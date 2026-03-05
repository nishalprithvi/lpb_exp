#!/usr/bin/env python3
import argparse
import csv
import json
from collections import deque
from pathlib import Path

import dill
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv


def load_policy(checkpoint_path: str, device: str, use_fp16: bool = True):
    with open(checkpoint_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")

    cfg = payload["cfg"]
    target = str(getattr(cfg, "_target_", ""))
    if "lowdim" in target.lower():
        raise RuntimeError(f"Incompatible low-dim checkpoint for PushT image eval: {target}")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    target_device = torch.device(device)
    use_half = bool(use_fp16 and target_device.type == "cuda")
    if use_half:
        policy = policy.half()
    policy.to(target_device)
    policy.eval()

    normalizer_path = Path(checkpoint_path).resolve().parent.parent / "normalizer.pth"
    if normalizer_path.exists():
        policy.normalizer.load_state_dict(torch.load(normalizer_path, map_location=device))
    if use_half:
        policy.normalizer.to(device=target_device, dtype=torch.float16)
    else:
        policy.normalizer.to(target_device)
    print(f"[ckpt-ok] {checkpoint_path} target={target} fp16={use_half}")
    return policy, cfg


def make_obs_window(history, n_obs_steps):
    images = np.stack([h["image"] for h in list(history)[-n_obs_steps:]], axis=0)
    agent_pos = np.stack([h["agent_pos"] for h in list(history)[-n_obs_steps:]], axis=0)
    return {
        "image": images[None, ...].astype(np.float32),
        "agent_pos": agent_pos[None, ...].astype(np.float32),
    }


def summarize_actions(actions_arr):
    if actions_arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(actions_arr)),
        "max": float(np.max(actions_arr)),
        "mean": float(np.mean(actions_arr)),
        "std": float(np.std(actions_arr)),
    }


def maybe_make_waypoint_plot(plot_path, waypoints_xy, obstacle_center, obstacle_radius):
    if len(waypoints_xy) == 0:
        return
    pts = np.asarray(waypoints_xy, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pts[:, 0], pts[:, 1], "-o", markersize=2, linewidth=1, label="denorm waypoints")
    circle = plt.Circle((obstacle_center[0], obstacle_center[1]), obstacle_radius, color="r", fill=False, linewidth=2, label="obstacle")
    ax.add_patch(circle)
    ax.scatter([obstacle_center[0]], [obstacle_center[1]], c="r", s=30)
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("PushT waypoints vs obstacle")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def rollout_episode(
    policy,
    seed,
    obstacle_enabled,
    guidance_mode,
    obstacle_center,
    obstacle_radius,
    obstacle_guidance_scale,
    obstacle_margin,
    guidance_start_timestep,
    max_steps,
    fps,
    video_path,
    debug_plot_path,
    render_size,
    legacy_mode,
):
    env = PushTImageEnv(
        legacy=bool(legacy_mode),
        render_size=int(render_size),
        obstacle_enabled=obstacle_enabled,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
    )
    env.seed(seed)
    obs = env.reset()
    policy.reset()
    if hasattr(policy, "reset_guidance_debug"):
        policy.reset_guidance_debug()

    policy.guidance_mode = guidance_mode
    policy.guidance_scale = 0.0
    policy.threshold = 1e9
    policy.obstacle_guidance_scale = float(obstacle_guidance_scale)
    policy.obstacle_margin = float(obstacle_margin)
    policy.guidance_start_timestep = int(guidance_start_timestep)
    policy.set_obstacle_guidance(
        obstacle_center,
        obstacle_radius,
        enabled=(guidance_mode == "obstacle" and obstacle_enabled),
    )

    history = deque(maxlen=policy.n_obs_steps)
    for _ in range(policy.n_obs_steps):
        history.append({"image": obs["image"], "agent_pos": obs["agent_pos"]})

    frames = [env.render("rgb_array")]
    done = False
    step_count = 0
    max_reward = 0.0
    collided = False
    action_chunks = []
    first_waypoints = []

    while not done and step_count < max_steps:
        np_obs = make_obs_window(history, policy.n_obs_steps)
        obs_dict = {k: torch.from_numpy(v).to(device=policy.device, dtype=policy.dtype) for k, v in np_obs.items()}

        action_dict = policy.predict_action(obs_dict)
        actions = action_dict["action"][0].detach().cpu().numpy()
        action_chunks.append(actions)

        if len(first_waypoints) < 3:
            for a in actions:
                first_waypoints.append([float(a[0]), float(a[1])])
                if len(first_waypoints) >= 3:
                    break

        for action in actions:
            obs, reward, done, info = env.step(action)
            max_reward = max(max_reward, float(reward))
            collided = collided or bool(info.get("collision_obstacle", False))
            history.append({"image": obs["image"], "agent_pos": obs["agent_pos"]})
            frames.append(env.render("rgb_array"))
            step_count += 1
            if done or step_count >= max_steps:
                break

    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=fps)
    env.close()

    all_actions = np.concatenate(action_chunks, axis=0) if action_chunks else np.zeros((0, 2), dtype=np.float32)
    action_stats = summarize_actions(all_actions)
    center_xy = np.asarray(obstacle_center, dtype=np.float32)
    first_wp = np.asarray(first_waypoints, dtype=np.float32) if len(first_waypoints) > 0 else np.zeros((0, 2), dtype=np.float32)
    dist_first = float(np.linalg.norm(first_wp[0] - center_xy[:2])) if first_wp.shape[0] > 0 else -1.0

    guidance_debug = {}
    if hasattr(policy, "get_guidance_debug_summary"):
        guidance_debug = policy.get_guidance_debug_summary()

    if debug_plot_path is not None and obstacle_enabled:
        maybe_make_waypoint_plot(debug_plot_path, first_waypoints, center_xy[:2], obstacle_radius)

    print(
        f"[episode-debug] seed={seed} mode={guidance_mode} obstacle={obstacle_enabled} "
        f"action[min,max,mean,std]=({action_stats['min']:.3f},{action_stats['max']:.3f},{action_stats['mean']:.3f},{action_stats['std']:.3f}) "
        f"first_waypoints={first_waypoints} obstacle_center={center_xy[:2].tolist()} dist_first={dist_first:.3f}"
    )
    if guidance_debug:
        print(
            "[guidance-debug] "
            f"loss_mean={guidance_debug.get('obstacle_loss_mean', 0.0):.6f} "
            f"grad_norm_mean={guidance_debug.get('grad_norm_mean', 0.0):.6f} "
            f"grad_norm_max={guidance_debug.get('grad_norm_max', 0.0):.6f} "
            f"applied_ratio={100.0 * guidance_debug.get('guidance_applied_ratio', 0.0):.1f}% "
            f"eligible={guidance_debug.get('eligible_steps', 0)} "
            f"applied={guidance_debug.get('applied_steps', 0)}"
        )

    return {
        "success": 1 if max_reward >= 0.95 else 0,
        "collision": 1 if collided else 0,
        "episode_length": int(step_count),
        "max_reward": float(max_reward),
        "action_min": action_stats["min"],
        "action_max": action_stats["max"],
        "action_mean": action_stats["mean"],
        "action_std": action_stats["std"],
        "debug_first_waypoints": json.dumps(first_waypoints),
        "debug_obstacle_center": json.dumps(center_xy[:2].tolist()),
        "debug_first_distance": dist_first,
        "guidance_loss_mean": float(guidance_debug.get("obstacle_loss_mean", 0.0)),
        "guidance_grad_norm_mean": float(guidance_debug.get("grad_norm_mean", 0.0)),
        "guidance_grad_norm_max": float(guidance_debug.get("grad_norm_max", 0.0)),
        "guidance_applied_ratio": float(guidance_debug.get("guidance_applied_ratio", 0.0)),
    }


def _row_key(row):
    return (int(row["seed"]), str(row["scenario"]), int(row.get("episode_idx", 0)))


def load_existing_rows(metrics_json: Path):
    if not metrics_json.exists():
        return {}
    rows = json.loads(metrics_json.read_text())
    return {_row_key(r): r for r in rows}


def save_rows(rows_by_key, metrics_csv: Path, metrics_json: Path):
    scenario_order = {"baseline": 0, "obstacle_no_guidance": 1, "obstacle_guidance": 2}
    rows = [rows_by_key[k] for k in sorted(rows_by_key.keys(), key=lambda x: (x[0], scenario_order.get(x[1], 99), x[2]))]
    if not rows:
        return
    metrics_json.write_text(json.dumps(rows, indent=2))
    fieldnames = list(rows[0].keys())
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def video_name(seed, episode_idx, scenario_filename, episodes_per_seed):
    base = f"seed_{seed}_{scenario_filename}"
    if episodes_per_seed > 1:
        return f"{base}_ep_{episode_idx:02d}.mp4"
    return f"{base}.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs_phase2_pusht")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--obstacle-center", nargs=2, type=float, default=[320.0, 256.0])
    parser.add_argument("--obstacle-radius", type=float, default=38.0)
    parser.add_argument("--obstacle-guidance-scale", type=float, default=1.0)
    parser.add_argument("--obstacle-margin", type=float, default=8.0)
    parser.add_argument("--guidance-start-timestep", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--scenario-mode", choices=["full", "baseline_only", "guidance_only"], default="full")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    if args.episodes_per_seed < 1:
        raise ValueError("--episodes-per-seed must be >= 1")

    policy, _cfg = load_policy(args.checkpoint, args.device, use_fp16=not args.no_fp16)
    cfg_render_size = int(_cfg.task.shape_meta.obs.image.shape[-1])
    cfg_legacy = bool(_cfg.task.env_runner.legacy_test)
    print(f"[cfg] using render_size={cfg_render_size} legacy={cfg_legacy} from checkpoint")
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    debug_dir = videos_dir / "debug"
    videos_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = videos_dir / "per_episode_metrics.csv"
    metrics_json = videos_dir / "per_episode_metrics.json"
    rows_by_key = load_existing_rows(metrics_json)

    all_specs = [
        {
            "scenario": "baseline",
            "obstacle_enabled": False,
            "guidance_mode": "none",
            "filename": "no_obstacle_baseline",
        },
        {
            "scenario": "obstacle_no_guidance",
            "obstacle_enabled": True,
            "guidance_mode": "none",
            "filename": "obstacle_no_guidance",
        },
        {
            "scenario": "obstacle_guidance",
            "obstacle_enabled": True,
            "guidance_mode": "obstacle",
            "filename": "obstacle_with_guidance",
        },
    ]
    if args.scenario_mode == "baseline_only":
        scenarios = [all_specs[0]]
    elif args.scenario_mode == "guidance_only":
        scenarios = [all_specs[2]]
    else:
        scenarios = all_specs

    episode_seed_stride = 1000

    for seed in args.seeds:
        for episode_idx in range(args.episodes_per_seed):
            rollout_seed = int(seed) + int(episode_idx) * episode_seed_stride
            for spec in scenarios:
                key = (int(seed), spec["scenario"], int(episode_idx))
                file_name = video_name(seed, episode_idx, spec["filename"], args.episodes_per_seed)
                video_path = videos_dir / file_name
                debug_plot_path = debug_dir / f"seed_{seed}_{spec['filename']}_debug.png"

                if key in rows_by_key and video_path.exists():
                    print(
                        f"[resume-skip] seed={seed} episode={episode_idx} "
                        f"scenario={spec['scenario']} (video+metrics present)"
                    )
                    continue

                print(f"[run] seed={seed} episode={episode_idx} scenario={spec['scenario']}")
                out = rollout_episode(
                    policy=policy,
                    seed=rollout_seed,
                    obstacle_enabled=spec["obstacle_enabled"],
                    guidance_mode=spec["guidance_mode"],
                    obstacle_center=args.obstacle_center,
                    obstacle_radius=args.obstacle_radius,
                    obstacle_guidance_scale=args.obstacle_guidance_scale,
                    obstacle_margin=args.obstacle_margin,
                    guidance_start_timestep=args.guidance_start_timestep,
                    max_steps=args.max_steps,
                    fps=args.fps,
                    video_path=video_path,
                    debug_plot_path=debug_plot_path,
                    render_size=cfg_render_size,
                    legacy_mode=cfg_legacy,
                )

                row = {
                    "seed": int(seed),
                    "episode_idx": int(episode_idx),
                    "rollout_seed": int(rollout_seed),
                    "obstacle_enabled": bool(spec["obstacle_enabled"]),
                    "guidance_mode": spec["guidance_mode"],
                    "success": out["success"],
                    "collision": out["collision"],
                    "episode_length": out["episode_length"],
                    "max_reward": out["max_reward"],
                    "action_min": out["action_min"],
                    "action_max": out["action_max"],
                    "action_mean": out["action_mean"],
                    "action_std": out["action_std"],
                    "guidance_loss_mean": out["guidance_loss_mean"],
                    "guidance_grad_norm_mean": out["guidance_grad_norm_mean"],
                    "guidance_grad_norm_max": out["guidance_grad_norm_max"],
                    "guidance_applied_ratio": out["guidance_applied_ratio"],
                    "debug_first_waypoints": out["debug_first_waypoints"],
                    "debug_obstacle_center": out["debug_obstacle_center"],
                    "debug_first_distance": out["debug_first_distance"],
                    "video_path": str(video_path),
                    "scenario": spec["scenario"],
                }
                rows_by_key[key] = row
                save_rows(rows_by_key, metrics_csv, metrics_json)
                print(
                    f"[done] seed={seed} scenario={spec['scenario']} success={row['success']} "
                    f"collision={row['collision']} max_reward={row['max_reward']:.3f}"
                )

    save_rows(rows_by_key, metrics_csv, metrics_json)
    print(f"Saved metrics to: {metrics_csv}")
    print(f"Saved metrics to: {metrics_json}")


if __name__ == "__main__":
    main()
