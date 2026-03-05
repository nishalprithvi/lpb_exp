#!/usr/bin/env python3
import argparse
import csv
import json
from collections import deque
from pathlib import Path

import dill
import imageio
import numpy as np
import torch
import hydra

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


def load_policy(checkpoint_path: str, device: str):
    with open(checkpoint_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")

    cfg = payload["cfg"]
    target = str(getattr(cfg, "_target_", ""))
    if "lowdim" not in target.lower():
        raise RuntimeError(f"Expected low-dim checkpoint, got: {target}")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    print(f"[ckpt-ok] {checkpoint_path} target={target}")
    return policy, cfg


def obs_to_policy(obs_vec: np.ndarray):
    do = obs_vec.shape[-1] // 2
    return obs_vec[..., :do]


def summarize_actions(actions_arr):
    if actions_arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(actions_arr)),
        "max": float(np.max(actions_arr)),
        "mean": float(np.mean(actions_arr)),
        "std": float(np.std(actions_arr)),
    }


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
):
    env = PushTKeypointsEnv(
        legacy=True,
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
    policy.obstacle_guidance_scale = float(obstacle_guidance_scale)
    policy.obstacle_margin = float(obstacle_margin)
    policy.guidance_start_timestep = int(guidance_start_timestep)
    if hasattr(policy, "set_obstacle_guidance"):
        policy.set_obstacle_guidance(
            obstacle_center,
            obstacle_radius,
            enabled=(guidance_mode == "obstacle" and obstacle_enabled),
        )

    history = deque(maxlen=policy.n_obs_steps)
    for _ in range(policy.n_obs_steps):
        history.append(obs)

    frames = [env.render("rgb_array")]
    done = False
    step_count = 0
    max_reward = 0.0
    collided = False
    action_chunks = []

    while not done and step_count < max_steps:
        nobs = np.stack([obs_to_policy(x) for x in list(history)[-policy.n_obs_steps:]], axis=0)
        obs_dict = {
            "obs": torch.from_numpy(nobs[None, ...].astype(np.float32)).to(device=policy.device, dtype=policy.dtype)
        }

        if guidance_mode == "obstacle":
            action_dict = policy.predict_action(obs_dict)
        else:
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
        actions = action_dict["action"][0].detach().cpu().numpy()
        action_chunks.append(actions)

        for action in actions:
            obs, reward, done, info = env.step(action)
            max_reward = max(max_reward, float(reward))
            collided = collided or bool(info.get("collision_obstacle", False))
            history.append(obs)
            frames.append(env.render("rgb_array"))
            step_count += 1
            if done or step_count >= max_steps:
                break

    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=fps)
    env.close()

    all_actions = np.concatenate(action_chunks, axis=0) if action_chunks else np.zeros((0, 2), dtype=np.float32)
    action_stats = summarize_actions(all_actions)

    guidance_debug = {}
    if hasattr(policy, "get_guidance_debug_summary"):
        guidance_debug = policy.get_guidance_debug_summary()

    print(
        f"[episode-debug] seed={seed} mode={guidance_mode} obstacle={obstacle_enabled} "
        f"action[min,max,mean,std]=({action_stats['min']:.3f},{action_stats['max']:.3f},{action_stats['mean']:.3f},{action_stats['std']:.3f})"
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
        "guidance_loss_mean": float(guidance_debug.get("obstacle_loss_mean", 0.0)),
        "guidance_grad_norm_mean": float(guidance_debug.get("grad_norm_mean", 0.0)),
        "guidance_grad_norm_max": float(guidance_debug.get("grad_norm_max", 0.0)),
        "guidance_applied_ratio": float(guidance_debug.get("guidance_applied_ratio", 0.0)),
    }


def video_name(seed, scenario_filename):
    return f"seed_{seed}_{scenario_filename}.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs_phase2_pusht")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--obstacle-center", nargs=2, type=float, default=[320.0, 256.0])
    parser.add_argument("--obstacle-radius", type=float, default=38.0)
    parser.add_argument("--obstacle-guidance-scale", type=float, default=1.0)
    parser.add_argument("--obstacle-margin", type=float, default=8.0)
    parser.add_argument("--guidance-start-timestep", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--scenario-mode", choices=["full", "baseline_only", "guidance_only"], default="full")
    parser.add_argument("--seed-offset", type=int, default=10000)
    args = parser.parse_args()

    policy, cfg = load_policy(args.checkpoint, args.device)
    policy.num_inference_steps = int(args.num_inference_steps)

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = videos_dir / "per_episode_metrics.csv"
    metrics_json = videos_dir / "per_episode_metrics.json"

    all_specs = [
        {"scenario": "baseline", "obstacle_enabled": False, "guidance_mode": "none", "filename": "no_obstacle_baseline"},
        {"scenario": "obstacle_no_guidance", "obstacle_enabled": True, "guidance_mode": "none", "filename": "obstacle_no_guidance"},
        {"scenario": "obstacle_guidance", "obstacle_enabled": True, "guidance_mode": "obstacle", "filename": "obstacle_with_guidance"},
    ]
    if args.scenario_mode == "baseline_only":
        scenarios = [all_specs[0]]
    elif args.scenario_mode == "guidance_only":
        scenarios = [all_specs[2]]
    else:
        scenarios = all_specs

    rows = []
    for seed in args.seeds:
        for spec in scenarios:
            video_path = videos_dir / video_name(seed, spec["filename"])
            print(f"[run] seed={seed} scenario={spec['scenario']}")
            rollout_seed = int(seed) + int(args.seed_offset)
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
            )
            row = {
                "seed": int(seed),
                "episode_idx": 0,
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
                "video_path": str(video_path),
                "scenario": spec["scenario"],
            }
            rows.append(row)
            print(
                f"[done] seed={seed} scenario={spec['scenario']} "
                f"success={row['success']} collision={row['collision']} max_reward={row['max_reward']:.3f}"
            )

    metrics_json.write_text(json.dumps(rows, indent=2))
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metrics to: {metrics_csv}")
    print(f"Saved metrics to: {metrics_json}")


if __name__ == "__main__":
    main()
