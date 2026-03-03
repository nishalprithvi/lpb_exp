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
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv


def load_policy(checkpoint_path: str, device: str):
    with open(checkpoint_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")

    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()

    normalizer_path = Path(checkpoint_path).resolve().parent.parent / "normalizer.pth"
    if normalizer_path.exists():
        policy.normalizer.load_state_dict(torch.load(normalizer_path, map_location=device))
    policy.normalizer.to(torch.device(device))
    return policy, cfg


def make_obs_window(history, n_obs_steps):
    images = np.stack([h["image"] for h in list(history)[-n_obs_steps:]], axis=0)
    agent_pos = np.stack([h["agent_pos"] for h in list(history)[-n_obs_steps:]], axis=0)
    return {
        "image": images[None, ...].astype(np.float32),
        "agent_pos": agent_pos[None, ...].astype(np.float32),
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
    env = PushTImageEnv(
        legacy=True,
        render_size=140,
        obstacle_enabled=obstacle_enabled,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
    )
    env.seed(seed)
    obs = env.reset()
    policy.reset()

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

    while not done and step_count < max_steps:
        np_obs = make_obs_window(history, policy.n_obs_steps)
        obs_dict = {k: torch.from_numpy(v).to(device=policy.device, dtype=policy.dtype) for k, v in np_obs.items()}

        action_dict = policy.predict_action(obs_dict)
        actions = action_dict["action"][0].detach().cpu().numpy()

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

    return {
        "success": 1 if max_reward >= 0.95 else 0,
        "collision": 1 if collided else 0,
        "episode_length": int(step_count),
        "max_reward": float(max_reward),
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
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--obstacle-center", nargs=2, type=float, default=[320.0, 256.0])
    parser.add_argument("--obstacle-radius", type=float, default=38.0)
    parser.add_argument("--obstacle-guidance-scale", type=float, default=0.15)
    parser.add_argument("--obstacle-margin", type=float, default=12.0)
    parser.add_argument("--guidance-start-timestep", type=int, default=35)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    args = parser.parse_args()

    if args.episodes_per_seed < 1:
        raise ValueError("--episodes-per-seed must be >= 1")

    policy, _cfg = load_policy(args.checkpoint, args.device)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = videos_dir / "per_episode_metrics.csv"
    metrics_json = videos_dir / "per_episode_metrics.json"
    rows_by_key = load_existing_rows(metrics_json)

    scenarios = [
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

    # Each episode for a seed uses a deterministic offset seed so we can estimate variability.
    episode_seed_stride = 1000

    for seed in args.seeds:
        for episode_idx in range(args.episodes_per_seed):
            rollout_seed = int(seed) * episode_seed_stride + int(episode_idx)
            for spec in scenarios:
                key = (int(seed), spec["scenario"], int(episode_idx))
                file_name = video_name(seed, episode_idx, spec["filename"], args.episodes_per_seed)
                video_path = videos_dir / file_name

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
                    "video_path": str(video_path),
                    "scenario": spec["scenario"],
                }
                rows_by_key[key] = row
                save_rows(rows_by_key, metrics_csv, metrics_json)

    save_rows(rows_by_key, metrics_csv, metrics_json)
    print(f"Saved metrics to: {metrics_csv}")
    print(f"Saved metrics to: {metrics_json}")


if __name__ == "__main__":
    main()
