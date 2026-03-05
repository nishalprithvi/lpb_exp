#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import dill
import h5py
import hydra
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.libero_image_sequential_runner import create_env
import robomimic.utils.file_utils as FileUtils


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


def scenario_specs():
    return [
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


def save_rows(rows, metrics_csv: Path, metrics_json: Path):
    rows = sorted(rows, key=lambda r: (int(r["seed"]), str(r["scenario"]), int(r["episode_idx"])))
    metrics_json.write_text(json.dumps(rows, indent=2))
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def estimate_obstacle_center_from_dataset(dataset_path: Path, max_demos: int = 30):
    pts = []
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")])[:max_demos]
        for d in demos:
            k = f"data/{d}/obs/robot0_eef_pos"
            if k in f:
                arr = f[k][:]
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    pts.append(arr[:, :3])
    if not pts:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cat = np.concatenate(pts, axis=0)
    return np.median(cat, axis=0).astype(np.float32)


def apply_action_obstacle_guidance(actions, obstacle_center, obstacle_radius, obstacle_margin, obstacle_guidance_scale):
    out = np.array(actions, copy=True)
    safe_r = float(obstacle_radius) + float(obstacle_margin)
    center = np.asarray(obstacle_center, dtype=np.float32)

    for i in range(out.shape[0]):
        pos = out[i, :3]
        vec = pos - center
        dist = float(np.linalg.norm(vec) + 1e-8)
        if dist < safe_r:
            direction = vec / dist
            push = (safe_r - dist) * float(obstacle_guidance_scale)
            out[i, :3] = pos + push * direction
    return out


def rot6d_to_matrix_np(rot6d):
    x = rot6d[..., 0:3]
    y = rot6d[..., 3:6]
    x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)
    y = y - np.sum(x * y, axis=-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True).clip(min=1e-8)
    z = np.cross(x, y, axis=-1)
    return np.stack([x, y, z], axis=-1)


def undo_transform_action(action):
    raw_shape = action.shape
    arr = action
    if raw_shape[-1] == 20:
        arr = action.reshape(-1, 2, 10)

    d_rot = arr.shape[-1] - 4
    pos = arr[..., :3]
    rot6d = arr[..., 3:3 + d_rot]
    gripper = arr[..., [-1]]

    rot_mat = rot6d_to_matrix_np(rot6d)
    rotvec = Rotation.from_matrix(rot_mat.reshape(-1, 3, 3)).as_rotvec().reshape(rot_mat.shape[:-2] + (3,))
    uaction = np.concatenate([pos, rotvec, gripper], axis=-1)

    if raw_shape[-1] == 20:
        uaction = uaction.reshape(*raw_shape[:-1], 14)
    return uaction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", default="outputs_phase2_libero")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--obstacle-center", nargs=3, type=float, default=None)
    parser.add_argument("--obstacle-radius", type=float, default=0.08)
    parser.add_argument("--obstacle-guidance-scale", type=float, default=0.6)
    parser.add_argument("--obstacle-margin", type=float, default=0.03)
    parser.add_argument("--guidance-start-timestep", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset_path)
    policy, cfg = load_policy(args.checkpoint, args.device)
    policy.num_inference_steps = int(args.num_inference_steps)
    policy.guidance_scale = 0.0
    policy.threshold = 1e9

    if args.obstacle_center is None:
        obstacle_center = estimate_obstacle_center_from_dataset(dataset_path)
    else:
        obstacle_center = np.asarray(args.obstacle_center, dtype=np.float32)

    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
    shape_meta = cfg.task.shape_meta
    abs_action = bool(cfg.task.env_runner.abs_action)

    language_goal = " ".join(dataset_path.name[:-10].split("_"))

    metrics_csv = videos_dir / "per_episode_metrics.csv"
    metrics_json = videos_dir / "per_episode_metrics.json"
    rows = []

    for seed in args.seeds:
        for ep_idx in range(args.episodes_per_seed):
            rollout_seed = int(seed) * 1000 + int(ep_idx)
            for spec in scenario_specs():
                video_path = videos_dir / f"seed_{seed}_{spec['filename']}.mp4"
                if args.episodes_per_seed > 1:
                    video_path = videos_dir / f"seed_{seed}_{spec['filename']}_ep_{ep_idx:02d}.mp4"

                env = create_env(
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    enable_render=True,
                    render_obs_key="agentview_image",
                    fps=args.fps,
                    n_obs_steps=policy.n_obs_steps,
                    n_action_steps=policy.n_action_steps,
                    max_steps=args.max_steps,
                )

                env.env.video_recoder.stop()
                env.env.file_path = str(video_path)
                env.env.env.init_state = None
                env.seed(rollout_seed)

                obs = env.reset()
                policy.reset()

                policy.guidance_mode = spec["guidance_mode"]
                policy.guidance_start_timestep = int(args.guidance_start_timestep)
                policy.obstacle_guidance_scale = float(args.obstacle_guidance_scale)
                policy.obstacle_margin = float(args.obstacle_margin)
                policy.set_obstacle_guidance(
                    obstacle_center,
                    args.obstacle_radius,
                    enabled=(spec["guidance_mode"] == "obstacle" and spec["obstacle_enabled"]),
                )

                done = False
                step_count = 0
                max_reward = 0.0
                collided = False

                while not done:
                    np_obs_dict = dict(obs)
                    np_obs_dict = dict_apply(np_obs_dict, lambda x: np.expand_dims(x, axis=0))
                    obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(policy.device))

                    if "agentview_image" in obs_dict:
                        obs_dict["agentview_rgb"] = obs_dict.pop("agentview_image")
                    if "robot0_eye_in_hand_image" in obs_dict:
                        obs_dict["eye_in_hand_rgb"] = obs_dict.pop("robot0_eye_in_hand_image")
                    if "robot0_joint_pos" in obs_dict:
                        obs_dict["joint_states"] = obs_dict.pop("robot0_joint_pos")
                    if "robot0_eef_pos" in obs_dict:
                        obs_dict["ee_pos"] = obs_dict.pop("robot0_eef_pos")
                    if "robot0_eef_quat" in obs_dict:
                        obs_dict["ee_ori"] = obs_dict.pop("robot0_eef_quat")

                    action_dict = policy.predict_action(
                        obs_dict,
                        language_goal=[language_goal] * obs_dict["agentview_rgb"].size(0),
                    )
                    action = action_dict["action"][0].detach().cpu().numpy()

                    if spec["scenario"] == "obstacle_guidance":
                        action = apply_action_obstacle_guidance(
                            action,
                            obstacle_center,
                            args.obstacle_radius,
                            args.obstacle_margin,
                            args.obstacle_guidance_scale,
                        )

                    env_action = undo_transform_action(action) if abs_action else action
                    obs, reward, done_flag, _info = env.step(env_action)

                    if reward == 1.0:
                        done_flag = np.array([True])
                    done = bool(np.all(done_flag))

                    eef = obs.get("robot0_eef_pos", None)
                    if spec["obstacle_enabled"] and eef is not None:
                        eef_pt = np.asarray(eef)[-1, :3] if np.asarray(eef).ndim == 2 else np.asarray(eef)[:3]
                        if np.linalg.norm(eef_pt - obstacle_center) <= float(args.obstacle_radius):
                            collided = True

                    max_reward = max(max_reward, float(reward))
                    step_count += int(action.shape[0]) if action.ndim == 2 else 1
                    if step_count >= args.max_steps:
                        break

                env.render()
                env.close()

                rows.append({
                    "seed": int(seed),
                    "episode_idx": int(ep_idx),
                    "rollout_seed": int(rollout_seed),
                    "obstacle_enabled": bool(spec["obstacle_enabled"]),
                    "guidance_mode": spec["guidance_mode"],
                    "success": 1 if max_reward >= 1.0 else 0,
                    "collision": 1 if collided else 0,
                    "episode_length": int(step_count),
                    "max_reward": float(max_reward),
                    "video_path": str(video_path),
                    "scenario": spec["scenario"],
                })
                save_rows(rows, metrics_csv, metrics_json)
                print(f"[done] seed={seed} episode={ep_idx} scenario={spec['scenario']} success={rows[-1]['success']} collision={rows[-1]['collision']}")

    save_rows(rows, metrics_csv, metrics_json)
    print(f"Saved metrics CSV: {metrics_csv}")
    print(f"Saved metrics JSON: {metrics_json}")


if __name__ == "__main__":
    main()
