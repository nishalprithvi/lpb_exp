import argparse
import csv
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import dill
import hydra
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.workspace.base_workspace import BaseWorkspace


SUCCESS_THRESHOLD = 0.95
CHUNK_SIZE_DEFAULT = 20


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


@dataclass
class PipelinePaths:
    root: Path
    rollout_bank: Path
    failure_bank: Path
    localization: Path
    plots: Path
    videos: Path


@dataclass
class RunConfig:
    task: str
    ckpt: Path
    output_root: Path
    guidance_mode: str
    episodes: int
    test_start_seed: int
    max_steps: int
    chunk_size: int
    device: str
    force: bool


class PolicyWrapper:
    def __init__(self, ckpt_path: Path, device: str):
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)
        self.workspace = None
        self.policy = None
        self.n_obs_steps = 2
        self.n_action_steps = 8

    def load(self):
        log(f"Loading checkpoint: {self.ckpt_path}")
        payload = torch.load(self.ckpt_path.open("rb"), map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg, output_dir=str(self.ckpt_path.parent.parent))
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model

        normalizer_path = self.ckpt_path.parent.parent / "normalizer.pth"
        if normalizer_path.exists() and hasattr(policy, "normalizer"):
            log(f"Loading normalizer: {normalizer_path}")
            policy.normalizer.load_state_dict(torch.load(normalizer_path, map_location="cpu"))

        policy.to(self.device)
        policy.eval()

        self.workspace = workspace
        self.policy = policy
        self.n_obs_steps = int(getattr(policy, "n_obs_steps", 2))
        self.n_action_steps = int(getattr(policy, "n_action_steps", 8))
        log(f"Policy loaded. n_obs_steps={self.n_obs_steps}, n_action_steps={self.n_action_steps}")

    def predict_actions(self, obs_hist: Dict[str, np.ndarray]) -> np.ndarray:
        obs_t = {}
        for k, v in obs_hist.items():
            obs_t[k] = torch.from_numpy(v).to(self.device)
        with torch.no_grad():
            out = self.policy.predict_action(obs_t)
        action = out["action"].detach().cpu().numpy()[0]
        return action



def build_paths(output_root: Path, task: str, model_name: str, guidance_mode: str) -> PipelinePaths:
    base = output_root / task / model_name / guidance_mode
    paths = PipelinePaths(
        root=base,
        rollout_bank=base / "rollout_bank",
        failure_bank=base / "failure_bank",
        localization=base / "localization",
        plots=base / "plots",
        videos=base / "videos",
    )
    for p in [paths.root, paths.rollout_bank, paths.failure_bank, paths.localization, paths.plots, paths.videos]:
        p.mkdir(parents=True, exist_ok=True)
    (paths.rollout_bank / "chunks").mkdir(parents=True, exist_ok=True)
    (paths.rollout_bank / "states").mkdir(parents=True, exist_ok=True)
    (paths.rollout_bank / "chunk_rewards").mkdir(parents=True, exist_ok=True)
    return paths



def make_obs_tensor(history: deque, n_obs_steps: int) -> Dict[str, np.ndarray]:
    items = list(history)
    if len(items) < n_obs_steps:
        items = [items[0]] * (n_obs_steps - len(items)) + items
    else:
        items = items[-n_obs_steps:]

    img = np.stack([x["image"] for x in items], axis=0).astype(np.float32)
    pos = np.stack([x["agent_pos"] for x in items], axis=0).astype(np.float32)
    return {
        "image": img[None, ...],
        "agent_pos": pos[None, ...],
    }



def collect_rollouts(cfg: RunConfig, paths: PipelinePaths, policy: PolicyWrapper):
    manifest_csv = paths.rollout_bank / "rollout_manifest.csv"

    existing_rows = []
    if manifest_csv.exists() and not cfg.force:
        with manifest_csv.open("r") as f:
            existing_rows = list(csv.DictReader(f))
        if len(existing_rows) >= cfg.episodes:
            log(f"Rollouts already exist ({len(existing_rows)} >= {cfg.episodes}), skipping collection.")
            return

    log("Collecting rollout trajectories...")
    fieldnames = [
        "episode_id", "timestamp", "task", "checkpoint", "guidance_mode", "seed", "success", "failure",
        "max_reward", "steps", "chunk_size", "chunk_count", "chunk_path", "state_path", "chunk_rewards_path",
        "video_path"
    ]

    rows: List[dict] = []
    env = PushTImageEnv(legacy=True, render_size=140)

    for i in range(cfg.episodes):
        ep_id = f"ep_{i:06d}"
        seed = cfg.test_start_seed + i
        np.random.seed(seed)
        torch.manual_seed(seed)

        env.seed(seed)
        obs = env.reset()
        history = deque([obs], maxlen=policy.n_obs_steps)
        frames = [env.render("rgb_array")]

        rewards = []
        states = [{"step": 0, "state": obs["state"][:5].tolist()}]
        chunk_rewards = []
        chunk_starts = [0]
        step_count = 0
        done = False

        while (not done) and step_count < cfg.max_steps:
            obs_t = make_obs_tensor(history, policy.n_obs_steps)
            action_seq = policy.predict_actions(obs_t)

            for j in range(action_seq.shape[0]):
                action = action_seq[j]
                obs, reward, done, _ = env.step(action)
                history.append(obs)
                rewards.append(float(reward))
                frames.append(env.render("rgb_array"))
                step_count += 1

                if step_count % cfg.chunk_size == 0 or done:
                    start = chunk_starts[-1]
                    end = step_count
                    cmax = max(rewards[start:end]) if end > start else 0.0
                    chunk_rewards.append(float(cmax))
                    states.append({"step": step_count, "state": obs["state"][:5].tolist()})
                    chunk_starts.append(step_count)

                if done or step_count >= cfg.max_steps:
                    break

        max_reward = float(max(rewards) if rewards else 0.0)
        success = 1 if max_reward >= SUCCESS_THRESHOLD else 0

        video_path = paths.videos / f"{ep_id}.mp4"
        imageio.mimsave(video_path, frames, fps=10)

        chunk_path = paths.rollout_bank / "chunks" / f"{ep_id}_chunks.json"
        state_path = paths.rollout_bank / "states" / f"{ep_id}_boundary_states.json"
        chunk_rewards_path = paths.rollout_bank / "chunk_rewards" / f"{ep_id}_chunk_rewards.json"

        with chunk_path.open("w") as f:
            json.dump({"chunk_size": cfg.chunk_size, "steps": step_count, "chunk_count": len(chunk_rewards)}, f, indent=2)
        with state_path.open("w") as f:
            json.dump({"episode_id": ep_id, "seed": seed, "states": states}, f, indent=2)
        with chunk_rewards_path.open("w") as f:
            json.dump({"episode_id": ep_id, "chunk_rewards": chunk_rewards}, f, indent=2)

        row = {
            "episode_id": ep_id,
            "timestamp": datetime.utcnow().isoformat(),
            "task": cfg.task,
            "checkpoint": str(cfg.ckpt),
            "guidance_mode": cfg.guidance_mode,
            "seed": seed,
            "success": success,
            "failure": 1 - success,
            "max_reward": max_reward,
            "steps": step_count,
            "chunk_size": cfg.chunk_size,
            "chunk_count": len(chunk_rewards),
            "chunk_path": str(chunk_path),
            "state_path": str(state_path),
            "chunk_rewards_path": str(chunk_rewards_path),
            "video_path": str(video_path),
        }
        rows.append(row)
        log(f"Collected {ep_id}: success={success}, max_reward={max_reward:.3f}, steps={step_count}, chunks={len(chunk_rewards)}")

    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"Rollout manifest written: {manifest_csv}")



def read_csv(path: Path) -> List[dict]:
    with path.open("r") as f:
        return list(csv.DictReader(f))



def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def build_failure_bank(paths: PipelinePaths):
    manifest_csv = paths.rollout_bank / "rollout_manifest.csv"
    rows = read_csv(manifest_csv)

    failures = []
    for r in rows:
        fail = int(r["failure"])
        has_state = Path(r["state_path"]).exists()
        if fail == 1 and has_state:
            failures.append(r)

    out_csv = paths.failure_bank / "failure_bank.csv"
    if failures:
        write_csv(out_csv, failures, list(failures[0].keys()))
    else:
        write_csv(out_csv, [], list(rows[0].keys()))

    summary = {
        "num_episodes": len(rows),
        "num_failures": len(failures),
        "failure_rate": float(len(failures) / max(1, len(rows))),
        "rollout_manifest_csv": str(manifest_csv),
        "failure_bank_csv": str(out_csv),
    }
    with (paths.failure_bank / "failure_bank_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    log(f"Failure bank built: {len(failures)} failures / {len(rows)} episodes")



def propose_candidate_chunks(paths: PipelinePaths):
    failure_csv = paths.failure_bank / "failure_bank.csv"
    failures = read_csv(failure_csv)

    cand_rows = []
    diagnostics = {}
    for r in failures:
        ep = r["episode_id"]
        chunk_rewards_path = Path(r["chunk_rewards_path"])
        data = json.loads(chunk_rewards_path.read_text())
        rewards = data["chunk_rewards"]

        candidate = 0
        best_improve = -1e9
        running_best = -1e9
        stall_idx = None
        for i, rv in enumerate(rewards):
            improve = rv - running_best
            if improve > best_improve:
                best_improve = improve
            if rv > running_best:
                running_best = rv
            if i >= 2:
                recent = rewards[max(0, i-2):i+1]
                if max(recent) - min(recent) < 0.02 and stall_idx is None:
                    stall_idx = i - 1
        if stall_idx is not None:
            candidate = int(stall_idx)
        else:
            candidate = int(max(0, len(rewards) - 2))

        cand_rows.append({
            "episode_id": ep,
            "candidate_chunk": candidate,
            "chunk_count": len(rewards),
            "reason": "stall_or_late_failure"
        })
        diagnostics[ep] = {
            "chunk_rewards": rewards,
            "running_best": float(max(rewards) if rewards else 0.0),
            "candidate_chunk": candidate
        }

    out_csv = paths.localization / "candidate_failure_chunks.csv"
    write_csv(out_csv, cand_rows, ["episode_id", "candidate_chunk", "chunk_count", "reason"]) if cand_rows else write_csv(out_csv, [], ["episode_id", "candidate_chunk", "chunk_count", "reason"])
    with (paths.localization / "candidate_diagnostics.json").open("w") as f:
        json.dump(diagnostics, f, indent=2)
    log(f"Candidate chunks proposed: {len(cand_rows)}")



def run_from_state(policy: PolicyWrapper, state5: List[float], max_steps: int, seed: int) -> Tuple[int, float]:
    env = PushTImageEnv(legacy=True, render_size=140)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.reset()
    env._set_state(np.array(state5))
    obs = env._get_obs()
    history = deque([obs], maxlen=policy.n_obs_steps)

    rewards = []
    done = False
    steps = 0
    while (not done) and steps < max_steps:
        obs_t = make_obs_tensor(history, policy.n_obs_steps)
        action_seq = policy.predict_actions(obs_t)
        for j in range(action_seq.shape[0]):
            obs, reward, done, _ = env.step(action_seq[j])
            history.append(obs)
            rewards.append(float(reward))
            steps += 1
            if done or steps >= max_steps:
                break

    max_reward = float(max(rewards) if rewards else 0.0)
    success = 1 if max_reward >= SUCCESS_THRESHOLD else 0
    return success, max_reward



def eval_recoverability(policy: PolicyWrapper, state5: List[float], n: int, seed_base: int, max_steps: int) -> Tuple[int, float]:
    succ = 0
    for k in range(n):
        s, _ = run_from_state(policy, state5, max_steps=max_steps, seed=seed_base + k)
        succ += s
    return succ, float(succ / max(1, n))



def localize_failure_chunks(paths: PipelinePaths, policy: PolicyWrapper, rho: float, coarse_n: int, refine_n: int, verify_n: int, max_steps: int):
    cand_csv = paths.localization / "candidate_failure_chunks.csv"
    failure_csv = paths.failure_bank / "failure_bank.csv"
    cands = read_csv(cand_csv)
    failures = {r["episode_id"]: r for r in read_csv(failure_csv)}

    recover_rows = []
    loc_rows = []

    for c in cands:
        ep = c["episode_id"]
        fail_row = failures[ep]
        state_data = json.loads(Path(fail_row["state_path"]).read_text())
        states = state_data["states"]
        chunk_count = int(c["chunk_count"])

        cand = min(int(c["candidate_chunk"]), max(0, chunk_count - 1))
        prev_idx = max(0, cand - 1)
        next_idx = min(chunk_count - 1, cand + 1)

        idx_to_budget = [(prev_idx, coarse_n), (cand, coarse_n), (next_idx, coarse_n)]
        rec = {}

        for idx, n in idx_to_budget:
            state5 = states[idx]["state"]
            succ, rscore = eval_recoverability(policy, state5, n=n, seed_base=300000 + idx * 1000, max_steps=max_steps)
            rec[idx] = rscore
            recover_rows.append({
                "episode_id": ep,
                "chunk_index": idx,
                "n_rollouts": n,
                "successes": succ,
                "recoverability": rscore,
            })

        localized = None
        status = "not_localized"
        if rec[prev_idx] >= rho and rec[cand] < rho:
            localized = cand
            status = "localized"
        elif rec[cand] >= rho and rec[next_idx] < rho:
            localized = next_idx
            status = "localized"
        else:
            # Refine uncertainty around candidate neighborhood.
            for idx in sorted({prev_idx, cand, next_idx}):
                state5 = states[idx]["state"]
                succ, rscore = eval_recoverability(policy, state5, n=refine_n, seed_base=400000 + idx * 1000, max_steps=max_steps)
                rec[idx] = rscore
                recover_rows.append({
                    "episode_id": ep,
                    "chunk_index": idx,
                    "n_rollouts": refine_n,
                    "successes": succ,
                    "recoverability": rscore,
                })
            if rec[prev_idx] >= rho and rec[cand] < rho:
                localized = cand
                status = "localized_after_refine"
            elif rec[cand] >= rho and rec[next_idx] < rho:
                localized = next_idx
                status = "localized_after_refine"

        if localized is None:
            # Fast fallback: avoid expensive full-trajectory counterfactual scan.
            # Use lowest recoverability among evaluated neighborhood, else candidate.
            neigh = [(k, v) for k, v in rec.items() if k in {prev_idx, cand, next_idx}]
            if neigh:
                localized = min(neigh, key=lambda kv: kv[1])[0]
                status = "localized_fallback_neighborhood_min"
            else:
                localized = cand
                status = "localized_fallback_candidate"

        if localized is not None:
            state5 = states[localized]["state"]
            succ, rscore = eval_recoverability(policy, state5, n=verify_n, seed_base=500000 + localized * 1000, max_steps=max_steps)
            rec[localized] = rscore
            recover_rows.append({
                "episode_id": ep,
                "chunk_index": localized,
                "n_rollouts": verify_n,
                "successes": succ,
                "recoverability": rscore,
            })

        loc_rows.append({
            "episode_id": ep,
            "candidate_chunk": cand,
            "localized_failure_chunk": localized if localized is not None else "",
            "R_prev": rec.get(prev_idx, float("nan")),
            "R_curr": rec.get(cand, float("nan")),
            "R_next": rec.get(next_idx, float("nan")),
            "status": status,
        })
        log(f"Localized {ep}: status={status}, candidate={cand}, localized={localized}")

    rec_csv = paths.localization / "recoverability_results.csv"
    loc_csv = paths.localization / "localized_failure_chunks.csv"
    write_csv(rec_csv, recover_rows, ["episode_id", "chunk_index", "n_rollouts", "successes", "recoverability"]) if recover_rows else write_csv(rec_csv, [], ["episode_id", "chunk_index", "n_rollouts", "successes", "recoverability"])
    write_csv(loc_csv, loc_rows, ["episode_id", "candidate_chunk", "localized_failure_chunk", "R_prev", "R_curr", "R_next", "status"]) if loc_rows else write_csv(loc_csv, [], ["episode_id", "candidate_chunk", "localized_failure_chunk", "R_prev", "R_curr", "R_next", "status"])

    summary = {
        "num_candidates": len(loc_rows),
        "num_localized": int(sum(1 for r in loc_rows if str(r["localized_failure_chunk"]) != "")),
        "rho": rho,
        "coarse_n": coarse_n,
        "refine_n": refine_n,
        "verify_n": verify_n,
        "recoverability_csv": str(rec_csv),
        "localized_csv": str(loc_csv),
    }
    with (paths.localization / "localization_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    log(f"Localization summary: {summary}")



def summarize_results(paths: PipelinePaths):
    manifest = read_csv(paths.rollout_bank / "rollout_manifest.csv")
    failure_summary = json.loads((paths.failure_bank / "failure_bank_summary.json").read_text())
    loc_summary = json.loads((paths.localization / "localization_summary.json").read_text())
    loc_rows = read_csv(paths.localization / "localized_failure_chunks.csv")
    rec_rows = read_csv(paths.localization / "recoverability_results.csv")

    total = len(manifest)
    succ = sum(int(r["success"]) for r in manifest)
    fail = total - succ

    rollout_summary = {
        "num_episodes": total,
        "num_successes": succ,
        "num_failures": fail,
        "success_rate": float(succ / max(1, total)),
        "failure_rate": float(fail / max(1, total)),
        "num_localized": loc_summary["num_localized"],
        "num_candidates": loc_summary["num_candidates"],
    }

    with (paths.plots / "rollout_summary.json").open("w") as f:
        json.dump(rollout_summary, f, indent=2)
    with (paths.plots / "rollout_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in rollout_summary.items():
            w.writerow([k, v])

    with (paths.plots / "failure_localization_table.csv").open("w", newline="") as f:
        fieldnames = ["episode_id", "candidate_chunk", "localized_failure_chunk", "R_prev", "R_curr", "R_next", "status"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(loc_rows)

    plt.figure(figsize=(6, 4))
    plt.bar(["success", "failure"], [succ, fail])
    plt.title("PushT Rollout Counts")
    plt.ylabel("Episodes")
    plt.tight_layout()
    plt.savefig(paths.plots / "success_failure_counts.png", dpi=150)
    plt.close()

    localized_positions = [int(r["localized_failure_chunk"]) for r in loc_rows if str(r["localized_failure_chunk"]) not in ["", "None"]]
    plt.figure(figsize=(6, 4))
    if localized_positions:
        bins = np.arange(0, max(localized_positions) + 2) - 0.5
        plt.hist(localized_positions, bins=bins)
    plt.title("Localized Failure Chunk Histogram")
    plt.xlabel("Chunk index")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(paths.plots / "failure_chunk_histogram.png", dpi=150)
    plt.close()

    if rec_rows:
        ep0 = rec_rows[0]["episode_id"]
        ep_rows = [r for r in rec_rows if r["episode_id"] == ep0]
        x = [int(r["chunk_index"]) for r in ep_rows]
        y = [float(r["recoverability"]) for r in ep_rows]
        plt.figure(figsize=(7, 4))
        plt.scatter(x, y, alpha=0.8)
        plt.title(f"Recoverability vs Chunk ({ep0})")
        plt.xlabel("Chunk index")
        plt.ylabel("Recoverability")
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(paths.plots / "recoverability_vs_chunk.png", dpi=150)
        plt.close()


    # Additional observation plots for culprit analysis.
    status_counts = {}
    for r in loc_rows:
        st = r.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1
    if status_counts:
        plt.figure(figsize=(7, 4))
        keys = list(status_counts.keys())
        vals = [status_counts[k] for k in keys]
        plt.bar(keys, vals)
        plt.xticks(rotation=20, ha="right")
        plt.title("Localization Status Counts")
        plt.ylabel("Episodes")
        plt.tight_layout()
        plt.savefig(paths.plots / "localization_status_counts.png", dpi=150)
        plt.close()

    cand = []
    locd = []
    for r in loc_rows:
        if str(r.get("localized_failure_chunk", "")) not in ["", "None"]:
            cand.append(int(r["candidate_chunk"]))
            locd.append(int(r["localized_failure_chunk"]))
    if cand:
        plt.figure(figsize=(6, 6))
        plt.scatter(cand, locd, alpha=0.8)
        m = max(max(cand), max(locd))
        plt.plot([0, m], [0, m], "--", linewidth=1)
        plt.xlabel("Candidate chunk")
        plt.ylabel("Localized chunk")
        plt.title("Candidate vs Localized Chunk")
        plt.tight_layout()
        plt.savefig(paths.plots / "candidate_vs_localized.png", dpi=150)
        plt.close()

    diag_path = paths.localization / "candidate_diagnostics.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text())
        if diag:
            plt.figure(figsize=(8, 4.5))
            for ep, d in diag.items():
                rr = d.get("chunk_rewards", [])
                if rr:
                    plt.plot(list(range(len(rr))), rr, alpha=0.55, linewidth=1)
            plt.title("Failure Chunk Reward Profiles")
            plt.xlabel("Chunk index")
            plt.ylabel("Chunk max reward")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(paths.plots / "failure_chunk_reward_profiles.png", dpi=150)
            plt.close()

    if rec_rows:
        by_chunk = {}
        for r in rec_rows:
            idx = int(r["chunk_index"])
            by_chunk.setdefault(idx, []).append(float(r["recoverability"]))
        xs = sorted(by_chunk.keys())
        means = [float(np.mean(by_chunk[k])) for k in xs]
        stds = [float(np.std(by_chunk[k])) for k in xs]
        plt.figure(figsize=(7, 4))
        plt.errorbar(xs, means, yerr=stds, fmt="-o", capsize=3)
        plt.ylim(0, 1.05)
        plt.xlabel("Chunk index")
        plt.ylabel("Recoverability (mean +/- std)")
        plt.title("Recoverability by Chunk Index")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(paths.plots / "recoverability_by_chunk.png", dpi=150)
        plt.close()

    log(f"Summary/plots written to {paths.plots}")
    log(f"Failure bank summary: {failure_summary}")
    log(f"Localization summary: {loc_summary}")



def parse_args():
    parser = argparse.ArgumentParser(description="PushT failure localization pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", default="pusht")
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--output-root", required=True)
    common.add_argument("--guidance-mode", default="threshold_disabled", choices=["full", "threshold_disabled", "fully_disabled"])
    common.add_argument("--episodes", type=int, default=50)
    common.add_argument("--test-start-seed", type=int, default=200000)
    common.add_argument("--max-steps", type=int, default=300)
    common.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    common.add_argument("--device", default="cuda")
    common.add_argument("--force", action="store_true")

    for name in ["collect_rollouts", "build_failure_bank", "propose_failure_chunks", "summarize_results", "run_pipeline"]:
        sub.add_parser(name, parents=[common])

    loc = sub.add_parser("localize_failure_chunks", parents=[common])
    loc.add_argument("--rho", type=float, default=0.3)
    loc.add_argument("--coarse-n", type=int, default=3)
    loc.add_argument("--refine-n", type=int, default=5)
    loc.add_argument("--verify-n", type=int, default=10)

    run = sub.choices["run_pipeline"]
    run.add_argument("--rho", type=float, default=0.3)
    run.add_argument("--coarse-n", type=int, default=3)
    run.add_argument("--refine-n", type=int, default=5)
    run.add_argument("--verify-n", type=int, default=10)

    return parser.parse_args()



def main():
    args = parse_args()
    cfg = RunConfig(
        task=args.task,
        ckpt=Path(args.checkpoint).resolve(),
        output_root=Path(args.output_root).resolve(),
        guidance_mode=args.guidance_mode,
        episodes=args.episodes,
        test_start_seed=args.test_start_seed,
        max_steps=args.max_steps,
        chunk_size=args.chunk_size,
        device=args.device,
        force=args.force,
    )

    model_name = cfg.ckpt.stem
    paths = build_paths(cfg.output_root, cfg.task, model_name, cfg.guidance_mode)

    def need_policy(cmd: str) -> bool:
        return cmd in {"collect_rollouts", "localize_failure_chunks", "run_pipeline"}

    policy = None
    if need_policy(args.cmd):
        policy = PolicyWrapper(cfg.ckpt, cfg.device)
        policy.load()

    if args.cmd == "collect_rollouts":
        collect_rollouts(cfg, paths, policy)
    elif args.cmd == "build_failure_bank":
        build_failure_bank(paths)
    elif args.cmd == "propose_failure_chunks":
        propose_candidate_chunks(paths)
    elif args.cmd == "localize_failure_chunks":
        localize_failure_chunks(paths, policy, rho=args.rho, coarse_n=args.coarse_n, refine_n=args.refine_n, verify_n=args.verify_n, max_steps=cfg.max_steps)
    elif args.cmd == "summarize_results":
        summarize_results(paths)
    elif args.cmd == "run_pipeline":
        collect_rollouts(cfg, paths, policy)
        build_failure_bank(paths)
        propose_candidate_chunks(paths)
        localize_failure_chunks(paths, policy, rho=args.rho, coarse_n=args.coarse_n, refine_n=args.refine_n, verify_n=args.verify_n, max_steps=cfg.max_steps)
        summarize_results(paths)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
