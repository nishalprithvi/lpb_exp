#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    radius = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    lo = max(0.0, center - radius)
    hi = min(1.0, center + radius)
    return float(lo), float(hi)


def mean_ci(values):
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mu = float(np.mean(arr))
    if n == 1:
        return mu, mu, mu
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    delta = 1.96 * se
    return mu, mu - delta, mu + delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title-prefix", default="Phase-2")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    scenarios = sorted(set(r["scenario"] for r in rows))
    summary = []
    for scenario in scenarios:
        subset = [r for r in rows if r["scenario"] == scenario]
        n = len(subset)
        success_vals = [int(float(r["success"])) for r in subset]
        collision_vals = [int(float(r["collision"])) for r in subset]
        episode_lengths = [float(r["episode_length"]) for r in subset]

        success_sum = int(sum(success_vals))
        collision_sum = int(sum(collision_vals))
        success_rate = success_sum / n if n > 0 else 0.0
        collision_rate = collision_sum / n if n > 0 else 0.0
        success_lo, success_hi = wilson_ci(success_sum, n)
        collision_lo, collision_hi = wilson_ci(collision_sum, n)
        avg_len, avg_len_lo, avg_len_hi = mean_ci(episode_lengths)

        summary.append({
            "scenario": scenario,
            "success_rate": float(success_rate),
            "success_ci95_low": float(success_lo),
            "success_ci95_high": float(success_hi),
            "collision_rate": float(collision_rate),
            "collision_ci95_low": float(collision_lo),
            "collision_ci95_high": float(collision_hi),
            "avg_episode_length": float(avg_len),
            "avg_episode_length_ci95_low": float(avg_len_lo),
            "avg_episode_length_ci95_high": float(avg_len_hi),
            "num_episodes": int(n),
        })

    summary_csv = output_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    summary_json = output_dir / "summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    labels = [s["scenario"] for s in summary]
    success_values = [s["success_rate"] for s in summary]
    success_yerr = [
        [s["success_rate"] - s["success_ci95_low"] for s in summary],
        [s["success_ci95_high"] - s["success_rate"] for s in summary],
    ]
    collision_values = [s["collision_rate"] for s in summary]
    collision_yerr = [
        [s["collision_rate"] - s["collision_ci95_low"] for s in summary],
        [s["collision_ci95_high"] - s["collision_rate"] for s in summary],
    ]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, success_values, yerr=success_yerr, capsize=4)
    plt.ylim(0, 1)
    plt.ylabel("Success Rate")
    plt.title(f"{args.title_prefix} Success Rate (95% CI)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "success_rate.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(labels, collision_values, yerr=collision_yerr, capsize=4)
    plt.ylim(0, 1)
    plt.ylabel("Collision Rate")
    plt.title(f"{args.title_prefix} Collision Rate (95% CI)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "collision_rate.png", dpi=150)
    plt.close()

    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
