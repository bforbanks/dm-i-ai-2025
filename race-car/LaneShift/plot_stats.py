#!/usr/bin/env python3
"""
Generate summary statistics and plots for laneshift_dataset.npz.

Run from the project root (dm-i-ai-2025/):
    python race-car/LaneShift/plot_stats.py [path/to/dataset.npz] [--bins 30] [--out-dir .]

Outputs:
    distance_histogram.png  – probability histogram of per-game distances
    stats.txt               – full text summary
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt


def load(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def print_and_write(lines: list[str], f):
    for line in lines:
        print(line)
        f.write(line + "\n")


def run(dataset_path: str, bins: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    d = load(dataset_path)

    lengths    = d["game_lengths"].astype(int)
    distances  = d["game_distances"].astype(float)
    car_x      = d["car_x"]
    car_vx     = d["car_vx"]
    sensors    = d["sensors"]

    n_games     = len(lengths)
    total_ticks = int(lengths.sum())
    MAX_TICKS   = 60 * 60
    crashed     = lengths < MAX_TICKS

    stats_path = os.path.join(out_dir, "stats.txt")
    with open(stats_path, "w") as f:
        lines = [
            "=" * 60,
            "  LaneShift Dataset Summary",
            "=" * 60,
            f"  Source : {os.path.abspath(dataset_path)}",
            "",
            "── Games ──────────────────────────────────────────────────",
            f"  Total games      : {n_games:,}",
            f"  Total ticks      : {total_ticks:,}",
            f"  Crashed          : {crashed.sum():,}  ({crashed.mean()*100:.1f}%)",
            f"  Timeout          : {(~crashed).sum():,}  ({(~crashed).mean()*100:.1f}%)",
            "",
            "── Game length (ticks) ─────────────────────────────────────",
            f"  mean   : {lengths.mean():.1f}",
            f"  median : {np.median(lengths):.1f}",
            f"  std    : {lengths.std():.1f}",
            f"  min    : {lengths.min()}",
            f"  max    : {lengths.max()}",
            f"  p25    : {np.percentile(lengths, 25):.0f}",
            f"  p75    : {np.percentile(lengths, 75):.0f}",
            "",
            "── Distance (px driven by ego) ─────────────────────────────",
            f"  mean   : {distances.mean():.0f}",
            f"  median : {np.median(distances):.0f}",
            f"  std    : {distances.std():.0f}",
            f"  min    : {distances.min():.0f}",
            f"  max    : {distances.max():.0f}",
            f"  p25    : {np.percentile(distances, 25):.0f}",
            f"  p75    : {np.percentile(distances, 75):.0f}",
            f"  p90    : {np.percentile(distances, 90):.0f}",
            "",
            "── Lane occupancy (fraction of ticks a car is present) ─────",
        ]
        for lane in range(5):
            occ = np.mean(~np.isnan(car_x[:, lane]))
            lines.append(f"  Lane {lane+1} : {occ*100:.1f}%")

        lines += [
            "",
            "── NPC relative velocity |vx| per lane ────────────────────",
        ]
        for lane in range(5):
            vals = car_vx[:, lane]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                lines.append(
                    f"  Lane {lane+1} : mean={vals.mean():.3f}  std={vals.std():.3f}"
                    f"  min={vals.min():.2f}  max={vals.max():.2f}"
                )
            else:
                lines.append(f"  Lane {lane+1} : no data")

        lines += [
            "",
            "── Sensor fire rate (fraction of ticks with a reading) ─────",
        ]
        SENSOR_LABELS = [
            "front",      "right_front",  "right_side",  "right_back",
            "back",       "left_back",    "left_side",   "left_front",
            "l_side_frt", "front_l_frt",  "front_r_frt", "r_side_frt",
            "r_side_bck", "back_r_bck",   "back_l_bck",  "l_side_bck",
        ]
        for i, label in enumerate(SENSOR_LABELS):
            rate = np.mean(~np.isnan(sensors[:, i]))
            lines.append(f"  {label:<12} : {rate*100:.1f}%")

        lines += ["", "=" * 60]
        print_and_write(lines, f)

    print(f"\nStats written to {stats_path}")

    # ── Distance probability histogram ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    counts, edges = np.histogram(distances, bins=bins)
    probs = counts / counts.sum()
    ax.bar(edges[:-1], probs, width=np.diff(edges), align="edge",
           color="#2196a8", edgecolor="white", linewidth=0.4)

    ax.set_xlabel("Distance", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"Probability Histogram of Distances ({bins} bins)", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    hist_path = os.path.join(out_dir, "distance_histogram.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Histogram saved to {hist_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path",      nargs="?", default="laneshift_dataset.npz")
    ap.add_argument("--bins",    type=int,  default=30)
    ap.add_argument("--out-dir", type=str,  default=".")
    args = ap.parse_args()
    run(args.path, args.bins, args.out_dir)
