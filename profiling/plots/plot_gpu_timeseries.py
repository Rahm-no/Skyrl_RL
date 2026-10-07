#!/usr/bin/env python3
"""Plot per-GPU SM utilization time series with individual lines + average."""
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="gpu_util.csv from profiling output")
    parser.add_argument("--output", "-o", default=None, help="Output image path")
    parser.add_argument("--window", "-w", type=int, default=30, help="Rolling average window (in samples)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    gpu_ids = sorted(df["gpu"].unique())
    stages = df["stage"].unique()

    stage_colors = {
        "generate": "#e8f5e9",
        "postprocess_generator_output": "#fff3e0",
        "convert_to_training_input": "#fce4ec",
        "fwd_logprobs_values_reward": "#e3f2fd",
        "compute_advantages_and_returns": "#f3e5f5",
        "train_critic_and_policy": "#fff9c4",
        "sync_weights": "#e0f2f1",
        "eval": "#f5f5f5",
        "init": "#eeeeee",
        "load_checkpoints": "#eeeeee",
        "init_weight_sync_state": "#eeeeee",
        "save_checkpoints": "#e8eaf6",
    }

    fig, ax = plt.subplots(figsize=(20, 6))

    gpu_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    elapsed = df["elapsed_s"].values

    # Plot stage background bands
    stage_changes = df.drop_duplicates(subset=["elapsed_s"])[["elapsed_s", "stage"]].reset_index(drop=True)
    prev_stage = None
    band_start = 0
    for _, row in stage_changes.iterrows():
        if row["stage"] != prev_stage:
            if prev_stage is not None:
                color = stage_colors.get(prev_stage, "#f0f0f0")
                ax.axvspan(band_start / 60, row["elapsed_s"] / 60, alpha=0.3, color=color, lw=0)
            band_start = row["elapsed_s"]
            prev_stage = row["stage"]
    if prev_stage is not None:
        color = stage_colors.get(prev_stage, "#f0f0f0")
        ax.axvspan(band_start / 60, elapsed.max() / 60, alpha=0.3, color=color, lw=0)

    # Plot each GPU
    all_smooth = []
    for i, gid in enumerate(gpu_ids):
        gdf = df[df["gpu"] == gid].sort_values("elapsed_s")
        t = gdf["elapsed_s"].values / 60
        util = gdf["sm_util_pct"].values
        smooth = pd.Series(util).rolling(args.window, min_periods=1, center=True).mean().values
        ax.plot(t, smooth, color=gpu_colors[i % len(gpu_colors)], alpha=0.5, lw=0.8, label=f"GPU {gid}")
        all_smooth.append(pd.DataFrame({"t": t, "util": smooth}))

    # Compute and plot average across GPUs
    merged = all_smooth[0].rename(columns={"util": "gpu0"})
    for i in range(1, len(all_smooth)):
        merged[f"gpu{i}"] = all_smooth[i]["util"].values
    avg = merged[[c for c in merged.columns if c.startswith("gpu")]].mean(axis=1).values
    ax.plot(merged["t"].values, avg, color="black", lw=2, label="Average", zorder=10)

    # Add stage labels at top
    prev_stage = None
    band_start = 0
    for _, row in stage_changes.iterrows():
        if row["stage"] != prev_stage:
            if prev_stage is not None:
                mid = (band_start + row["elapsed_s"]) / 2 / 60
                width_min = (row["elapsed_s"] - band_start) / 60
                if width_min > 1.5:
                    short = prev_stage.replace("_", "\n")
                    ax.text(mid, 102, short, ha="center", va="bottom", fontsize=6, color="#555")
            band_start = row["elapsed_s"]
            prev_stage = row["stage"]
    if prev_stage is not None:
        mid = (band_start + elapsed.max()) / 2 / 60
        width_min = (elapsed.max() - band_start) / 60
        if width_min > 1.5:
            short = prev_stage.replace("_", "\n")
            ax.text(mid, 102, short, ha="center", va="bottom", fontsize=6, color="#555")

    ax.set_xlabel("Elapsed Time (minutes)")
    ax.set_ylabel("SM Utilization (%)")
    ax.set_title("GPU SM Utilization — Round-Robin APPS Run (job 1689780)")
    ax.set_ylim(-2, 110)
    ax.set_xlim(0, elapsed.max() / 60)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = args.output or args.csv.replace(".csv", "_timeseries.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
