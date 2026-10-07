"""Summary plots from monitor CSV: average CPU, RAM, GPU util, VRAM (pie + bar)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Drop idle rows at start/end where all GPUs are 0 util
    gpu_util_cols = [c for c in df.columns if c.endswith("_util_pct") and "mem" not in c]
    active = df[gpu_util_cols].sum(axis=1) > 0
    first, last = active.idxmax(), len(df) - active[::-1].values.argmax() - 1
    return df.iloc[first:last].reset_index(drop=True)


def plot_averages_bar(df: pd.DataFrame, out_dir: Path):
    gpu_ids = [0, 1, 2, 3]
    avg_cpu = df["cpu_pct"].mean()
    avg_ram_pct = (df["ram_used_mb"] / df["ram_total_mb"] * 100).mean()
    avg_gpu_util = [df[f"gpu{i}_util_pct"].mean() for i in gpu_ids]
    avg_vram_pct = [(df[f"gpu{i}_mem_used_mb"] / df[f"gpu{i}_mem_total_mb"] * 100).mean() for i in gpu_ids]
    avg_power = [df[f"gpu{i}_power_w"].mean() for i in gpu_ids]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Average Resource Utilisation (active training window)", fontsize=13, fontweight="bold")

    # --- GPU compute util (%) ---
    ax = axes[0, 0]
    bars = ax.bar([f"GPU {i}" for i in gpu_ids], avg_gpu_util, color=colors, edgecolor="white")
    ax.axhline(np.mean(avg_gpu_util), color="black", linestyle="--", linewidth=1,
               label=f"Mean {np.mean(avg_gpu_util):.1f}%")
    for bar, val in zip(bars, avg_gpu_util):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Utilisation %")
    ax.set_title("Avg GPU Compute Utilisation (%)")
    ax.legend(fontsize=8)

    # --- VRAM (%) ---
    ax = axes[0, 1]
    bars = ax.bar([f"GPU {i}" for i in gpu_ids], avg_vram_pct, color=colors, edgecolor="white")
    ax.axhline(np.mean(avg_vram_pct), color="black", linestyle="--", linewidth=1,
               label=f"Mean {np.mean(avg_vram_pct):.1f}%")
    for bar, val in zip(bars, avg_vram_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("VRAM %")
    ax.set_title("Avg GPU VRAM Usage (%)")
    ax.legend(fontsize=8)

    # --- CPU & RAM (%) ---
    ax = axes[1, 0]
    bars = ax.bar(["CPU", "RAM"], [avg_cpu, avg_ram_pct], color=["#8172B3", "#64B5CD"], edgecolor="white")
    for bar, val in zip(bars, [avg_cpu, avg_ram_pct]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("%")
    ax.set_title("Avg CPU & RAM Utilisation (%)")

    # --- GPU Power (W) — separate axis, clearly in watts ---
    ax = axes[1, 1]
    bars = ax.bar([f"GPU {i}" for i in gpu_ids], avg_power, color=colors, edgecolor="white")
    ax.axhline(np.mean(avg_power), color="black", linestyle="--", linewidth=1,
               label=f"Mean {np.mean(avg_power):.0f} W")
    for bar, val in zip(bars, avg_power):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val:.0f}W", ha="center", fontsize=9)
    ax.set_ylabel("Power (W)")
    ax.set_title("Avg GPU Power Draw (Watts)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = out_dir / "avg_resource_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_utilisation_pie(df: pd.DataFrame, out_dir: Path):
    gpu_ids = [0, 1, 2, 3]
    avg_gpu_util = [df[f"gpu{i}_util_pct"].mean() for i in gpu_ids]
    avg_vram_pct = [(df[f"gpu{i}_mem_used_mb"] / df[f"gpu{i}_mem_total_mb"] * 100).mean() for i in gpu_ids]
    avg_cpu = df["cpu_pct"].mean()
    avg_ram_pct = (df["ram_used_mb"] / df["ram_total_mb"] * 100).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Resource Utilisation Breakdown (avg over active training window)", fontsize=12, fontweight="bold")

    # Pie 1: GPU compute
    labels_gpu = [f"GPU {i}\n{v:.1f}%" for i, v in enumerate(avg_gpu_util)]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    wedges, texts = axes[0].pie(
        avg_gpu_util, labels=labels_gpu, colors=colors,
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=10),
    )
    axes[0].set_title(f"GPU Compute Util\n(overall mean: {np.mean(avg_gpu_util):.1f}%)", fontsize=11)

    # Pie 2: VRAM
    labels_vram = [f"GPU {i}\n{v:.1f}%" for i, v in enumerate(avg_vram_pct)]
    wedges2, texts2 = axes[1].pie(
        avg_vram_pct, labels=labels_vram, colors=colors,
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=10),
    )
    axes[1].set_title(f"GPU VRAM Usage\n(overall mean: {np.mean(avg_vram_pct):.1f}%)", fontsize=11)

    fig.text(0.5, 0.01,
             f"CPU avg: {avg_cpu:.1f}%   |   RAM avg: {avg_ram_pct:.1f}%",
             ha="center", fontsize=10, color="#444")

    plt.tight_layout()
    out = out_dir / "avg_resource_pie.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_timeseries_stacked(df: pd.DataFrame, out_dir: Path):
    """Time-series: mean across 4 GPUs at each timestep for util and VRAM, plus CPU/RAM."""
    gpu_ids = [0, 1, 2, 3]
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 3600

    mean_gpu_util = np.mean([df[f"gpu{i}_util_pct"].values for i in gpu_ids], axis=0)
    mean_vram_gb = np.mean([df[f"gpu{i}_mem_used_mb"].values for i in gpu_ids], axis=0) / 1024
    total_gb = df["gpu0_mem_total_mb"].iloc[0] / 1024
    ram_pct = df["ram_used_mb"] / df["ram_total_mb"] * 100

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Resource Usage Over Training Time (mean across 4 GPUs)", fontsize=13, fontweight="bold")

    # Mean GPU util
    ax = axes[0]
    ax.plot(t, mean_gpu_util, color="#4C72B0", linewidth=1.2, label="Mean GPU util")
    ax.fill_between(t, mean_gpu_util, alpha=0.25, color="#4C72B0")
    ax.axhline(mean_gpu_util.mean(), color="black", linestyle="--", linewidth=1,
               label=f"Avg {mean_gpu_util.mean():.1f}%")
    ax.set_ylabel("GPU Util %")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Mean GPU Compute Utilisation (avg of GPU 0-3)")

    # Mean VRAM
    ax = axes[1]
    ax.plot(t, mean_vram_gb, color="#DD8452", linewidth=1.2, label="Mean VRAM used")
    ax.fill_between(t, mean_vram_gb, alpha=0.25, color="#DD8452")
    ax.axhline(total_gb, color="gray", linestyle=":", linewidth=1, label=f"Max {total_gb:.0f} GB")
    ax.axhline(mean_vram_gb.mean(), color="black", linestyle="--", linewidth=1,
               label=f"Avg {mean_vram_gb.mean():.1f} GB")
    ax.set_ylabel("VRAM (GB)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Mean GPU VRAM Usage (avg of GPU 0-3)")

    # CPU + RAM
    ax = axes[2]
    ax.plot(t, df["cpu_pct"], color="#8172B3", linewidth=1.2, label=f"CPU % (avg {df['cpu_pct'].mean():.1f}%)")
    ax.plot(t, ram_pct, color="#64B5CD", linewidth=1.2, label=f"RAM % (avg {ram_pct.mean():.1f}%)")
    ax.fill_between(t, df["cpu_pct"], alpha=0.15, color="#8172B3")
    ax.fill_between(t, ram_pct, alpha=0.15, color="#64B5CD")
    ax.set_ylabel("%")
    ax.set_xlabel("Elapsed time (h)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("CPU & RAM Utilisation")

    plt.tight_layout()
    out = out_dir / "resource_timeseries_stacked.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to monitor_<jobid>.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(csv_path)
    print(f"Loaded {len(df)} rows (active training window)")

    plot_averages_bar(df, out_dir)
    plot_utilisation_pie(df, out_dir)
    plot_timeseries_stacked(df, out_dir)


if __name__ == "__main__":
    main()
