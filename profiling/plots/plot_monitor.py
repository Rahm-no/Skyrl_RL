"""Plot system resource monitor CSV (CPU, RAM, disk, per-GPU util/VRAM/power/temp)."""
import argparse
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

GPU_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to monitor_<jobid>.csv")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: same as csv)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data.")
        return

    # Parse elapsed time from timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    t = df["elapsed_s"].values

    # ── Figure 1: System resources (CPU, RAM, Disk) ──────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, df["cpu_pct"].values, color="#4C72B0", linewidth=1.5)
    axes[0].set_ylabel("CPU %", fontsize=11)
    axes[0].set_title("CPU Utilization (job-scoped, 32 cores)", fontsize=12)
    axes[0].axhline(100, color="red", linestyle="--", linewidth=0.8, label="100%")
    axes[0].set_ylim(0, max(df["cpu_pct"].max() * 1.15, 10))
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    ram_used = df["ram_used_mb"].values / 1024
    ram_total = df["ram_total_mb"].values[0] / 1024
    axes[1].fill_between(t, 0, ram_used, alpha=0.5, color="#55A868")
    axes[1].plot(t, ram_used, color="#55A868", linewidth=1.2)
    axes[1].axhline(ram_total, color="red", linestyle="--", linewidth=0.8,
                    label=f"Total {ram_total:.0f} GiB")
    axes[1].set_ylabel("RAM (GiB)", fontsize=11)
    axes[1].set_title(f"RAM Usage  (total {ram_total:.0f} GiB)", fontsize=12)
    axes[1].set_ylim(0, ram_total * 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    axes[2].plot(t, df["disk_read_mbps"].values,  color="#4C72B0", linewidth=1.2, label="Read")
    axes[2].plot(t, df["disk_write_mbps"].values, color="#DD8452", linewidth=1.2, label="Write")
    axes[2].set_ylabel("MB/s", fontsize=11)
    axes[2].set_title("Disk I/O", fontsize=12)
    axes[2].set_xlabel("Elapsed time (s)", fontsize=11)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=9)

    fig.suptitle(f"System Resources — {csv_path.name}", fontsize=14)
    fig.tight_layout()
    out1 = out_dir / "monitor_system.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── Figure 2: GPU metrics (util, VRAM, power, temp) ─────────────────────
    n_gpus = sum(1 for g in range(4) if f"gpu{g}_util_pct" in df.columns
                 and df[f"gpu{g}_mem_used_mb"].max() > 10)
    if n_gpus == 0:
        print("No GPU data found.")
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    titles = ["SM Utilization (%)", "VRAM Used (GiB)", "Power Draw (W)", "Temperature (°C)"]
    keys   = ["util_pct", "mem_used_mb", "power_w", "temp_c"]
    scales = [1, 1/1024, 1, 1]

    for gi in range(n_gpus):
        col = GPU_COLORS[gi]
        for ai, (key, scale) in enumerate(zip(keys, scales)):
            col_name = f"gpu{gi}_{key}"
            if col_name not in df.columns:
                continue
            vals = pd.to_numeric(df[col_name], errors="coerce").fillna(0).values * scale
            axes[ai].plot(t, vals, color=col, linewidth=1.2, alpha=0.85, label=f"GPU {gi}")

    vram_total = df["gpu0_mem_total_mb"].values[0] / 1024 if "gpu0_mem_total_mb" in df.columns else 40
    axes[0].axhline(80, color="red",    linestyle="--", linewidth=0.8, label="80% (compute-bound)")
    axes[0].axhline(40, color="orange", linestyle="--", linewidth=0.8, label="40% (BW-bound)")
    axes[0].set_ylim(0, 105)
    axes[1].axhline(vram_total, color="red", linestyle="--", linewidth=0.8,
                    label=f"Total {vram_total:.0f} GiB")
    axes[1].set_ylim(0, vram_total * 1.05)

    for ai, title in enumerate(titles):
        axes[ai].set_ylabel(title.split("(")[0].strip(), fontsize=10)
        axes[ai].set_title(title, fontsize=11)
        axes[ai].grid(alpha=0.3)
        axes[ai].legend(fontsize=8, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Elapsed time (s)", fontsize=11)
    fig.suptitle(f"GPU Metrics — {csv_path.name}", fontsize=14)
    fig.tight_layout()
    out2 = out_dir / "monitor_gpu.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # ── Summary stats ────────────────────────────────────────────────────────
    print(f"\n── Monitor Summary ({len(df)} samples over {t[-1]:.0f}s) ──")
    print(f"  Peak CPU:   {df['cpu_pct'].max():.1f}%  mean={df['cpu_pct'].mean():.1f}%")
    print(f"  Peak RAM:   {(df['ram_used_mb'].max()/1024):.1f} GiB / {ram_total:.0f} GiB")
    for gi in range(n_gpus):
        util = pd.to_numeric(df[f"gpu{gi}_util_pct"], errors="coerce").dropna()
        vram = pd.to_numeric(df[f"gpu{gi}_mem_used_mb"], errors="coerce").dropna() / 1024
        pwr  = pd.to_numeric(df[f"gpu{gi}_power_w"], errors="coerce").dropna()
        print(f"  GPU {gi}: util mean={util.mean():.0f}% peak={util.max():.0f}%  "
              f"VRAM peak={vram.max():.1f}GiB  power mean={pwr.mean():.0f}W peak={pwr.max():.0f}W")


if __name__ == "__main__":
    main()
