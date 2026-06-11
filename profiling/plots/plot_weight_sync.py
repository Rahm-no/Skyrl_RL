"""Task 3 plot — weight sync phase breakdown horizontal bar chart."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Path to weight_sync.jsonl")
    args = parser.parse_args()

    records = load(Path(args.jsonl))
    out_dir = Path(args.jsonl).parent

    if not records:
        print("No records in weight_sync.jsonl")
        return

    totals      = np.array([r["total_s"] for r in records]) * 1000
    gather_est  = np.array([r["gather_s_est"] for r in records]) * 1000
    bcast_est   = np.array([r["broadcast_s_est"] for r in records]) * 1000
    load_ms     = np.array([r["load_s_est"] for r in records]) * 1000
    bws         = np.array([r["achieved_bw_gbs"] for r in records])
    peak_bw     = records[0]["theoretical_peak_gbs"]
    model_gb    = records[0]["model_size_gb"]

    # ── horizontal stacked bar for one representative sync (median) ──────────
    mid = len(records) // 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 2]})

    # top: bandwidth over syncs
    ax1.plot(range(len(records)), bws, marker="o", color="#4C72B0", linewidth=1.5, markersize=4)
    ax1.axhline(peak_bw, color="red", linestyle="--", linewidth=1, label=f"NVLink peak ({peak_bw} GB/s)")
    ax1.set_ylabel("Achieved BW (GB/s)", fontsize=10)
    ax1.set_title("Achieved Bandwidth per Weight Sync", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlabel("Sync index", fontsize=10)

    # bottom: phase breakdown bar (mean values)
    phases = ["Gather\n(FSDP all-gather)", "Broadcast\n(NCCL → vLLM)", "Load\n(vLLM install)"]
    values = [gather_est.mean(), bcast_est.mean(), load_ms.mean()]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax2.barh(phases, values, color=colors, height=0.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f} ms", va="center", fontsize=10)
    ax2.set_xlabel("Time (ms)", fontsize=11)
    ax2.set_title(f"Mean Weight Sync Phase Breakdown  (total = {totals.mean():.1f} ms, model = {model_gb:.2f} GB BF16)", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)

    # annotate bandwidth
    ax2.text(0.98, 0.05,
             f"Mean BW: {bws.mean():.1f} GB/s\nEfficiency: {bws.mean()/peak_bw*100:.0f}% of NVLink peak",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    out = out_dir / "weight_sync.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

    print(f"\n── Weight Sync Summary ──────────────────────────────────")
    print(f"  Syncs:           {len(records)}")
    print(f"  Total time:      {totals.mean():.1f} ms  (p95: {np.percentile(totals,95):.1f} ms)")
    print(f"  Achieved BW:     {bws.mean():.1f} GB/s  (peak: {peak_bw} GB/s)")
    print(f"  Efficiency:      {bws.mean()/peak_bw*100:.0f}%")
    print(f"  Bottleneck:      {'gather' if gather_est.mean() > bcast_est.mean() else 'broadcast'}")
    print(f"  Phases (est.):   gather={gather_est.mean():.1f} ms  broadcast={bcast_est.mean():.1f} ms  load={load_ms.mean():.1f} ms")
    print(f"  Note: gather/broadcast are modelled from NVLink BW; load is the remainder.")


if __name__ == "__main__":
    main()
