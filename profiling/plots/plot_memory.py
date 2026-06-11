"""Task 2 plot — VRAM usage vs time averaged across all GPUs with stage shading."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Same high-level phase mapping as plot_gpu_util.py
HIGH_LEVEL = {
    "generate":               ("Generate (vLLM inference)", "#2CA02C"),
    "fwd_logprobs":           ("Train (FSDP compute)",      "#C44E52"),
    "advantage":              ("Train (FSDP compute)",      "#C44E52"),
    "policy_train":           ("Train (FSDP compute)",      "#C44E52"),
    "stage_switch":           ("Overhead",                  "#bbbbbb"),
    "weight_sync":            ("Overhead",                  "#bbbbbb"),
    "env_step":               ("Overhead",                  "#bbbbbb"),
    "convert_to_training_input": ("Overhead",               "#bbbbbb"),
    "eval":                   ("Eval",                      "#FF7F0E"),
    "init":                   ("Overhead",                  "#bbbbbb"),
    "done":                   ("Overhead",                  "#bbbbbb"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("memory_csv", help="Path to memory.csv")
    args = parser.parse_args()

    out_dir = Path(args.memory_csv).parent
    df = pd.read_csv(args.memory_csv)
    if df.empty:
        print("No data in memory.csv")
        return

    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    df["mem_used_mb"] = pd.to_numeric(df["mem_used_mb"], errors="coerce")
    df["mem_total_mb"] = pd.to_numeric(df["mem_total_mb"], errors="coerce")
    df["gpu"] = df["gpu"].astype(str)
    gpus = df["gpu"].unique()

    # Average across all GPUs at each timestamp
    avg = (df.groupby(["elapsed_s", "stage"], sort=False)
             .agg(
                 mem_used_mb=("mem_used_mb", "mean"),
                 mem_total_mb=("mem_total_mb", "mean"),
             )
             .reset_index()
             .sort_values("elapsed_s"))

    avg["phase"] = avg["stage"].map(lambda s: HIGH_LEVEL.get(s, ("Overhead", "#bbbbbb"))[0])

    t     = avg["elapsed_s"].values
    used  = avg["mem_used_mb"].values / 1024   # GiB
    total = avg["mem_total_mb"].values / 1024
    total_val = float(total[0]) if len(total) else 40.0

    fig, ax = plt.subplots(figsize=(18, 5))

    # Shade background by high-level phase
    phase_color = {label: color for label, color in HIGH_LEVEL.values()}
    phase_arr = avg["phase"].values
    changes = [0]
    for i in range(1, len(phase_arr)):
        if phase_arr[i] != phase_arr[i - 1]:
            changes.append(i)
    changes.append(len(phase_arr))

    seen = set()
    for i in range(len(changes) - 1):
        s_idx, e_idx = changes[i], changes[i + 1]
        phase = phase_arr[s_idx]
        c = phase_color.get(phase, "#eeeeee")
        lbl = phase if phase not in seen else "_nolegend_"
        seen.add(phase)
        ax.axvspan(t[s_idx], t[e_idx - 1], alpha=0.22, color=c, label=lbl)

    # VRAM fill areas
    ax.fill_between(t, 0, used, alpha=0.55, color="#4C72B0", label="Used VRAM (avg)")
    ax.fill_between(t, used, total, alpha=0.12, color="#aaaaaa", label="Free VRAM (avg)")
    ax.axhline(total_val, color="red", linestyle="--", linewidth=0.8,
               label=f"Total ({total_val:.0f} GiB)")

    # Annotate peak
    peak_idx = int(np.argmax(used))
    ax.annotate(
        f"Peak: {used[peak_idx]:.1f} GiB\n(headroom: {(total[peak_idx] - used[peak_idx]):.1f} GiB)",
        xy=(t[peak_idx], used[peak_idx]),
        xytext=(t[peak_idx] + max(t) * 0.02, used[peak_idx] * 0.85),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    ax.set_ylabel("VRAM (GiB)", fontsize=11)
    ax.set_xlabel("Elapsed time (s)", fontsize=11)
    ax.set_ylim(0, total_val * 1.05)
    ax.grid(alpha=0.25)

    # Legend — phase patches first, then VRAM series
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    seen_phases: dict[str, str] = {}
    for label, color in HIGH_LEVEL.values():
        if label not in seen_phases:
            seen_phases[label] = color
    phase_handles = [Patch(color=c, alpha=0.5, label=l) for l, c in seen_phases.items()]
    vram_handles = [
        Patch(color="#4C72B0", alpha=0.55, label="Used VRAM (avg)"),
        Patch(color="#aaaaaa", alpha=0.3,  label="Free VRAM (avg)"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=0.8,
               label=f"Total ({total_val:.0f} GiB)"),
    ]
    ax.legend(
        handles=phase_handles + vram_handles,
        loc="upper center", bbox_to_anchor=(0.5, 1.13),
        ncol=len(phase_handles) + len(vram_handles),
        fontsize=9, framealpha=0.9,
    )

    fig.suptitle(f"GPU VRAM Usage — Average over {len(gpus)} GPUs", fontsize=14)
    fig.tight_layout()
    out = out_dir / "memory_waterfall.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")

    # Peak summary
    print(f"\n── VRAM Peak Summary (averaged over {len(gpus)} GPUs) ──────────")
    peak_used = float(used[peak_idx])
    print(f"  Peak: {peak_used:.2f} GiB / {total_val:.0f} GiB  headroom={total_val - peak_used:.2f} GiB")
    for gpu in sorted(gpus, key=lambda x: int(x) if x.isdigit() else 0):
        gdf = df[df["gpu"] == gpu]
        p = gdf["mem_used_mb"].max() / 1024
        tot = gdf["mem_total_mb"].max() / 1024
        peak_stage = gdf.loc[gdf["mem_used_mb"].idxmax(), "stage"] if not gdf.empty else "?"
        print(f"  GPU {gpu}:  peak={p:.2f} GiB / {tot:.0f} GiB  headroom={tot - p:.2f} GiB  at_stage={peak_stage}")


if __name__ == "__main__":
    main()
