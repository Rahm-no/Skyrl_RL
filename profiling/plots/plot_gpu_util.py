"""Task 5 plot — GPU SM utilization time-series with stage shading."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    parser.add_argument("csv", help="Path to gpu_util.csv")
    args = parser.parse_args()

    out_dir = Path(args.csv).parent
    df = pd.read_csv(args.csv)
    if df.empty:
        print("No GPU util data.")
        return

    for col in ["elapsed_s", "sm_util_pct", "mem_util_pct", "power_w", "temp_c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gpu"] = df["gpu"].astype(str)

    # ── Average across all GPUs at each timestamp ────────────────────────────
    avg = (df.groupby(["elapsed_s", "stage"], sort=False)
             .agg(sm_util_pct=("sm_util_pct", "mean"))
             .reset_index()
             .sort_values("elapsed_s"))

    t     = avg["elapsed_s"].values
    util  = avg["sm_util_pct"].values
    stage_arr = avg["stage"].values

    fig, ax = plt.subplots(figsize=(18, 5))

    # map raw stages to high-level phase names
    avg["phase"] = avg["stage"].map(
        lambda s: HIGH_LEVEL.get(s, ("Overhead", "#bbbbbb"))[0]
    )

    # shade by high-level phase
    phase_arr = avg["phase"].values
    changes = [0]
    for i in range(1, len(phase_arr)):
        if phase_arr[i] != phase_arr[i - 1]:
            changes.append(i)
    changes.append(len(phase_arr))

    phase_color = {label: color for label, color in HIGH_LEVEL.values()}
    seen = set()
    for i in range(len(changes) - 1):
        s_idx, e_idx = changes[i], changes[i + 1]
        phase = phase_arr[s_idx]
        c = phase_color.get(phase, "#eeeeee")
        lbl = phase if phase not in seen else "_nolegend_"
        seen.add(phase)
        ax.axvspan(t[s_idx], t[e_idx - 1], alpha=0.22, color=c, label=lbl)

    overall_avg = float(np.mean(util))
    ax.plot(t, util, color="#222222", linewidth=1.2, alpha=0.9)
    ax.axhline(overall_avg, color="#1f77b4", linestyle="-", linewidth=1.2,
               label=f"Avg GPU util ({overall_avg:.0f}%)")
    ax.set_ylabel("SM Utilization (%)", fontsize=11)
    ax.set_xlabel("Elapsed time (s)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)

    gpus = df["gpu"].unique()

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    seen_phases = {}
    for label, color in HIGH_LEVEL.values():
        if label not in seen_phases:
            seen_phases[label] = color
    phase_handles = [Patch(color=c, alpha=0.5, label=l) for l, c in seen_phases.items()]
    line_handles  = [Line2D([0], [0], color="#222222", linewidth=1.5, label="Avg SM util (all GPUs)"),
                     Line2D([0], [0], color="#1f77b4", linewidth=1.2, label=f"Avg GPU util ({overall_avg:.0f}%)")]
    ax.legend(handles=phase_handles + line_handles,
              loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=len(phase_handles) + len(line_handles),
              fontsize=9, framealpha=0.9)

    fig.suptitle(f"GPU SM Utilization — Average over {len(gpus)} GPUs", fontsize=14)
    fig.tight_layout()
    out = out_dir / "gpu_util.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")

    # ── per-phase summary ─────────────────────────────────────────────────────
    print(f"\n── GPU Util by Phase (averaged over {len(gpus)} GPUs) ──────────")
    for phase in dict.fromkeys(l for l, _ in HIGH_LEVEL.values()):
        sdf = avg[avg["phase"] == phase]
        if sdf.empty:
            continue
        vals = sdf["sm_util_pct"].values
        print(f"  {phase:<26}: mean={vals.mean():.0f}%  median={np.median(vals):.0f}%")


if __name__ == "__main__":
    main()
