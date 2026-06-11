"""Task 1 plot — stacked bar chart of per-stage times + summary table."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGES = ["generate", "stage_switch", "env_step", "fwd_logprobs", "advantage", "policy_train", "weight_sync"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]
LABELS = ["Generate (vLLM)", "Stage switch (sleep)", "Env step", "Fwd logprobs", "Advantage est.", "Policy train", "Weight sync"]


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
    parser.add_argument("jsonl", help="Path to stage_times.jsonl")
    parser.add_argument("--warmup", type=int, default=1, help="Epochs to skip as warmup")
    args = parser.parse_args()

    records = load(Path(args.jsonl))
    out_dir = Path(args.jsonl).parent

    # skip warmup epoch
    max_epoch = max(r["epoch"] for r in records) if records else 0
    if args.warmup > 0 and max_epoch >= args.warmup:
        records = [r for r in records if r["epoch"] >= args.warmup]

    if not records:
        print("No records after warmup. Lower --warmup.")
        return

    steps = [r["global_step"] for r in records]
    data = {s: [r.get(s, 0) for r in records] for s in STAGES}

    # ── stacked bar chart ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(steps) * 0.4), 6))
    bottoms = np.zeros(len(steps))
    for stage, color, label in zip(STAGES, COLORS, LABELS):
        vals = np.array(data[stage])
        ax.bar(steps, vals, bottom=bottoms, color=color, label=label, width=0.8)
        bottoms += vals

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Per-Stage Time Breakdown per Training Step", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "stage_times_stacked_bar.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

    # ── summary table ────────────────────────────────────────────────────────
    totals = [r.get("step_total", 0) for r in records]
    grand_mean = np.mean(totals)
    lit_pcts = {"generate": 65, "policy_train": 20, "weight_sync": 10}

    print(f"\n{'Stage':<22} {'Mean':>8} {'Median':>8} {'P95':>8} {'% Total':>9}  {'Lit. ref':>9}  Flag")
    print("-" * 80)
    for stage, label in zip(STAGES, LABELS):
        vals = np.array(data[stage])
        mean = np.mean(vals)
        median = np.median(vals)
        p95 = np.percentile(vals, 95)
        pct = 100.0 * mean / grand_mean if grand_mean > 0 else 0
        lit = lit_pcts.get(stage, None)
        lit_str = f"{lit}%" if lit else "n/a"
        flag = ""
        if stage not in ("generate", "policy_train", "weight_sync", "step_total") and pct > 10:
            flag = "← HIGH"
        print(f"{label:<22} {mean:>7.2f}s {median:>7.2f}s {p95:>7.2f}s {pct:>8.1f}%  {lit_str:>9}  {flag}")
    print(f"\n{'Step total':<22} {grand_mean:>7.2f}s")

    # ── pie chart (mean split) ────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    means = [np.mean(data[s]) for s in STAGES]
    other = max(0, grand_mean - sum(means))
    pie_colors = list(COLORS)
    pie_labels = list(LABELS)
    if other > 0.01:
        means.append(other)
        pie_colors.append("#aaaaaa")
        pie_labels.append("Other")

    def autopct_fmt(pct):
        return f"{pct:.1f}%" if pct >= 3 else ""

    wedges, _, autotexts = ax2.pie(
        means,
        colors=pie_colors[:len(means)],
        autopct=autopct_fmt,
        pctdistance=0.75,
        startangle=140,
        wedgeprops=dict(linewidth=0.5, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    legend_labels = [f"{l}  ({m:.1f}s, {m/grand_mean*100:.0f}%)"
                     for l, m in zip(pie_labels, means)]
    ax2.legend(wedges, legend_labels, loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)
    ax2.set_title("Mean Stage Time Distribution", fontsize=13)
    fig2.tight_layout()
    out2 = out_dir / "stage_times_pie.png"
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
