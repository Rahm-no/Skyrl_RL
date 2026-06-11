"""Task 4 plot — long-tail rollout distribution (tokens, turns, CDF, scatter)."""
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
    parser.add_argument("jsonl", help="Path to rollout_stats.jsonl")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    args = parser.parse_args()

    records = load(Path(args.jsonl))
    out_dir = Path(args.jsonl).parent

    if not records:
        print("No rollout records.")
        return

    # skip warmup: take last 80% of records if warmup requested
    if args.warmup_epochs > 0:
        cutoff = max(1, int(len(records) * 0.2))
        records = records[cutoff:]

    tokens = np.array([r["total_response_tokens"] for r in records])
    turns  = np.array([r["num_turns"] for r in records])
    wall_s = np.array([r["wall_s"] for r in records])
    solved = np.array([r["solve_turn"] > 0 for r in records])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── 1. Histogram of total tokens ─────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(tokens, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
    p50, p99 = np.percentile(tokens, 50), np.percentile(tokens, 99)
    ax.axvline(p50, color="orange", linestyle="--", label=f"P50={p50:.0f}")
    ax.axvline(p99, color="red",    linestyle="--", label=f"P99={p99:.0f}")
    ax.set_xlabel("Total response tokens", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Token Distribution per Prompt", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    ratio = p99 / p50 if p50 > 0 else 0
    ax.text(0.97, 0.95, f"P99/P50 = {ratio:.1f}x", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # ── 2. Histogram of turns used ───────────────────────────────────────────
    ax = axes[0, 1]
    max_turn = int(turns.max()) if len(turns) else 5
    ax.hist(turns, bins=np.arange(0.5, max_turn + 1.5, 1), color="#55A868", edgecolor="white")
    ax.set_xlabel("Number of turns", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Turn Count per Prompt", fontsize=12)
    ax.set_xticks(range(1, max_turn + 1))
    ax.grid(alpha=0.3)
    solve_rate = solved.mean() * 100
    ax.text(0.97, 0.95, f"Solve rate: {solve_rate:.1f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # ── 3. CDF of wall-clock time (Pareto curve) ─────────────────────────────
    ax = axes[1, 0]
    sorted_wall = np.sort(wall_s)[::-1]
    cum_frac = np.cumsum(sorted_wall) / sorted_wall.sum()
    prompt_frac = np.arange(1, len(sorted_wall) + 1) / len(sorted_wall)
    ax.plot(prompt_frac * 100, cum_frac * 100, color="#C44E52", linewidth=2)
    ax.axvline(10, color="gray", linestyle=":", linewidth=1)
    pct_time_from_slowest_10 = cum_frac[int(len(cum_frac) * 0.1)] * 100
    ax.axhline(pct_time_from_slowest_10, color="gray", linestyle=":", linewidth=1)
    ax.fill_between(prompt_frac[:int(len(prompt_frac)*0.1)]*100,
                    cum_frac[:int(len(cum_frac)*0.1)]*100,
                    alpha=0.2, color="red", label=f"Slowest 10%: {pct_time_from_slowest_10:.0f}% of time")
    ax.set_xlabel("Slowest X% of prompts", fontsize=11)
    ax.set_ylabel("% of total generation time", fontsize=11)
    ax.set_title("Pareto: Prompt Fraction vs Generation Time", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 4. Scatter: turns vs tokens ───────────────────────────────────────────
    ax = axes[1, 1]
    scatter = ax.scatter(turns + np.random.uniform(-0.15, 0.15, size=len(turns)),
                         tokens, c=wall_s, cmap="RdYlGn_r", alpha=0.4, s=8)
    plt.colorbar(scatter, ax=ax, label="Wall time (s)")
    ax.set_xlabel("Turns used", fontsize=11)
    ax.set_ylabel("Total response tokens", fontsize=11)
    ax.set_title("Turns vs Tokens (color = wall time)", fontsize=12)
    ax.set_xticks(range(1, max_turn + 1))
    ax.grid(alpha=0.3)

    fig.suptitle("Multi-Turn Rollout Long-Tail Analysis", fontsize=14)
    fig.tight_layout()
    out = out_dir / "rollout_analysis.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

    print(f"\n── Rollout Summary ──────────────────────────────────────────")
    print(f"  Prompts:          {len(records)}")
    print(f"  Solve rate:       {solve_rate:.1f}%")
    print(f"  Tokens P50/P99:   {p50:.0f} / {p99:.0f}  (P99/P50 = {ratio:.1f}x)")
    print(f"  Turns P50:        {np.median(turns):.1f}")
    print(f"  Slowest 10% acct: {pct_time_from_slowest_10:.0f}% of total generation time")
    for t in range(1, max_turn + 1):
        pct = 100.0 * (turns == t).sum() / len(turns)
        print(f"    Turns={t}: {pct:.1f}%")


if __name__ == "__main__":
    main()
