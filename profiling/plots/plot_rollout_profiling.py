"""
Analyze rollout profiling JSONL to understand GPU vs CPU time split.

Produces:
  1. GPU generation time vs CPU sandbox time per trajectory (scatter)
  2. Distribution of sandbox time (histogram, log scale)
  3. Sandbox time vs output token count (scatter + regression)
  4. Sandbox time vs number of test cases (box plot)
  5. Per-step aggregate: GPU idle fraction during sandbox execution
  6. Pass/fail breakdown by time bucket

Usage:
    uv run --isolated profiling/plots/plot_rollout_profiling.py \\
        --input profiling_results/run_01/rollout_profiling.jsonl \\
        --output profiling_results/run_01/plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def plot_gpu_vs_cpu_scatter(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    passed = df["is_correct"] == True  # noqa: E712
    ax.scatter(
        df.loc[passed, "generation_time_s"],
        df.loc[passed, "sandbox_time_s"],
        alpha=0.4, s=15, label="Pass", color="green",
    )
    ax.scatter(
        df.loc[~passed, "generation_time_s"],
        df.loc[~passed, "sandbox_time_s"],
        alpha=0.4, s=15, label="Fail", color="red",
    )
    ax.set_xlabel("GPU Generation Time (s)")
    ax.set_ylabel("CPU Sandbox Time (s)")
    ax.set_title("Per-Trajectory: GPU Generation vs CPU Sandbox Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gpu_vs_cpu_scatter.png", dpi=150)
    plt.close(fig)


def plot_sandbox_distribution(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["sandbox_time_s"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Sandbox Time (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sandbox Time Distribution")

    nonzero = df["sandbox_time_s"][df["sandbox_time_s"] > 0]
    if len(nonzero) > 0:
        axes[1].hist(np.log10(nonzero), bins=50, edgecolor="black", alpha=0.7, color="orange")
        axes[1].set_xlabel("log10(Sandbox Time)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Sandbox Time Distribution (log scale)")

    fig.tight_layout()
    fig.savefig(out_dir / "sandbox_time_distribution.png", dpi=150)
    plt.close(fig)


def plot_sandbox_vs_tokens(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["output_token_count"], df["sandbox_time_s"], alpha=0.3, s=10)

    if len(df) > 10:
        z = np.polyfit(df["output_token_count"], df["sandbox_time_s"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["output_token_count"].min(), df["output_token_count"].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2,
                label=f"Linear fit (r={np.corrcoef(df['output_token_count'], df['sandbox_time_s'])[0,1]:.3f})")
        ax.legend()

    ax.set_xlabel("Output Token Count")
    ax.set_ylabel("Sandbox Time (s)")
    ax.set_title("Sandbox Execution Time vs Output Length")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sandbox_vs_tokens.png", dpi=150)
    plt.close(fig)


def plot_sandbox_vs_test_cases(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    test_counts = sorted(df["num_test_cases"].unique())

    data_by_ntests = [df[df["num_test_cases"] == n]["sandbox_time_s"].values for n in test_counts]
    bp = ax.boxplot(data_by_ntests, labels=[str(n) for n in test_counts], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    ax.set_xlabel("Number of Test Cases")
    ax.set_ylabel("Sandbox Time (s)")
    ax.set_title("Sandbox Time by Number of Test Cases")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "sandbox_vs_test_cases.png", dpi=150)
    plt.close(fig)


def plot_per_step_gpu_idle(df: pd.DataFrame, out_dir: Path):
    steps = sorted(df["global_step"].unique())
    gpu_fracs = []
    cpu_fracs = []
    step_labels = []

    for step in steps:
        step_df = df[df["global_step"] == step]
        gen_total = step_df["generation_time_s"].sum()
        sandbox_total = step_df["sandbox_time_s"].sum()
        total = gen_total + sandbox_total
        if total > 0:
            gpu_fracs.append(gen_total / total)
            cpu_fracs.append(sandbox_total / total)
            step_labels.append(step)

    if not step_labels:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(step_labels))
    ax.bar(x, gpu_fracs, label="GPU (generation)", color="steelblue")
    ax.bar(x, cpu_fracs, bottom=gpu_fracs, label="CPU (sandbox)", color="coral")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Time Fraction")
    ax.set_title("GPU vs CPU Time Fraction per Training Step")
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "per_step_gpu_idle.png", dpi=150)
    plt.close(fig)


def plot_pass_fail_by_time(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, label in [
        (axes[0], "sandbox_time_s", "Sandbox Time (s)"),
        (axes[1], "generation_time_s", "Generation Time (s)"),
    ]:
        passed = df[df["is_correct"] == True][col]  # noqa: E712
        failed = df[df["is_correct"] == False][col]  # noqa: E712
        bins = np.linspace(0, max(df[col].quantile(0.99), 0.1), 40)
        ax.hist(passed, bins=bins, alpha=0.6, label="Pass", color="green")
        ax.hist(failed, bins=bins, alpha=0.6, label="Fail", color="red")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Pass/Fail Distribution by {label}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "pass_fail_by_time.png", dpi=150)
    plt.close(fig)


def print_summary(df: pd.DataFrame):
    print(f"\n{'='*65}")
    print(f"{'Rollout Profiling Summary':^65}")
    print(f"{'='*65}")
    print(f"Total trajectories: {len(df)}")
    print(f"Training steps: {df['global_step'].nunique()}")
    if "is_correct" in df.columns:
        pass_rate = df["is_correct"].mean()
        print(f"Pass rate: {pass_rate:.1%}")

    print(f"\n{'Metric':<28} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8}")
    print("-" * 65)
    for col in ["generation_time_s", "sandbox_time_s", "sandbox_execution_s",
                "code_extraction_s", "output_token_count", "num_test_cases"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        fmt = ".3f" if "time" in col or "extraction" in col or "execution" in col else ".1f"
        unit = "s" if "time" in col or "extraction" in col or "execution" in col else ""
        print(f"{col:<28} {vals.mean():{fmt}}{unit:>1} {vals.median():{fmt}}{unit:>1} "
              f"{vals.quantile(0.95):{fmt}}{unit:>1} {vals.max():{fmt}}{unit:>1}")

    gen_total = df["generation_time_s"].sum()
    sandbox_total = df["sandbox_time_s"].sum()
    total = gen_total + sandbox_total
    if total > 0:
        print(f"\nAggregate time split:")
        print(f"  GPU (generation): {gen_total:.1f}s ({100*gen_total/total:.1f}%)")
        print(f"  CPU (sandbox):    {sandbox_total:.1f}s ({100*sandbox_total/total:.1f}%)")

    if len(df) > 10:
        r_tokens = df["output_token_count"].corr(df["sandbox_time_s"])
        r_tests = df["num_test_cases"].corr(df["sandbox_time_s"])
        print(f"\nCorrelations with sandbox_time:")
        print(f"  vs output_token_count: r={r_tokens:+.3f}")
        print(f"  vs num_test_cases:     r={r_tests:+.3f}")

    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout profiling data")
    parser.add_argument("--input", required=True, help="Path to rollout_profiling.jsonl")
    parser.add_argument("--output", default=None, help="Directory for plots (default: same as input)")
    args = parser.parse_args()

    df = load_data(args.input)
    print(f"Loaded {len(df)} trajectory records from {args.input}")

    print_summary(df)

    out_dir = Path(args.output) if args.output else Path(args.input).parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_gpu_vs_cpu_scatter(df, out_dir)
    plot_sandbox_distribution(df, out_dir)
    plot_sandbox_vs_tokens(df, out_dir)
    if df["num_test_cases"].nunique() > 1:
        plot_sandbox_vs_test_cases(df, out_dir)
    plot_per_step_gpu_idle(df, out_dir)
    if "is_correct" in df.columns:
        plot_pass_fail_by_time(df, out_dir)

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
