"""
Per-trajectory rollout profiler — GPU generation vs. CPU sandbox breakdown.

Collects timing from two sources wired into the rollout loop:
  1. ``rollout_profiling`` in env_metrics — generation_time_s, sandbox_time_s
     (added in skyrl_gym_generator.agent_loop)
  2. ``sandbox_profiling`` in env_metrics — code_extraction_s, sandbox_execution_s,
     num_test_cases, is_correct  (added in LCBEnv.get_metrics)

Output: one JSONL line per trajectory, written to ``<out_dir>/rollout_profiling.jsonl``.

Usage:
    profiler = RolloutProfiler(Path("profiling_results/run_01"))
    # after each generate() call:
    profiler.record_batch(
        global_step=step,
        env_metrics=generator_output["env_metrics"],
        trajectory_ids=input_batch.get("trajectory_ids"),
        rewards=generator_output["rewards"],
        response_ids=generator_output["response_ids"],
        env_extras=input_batch["env_extras"],
    )
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class RolloutProfiler:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / "rollout_profiling.jsonl"
        self._record_count = 0
        print(f"[rollout_profiler] active → {self._path}", flush=True)

    def record_batch(
        self,
        global_step: int,
        env_metrics: List[Dict[str, Any]],
        trajectory_ids: Optional[List] = None,
        rewards: Optional[List] = None,
        response_ids: Optional[List[List[int]]] = None,
        env_extras: Optional[List[Dict[str, Any]]] = None,
    ):
        lines = []
        for i, metrics in enumerate(env_metrics):
            rollout_prof = metrics.get("rollout_profiling", {})
            sandbox_prof = metrics.get("sandbox_profiling", {})

            extra_info = {}
            if env_extras and i < len(env_extras):
                extra_info = env_extras[i].get("extra_info", {})
                if isinstance(extra_info, str):
                    try:
                        extra_info = json.loads(extra_info)
                    except (json.JSONDecodeError, TypeError):
                        extra_info = {}

            reward_val = 0.0
            if rewards and i < len(rewards):
                r = rewards[i]
                reward_val = float(r) if isinstance(r, (int, float)) else float(sum(r))

            output_tokens = 0
            if response_ids and i < len(response_ids):
                output_tokens = len(response_ids[i])

            traj_id = ""
            if trajectory_ids and i < len(trajectory_ids):
                tid = trajectory_ids[i]
                traj_id = tid.to_string() if hasattr(tid, "to_string") else str(tid)

            record = {
                "global_step": global_step,
                "trajectory_idx": i,
                "trajectory_id": traj_id,
                "generation_time_s": rollout_prof.get("generation_time_s", 0.0),
                "sandbox_time_s": rollout_prof.get("sandbox_time_s", 0.0),
                "num_turns": rollout_prof.get("num_turns", 0),
                "output_token_count": output_tokens,
                "reward": reward_val,
                "num_test_cases": sandbox_prof.get("num_test_cases", 0),
                "code_valid": sandbox_prof.get("code_valid", None),
                "is_correct": sandbox_prof.get("is_correct", None),
                "code_extraction_s": sandbox_prof.get("code_extraction_s", 0.0),
                "sandbox_execution_s": sandbox_prof.get("sandbox_execution_s", 0.0),
                "difficulty": extra_info.get("difficulty", ""),
                "problem_index": extra_info.get("index", -1),
                "problem_url": extra_info.get("url", ""),
            }
            lines.append(json.dumps(record))

        with open(self._path, "a") as f:
            f.write("\n".join(lines) + "\n")
        self._record_count += len(lines)

    def summary(self) -> str:
        if not self._path.exists():
            return "[rollout_profiler] no data recorded."
        try:
            import numpy as np
        except ImportError:
            return f"[rollout_profiler] {self._record_count} records (numpy unavailable for summary)."

        records = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return "[rollout_profiler] no records."

        gen_times = [r["generation_time_s"] for r in records]
        sandbox_times = [r["sandbox_time_s"] for r in records]
        token_counts = [r["output_token_count"] for r in records]
        num_tests = [r["num_test_cases"] for r in records]
        rewards = [r["reward"] for r in records]
        sandbox_exec = [r["sandbox_execution_s"] for r in records]

        pass_rate = np.mean([1.0 if r > 0.5 else 0.0 for r in rewards])

        lines = [
            f"\n{'Rollout Profiling Summary':^65}",
            f"{'='*65}",
            f"Total trajectories: {len(records)}",
            f"Pass rate: {pass_rate:.1%}",
            "",
            f"{'Metric':<28} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8}",
            f"{'-'*65}",
        ]
        for name, vals in [
            ("generation_time_s", gen_times),
            ("sandbox_time_s", sandbox_times),
            ("sandbox_execution_s", sandbox_exec),
            ("output_tokens", token_counts),
            ("num_test_cases", num_tests),
        ]:
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            lines.append(
                f"{name:<28} {np.mean(arr):>7.3f}s {np.median(arr):>7.3f}s "
                f"{np.percentile(arr, 95):>7.3f}s {np.max(arr):>7.3f}s"
                if "time" in name or "execution" in name
                else f"{name:<28} {np.mean(arr):>8.1f} {np.median(arr):>8.1f} "
                f"{np.percentile(arr, 95):>8.1f} {np.max(arr):>8.1f}"
            )

        gpu_total = np.sum(gen_times)
        cpu_total = np.sum(sandbox_times)
        total = gpu_total + cpu_total
        if total > 0:
            lines.append(f"\n{'Time split':^65}")
            lines.append(f"  GPU (generation): {gpu_total:.1f}s ({100*gpu_total/total:.1f}%)")
            lines.append(f"  CPU (sandbox):    {cpu_total:.1f}s ({100*cpu_total/total:.1f}%)")

        # Correlation: sandbox time vs output tokens
        if len(sandbox_times) > 10:
            corr_tokens = np.corrcoef(token_counts, sandbox_times)[0, 1]
            corr_ntests = np.corrcoef(num_tests, sandbox_times)[0, 1]
            lines.append(f"\n{'Correlations with sandbox_time':^65}")
            lines.append(f"  vs output_tokens:  r={corr_tokens:+.3f}")
            lines.append(f"  vs num_test_cases: r={corr_ntests:+.3f}")

        lines.append(f"{'='*65}")
        return "\n".join(lines)
