"""
Task 4 — Long-tail distribution in multi-turn rollouts.

Subclasses SkyRLGymGenerator to time each trajectory's agent_loop() independently
(all run concurrently via asyncio.gather, so per-prompt wall-clock is meaningful).

Turn structure is inferred from the token-level reward list:
  - Non-zero positions = turn boundaries
  - Number of turns = count of non-zero reward entries
  - Solve turn = 1-based index of first reward >= 1.0, or -1 if unsolved

Also captures fine-grained GPU vs CPU breakdown per trajectory via the
``rollout_profiling`` and ``sandbox_profiling`` dicts that agent_loop() and
LCBEnv inject into env_metrics.

Output:
  profiling_results/<run>/rollout_stats.jsonl      (per-trajectory wall-clock + turns)
  profiling_results/<run>/rollout_profiling.jsonl   (GPU/CPU breakdown per trajectory)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.generators.base import TrajectoryID
from profiling.rollout_profiler import RolloutProfiler


class ProfiledSkyRLGymGenerator(SkyRLGymGenerator):
    """
    Minimal subclass that wraps agent_loop() with a wall-clock timer and records
    per-trajectory stats to JSONL.

    Multi-turn behavior is completely preserved — only timing is added.

    Also collects fine-grained GPU generation vs. CPU sandbox timing from
    env_metrics (populated by the base agent_loop and LCBEnv instrumentation)
    and writes it via RolloutProfiler.
    """

    def __init__(self, *args, profiling_out_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        profiling_out_dir.mkdir(parents=True, exist_ok=True)
        self._stats_path = profiling_out_dir / "rollout_stats.jsonl"
        self._rollout_profiler = RolloutProfiler(profiling_out_dir)
        self._global_step = 0
        print(f"[rollout_stats] profiler active → {self._stats_path}", flush=True)

    def set_global_step(self, step: int):
        self._global_step = step

    async def agent_loop(
        self,
        prompt,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params=None,
        trajectory_id: Optional[TrajectoryID] = None,
    ):
        t0 = time.perf_counter()
        result = await super().agent_loop(
            prompt=prompt,
            env_class=env_class,
            env_extras=env_extras,
            max_tokens=max_tokens,
            max_input_length=max_input_length,
            sampling_params=sampling_params,
            trajectory_id=trajectory_id,
        )
        wall_s = time.perf_counter() - t0

        # result is TrajectoryOutput (non-stepwise path)
        # Infer turn stats from per-token reward list
        response_ids = result.response_ids if hasattr(result, "response_ids") else []
        reward_list = result.reward if hasattr(result, "reward") else []
        stop_reason = result.stop_reason if hasattr(result, "stop_reason") else ""

        num_turns, solve_turn, final_reward = _infer_turn_stats(reward_list)
        tokens_per_turn = len(response_ids) / max(num_turns, 1)

        record = {
            "global_step": self._global_step,
            "trajectory_id": trajectory_id.to_string() if trajectory_id else "",
            "wall_s": round(wall_s, 4),
            "num_turns": num_turns,
            "total_response_tokens": len(response_ids),
            "tokens_per_turn_avg": round(tokens_per_turn, 1),
            "solve_turn": solve_turn,
            "final_reward": final_reward,
            "stop_reason": stop_reason,
        }
        with open(self._stats_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return result

    async def generate(self, input_batch, disable_tqdm: bool = False):
        batch_meta = input_batch.get("batch_metadata") if isinstance(input_batch, dict) else None
        if batch_meta is not None and hasattr(batch_meta, "global_step"):
            self._global_step = batch_meta.global_step
        result = await super().generate(input_batch, disable_tqdm=disable_tqdm)

        env_metrics = result.get("env_metrics", [])
        if env_metrics:
            self._rollout_profiler.record_batch(
                global_step=self._global_step,
                env_metrics=env_metrics,
                trajectory_ids=input_batch.get("trajectory_ids"),
                rewards=result.get("rewards"),
                response_ids=result.get("response_ids"),
                env_extras=input_batch.get("env_extras"),
            )

        return result

    def get_profiling_summary(self) -> str:
        return self._rollout_profiler.summary()


def _infer_turn_stats(reward_list: Union[List[float], float]):
    """
    Infer number of turns and solve turn from per-token reward list.

    In non-stepwise GRPO with multi-turn, rewards are placed at the last
    token of each assistant turn (i.e. one non-zero value per turn).
    """
    if isinstance(reward_list, (int, float)):
        # scalar reward — single turn
        return 1, (1 if float(reward_list) >= 1.0 else -1), float(reward_list)

    nonzero = [(i, v) for i, v in enumerate(reward_list) if abs(v) > 1e-9]
    num_turns = len(nonzero)
    if num_turns == 0:
        return 1, -1, 0.0

    final_reward = nonzero[-1][1]
    solve_turn = -1
    for rank, (_, v) in enumerate(nonzero, start=1):
        if v >= 1.0:
            solve_turn = rank
            break

    return num_turns, solve_turn, float(final_reward)
