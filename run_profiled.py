"""
Profiled training entrypoint.

Wraps main_multiply.py without modifying any SkyRL source files:
  - Subclasses BasePPOExp to inject ProfiledRayPPOTrainer + ProfiledSkyRLGymGenerator
  - Launches background monitors (memory waterfall, GPU util)
  - Patches dispatch.save_weights_for_sampler for weight sync timing
  - Saves all results to profiling_results/<timestamp>/

Usage (inside Singularity, from /skyrl):
    python run_profiled.py \\
        "data.train_data=[...]" \\
        "data.val_data=[...]" \\
        trainer.epochs=5 \\
        ... (same args as main_multiply)

Flags:
    --profiling-off     Run without any profiling (baseline for overhead check)
"""

import sys
import os
import asyncio
from datetime import datetime
from pathlib import Path

import ray

# ── Resolve output directory ──────────────────────────────────────────────────
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
PROFILING_OUT = Path(__file__).parent / "profiling_results" / f"{_TS}_job{_JOB_ID}"

# ── Check for --profiling-off flag ────────────────────────────────────────────
PROFILING_ENABLED = "--profiling-off" not in sys.argv
if not PROFILING_ENABLED:
    sys.argv.remove("--profiling-off")


from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray, validate_cfg
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name


# ── Profiled experiment class ─────────────────────────────────────────────────

class ProfiledPPOExp(BasePPOExp):
    """
    Subclass of BasePPOExp that injects profiled trainer and generator.
    All logic runs inside a Ray remote task (same as skyrl_entrypoint in main_multiply).
    """

    def __init__(self, cfg, profiling_out: Path, profiling_enabled: bool):
        super().__init__(cfg)
        self._profiling_out = profiling_out
        self._profiling_enabled = profiling_enabled
        self._stage_state = {"current": "init"}

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        if not self._profiling_enabled:
            return super().get_generator(cfg, tokenizer, inference_engine_client)

        from profiling.rollout_stats import ProfiledSkyRLGymGenerator
        print(f"[profiling] Using ProfiledSkyRLGymGenerator → {self._profiling_out}", flush=True)
        return ProfiledSkyRLGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
            profiling_out_dir=self._profiling_out,
        )

    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset,
                    inference_engine_client, generator, colocate_pg):
        if not self._profiling_enabled:
            return super().get_trainer(cfg, tracker, tokenizer, train_dataset, eval_dataset,
                                       inference_engine_client, generator, colocate_pg)

        from profiling.stage_timer import ProfiledRayPPOTrainer
        print(f"[profiling] Using ProfiledRayPPOTrainer → {self._profiling_out}", flush=True)
        return ProfiledRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
            profiling_out_dir=self._profiling_out,
            stage_state=self._stage_state,
        )

    def run(self):
        if not self._profiling_enabled:
            super().run()
            return

        from profiling.memory_waterfall import MemoryWaterfallMonitor
        from profiling.gpu_util import GPUUtilMonitor
        from profiling.weight_sync import WeightSyncProfiler

        # Start background monitors
        mem_monitor = MemoryWaterfallMonitor(self._profiling_out, self._stage_state, interval_ms=200)
        gpu_monitor = GPUUtilMonitor(self._profiling_out, self._stage_state, interval_s=1)
        mem_monitor.start()
        gpu_monitor.start()

        try:
            trainer = self._setup_trainer()

            # Patch weight sync profiler onto dispatch
            ws_profiler = WeightSyncProfiler(self._profiling_out)
            ws_profiler.patch(trainer.dispatch)

            # Run training
            asyncio.run(trainer.train())

            # Print weight sync summary
            print(ws_profiler.summary())

        finally:
            mem_monitor.stop()
            gpu_summary = gpu_monitor.stop()
            print(f"\n[gpu_util] Generation stats: {gpu_summary}")

        print(f"\n[profiling] All results saved to: {self._profiling_out}")
        _print_plot_instructions(self._profiling_out)


def _print_plot_instructions(out_dir: Path):
    print("\n── To generate plots, run: ─────────────────────────────────────────────")
    print(f"  python profiling/plots/plot_stage_times.py {out_dir}/stage_times.jsonl")
    print(f"  python profiling/plots/plot_memory.py      {out_dir}/memory.csv")
    print(f"  python profiling/plots/plot_weight_sync.py {out_dir}/weight_sync.jsonl")
    print(f"  python profiling/plots/plot_rollout.py     {out_dir}/rollout_stats.jsonl")
    print(f"  python profiling/plots/plot_gpu_util.py    {out_dir}/gpu_util.csv")
    print("────────────────────────────────────────────────────────────────────────")


# ── Ray remote entrypoint ─────────────────────────────────────────────────────

@ray.remote(num_cpus=1)
def profiled_entrypoint(cfg: SkyRLTrainConfig, profiling_out_str: str, profiling_enabled: bool):
    profiling_out = Path(profiling_out_str)
    exp = ProfiledPPOExp(cfg, profiling_out=profiling_out, profiling_enabled=profiling_enabled)
    exp.run()


def main():
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)

    PROFILING_OUT.mkdir(parents=True, exist_ok=True)
    print(f"[profiling] Output directory: {PROFILING_OUT}", flush=True)
    print(f"[profiling] Profiling enabled: {PROFILING_ENABLED}", flush=True)

    ray.get(profiled_entrypoint.remote(cfg, str(PROFILING_OUT), PROFILING_ENABLED))


if __name__ == "__main__":
    main()
