"""
Task 1 — Per-stage wall-clock timer.

Wraps RayPPOTrainer with explicit timing for all 6 buckets:
  1. generate          — vLLM rollout
  2. env_step          — postprocess_generator_output (reward reshape)
  3. advantage         — compute_advantages_and_returns
  4. fwd_logprobs      — ref + policy forward passes
  5. policy_train      — FSDP backward + optimizer step
  6. weight_sync       — FSDP → vLLM NCCL broadcast
  + stage_switch       — vLLM sleep between generation and training

All timings written to profiling_results/<run>/stage_times.jsonl (one JSON line
per step) and printed as a summary table at the end.
"""

import json
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from skyrl.train.trainer import RayPPOTrainer


# ── Simple synchronized timer ─────────────────────────────────────────────────

@contextmanager
def wall_timer(name: str, store: dict):
    t0 = time.perf_counter()
    yield
    store[name] = time.perf_counter() - t0


@asynccontextmanager
async def async_wall_timer(name: str, store: dict):
    t0 = time.perf_counter()
    yield
    store[name] = time.perf_counter() - t0


# ── Stage recorder (writes JSONL) ─────────────────────────────────────────────

class StageRecorder:
    STAGE_KEYS = [
        "generate",
        "stage_switch",
        "env_step",
        "convert_to_training_input",
        "fwd_logprobs",
        "advantage",
        "policy_train",
        "weight_sync",
        "step_total",
    ]

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / "stage_times.jsonl"
        self._records: List[dict] = []

    def record(self, global_step: int, epoch: int, timings: dict):
        row = {"global_step": global_step, "epoch": epoch, **timings}
        self._records.append(row)
        with open(self._path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def summary(self) -> str:
        if not self._records:
            return "No records yet."
        import numpy as np

        lines = [
            f"\n{'Stage':<28} {'Mean':>8} {'Median':>8} {'P95':>8} {'% Total':>9}",
            "-" * 65,
        ]
        totals = [r.get("step_total", 0) for r in self._records]
        grand_mean = float(np.mean(totals)) if totals else 1.0

        for key in self.STAGE_KEYS:
            vals = [r[key] for r in self._records if key in r]
            if not vals:
                continue
            mean = float(np.mean(vals))
            median = float(np.median(vals))
            p95 = float(np.percentile(vals, 95))
            pct = 100.0 * mean / grand_mean if grand_mean > 0 else 0.0
            flag = "  ← high" if key not in ("generate", "policy_train", "weight_sync", "step_total") and pct > 10 else ""
            lines.append(f"{key:<28} {mean:>7.2f}s {median:>7.2f}s {p95:>7.2f}s {pct:>8.1f}%{flag}")
        lines.append("-" * 65)
        return "\n".join(lines)


# ── Profiled trainer ──────────────────────────────────────────────────────────

class ProfiledRayPPOTrainer(RayPPOTrainer):
    """
    Subclass of RayPPOTrainer that:
    - Adds explicit timing for stage_switch (vLLM sleep) which the base class omits.
    - Dumps all_timings to JSONL after every step.
    - Publishes current stage name to a shared dict for GPU-util tagging (Task 5).
    """

    def __init__(self, *args, profiling_out_dir: Path, stage_state: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._recorder = StageRecorder(profiling_out_dir)
        # Shared dict updated in-place so other monitors can read current stage
        self._stage_state: dict = stage_state if stage_state is not None else {}
        self._stage_state["current"] = "init"

    def _set_stage(self, name: str):
        self._stage_state["current"] = name

    async def train(self):
        """
        Override train() to splice in stage_switch timing and JSONL dumps.
        The full logic mirrors the parent; only differences are marked with ##.
        """
        import math

        from loguru import logger
        from tqdm import tqdm

        from skyrl.backends.skyrl_train.inference_engines.utils import (
            get_sampling_params_for_backend,
        )
        from skyrl.train.utils import Timer
        from skyrl.train.generators.utils import prepare_generator_input
        from skyrl.train.utils.logging_utils import log_example
        from skyrl.backends.skyrl_train.utils.ppo_utils import get_kl_controller

        # -- init (mirrors parent) --
        with Timer("init_weight_sync_state"):
            self.init_weight_sync_state()

        if self.resume_mode.value != "none":
            with Timer("load_checkpoints"):
                self.global_step, _ = self.load_checkpoints()

        with Timer("sync_weights"):
            await self.dispatch.save_weights_for_sampler()

        if self.cfg.trainer.eval_interval > 0 and self.cfg.trainer.eval_before_train:
            with Timer("eval", self.all_timings):
                eval_metrics = await self.eval()
                self.tracker.log(eval_metrics, step=self.global_step, commit=True)

        if self.cfg.trainer.algorithm.use_kl_in_reward:
            self.reward_kl_controller = get_kl_controller(self.cfg.trainer.algorithm)

        pbar = tqdm(total=self.total_training_steps, initial=self.global_step, desc="Training Batches Processed")
        start_epoch = self.global_step // len(self.train_dataloader)
        self.global_step += 1

        for epoch in range(start_epoch, self.cfg.trainer.epochs):
            for _, rand_prompts in enumerate(self.train_dataloader):
                step_t0 = time.perf_counter()  ## wall-clock for full step
                profiled_timings: dict = {}

                with Timer("step", self.all_timings):
                    rand_prompts = self._remove_tail_data(rand_prompts)
                    generator_input, uids = prepare_generator_input(
                        rand_prompts,
                        self.cfg.generator.n_samples_per_prompt,
                        get_sampling_params_for_backend(
                            self.cfg.generator.inference_engine.backend,
                            self.cfg.generator.sampling_params,
                        ),
                        self.cfg.environment.env_class,
                        "train",
                        self.global_step,
                    )

                    ## 1. Generate
                    self._set_stage("generate")
                    async with async_wall_timer("generate", profiled_timings):
                        with Timer("generate", self.all_timings):
                            from skyrl.train.generators.base import GeneratorOutput
                            generator_output: GeneratorOutput = await self.generate(generator_input)

                    if self.cfg.generator.step_wise_trajectories:
                        uids = [t.instance_id for t in generator_output["trajectory_ids"]]

                    if self.cfg.trainer.algorithm.dynamic_sampling.type is not None:
                        generator_output, uids, keep_sampling = self.handle_dynamic_sampling(generator_output, uids)
                        if keep_sampling:
                            pbar.update(1)
                            continue

                    ## 2. Stage-switch (vLLM sleep) — the overhead the literature ignores
                    self._set_stage("stage_switch")
                    if self.colocate_all:
                        async with async_wall_timer("stage_switch", profiled_timings):
                            await self.inference_engine_client.sleep()
                    else:
                        profiled_timings["stage_switch"] = 0.0

                    ## 3. Env step (reward reshape)
                    self._set_stage("env_step")
                    async with async_wall_timer("env_step", profiled_timings):
                        with Timer("postprocess_generator_output", self.all_timings):
                            generator_output, uids = self.postprocess_generator_output(generator_output, uids)

                    log_interval = self.cfg.trainer.log_example_interval
                    if log_interval > 0 and self.global_step % log_interval == 0:
                        vis = self.tokenizer.decode(generator_output["response_ids"][0])
                        log_example(logger, prompt=generator_input["prompts"][0], response=vis,
                                    reward=generator_output["rewards"][0])

                    ## 4. Convert to tensors
                    self._set_stage("convert")
                    with Timer("convert_to_training_input", self.all_timings):
                        training_input = self.convert_to_training_input(generator_output, uids)
                    profiled_timings["convert_to_training_input"] = self.all_timings.get("convert_to_training_input", 0)

                    ## 5. Forward logprobs + values (ref + policy)
                    self._set_stage("fwd_logprobs")
                    async with async_wall_timer("fwd_logprobs", profiled_timings):
                        with Timer("fwd_logprobs_values_reward", self.all_timings):
                            training_input = self.fwd_logprobs_values_reward(training_input)

                    if self.cfg.trainer.algorithm.use_kl_in_reward:
                        with Timer("apply_reward_kl_penalty", self.all_timings):
                            training_input = self.apply_reward_kl_penalty(training_input)

                    ## 6. Advantage estimation
                    self._set_stage("advantage")
                    async with async_wall_timer("advantage", profiled_timings):
                        with Timer("compute_advantages_and_returns", self.all_timings):
                            training_input = self.compute_advantages_and_returns(training_input)
                            for key in ["rewards"]:
                                training_input.pop(key)
                            training_input.metadata.pop("uids")
                            training_input.metadata.pop("is_last_step", None)

                    ## 7. Policy update (FSDP fwd/bwd)
                    self._set_stage("policy_train")
                    async with async_wall_timer("policy_train", profiled_timings):
                        with Timer("train_critic_and_policy", self.all_timings):
                            status = self.train_critic_and_policy(training_input)

                    # checkpoints
                    is_epoch_end = self.global_step % len(self.train_dataloader) == 0
                    if self.cfg.trainer.ckpt_interval > 0:
                        if is_epoch_end or self.global_step % self.cfg.trainer.ckpt_interval == 0:
                            with Timer("save_checkpoints", self.all_timings):
                                self.save_checkpoints()

                    if self.cfg.trainer.update_ref_every_epoch and self.ref_model is not None and is_epoch_end and epoch != self.cfg.trainer.epochs - 1:
                        with Timer("update_ref_with_policy", self.all_timings):
                            self.update_ref_with_policy()

                    ## 8. Weight sync
                    self._set_stage("weight_sync")
                    async with async_wall_timer("weight_sync", profiled_timings):
                        with Timer("sync_weights", self.all_timings):
                            await self.dispatch.save_weights_for_sampler()

                profiled_timings["step_total"] = time.perf_counter() - step_t0  ##

                # eval
                logger.info(status)
                self.all_metrics.update({"trainer/epoch": epoch, "trainer/global_step": self.global_step})
                if self.cfg.trainer.eval_interval > 0 and (
                    self.global_step % self.cfg.trainer.eval_interval == 0
                    or self.global_step == self.total_training_steps
                ):
                    self._set_stage("eval")
                    with Timer("eval", self.all_timings):
                        eval_metrics = await self.eval()
                        self.all_metrics.update(eval_metrics)

                log_payload = {
                    **self.all_metrics,
                    **{f"timing/{k}": v for k, v in self.all_timings.items()},
                }
                if self._vllm_metrics_scraper is not None:
                    log_payload.update(await self._vllm_metrics_scraper.sample())
                self.tracker.log(log_payload, step=self.global_step, commit=True)
                self.all_metrics = {}
                self.all_timings = {}

                ## write to JSONL
                self._recorder.record(self.global_step, epoch, profiled_timings)

                pbar.update(1)
                self.global_step += 1
                del training_input, generator_output

        pbar.close()
        self._set_stage("done")

        if self.colocate_all:
            await self.inference_engine_client.sleep()
        if self.cfg.trainer.ckpt_interval > 0:
            with Timer("save_checkpoints", self.all_timings):
                self.save_checkpoints()
        if self._vllm_metrics_scraper is not None:
            await self._vllm_metrics_scraper.aclose()
        self.tracker.finish()
        logger.info("Training done!")
        logger.info(self._recorder.summary())
