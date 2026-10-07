# Generation Analysis: APPS GRPO — Qwen3-4B — 5 Epochs

**Date**: 2026-06-15
**Jobs**: 1564511 (full 5-epoch run), 1651919 (profiled single-step)
**Model**: Qwen/Qwen3-4B
**Hardware**: 4x A100-40GB, 32 CPUs, 256 GB RAM (single node)
**Status**: Completed successfully — 15 training steps across 5 epochs

---

## Executive Summary

Generation dominates the training loop at **63% of wall time** (mean 18.4 min/step),
making it the primary optimization target. The model hits the 2048-token cap on
**73.5% of trajectories**, indicating extensive "thinking" output that doesn't
translate to correctness — only **24.5% pass rate** across all difficulty levels.
GPU straggler effects are minimal (mean gap <1s), so the bottleneck is raw
token generation volume, not load imbalance.

---

## Training Time Breakdown

| Phase | Mean Time/Step | % of Step |
|-------|---------------|-----------|
| **Generate** | **1101.5s (18.4 min)** | **62.9%** |
| Policy train | 436.4s (7.3 min) | 24.9% |
| Fwd logprobs | 188.8s (3.1 min) | 10.8% |
| Weight sync | 7.3s | 0.4% |
| Other (convert, postprocess) | ~4s | 0.2% |
| **Total step** | **1750.6s (29.2 min)** | **100%** |

**Total training time**: 7.3 hours across 15 steps (3 steps/epoch x 5 epochs)

### Per-Step Detail

| Step | Total (s) | Generate (s) | Fwd Logprobs (s) | Policy Train (s) |
|------|-----------|-------------|-------------------|-------------------|
| 1 | 1658 | 1014 | 191 | 441 |
| 2 | 1698 | 1066 | 186 | 434 |
| 3 | 1780 | 1116 | 189 | 440 |
| 4 | 1713 | 1072 | 192 | 438 |
| 5 | 1735 | 1098 | 188 | 437 |
| 6 | 1802 | 1146 | 187 | 440 |
| 7 | 1668 | 1027 | 191 | 439 |
| 8 | 1818 | 1176 | 190 | 441 |
| 9 | 1896 | 1162 | 187 | 428 |
| 10 | 1822 | 1186 | 191 | 433 |
| 11 | 1718 | 1084 | 187 | 435 |
| 12 | 1675 | 1024 | 189 | 435 |
| 13 | 1741 | 1102 | 190 | 436 |
| 14 | 1813 | 1179 | 187 | 436 |
| 15 | 1721 | 1071 | 187 | 434 |

Generate time varies from 1014s to 1186s (16.9–19.8 min), with no clear upward
trend over training — the variance comes from the stochastic nature of generation
length per batch.

---

## Per-Trajectory Generation Profiling (Step 16)

Profiled from job 1651919: 98 trajectories from the validation set.

### Generation Time Distribution

| Metric | Value |
|--------|-------|
| Mean | 51.4s |
| Median | 55.4s |
| P95 | 57.5s |
| Min | 23.1s |
| Max | 67.4s |
| Std dev | ~10.5s |

The distribution is **bimodal**: a large peak at 54–57s (trajectories that hit the
token cap) and a long left tail (trajectories that finish early with correct solutions).

### Token Output

| Metric | Value |
|--------|-------|
| Mean tokens | 1869 |
| Median tokens | 2048 |
| Max tokens (cap) | 2048 |
| At token cap | 72 / 98 (73.5%) |
| Below cap | 26 / 98 (26.5%) |

**73.5% of trajectories exhaust the full 2048-token budget.** The correlation
between output tokens and generation time is very high (**r = 0.980**), confirming
that generation time is almost entirely determined by how many tokens the model
produces — not by prompt complexity or scheduling overhead.

### Correctness Analysis

| Metric | Incorrect | Correct |
|--------|-----------|---------|
| Count | 74 (75.5%) | 24 (24.5%) |
| Median gen time | 55.5s | 37.8s |
| Median tokens | 2048 | ~1350 |

**Incorrect solutions are overwhelmingly token-capped**: they produce the maximum
2048 tokens without arriving at a working answer. Correct solutions terminate
earlier (median ~1350 tokens), spending less time generating.

This means the majority of the generation budget is spent on failed trajectories
that exhaust the token limit — a key inefficiency in the generate phase.

### Difficulty Breakdown

All 98 profiled trajectories are labeled "introductory" difficulty:

| Difficulty | Count | Pass Rate | Mean Gen Time | Mean Tokens |
|------------|-------|-----------|---------------|-------------|
| Introductory | 98 | 24.5% | 51.4s | 1869 |

---

## GPU Utilization & Straggler Analysis

GPU straggler effects during the generate phase are **negligible**:

| Metric | Value |
|--------|-------|
| Mean straggler gap | 0.7s |
| Max straggler gap | 10s (step 1 only, GPU 3) |
| Typical gap (steps 2–15) | 0s |

The 10s outlier on step 1 for GPU 3 is a warmup artifact. After step 1, all
4 GPUs finish generation within the same monitoring interval. This rules out
load imbalance as a contributor to generation time — the bottleneck is purely
the volume of tokens generated per batch.

---

## Key Findings

1. **Generation is the bottleneck**: 63% of each training step is spent generating
   rollouts. At ~18 min/step, the generate phase alone accounts for 4.6 hours of
   the 7.3-hour total training time.

2. **Token cap saturation**: 73.5% of trajectories hit the 2048-token cap. Most
   of these are incorrect — the model "thinks" at length without producing valid
   code. This wastes generation budget on hopeless trajectories.

3. **Strong token-time correlation (r=0.98)**: Generation time is almost entirely
   driven by output length. There is negligible overhead from scheduling, prefill,
   or sandbox execution.

4. **No GPU straggler problem**: All 4 GPUs finish within ~0s of each other
   (after warmup), so the batched generation is well-balanced.

5. **Low pass rate (24.5%)**: Only 1 in 4 trajectories produces a correct solution
   on introductory-level problems, suggesting the 4B model struggles with the APPS
   task at this training stage.

---

## Optimization Opportunities

| Opportunity | Potential Savings | Complexity |
|-------------|-------------------|------------|
| Early stopping on low-quality generations | High — cut ~50% of token-capped incorrect trajectories | Medium |
| Reduce `max_generate_length` to 1536 | Moderate — saves ~25% on capped trajectories | Low |
| Increase `n_samples_per_prompt` with shorter cap | Trade breadth for depth per trajectory | Low |
| Switch to a larger model (7B) | May reduce token waste via better code quality | Already planned (job 1657552) |

---

## Run Configuration

### Infrastructure

| Parameter | Value |
|-----------|-------|
| Partition | normal-a100-40 |
| GPUs | 4x A100-40GB |
| CPUs | 32 |
| Memory | 256 GB |
| Container | `singularity/skyrl_fsdp.sif` |

### Model

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-4B |
| Dtype | bfloat16 |
| Strategy | FSDP |

### Data

| Parameter | Value |
|-----------|-------|
| Train data | `/data/apps/train.parquet` (891 total, 875 after filtering) |
| Val data | `/data/apps/validation.parquet` (99 total, 98 after filtering) |
| Environment | `lcb` (LiveCodeBench) |
| Max prompt length | 1024 tokens |

### Training (GRPO)

| Parameter | Value |
|-----------|-------|
| Algorithm | GRPO |
| Epochs | 5 |
| Steps per epoch | 3 (ceil(875 / 256) = ~3.4) |
| Update epochs per batch | 1 |
| Train batch size | 256 |
| Policy mini-batch size | 64 |
| Micro forward batch/GPU | 8 |
| Micro train batch/GPU | 2 |
| Eval batch size | 256 |
| Eval before train | true |
| Eval interval | 5 epochs |
| Checkpoint interval | 3 epochs |
| Learning rate (policy) | 1.0e-6 |
| Learning rate (critic) | 5.0e-6 |
| Scheduler | constant_with_warmup |
| KL loss | enabled (coef=0.001) |
| KL estimator | k3 |
| Loss reduction | token_mean |
| Gradient checkpointing | true |
| Flash attention | true |

### Generation / Inference

| Parameter | Value |
|-----------|-------|
| Backend | vLLM (v1) |
| Num engines | 4 |
| Tensor parallel size | 1 |
| GPU memory utilization | 0.4 |
| Async engine | true |
| Run engines locally | true |
| Batched generation | true |
| Weight sync backend | NCCL |
| Enforce eager | true |
| Chunked prefill | true |
| Prefix caching | true |
| Max num seqs | 1024 |
| Max batched tokens | 8192 |
| N samples per prompt | 8 |
| Max generate length | 2048 |
| Temperature (train) | 1.0 |
| Temperature (eval) | 0.0 |
| Top-p | 1.0 |

### Placement

| Parameter | Value |
|-----------|-------|
| Colocate all | true |
| Policy GPUs/node | 4 |
| Ref GPUs/node | 4 |
| Critic GPUs/node | 1 |
| Nodes | 1 |

### FSDP Config

| Parameter | Value |
|-----------|-------|
| CPU offload | false |
| FSDP size | -1 (full shard) |
| Reshard after forward | true |

---

## Plots

All plots are in the same directory as this document:

| File | Description |
|------|-------------|
| `01_gen_time_histogram.png` | Per-trajectory generation time distribution |
| `02_gen_time_vs_tokens.png` | Generation time vs output tokens (r=0.98) |
| `03_gen_time_by_correctness.png` | Generation time & tokens by correct/incorrect |
| `04_gen_time_by_difficulty.png` | Generation time & pass rate by difficulty |
| `05_token_distribution.png` | Output token count distribution (73.5% at cap) |
| `06_step_gen_time_trend.png` | Per-step generation time across all 15 steps |
| `07_gpu_straggler_analysis.png` | Per-GPU straggler lag analysis |
| `08_gpu_detail_generate_tail.png` | Per-GPU utilization during generate tail |

## Data Sources

| File | Description |
|------|-------------|
| `profiling_results/20260618_090402_job1651919/rollout_profiling.jsonl` | Per-trajectory profiling (98 records) |
| `profiling_results/20260618_090402_job1651919/rollout_stats.jsonl` | Per-trajectory stats (98 records) |
| `logs/apps_prof_1564511_err.log` | Full run stderr with step timings |
| `logs/monitor_apps_1564511.csv` | GPU/CPU resource monitor (5s interval) |
