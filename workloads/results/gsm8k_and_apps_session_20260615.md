# SkyRL Profiling Session — 2026-06-15

## Context

Ongoing effort to benchmark SkyRL against verl (full fine-tune) and verl LoRA on RL training
tasks (GSM8K math, APPS coding). The goal is to profile training efficiency (step time breakdown,
GPU/VRAM/CPU/RAM usage) and track reward convergence across frameworks.

Previous comparison used SkyRL job 1400905 (GSM8K, 58 steps). This session adds two new SkyRL runs.

## Workload Description

### What we are running
**Reinforcement Learning from Verifiable Rewards (RLVR)** using the **GRPO** algorithm to
fine-tune a **Qwen3-4B** language model on two tasks:

| Task | Dataset | Reward signal | Difficulty |
|------|---------|---------------|------------|
| **GSM8K** | Grade-school math word problems (~7.5k train) | Exact numeric match after `####` | Easy — short answers, 512 token responses |
| **APPS** | Competitive programming problems (~875 filtered train) | Code execution correctness (pass all test cases) | Hard — long code solutions, 2048 token responses |

### Training setup
- **Framework**: SkyRL with FSDP backend, colocated vLLM inference
- **Hardware**: 4× A100-40GB on a single node
- **Algorithm**: GRPO — generates 8 candidate responses per prompt, uses reward signal to compute advantages, updates policy with PPO-style clipping
- **Weight sync**: NCCL broadcast from training weights into vLLM after each step

### Scenario
Each training step: **(1)** vLLM generates N responses per prompt → **(2)** environment scores each response → **(3)** GRPO computes advantages → **(4)** policy is trained on the batch → **(5)** weights are synced back to vLLM.
The goal is to improve the model's ability to solve the target task through repeated self-improvement loops, measured by pass@1 on a held-out eval set.

---

## Run 1 — GSM8K Aligned, bs=512 (job 1576177)

**Script**: `singularity/run_gsm8k_aligned_bs512.sh`
**Model**: Qwen3-4B | **GPUs**: 4× A100-40GB | **Duration**: 3h 20m

### Config vs baseline (job 1400905)
| | Baseline (job 1400905) | This run (job 1576177) |
|---|---|---|
| `train_batch_size` | 256 | **512** |
| Sequences/step | 2048 | **4096** |
| `max_generate_length` | 512 | 512 |
| Epochs | 1 | **2** |

### Results
| Step | GSM8K pass@1 |
|------|-------------|
| 0    | 11.5% |
| 5    | 29.6% |
| 10   | 45.6% |
| 15   | 69.2% |
| 20   | 80.5% |
| 25   | 85.4% |
| 28   | **88.2%** |

- Completed 2 epochs (28 steps), `Training done!` at 20:32:41
- Still improving at end (+2.8pp step 25→28) — not fully saturated
- **No checkpoint saved** (`ckpt_interval=0`, path was `/tmp`) — cannot resume

### Profiling (mean per step)
| Phase | Time | % |
|---|---|---|
| Generate | 82.4s | ~22% |
| Fwd LogProbs | 91.7s | ~25% |
| Policy Train | 186.2s | ~50% |
| Weight Sync | 7.8s | ~2% |
| **Step total** | **371.5s** | |

### Plots
`profiling_results/bs512_summary_plots/`
- `stage_times_pie.png`, `stage_times_stacked_bar.png`
- `avg_resource_pie.png`, `avg_resource_bar.png`
- `resource_timeseries_stacked.png` — mean GPU util, VRAM, CPU, RAM over time


## Run 2 — APPS Coding, GRPO (job 1564511)

See `profiling_results/apps_1564511_plots/summary.md` for full details and plots.

---

## Issues Fixed During Session

| Issue | Fix |
|---|---|
| Wrong timing key names (`fwd_logprobs` → `fwd_logprobs_values_reward`, `weight_sync` → `sync_weights`) | Added `TIMING_KEY` map in `plot_apps_run.py` |
| Step 0 (pre-train eval) inflating phase averages | Filter to `timing/step > 0` before averaging |
| `uv --isolated` failing (inode quota exceeded on home) | Use `/projects/I20240005/rnouaj/plot_env2` Python |

---

## Next Steps

1. **APPS**: Resume from `global_step_16` checkpoint, run ~30 more steps (10 epochs)
2. **Checkpointing**: Enable `ckpt_interval > 0` and set `ckpt_path` to persistent storage for future runs
3. **Comparison**: Update `comparison_plots/compare_runs.py` with APPS run data once converged
