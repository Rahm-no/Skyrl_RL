# Workloads: GSM8K and APPS on SkyRL

RL training (GRPO) of **Qwen3-4B** with SkyRL on **1 node, 4× A100**, plus profiling of where the time goes. Everything runs inside a Singularity image built from `singularity/skyrl_fsdp.def`.

## The two workloads

| | GSM8K | APPS |
|---|---|---|
| Task | Grade-school math word problems | Python programming problems (introductory split) |
| Data | ~7.5k train | ~875 train / ~98 val (stdin/stdout problems only) |
| Reward | 1 if the final number matches | 1 if the code passes **all** hidden test cases (run in a CPU sandbox) |
| Max response | 512 tokens | 2048 tokens |
| Prepare data | `singularity/prep_gsm8k_data.sh` | `singularity/prep_apps_data.sh` |
| Main run | `singularity/run_gsm8k_aligned.sh` | `singularity/run_apps_grpo_a100.sh` |
| Profiled run | `singularity/run_gsm8k_profiled.sh` | `singularity/run_apps_grpo_a100_profiled.sh` |

Submit from the repo root, e.g. `sbatch singularity/run_apps_grpo_a100.sh`.

## What we tested

**GSM8K** (batch 512, 2 epochs, job 1576177): pass@1 went from **11.5% to 88.2%** in 28 steps. Each step took about 6 minutes: 50% policy training, 25% reference log-probs, 22% generation, 2% weight sync.

**APPS**: each step takes about 29 minutes, and **63% of that is generation**. Generation is slow because 73% of answers hit the 2048-token cap, and those are mostly wrong.

| APPS experiment | Script | Result |
|---|---|---|
| Baseline (async, 2048 tokens) | `run_apps_grpo_a100_profiled.sh` | pass@1 12.2% → **22.5%**, 7.3 h |
| `batched=true` | `run_apps_grpo_a100_batched.sh` | Same accuracy, generation **52% slower** |
| `max_gen=1024` | `run_apps_grpo_a100_short_gen.sh` | 2.4× faster steps, but pass@1 **collapses to 4.1%** (the model's thinking doesn't fit) |
| Round-robin routing | `run_apps_grpo_a100_roundrobin.sh` | Fixes the GPU-3 straggler (from consistent-hash routing); pass@1 21.4% |
| Thinking on | `run_apps_grpo_a100_think.sh` | pass@1 22.4% |
| Thinking off | `run_apps_grpo_a100_nothink.sh` | pass@1 **50%** at step 5 (partial run) |
| Mixed difficulty (intro+interview+competition) | `run_apps_mix_think.sh`, `run_apps_mix_nothink.sh` | Launched; no results logged yet |

**Main takeaways**
- APPS generation time is set by token count, and most tokens go to wrong answers.
- Keep async generation; `batched=true` makes it slower.
- Use round-robin routing so that one GPU doesn't end up doing the long tail alone.
- Turning thinking off helps a lot on APPS.

## Layout

- `profiling/`: stage timer, GPU and memory monitors, rollout stats. `profiling/plots/` holds the plotting scripts.
- `run_profiled.py`: training entrypoint with profiling enabled.
- `examples/train/apps/apps_dataset.py`: APPS preprocessing.
- `singularity/`: image definition and all SLURM job scripts.
- `workloads/docs/`: APPS workload details (`apps.md`) and the profiling spec.
- `workloads/results/`: write-ups of the results, with key figures in `figures/`.

`logs/`, `profiling_results/` and the `.sif` image stay local and are gitignored.
