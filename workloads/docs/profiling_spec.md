# Profiling task: characterize bottlenecks in single-node multi-turn GRPO

## Context

I'm running SkyRL with Qwen2.5-1.5B-Instruct on 4× A100-40GB GPUs (single node, colocated). The training config is GRPO with multi-turn (up to 5 turns), synthetic multiplication problems, group size 5, batch size 256. Policy and reference are FSDP-sharded; vLLM engines (TP=1, one per GPU) handle generation. Weights sync from FSDP to vLLM via NCCL broadcast each epoch.

I want to profile this setup to identify systems bottlenecks at small scale. The big-cluster literature assumes generation dominates wall-clock time (60–80%), but at my scale stage switching, memory pressure, and weight sync may matter more. I need data to know which.

## Working agreement

- Treat this document as the source of truth for what to build.
- Before any code changes, read the relevant SkyRL source files and propose your plan.
- Wait for explicit approval before modifying files.
- Prefer monkey-patches in a separate `profiling/` directory over edits to SkyRL internals.
- Log all measurements to `./profiling_results/<timestamp>/` so runs don't overwrite each other.
- If the spec is ambiguous about a specific integration point, ask rather than guess.

## General requirements

Help me implement the following five profiling tasks. For each, produce instrumented code I can drop into SkyRL, a script to run the profiled training, and a plotting script that generates the figure described. Use Python, matplotlib, and torch profiling APIs. Save all data to `./profiling_results/` with timestamped subdirectories. Run for at least 2 full epochs after a 1-epoch warmup so I get stable numbers.

---

## Task 1: Per-stage time breakdown

Instrument each of the 5 stages of the GRPO training loop and record wall-clock time per stage per iteration:

1. Generation (vLLM rollout)
2. Environment step (reward computation)
3. Advantage estimation
4. Policy update (FSDP forward/backward)
5. Weight sync (FSDP → vLLM NCCL broadcast)

Also capture **stage-switching overhead** — the time between when one stage ends and the next begins (e.g., vLLM sleep/wake, KV cache release, FSDP gather). Treat this as a sixth bucket. Use `torch.cuda.synchronize()` before each timer to avoid attributing async work to the wrong stage.

**Output:** a stacked bar chart, one bar per iteration, segmented by stage. Also print a summary table with mean, median, p95 time per stage and percentage of total iteration time. Compare against the rough 60/15/25 split from large-scale RLHF papers — for a 1.5B model with short responses, I expect generation to be a smaller fraction and stage switching a bigger fraction. Highlight any stage taking >10% that the literature treats as negligible.

---

## Task 2: Memory waterfall

Use `torch.cuda.memory_snapshot()` plus high-frequency `nvidia-smi` polling (every 100 ms) on all 4 GPUs through one full iteration. Record:

- Total allocated VRAM
- Reserved VRAM
- KV cache footprint (query vLLM if API available; otherwise infer from vLLM engine state)
- Optimizer state size
- Activation memory peak

**Output:** a stacked area chart of VRAM usage vs time for GPU 0 (with per-GPU charts as subplots), with vertical lines annotating stage transitions. Identify peaks (when KV cache + optimizer state coexist) and valleys (between stages when memory is released). Quantify the maximum simultaneous footprint and how much headroom remains on the 40 GB budget.

---

## Task 3: Weight sync cost breakdown

For the FSDP → vLLM NCCL broadcast at the end of each epoch, decompose into:

1. **Gather phase:** time to gather FSDP shards into full tensors
2. **Conversion phase:** any layout/dtype conversion between FSDP and vLLM format
3. **Broadcast phase:** time to broadcast full weights to all 4 vLLM engines
4. **Load phase:** time for vLLM engines to install new weights and rebuild any internal state

Use `torch.cuda.Event` for fine-grained timing. For a 1.5B model in BF16 that's ~3 GB total; report achieved bandwidth (GB/s) for each phase and compare to NVLink peak bandwidth on A100 (~600 GB/s intra-node).

**Output:** a horizontal bar chart breaking down one weight sync event into the four phases, with bandwidth annotations. Print whether the bottleneck is gather or broadcast and the theoretical-vs-achieved bandwidth ratio.

---

## Task 4: Long-tail distribution in multi-turn rollouts

For each prompt in one full epoch's rollout, record:

- Number of turns used (1 to 5)
- Total tokens generated across all turns
- Tokens per turn
- Whether the prompt was solved (reward = 1.0) and at which turn
- Per-prompt wall-clock generation time

**Output:**

- Histogram of total tokens per prompt
- Histogram of turns used
- A CDF showing what fraction of prompts account for what fraction of total generation time (a Pareto-style curve is expected)
- A scatter plot of (turns used) vs (total tokens) to see if late-turn prompts are also long-token prompts

**Quantify:** the p99/p50 token ratio (the long-tail severity), and what fraction of total generation time comes from the slowest 10% of prompts. This is the small-scale, multi-turn version of the long-tail problem.

---

## Task 5: GPU utilization during generation

Run `nvidia-smi dmon -s u -d 1` (or DCGM if available) in parallel with training, capturing per-GPU SM utilization, memory utilization, and memory bandwidth utilization at 1-second granularity. Tag each sample with the current training stage by syncing timestamps with the per-stage timer from Task 1.

**Output:** a time-series line chart of SM utilization per GPU with the stage transitions shaded in the background. Specifically isolate the generation stage and report:

- Mean and median SM util during generation
- Time spent above 80% util (compute-bound regime)
- Time spent below 40% util (likely memory-bandwidth-bound or tail-sample regime)
- Whether util drops sharply near the end of each generation phase (long-tail signature)

---

## Deliverables

1. A single instrumented training script that runs all 5 profilers concurrently with minimal overhead (use sampling rather than per-op tracing where possible; per-op tracing should be off by default and toggled by a flag).
2. Five plotting scripts, one per task, each producing a publication-ready figure (300 DPI, clear axes, labeled legends).
3. A summary `report.md` that pulls together the headline numbers from all five tasks: time breakdown table, peak VRAM, weight sync cost, long-tail severity, generation GPU util. End with a paragraph diagnosing which bottleneck is dominant for this configuration.

## Constraints

- Keep instrumentation overhead below 5% of total iteration time — verify this by running with profiling off as a baseline and comparing. If any profiler adds more than 5%, sample less frequently or move to async logging.
- Start by inspecting the SkyRL codebase to find where each of the 5 stages is implemented, then propose the integration points before writing code.
- Ask me before modifying any files in the SkyRL source tree — prefer monkey-patching or wrapping at the entrypoint script (`main_multiply.py`) over editing library internals.

## Suggested file layout

```
your-skyrl-workspace/
├── docs/
│   └── profiling_spec.md          ← this file
├── profiling/                     ← new directory for instrumentation code
│   ├── __init__.py
│   ├── stage_timer.py
│   ├── memory_waterfall.py
│   ├── weight_sync.py
│   ├── rollout_stats.py
│   ├── gpu_util.py
│   └── plots/
├── profiling_results/             ← outputs, gitignored
└── run_profiled.py                ← entrypoint that wraps main_multiply.py
```