# APPS Ablation Story: From Baseline to Throughput Experiments

**Model**: Qwen3-4B | **Hardware**: 4x A100-40GB | **Algorithm**: GRPO | **Task**: APPS introductory (competitive programming)

---

## Act 1 — The Baseline (job 1564511): `batched=false`, `max_gen=2048`

We ran Qwen3-4B on APPS introductory problems with GRPO for 5 epochs (15 training steps). The model learned steadily:

| Eval Point | pass@1 |
|------------|--------|
| Step 0 (before training) | 12.2% |
| Step 5 | 17.4% |
| Step 10 | 16.3% (dip) |
| Step 15 | **22.5%** |

But each step took **29.4 minutes**, and generation alone consumed **62.9%** of that (~18.4 min/step). For comparison, generation on GSM8K (a math task with short answers) takes only ~82 seconds — here it was **1100 seconds**, roughly 13x slower. The total training time was 7.3 hours for 15 steps.

The natural question: **why is generation so slow, and what can we do about it?**

---

## Act 2 — Generation Phase Deep-Dive

To answer that question, we profiled a single training step at the per-trajectory level (job 1651919, 98 validation trajectories). This analysis produced the key findings that motivate the subsequent experiments.

### Finding 1: Most trajectories hit the token cap

| Metric | Value |
|--------|-------|
| Trajectories hitting the 2048-token cap | **73.5%** (72 / 98) |
| Trajectories finishing below cap | 26.5% (26 / 98) |
| Mean tokens generated | 1869 |
| Median tokens generated | 2048 (the cap itself) |

Nearly three-quarters of all generated sequences exhaust the full 2048-token budget without producing a valid solution.

### Finding 2: Generation time is almost entirely determined by token count

The correlation between output tokens and generation wall-clock time is **r = 0.98**. This means scheduling overhead, prefill time, and sandbox execution are negligible — if you know how many tokens a trajectory will produce, you know how long it will take.

### Finding 3: Incorrect solutions are the expensive ones

| Outcome | Count | Median gen time | Median tokens |
|---------|-------|----------------|---------------|
| Correct (reward = 1.0) | 24 (24.5%) | 37.8s | ~1350 |
| Incorrect (reward = 0.0) | 74 (75.5%) | 55.5s | **2048 (capped)** |

Correct solutions finish early (~1350 tokens) because they produce a working program and stop. Incorrect solutions almost always run to the 2048-token cap — the model keeps generating without arriving at a valid answer. Failed attempts take roughly **1.5x longer** to generate than successful ones.

### Finding 4: GPU 3 is a persistent straggler during generation

The per-trajectory profiling (job 1651919) measured the **end-of-phase straggler gap** — the time between the first and last vLLM engine reporting "done." That gap was small (~0.7s mean). But the nvidia-smi resource monitor (5-second sampling) reveals a much larger imbalance **within** each generate phase.

Using precise phase boundaries from the stage timer logs, GPU activity during the generate phase (active = util > 10%) across all 15 training steps:

| GPU | Avg % of generate time active | Role |
|-----|-------------------------------|------|
| GPU 0 | **53%** | Finishes early, idles ~47% of generate |
| GPU 1 | **53%** | Finishes early, idles ~47% of generate |
| GPU 2 | **54%** | Finishes early, idles ~46% of generate |
| GPU 3 | **84%** | Last to finish in **all 15 steps** |

*(See `apps_per_gpu_active_idle.png`, bottom panel — GPU 3 avg: 84%, others avg: 53-54%)*

The spread between GPU 3 and the others averages **~31 percentage points** per step (range: 8pp to 51pp). This is also visible in `apps_gpu_straggler_timeline.png` (middle panel), which shows the GPU divergence metric averaging **21.9%** across the full run.

**The tail is where the waste concentrates.** The `08_gpu_detail_generate_tail.png` plot zooms into the last 3 minutes of the generate phase for steps 1, 8, and 15. In each case, GPUs 0-2 drop to idle (<5% util) while GPU 3 continues at ~90% for 2-6 minutes before generate ends. Across all 15 steps, GPU 3 runs **solo** (the other 3 GPUs idle) for an average of **275 seconds per step** — that is **25% of total generate time**.

| Metric | Value |
|--------|-------|
| GPU 3 solo time per step (mean) | **275s (4.6 min)** |
| GPU 3 solo as % of generate | **25%** |
| Total GPU 3 solo across 15 steps | **4125s (69 min)** |
| Wasted GPU-seconds (3 idle GPUs during solo) | **12375s (206 GPU-min)** |
| Waste as % of all GPU-seconds in generate | **44%** |
| Theoretical speedup from perfect load balance | **~2.1x on generate** |

This straggler effect is **systematic, not random** — GPU 3 is the last to finish in all 15 steps, never GPUs 0-2. This points to a structural cause rather than stochastic variation in request lengths. Possible explanations:
1. **Non-uniform request routing**: the vLLM router may not distribute requests evenly across the 4 engines
2. **GPU 3 hosting additional processes** (e.g., Ray head node, training coordinator) that steal compute cycles
3. **NUMA locality or PCIe topology effects** that make GPU 3 slower at serving requests

The practical impact: if load were perfectly balanced across all 4 GPUs, generate would be ~2.1x faster (468s → 222s per step), saving ~4 min/step or ~1 hour off the total 7.3h run. Combined with the token-cap waste (Finding 2-3), this represents a compounding inefficiency: reducing token-cap saturation would both cut raw token volume *and* reduce per-engine length variance, which would in turn reduce the straggler effect.

### Why Qwen3-4B is especially affected

Qwen3-4B uses `<think>` tags to produce chain-of-thought reasoning before writing code. This "thinking" phase consumes ~500-700 tokens before any actual Python code appears. On incorrect solutions, the model thinks at length, writes partial code, then runs out of space — wasting the entire token budget.

### Diagnosis

The generation bottleneck comes from a specific inefficiency: **the majority of the token budget is spent on failed trajectories that exhaust the token limit**. This pointed to two optimization levers:

1. **Scheduling efficiency** — can vLLM process the 2048 requests faster if we submit them differently? (test: `batched=true`)
2. **Token budget** — can we cap generation shorter and accept a quality tradeoff? (test: `max_gen=1024`)

---

## Interlude — What the CPU and GPU are doing during generate

### The generate phase is not GPU-only

With `batched=false`, the generator runs **2048 concurrent async `agent_loop` tasks** (one per trajectory). Each trajectory follows this cycle:

```
Per trajectory (2048 total, running in parallel):
  1. Submit prompt to vLLM engine (GPU) → generate code solution
  2. Receive generated code
  3. Run env.step() = LCB sandbox (CPU) → execute code against test cases → get reward
  4. Done (APPS is single-turn: max_turns=1)
```

The sandbox (`env.step`) runs via a **ThreadPoolExecutor** with `max_env_workers=32` threads, all on CPU. Generation (GPU) and sandbox execution (CPU) are **interleaved per-trajectory, not sequential** — both happen concurrently inside the single `generate` timing bucket.

When a trajectory finishes generating on GPU, its code is immediately handed to a CPU sandbox thread for execution. Meanwhile, other trajectories are still generating on GPU. The sandbox runs test cases with a 6-second wall-clock timeout per test case, using `signal.alarm()` and `reliability_guard()` to isolate destructive calls.

### CPU and GPU utilization within a generate phase

Data from Step 8 (1176s generate phase), split into thirds:

| Segment | Time | CPU util | GPUs 0-2 util | GPU 3 util | What's happening |
|---------|------|----------|---------------|------------|------------------|
| **Early** (0-33%) | 0-388s | **28%** | **80%** (99% active) | **79%** | All 4 vLLM engines generating at full throughput. CPU busy sandboxing completed solutions as they arrive. |
| **Mid** (33-66%) | 394-777s | **23%** | 25-51% (33-66% active) | **81%** | GPUs 0-2 start draining their queues and going idle. Fewer new solutions → fewer sandbox tasks → CPU drops. |
| **Late** (66-100%) | 783-1171s | **18%** | **0%** (all idle) | **37%** | GPUs 0-2 completely done. Only GPU 3 still generating. Almost no new code to sandbox → CPU at its lowest. |

*(CPU utilization from `monitor_apps_1564511.csv`, 5-second sampling. See also `apps_resource_timeseries_phases.png`, CPU panel.)*

**Key insight: CPU utilization tracks GPU activity, not the other way around.** The CPU is not a bottleneck — it is *starved*. When GPUs are busy generating many solutions in parallel, CPU sandbox workers have plenty of code to execute (~28% = ~9 cores busy on a 32-core node). As GPUs finish and go idle, fewer new solutions arrive for sandboxing, so CPU drops to ~18%.

### Why CPU utilization is ~24% on average, not higher

On a 32-core node, 24% CPU utilization means ~8 cores busy on average. The 32-thread sandbox pool has capacity to spare because:

1. **Sandbox execution is I/O-bound**: most time is spent waiting on subprocess timeouts (6s per test case), not doing compute
2. **Solutions arrive at the rate vLLM produces them**: the sandbox is bottlenecked by GPU generation throughput, not CPU capacity
3. **In the late phase, there is nothing left to sandbox**: with 3 GPUs idle and only GPU 3 producing new solutions, the CPU sandbox pool is mostly idle

### The gap between generate and training is zero

The `apps_gap_analysis.png` plot confirms: the idle gap after generate ends and before fwd_logprobs begins is **~0 seconds** on every step. There is no separate CPU-bound reward phase between generate and training — the sandbox work is fully absorbed into the generate phase because it runs concurrently with vLLM generation via the async agent_loop.

The `postprocess_generator_output` phase after generate takes only **0.02 seconds**, confirming all sandbox work finishes before the last vLLM request completes.

### Summary: resource utilization during generate

| Resource | During generate | Role |
|----------|----------------|------|
| **GPU 0-2** | Active 53% of the time, idle 47% | vLLM inference engines. Finish early due to `consistent_hash` routing bias. |
| **GPU 3** | Active 84% of the time | vLLM inference engine. Overloaded by routing, always last to finish. |
| **CPU** (32 cores) | ~24% average (~8 cores) | LCB sandbox: execute generated Python solutions against test cases. Throughput-limited by GPU, not by CPU capacity. |
| **RAM** | ~27% (~140 GB of 516 GB) | Stable throughout; no pressure. |

---

## Finding 5: GPU straggler pattern depends on routing and sequence length variance

The GPU 3 straggler is **not hardware-specific** — it is caused by `consistent_hash` request routing combined with high sequence length variance. This is confirmed by comparing the straggler pattern across all three runs:

| Run | Routing path | Seq length variance | Consistent straggler? | Last GPU | Solo tail/step |
|-----|-------------|--------------------|-----------------------|----------|---------------|
| **Baseline** (batched=false, 2048) | `consistent_hash` via vllm-router | High (1350-2048 tokens) | **Yes — GPU 3 in 15/15 steps** | GPU 3 | 275s (25%) |
| **Batched=true** (2048) | Single batch call (different path) | High (1350-2048 tokens) | **Yes — GPU 1 in 14/15 steps** | GPU 1 | 139s |
| **Short-gen** (batched=false, 1024) | `consistent_hash` via vllm-router | **Low** (~980 tokens, nearly all capped) | **No straggler** — rotates | varies | ~5s (~0%) |

*(Per-GPU activity computed from `monitor_apps_*.csv` using stage-timer phase boundaries.)*

The mechanism:
- **Baseline**: `consistent_hash` routing hashes each trajectory's `session_id` to a fixed engine. The hash distribution is uneven, and GPU 3's engine gets more long (token-capped) requests. With high length variance (1350 vs 2048 tokens), this creates a persistent ~275s tail.
- **Batched=true**: Different scheduling path (single batch call), so a *different* engine ends up overloaded (GPU 1 instead of GPU 3). The imbalance persists because length variance is still high.
- **Short-gen (1024)**: Same `consistent_hash` routing, so the hash bias still exists — but with `max_gen=1024`, almost all sequences hit the cap (~980 avg tokens). **Length variance collapses**, so even uneven request counts barely matter. All GPUs stay >93% active and finish within seconds of each other.

**Root cause** (`skyrl/backends/skyrl_train/inference_servers/utils.py:218`): the router uses `policy="consistent_hash"`, which maps deterministic `session_id` headers (from `trajectory_id`, line 308-310 in `skyrl_gym_generator.py`) to engine indices. Switching to `round_robin` routing via `generator.inference_engine.router_init_kwargs.policy=round_robin` would guarantee even distribution.

*(See `apps_gpu3_straggler_waste.png` for the visualization of wasted GPU-time.)*

---

## Act 3 — Ablation A: `batched=true` (job 1657846)

### Hypothesis

With `batched=false` (the baseline), SkyRL submits the 2048 generation requests individually to vLLM's async engine. vLLM processes them via **continuous batching** — it dynamically schedules requests and reclaims GPU capacity as individual requests finish.

With `batched=true`, SkyRL submits all 2048 requests as a single synchronous batch call. The hypothesis was that vLLM might schedule a large batch more efficiently than thousands of individually submitted requests.

### Result: Slower, no quality difference

| Metric | Baseline (`batched=false`) | `batched=true` | Delta |
|--------|---------------------------|----------------|-------|
| Mean generate time | 1101s (18.4 min) | **1674s (27.9 min)** | **+52%** |
| Mean step total | 1751s (29.2 min) | **2318s (38.6 min)** | **+32%** |
| Generate % of step | 62.9% | **72.2%** | +9.3pp |
| Final eval pass@1 | 22.5% | 22.5% | 0 |
| Total wall time | 7.3h | **9.7h** | +2.4h |

The model learned exactly the same thing (22.5% pass@1) but took 2.4 hours longer to get there. Generation became the even more dominant bottleneck, growing from 63% to 72% of step time.

### Explanation

With `batched=false`, vLLM's continuous batching works optimally for **variable-length** workloads:
- Correct solutions finish at ~1350 tokens and **immediately free their GPU slot**
- New requests fill that slot, maintaining high GPU utilization throughout
- The "fast finishers" effectively subsidize the "slow finishers" by releasing resources early

With `batched=true`, vLLM processes the whole batch synchronously:
- All 2048 requests are submitted at once
- GPU slots occupied by short (correct) solutions **cannot be reclaimed** until the batch completes
- The batch completion time is gated by the slowest requests (the token-capped failures)
- Effectively, the entire batch runs at the speed of the worst case

On a workload where correct solutions are ~1350 tokens and incorrect ones are ~2048 tokens, the variance is high, and continuous batching's ability to reclaim capacity from early finishers is critical. Removing that ability (via `batched=true`) wastes ~50% more wall time for zero quality benefit.

**Conclusion: `batched=true` is a trap for variable-length generation.** Only use it when response lengths are uniform.

---

## Act 4 — Ablation B: `max_gen=1024` (job 1657847)

### Hypothesis

Since 73.5% of trajectories hit the 2048-token cap (and most are wrong anyway), halving the cap should nearly halve generation time. The risk: Qwen3-4B's `<think>` reasoning phase consumes ~500-700 tokens before code begins, so 1024 tokens might not leave enough room for the actual solution.

### Result: 3.7x faster generation, but quality collapsed

| Metric | Baseline (`max_gen=2048`) | `max_gen=1024` | Delta |
|--------|--------------------------|----------------|-------|
| Mean generate time | 1101s (18.4 min) | **301s (5.0 min)** | **-73% (3.7x faster)** |
| Mean step total | 1751s (29.2 min) | **731s (12.2 min)** | **-58% (2.4x faster)** |
| Generate % of step | 62.9% | **41.2%** | -21.7pp |
| Eval pass@1 at step 0 | 12.2% | **0.0%** | -12.2pp |
| Final eval pass@1 (step 15) | 22.5% | **4.1%** | **-18.4pp** |
| Final train avg_raw_reward | 0.36 | 0.18 | -50% |
| Avg response length | ~1870 tokens | ~980 tokens | -48% |
| Total wall time | 7.3h | **3.0h** | -4.3h |

The throughput improvement was dramatic — steps ran 2.4x faster and total wall time dropped from 7.3 to 3.0 hours. But pass@1 was catastrophic: **0%** at step 0 (the model literally cannot produce any correct code in 1024 tokens with thinking enabled), climbing to only **4.1%** after 15 steps of training.

### Explanation

The quality collapse has a clear mechanism:

1. **Thinking fills the budget**: Qwen3-4B's `<think>` tags produce 500-700 tokens of chain-of-thought reasoning before any Python code. At 1024 max tokens, this leaves only 300-500 tokens for the actual solution.

2. **Code gets truncated**: Most APPS solutions require 200-400 tokens of code. With the thinking overhead, there is barely enough room — and any slightly longer solution gets cut off mid-function, producing syntactically invalid Python.

3. **No learning signal at step 0**: With 0% pass@1 at initialization, the model starts with essentially no gradient signal. GRPO needs some trajectories to succeed (reward=1) and some to fail (reward=0) within each group of 8 — if all 8 fail, the advantage is zero and no learning happens. The model slowly bootstraps to 4.1% but can never catch up because it can't generate long enough solutions to learn from.

4. **The avg response length (~980 tokens) confirms near-universal cap saturation** — responses are still hitting the ceiling, just a lower one. The model didn't get shorter; it just got truncated.

**Conclusion: For thinking models, there is a hard floor on `max_generate_length` below which the model cannot fit reasoning + code.** 1024 is below that floor for Qwen3-4B on APPS. A middle ground (e.g., 1536) was not tested but might preserve most of the speedup with less quality loss.

---

## Summary Comparison

| Config | Step Time | Gen Time | Gen % | Final pass@1 | Wall Time |
|--------|-----------|----------|-------|-------------|-----------|
| **Baseline** (batched=false, 2048) | 29.2 min | 18.4 min | 63% | **22.5%** | 7.3h |
| batched=true, 2048 | 38.6 min | 27.9 min | 72% | 22.5% | 9.7h |
| batched=false, 1024 | 12.2 min | 5.0 min | 41% | 4.1% | 3.0h |

---

## Key Takeaways

1. **Generation dominates APPS training time** (63% of each step) because code solutions are long (2048 tokens) and 73.5% of trajectories hit the cap. This is the primary systems bottleneck.

2. **Generation time is driven by token count** (r=0.98), but **GPU 3 is a persistent straggler** — active 84% of generate time vs 53% for GPUs 0-2 (`apps_per_gpu_active_idle.png`). GPU 3 runs solo for 25% of generate time (275s/step), wasting 44% of all GPU-seconds during generation on idle GPUs. This is structural (GPU 3 is last in all 15 steps), not random. Reducing token-cap saturation would both cut raw token volume and reduce the length variance that feeds the straggler.

3. **`batched=true` makes things worse (+52% gen time)** because it disables vLLM's ability to reclaim GPU slots from early-finishing requests. On variable-length workloads, continuous batching (the `batched=false` default) is strictly better.

4. **Cutting `max_gen` gives linear speedup but non-linear quality loss.** Halving the token budget (2048 -> 1024) gave 3.7x faster generation but pass@1 dropped from 22.5% to 4.1%. For thinking models, the reasoning tokens are not optional — they are load-bearing.

5. **The real optimization target is reducing wasted tokens on incorrect trajectories.** Possible approaches (not yet tested):
   - Early stopping on low-quality partial generations
   - A middle-ground token cap (e.g., 1536) that preserves most quality
   - A larger model (7B+) that fails less often, reducing cap saturation rate
   - Disabling or shortening the `<think>` phase during training

---

## Data Sources

| Run | Job ID | Logs | Profiling Results |
|-----|--------|------|-------------------|
| Baseline | 1564511 | `logs/apps_prof_1564511_err.log` | `profiling_results/apps_1564511_plots/` |
| Generation analysis | 1651919 | — | `profiling_results/apps_generate_analysis/` |
| batched=true | 1657846 | `logs/apps_batched_1657846_err.log` | `profiling_results/apps_batched_true_job1657846/` |
| max_gen=1024 | 1657847 | `logs/apps_short_1657847_err.log` | `profiling_results/apps_short_gen_1024_job1657847/` |
