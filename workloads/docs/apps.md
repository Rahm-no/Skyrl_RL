# APPS Workload

## What is APPS?

APPS (Automated Programming Progress Standard) is a competitive-programming benchmark of natural-language coding problems sourced from platforms like Codeforces and Kattis. Problems are grouped by difficulty: introductory, interview, and competition. This workload uses the **introductory** split exclusively, which contains ~1000 problems tractable for a 7B model.

The dataset is hosted as `codeparrot/apps` on HuggingFace. Because the train split was originally designed for few-shot prompting and lacks test cases, we use `test.jsonl` and split it 90/10 into train and validation. Problems that only have function-call-based tests (where input format differs) are dropped — only stdin/stdout problems are kept.

**After filtering:** ~875 train problems, ~98 validation problems.

Each problem has:
- A natural-language problem statement
- A set of hidden input/output test cases (used only for reward, never shown to the model)

---

## Example Problem (Input to the Model)

The model receives a system message followed by the problem statement as a user message:

```
[system]
You are an expert Python programmer. Read the problem statement carefully,
then write a correct Python solution that reads from stdin and writes the
answer to stdout. Enclose your code within triple backticks:
```python
# YOUR CODE HERE
```

[user]
Given a list of N integers, print the sum of all even numbers and the
product of all odd numbers on separate lines.

Input:
  First line: integer N (1 ≤ N ≤ 100)
  Next N lines: one integer per line

Output:
  Line 1: sum of even numbers (0 if none)
  Line 2: product of odd numbers (1 if none)

Sample Input:
4
3
4
7
2

Sample Output:
6
21
```

The hidden test cases for this problem might be:

| Input | Expected Output |
|-------|----------------|
| `4\n3\n4\n7\n2` | `6\n21` |
| `3\n1\n2\n3`    | `2\n3`  |
| `1\n8`          | `8\n1`  |

These test cases are stored in the parquet file under `reward_spec.ground_truth` and are never put in the prompt.

---

## Training Setup

**Hardware:** 1 node, 4× A100 GPU (40 GB or 80 GB)

**Model:** Qwen/Qwen3-4B (profiled run, job 1564511)

**Algorithm:** GRPO (Group Relative Policy Optimization)

**Environment:** `lcb` (LiveCodeBench executor) — runs generated Python code against hidden test cases

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `train_batch_size` | 256 prompts/step |
| `n_samples_per_prompt` | 8 |
| Sequences per step | 2048 |
| `max_prompt_length` | 1024 tokens |
| `max_generate_length` | 2048 tokens |
| `policy_mini_batch_size` | 64 |
| `micro_train_batch_size_per_gpu` | 4 |
| `lr` | 1e-6 |
| KL loss | enabled |
| Epochs | 5 (≈15 steps total, 3 steps/epoch) |

**Data preparation** (run once, needs internet access):

```bash
bash singularity/prep_apps_data.sh
# → /data/apps/train.parquet, /data/apps/validation.parquet
```

**Training launch:**

```bash
sbatch singularity/run_apps_grpo_a100_80gb.sh
```

---

## How the Model Is Trained — Step by Step

Each training step processes 256 problems. Here is what happens end-to-end:

### Step 1 — Generate (vLLM, ~62% of step time)

256 problems are sampled from the dataset. vLLM generates **8 candidate Python solutions per problem** = 2048 total sequences, with `max_generate_length=2048` tokens. Generation uses async continuous batching (`async_engine=true`): vLLM doesn't wait for all 2048 requests to be submitted before starting — it schedules dynamically and reuses GPU capacity as requests finish.

A typical model response looks like:

````
```python
n = int(input())
nums = [int(input()) for _ in range(n)]
even_sum = sum(x for x in nums if x % 2 == 0)
odd_prod = 1
for x in nums:
    if x % 2 != 0:
        odd_prod *= x
print(even_sum)
print(odd_prod)
```
````

### Step 2 — Reward / CPU Sandbox (LCB executor)

Each of the 2048 generated programs is executed against the hidden test cases. This runs **entirely on CPU** — the GPU is idle during this phase.

Execution flow per response (`livecodebench.py`):

1. Extract the Python code block from between the triple backticks using a regex.
2. Wrap it in a function to isolate stdin/stdout.
3. Spawn a `multiprocessing.Process` with a wall-clock timeout (6 s × number of test cases).
4. Inside the subprocess, run each test case by patching `sys.stdin` with the expected input and capturing `sys.stdout`.
5. Compare the captured output to expected output line-by-line (exact match, or floating-point via `Decimal`).

Reward signal is binary:
- **+1.0** — program passes **all** test cases
- **0.0** — any test case fails (wrong answer, runtime error, timeout, or no code block found)

Destructive OS calls (`os.fork`, `os.kill`, `shutil.rmtree`, `subprocess.Popen`, etc.) are disabled before execution via `reliability_guard()`.

Up to 32 parallel sandbox workers run concurrently (`max_env_workers=32`).

**Observed token-length asymmetry:**

| Outcome | Avg tokens generated |
|---------|---------------------|
| Correct (reward = 1.0) | ~1084 tokens |
| Wrong   (reward = 0.0) | ~2025 tokens |

Failed solutions almost always hit the 2048-token cap, so they take roughly 2× longer to generate than correct ones.

### Step 3 — Reference Log-probs (FSDP, ~11% of step time)

The frozen reference model (same weights as initialization, loaded under FSDP) computes per-token log-probabilities for all 2048 sequences. These are used for the KL penalty term in the GRPO loss.

### Step 4 — Advantage Computation (GRPO, CPU)

For each group of 8 responses to the same prompt, GRPO normalizes rewards within the group:

```
A_i = (r_i - mean(r)) / (std(r) + ε)
```

If all 8 responses have the same reward (all correct or all wrong), the group's advantages are zero and contributes no gradient. This means the model only learns from prompts where it sometimes succeeds and sometimes fails.

The pipeline waits for **all 2048 responses** before this step — GRPO requires the full group of 8 to compute the mean and std.

### Step 5 — Policy Training (FSDP, ~25% of step time)

The policy model is updated with a PPO-style clipped surrogate loss over all 2048 sequences using the computed per-token advantages. Gradient accumulation:

```
policy_mini_batch_size=64  /  micro_train_batch_size_per_gpu=4
→ 16 micro-steps per GPU per mini-batch
```

KL divergence against the reference log-probs is added to the loss with a fixed coefficient.

### Step 6 — Weight Sync (NCCL, ~0.4% of step time)

Updated FSDP training weights are broadcast via NCCL into the vLLM inference engine so the next generation step uses the freshly trained model. Uses a 3-phase chunked protocol: `start_weight_update` → `update_weights_chunk` → `finish_weight_update`.

---

## Step Timeline (mean, training steps only)

| Phase | Time | % of step |
|-------|------|-----------|
| Generate (vLLM) | ~1100 s | ~62% |
| Fwd log-probs (FSDP ref) | ~187 s | ~11% |
| Policy train (FSDP) | ~435 s | ~25% |
| Weight sync (NCCL) | ~7 s | ~0.4% |
| **Total per step** | **~1764 s (29.4 min)** | |

Generate is ~13× slower than GSM8K (1100 s vs 82 s) because code solutions are 4× longer and the model fails more often, hitting the 2048-token cap.

---

## Observed Results (job 1564511, Qwen3-4B, 4× A100-40GB, 15 steps)

Model path: `models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`

| Step | APPS pass@1 |
|------|------------|
| 0    | 12.2% |
| 5    | 17.4% |
| 10   | 16.3% |
| 15   | **22.5%** |

Training was still improving at step 15 — estimated ~30 more steps to approach convergence.
