"""Preprocess codeparrot/apps to parquet for RL training with the lcb env.

codeparrot/apps uses a custom dataset script that datasets>=3.x no longer supports,
so we download the raw JSONL files via huggingface_hub and parse them manually.

Usage:
    uv run examples/train/apps/apps_dataset.py --output_dir ~/data/apps
    uv run examples/train/apps/apps_dataset.py --output_dir ~/data/apps_mix --difficulty all
    uv run examples/train/apps/apps_dataset.py --output_dir ~/data/apps_hard --difficulty interview,competition
"""

import argparse
import json
import os
import sys

sys.set_int_max_str_digits(0)  # APPS test cases contain large integers (>4300 digits)

import pandas as pd
from huggingface_hub import hf_hub_download


SYSTEM_MESSAGE = (
    "You are an expert Python programmer. "
    "Read the problem statement carefully, then write a correct Python solution that "
    "reads from stdin and writes the answer to stdout. "
    "Enclose your code within triple backticks:\n"
    "```python\n# YOUR CODE HERE\n```"
)


def _parse_tests(input_output_str: str):
    """Return a list of {input, output, testtype} dicts, or None if unusable."""
    try:
        io = json.loads(input_output_str or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    # Skip call-based problems (fn_name present) — their input format differs.
    if io.get("fn_name"):
        return None
    inputs = io.get("inputs", [])
    outputs = io.get("outputs", [])
    if not inputs or not outputs or len(inputs) != len(outputs):
        return None
    return [{"input": inp, "output": out, "testtype": "stdin"} for inp, out in zip(inputs, outputs)]


def load_jsonl(path: str, difficulties=None):
    """Read a JSONL file, filter by difficulty, return a list of dicts.

    Args:
        difficulties: None or "all" to include everything, or a set/list of
            difficulty strings like {"introductory", "interview", "competition"}.
    """
    include_all = difficulties is None or difficulties == "all"
    if not include_all and isinstance(difficulties, str):
        difficulties = {d.strip() for d in difficulties.split(",")}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not include_all and obj.get("difficulty") not in difficulties:
                continue
            rows.append(obj)
    return rows


def process_rows(rows, split: str):
    """Convert raw APPS rows to SkyRL schema, dropping unusable examples."""
    processed = []
    for idx, obj in enumerate(rows):
        tests = _parse_tests(obj.get("input_output", ""))
        if tests is None:
            continue
        question = obj.get("question", "")
        processed.append({
            "data_source": "codeparrot/apps",
            "prompt": [{"role": "user", "content": SYSTEM_MESSAGE + "\n\n" + question}],
            "env_class": "lcb",
            "reward_spec": {
                "method": "rule",
                "ground_truth": json.dumps(tests),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "difficulty": obj.get("difficulty", "introductory"),
                "url": obj.get("url", ""),
            },
        })
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/apps")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="introductory",
        help="Comma-separated difficulty levels: introductory,interview,competition or 'all'",
    )
    parser.add_argument(
        "--max_train_dataset_length",
        type=int,
        default=None,
        help="Truncate the training split to this many examples after filtering.",
    )
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    if args.difficulty == "all":
        difficulties = "all"
    else:
        difficulties = {d.strip() for d in args.difficulty.split(",")}

    print("Downloading codeparrot/apps JSONL files from HuggingFace...")
    train_path = hf_hub_download("codeparrot/apps", "train.jsonl", repo_type="dataset")
    test_path = hf_hub_download("codeparrot/apps", "test.jsonl", repo_type="dataset")
    print(f"  train: {train_path}")
    print(f"  test:  {test_path}")

    # Note: the APPS train.jsonl rarely includes input_output (it was designed for few-shot
    # prompting, not automated evaluation).  The test.jsonl has test cases for nearly all
    # problems and is the right source for RL reward computation.  We therefore split
    # test.jsonl 90/10 into train and validation, and ignore train.jsonl.
    print(f"Loading problems from test.jsonl (difficulties={difficulties})...")
    all_rows = load_jsonl(test_path, difficulties=difficulties)
    print(f"  {len(all_rows)} rows after difficulty filter")

    # Count per difficulty
    diff_counts = {}
    for r in all_rows:
        d = r.get("difficulty", "unknown")
        diff_counts[d] = diff_counts.get(d, 0) + 1
    for d, c in sorted(diff_counts.items()):
        print(f"    {d}: {c}")

    all_data = process_rows(all_rows, "mixed")
    print(f"  {len(all_data)} rows after dropping problems with no valid tests")

    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    print(f"  90/10 split → {len(train_data)} train, {len(val_data)} val")

    if args.max_train_dataset_length is not None:
        train_data = train_data[: args.max_train_dataset_length]
        print(f"  Truncated train to {len(train_data)} examples")

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(train_data).to_parquet(os.path.join(args.output_dir, "train.parquet"), index=False)
    pd.DataFrame(val_data).to_parquet(os.path.join(args.output_dir, "validation.parquet"), index=False)
    print(f"Wrote {len(train_data)} train, {len(val_data)} val rows to {args.output_dir}")
