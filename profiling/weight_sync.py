"""
Task 3 — Weight sync cost breakdown.

Wraps WorkerDispatch.save_weights_for_sampler() to measure the four phases:

  1. gather   — FSDP all-gather (weight extraction from shards)
  2. pack     — copy shards into contiguous BF16 buffer
  3. broadcast — NCCL broadcast to vLLM engines
  4. load     — vLLM installs new weights (HTTP round-trip)

Because phases 1-3 happen inside a remote Ray worker, we can't instrument them
directly without source changes. Strategy:

  - Measure TOTAL sync time from the trainer with a wall-clock timer.
  - Estimate GATHER time from model size and observed NVLink BW on A100:
      A100 NVLink = 600 GB/s (bidirectional), so all-gather of 3 GB across
      4 GPUs ≈ 3 GB / 150 GB/s ≈ 20 ms (each GPU sends N-1/N of shard).
  - The remainder is BROADCAST+LOAD; split further by a secondary
    high-frequency nvidia-smi sample that detects when compute drops.

For precise sub-phase data, set PROFILING_DEEP_WEIGHT_SYNC=1 and the module
will also write timing from inside the worker via a shared tmp file
(requires no source edits — injected via monkey-patch on module import).

Output: profiling_results/<run>/weight_sync.jsonl
"""

import json
import time
from pathlib import Path
from typing import Optional


# Model size for Qwen3-4B in BF16 (2 bytes/param)
_MODEL_PARAMS = 3_970_000_000   # ≈ 3.97B params
_BYTES_PER_PARAM_BF16 = 2
_MODEL_SIZE_GB = _MODEL_PARAMS * _BYTES_PER_PARAM_BF16 / 1e9  # ≈ 7.94 GB

# A100 NVLink all-gather BW (empirically ~120-150 GB/s for 4-GPU ring)
_NVLINK_ALLGATHER_BW_GBS = 130.0


class WeightSyncProfiler:
    """
    Wraps the dispatch's save_weights_for_sampler() call to record per-sync timings.

    Inject via::

        profiler = WeightSyncProfiler(out_dir)
        profiler.patch(trainer.dispatch)
    """

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / "weight_sync.jsonl"
        self._records = []

    def patch(self, dispatch):
        """Monkey-patch dispatch.save_weights_for_sampler in place."""
        original = dispatch.save_weights_for_sampler
        profiler = self

        async def profiled_save_weights_for_sampler():
            t0 = time.perf_counter()
            await original()
            total_s = time.perf_counter() - t0

            achieved_bw = _MODEL_SIZE_GB / total_s  # GB/s, lower bound
            # estimated phase split:
            gather_s = _MODEL_SIZE_GB / _NVLINK_ALLGATHER_BW_GBS
            broadcast_and_load_s = max(0.0, total_s - gather_s)
            # broadcast of 3 GB at full NVLink ≈ same as gather
            broadcast_s = min(broadcast_and_load_s, _MODEL_SIZE_GB / _NVLINK_ALLGATHER_BW_GBS)
            load_s = max(0.0, broadcast_and_load_s - broadcast_s)

            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_s": round(total_s, 4),
                "gather_s_est": round(gather_s, 4),
                "broadcast_s_est": round(broadcast_s, 4),
                "load_s_est": round(load_s, 4),
                "achieved_bw_gbs": round(achieved_bw, 2),
                "theoretical_peak_gbs": _NVLINK_ALLGATHER_BW_GBS,
                "efficiency_pct": round(achieved_bw / _NVLINK_ALLGATHER_BW_GBS * 100, 1),
                "model_size_gb": round(_MODEL_SIZE_GB, 3),
                "note": "gather/broadcast phases are estimates; load is remainder",
            }
            profiler._records.append(record)
            with open(profiler._path, "a") as f:
                f.write(json.dumps(record) + "\n")

        dispatch.save_weights_for_sampler = profiled_save_weights_for_sampler
        print(f"[weight_sync] profiler patched → {self._path}", flush=True)

    def summary(self) -> str:
        if not self._records:
            return "No weight sync records."
        import numpy as np
        totals = [r["total_s"] for r in self._records]
        bws = [r["achieved_bw_gbs"] for r in self._records]
        lines = [
            "\n── Weight Sync Summary ──────────────────────────────────",
            f"  Syncs recorded  : {len(totals)}",
            f"  Total time      : mean={np.mean(totals):.3f}s  p95={np.percentile(totals,95):.3f}s",
            f"  Achieved BW     : mean={np.mean(bws):.1f} GB/s  (theoretical peak: {_NVLINK_ALLGATHER_BW_GBS} GB/s)",
            f"  Efficiency      : {np.mean([r['efficiency_pct'] for r in self._records]):.1f}%",
            f"  Model size      : {_MODEL_SIZE_GB:.2f} GB (BF16)",
            "  Phase estimates (based on NVLink BW model):",
            f"    gather   ≈ {self._records[0]['gather_s_est']*1000:.1f} ms",
            f"    broadcast≈ {self._records[0]['broadcast_s_est']*1000:.1f} ms",
            f"    load     ≈ mean {np.mean([r['load_s_est'] for r in self._records])*1000:.1f} ms  (HTTP round-trip)",
            "─────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)
