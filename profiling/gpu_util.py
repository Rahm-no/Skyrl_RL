"""
Task 5 — GPU utilization time-series.

Runs nvidia-smi dmon at 1-second granularity in a background thread.
Each sample is tagged with the current training stage (from stage_state shared
with ProfiledRayPPOTrainer) so plots can shade stage regions.

Output: profiling_results/<run>/gpu_util.csv
"""

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


FIELDNAMES = [
    "timestamp",
    "elapsed_s",
    "stage",
    "gpu",
    "sm_util_pct",
    "mem_util_pct",
    "power_w",
    "temp_c",
]

_NVSMI_QUERY = "index,utilization.gpu,utilization.memory,power.draw,temperature.gpu"


def _find_nvidia_smi() -> Optional[str]:
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        return nvsmi
    for p in ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi", "/opt/cuda/bin/nvidia-smi"):
        if Path(p).is_file():
            return p
    return None


class GPUUtilMonitor:
    """
    Background thread sampling GPU SM util, memory util, power, temperature at
    ~1 s intervals. Each row is tagged with the current training stage.

    Usage::

        monitor = GPUUtilMonitor(out_dir, stage_state)
        monitor.start()
        ...training...
        stats = monitor.stop()   # returns summary dict
    """

    def __init__(self, out_dir: Path, stage_state: dict, interval_s: float = 1.0):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / "gpu_util.csv"
        self._stage_state = stage_state
        self._interval = interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvsmi = _find_nvidia_smi()
        self._t0 = time.monotonic()
        self._records = []

    def start(self):
        if self._nvsmi is None:
            print("[gpu_util] nvidia-smi not found — monitor disabled", flush=True)
            return
        self._t0 = time.monotonic()
        with open(self._path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-util")
        self._thread.start()
        print(f"[gpu_util] started → {self._path}", flush=True)

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self._summarize()

    def _poll(self):
        try:
            result = subprocess.run(
                [self._nvsmi, f"--query-gpu={_NVSMI_QUERY}", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            stdout = result.stdout.decode("utf-8", errors="replace")
            rows = []
            for line in stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    rows.append({
                        "gpu": parts[0],
                        "sm_util_pct": parts[1],
                        "mem_util_pct": parts[2],
                        "power_w": parts[3],
                        "temp_c": parts[4],
                    })
            return rows
        except Exception as exc:
            print(f"[gpu_util] poll error: {exc}", flush=True)
            return []

    def _loop(self):
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            while not self._stop_event.is_set():
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                elapsed = round(time.monotonic() - self._t0, 2)
                stage = self._stage_state.get("current", "unknown")
                for gpu_row in self._poll():
                    row = {
                        "timestamp": ts,
                        "elapsed_s": elapsed,
                        "stage": stage,
                        **gpu_row,
                    }
                    writer.writerow(row)
                    self._records.append(row)
                f.flush()
                self._stop_event.wait(self._interval)

    def _summarize(self) -> dict:
        if not self._records:
            return {}
        try:
            import numpy as np
            gen_rows = [r for r in self._records if r["stage"] == "generate"]
            def safe_mean(rows, key):
                vals = []
                for r in rows:
                    try:
                        vals.append(float(r[key]))
                    except (ValueError, KeyError):
                        pass
                return float(np.mean(vals)) if vals else 0.0

            gen_sm = [float(r["sm_util_pct"]) for r in gen_rows if r["sm_util_pct"] not in ("", "[N/A]")]
            return {
                "generate_samples": len(gen_rows),
                "mean_sm_util_generate": round(safe_mean(gen_rows, "sm_util_pct"), 1),
                "pct_above_80_util": round(100.0 * sum(v >= 80 for v in gen_sm) / max(len(gen_sm), 1), 1),
                "pct_below_40_util": round(100.0 * sum(v < 40 for v in gen_sm) / max(len(gen_sm), 1), 1),
            }
        except Exception:
            return {}
