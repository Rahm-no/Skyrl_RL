"""
Task 2 — GPU memory waterfall monitor.

Polls nvidia-smi at 200 ms intervals in a background thread. Accepts stage
transition events from the profiled trainer so every sample is tagged with the
current training stage.

Output: profiling_results/<run>/memory.csv
"""

import csv
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
    "mem_used_mb",
    "mem_free_mb",
    "mem_total_mb",
    "mem_reserved_mb",
]

# nvidia-smi query columns
_NVSMI_QUERY = (
    "index,memory.used,memory.free,memory.total"
)


def _find_nvidia_smi() -> Optional[str]:
    import shutil
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        return nvsmi
    for p in ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi", "/opt/cuda/bin/nvidia-smi"):
        if Path(p).is_file():
            return p
    return None


class MemoryWaterfallMonitor:
    """
    Background thread that polls nvidia-smi every `interval_ms` milliseconds.

    Usage::

        monitor = MemoryWaterfallMonitor(out_dir, stage_state)
        monitor.start()
        ...training...
        monitor.stop()
    """

    def __init__(self, out_dir: Path, stage_state: dict, interval_ms: int = 200):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / "memory.csv"
        self._stage_state = stage_state
        self._interval = interval_ms / 1000.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvsmi = _find_nvidia_smi()
        self._t0 = time.monotonic()

    def start(self):
        if self._nvsmi is None:
            print("[memory_waterfall] nvidia-smi not found — monitor disabled", flush=True)
            return
        self._t0 = time.monotonic()
        with open(self._path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mem-waterfall")
        self._thread.start()
        print(f"[memory_waterfall] started → {self._path}", flush=True)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _poll(self):
        """Run one nvidia-smi query, return list of per-GPU dicts."""
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
                if len(parts) >= 4:
                    rows.append({
                        "gpu": parts[0],
                        "mem_used_mb": parts[1],
                        "mem_free_mb": parts[2],
                        "mem_total_mb": parts[3],
                    })
            return rows
        except Exception as exc:
            print(f"[memory_waterfall] poll error: {exc}", flush=True)
            return []

    def _loop(self):
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            while not self._stop_event.is_set():
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                elapsed = time.monotonic() - self._t0
                stage = self._stage_state.get("current", "unknown")
                gpu_rows = self._poll()
                for gpu_row in gpu_rows:
                    writer.writerow({
                        "timestamp": ts,
                        "elapsed_s": round(elapsed, 3),
                        "stage": stage,
                        "gpu": gpu_row["gpu"],
                        "mem_used_mb": gpu_row["mem_used_mb"],
                        "mem_free_mb": gpu_row["mem_free_mb"],
                        "mem_total_mb": gpu_row["mem_total_mb"],
                        "mem_reserved_mb": "",  # requires torch.cuda.memory_reserved — not available host-side
                    })
                f.flush()
                self._stop_event.wait(self._interval)
