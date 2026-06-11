#!/usr/bin/env python3
"""
Resource monitor: polls CPU, RAM, GPU (via nvidia-smi), and disk I/O
at a fixed interval and appends rows to a CSV file.

CPU % is measured against allocated SLURM cores via cgroup cpuacct, so
100% means all allocated cores are saturated (not machine-wide).

Usage:
    python monitor_resources.py --output /path/to/monitor.csv --interval 5
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


# ── CPU via cgroup (job-scoped) ───────────────────────────────────────────────

def _find_cgroup_cpuacct():
    """
    Return path to the cpuacct.usage file for this SLURM job, or None.
    Tries cgroup v1 paths first, then parses /proc/self/cgroup for v2.
    """
    job_id = os.environ.get("SLURM_JOB_ID", "")
    uid = os.getuid()

    # cgroup v1 — common SLURM layouts
    v1_candidates = [
        f"/sys/fs/cgroup/cpuacct/slurm/uid_{uid}/job_{job_id}/cpuacct.usage",
        f"/sys/fs/cgroup/cpu,cpuacct/slurm/uid_{uid}/job_{job_id}/cpuacct.usage",
        f"/sys/fs/cgroup/cpuacct/system.slice/slurmstepd.scope/job_{job_id}/cpuacct.usage",
    ]
    for p in v1_candidates:
        if os.path.isfile(p):
            return ("v1", p)

    # cgroup v2 — derive path from /proc/self/cgroup
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                # "0::/slice/slurmstepd.scope/job_NNNNN/..."
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    rel = parts[2].lstrip("/")
                    # Walk up to the job-level directory
                    segs = rel.split("/")
                    for i, seg in enumerate(segs):
                        if seg.startswith("job_"):
                            job_cg = "/sys/fs/cgroup/" + "/".join(segs[: i + 1])
                            stat_path = os.path.join(job_cg, "cpu.stat")
                            if os.path.isfile(stat_path):
                                return ("v2", stat_path)
    except OSError:
        pass

    return None


def _read_cgroup_cpu_ns(cg):
    """Read cumulative CPU nanoseconds from a cgroup descriptor (kind, path)."""
    kind, path = cg
    if kind == "v1":
        with open(path) as f:
            return int(f.read().strip())
    else:  # v2 cpu.stat — look for usage_usec
        with open(path) as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1]) * 1000  # µs → ns
    return 0


def _cpu_pct_cgroup(prev_ns, curr_ns, elapsed_s, num_cores):
    """% of allocated cores used: 100% = all num_cores fully busy."""
    delta_ns = curr_ns - prev_ns
    if elapsed_s <= 0 or num_cores <= 0:
        return 0.0
    return round(delta_ns / 1e9 / elapsed_s / num_cores * 100.0, 2)


# ── CPU fallback: machine-wide via /proc/stat ─────────────────────────────────

def _read_proc_stat():
    with open("/proc/stat") as f:
        line = f.readline()
    return [int(x) for x in line.split()[1:9]]


def _cpu_percent_proc(prev, curr):
    delta_total = sum(curr) - sum(prev)
    delta_idle = curr[3] - prev[3]
    if delta_total == 0:
        return 0.0
    return round(100.0 * (1.0 - delta_idle / delta_total), 2)


def _read_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])  # kB
    return info


def _read_diskstats():
    """Returns {device: (reads, read_sectors, writes, write_sectors)}."""
    stats = {}
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if len(p) >= 14 and not p[2].startswith("loop") and not p[2].startswith("dm-"):
                stats[p[2]] = (int(p[3]), int(p[5]), int(p[7]), int(p[9]))
    return stats


def _disk_delta_mbps(prev, curr, elapsed):
    """Aggregate read/write MB/s across all tracked devices."""
    total_read_sectors = 0
    total_write_sectors = 0
    for dev in curr:
        if dev in prev:
            total_read_sectors += curr[dev][1] - prev[dev][1]
            total_write_sectors += curr[dev][3] - prev[dev][3]
    sector = 512  # bytes
    read_mbps = round(total_read_sectors * sector / 1e6 / elapsed, 3)
    write_mbps = round(total_write_sectors * sector / 1e6 / elapsed, 3)
    return read_mbps, write_mbps


def _find_nvidia_smi():
    """Return path to nvidia-smi, searching common HPC locations."""
    import shutil
    candidate = shutil.which("nvidia-smi")
    if candidate:
        return candidate
    for path in (
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/cuda/bin/nvidia-smi",
        "/opt/cuda/bin/nvidia-smi",
    ):
        if os.path.isfile(path):
            return path
    return None


def _gpu_stats(num_gpus=4):
    """Query nvidia-smi; returns list of per-GPU dicts."""
    nvsmi = _find_nvidia_smi()
    if nvsmi is None:
        print("[monitor] nvidia-smi not found", file=sys.stderr, flush=True)
        return []
    try:
        result = subprocess.run(
            [
                nvsmi,
                "--query-gpu=index,utilization.gpu,utilization.memory,"
                "memory.total,memory.used,memory.free,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            env={**os.environ, "PATH": os.environ.get("PATH", "") + ":/usr/bin:/usr/local/bin"},
        )
        result.stdout = result.stdout.decode("utf-8", errors="replace")
        result.stderr = result.stderr.decode("utf-8", errors="replace")
        if result.returncode != 0:
            print(f"[monitor] nvidia-smi error (rc={result.returncode}): {result.stderr.strip()}",
                  file=sys.stderr, flush=True)
            return []
        rows = []
        for line in result.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 8:
                rows.append(
                    {
                        "gpu_util_pct": parts[1],
                        "gpu_mem_util_pct": parts[2],
                        "gpu_mem_total_mb": parts[3],
                        "gpu_mem_used_mb": parts[4],
                        "gpu_mem_free_mb": parts[5],
                        "gpu_power_w": parts[6],
                        "gpu_temp_c": parts[7],
                    }
                )
        return rows
    except Exception as exc:
        print(f"[monitor] nvidia-smi exception: {exc}", file=sys.stderr, flush=True)
        return []


# ── CSV header ────────────────────────────────────────────────────────────────

def _build_header(num_gpus):
    base = [
        "timestamp",
        "cpu_pct",
        "ram_used_mb",
        "ram_free_mb",
        "ram_available_mb",
        "ram_total_mb",
        "swap_used_mb",
        "swap_total_mb",
        "disk_read_mbps",
        "disk_write_mbps",
    ]
    for i in range(num_gpus):
        base += [
            f"gpu{i}_util_pct",
            f"gpu{i}_mem_util_pct",
            f"gpu{i}_mem_total_mb",
            f"gpu{i}_mem_used_mb",
            f"gpu{i}_mem_free_mb",
            f"gpu{i}_power_w",
            f"gpu{i}_temp_c",
        ]
    return base


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Resource monitor → CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval in seconds")
    parser.add_argument("--num-gpus", type=int, default=4, help="Expected number of GPUs")
    parser.add_argument(
        "--num-cpus", type=int,
        default=int(os.environ.get("SLURM_CPUS_ON_NODE", 0)) or None,
        help="Allocated CPU cores (default: SLURM_CPUS_ON_NODE). Used to express CPU%% relative to allocated cores.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    interval = args.interval
    num_gpus = args.num_gpus
    num_cpus = args.num_cpus  # may be None if unknown

    header = _build_header(num_gpus)

    write_header = not out_path.exists() or out_path.stat().st_size == 0

    # Detect CPU measurement method
    cgroup = _find_cgroup_cpuacct()
    if cgroup and num_cpus:
        cpu_method = "cgroup"
        print(f"[monitor] CPU method: cgroup ({cgroup[0]}: {cgroup[1]})  allocated_cores={num_cpus}", flush=True)
    else:
        cpu_method = "procstat"
        if not cgroup:
            print("[monitor] CPU method: /proc/stat (machine-wide — cgroup path not found)", flush=True)
        else:
            print("[monitor] CPU method: /proc/stat (machine-wide — --num-cpus not set)", flush=True)

    with open(out_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            csvfile.flush()

        # Seed initial readings for delta calculations
        if cpu_method == "cgroup":
            prev_cpu = _read_cgroup_cpu_ns(cgroup)
        else:
            prev_cpu = _read_proc_stat()
        prev_disk = _read_diskstats()
        prev_time = time.monotonic()
        time.sleep(interval)

        print(f"[monitor] Writing to {out_path}  (interval={interval}s, gpus={num_gpus})", flush=True)

        while True:
            try:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                curr_time = time.monotonic()
                elapsed = curr_time - prev_time

                # CPU
                if cpu_method == "cgroup":
                    curr_cpu = _read_cgroup_cpu_ns(cgroup)
                    cpu_pct = _cpu_pct_cgroup(prev_cpu, curr_cpu, elapsed, num_cpus)
                    prev_cpu = curr_cpu
                else:
                    curr_cpu = _read_proc_stat()
                    cpu_pct = _cpu_percent_proc(prev_cpu, curr_cpu)
                    prev_cpu = curr_cpu

                # RAM
                mem = _read_meminfo()
                ram_total_mb = round(mem.get("MemTotal", 0) / 1024, 1)
                ram_free_mb = round(mem.get("MemFree", 0) / 1024, 1)
                ram_available_mb = round(mem.get("MemAvailable", 0) / 1024, 1)
                ram_used_mb = round(ram_total_mb - ram_available_mb, 1)
                swap_total_mb = round(mem.get("SwapTotal", 0) / 1024, 1)
                swap_free_mb = round(mem.get("SwapFree", 0) / 1024, 1)
                swap_used_mb = round(swap_total_mb - swap_free_mb, 1)

                # Disk I/O
                curr_disk = _read_diskstats()
                disk_read_mbps, disk_write_mbps = _disk_delta_mbps(prev_disk, curr_disk, elapsed)
                prev_disk = curr_disk
                prev_time = curr_time

                # GPU
                gpu_rows = _gpu_stats(num_gpus)

                row = {
                    "timestamp": ts,
                    "cpu_pct": cpu_pct,
                    "ram_used_mb": ram_used_mb,
                    "ram_free_mb": ram_free_mb,
                    "ram_available_mb": ram_available_mb,
                    "ram_total_mb": ram_total_mb,
                    "swap_used_mb": swap_used_mb,
                    "swap_total_mb": swap_total_mb,
                    "disk_read_mbps": disk_read_mbps,
                    "disk_write_mbps": disk_write_mbps,
                }

                for i, gpu in enumerate(gpu_rows[:num_gpus]):
                    row[f"gpu{i}_util_pct"] = gpu.get("gpu_util_pct", "")
                    row[f"gpu{i}_mem_util_pct"] = gpu.get("gpu_mem_util_pct", "")
                    row[f"gpu{i}_mem_total_mb"] = gpu.get("gpu_mem_total_mb", "")
                    row[f"gpu{i}_mem_used_mb"] = gpu.get("gpu_mem_used_mb", "")
                    row[f"gpu{i}_mem_free_mb"] = gpu.get("gpu_mem_free_mb", "")
                    row[f"gpu{i}_power_w"] = gpu.get("gpu_power_w", "")
                    row[f"gpu{i}_temp_c"] = gpu.get("gpu_temp_c", "")

                writer.writerow(row)
                csvfile.flush()

            except Exception as exc:
                print(f"[monitor] error: {exc}", file=sys.stderr, flush=True)

            time.sleep(interval)


if __name__ == "__main__":
    main()
