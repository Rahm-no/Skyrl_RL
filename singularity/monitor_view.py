#!/usr/bin/env python3
"""
Live dashboard for monitor_resources.py CSV output.

Usage:
    python monitor_view.py /path/to/monitor_<jobid>.csv
    python monitor_view.py /path/to/monitor_<jobid>.csv --interval 3
"""
import argparse
import csv
import os
import sys
import time

# ── ANSI helpers ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

BLACK  = "\033[30m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
CYAN   = "\033[36m"
WHITE  = "\033[37m"

BG_DARK  = "\033[48;5;235m"
BG_HEAD  = "\033[48;5;17m"
BG_GPU   = "\033[48;5;22m"
BG_SYS   = "\033[48;5;18m"

CLEAR_SCREEN = "\033[2J\033[H"


def clr(text, *codes):
    return "".join(codes) + str(text) + RESET


def bar(value, total, width=20, low_color=GREEN, high_color=RED, threshold=80):
    try:
        pct = float(value) / float(total) * 100 if float(total) > 0 else 0
    except (ValueError, ZeroDivisionError):
        pct = 0
    filled = int(pct / 100 * width)
    color = high_color if pct >= threshold else low_color
    b = clr("█" * filled, color) + clr("░" * (width - filled), DIM)
    return f"{b} {pct:5.1f}%"


def pct_bar(pct_str, width=20, threshold=80):
    try:
        pct = float(pct_str)
    except (ValueError, TypeError):
        return clr("N/A", DIM)
    color = RED if pct >= threshold else (YELLOW if pct >= 50 else GREEN)
    filled = int(pct / 100 * width)
    b = clr("█" * filled, color) + clr("░" * (width - filled), DIM)
    return f"{b} {pct:5.1f}%"


def fmt_mb(val, decimals=1):
    try:
        v = float(val)
        if v >= 1024:
            return f"{v/1024:.{decimals}f} GB"
        return f"{v:.{decimals}f} MB"
    except (ValueError, TypeError):
        return "N/A"


def fmt_val(val, unit="", na="N/A"):
    try:
        return f"{float(val):.1f}{unit}"
    except (ValueError, TypeError):
        return na


def divider(width=72, char="─", color=DIM):
    return clr(char * width, color)


def header_line(title, width=72):
    pad = width - len(title) - 4
    left = pad // 2
    right = pad - left
    return clr("┌" + "─" * left + "  " + BOLD + WHITE + title + RESET + DIM + "  " + "─" * right + "┐", DIM)


def section(label, width=72):
    inner = f"  {BOLD}{CYAN}{label}{RESET}"
    raw_len = len(f"  {label}")
    pad = width - raw_len - 2
    return clr("│", DIM) + inner + " " * pad + clr("│", DIM)


def row(label, value_str, label_w=18, total_w=72):
    inner = f"  {YELLOW}{label:<{label_w}}{RESET}  {value_str}"
    # strip ANSI for length calculation
    import re
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    visible = ansi_escape.sub('', inner)
    pad = max(0, total_w - len(visible) - 2)
    return clr("│", DIM) + inner + " " * pad + clr("│", DIM)


def footer(width=72):
    return clr("└" + "─" * (width - 2) + "┘", DIM)


# ── CSV reading ───────────────────────────────────────────────────────────────

def read_last_row(path):
    """Return (header, last_data_row_dict) or (None, None) if file not ready."""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            last = None
            for last in reader:
                pass
            if last is None:
                return None, None
            return reader.fieldnames, last
    except (FileNotFoundError, PermissionError):
        return None, None


def read_all_rows(path):
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except (FileNotFoundError, PermissionError):
        return []


# ── Sparkline ─────────────────────────────────────────────────────────────────

SPARK_CHARS = " ▁▂▃▄▅▆▇█"

def sparkline(values, width=20):
    if not values:
        return " " * width
    vals = values[-width:]
    try:
        vmin, vmax = min(float(v) for v in vals), max(float(v) for v in vals)
    except (ValueError, TypeError):
        return " " * width
    span = vmax - vmin or 1
    chars = [SPARK_CHARS[min(8, int((float(v) - vmin) / span * 8))] for v in vals]
    pad = width - len(chars)
    return " " * pad + "".join(chars)


# ── Dashboard render ──────────────────────────────────────────────────────────

def render(path, fieldnames, latest, history):
    W = 72
    num_gpus = sum(1 for k in (fieldnames or []) if k.endswith("_util_pct"))

    lines = []
    lines.append("")
    lines.append(header_line(f"Resource Monitor  ·  {latest.get('timestamp','?')}", W))

    # ── System ────────────────────────────────────────────────────────────────
    lines.append(section("SYSTEM", W))

    cpu = latest.get("cpu_pct", "?")
    cpu_hist = [r.get("cpu_pct", 0) for r in history]
    lines.append(row("CPU", f"{pct_bar(cpu, 16)}  {clr(fmt_val(cpu,'%'), BOLD)}   spark: {sparkline(cpu_hist, 18)}"))

    ram_used  = latest.get("ram_used_mb", "?")
    ram_total = latest.get("ram_total_mb", "?")
    ram_hist  = [r.get("ram_used_mb", 0) for r in history]
    lines.append(row("RAM", f"{bar(ram_used, ram_total, 16)}  {clr(fmt_mb(ram_used), BOLD)} / {fmt_mb(ram_total)}   spark: {sparkline(ram_hist, 10)}"))

    swap_used  = latest.get("swap_used_mb", "?")
    swap_total = latest.get("swap_total_mb", "?")
    lines.append(row("Swap", f"{clr(fmt_mb(swap_used), BOLD)} / {fmt_mb(swap_total)}"))

    dr = latest.get("disk_read_mbps", "?")
    dw = latest.get("disk_write_mbps", "?")
    dr_hist = [r.get("disk_read_mbps", 0) for r in history]
    dw_hist = [r.get("disk_write_mbps", 0) for r in history]
    lines.append(row("Disk Read",  f"{clr(fmt_val(dr,' MB/s'), BOLD)}   spark: {sparkline(dr_hist, 20)}"))
    lines.append(row("Disk Write", f"{clr(fmt_val(dw,' MB/s'), BOLD)}   spark: {sparkline(dw_hist, 20)}"))

    lines.append(clr("├" + "─" * (W - 2) + "┤", DIM))

    # ── GPUs ──────────────────────────────────────────────────────────────────
    lines.append(section("GPUs", W))

    for i in range(num_gpus):
        util      = latest.get(f"gpu{i}_util_pct", "")
        mem_used  = latest.get(f"gpu{i}_mem_used_mb", "")
        mem_total = latest.get(f"gpu{i}_mem_total_mb", "")
        power     = latest.get(f"gpu{i}_power_w", "")
        temp      = latest.get(f"gpu{i}_temp_c", "")
        mem_util  = latest.get(f"gpu{i}_mem_util_pct", "")
        util_hist = [r.get(f"gpu{i}_util_pct", 0) for r in history]

        lines.append(clr("├" + "─" * (W - 2) + "┤", DIM) if i > 0 else "")
        if i > 0:
            lines.pop(-2)  # remove blank before divider after first gpu

        lines.append(row(f"GPU {i}  Compute", f"{pct_bar(util, 16)}  {clr(fmt_val(util,'%'), BOLD)}   spark: {sparkline(util_hist, 18)}"))
        lines.append(row(f"GPU {i}  VRAM",    f"{bar(mem_used, mem_total, 16)}  {clr(fmt_mb(mem_used), BOLD)} / {fmt_mb(mem_total)}  (BW {fmt_val(mem_util,'%')})"))
        lines.append(row(f"GPU {i}  Power",   f"{clr(fmt_val(power,' W'), BOLD)}   Temp: {clr(fmt_val(temp,'°C'), BOLD)}"))

    lines.append(footer(W))

    rows_collected = len(history)
    lines.append(clr(f"  {path}  ·  {rows_collected} samples  ·  refreshing every {INTERVAL}s  (Ctrl-C to quit)", DIM))
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

INTERVAL = 5

def main():
    global INTERVAL
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to monitor CSV file")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    INTERVAL = args.interval

    print(f"Watching {args.csv} … (Ctrl-C to quit)", flush=True)

    try:
        while True:
            fieldnames, latest = read_last_row(args.csv)
            history = read_all_rows(args.csv)

            print(CLEAR_SCREEN, end="")

            if latest is None:
                print(f"\n  {clr('Waiting for data…', DIM)}  ({args.csv})\n")
            else:
                print(render(args.csv, fieldnames, latest, history))

            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
