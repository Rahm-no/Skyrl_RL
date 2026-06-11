"""
Self-contained HTML report generator — stdlib only (Python 3.6+).

Reads all profiling output files and the monitor CSV, writes a single
report.html with inline SVG charts.

Usage:
    python3 profiling/plots/generate_report.py \
        profiling_results/<run>/ \
        logs/monitor_<jobid>.csv
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path


# ── Colour palette ─────────────────────────────────────────────────────────────

STAGE_COLORS = {
    "generate":               "#4C72B0",
    "stage_switch":           "#DD8452",
    "env_step":               "#55A868",
    "convert_to_training_input": "#aaaaaa",
    "fwd_logprobs":           "#C44E52",
    "advantage":              "#8172B2",
    "policy_train":           "#937860",
    "weight_sync":            "#DA8BC3",
    "eval":                   "#64B5CD",
}
GPU_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ── SVG primitives ─────────────────────────────────────────────────────────────

def _scale(val, vmin, vmax, out_min, out_max):
    if vmax == vmin:
        return (out_min + out_max) / 2
    return out_min + (val - vmin) / (vmax - vmin) * (out_max - out_min)


def svg_line_chart(series, width=700, height=220, title="", x_label="", y_label="",
                   y_min_zero=True, ref_lines=None):
    """series = list of (label, color, [(x, y), ...])"""
    ml, mr, mt, mb = 55, 20, 30, 40
    iw, ih = width - ml - mr, height - mt - mb

    all_x = [p[0] for _, _, pts in series for p in pts]
    all_y = [p[1] for _, _, pts in series for p in pts]
    if not all_x:
        return f'<svg width="{width}" height="{height}"><text x="10" y="20">No data</text></svg>'

    x0, x1 = min(all_x), max(all_x)
    y0 = 0 if y_min_zero else min(all_y) * 0.95
    y1 = max(all_y) * 1.1 if max(all_y) > 0 else 1

    def sx(x): return ml + _scale(x, x0, x1, 0, iw)
    def sy(y): return mt + ih - _scale(y, y0, y1, 0, ih)

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif;font-size:11px">',
        f'<text x="{width/2:.0f}" y="18" text-anchor="middle" font-size="13" font-weight="bold">{title}</text>',
        # axes
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ih}" stroke="#888" stroke-width="1"/>',
        f'<line x1="{ml}" y1="{mt+ih}" x2="{ml+iw}" y2="{mt+ih}" stroke="#888" stroke-width="1"/>',
        # axis labels
        f'<text x="{ml+iw/2:.0f}" y="{height-4}" text-anchor="middle" fill="#555">{x_label}</text>',
        f'<text x="12" y="{mt+ih/2:.0f}" text-anchor="middle" fill="#555" '
        f'transform="rotate(-90,12,{mt+ih/2:.0f})">{y_label}</text>',
    ]

    # y grid + tick labels
    n_ticks = 5
    for i in range(n_ticks + 1):
        yv = y0 + (y1 - y0) * i / n_ticks
        yp = sy(yv)
        parts.append(f'<line x1="{ml}" y1="{yp:.1f}" x2="{ml+iw}" y2="{yp:.1f}" '
                     f'stroke="#eee" stroke-width="1"/>')
        label = f"{yv:.0f}" if yv >= 10 else f"{yv:.1f}"
        parts.append(f'<text x="{ml-4}" y="{yp+4:.1f}" text-anchor="end" fill="#666">{label}</text>')

    # reference lines
    if ref_lines:
        for rv, rl, rc in ref_lines:
            if y0 <= rv <= y1:
                yp = sy(rv)
                parts.append(f'<line x1="{ml}" y1="{yp:.1f}" x2="{ml+iw}" y2="{yp:.1f}" '
                              f'stroke="{rc}" stroke-width="1" stroke-dasharray="4,3"/>')
                parts.append(f'<text x="{ml+iw-2}" y="{yp-3:.1f}" text-anchor="end" '
                              f'fill="{rc}" font-size="10">{rl}</text>')

    # series lines
    for label, color, pts in series:
        if not pts:
            continue
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                     f'stroke-width="1.5" opacity="0.85"/>')

    # legend
    lx = ml + 6
    for i, (label, color, _) in enumerate(series):
        ly = mt + 6 + i * 16
        parts.append(f'<rect x="{lx}" y="{ly}" width="12" height="10" fill="{color}" opacity="0.8"/>')
        parts.append(f'<text x="{lx+16}" y="{ly+9}" fill="#333">{label}</text>')

    parts.append('</svg>')
    return '\n'.join(parts)


def svg_bar_chart(categories, values, colors, width=700, height=220, title="",
                  x_label="", y_label="", annotations=None):
    ml, mr, mt, mb = 55, 20, 30, 60
    iw, ih = width - ml - mr, height - mt - mb
    n = len(values)
    if n == 0:
        return f'<svg width="{width}" height="{height}"><text x="10" y="20">No data</text></svg>'

    y1 = max(values) * 1.15 if max(values) > 0 else 1
    bw = iw / n * 0.7
    gap = iw / n

    def sy(y): return mt + ih - _scale(y, 0, y1, 0, ih)

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif;font-size:11px">',
        f'<text x="{width/2:.0f}" y="18" text-anchor="middle" font-size="13" font-weight="bold">{title}</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ih}" stroke="#888" stroke-width="1"/>',
        f'<line x1="{ml}" y1="{mt+ih}" x2="{ml+iw}" y2="{mt+ih}" stroke="#888" stroke-width="1"/>',
        f'<text x="{ml+iw/2:.0f}" y="{height-4}" text-anchor="middle" fill="#555">{x_label}</text>',
        f'<text x="12" y="{mt+ih/2:.0f}" text-anchor="middle" fill="#555" '
        f'transform="rotate(-90,12,{mt+ih/2:.0f})">{y_label}</text>',
    ]

    for i in range(5):
        yv = y1 * i / 4
        yp = sy(yv)
        parts.append(f'<line x1="{ml}" y1="{yp:.1f}" x2="{ml+iw}" y2="{yp:.1f}" '
                     f'stroke="#eee" stroke-width="1"/>')
        parts.append(f'<text x="{ml-4}" y="{yp+4:.1f}" text-anchor="end" fill="#666">'
                     f'{yv:.1f}</text>')

    for i, (cat, val, col) in enumerate(zip(categories, values, colors)):
        bx = ml + gap * i + (gap - bw) / 2
        by = sy(val)
        bh = mt + ih - by
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                     f'fill="{col}" opacity="0.85"/>')
        ann = annotations[i] if annotations else f"{val:.1f}"
        parts.append(f'<text x="{bx+bw/2:.1f}" y="{by-3:.1f}" text-anchor="middle" '
                     f'fill="#333" font-size="10">{ann}</text>')
        # x tick
        cx = bx + bw / 2
        parts.append(f'<text x="{cx:.1f}" y="{mt+ih+14}" text-anchor="middle" '
                     f'fill="#555" font-size="10" transform="rotate(-30,{cx:.1f},{mt+ih+14})">'
                     f'{cat}</text>')

    parts.append('</svg>')
    return '\n'.join(parts)


def svg_stacked_bar(steps, stage_keys, stage_data, width=700, height=260, title=""):
    """stage_data[stage] = [val_per_step]"""
    ml, mr, mt, mb = 55, 140, 30, 40
    iw, ih = width - ml - mr, height - mt - mb
    n = len(steps)
    if n == 0:
        return f'<svg width="{width}" height="{height}"><text x="10" y="20">No data</text></svg>'

    totals = [sum(stage_data[s][i] for s in stage_keys if i < len(stage_data[s]))
              for i in range(n)]
    y1 = max(totals) * 1.1 if totals else 1
    bw = iw / n * 0.75
    gap = iw / n

    def sy(y): return mt + ih - _scale(y, 0, y1, 0, ih)

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif;font-size:11px">',
        f'<text x="{(width-mr)/2:.0f}" y="18" text-anchor="middle" font-size="13" '
        f'font-weight="bold">{title}</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ih}" stroke="#888" stroke-width="1"/>',
        f'<line x1="{ml}" y1="{mt+ih}" x2="{ml+iw}" y2="{mt+ih}" stroke="#888" stroke-width="1"/>',
        f'<text x="12" y="{mt+ih/2:.0f}" text-anchor="middle" fill="#555" '
        f'transform="rotate(-90,12,{mt+ih/2:.0f})">Seconds</text>',
    ]

    for i in range(5):
        yv = y1 * i / 4
        yp = sy(yv)
        parts.append(f'<line x1="{ml}" y1="{yp:.1f}" x2="{ml+iw}" y2="{yp:.1f}" '
                     f'stroke="#eee" stroke-width="1"/>')
        parts.append(f'<text x="{ml-4}" y="{yp+4:.1f}" text-anchor="end" fill="#666">'
                     f'{yv:.0f}s</text>')

    for i, step in enumerate(steps):
        bx = ml + gap * i + (gap - bw) / 2
        base = 0
        for stage in stage_keys:
            vals = stage_data[stage]
            val = vals[i] if i < len(vals) else 0
            if val <= 0:
                base += val
                continue
            by = sy(base + val)
            bh = sy(base) - by
            col = STAGE_COLORS.get(stage, "#cccccc")
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                         f'fill="{col}" opacity="0.9"/>')
            base += val
        cx = bx + bw / 2
        parts.append(f'<text x="{cx:.1f}" y="{mt+ih+14}" text-anchor="middle" '
                     f'fill="#555" font-size="10">{step}</text>')

    # legend (right side)
    lx = ml + iw + 10
    for j, stage in enumerate(stage_keys):
        ly = mt + j * 18
        col = STAGE_COLORS.get(stage, "#cccccc")
        parts.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{col}" opacity="0.9"/>')
        parts.append(f'<text x="{lx+16}" y="{ly+11}" fill="#333" font-size="11">{stage}</text>')

    parts.append('</svg>')
    return '\n'.join(parts)


def svg_histogram(values, bins=20, width=600, height=220, title="", x_label="", color="#4C72B0",
                  vlines=None):
    if not values:
        return f'<svg width="{width}" height="{height}"><text x="10" y="20">No data</text></svg>'
    ml, mr, mt, mb = 55, 20, 30, 40
    iw, ih = width - ml - mr, height - mt - mb

    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        vmax = vmin + 1
    bsize = (vmax - vmin) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - vmin) / bsize), bins - 1)
        counts[idx] += 1
    y1 = max(counts) * 1.15

    def sx(x): return ml + _scale(x, vmin, vmax, 0, iw)
    def sy(y): return mt + ih - _scale(y, 0, y1, 0, ih)

    bw = iw / bins

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif;font-size:11px">',
        f'<text x="{width/2:.0f}" y="18" text-anchor="middle" font-size="13" font-weight="bold">{title}</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ih}" stroke="#888" stroke-width="1"/>',
        f'<line x1="{ml}" y1="{mt+ih}" x2="{ml+iw}" y2="{mt+ih}" stroke="#888" stroke-width="1"/>',
        f'<text x="{ml+iw/2:.0f}" y="{height-4}" text-anchor="middle" fill="#555">{x_label}</text>',
        f'<text x="12" y="{mt+ih/2:.0f}" text-anchor="middle" fill="#555" '
        f'transform="rotate(-90,12,{mt+ih/2:.0f})">Count</text>',
    ]

    for i, cnt in enumerate(counts):
        bx = ml + i * bw
        by = sy(cnt)
        bh = mt + ih - by
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw-1:.1f}" height="{bh:.1f}" '
                     f'fill="{color}" opacity="0.8"/>')

    # x ticks
    for i in range(6):
        xv = vmin + (vmax - vmin) * i / 5
        xp = sx(xv)
        parts.append(f'<text x="{xp:.1f}" y="{mt+ih+14}" text-anchor="middle" fill="#555">'
                     f'{xv:.0f}</text>')

    # y ticks
    for i in range(4):
        yv = y1 * i / 3
        yp = sy(yv)
        parts.append(f'<text x="{ml-4}" y="{yp+4:.1f}" text-anchor="end" fill="#666">'
                     f'{yv:.0f}</text>')

    if vlines:
        for xv, label, col in vlines:
            if vmin <= xv <= vmax:
                xp = sx(xv)
                parts.append(f'<line x1="{xp:.1f}" y1="{mt}" x2="{xp:.1f}" y2="{mt+ih}" '
                              f'stroke="{col}" stroke-width="1.5" stroke-dasharray="5,3"/>')
                parts.append(f'<text x="{xp+3:.1f}" y="{mt+12}" fill="{col}" font-size="10">'
                              f'{label}</text>')

    parts.append('</svg>')
    return '\n'.join(parts)


def svg_pie(labels, values, colors, width=340, height=260, title=""):
    cx, cy, r = width // 2, height // 2 + 10, min(width, height) // 2 - 30
    total = sum(values)
    if total == 0:
        return ''

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif;font-size:11px">',
        f'<text x="{width/2:.0f}" y="18" text-anchor="middle" font-size="13" '
        f'font-weight="bold">{title}</text>',
    ]

    angle = -math.pi / 2
    for label, val, col in zip(labels, values, colors):
        sweep = 2 * math.pi * val / total
        x1 = cx + r * math.cos(angle)
        y1 = cy + r * math.sin(angle)
        x2 = cx + r * math.cos(angle + sweep)
        y2 = cy + r * math.sin(angle + sweep)
        large = 1 if sweep > math.pi else 0
        pct = val / total * 100
        parts.append(
            f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} '
            f'A{r},{r} 0 {large},1 {x2:.1f},{y2:.1f} Z" '
            f'fill="{col}" opacity="0.85" stroke="white" stroke-width="1"/>'
        )
        if pct > 4:
            mid_a = angle + sweep / 2
            tx = cx + r * 0.65 * math.cos(mid_a)
            ty = cy + r * 0.65 * math.sin(mid_a)
            parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
                         f'fill="white" font-size="10" font-weight="bold">{pct:.0f}%</text>')
        angle += sweep

    parts.append('</svg>')
    return '\n'.join(parts)


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── Section builders ───────────────────────────────────────────────────────────

def section(title, content):
    return f'''
<div class="section">
  <h2>{title}</h2>
  {content}
</div>'''


def stat_box(label, value, sub=""):
    return (f'<div class="stat"><div class="stat-val">{value}</div>'
            f'<div class="stat-label">{label}</div>'
            f'{"<div class=stat-sub>" + sub + "</div>" if sub else ""}</div>')


def build_stage_times_section(records):
    if not records:
        return "<p>No stage_times.jsonl data.</p>"

    stage_keys = ["generate", "stage_switch", "env_step", "convert_to_training_input",
                  "fwd_logprobs", "advantage", "policy_train", "weight_sync"]
    steps = [str(r.get("global_step", i)) for i, r in enumerate(records)]
    stage_data = {s: [safe_float(r.get(s, 0)) for r in records] for s in stage_keys}
    totals = [safe_float(r.get("step_total", 0)) for r in records]

    mean_total = statistics.mean(totals) if totals else 0

    # Stats row
    stats_html = '<div class="stats-row">'
    for s in stage_keys:
        vals = stage_data[s]
        m = statistics.mean(vals) if vals else 0
        pct = m / mean_total * 100 if mean_total > 0 else 0
        col = STAGE_COLORS.get(s, "#aaa")
        stats_html += f'<div class="stat" style="border-left:4px solid {col}">'
        stats_html += f'<div class="stat-val">{m:.1f}s</div>'
        stats_html += f'<div class="stat-label">{s}</div>'
        stats_html += f'<div class="stat-sub">{pct:.1f}% of step</div></div>'
    stats_html += f'<div class="stat"><div class="stat-val">{mean_total:.1f}s</div>'
    stats_html += f'<div class="stat-label">step total</div></div>'
    stats_html += '</div>'

    stacked = svg_stacked_bar(steps, stage_keys, stage_data,
                              width=760, height=280,
                              title="Stage Time per Step (stacked)")

    pie_vals = [statistics.mean(stage_data[s]) for s in stage_keys]
    pie_colors = [STAGE_COLORS.get(s, "#ccc") for s in stage_keys]
    pie = svg_pie(stage_keys, pie_vals, pie_colors, width=360, height=280,
                  title="Mean Stage Distribution")

    return stats_html + f'<div style="display:flex;gap:16px;flex-wrap:wrap">{stacked}{pie}</div>'


def build_weight_sync_section(records):
    if not records:
        return "<p>No weight_sync.jsonl data.</p>"

    totals_ms = [safe_float(r.get("total_s", 0)) * 1000 for r in records]
    bws = [safe_float(r.get("achieved_bw_gbs", 0)) for r in records]
    peak = safe_float(records[0].get("theoretical_peak_gbs", 130))
    model_gb = safe_float(records[0].get("model_size_gb", 3.08))
    mean_total = statistics.mean(totals_ms)
    mean_bw = statistics.mean(bws)

    stats_html = '<div class="stats-row">'
    stats_html += stat_box("Mean sync time", f"{mean_total:.0f} ms",
                           f"P95: {sorted(totals_ms)[int(len(totals_ms)*0.95)]:.0f} ms")
    stats_html += stat_box("Mean BW", f"{mean_bw:.1f} GB/s",
                           f"{mean_bw/peak*100:.0f}% of NVLink peak ({peak:.0f} GB/s)")
    stats_html += stat_box("Model size", f"{model_gb:.2f} GB", "BF16")
    stats_html += stat_box("Syncs recorded", str(len(records)))
    stats_html += '</div>'

    bw_chart = svg_line_chart(
        [("Achieved BW (GB/s)", "#4C72B0", list(enumerate(bws)))],
        width=520, height=200,
        title="Weight Sync Bandwidth per Sync",
        x_label="Sync index", y_label="GB/s",
        ref_lines=[(peak, f"NVLink peak {peak:.0f}", "red")],
    )

    gather = safe_float(records[0].get("gather_s_est", 0)) * 1000
    bcast  = safe_float(records[0].get("broadcast_s_est", 0)) * 1000
    load   = statistics.mean(safe_float(r.get("load_s_est", 0)) * 1000 for r in records)
    phase_chart = svg_bar_chart(
        ["Gather\n(FSDP all-gather)", "Broadcast\n(NCCL→vLLM)", "Load\n(vLLM install)"],
        [gather, bcast, load],
        ["#4C72B0", "#DD8452", "#55A868"],
        width=380, height=200,
        title="Mean Phase Breakdown (estimated)",
        y_label="ms",
        annotations=[f"{gather:.0f}ms", f"{bcast:.0f}ms", f"{load:.0f}ms"],
    )

    return stats_html + f'<div style="display:flex;gap:16px;flex-wrap:wrap">{bw_chart}{phase_chart}</div>'


def build_rollout_section(records):
    if not records:
        return "<p>No rollout_stats.jsonl data.</p>"

    tokens = [safe_float(r.get("total_response_tokens", 0)) for r in records]
    turns  = [safe_float(r.get("num_turns", 1)) for r in records]
    wall_s = [safe_float(r.get("wall_s", 0)) for r in records]
    solved = [r.get("solve_turn", -1) != -1 for r in records]

    p50 = sorted(tokens)[len(tokens)//2]
    p99 = sorted(tokens)[int(len(tokens)*0.99)]
    ratio = p99 / p50 if p50 > 0 else 0
    solve_rate = sum(solved) / len(solved) * 100 if solved else 0

    stats_html = '<div class="stats-row">'
    stats_html += stat_box("Trajectories", str(len(records)))
    stats_html += stat_box("Solve rate", f"{solve_rate:.1f}%")
    stats_html += stat_box("Token P50", f"{p50:.0f}")
    stats_html += stat_box("Token P99", f"{p99:.0f}", f"P99/P50 = {ratio:.1f}x")
    stats_html += stat_box("Avg wall time", f"{statistics.mean(wall_s):.2f}s")
    stats_html += '</div>'

    tok_hist = svg_histogram(tokens, bins=25, width=480, height=220,
                             title="Token Distribution per Trajectory",
                             x_label="Total response tokens", color="#4C72B0",
                             vlines=[(p50, f"P50={p50:.0f}", "orange"),
                                     (p99, f"P99={p99:.0f}", "red")])

    max_turn = int(max(turns)) if turns else 5
    turn_counts = [int(t) for t in turns]
    turn_bins = list(range(1, max_turn + 1))
    turn_vals = [turn_counts.count(t) for t in turn_bins]
    turn_chart = svg_bar_chart(
        [str(t) for t in turn_bins], turn_vals,
        ["#55A868"] * len(turn_bins),
        width=320, height=220,
        title="Turn Count Distribution",
        x_label="Turns used", y_label="Count",
    )

    return stats_html + f'<div style="display:flex;gap:16px;flex-wrap:wrap">{tok_hist}{turn_chart}</div>'


def build_monitor_section(rows):
    if not rows:
        return "<p>No monitor CSV data.</p>"

    # Parse timestamps to elapsed seconds
    import time as _time
    t0 = None
    elapsed = []
    cpu = []
    ram_used = []
    ram_total_val = safe_float(rows[0].get("ram_total_mb", 0)) / 1024

    for row in rows:
        ts_str = row.get("timestamp", "")
        try:
            import datetime
            ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            if t0 is None:
                t0 = ts
            elapsed.append((ts - t0).total_seconds())
        except Exception:
            elapsed.append(len(elapsed) * 5)
        cpu.append(safe_float(row.get("cpu_pct", 0)))
        ram_used.append(safe_float(row.get("ram_used_mb", 0)) / 1024)

    disk_read  = [safe_float(r.get("disk_read_mbps", 0)) for r in rows]
    disk_write = [safe_float(r.get("disk_write_mbps", 0)) for r in rows]

    cpu_chart = svg_line_chart(
        [("CPU %", "#4C72B0", list(zip(elapsed, cpu)))],
        width=700, height=180, title="CPU Utilization (job-scoped, 32 cores)",
        x_label="Elapsed (s)", y_label="CPU %",
        ref_lines=[(100, "100%", "red")],
    )
    ram_chart = svg_line_chart(
        [("RAM used (GiB)", "#55A868", list(zip(elapsed, ram_used)))],
        width=700, height=180, title=f"RAM Usage (total {ram_total_val:.0f} GiB)",
        x_label="Elapsed (s)", y_label="GiB",
        ref_lines=[(ram_total_val, f"Total {ram_total_val:.0f}GiB", "red")],
    )
    disk_chart = svg_line_chart(
        [("Read MB/s", "#4C72B0", list(zip(elapsed, disk_read))),
         ("Write MB/s", "#DD8452", list(zip(elapsed, disk_write)))],
        width=700, height=180, title="Disk I/O",
        x_label="Elapsed (s)", y_label="MB/s",
    )

    # GPU charts — up to 4 GPUs
    gpu_util_series, vram_series, power_series = [], [], []
    for g in range(4):
        util = [safe_float(r.get(f"gpu{g}_util_pct", 0)) for r in rows]
        vram = [safe_float(r.get(f"gpu{g}_mem_used_mb", 0)) / 1024 for r in rows]
        pwr  = [safe_float(r.get(f"gpu{g}_power_w", 0)) for r in rows]
        if max(util) > 0 or max(vram) > 0:
            gpu_util_series.append((f"GPU{g}", GPU_COLORS[g], list(zip(elapsed, util))))
            vram_total = safe_float(rows[0].get(f"gpu{g}_mem_total_mb", 40960)) / 1024
            vram_series.append((f"GPU{g}", GPU_COLORS[g], list(zip(elapsed, vram))))
            power_series.append((f"GPU{g} power", GPU_COLORS[g], list(zip(elapsed, pwr))))

    vram_total_gib = safe_float(rows[0].get("gpu0_mem_total_mb", 40960)) / 1024
    gpu_util_chart = svg_line_chart(gpu_util_series, width=700, height=200,
                                    title="GPU SM Utilization (%)",
                                    x_label="Elapsed (s)", y_label="%",
                                    ref_lines=[(80, "80%", "red"), (40, "40%", "orange")])
    vram_chart = svg_line_chart(vram_series, width=700, height=200,
                                title=f"VRAM Usage (GiB, total {vram_total_gib:.0f} GiB/GPU)",
                                x_label="Elapsed (s)", y_label="GiB",
                                ref_lines=[(vram_total_gib, f"{vram_total_gib:.0f}GiB", "red")])
    power_chart = svg_line_chart(power_series, width=700, height=200,
                                 title="GPU Power Draw (W)",
                                 x_label="Elapsed (s)", y_label="W")

    # Summary stats
    stats_html = '<div class="stats-row">'
    stats_html += stat_box("Duration", f"{elapsed[-1]:.0f}s" if elapsed else "?")
    stats_html += stat_box("Peak CPU", f"{max(cpu):.1f}%")
    stats_html += stat_box("Peak RAM", f"{max(ram_used):.1f} GiB")
    for g in range(4):
        peak_vram = max(safe_float(r.get(f"gpu{g}_mem_used_mb", 0)) for r in rows) / 1024
        total_vram = safe_float(rows[0].get(f"gpu{g}_mem_total_mb", 40960)) / 1024
        if peak_vram > 0.1:
            stats_html += stat_box(f"GPU{g} peak VRAM", f"{peak_vram:.1f} GiB",
                                   f"headroom {total_vram - peak_vram:.1f} GiB")
    stats_html += '</div>'

    charts = "\n".join([cpu_chart, ram_chart, disk_chart, gpu_util_chart, vram_chart, power_chart])
    return stats_html + charts


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profiling_dir", help="profiling_results/<run>/ directory")
    parser.add_argument("monitor_csv", help="logs/monitor_<jobid>.csv")
    args = parser.parse_args()

    out_dir = Path(args.profiling_dir)
    out_html = out_dir / "report.html"

    print(f"Reading from: {out_dir}")

    stage_recs   = load_jsonl(out_dir / "stage_times.jsonl")   if (out_dir / "stage_times.jsonl").exists()   else []
    ws_recs      = load_jsonl(out_dir / "weight_sync.jsonl")   if (out_dir / "weight_sync.jsonl").exists()   else []
    rollout_recs = load_jsonl(out_dir / "rollout_stats.jsonl") if (out_dir / "rollout_stats.jsonl").exists() else []
    monitor_rows = load_csv(args.monitor_csv)                  if Path(args.monitor_csv).exists()            else []

    print(f"  stage_times:   {len(stage_recs)} records")
    print(f"  weight_sync:   {len(ws_recs)} records")
    print(f"  rollout_stats: {len(rollout_recs)} records")
    print(f"  monitor CSV:   {len(monitor_rows)} rows")

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SkyRL Profiling Report</title>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f8f9fa; color: #333; }}
  h1 {{ color: #222; border-bottom: 2px solid #4C72B0; padding-bottom: 8px; }}
  h2 {{ color: #444; margin-top: 0; }}
  .section {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
              box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .stats-row {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }}
  .stat {{ background: #f0f4ff; border-radius: 6px; padding: 10px 16px; min-width: 120px;
           border-left: 4px solid #4C72B0; }}
  .stat-val {{ font-size: 1.4em; font-weight: bold; color: #222; }}
  .stat-label {{ font-size: 0.85em; color: #666; margin-top: 2px; }}
  .stat-sub {{ font-size: 0.78em; color: #999; margin-top: 2px; }}
  svg {{ display: block; max-width: 100%; }}
</style>
</head>
<body>
<h1>SkyRL Profiling Report — {out_dir.name}</h1>
{section("1. Training Stage Timing", build_stage_times_section(stage_recs))}
{section("2. System Resources (monitor CSV)", build_monitor_section(monitor_rows))}
{section("3. Weight Sync Analysis", build_weight_sync_section(ws_recs))}
{section("4. Rollout Statistics", build_rollout_section(rollout_recs))}
</body>
</html>'''

    with open(out_html, "w") as f:
        f.write(html)
    print(f"\nSaved: {out_html}")


if __name__ == "__main__":
    main()
