#!/bin/bash
# Submit from the project root: sbatch singularity/run_plots.sh <profiling_dir> <monitor_csv>
# Example:
#   sbatch singularity/run_plots.sh profiling_results/20260525_213703_job1327572 logs/monitor_1327572.csv

#SBATCH --job-name=skyrl-plots
#SBATCH --account=i20240005x
#SBATCH --partition=dev-x86
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=logs/plots_%j.log
#SBATCH --error=logs/plots_%j_err.log

set -e

# ── Configurable paths (override via environment variables) ──────────────────
PROJECT="${SKYRL_PROJECT:-/projects/I20240005/rnouaj/skyrl}"
SIF="${SKYRL_SIF:-$PROJECT/singularity/skyrl_fsdp.sif}"
# ─────────────────────────────────────────────────────────────────────────────

PROFILING_DIR="$1"
MONITOR_CSV="$2"

if [ -z "$PROFILING_DIR" ] || [ -z "$MONITOR_CSV" ]; then
    echo "Usage: sbatch singularity/run_plots.sh <profiling_dir> <monitor_csv>"
    echo "Example:"
    echo "  sbatch singularity/run_plots.sh profiling_results/20260525_213703_job1327572 logs/monitor_1327572.csv"
    exit 1
fi

echo "Profiling dir: $PROFILING_DIR"
echo "Monitor CSV:   $MONITOR_CSV"

singularity exec \
    --bind "$PROJECT":/skyrl \
    "$SIF" \
    bash -c '
set -e
cd /skyrl

pip install --break-system-packages -q matplotlib pandas

OUT="'"$PROFILING_DIR"'"
MON="'"$MONITOR_CSV"'"

echo "=== Generating plots ==="

python profiling/plots/plot_stage_times.py "$OUT/stage_times.jsonl"
python profiling/plots/plot_memory.py      "$OUT/memory.csv"
python profiling/plots/plot_weight_sync.py "$OUT/weight_sync.jsonl"
python profiling/plots/plot_rollout.py     "$OUT/rollout_stats.jsonl"
python profiling/plots/plot_gpu_util.py    "$OUT/gpu_util.csv"

echo "=== Generating monitor plot ==="
python profiling/plots/plot_monitor.py "$MON" --out-dir "$OUT"

echo ""
echo "All PNGs saved to: $OUT"
ls -lh "$OUT"/*.png
'
