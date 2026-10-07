#!/bin/bash
# Run from the project root on the login/CPU node (has internet access).
# Generates properly formatted APPS parquet files needed by SkyRL training.
# Override paths via env vars, e.g.:
#   SKYRL_DATA_DIR=/my/data/apps bash singularity/prep_apps_data.sh
#   SKYRL_DATA_DIR=/my/data/apps_mix SKYRL_DIFFICULTY=all bash singularity/prep_apps_data.sh
#   SKYRL_DIFFICULTY=interview,competition bash singularity/prep_apps_data.sh

set -e

# ── Configurable paths (override via environment variables) ──────────────────
PROJECT="${SKYRL_PROJECT:-/projects/I20240005/rnouaj/skyrl}"
SIF="${SKYRL_SIF:-$PROJECT/singularity/skyrl_fsdp.sif}"
DATA_DIR="${SKYRL_DATA_DIR:-/projects/I20240005/rnouaj/data/apps}"
HF_CACHE="${SKYRL_HF_CACHE:-/projects/I20240005/rnouaj/cache/huggingface}"
PIP_USERBASE="${SKYRL_PIP_USERBASE:-/projects/I20240005/rnouaj/pip_userbase}"
DIFFICULTY="${SKYRL_DIFFICULTY:-introductory}"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$DATA_DIR" "$PIP_USERBASE"

echo "Generating APPS parquet → $DATA_DIR (difficulty=$DIFFICULTY)"

singularity exec \
    --bind "$PROJECT":/skyrl \
    --bind "$DATA_DIR":/data/apps \
    --bind "$HF_CACHE":/hf_cache \
    --bind "$PIP_USERBASE":/pip_userbase \
    --env HF_HOME=/hf_cache \
    --env HF_DATASETS_CACHE=/hf_cache/datasets \
    --env HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    --env PYTHONUSERBASE=/pip_userbase \
    --env PYTHONDONTWRITEBYTECODE=1 \
    "$SIF" \
    bash -c '
set -e
pip install --break-system-packages --no-deps -q -e /skyrl -e /skyrl/skyrl-gym
python /skyrl/examples/train/apps/apps_dataset.py --output_dir /data/apps --difficulty '"$DIFFICULTY"'
'

echo "Done. Files written:"
ls -lh "$DATA_DIR"/*.parquet
