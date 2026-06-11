#!/bin/bash
# Run this from the login/CPU node (has internet access).
# Generates properly formatted GSM8K parquet files needed by SkyRL training.

set -e

PROJECT=/projects/I20240005/rnouaj/skyrl
SIF=$PROJECT/singularity/skyrl_fsdp.sif
DATA_DIR=/projects/I20240005/rnouaj/data/gsm8k
HF_CACHE=/projects/I20240005/rnouaj/cache/huggingface
PIP_USERBASE=/projects/I20240005/rnouaj/pip_userbase

mkdir -p "$DATA_DIR" "$PIP_USERBASE"

echo "Generating GSM8K parquet → $DATA_DIR"

singularity exec \
    --bind "$PROJECT":/skyrl \
    --bind "$DATA_DIR":/data/gsm8k \
    --bind "$HF_CACHE":/hf_cache \
    --bind "$PIP_USERBASE":/pip_userbase \
    --env HF_HOME=/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    --env PYTHONUSERBASE=/pip_userbase \
    --env PYTHONDONTWRITEBYTECODE=1 \
    "$SIF" \
    bash -c '
set -e
pip install --break-system-packages --no-deps -q -e /skyrl -e /skyrl/skyrl-gym
python /skyrl/examples/train/gsm8k/gsm8k_dataset.py --output_dir /data/gsm8k
'

echo "Done. Files written:"
ls -lh "$DATA_DIR"/*.parquet
