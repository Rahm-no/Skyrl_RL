#!/bin/bash
# Launch the Search-R1 retrieval server inside Singularity.
# Run in a separate tmux pane BEFORE starting training.
#
# Usage:  bash singularity/launch_retriever.sh
#
# The server listens on http://0.0.0.0:8000/retrieve
# It needs ~6 GB GPU memory per GPU used by faiss + ~30 GB RAM for corpus.

set -e

PROJECT="${SKYRL_PROJECT:-/projects/I20240005/rnouaj/skyrl}"
SIF="${SKYRL_SIF:-$PROJECT/singularity/skyrl_fsdp.sif}"
DATA_DIR="${SKYRL_DATA_DIR:-/projects/I20240005/rnouaj/data/search_r1}"
HF_CACHE="${SKYRL_HF_CACHE:-/projects/I20240005/rnouaj/cache/huggingface}"
PIP_USERBASE="${SKYRL_PIP_USERBASE:-/projects/I20240005/rnouaj/pip_userbase_retriever}"
LOG_DIR="${SKYRL_LOG_DIR:-$PROJECT/logs}"

mkdir -p "$LOG_DIR" "$PIP_USERBASE"

echo "========================================================"
echo "  Search-R1 Retrieval Server (e5 dense + faiss-gpu)"
echo "========================================================"
echo "SIF:      $SIF"
echo "Data:     $DATA_DIR"
echo "Index:    $DATA_DIR/e5_Flat.index"
echo "Corpus:   $DATA_DIR/wiki-18.jsonl"
echo "HF cache: $HF_CACHE"
echo ""
echo "Installing faiss-gpu into user site-packages..."

singularity exec --nv \
    --bind "$PROJECT":/skyrl \
    --bind "$DATA_DIR":/data/search_r1 \
    --bind "$HF_CACHE":/hf_cache \
    --bind "$PIP_USERBASE":/pip_userbase \
    --env HF_HOME=/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    --env PYTHONUSERBASE=/pip_userbase \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$SIF" \
    bash -c '
set -e
cd /skyrl

pip install --break-system-packages --no-deps -q faiss-gpu 2>/dev/null || true
pip install --break-system-packages --no-deps -q -e . -e ./skyrl-gym 2>/dev/null || true

echo "=== Verifying faiss-gpu ==="
python -c "import faiss; print(f\"faiss {faiss.__version__}, GPUs: {faiss.get_num_gpus()}\")"

echo "=== Starting retrieval server on port 8000 ==="
python examples/train/search/retriever/retrieval_server.py \
    --index_path /data/search_r1/e5_Flat.index \
    --corpus_path /data/search_r1/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
'
