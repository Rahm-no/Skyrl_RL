#!/bin/bash
# Submit from the project root: sbatch singularity/run_gsm8k_aligned_bs512.sh
# Override any path via env vars before submitting, e.g.:
#   SKYRL_HF_CACHE=/my/cache SKYRL_MODEL_PATH=/hf_cache/hub/... sbatch singularity/run_gsm8k_aligned_bs512.sh
#
# Aligned run with train_batch_size=512 (2× the baseline job1400905).
#
# Config matches run_gsm8k_aligned.sh exactly except:
#   train_batch_size 256 → 512  (policy_mini_batch_size kept at 64 → 8 accum steps)

#SBATCH --job-name=skyrl-gsm8k-bs512
#SBATCH --account=i20240005g
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/aligned_gsm8k_bs512_%j.log
#SBATCH --error=logs/aligned_gsm8k_bs512_%j_err.log

set -e

# ── Configurable paths (override via environment variables) ──────────────────
PROJECT="${SKYRL_PROJECT:-/projects/I20240005/rnouaj/skyrl}"
SIF="${SKYRL_SIF:-$PROJECT/singularity/skyrl_fsdp.sif}"
DATA_DIR="${SKYRL_DATA_DIR:-/projects/I20240005/rnouaj/data/gsm8k}"
HF_CACHE="${SKYRL_HF_CACHE:-/projects/I20240005/rnouaj/cache/huggingface}"
VLLM_CACHE="${SKYRL_VLLM_CACHE:-/projects/I20240005/rnouaj/cache/vllm}"
PIP_USERBASE="${SKYRL_PIP_USERBASE:-/projects/I20240005/rnouaj/pip_userbase}"
LOG_DIR="${SKYRL_LOG_DIR:-$PROJECT/logs}"
MODEL_PATH="${SKYRL_MODEL_PATH:-/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================================"
echo "  SkyRL GSM8K GRPO — ALIGNED — 4x A100-40 — bs=512"
echo "  Model: Qwen/Qwen3-4B"
echo "  batch=512  n=8  max_new_tokens=512  TP=1"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Time:   $(date)"
echo ""

MONITOR_CSV="${LOG_DIR}/monitor_aligned_bs512_${SLURM_JOB_ID}.csv"
MONITOR_SCRIPT="$PROJECT/singularity/monitor_resources.py"

mkdir -p "$LOG_DIR" "$PIP_USERBASE"

echo "SIF:      $SIF"
echo "Data:     $DATA_DIR"
echo "HF cache: $HF_CACHE"
echo "Monitor:  $MONITOR_CSV"
echo ""

python3 "$MONITOR_SCRIPT" \
    --output "$MONITOR_CSV" \
    --interval 5 \
    --num-gpus 4 \
    --num-cpus "${SLURM_CPUS_ON_NODE:-32}" \
    > "${LOG_DIR}/monitor_aligned_bs512_${SLURM_JOB_ID}.log" 2>&1 &
MONITOR_PID=$!

cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== Resource monitor started (PID=$MONITOR_PID) → $MONITOR_CSV ==="
echo ""

RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

singularity exec --nv \
    --bind "$PROJECT":/skyrl \
    --bind "$DATA_DIR":/data/gsm8k \
    --bind "$HF_CACHE":/hf_cache \
    --bind "$PIP_USERBASE":/pip_userbase \
    --bind "$RAY_TMPDIR":/tmp/ray \
    --bind "$VLLM_CACHE":/vllm_cache \
    --env HF_HOME=/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    --env PYTHONUSERBASE=/pip_userbase \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env TOKENIZERS_PARALLELISM=false \
    --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env OMP_NUM_THREADS=4 \
    --env RAY_TMPDIR=/tmp/ray \
    --env NCCL_DEBUG=WARN \
    --env VLLM_CACHE_ROOT=/vllm_cache \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env MODEL_PATH="$MODEL_PATH" \
    "$SIF" \
    bash -c '
set -e
cd /skyrl

echo "=== Installing skyrl packages ==="
pip install --break-system-packages --no-deps -q -e . -e ./skyrl-gym

echo "=== Verifying GPU visibility ==="
python -c "import torch; print(f\"GPUs: {torch.cuda.device_count()}, CUDA: {torch.version.cuda}\")"

echo "=== Starting aligned GSM8K+GRPO training (bs=512) ==="
python run_profiled.py \
    "data.train_data=[\"/data/gsm8k/train.parquet\"]" \
    "data.val_data=[\"/data/gsm8k/validation.parquet\"]" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path=$MODEL_PATH \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.ref_num_gpus_per_node=4 \
    generator.inference_engine.num_engines=4 \
    generator.inference_engine.tensor_parallel_size=1 \
    trainer.epochs=2 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=512 \
    trainer.policy_mini_batch_size=64 \
    trainer.critic_mini_batch_size=64 \
    trainer.micro_forward_batch_size_per_gpu=4 \
    trainer.micro_train_batch_size_per_gpu=4 \
    trainer.eval_batch_size=100 \
    trainer.eval_before_train=true \
    trainer.eval_interval=5 \
    trainer.ckpt_interval=0 \
    trainer.max_prompt_length=512 \
    generator.sampling_params.max_generate_length=512 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.algorithm.use_kl_loss=true \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    environment.env_class=gsm8k \
    generator.n_samples_per_prompt=8 \
    generator.inference_engine.enforce_eager=false \
    generator.inference_engine.gpu_memory_utilization=0.5 \
    trainer.logger=console \
    trainer.project_name=skyrl-gsm8k-aligned \
    trainer.run_name=qwen3-4b-grpo-gsm8k-aligned-bs512_${SLURM_JOB_ID} \
    trainer.ckpt_path=/tmp/ckpts \
    trainer.export_path=/tmp/exports
'

echo ""
echo "=== Aligned bs=512 run complete at: $(date) ==="
echo ""
echo "── Profiling results ───────────────────────────────────────────────────"
echo "  Stage timings + memory + GPU util saved to:"
echo "    $PROJECT/profiling_results/<timestamp>_job$SLURM_JOB_ID/"
echo ""
echo "── Resource monitor CSV ────────────────────────────────────────────────"
echo "  $MONITOR_CSV"
echo ""
echo "── To generate plots, run from the login node: ─────────────────────────"
echo "  OUT=\$(ls -dt $PROJECT/profiling_results/*_job${SLURM_JOB_ID} | head -1)"
echo "  python $PROJECT/profiling/plots/plot_stage_times.py \$OUT/stage_times.jsonl"
echo "  python $PROJECT/profiling/plots/plot_memory.py      \$OUT/memory.csv"
echo "  python $PROJECT/profiling/plots/plot_weight_sync.py \$OUT/weight_sync.jsonl"
echo "  python $PROJECT/profiling/plots/plot_rollout.py     \$OUT/rollout_stats.jsonl"
echo "  python $PROJECT/profiling/plots/plot_gpu_util.py    \$OUT/gpu_util.csv"
echo "  python $PROJECT/singularity/monitor_view.py         $MONITOR_CSV"
echo "────────────────────────────────────────────────────────────────────────"
