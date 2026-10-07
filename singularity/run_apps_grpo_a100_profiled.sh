#!/bin/bash
# Submit from the project root: sbatch singularity/run_apps_grpo_a100_profiled.sh
# Override any path via env vars before submitting, e.g.:
#   SKYRL_DATA_DIR=/my/data/apps sbatch singularity/run_apps_grpo_a100_profiled.sh

#SBATCH --job-name=skyrl-apps-40-p
#SBATCH --account=i20240005g
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/apps_prof_%j.log
#SBATCH --error=logs/apps_prof_%j_err.log

set -e

# ── Configurable paths (override via environment variables) ──────────────────
PROJECT="${SKYRL_PROJECT:-/projects/I20240005/rnouaj/skyrl}"
SIF="${SKYRL_SIF:-$PROJECT/singularity/skyrl_fsdp.sif}"
DATA_DIR="${SKYRL_DATA_DIR:-/projects/I20240005/rnouaj/data/apps}"
HF_CACHE="${SKYRL_HF_CACHE:-/projects/I20240005/rnouaj/cache/huggingface}"
PIP_USERBASE="${SKYRL_PIP_USERBASE:-/projects/I20240005/rnouaj/pip_userbase}"
LOG_DIR="${SKYRL_LOG_DIR:-$PROJECT/logs}"
MODEL_PATH="${SKYRL_MODEL_PATH:-/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
CKPT_DIR="${SKYRL_CKPT_DIR:-/projects/I20240005/rnouaj/ckpts/apps}"
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================================"
echo "  SkyRL APPS GRPO PROFILING on 4x A100-40"
echo "  Model: Qwen/Qwen3-4B"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Time:   $(date)"
echo ""

MONITOR_CSV="${LOG_DIR}/monitor_apps_${SLURM_JOB_ID}.csv"
MONITOR_SCRIPT="$PROJECT/singularity/monitor_resources.py"

mkdir -p "$LOG_DIR" "$PIP_USERBASE" "$CKPT_DIR"

echo "SIF:      $SIF"
echo "Data:     $DATA_DIR"
echo "HF cache: $HF_CACHE"
echo "Monitor:  $MONITOR_CSV"
echo ""

# ── Resource monitor — runs on the host, outside Singularity ─────────────────
# Tracks: CPU % (cgroup-scoped), RAM, swap, disk read/write MB/s (local devices
# via /proc/diskstats), and per-GPU util/mem/power/temp every 5 s.
# Note: Lustre (the /projects filesystem) uses a network transport so its
# traffic appears as network I/O, not in /proc/diskstats.  The disk columns
# here cover local NVMe (/tmp, /var).  See monitor_view.py to visualise.

python3 "$MONITOR_SCRIPT" \
    --output "$MONITOR_CSV" \
    --interval 5 \
    --num-gpus 4 \
    --num-cpus "${SLURM_CPUS_ON_NODE:-32}" \
    > "${LOG_DIR}/monitor_apps_${SLURM_JOB_ID}.log" 2>&1 &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Resource monitor started (PID=$MONITOR_PID) → $MONITOR_CSV ==="
echo ""

# ── Ray tmp dir ───────────────────────────────────────────────────────────────
RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

# ── Training inside Singularity ───────────────────────────────────────────────
singularity exec --nv \
    --bind "$PROJECT":/skyrl \
    --bind "$DATA_DIR":/data/apps \
    --bind "$HF_CACHE":/hf_cache \
    --bind "$PIP_USERBASE":/pip_userbase \
    --bind "$RAY_TMPDIR":/tmp/ray \
    --bind "$CKPT_DIR":/ckpts \
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

echo "=== Starting profiled APPS+GRPO training ==="
echo "    Using run_profiled.py for GPU vs CPU rollout breakdown"
python run_profiled.py \
    "data.train_data=[\"/data/apps/train.parquet\"]" \
    "data.val_data=[\"/data/apps/validation.parquet\"]" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path=$MODEL_PATH \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.ref_num_gpus_per_node=4 \
    generator.inference_engine.num_engines=4 \
    generator.inference_engine.tensor_parallel_size=1 \
    trainer.epochs=5 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=256 \
    trainer.policy_mini_batch_size=64 \
    trainer.micro_forward_batch_size_per_gpu=8 \
    trainer.micro_train_batch_size_per_gpu=2 \
    trainer.eval_batch_size=256 \
    trainer.eval_before_train=true \
    trainer.eval_interval=5 \
    trainer.ckpt_interval=100 \
    trainer.max_prompt_length=1024 \
    generator.sampling_params.max_generate_length=2048 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.algorithm.use_kl_loss=true \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    environment.env_class=lcb \
    generator.n_samples_per_prompt=8 \
    generator.inference_engine.gpu_memory_utilization=0.4 \
    trainer.logger=console \
    trainer.project_name=skyrl-apps-profiling \
    trainer.run_name=qwen3-4b-grpo-apps-prof_${SLURM_JOB_ID} \
    trainer.ckpt_path=/ckpts \
    trainer.export_path=/ckpts/exports
'

echo ""
echo "=== Profiling run complete at: $(date) ==="
echo ""
echo "── Profiling results ───────────────────────────────────────────────────"
echo "  Stage timings + rollout profiling saved to:"
echo "    $PROJECT/profiling_results/<timestamp>_job$SLURM_JOB_ID/"
echo ""
echo "── Resource monitor CSV ────────────────────────────────────────────────"
echo "  $MONITOR_CSV"
echo ""
echo "── To generate plots, run from the login node: ─────────────────────────"
echo "  OUT=\$(ls -dt $PROJECT/profiling_results/*_job${SLURM_JOB_ID} | head -1)"
echo "  python $PROJECT/profiling/plots/plot_stage_times.py       \$OUT/stage_times.jsonl"
echo "  python $PROJECT/profiling/plots/plot_memory.py            \$OUT/memory.csv"
echo "  python $PROJECT/profiling/plots/plot_weight_sync.py       \$OUT/weight_sync.jsonl"
echo "  python $PROJECT/profiling/plots/plot_rollout.py           \$OUT/rollout_stats.jsonl"
echo "  python $PROJECT/profiling/plots/plot_gpu_util.py          \$OUT/gpu_util.csv"
echo "  python $PROJECT/profiling/plots/plot_rollout_profiling.py --input \$OUT/rollout_profiling.jsonl"
echo "  python $PROJECT/singularity/monitor_view.py $MONITOR_CSV"
echo "────────────────────────────────────────────────────────────────────────"
