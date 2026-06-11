#!/bin/bash
#SBATCH --job-name=skyrl-gsm8k
#SBATCH --account=i20240005g
#SBATCH --partition=dev-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=4:00:00
#SBATCH --output=/projects/I20240005/rnouaj/skyrl/logs/gsm8k_%j.log
#SBATCH --error=/projects/I20240005/rnouaj/skyrl/logs/gsm8k_%j_err.log

set -e

echo "========================================================"
echo "  SkyRL GSM8K GRPO training on 4x A100-40"
echo "  Model: Qwen/Qwen3-4B"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Time:   $(date)"
echo ""

PROJECT=/projects/I20240005/rnouaj/skyrl
SIF=$PROJECT/singularity/skyrl_fsdp.sif
DATA_DIR=/projects/I20240005/rnouaj/data/gsm8k
HF_CACHE=/projects/I20240005/rnouaj/cache/huggingface

echo "SIF:      $SIF"
echo "Data:     $DATA_DIR"
echo "HF cache: $HF_CACHE"
echo ""

mkdir -p /projects/I20240005/rnouaj/skyrl/logs

PIP_USERBASE=/projects/I20240005/rnouaj/pip_userbase
mkdir -p "$PIP_USERBASE"

# Ray needs a unique tmp dir per job to avoid cross-job collisions
RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

singularity exec --nv \
    --bind "$PROJECT":/skyrl \
    --bind "$DATA_DIR":/data/gsm8k \
    --bind "$HF_CACHE":/hf_cache \
    --bind "$PIP_USERBASE":/pip_userbase \
    --env HF_HOME=/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    --bind "$RAY_TMPDIR":/tmp/ray \
    --env PYTHONUSERBASE=/pip_userbase \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env TOKENIZERS_PARALLELISM=false \
    --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env OMP_NUM_THREADS=4 \
    --env RAY_TMPDIR=/tmp/ray \
    --env NCCL_DEBUG=WARN \
    "$SIF" \
    bash -c '
set -e
cd /skyrl

echo "=== Installing skyrl packages ==="
pip install --break-system-packages --no-deps -q -e . -e ./skyrl-gym

echo "=== Verifying GPU visibility ==="
python -c "import torch; print(f\"GPUs: {torch.cuda.device_count()}, CUDA: {torch.version.cuda}\")"

echo "=== Starting GRPO+GSM8K training ==="
python -m skyrl.train.entrypoints.main_base \
    "data.train_data=[\"/data/gsm8k/train.parquet\"]" \
    "data.val_data=[\"/data/gsm8k/validation.parquet\"]" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path=/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.ref_num_gpus_per_node=4 \
    generator.inference_engine.num_engines=4 \
    generator.inference_engine.tensor_parallel_size=1 \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=256 \
    trainer.policy_mini_batch_size=64 \
    trainer.critic_mini_batch_size=64 \
    trainer.micro_forward_batch_size_per_gpu=4 \
    trainer.micro_train_batch_size_per_gpu=4 \
    trainer.eval_batch_size=100 \
    trainer.eval_before_train=true \
    trainer.eval_interval=5 \
    trainer.ckpt_interval=10 \
    trainer.max_prompt_length=512 \
    generator.sampling_params.max_generate_length=2048 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.algorithm.use_kl_loss=true \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    environment.env_class=gsm8k \
    generator.n_samples_per_prompt=8 \
    generator.inference_engine.gpu_memory_utilization=0.4 \
    trainer.logger=console \
    trainer.project_name=skyrl-gsm8k \
    trainer.run_name=qwen3-4b-grpo-gsm8k_${SLURM_JOB_ID} \
    trainer.ckpt_path=/tmp/ckpts \
    trainer.export_path=/tmp/exports
'

echo ""
echo "Training complete at: $(date)"
