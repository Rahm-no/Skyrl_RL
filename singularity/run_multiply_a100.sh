#!/bin/bash
#SBATCH --job-name=skyrl-multiply
#SBATCH --account=i20240005g
#SBATCH --partition=dev-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=2:00:00
#SBATCH --output=/projects/I20240005/rnouaj/skyrl/logs/multiply_%j.log
#SBATCH --error=/projects/I20240005/rnouaj/skyrl/logs/multiply_%j_err.log

set -e

echo "========================================================"
echo "  SkyRL multiply training on 4x A100-40"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Time:   $(date)"
echo ""

PROJECT=/projects/I20240005/rnouaj/skyrl
SIF=$PROJECT/singularity/skyrl_fsdp.sif
DATA_DIR=/projects/I20240005/rnouaj/data/multiply_parquet
HF_CACHE=/projects/I20240005/rnouaj/cache/huggingface

echo "SIF:      $SIF"
echo "Data:     $DATA_DIR"
echo "HF cache: $HF_CACHE"
echo ""

# Ray needs a unique tmp dir per job to avoid cross-job collisions
RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

singularity exec --nv \
    --bind "$PROJECT":/skyrl \
    --bind "$DATA_DIR":/data/multiply \
    --bind "$HF_CACHE":/hf_cache \
    --env HF_HOME=/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    --bind "$RAY_TMPDIR":/tmp/ray \
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

echo "=== Starting training ==="
python -m examples.train.multiply.main_multiply \
    "data.train_data=[\"/data/multiply/train.parquet\"]" \
    "data.val_data=[\"/data/multiply/validation.parquet\"]" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path=/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.ref_num_gpus_per_node=4 \
    generator.inference_engine.num_engines=4 \
    generator.inference_engine.tensor_parallel_size=1 \
    trainer.epochs=20 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=256 \
    trainer.policy_mini_batch_size=64 \
    trainer.critic_mini_batch_size=64 \
    trainer.micro_forward_batch_size_per_gpu=16 \
    trainer.micro_train_batch_size_per_gpu=16 \
    trainer.eval_batch_size=100 \
    trainer.eval_before_train=true \
    trainer.eval_interval=5 \
    trainer.ckpt_interval=10 \
    trainer.max_prompt_length=512 \
    generator.sampling_params.max_generate_length=1024 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.algorithm.use_kl_loss=true \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    environment.env_class=multiply \
    generator.n_samples_per_prompt=5 \
    generator.inference_engine.gpu_memory_utilization=0.8 \
    trainer.logger=console \
    trainer.project_name=multiply \
    trainer.run_name=multiply_test_${SLURM_JOB_ID} \
    trainer.ckpt_path=/tmp/ckpts \
    trainer.export_path=/tmp/exports
'

echo ""
echo "Training complete at: $(date)"
