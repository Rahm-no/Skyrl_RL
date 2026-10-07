#!/bin/bash
# GRPO + FSDP training on APPS (introductory) with Qwen2.5-7B-Instruct on 4x A100.
# Intended for pipeline stress-testing / bottleneck characterisation.
#
# Prerequisites:
#   uv run examples/train/apps/apps_dataset.py --output_dir ~/data/apps
#
# Run directly (outside Singularity):
#   bash singularity/run_apps.sh
#
# Or wrap inside the SIF:
#   singularity exec --nv singularity/skyrl_fsdp.sif bash singularity/run_apps.sh
#
# Override defaults via env vars, e.g.:
#   NUM_GPUS=4 MODEL_PATH=/local/Qwen2.5-7B-Instruct bash singularity/run_apps.sh

set -x

: "${DATA_DIR:=$HOME/data/apps}"
: "${NUM_GPUS:=4}"
: "${MODEL_PATH:=Qwen/Qwen2.5-7B-Instruct}"

# With colocate_all=true on 40 GB A100s, vLLM loads the full 7B model (~14 GB)
# on each GPU while FSDP shards across all 4 GPUs (~3.5 GB weights + ~10.5 GB
# grad/optim per GPU).  0.5 gpu_memory_utilization (~20 GB for vLLM) leaves
# ~20 GB for FSDP — sufficient for 7B at these micro-batch sizes.
# Reduce to 0.4 if OOM occurs during the FSDP forward/backward pass.

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
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
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=2048 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=lcb \
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.logger=console \
  trainer.project_name="apps-stress" \
  trainer.run_name="apps_7b_grpo" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.ckpt_path="$HOME/ckpts/apps_7b_ckpt" \
  $@
