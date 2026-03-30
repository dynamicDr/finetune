#!/usr/bin/env bash
set -euo pipefail

# 任务：评测微调后的模型在 0.2 测试集效果（4组）
pids=()

# 第1组：4B + mcq
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=test_vsibench.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  num_samples=all \
  task_filter=mcq \
  model=qwen3_4b_thinking \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-4B-Thinking/mcq_frames4 &
pids+=($!)

# 第2组：4B + numeric
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=test_vsibench.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  num_samples=all \
  task_filter=numeric \
  model=qwen3_4b_thinking \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-4B-Thinking/numeric_frames4 &
pids+=($!)

# 第3组：8B + mcq
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=test_vsibench.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  num_samples=all \
  task_filter=mcq \
  model=qwen3_8b_thinking \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-8B-Thinking/mcq_frames4 &
pids+=($!)

# 第4组：8B + numeric
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=test_vsibench.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  num_samples=all \
  task_filter=numeric \
  model=qwen3_8b_thinking \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-8B-Thinking/numeric_frames4 &
pids+=($!)

for pid in "${pids[@]}"; do
  wait "$pid"
done

