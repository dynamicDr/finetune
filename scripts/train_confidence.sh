#!/usr/bin/env bash
set -euo pipefail

# 任务：批量训练 4 个模型的 Confidence Head（4组）
pids=()

# 第1组：4B + mcq
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=train_vsibench_confidence.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  train.max_samples=200 \
  +confidence.num_train_epochs=1 \
  +confidence.learning_rate=1e-3 \
  model=qwen3_4b_thinking \
  task_filter=mcq \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-4B-Thinking/mcq_frames4 \
  train.output_dir=outputs/vsibench_confidence/Qwen/Qwen3-VL-4B-Thinking/mcq_frames4 &
pids+=($!)

# 第2组：4B + numeric
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=train_vsibench_confidence.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  train.max_samples=200 \
  +confidence.num_train_epochs=1 \
  +confidence.learning_rate=1e-3 \
  model=qwen3_4b_thinking \
  task_filter=numeric \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-4B-Thinking/numeric_frames4 \
  train.output_dir=outputs/vsibench_confidence/Qwen/Qwen3-VL-4B-Thinking/numeric_frames4 &
pids+=($!)

# 第3组：8B + mcq
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=train_vsibench_confidence.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  train.max_samples=200 \
  +confidence.num_train_epochs=1 \
  +confidence.learning_rate=1e-3 \
  model=qwen3_8b_thinking \
  task_filter=mcq \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-8B-Thinking/mcq_frames4 \
  train.output_dir=outputs/vsibench_confidence/Qwen/Qwen3-VL-8B-Thinking/mcq_frames4 &
pids+=($!)

# 第4组：8B + numeric
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=train_vsibench_confidence.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  train.max_samples=200 \
  +confidence.num_train_epochs=1 \
  +confidence.learning_rate=1e-3 \
  model=qwen3_8b_thinking \
  task_filter=numeric \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-8B-Thinking/numeric_frames4 \
  train.output_dir=outputs/vsibench_confidence/Qwen/Qwen3-VL-8B-Thinking/numeric_frames4 &
pids+=($!)

for pid in "${pids[@]}"; do
  wait "$pid"
done

