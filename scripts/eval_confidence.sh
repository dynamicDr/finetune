#!/usr/bin/env bash
set -euo pipefail

# 任务：批量评测 4 个模型的 Confidence（高/低信心准确率，4组）
pids=()

# 第1组：4B + mcq
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=visbench/eval_vsibench_confidence.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  num_samples=all \
  +confidence.threshold=0.7 \
  model=qwen3_4b_thinking \
  task_filter=mcq \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-4B-Thinking/mcq_frames4 \
  +confidence.head_path=outputs/vsibench_confidence/Qwen/Qwen3-VL-4B-Thinking/mcq_frames4/confidence_head.pt &
pids+=($!)

# 第2组：4B + numeric
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=visbench/eval_vsibench_confidence.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  num_samples=all \
  +confidence.threshold=0.7 \
  model=qwen3_4b_thinking \
  task_filter=numeric \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-4B-Thinking/numeric_frames4 \
  +confidence.head_path=outputs/vsibench_confidence/Qwen/Qwen3-VL-4B-Thinking/numeric_frames4/confidence_head.pt &
pids+=($!)

# 第3组：8B + mcq
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=visbench/eval_vsibench_confidence.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  num_samples=all \
  +confidence.threshold=0.7 \
  model=qwen3_8b_thinking \
  task_filter=mcq \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-8B-Thinking/mcq_frames4 \
  +confidence.head_path=outputs/vsibench_confidence/Qwen/Qwen3-VL-8B-Thinking/mcq_frames4/confidence_head.pt &
pids+=($!)

# 第4组：8B + numeric
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=visbench/eval_vsibench_confidence.py \
  slurm=hgpu_batch \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  num_samples=all \
  +confidence.threshold=0.7 \
  model=qwen3_8b_thinking \
  task_filter=numeric \
  lora.enabled=true \
  lora.adapter_path=outputs/vsibench_train/Qwen/Qwen3-VL-8B-Thinking/numeric_frames4 \
  +confidence.head_path=outputs/vsibench_confidence/Qwen/Qwen3-VL-8B-Thinking/numeric_frames4/confidence_head.pt &
pids+=($!)

for pid in "${pids[@]}"; do
  wait "$pid"
done

