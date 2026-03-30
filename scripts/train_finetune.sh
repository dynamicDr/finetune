#!/usr/bin/env bash
set -euo pipefail

# 在 0.8 训练集上微调（4组组合：4B/8B x mcq/numeric）
python run.py --multirun \
  hydra/launcher=submitit_slurm \
  script=train_vsibench.py \
  model=qwen3_4b_thinking,qwen3_8b_thinking \
  task_filter=mcq,numeric \
  num_frames=4 \
  train_ratio=0.8 \
  seed=42 \
  slurm=hgpu_batch

