#!/usr/bin/env bash
set -euo pipefail

python egoschema/eval_egoschema.py \
  --use_lora \
  --base_model Qwen/Qwen3-VL-4B-Instruct \
  --model_path outputs/egoschema_train/Qwen3-VL-4B-Instruct/base \
  --video_dir ~/dataset/egoschema/videos \
  --num_frames 4 \
  --num_samples all \
  --train_ratio 0.8 \
  --seed 42
