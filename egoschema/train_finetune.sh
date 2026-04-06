#!/usr/bin/env bash
set -euo pipefail

python egoschema/train_egoschema.py \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --video_dir ~/dataset/egoschema/videos \
  --output_dir outputs/egoschema_train/Qwen3-VL-4B-Instruct/base \
  --num_frames 4 \
  --train_ratio 0.8 \
  --seed 42
