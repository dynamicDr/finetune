#!/bin/bash

# 测试 Qwen3 不同模型在 VSI-Bench 上的表现
# 会依次测试：
# - Qwen3-VL-4B-Instruct
# - Qwen3-VL-8B-Instruct
# - Qwen3-VL-30B-A3B-Instruct

set -e

VIDEO_DIR="$HOME/dataset/vsi_bench"
SEED=42

# 三个要测试的模型（Hugging Face 名称后缀）
MODELS=(
  "Qwen3-VL-4B-Instruct"
  "Qwen3-VL-8B-Instruct"
  "Qwen3-VL-30B-A3B-Instruct"
)

# 不同任务下使用的帧数
FRAME_COUNTS_MCQ=(4 8)
FRAME_COUNTS_NUM=(4 8)

# 计算总实验次数（3 个模型 × (mcq 帧数 + numeric 帧数)）
TOTAL_EXPERIMENTS=$(( ${#MODELS[@]} * ( ${#FRAME_COUNTS_MCQ[@]} + ${#FRAME_COUNTS_NUM[@]} ) ))
CURRENT=0

resolve_model_path() {
  local model_suffix="$1"
  local base_dir="$HOME/.cache/huggingface/hub/models--Qwen--${model_suffix}"

  if [ ! -d "$base_dir" ]; then
    echo "错误：未在缓存中找到模型目录：$base_dir" >&2
    exit 1
  fi

  # 取 snapshots 下的第一个快照目录作为 model_path
  local snapshot_dir
  snapshot_dir=$(ls -d "$base_dir"/snapshots/* 2>/dev/null | head -n 1)

  if [ -z "$snapshot_dir" ]; then
    echo "错误：模型目录下未找到 snapshots：$base_dir" >&2
    exit 1
  fi

  echo "$snapshot_dir"
}

for model_suffix in "${MODELS[@]}"; do
  MODEL_PATH="$(resolve_model_path "$model_suffix")"
  LOG_FILE="vsibench_evaluation_log.csv"

  echo "=================================================="
  echo "开始测试模型: $model_suffix"
  echo "模型路径: $MODEL_PATH"
  echo "日志文件: $LOG_FILE"
  echo "=================================================="

  # mcq 任务
  for num_frames in "${FRAME_COUNTS_MCQ[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "实验 $CURRENT/$TOTAL_EXPERIMENTS - 模型: $model_suffix, 任务类型: mcq, 帧数: $num_frames"

    python test_vsibench.py \
      --model_path "$MODEL_PATH" \
      --video_dir "$VIDEO_DIR" \
      --num_samples "all" \
      --num_frames "$num_frames" \
      --seed "$SEED" \
      --task_filter "mcq" \
      --log_file "$LOG_FILE"

    sleep 2
  done

  # numeric 任务
  for num_frames in "${FRAME_COUNTS_NUM[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "实验 $CURRENT/$TOTAL_EXPERIMENTS - 模型: $model_suffix, 任务类型: numeric, 帧数: $num_frames"

    python test_vsibench.py \
      --model_path "$MODEL_PATH" \
      --video_dir "$VIDEO_DIR" \
      --num_samples "all" \
      --num_frames "$num_frames" \
      --seed "$SEED" \
      --task_filter "numeric" \
      --log_file "$LOG_FILE"

    sleep 2
  done
done

echo "所有模型的所有实验均已完成。"
