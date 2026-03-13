#!/bin/bash
# 测试每不同的 num_frame对结果的影响
MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
VIDEO_DIR="$HOME/dataset/vsi_bench"

SEED=42
LOG_FILE="vsibench_evaluation_log.csv"

FRAME_COUNTS_MCQ=(16 32 64 128)
FRAME_COUNTS_NUM=(1 2 4 8 16 32 64)
TASK_TYPES=("mcq" "numeric")

TOTAL_EXPERIMENTS=$((${#TASK_TYPES[@]} * ${#FRAME_COUNTS[@]}))
CURRENT=0


#for num_frames in "${FRAME_COUNTS_MCQ[@]}"; do
#    CURRENT=$((CURRENT + 1))
#    echo "实验 $CURRENT/$TOTAL_EXPERIMENTS - 任务类型: mcq, 帧数: $num_frames"
#
#    python test_vsibench.py \
#        --model_path "$MODEL_PATH" \
#        --video_dir "$VIDEO_DIR" \
#        --num_samples "all" \
#        --num_frames $num_frames \
#        --seed $SEED \
#        --task_filter "mcq" \
#        --log_file "$LOG_FILE"
#
#    sleep 2
#done

for num_frames in "${FRAME_COUNTS_NUM[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "实验 $CURRENT/$TOTAL_EXPERIMENTS - 任务类型: numeric, 帧数: $num_frames"

    python test_vsibench.py \
        --model_path "$MODEL_PATH" \
        --video_dir "$VIDEO_DIR" \
        --num_samples "all" \
        --num_frames $num_frames \
        --seed $SEED \
        --task_filter "numeric" \
        --log_file "$LOG_FILE"

    sleep 2
done

echo "所有实验完成"