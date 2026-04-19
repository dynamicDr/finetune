#!/bin/bash
# 用法示例（测试 frame_samplers/clip.py 选帧是否可用）:
# sbatch single_run.sh run.py \
#     script=vqa_eval.py \
#     dataset=egoschema \
#     dataset_name=lmms-lab/EgoSchema \
#     dataset_config=Subset \
#     dataset_split=test \
#     video_dir=~/dataset/egoschema/videos/videos \
#     num_samples=2 \
#     num_frames=8 \
#     model=qwen3_4b_thinking \
#     task_filter=mcq \
#     frame_sampling_method=clip \
#     train_ratio=0.0 \
#     seed=42
#
# 说明:
# - 该命令直接走 frame_samplers/clip.py 的 sample_clip_frames 逻辑。
# - 用 num_samples=2 做快速连通性验证（非完整评测）。
#SBATCH --job-name=python_job
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --time=6:00:00
#SBATCH --output=/dev/null    # ← 添加这行
#SBATCH --error=/dev/null     # ← 添加这行

# 检查是否提供了Python脚本参数
if [ -z "$1" ]; then
    echo "错误: 请提供Python脚本路径"
    echo "用法: sbatch run_sbatch.sh <python_script.py> [额外参数...]"
    exit 1
fi

PYTHON_SCRIPT=$1
shift  # 移除第一个参数，保留其余参数

# 创建带时间戳的输出目录
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="outputs/singlerun/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# 设置日志文件路径
LOG_FILE="${OUTPUT_DIR}/output.log"
ERROR_FILE="${OUTPUT_DIR}/error.log"

# 重定向所有输出到日志文件
exec > >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$ERROR_FILE" >&2)

# 打印任务信息
echo "=========================================="
echo "任务开始时间: $(date)"
echo "Python脚本: $PYTHON_SCRIPT"
echo "额外参数: $@"
echo "输出目录: $OUTPUT_DIR"
echo "节点: $SLURM_NODELIST"
echo "任务ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 激活conda环境
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate finetune

# 检查环境是否激活成功
if [ $? -ne 0 ]; then
    echo "错误: 无法激活conda环境 finetune"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 文件 $PYTHON_SCRIPT 不存在"
    exit 1
fi

# 显示环境信息
echo "Python版本: $(python --version 2>&1)"
echo "当前工作目录: $(pwd)"
echo "=========================================="

# 运行Python脚本
python "$PYTHON_SCRIPT" "$@"

EXIT_CODE=$?

# 打印任务结束信息
echo "=========================================="
echo "任务结束时间: $(date)"
echo "退出代码: $EXIT_CODE"
echo "输出保存在: $OUTPUT_DIR"
echo "=========================================="

# 保存任务信息到文件
cat > "${OUTPUT_DIR}/job_info.txt" << EOF
任务ID: $SLURM_JOB_ID
任务名称: $SLURM_JOB_NAME
Python脚本: $PYTHON_SCRIPT
脚本参数: $@
节点: $SLURM_NODELIST
GPU: $CUDA_VISIBLE_DEVICES
开始时间: $(date)
退出代码: $EXIT_CODE
EOF

exit $EXIT_CODE