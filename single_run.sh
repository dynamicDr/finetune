#!/bin/bash
# 用法A（兼容旧方式，资源由 sbatch CLI 指定）:
#   sbatch --partition=q-h800 --gres=gpu:h800:1 --cpus-per-task=4 --mem=240G --time=2-00:00:00 \
#     single_run.sh sandbox/011_test_videomme_short_random_confidence.py
#
# 用法B（自提交模式，可参数化指定卡型/资源）:
#   bash single_run.sh --submit \
#     --partition q-h800 \
#     --gpu-type h800 \
#     --gpus 1 \
#     --cpus 4 \
#     --mem 240G \
#     --time 2-00:00:00 \
#     sandbox/011_test_videomme_short_random_confidence.py
#
# 用法C（自提交 + 读取 config/slurm/*.yaml）:
#   bash single_run.sh --submit \
#     --slurm-config h800 \
#     sandbox/011_test_videomme_short_random_confidence.py
#
# 也可写完整路径:
#   bash single_run.sh --submit \
#     --slurm-config config/slurm/hgpu_batch.yaml \
#     sandbox/011_test_videomme_short_random_confidence.py
#
# 说明:
# - 用法B会在脚本内自动调用 sbatch 提交任务。
# - 用法C会先从 slurm yaml 读取默认资源，再叠加你显式传入的资源参数。
# - 如果已在 Slurm 任务内运行（存在 SLURM_JOB_ID），会直接执行 Python 脚本。
#SBATCH --job-name=python_job
#SBATCH --gres=gpu:1
#SBATCH --partition=q-hgpu-batch
#SBATCH --account=$USER
#SBATCH --cpus-per-task=4
#SBATCH --mem=190G
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00
#SBATCH --output=/dev/null    # ← 添加这行
#SBATCH --error=/dev/null     # ← 添加这行

# ---------- 解析可选资源参数 ----------
SELF_SUBMIT=0
PARTITION=""
GPU_TYPE=""
GPUS=""
CPUS=""
MEMORY=""
TIME_LIMIT=""
ACCOUNT=""
JOB_NAME=""
SLURM_CONFIG=""

POSITIONAL_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --submit)
            SELF_SUBMIT=1
            shift
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --slurm-config)
            SLURM_CONFIG="$2"
            shift 2
            ;;
        --)
            shift
            while [ $# -gt 0 ]; do
                POSITIONAL_ARGS+=("$1")
                shift
            done
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

extract_yaml_scalar() {
    local key="$1"
    local file="$2"
    awk -F':' -v k="$key" '
        $1 ~ "^[[:space:]]*"k"[[:space:]]*$" {
            v = substr($0, index($0, ":") + 1)
            sub(/^[[:space:]]+/, "", v)
            sub(/[[:space:]]+#.*/, "", v)
            sub(/[[:space:]]+$/, "", v)
            gsub(/^["'\'']|["'\'']$/, "", v)
            print v
            exit
        }
    ' "$file"
}

if [ "$SELF_SUBMIT" -eq 1 ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    if [ -z "$1" ]; then
        echo "错误: 自提交模式下请提供 Python 脚本路径"
        echo "示例: bash single_run.sh --submit --slurm-config h800 script.py"
        exit 1
    fi

    if [ -n "$SLURM_CONFIG" ]; then
        cfg="$SLURM_CONFIG"
        if [ ! -f "$cfg" ]; then
            if [ -f "config/slurm/${cfg}.yaml" ]; then
                cfg="config/slurm/${cfg}.yaml"
            elif [ -f "config/slurm/${cfg}" ]; then
                cfg="config/slurm/${cfg}"
            fi
        fi
        if [ ! -f "$cfg" ]; then
            echo "错误: slurm 配置文件不存在: $SLURM_CONFIG"
            exit 1
        fi

        cfg_partition="$(extract_yaml_scalar partition "$cfg")"
        cfg_gpus="$(extract_yaml_scalar gpus_per_node "$cfg")"
        cfg_cpus="$(extract_yaml_scalar cpus_per_task "$cfg")"
        cfg_mem_gb="$(extract_yaml_scalar mem_gb "$cfg")"
        cfg_timeout_min="$(extract_yaml_scalar timeout_min "$cfg")"
        cfg_account="$(extract_yaml_scalar account "$cfg")"

        if [ -z "$PARTITION" ] && [ -n "$cfg_partition" ]; then
            PARTITION="$cfg_partition"
        fi
        if [ -z "$GPUS" ] && [ -n "$cfg_gpus" ]; then
            GPUS="$cfg_gpus"
        fi
        if [ -z "$CPUS" ] && [ -n "$cfg_cpus" ]; then
            CPUS="$cfg_cpus"
        fi
        if [ -z "$MEMORY" ] && [ -n "$cfg_mem_gb" ]; then
            MEMORY="${cfg_mem_gb}G"
        fi
        if [ -z "$TIME_LIMIT" ] && [ -n "$cfg_timeout_min" ]; then
            TIME_LIMIT="$cfg_timeout_min"
        fi
        if [ -z "$ACCOUNT" ] && [ -n "$cfg_account" ]; then
            cfg_account="${cfg_account//'${oc.env:USER}'/$USER}"
            ACCOUNT="$cfg_account"
        fi

        if [ -z "$GPU_TYPE" ]; then
            cfg_base="$(basename "$cfg")"
            if [[ "$cfg_base" == *h800* ]] || [[ "$PARTITION" == *h800* ]]; then
                GPU_TYPE="h800"
            fi
        fi
    fi

    if [ -z "$ACCOUNT" ]; then
        ACCOUNT="${SLURM_ACCOUNT:-$USER}"
    fi

    SBATCH_CMD=(sbatch)
    if [ -n "$PARTITION" ]; then
        SBATCH_CMD+=("--partition=$PARTITION")
    fi
    if [ -n "$CPUS" ]; then
        SBATCH_CMD+=("--cpus-per-task=$CPUS")
    fi
    if [ -n "$MEMORY" ]; then
        SBATCH_CMD+=("--mem=$MEMORY")
    fi
    if [ -n "$TIME_LIMIT" ]; then
        if [[ "$TIME_LIMIT" =~ ^[0-9]+$ ]]; then
            mins="$TIME_LIMIT"
            days=$((mins / 1440))
            rem=$((mins % 1440))
            hours=$((rem / 60))
            minutes=$((rem % 60))
            if [ "$days" -gt 0 ]; then
                TIME_LIMIT="$(printf "%d-%02d:%02d:00" "$days" "$hours" "$minutes")"
            else
                TIME_LIMIT="$(printf "%02d:%02d:00" "$hours" "$minutes")"
            fi
        fi
        SBATCH_CMD+=("--time=$TIME_LIMIT")
    fi
    if [ -n "$ACCOUNT" ]; then
        SBATCH_CMD+=("--account=$ACCOUNT")
    fi
    if [ -n "$JOB_NAME" ]; then
        SBATCH_CMD+=("--job-name=$JOB_NAME")
    fi
    if [ -n "$GPUS" ]; then
        if [ -n "$GPU_TYPE" ]; then
            SBATCH_CMD+=("--gres=gpu:${GPU_TYPE}:${GPUS}")
        else
            SBATCH_CMD+=("--gres=gpu:${GPUS}")
        fi
    fi

    SBATCH_CMD+=("$(realpath "$0")")
    for arg in "$@"; do
        SBATCH_CMD+=("$arg")
    done

    echo "提交命令: ${SBATCH_CMD[*]}"
    "${SBATCH_CMD[@]}"
    exit $?
fi

# 检查是否提供了 Python 脚本参数
if [ -z "$1" ]; then
    echo "错误: 请提供 Python 脚本路径"
    echo "用法A: sbatch single_run.sh <python_script.py> [额外参数...]"
    echo "用法B: bash single_run.sh --submit [资源参数] <python_script.py> [额外参数...]"
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

# 确保可导入仓库内模块（如 data_loaders、frame_samplers 等）
export PYTHONPATH="$(pwd):${PYTHONPATH}"

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