#!/bin/bash

# SLURM 资源配置
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH -t 2-00:00:00
#SBATCH --job-name=test_frame_batch
#SBATCH --output=test_frame_batch_%j.out
#SBATCH --error=test_frame_batch_%j.err

# 初始化 Miniconda
. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate finetune  # 改成你的环境名

# 打印作业信息
echo "作业开始时间: $(date)"
echo "节点: $SLURMD_NODENAME"
echo "作业ID: $SLURM_JOB_ID"
nvidia-smi
echo "================================"

# 进入脚本所在目录（如果需要）
# cd /path/to/your/script/directory

# 调用要测试的脚本
bash test_model.sh

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "================================"
    echo "执行成功"
    echo "完成时间: $(date)"
else
    echo "================================"
    echo "执行失败"
    echo "失败时间: $(date)"
    exit 1
fi