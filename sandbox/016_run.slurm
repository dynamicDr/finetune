#!/bin/bash
#SBATCH --job-name=frame_hm_016
#SBATCH --partition=debug
#SBATCH --nodelist=gpu-4090-a02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/userhome/cs3/duanty/finetune/sandbox/016_outputs/slurm_%j.out
#SBATCH --error=/userhome/cs3/duanty/finetune/sandbox/016_outputs/slurm_%j.err

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate finetune
export NO_PROXY="api.aiohub.org,aiohub.org,api.poe.com,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

cd /userhome/cs3/duanty/finetune
mkdir -p sandbox/016_outputs
python sandbox/016_baseline_frame_selection_heatmap.py
