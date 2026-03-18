#!/bin/bash

# Parameters
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=4
#SBATCH --error=/userhome/cs3/duanty/finetune/outputs/multirun/2026-03-14_01-07-05/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=run
#SBATCH --mem=96GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/userhome/cs3/duanty/finetune/outputs/multirun/2026-03-14_01-07-05/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=q-hgpu-batch
#SBATCH --signal=USR2@120
#SBATCH --time=10080
#SBATCH --wckey=submitit

# setup
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate finetune

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /userhome/cs3/duanty/finetune/outputs/multirun/2026-03-14_01-07-05/.submitit/%A_%a/%A_%a_%t_log.out --error /userhome/cs3/duanty/finetune/outputs/multirun/2026-03-14_01-07-05/.submitit/%A_%a/%A_%a_%t_log.err /userhome/cs3/duanty/miniconda3/bin/python -u -m submitit.core._submit /userhome/cs3/duanty/finetune/outputs/multirun/2026-03-14_01-07-05/.submitit/%j
