python run.py --multirun \
    hydra/launcher=submitit_slurm \
    script=test_vsibench.py \
    model=qwen3_4b,qwen3_8b,qwen3_30b_a3b,qwen3_32b \
    task_filter=mcq,numeric \
    num_frames=4 \
    train_ratio=0.8 \
    seed=42 \
    slurm=hgpu_batch