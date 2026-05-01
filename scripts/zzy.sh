python run.py --multirun \
    hydra/launcher=submitit_slurm \
    script=visbench/eval_vsibench.py \
    model=qwen3_4b_instruct,qwen3_8b_instruct,qwen3_30b_a3b_instruct,qwen3_32b_instruct \
    task_filter=mcq,numeric \
    num_frames=4 \
    train_ratio=0.8 \
    seed=42 \
    slurm=hgpu_batch