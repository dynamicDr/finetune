# finetune 环境快照（用于回退）

- 记录时间：2026-05-01
- 环境名：`finetune`
- 前缀：`/userhome/cs3/duanty/miniconda3/envs/finetune`

## Conda 基础信息

```yaml
name: finetune
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - defaults
dependencies:
  - python=3.10
  - unzip
prefix: /userhome/cs3/duanty/miniconda3/envs/finetune
```

## Python / Torch 关键信息

- Python: `3.10.19`
- Python executable: `/userhome/cs3/duanty/miniconda3/envs/finetune/bin/python`
- Torch: `2.9.1+cu128`
- Torch CUDA 编译版本: `12.8`
- Platform: `Linux-5.15.0-168-generic-x86_64-with-glibc2.35`

## 关键 Python 包版本（回退锚点）

```text
torch==2.9.1
torchvision==0.24.1
transformers==5.5.3
accelerate==1.12.0
tokenizers==0.22.2
numpy==1.26.4
peft==0.18.1
datasets==4.5.0
triton==3.5.1
bitsandbytes==0.49.2
qwen-vl-utils==0.0.14
```

## CUDA 相关轮子版本（来自 conda list）

```text
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.3.20
nvidia-nvtx-cu12==12.8.90
```

## 建议回退命令（按本快照）

```bash
conda activate finetune
pip install --upgrade --no-cache-dir \
  torch==2.9.1 torchvision==0.24.1 transformers==5.5.3 \
  accelerate==1.12.0 tokenizers==0.22.2 numpy==1.26.4 \
  peft==0.18.1 datasets==4.5.0 triton==3.5.1 \
  bitsandbytes==0.49.2 qwen-vl-utils==0.0.14
```
