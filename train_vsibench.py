"""
train_vsibench.py - 在 VSI-Bench 标注上对已下载视频帧做 Qwen3-VL 的 SFT（QLoRA/LoRA），流程对齐 test_vsibench.py，训练方式参照 notebooks/sft_qwen_vl.ipynb（TRL SFTTrainer + PEFT）。

依赖（与 notebook 一致）：
  pip install -U "trl[peft]" bitsandbytes

示例：
  python train_vsibench.py --model_path Qwen/Qwen3-VL-4B-Instruct \\
    --video_dir ~/dataset/vsi_bench --num_frames 4 --task_filter all \\
    --output_dir ./outputs/vsibench_sft --max_samples 500 \\
    --train_ratio 0.8 --seed 42

注意：使用 --train_ratio 和 --seed 参数划分训练/测试集，测试时需使用相同参数以保证划分一致。
"""
from __future__ import annotations

import argparse
import inspect
import os
import random
from typing import Any

import cv2
import torch
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig
from hf_dataset_loader import load_dataset

from trl import SFTConfig, SFTTrainer


def extract_video_frames(video_path: str, num_frames: int = 8) -> list[Image.Image]:
    """从视频中均匀提取指定数量的帧（与 test_vsibench 一致）"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames: list[Image.Image] = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def should_include_sample(sample: dict, task_filter: str) -> bool:
    """与 test_vsibench 一致：按题型筛选"""
    return task_filter == "all" or (sample.get("options") is not None) == (task_filter == "mcq")


def build_user_text(question: str, options: list[str] | None) -> str:
    if options:
        return (
            f"{question}\n\nOptions:\n"
            + "\n".join(options)
            + "\n\nPlease answer with the option letter directly (A, B, C, or D)."
        )
    return f"{question}\n\nPlease provide the numerical answer directly."


def collect_visual_token_ids(processor) -> list[int]:
    """在 labels 中屏蔽视觉占位符 token，避免对图像占位算 loss。"""
    tok = processor.tokenizer
    candidates = [
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
    ]
    ids: list[int] = []
    for s in candidates:
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid != tok.unk_token_id and tid not in ids:
            ids.append(tid)
    if hasattr(processor, "image_token") and processor.image_token:
        tid = tok.convert_tokens_to_ids(processor.image_token)
        if tid not in ids:
            ids.append(tid)
    return ids


def split_indices(
    indices: list[int],
    seed: int,
    train_ratio: float,
    use_train_split: bool,
) -> list[int]:
    """
    根据 seed 和 train_ratio 划分索引，返回训练集或测试集部分。
    必须保证 train 和 test 脚本使用相同的 seed 和 train_ratio。
    """
    indices_copy = indices.copy()
    random.seed(seed)
    random.shuffle(indices_copy)
    
    split_point = int(len(indices_copy) * train_ratio)
    if use_train_split:
        return indices_copy[:split_point]
    else:
        return indices_copy[split_point:]


def build_training_examples(
    video_dir: str,
    num_frames: int,
    seed: int,
    task_filter: str,
    max_samples: int | None,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    train_ratio: float = 0.8,
    use_train_split: bool = True,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    if dataset_config:
        kwargs["name"] = dataset_config
    ds = load_dataset(dataset_name, split=split, **kwargs)

    # 筛选符合条件的样本
    indices = [i for i in range(len(ds)) if should_include_sample(ds[i], task_filter)]
    
    # 划分训练集/测试集
    indices = split_indices(indices, seed, train_ratio, use_train_split)
    
    split_name = "训练集" if use_train_split else "测试集"
    print(f"划分后 {split_name} 样本数: {len(indices)} (train_ratio={train_ratio}, seed={seed})")
    
    # 限制最大样本数
    if max_samples is not None:
        indices = indices[:max_samples]

    records: list[dict[str, Any]] = []
    for i in tqdm(indices, desc="构建训练样本（检查视频路径）"):
        sample = ds[i]
        scene = sample.get("scene_name", "")
        video_path = os.path.join(video_dir, f"{scene}.mp4")
        if not os.path.isfile(video_path):
            continue
        question = sample.get("question", "")
        options = sample.get("options", None)
        gt = sample.get("ground_truth", "")
        answer = str(gt).strip()
        if not answer:
            continue
        user_text = build_user_text(question, options)
        records.append(
            {
                "video_path": video_path,
                "user_text": user_text,
                "answer": answer,
            }
        )
    if not records:
        raise RuntimeError(
            "没有可用训练样本：请确认 video_dir 下已有 mp4，且与 nyu-visionx/VSI-Bench 的 scene_name 一致。"
        )
    return records


def make_collate_fn(processor, num_frames: int, visual_token_ids: list[int]):
    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        batch_images: list[list[Image.Image]] = []

        for ex in examples:
            frames = extract_video_frames(ex["video_path"], num_frames=num_frames)
            if not frames:
                raise ValueError(f"无法从视频取帧: {ex['video_path']}")
            content: list[dict[str, Any]] = [{"type": "image", "image": f} for f in frames]
            content.append({"type": "text", "text": ex["user_text"]})
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]},
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False))
            batch_images.append(frames)

        inputs = processor(
            text=texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        labels = inputs["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for vid in visual_token_ids:
            labels[labels == vid] = -100

        inputs["labels"] = labels
        return inputs

    return collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="在 VSI-Bench 上对 Qwen3-VL 做 SFT（TRL + LoRA/QLoRA）")
    p.add_argument("--model_path", type=str, required=True, help="基座模型路径或 HuggingFace ID（建议 Qwen3-VL-Instruct）")
    p.add_argument("--video_dir", type=str, default="~/dataset/vsi_bench", help="VSI-Bench 视频目录")
    p.add_argument("--output_dir", type=str, default="./outputs/vsibench_sft_lora", help="checkpoint 输出目录")
    p.add_argument("--num_frames", type=int, default=4, help="每个视频采样帧数（与评测对齐可设 4/8）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task_filter", type=str, default="all", choices=["all", "mcq", "numeric"])
    p.add_argument("--max_samples", type=int, default=None, help="最多使用多少条样本（默认全部可用样本）")
    p.add_argument("--dataset_name", type=str, default="nyu-visionx/VSI-Bench")
    p.add_argument(
        "--dataset_config",
        type=str,
        default="full",
        help="VSI-Bench 子配置：full / debiased / pruned；留空则不加 name 参数",
    )
    p.add_argument("--split", type=str, default="test", help="数据 split（该数据集多为 test）")
    p.add_argument("--no_dataset_config", action="store_true", help="load_dataset 时不传 name=（兼容旧缓存）")

    # 数据划分参数
    p.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例，剩余为测试集（默认0.8）")
    p.add_argument("--use_test_split", action="store_true", help="使用测试部分进行训练（默认使用训练部分）")

    # 训练超参（对齐 sft_qwen_vl 思路，可按显存改）
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1, help=">0 时覆盖 num_train_epochs")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--bf16", action="store_true", help="非量化时用 bfloat16 训练（需 GPU 支持）")
    p.add_argument("--no_qlora", action="store_true", help="关闭 4bit 量化，配合 --bf16 全精度/半精度加载")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="push_to_hub 时可选，对应 TrainingArguments.hub_model_id",
    )

    # LoRA（与 sft_qwen_vl notebook 一致）
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )
    return p.parse_args()


def main():
    args = parse_args()
    video_dir = os.path.expanduser(args.video_dir)
    model_path = os.path.expanduser(args.model_path)

    # use_train_split: 默认 True（训练时用训练集），除非指定 --use_test_split
    use_train_split = not args.use_test_split

    train_rows = build_training_examples(
        video_dir=video_dir,
        num_frames=args.num_frames,
        seed=args.seed,
        task_filter=args.task_filter,
        max_samples=args.max_samples,
        dataset_name=args.dataset_name,
        dataset_config=None if args.no_dataset_config else args.dataset_config,
        split=args.split,
        train_ratio=args.train_ratio,
        use_train_split=use_train_split,
    )
    print(f"有效训练样本数: {len(train_rows)}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    visual_ids = collect_visual_token_ids(processor)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "down_proj",
            "o_proj",
            "k_proj",
            "q_proj",
            "gate_proj",
            "up_proj",
            "v_proj",
        ],
    )

    if args.no_qlora:
        dtype = torch.bfloat16 if args.bf16 else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    sft_kw: dict[str, Any] = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16 and args.no_qlora,
        optim="adamw_torch",
        max_length=None,
        remove_unused_columns=False,
        report_to="none",
        push_to_hub=args.push_to_hub,
        # 多模态在 collate 里做 processor；勿让 TRL 按纯文本字段 "text" 预 tokenize
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    if args.hub_model_id:
        sft_kw["hub_model_id"] = args.hub_model_id

    training_args = SFTConfig(**sft_kw)

    train_dataset = Dataset.from_list(train_rows)
    collate_fn = make_collate_fn(processor, args.num_frames, visual_ids)

    trainer_kw: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
    )
    proc_kw = (
        {"processing_class": processor}
        if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
        else {"tokenizer": processor}
    )
    trainer = SFTTrainer(**trainer_kw, **proc_kw)

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"训练结束，adapter 与 processor 已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()