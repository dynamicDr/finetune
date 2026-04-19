from __future__ import annotations

import argparse
import inspect
import os
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from data_loaders import get_data_loader, list_supported_datasets
from data_loaders.base import VQASample
from frame_samplers import sample_video_frames
from vl_common import collect_visual_token_ids


def build_user_text(question: str, options: list[str] | None) -> str:
    if options:
        return (
            f"{question}\n\nOptions:\n"
            + "\n".join(options)
            + "\n\nPlease answer with the option letter directly."
        )
    return f"{question}\n\nPlease provide the numerical answer directly."


def build_training_examples(samples: list[VQASample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.append(
            {
                "video_path": sample.video_path,
                "user_text": build_user_text(sample.question, sample.options),
                "question_text": sample.question,
                "answer": str(sample.answer).strip(),
            }
        )
    if not rows:
        raise RuntimeError("没有可用训练样本。")
    return rows


def make_collate_fn(processor, num_frames: int, visual_token_ids: list[int]):
    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        batch_images: list[list] = []
        for i, ex in enumerate(examples):
            frame_sampling_method = ex.get("frame_sampling_method", "uniform")
            random_seed = ex.get("sample_seed", None) if frame_sampling_method == "random" else None
            if random_seed is not None:
                random_seed = int(random_seed) + i
            frames = sample_video_frames(
                video_path=ex["video_path"],
                num_frames=num_frames,
                method=frame_sampling_method,
                random_seed=random_seed,
                question=ex.get("question_text", ""),
                answer=ex.get("answer", ""),
                focus_blip_model_name=ex.get("focus_blip_model_name", "Salesforce/blip-itm-base-coco"),
                focus_blip_device=ex.get("focus_blip_device", None),
                focus_blip_batch_size=int(ex.get("focus_blip_batch_size", 16)),
            )
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

        inputs = processor(text=texts, images=batch_images, return_tensors="pt", padding=True)
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
    p = argparse.ArgumentParser(description="通用 VQA 训练脚本（TRL + LoRA/QLoRA）")
    p.add_argument("--dataset", type=str, default="vsibench", choices=list_supported_datasets())
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--dataset_name", type=str, default="nyu-visionx/VSI-Bench")
    p.add_argument("--dataset_config", type=str, default="full")
    p.add_argument("--no_dataset_config", action="store_true")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--model_name", type=str, default=None, help="模型名称（可选，便于日志标识）")
    p.add_argument("--video_dir", type=str, default="~/dataset/vsi_bench")
    p.add_argument("--output_dir", type=str, default="./outputs/vqa_sft_lora")
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument(
        "--frame_sampling_method",
        type=str,
        default="uniform",
        choices=["uniform", "random", "focus", "sevila", "videoagent", "clip", "siglip2", "aks"],
    )
    p.add_argument("--focus_blip_model_name", type=str, default="Salesforce/blip-itm-base-coco")
    p.add_argument("--focus_blip_device", type=str, default=None)
    p.add_argument("--focus_blip_batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task_filter", type=str, default="all", choices=["all", "mcq", "numeric"])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--use_test_split", action="store_true")

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--no_qlora", action="store_true")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", type=str, default=None)

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    video_dir = os.path.expanduser(args.video_dir)
    model_path = os.path.expanduser(args.model_path)
    if args.model_name:
        print(f"训练模型: {args.model_name}")
    use_train_split = not args.use_test_split

    loader = get_data_loader(
        args.dataset,
        video_dir=video_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        task_filter=args.task_filter,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        no_dataset_config=args.no_dataset_config,
    )
    samples = loader.get_split_samples(
        split=args.dataset_split,
        use_train_split=use_train_split,
        max_samples=args.max_samples,
    )
    train_rows = build_training_examples(samples)
    for row_idx, row in enumerate(train_rows):
        row["frame_sampling_method"] = args.frame_sampling_method
        row["sample_seed"] = args.seed + row_idx
        row["focus_blip_model_name"] = args.focus_blip_model_name
        row["focus_blip_device"] = args.focus_blip_device
        row["focus_blip_batch_size"] = args.focus_blip_batch_size
    print(f"有效训练样本数: {len(train_rows)}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    visual_ids = collect_visual_token_ids(processor)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"],
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
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    if args.hub_model_id:
        sft_kw["hub_model_id"] = args.hub_model_id
    training_args = SFTConfig(**sft_kw)

    trainer_kw: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        data_collator=make_collate_fn(processor, args.num_frames, visual_ids),
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

