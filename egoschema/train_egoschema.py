"""
在 EgoSchema 上对 Qwen3-VL 做 SFT（LoRA/QLoRA）。

默认使用 HuggingFace: lmms-lab/EgoSchema 的 validation split，
并按 seed/train_ratio 划分训练/测试部分，确保可与 eval_egoschema.py 对齐。
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from hf_dataset_loader import load_dataset
from vl_common import build_mcq_prompt, collect_visual_token_ids, extract_video_frames, split_indices


def _normalize_options(sample: dict[str, Any]) -> list[str] | None:
    options = sample.get("options")
    if isinstance(options, list) and options:
        return [str(x).strip() for x in options]

    collected: list[str] = []
    for key in ("option", "choices", "candidates"):
        cand = sample.get(key)
        if isinstance(cand, list) and cand:
            return [str(x).strip() for x in cand]

    for i in range(8):
        for k in (f"option_{i}", f"option{i}", f"a{i}", f"choice_{i}"):
            if k in sample and str(sample[k]).strip():
                collected.append(str(sample[k]).strip())
                break
    return collected if collected else None


def _answer_to_letter(answer: Any, num_options: int) -> str | None:
    if answer is None:
        return None
    text = str(answer).strip().upper()
    if not text:
        return None

    if len(text) == 1 and "A" <= text <= "Z":
        return text

    m = re.search(r"-?\d+", text)
    if m:
        idx = int(m.group(0))
        if 0 <= idx < num_options:
            return chr(ord("A") + idx)
        if 1 <= idx <= num_options:
            return chr(ord("A") + idx - 1)
    return None


def _resolve_video_path(video_dir: str, sample: dict[str, Any]) -> str | None:
    # 按常见字段名尝试拼接本地视频路径
    candidates: list[str] = []
    for key in ("video_path", "video", "video_id", "video_uid", "video_idx", "sample_id", "uid", "q_uid"):
        val = sample.get(key)
        if val is not None and str(val).strip():
            candidates.append(str(val).strip())

    for c in candidates:
        if os.path.isfile(c):
            return c
        p = os.path.join(video_dir, c)
        if os.path.isfile(p):
            return p
        for ext in (".mp4", ".MP4", ".webm", ".mkv", ".avi"):
            p2 = os.path.join(video_dir, c + ext)
            if os.path.isfile(p2):
                return p2
    return None


def build_training_examples(
    video_dir: str,
    dataset_name: str,
    split: str,
    seed: int,
    train_ratio: float,
    use_train_split: bool,
    max_samples: int | None,
) -> list[dict[str, str]]:
    ds = load_dataset(dataset_name, split=split)
    indices = list(range(len(ds)))
    indices = split_indices(indices, seed, train_ratio, use_train_split)
    if max_samples is not None:
        indices = indices[:max_samples]

    records: list[dict[str, str]] = []
    for i in tqdm(indices, desc="构建 EgoSchema 训练样本"):
        sample = ds[i]
        question = str(sample.get("question", "")).strip()
        if not question:
            continue

        options = _normalize_options(sample)
        if not options:
            continue

        raw_answer = sample.get("answer", sample.get("ground_truth", sample.get("label")))
        answer_letter = _answer_to_letter(raw_answer, len(options))
        if not answer_letter:
            continue

        video_path = _resolve_video_path(video_dir, sample)
        if not video_path:
            continue

        records.append(
            {
                "video_path": video_path,
                "user_text": build_mcq_prompt(question, options),
                "answer": answer_letter,
            }
        )

    if not records:
        raise RuntimeError("没有可用训练样本，请检查 EgoSchema 字段和 video_dir 本地视频命名。")
    return records


def make_collate_fn(processor, num_frames: int, visual_token_ids: list[int]):
    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        batch_images: list[list] = []
        for ex in examples:
            frames = extract_video_frames(ex["video_path"], num_frames=num_frames)
            if not frames:
                raise ValueError(f"无法从视频取帧: {ex['video_path']}")
            content = [{"type": "image", "image": f} for f in frames]
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
    p = argparse.ArgumentParser(description="EgoSchema SFT 训练")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--video_dir", type=str, default="~/dataset/egoschema/videos")
    p.add_argument("--output_dir", type=str, default="./outputs/egoschema_sft_lora")
    p.add_argument("--dataset_name", type=str, default="lmms-lab/EgoSchema")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
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

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    video_dir = os.path.expanduser(args.video_dir)
    model_path = os.path.expanduser(args.model_path)
    use_train_split = not args.use_test_split

    rows = build_training_examples(
        video_dir=video_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        seed=args.seed,
        train_ratio=args.train_ratio,
        use_train_split=use_train_split,
        max_samples=args.max_samples,
    )
    print(f"有效 EgoSchema 训练样本数: {len(rows)}")

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

    training_args = SFTConfig(
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
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer_kw: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(rows),
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
    print(f"训练完成，已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
