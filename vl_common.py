from __future__ import annotations

import os
import random
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration


def split_indices(
    indices: list[int],
    seed: int,
    train_ratio: float,
    use_train_split: bool,
) -> list[int]:
    copied = indices.copy()
    random.seed(seed)
    random.shuffle(copied)
    split_point = int(len(copied) * train_ratio)
    return copied[:split_point] if use_train_split else copied[split_point:]


def collect_visual_token_ids(processor) -> list[int]:
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


def build_mcq_prompt(question: str, options: list[str]) -> str:
    lines = [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]
    return (
        f"{question}\n\nOptions:\n"
        + "\n".join(lines)
        + "\n\nPlease answer with the option letter directly."
    )


def load_model_and_processor(
    model_path: str,
    use_lora: bool = False,
    base_model: str | None = None,
    merge_lora: bool = False,
):
    model_path = os.path.expanduser(model_path)
    if use_lora and base_model:
        from peft import PeftModel

        base_id = os.path.expanduser(base_model)
        if "Qwen3-VL" in base_id:
            base = AutoModelForImageTextToText.from_pretrained(
                base_id,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, model_path)
            if merge_lora:
                model = model.merge_and_unload()
            proc_src = (
                model_path
                if os.path.isfile(os.path.join(model_path, "preprocessor_config.json"))
                or os.path.isfile(os.path.join(model_path, "tokenizer_config.json"))
                else base_id
            )
            processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)
        else:
            base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_id,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, model_path)
            if merge_lora:
                model = model.merge_and_unload()
            processor = AutoProcessor.from_pretrained(base_id)
        model.eval()
        return model, processor

    if "Qwen3-VL" in model_path:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        return model, processor

    if use_lora:
        from peft import PeftModel

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    model.eval()
    return model, processor


def generate_response(model, processor, frames: list[Image.Image], prompt: str) -> tuple[str, float]:
    import time

    content: list[dict[str, Any]] = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
    inference_time = time.time() - start_time

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response, inference_time
