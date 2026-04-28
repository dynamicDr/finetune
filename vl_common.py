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


def generate_response(
    model,
    processor,
    frames: list[Image.Image],
    prompt: str,
    max_new_tokens: int = 1024,
) -> tuple[str, float, int, bool]:
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
            max_new_tokens=max_new_tokens,
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
    generated_token_count = int(generated_ids_trimmed[0].shape[0]) if generated_ids_trimmed else 0
    hit_max_tokens = generated_token_count >= max_new_tokens
    return response, inference_time, generated_token_count, hit_max_tokens


def _decode_new_tokens(processor, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    prompt_len = int(input_ids.shape[1])
    token_ids = generated_ids[:, prompt_len:] if generated_ids.shape[1] > prompt_len else generated_ids
    text = processor.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def _build_image_fused_embeds(model, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if "pixel_values" not in model_inputs:
        return None
    if not hasattr(model, "get_image_features") or not hasattr(model, "get_input_embeddings"):
        return None
    if not hasattr(model, "model") or not hasattr(model.model, "get_placeholder_mask"):
        return None

    image_grid_thw = model_inputs.get("image_grid_thw")
    with torch.no_grad():
        image_outputs = model.get_image_features(
            pixel_values=model_inputs["pixel_values"],
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )

    image_embed_seq = getattr(image_outputs, "pooler_output", None)
    if not isinstance(image_embed_seq, (list, tuple)) or len(image_embed_seq) == 0:
        return None

    image_embeds = torch.cat(image_embed_seq, dim=0).to(model.device)
    token_embeds = model.get_input_embeddings()(model_inputs["input_ids"])
    image_mask, _ = model.model.get_placeholder_mask(
        model_inputs["input_ids"],
        inputs_embeds=token_embeds,
        image_features=image_embeds,
    )
    return token_embeds.masked_scatter(image_mask, image_embeds)


def generate_response_with_split_embedding(
    model,
    processor,
    frames: list[Image.Image],
    prompt: str,
    max_new_tokens: int = 1024,
) -> tuple[str, float, float, int, bool]:
    import time

    content: list[dict[str, Any]] = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to(model.device)

    build_start = time.time()
    fused_embeds = _build_image_fused_embeds(model, model_inputs)
    embedding_build_time = time.time() - build_start
    if fused_embeds is None:
        raise RuntimeError("模型不支持 embedding 分开输入，或构建图像 embedding 失败。")

    infer_start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            inputs_embeds=fused_embeds,
            attention_mask=model_inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    inference_time = time.time() - infer_start
    response = _decode_new_tokens(processor, model_inputs["input_ids"], generated_ids)
    prompt_len = int(model_inputs["input_ids"].shape[1])
    generated_token_count = int(generated_ids.shape[1] - prompt_len)
    hit_max_tokens = generated_token_count >= max_new_tokens
    return response, inference_time, embedding_build_time, generated_token_count, hit_max_tokens
