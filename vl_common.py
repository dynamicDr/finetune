from __future__ import annotations

import os
import random
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration


def _target_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _move_model_to_target_device(model):
    # 强制单设备路径，避免 device_map=auto 在不同环境触发不一致性能。
    return model.to(_target_device())


def _use_generic_vl_loader(model_id: str) -> bool:
    model_id_lower = model_id.lower()
    generic_model_hints = (
        "qwen3-vl",
        "llava-onevision",
        "llava-next",
    )
    return any(hint in model_id_lower for hint in generic_model_hints)


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
        + "\n\nDirectly answer with the option letter only. Do not explain."
    )


def load_model_and_processor(
    model_path: str,
    use_lora: bool = False,
    base_model: str | None = None,
    merge_lora: bool = False,
):
    model_path = os.path.expanduser(model_path)
    print(
        "[perf-debug][load] "
        f"model_path={model_path}, use_lora={use_lora}, base_model={base_model}, merge_lora={merge_lora}",
        flush=True,
    )
    if use_lora and base_model:
        from peft import PeftModel

        base_id = os.path.expanduser(base_model)
        if _use_generic_vl_loader(base_id):
            base = AutoModelForImageTextToText.from_pretrained(
                base_id,
                dtype=torch.bfloat16,
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
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, model_path)
            if merge_lora:
                model = model.merge_and_unload()
            processor = AutoProcessor.from_pretrained(base_id)
        model = _move_model_to_target_device(model)
        model.eval()
        attn_impl = getattr(model.config, "_attn_implementation", None)
        print(
            "[perf-debug][model] "
            f"dtype={model.dtype}, attn_impl={attn_impl}, device={next(model.parameters()).device}",
            flush=True,
        )
        image_processor = getattr(processor, "image_processor", None)
        min_pixels = getattr(image_processor, "min_pixels", None) if image_processor is not None else None
        max_pixels = getattr(image_processor, "max_pixels", None) if image_processor is not None else None
        print(
            "[perf-debug][processor] "
            f"processor={type(processor).__name__}, min_pixels={min_pixels}, max_pixels={max_pixels}",
            flush=True,
        )
        return model, processor

    if _use_generic_vl_loader(model_path):
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = _move_model_to_target_device(model)
        model.eval()
        attn_impl = getattr(model.config, "_attn_implementation", None)
        print(
            "[perf-debug][model] "
            f"dtype={model.dtype}, attn_impl={attn_impl}, device={next(model.parameters()).device}",
            flush=True,
        )
        image_processor = getattr(processor, "image_processor", None)
        min_pixels = getattr(image_processor, "min_pixels", None) if image_processor is not None else None
        max_pixels = getattr(image_processor, "max_pixels", None) if image_processor is not None else None
        print(
            "[perf-debug][processor] "
            f"processor={type(processor).__name__}, min_pixels={min_pixels}, max_pixels={max_pixels}",
            flush=True,
        )
        return model, processor

    if use_lora:
        from peft import PeftModel

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    model = _move_model_to_target_device(model)
    model.eval()
    attn_impl = getattr(model.config, "_attn_implementation", None)
    print(
        "[perf-debug][model] "
        f"dtype={model.dtype}, attn_impl={attn_impl}, device={next(model.parameters()).device}",
        flush=True,
    )
    image_processor = getattr(processor, "image_processor", None)
    min_pixels = getattr(image_processor, "min_pixels", None) if image_processor is not None else None
    max_pixels = getattr(image_processor, "max_pixels", None) if image_processor is not None else None
    print(
        "[perf-debug][processor] "
        f"processor={type(processor).__name__}, min_pixels={min_pixels}, max_pixels={max_pixels}",
        flush=True,
    )
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
    t_proc_start = time.time()
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    processor_time = time.time() - t_proc_start
    input_keys = sorted(list(inputs.keys()))
    print(
        f"[perf-debug][vision-input] frames_count={len(frames)}, input_keys={input_keys}",
        flush=True,
    )
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        print(
            "[perf-debug][vision-input] "
            f"pixel_values.shape={tuple(pixel_values.shape)}, pixel_values.dtype={pixel_values.dtype}",
            flush=True,
        )
    else:
        print("[perf-debug][vision-input] pixel_values=None", flush=True)

    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        print(
            "[perf-debug][vision-input] "
            f"image_grid_thw.shape={tuple(image_grid_thw.shape)}, image_grid_thw={image_grid_thw}",
            flush=True,
        )
    else:
        print("[perf-debug][vision-input] image_grid_thw=None", flush=True)

    visual_token_ids = collect_visual_token_ids(processor)
    if visual_token_ids and "input_ids" in inputs:
        input_ids = inputs["input_ids"]
        visual_token_count = int(sum((input_ids == tid).sum().item() for tid in visual_token_ids))
        print(
            "[perf-debug][vision-input] "
            f"visual_token_ids={visual_token_ids}, visual_token_count={visual_token_count}",
            flush=True,
        )
    else:
        print(
            "[perf-debug][vision-input] visual_token_ids unavailable or input_ids missing",
            flush=True,
        )
    print(
        "[perf-debug][inputs] "
        f"input_ids.shape={tuple(inputs['input_ids'].shape)}, attention_mask.shape={tuple(inputs['attention_mask'].shape)}",
        flush=True,
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    print(
        "[perf-debug][generate-config] "
        f"max_new_tokens={gen_kwargs['max_new_tokens']}, do_sample={gen_kwargs['do_sample']}, num_beams={gen_kwargs['num_beams']}",
        flush=True,
    )
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )
    inference_time = time.time() - start_time

    t_decode_start = time.time()
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    decode_time = time.time() - t_decode_start
    generated_token_count = int(generated_ids_trimmed[0].shape[0]) if generated_ids_trimmed else 0
    print(
        "[perf-debug][outputs] "
        f"generated_tokens={generated_token_count}, output_minus_input={generated_ids.shape[-1] - inputs['input_ids'].shape[-1]}",
        flush=True,
    )
    print(f"[perf-debug][decoded-full]\n{response}", flush=True)
    print(
        "[perf-debug][timing-split] "
        f"processor={processor_time:.2f}s, generate={inference_time:.2f}s, decode={decode_time:.2f}s",
        flush=True,
    )
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
        outputs = model.generate(
            inputs_embeds=fused_embeds,
            attention_mask=model_inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    inference_time = time.time() - infer_start
    generated_ids = outputs.sequences
    response = _decode_new_tokens(processor, model_inputs["input_ids"], generated_ids)
    generated_token_count = int(len(outputs.scores))
    hit_max_tokens = generated_token_count >= max_new_tokens
    return response, inference_time, embedding_build_time, generated_token_count, hit_max_tokens
