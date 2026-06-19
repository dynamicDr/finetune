"""lmms-eval 桥接层：模型加载、推理与 VQA 评测共用工具。

vqa_eval.py / vqa_eval_ours.py 通过本模块使用 lmms-eval 的模型实现，
替代原 vl_common.py 中的手写加载逻辑。
"""
from __future__ import annotations

import gc
import os
import random
import re
from pathlib import Path
from typing import Any

import llava_transformers_compat  # noqa: F401  # must run before llava / lmms-eval llava_vid

import torch
from PIL import Image

from lmms_eval.models import get_model

_LLAVA_HF_VIDEO_TOKEN = "<video>"
_LLAVA_HF_IMAGE_TOKEN = "<image>"
_DEFAULT_MIN_VISUAL_TOKENS = 256
# lmms-eval qwen2_vl / qwen2_5_vl / qwen3_vl 默认 processor 像素预算
_QWEN_VL_OFFICIAL_MIN_PIXELS = 256 * 28 * 28
_QWEN_VL_OFFICIAL_MAX_PIXELS = 1605632
_QWEN_VL_OFFICIAL_SYSTEM_PROMPT = "You are a helpful assistant."
_LLAVA_DEFAULT_MAX_IMAGE_EDGE = 768
_LLAVA_DEFAULT_CONTEXT_LENGTH = 131072
_LLAVA_TEXT_TOKEN_RESERVE = 2048
_LLAVA_EDGE_TOKEN_ESTIMATES: tuple[tuple[int, int], ...] = (
    (768, 5967),
    (640, 4500),
    (512, 3300),
    (448, 2600),
    (384, 1800),
    (336, 1500),
)


def _target_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _target_device_str() -> str:
    dev = _target_device()
    return f"cuda:{dev.index or 0}" if dev.type == "cuda" else str(dev)


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


def _is_llava_pixel_limit_model(model_id: str) -> bool:
    model_id_lower = model_id.lower()
    if "llava-onevision" in model_id_lower:
        return False
    return "llava-next" in model_id_lower


def _vl_pixel_factor(model_id: str) -> int | None:
    model_id_lower = model_id.lower()
    if "qwen3-vl" in model_id_lower:
        return 32
    if "qwen2.5-vl" in model_id_lower or "qwen2-vl" in model_id_lower:
        return 28
    return None


def model_uses_vl_pixel_limits(model_id: str) -> bool:
    return _is_llava_pixel_limit_model(model_id) or _vl_pixel_factor(model_id) is not None


def _llava_max_edge_for_frames(
    num_frames: int,
    *,
    context_length: int = _LLAVA_DEFAULT_CONTEXT_LENGTH,
) -> int:
    tokens_per_frame_budget = (context_length - _LLAVA_TEXT_TOKEN_RESERVE) // max(1, num_frames)
    for edge, est_tokens in _LLAVA_EDGE_TOKEN_ESTIMATES:
        if est_tokens <= tokens_per_frame_budget:
            return edge
    return _LLAVA_EDGE_TOKEN_ESTIMATES[-1][0]


def _apply_processor_pixel_limits(
    processor,
    model_id: str,
    *,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    num_frames: int | None = None,
) -> None:
    if _is_llava_pixel_limit_model(model_id):
        if max_pixels is not None:
            max_edge = int(max_pixels)
        elif num_frames is not None and num_frames > 0:
            max_edge = _llava_max_edge_for_frames(num_frames)
        else:
            max_edge = _LLAVA_DEFAULT_MAX_IMAGE_EDGE
        processor._finetune_max_image_edge = max_edge
        return

    factor = _vl_pixel_factor(model_id)
    if factor is None:
        return

    max_px = max_pixels if max_pixels is not None else _QWEN_VL_OFFICIAL_MAX_PIXELS
    min_px = min_pixels if min_pixels is not None else _QWEN_VL_OFFICIAL_MIN_PIXELS
    size = {"longest_edge": max_px, "shortest_edge": min_px}

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.size = size


def _maybe_resize_vlm_frames(frames: list[Image.Image], processor) -> list[Image.Image]:
    max_edge = getattr(processor, "_finetune_max_image_edge", None)
    if max_edge is None:
        return frames

    resized: list[Image.Image] = []
    scaled_count = 0
    for frame in frames:
        width, height = frame.size
        longest = max(width, height)
        if longest <= max_edge:
            resized.append(frame)
            continue
        scale = max_edge / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized.append(frame.resize(new_size, Image.Resampling.BICUBIC))
        scaled_count += 1
    if scaled_count:
        print(
            "[perf-debug][pixel-limits] "
            f"llava_max_image_edge={max_edge}, scaled_frames={scaled_count}/{len(frames)}",
            flush=True,
        )
    return resized


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_llava_video_qwen_model_id(model_id: str) -> bool:
    lowered = model_id.lower()
    return "llava-video-7b-qwen2" in lowered or (
        "lmms-lab" in lowered and "llava-video" in lowered and "qwen2" in lowered
    )


def _hub_id_from_hf_cache_path(model_path: str) -> str | None:
    for part in Path(model_path).parts:
        if not part.startswith("models--"):
            continue
        org, sep, name = part[len("models--") :].partition("--")
        if sep and org and name:
            return f"{org}/{name}"
    return None


def _resolve_llava_vid_pretrained(pretrained: str) -> str:
    """Use a hub-style id so llava's get_model_name_from_path detects qwen/llava."""
    expanded = os.path.expanduser(pretrained)
    hub_id = _hub_id_from_hf_cache_path(expanded)
    if hub_id is not None:
        return hub_id
    return expanded


def _is_vl_model_id(model_id: str) -> bool:
    model_id_lower = model_id.lower()
    vl_hints = (
        "qwen3-vl",
        "qwen2.5-vl",
        "qwen2-vl",
        "llava-onevision",
        "llava-next",
        "llava",
        "openvla",
        "internvl",
        "-vl-",
        "-vl/",
    )
    return any(h in model_id_lower for h in vl_hints)


def resolve_lmms_model_name(model_id: str) -> str:
    """将 HuggingFace 模型路径映射到 lmms-eval 注册的 model id。"""
    lowered = model_id.lower()
    if _is_llava_video_qwen_model_id(model_id):
        return "llava_vid"
    if "llava-onevision" in lowered or "llava_onevision" in lowered:
        # HF 官方权重走 transformers llava_hf；仅 lmms-lab 旧版走 llava 包
        if "llava-hf" in lowered:
            return "llava_hf"
        if "lmms-lab" in lowered:
            return "llava_onevision"
        return "llava_hf"
    if "llava-next" in lowered or "llava_next" in lowered:
        return "llava_hf"
    if "llava" in lowered and "video" in lowered:
        return "llava_vid"
    if "qwen3-vl" in lowered or "qwen3_vl" in lowered:
        return "qwen3_vl"
    if "qwen2.5-vl" in lowered or "qwen2_5-vl" in lowered:
        return "qwen2_5_vl"
    if "qwen2-vl" in lowered:
        return "qwen2_vl"
    return "qwen2_5_vl"


def _qwen_pixel_budget(model_id: str, apply_pixel_limits: bool) -> tuple[int | None, int | None]:
    _ = apply_pixel_limits
    if _vl_pixel_factor(model_id) is None:
        return None, None
    return _QWEN_VL_OFFICIAL_MIN_PIXELS, _QWEN_VL_OFFICIAL_MAX_PIXELS


class LlavaLegacyProcessor:
    """LLaVA-NeXT（llava_vid / llava_onevision）tokenizer + image_processor 包装。"""

    is_llava_video_qwen = True

    def __init__(self, tokenizer, image_processor, conv_template: str = "chatml_direct"):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_template = conv_template

    def batch_decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ):
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def apply_chat_template(self, *args, **kwargs):
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(*args, **kwargs)
        raise AttributeError("tokenizer 不支持 apply_chat_template")


def is_llava_video_qwen_inference(processor, model=None) -> bool:
    return getattr(processor, "is_llava_video_qwen", False)


def is_llava_onevision_video_inference(processor, model=None) -> bool:
    if is_llava_video_qwen_inference(processor, model=model):
        return False
    if type(processor).__name__ == "LlavaOnevisionProcessor":
        return True
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type == "llava_onevision"


def is_llava_next_video_inference(processor, model=None) -> bool:
    if is_llava_video_qwen_inference(processor, model=model):
        return False
    if type(processor).__name__ == "LlavaNextVideoProcessor":
        return True
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type in ("llava_next_video", "llava_next_video2")


def is_llava_hf_video_inference(processor, model=None) -> bool:
    return is_llava_next_video_inference(processor, model=model) or is_llava_onevision_video_inference(
        processor, model=model
    )


def is_llava_hf_inference(processor, model=None) -> bool:
    if is_llava_video_qwen_inference(processor, model=model) or isinstance(processor, LlavaLegacyProcessor):
        return False
    processor_name = type(processor).__name__
    if processor_name in {
        "LlavaProcessor",
        "LlavaNextProcessor",
        "LlavaOnevisionProcessor",
        "LlavaNextVideoProcessor",
    }:
        return True
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type in {"llava", "llava_next", "llava_onevision", "llava_next_video", "llava_next_video2"}


def is_qwen_vl_inference(processor, model=None) -> bool:
    if is_llava_video_qwen_inference(processor, model=model):
        return False
    if is_llava_hf_video_inference(processor, model=model):
        return False
    processor_name = type(processor).__name__
    if "Qwen" in processor_name and "VL" in processor_name:
        return True
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}


def _build_lmms_init_kwargs(
    lmms_model_name: str,
    pretrained: str,
    *,
    apply_pixel_limits: bool,
    num_frames: int | None,
) -> dict[str, Any]:
    device = _target_device_str()
    kwargs: dict[str, Any] = {
        "pretrained": pretrained,
        "device": device,
        "device_map": device,
        "batch_size": 1,
    }
    min_px, max_px = _qwen_pixel_budget(pretrained, apply_pixel_limits)
    if lmms_model_name in {"qwen2_5_vl", "qwen3_vl", "qwen2_vl"}:
        if min_px is not None:
            kwargs["min_pixels"] = min_px
        if max_px is not None:
            kwargs["max_pixels"] = max_px
        kwargs["system_prompt"] = _QWEN_VL_OFFICIAL_SYSTEM_PROMPT
        if num_frames is not None:
            kwargs["max_num_frames"] = num_frames
    elif lmms_model_name == "llava_hf":
        if num_frames is not None:
            kwargs["max_frames_num"] = num_frames
        kwargs["trust_remote_code"] = True
    elif lmms_model_name == "llava_vid":
        if num_frames is not None:
            kwargs["max_frames_num"] = num_frames
        kwargs["delay_load"] = True
        kwargs["overwrite"] = True
        kwargs["torch_dtype"] = "bfloat16"
        kwargs["conv_template"] = "chatml_direct"
    elif lmms_model_name == "llava_onevision":
        if num_frames is not None:
            kwargs["max_frames_num"] = num_frames
    return kwargs


def _normalize_model_processor(lmms_wrapper, lmms_model_name: str):
    model = lmms_wrapper.model
    if hasattr(lmms_wrapper, "processor"):
        processor = lmms_wrapper.processor
    elif lmms_model_name == "llava_hf" and hasattr(lmms_wrapper, "_image_processor"):
        # lmms-eval llava_hf 将 AutoProcessor 存在 _image_processor
        processor = lmms_wrapper._image_processor
    elif hasattr(lmms_wrapper, "_image_processor"):
        conv = getattr(lmms_wrapper, "conv_template", "chatml_direct")
        processor = LlavaLegacyProcessor(lmms_wrapper.tokenizer, lmms_wrapper._image_processor, conv_template=conv)
        if lmms_model_name == "llava_vid":
            processor.is_llava_video_qwen = True
        else:
            processor.is_llava_video_qwen = False
    else:
        raise RuntimeError(f"无法从 lmms-eval 模型 {lmms_model_name!r} 提取 processor")
    return model, processor


def load_model_and_processor(
    model_path: str,
    use_lora: bool = False,
    base_model: str | None = None,
    merge_lora: bool = False,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    apply_pixel_limits: bool = False,
    num_frames: int | None = None,
):
    model_path = os.path.expanduser(model_path)
    pixel_limit_id = os.path.expanduser(base_model) if (use_lora and base_model) else model_path
    pretrained = os.path.expanduser(base_model) if (use_lora and base_model) else model_path

    if use_lora and _is_llava_video_qwen_model_id(pretrained):
        raise ValueError("lmms-lab/LLaVA-Video-7B-Qwen2 暂不支持 LoRA 评测。")

    lmms_model_name = resolve_lmms_model_name(pretrained)
    if lmms_model_name == "llava_vid":
        pretrained = _resolve_llava_vid_pretrained(pretrained)
    print(
        "[perf-debug][load-lmms] "
        f"model_path={model_path}, lmms_model={lmms_model_name}, pretrained={pretrained}, "
        f"use_lora={use_lora}, merge_lora={merge_lora}",
        flush=True,
    )

    model_cls = get_model(lmms_model_name, force_simple=True)
    init_kwargs = _build_lmms_init_kwargs(
        lmms_model_name,
        pretrained,
        apply_pixel_limits=apply_pixel_limits,
        num_frames=num_frames,
    )
    if max_pixels is not None:
        init_kwargs["max_pixels"] = max_pixels
    if min_pixels is not None:
        init_kwargs["min_pixels"] = min_pixels

    lmms_wrapper = model_cls(**init_kwargs)
    model, processor = _normalize_model_processor(lmms_wrapper, lmms_model_name)

    if use_lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_path)
        if merge_lora:
            model = model.merge_and_unload()
        proc_src = pretrained
        if os.path.isfile(os.path.join(model_path, "preprocessor_config.json")) or os.path.isfile(
            os.path.join(model_path, "tokenizer_config.json")
        ):
            proc_src = model_path
        if lmms_model_name in {"qwen2_5_vl", "qwen3_vl", "qwen2_vl"} and hasattr(lmms_wrapper, "processor"):
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)

    model.eval()
    if apply_pixel_limits and lmms_model_name == "llava_hf":
        _apply_processor_pixel_limits(
            processor,
            pixel_limit_id,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            num_frames=num_frames,
        )
    elif lmms_model_name in {"qwen2_5_vl", "qwen3_vl", "qwen2_vl"}:
        _apply_processor_pixel_limits(
            processor,
            pixel_limit_id,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            num_frames=num_frames,
        )

    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    print(
        "[perf-debug][model] "
        f"dtype={getattr(model, 'dtype', None)}, attn_impl={attn_impl}, "
        f"device={next(model.parameters()).device}, lmms_model={lmms_model_name}",
        flush=True,
    )
    return model, processor


def load_text_model_and_processor(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.expanduser(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(_target_device())
    model.eval()
    return model, processor


def load_keyword_model_and_processor(model_path: str):
    if _is_vl_model_id(model_path):
        return load_model_and_processor(model_path)
    return load_text_model_and_processor(model_path)


def build_vlm_user_messages(
    frames: list[Image.Image],
    prompt: str,
    processor,
    *,
    model=None,
) -> list[dict[str, Any]]:
    if is_llava_hf_inference(processor, model=model):
        context = prompt
        if frames and _LLAVA_HF_VIDEO_TOKEN not in context and _LLAVA_HF_IMAGE_TOKEN not in context:
            token = _LLAVA_HF_VIDEO_TOKEN if is_llava_hf_video_inference(processor, model=model) else _LLAVA_HF_IMAGE_TOKEN
            prefix = " ".join([token] * len(frames)) if not is_llava_hf_video_inference(processor, model=model) else token
            context = f"{prefix}\n{context}"
        return [{"role": "user", "content": context}]
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    user_message = {"role": "user", "content": content}
    if is_qwen_vl_inference(processor, model=model):
        return [
            {"role": "system", "content": _QWEN_VL_OFFICIAL_SYSTEM_PROMPT},
            user_message,
        ]
    return [user_message]


def prepare_vlm_inputs(
    processor,
    frames: list[Image.Image],
    prompt: str,
    *,
    model=None,
) -> tuple[Any, str]:
    frames = _maybe_resize_vlm_frames(frames, processor)
    messages = build_vlm_user_messages(frames, prompt, processor, model=model)
    if is_llava_hf_inference(processor, model=model):
        tokenizer = processor.tokenizer
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if is_llava_hf_video_inference(processor, model=model) and frames:
            inputs = processor(videos=[frames], text=text, return_tensors="pt")
        elif frames:
            inputs = processor(images=frames, text=text, return_tensors="pt")
        else:
            inputs = processor(text=text, return_tensors="pt")
        return inputs, text

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt")
    return inputs, text


def _generate_response_llava_hf(
    model,
    processor,
    frames: list[Image.Image],
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, float, int, bool]:
    """与 lmms_eval.models.simple.llava_hf.generate_until 对齐。"""
    import time

    tokenizer = processor.tokenizer
    messages = build_vlm_user_messages(frames, prompt, processor, model=model)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t_proc_start = time.time()
    if is_llava_hf_video_inference(processor, model=model) and frames:
        inputs = processor(videos=[frames], text=text, return_tensors="pt")
    elif frames:
        inputs = processor(images=frames, text=text, return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")
    inputs = inputs.to(model.device, model.dtype)
    processor_time = time.time() - t_proc_start
    prompt_len = int(inputs["input_ids"].shape[-1])

    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = generated_ids[:, prompt_len:]
    inference_time = time.time() - start_time

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_token_count = int(generated_ids.shape[1]) if generated_ids.numel() else 0
    hit_max_tokens = generated_token_count >= max_new_tokens
    print(
        "[perf-debug][timing-split] "
        f"processor={processor_time:.2f}s, generate={inference_time:.2f}s",
        flush=True,
    )
    del inputs, generated_ids
    release_cuda_memory()
    return response, inference_time, generated_token_count, hit_max_tokens


def _generate_response_llava_video_qwen(
    model,
    processor: LlavaLegacyProcessor,
    frames: list[Image.Image],
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, float, int, bool]:
    import time

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

    t_proc_start = time.time()
    videos = []
    if frames:
        pixel_values = processor.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(model.device, dtype=model.dtype)
        videos.append(pixel_values)

    qs = prompt
    if videos:
        qs = DEFAULT_IMAGE_TOKEN * len(videos) + "\n" + qs

    conv = conv_templates[processor.conv_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text,
        processor.tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    pad_token_id = processor.tokenizer.pad_token_id or 151643
    attention_mask = input_ids.ne(pad_token_id).long()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], processor.tokenizer, input_ids)
    processor_time = time.time() - t_proc_start
    prompt_len = int(input_ids.shape[1])

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            inputs=input_ids,
            images=videos if videos else None,
            attention_mask=attention_mask,
            modalities="video" if videos else None,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    inference_time = time.time() - start_time

    if output_ids.shape[1] > prompt_len:
        generated_ids_trimmed = output_ids[:, prompt_len:]
    else:
        generated_ids_trimmed = output_ids
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    generated_token_count = int(generated_ids_trimmed.shape[1]) if generated_ids_trimmed.numel() else 0
    hit_max_tokens = generated_token_count >= max_new_tokens
    print(
        "[perf-debug][timing-split] "
        f"processor={processor_time:.2f}s, generate={inference_time:.2f}s",
        flush=True,
    )
    return response, inference_time, generated_token_count, hit_max_tokens


def generate_response(
    model,
    processor,
    frames: list[Image.Image],
    prompt: str,
    max_new_tokens: int = 1024,
) -> tuple[str, float, int, bool]:
    import time

    if is_llava_video_qwen_inference(processor, model=model):
        return _generate_response_llava_video_qwen(
            model,
            processor,
            frames,
            prompt,
            max_new_tokens,
        )
    if is_llava_hf_inference(processor, model=model):
        return _generate_response_llava_hf(
            model,
            processor,
            frames,
            prompt,
            max_new_tokens,
        )

    t_proc_start = time.time()
    inputs, _ = prepare_vlm_inputs(processor, frames, prompt, model=model)
    inputs = inputs.to(model.device)
    processor_time = time.time() - t_proc_start

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
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
    print(
        "[perf-debug][timing-split] "
        f"processor={processor_time:.2f}s, generate={inference_time:.2f}s",
        flush=True,
    )
    del inputs, generated_ids
    release_cuda_memory()
    return response, inference_time, generated_token_count, hit_max_tokens
