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


def _is_llava_video_qwen_model_id(model_id: str) -> bool:
    lowered = model_id.lower()
    return "llava-video-7b-qwen2" in lowered or (
        "lmms-lab" in lowered and "llava-video" in lowered and "qwen2" in lowered
    )


def _patch_llava_transformers_compat() -> None:
    """LLaVA-NeXT 面向 transformers 4.x；在 5.x 下补齐少量 API 差异。"""
    import transformers.modeling_utils as modeling_utils
    from transformers.configuration_utils import PretrainedConfig
    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 for h in already_pruned_heads if h < head)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    # llava_qwen.__init__ 里 config.rope_scaling = None 会在 tf5 清空 rope_parameters。
    if not getattr(PretrainedConfig, "_finetune_rope_none_patch", False):
        _orig_rope_setter = PretrainedConfig.rope_scaling.fset

        def _rope_scaling_setter(self, value):
            if value is None and getattr(self, "rope_parameters", None):
                return
            _orig_rope_setter(self, value)

        PretrainedConfig.rope_scaling = property(PretrainedConfig.rope_scaling.fget, _rope_scaling_setter)
        PretrainedConfig._finetune_rope_none_patch = True

    # llava 自带 SigLipVisionConfig.from_pretrained 仍调用已移除的 _set_token_in_kwargs。
    from llava.model.multimodal_encoder import siglip_encoder

    if not hasattr(siglip_encoder.SigLipVisionConfig, "_set_token_in_kwargs"):
        siglip_encoder.SigLipVisionConfig._set_token_in_kwargs = classmethod(lambda cls, kwargs: None)


def _llava_load_device_map() -> str | None:
    device = _target_device()
    if device.type == "cuda":
        return f"cuda:{device.index or 0}"
    return None


def _build_llava_video_qwen_overwrite_config() -> dict[str, Any]:
    """对齐 AKS llava_vid.py；delay_load 避免 transformers 5 meta-init 嵌套加载 vision tower。"""
    return {
        "mm_spatial_pool_stride": 2,
        "mm_spatial_pool_mode": "average",
        "mm_pooling_position": "before",
        "mm_newline_position": "grid",
        "delay_load": True,
        # checkpoint 的 mm_tunable_parts 含 mm_vision_tower 时会在 __init__ 强加载 vision，触发 meta device 报错。
        "mm_tunable_parts": "mm_mlp_adapter,mm_language_model",
    }


def _finalize_llava_video_qwen_model(model):
    target = _target_device()
    model = model.to(target)
    vision_tower = model.get_vision_tower()
    if vision_tower.is_loaded and getattr(vision_tower, "vision_tower", None) is not None:
        vision_tower.vision_tower = vision_tower.vision_tower.to(device=target, dtype=model.dtype)
    return model


class LlavaVideoQwenProcessor:
    """lmms-lab/LLaVA-Video-7B-Qwen2 原始权重推理包装（依赖 LLaVA-NeXT 包）。"""

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


def is_llava_video_qwen_inference(processor, model=None) -> bool:
    return getattr(processor, "is_llava_video_qwen", False)


def is_llava_next_video_inference(processor, model=None) -> bool:
    """LLaVA-NeXT-Video HF 权重需走单路 video 输入，不能与 Qwen / OneVision 共用多图路径。"""
    if is_llava_video_qwen_inference(processor, model=model):
        return False
    if type(processor).__name__ == "LlavaNextVideoProcessor":
        return True
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type in ("llava_next_video", "llava_next_video2")


def _is_vl_model_id(model_id: str) -> bool:
    model_id_lower = model_id.lower()
    if _use_generic_vl_loader(model_id):
        return True
    vl_hints = (
        "qwen2.5-vl",
        "qwen2-vl",
        "llava",
        "openvla",
        "internvl",
        "-vl-",
        "-vl/",
    )
    return any(h in model_id_lower for h in vl_hints)


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
    for token in ("<video>",):
        tid = tok.convert_tokens_to_ids(token)
        if tid is not None and tid != tok.unk_token_id and tid not in ids:
            ids.append(tid)
    return ids


def build_vlm_user_messages(
    frames: list[Image.Image],
    prompt: str,
    processor,
    *,
    model=None,
) -> list[dict[str, Any]]:
    if is_llava_next_video_inference(processor, model=model):
        content: list[dict[str, Any]] = []
        if frames:
            content.append({"type": "video", "video": frames})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def prepare_vlm_inputs(
    processor,
    frames: list[Image.Image],
    prompt: str,
    *,
    model=None,
) -> tuple[Any, str]:
    messages = build_vlm_user_messages(frames, prompt, processor, model=model)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if is_llava_next_video_inference(processor, model=model):
        proc_kwargs: dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
        }
        if frames:
            proc_kwargs["videos"] = [frames]
        inputs = processor(**proc_kwargs)
    else:
        inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt")
    return inputs, text


def _log_vision_inputs(
    processor,
    inputs: Any,
    frames: list[Image.Image],
    *,
    model=None,
) -> None:
    llava_video = is_llava_next_video_inference(processor, model=model)
    print(
        f"[perf-debug][vision-input] frames_count={len(frames)}, "
        f"llava_next_video={llava_video}, input_keys={sorted(list(inputs.keys()))}",
        flush=True,
    )
    pixel_values = inputs.get("pixel_values")
    pixel_values_videos = inputs.get("pixel_values_videos")
    if pixel_values is not None:
        print(
            "[perf-debug][vision-input] "
            f"pixel_values.shape={tuple(pixel_values.shape)}, pixel_values.dtype={pixel_values.dtype}",
            flush=True,
        )
    else:
        print("[perf-debug][vision-input] pixel_values=None", flush=True)
    if pixel_values_videos is not None:
        print(
            "[perf-debug][vision-input] "
            f"pixel_values_videos.shape={tuple(pixel_values_videos.shape)}, "
            f"pixel_values_videos.dtype={pixel_values_videos.dtype}",
            flush=True,
        )
    else:
        print("[perf-debug][vision-input] pixel_values_videos=None", flush=True)

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
        f"input_ids.shape={tuple(inputs['input_ids'].shape)}, "
        f"attention_mask.shape={tuple(inputs['attention_mask'].shape)}",
        flush=True,
    )


def build_mcq_prompt(question: str, options: list[str]) -> str:
    from utils import build_user_text

    return build_user_text(question, options)


# Qwen VL 官方推荐：单图/单帧视觉 token 预算 256–1280（经 spatial merge 后）
_DEFAULT_MAX_VISUAL_TOKENS = 1280
_DEFAULT_MIN_VISUAL_TOKENS = 256


def _vl_pixel_factor(model_id: str) -> int | None:
    model_id_lower = model_id.lower()
    if "qwen3-vl" in model_id_lower:
        return 32
    if "qwen2.5-vl" in model_id_lower or "qwen2-vl" in model_id_lower:
        return 28
    return None


def _apply_processor_pixel_limits(
    processor,
    model_id: str,
    *,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
) -> None:
    """限制 Qwen VL processor 单帧/单图像素，避免超高分辨率视频帧 OOM。"""
    factor = _vl_pixel_factor(model_id)
    if factor is None:
        return

    max_px = (
        max_pixels
        if max_pixels is not None
        else _DEFAULT_MAX_VISUAL_TOKENS * factor * factor
    )
    min_px = (
        min_pixels
        if min_pixels is not None
        else _DEFAULT_MIN_VISUAL_TOKENS * factor * factor
    )
    size = {"longest_edge": max_px, "shortest_edge": min_px}

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.size = size


def _log_processor_pixel_config(processor) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        print(
            f"[perf-debug][processor] processor={type(processor).__name__}, image_processor=None",
            flush=True,
        )
        return

    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        longest_edge = size.get("longest_edge")
        shortest_edge = size.get("shortest_edge")
    elif size is not None:
        longest_edge = getattr(size, "longest_edge", None)
        shortest_edge = getattr(size, "shortest_edge", None)
    else:
        longest_edge = None
        shortest_edge = None
    min_pixels = getattr(image_processor, "min_pixels", None)
    max_pixels = getattr(image_processor, "max_pixels", None)
    print(
        "[perf-debug][processor] "
        f"processor={type(processor).__name__}, "
        f"size.longest_edge={longest_edge}, size.shortest_edge={shortest_edge}, "
        f"min_pixels={min_pixels}, max_pixels={max_pixels}",
        flush=True,
    )


def load_model_and_processor(
    model_path: str,
    use_lora: bool = False,
    base_model: str | None = None,
    merge_lora: bool = False,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    apply_pixel_limits: bool = False,
):
    model_path = os.path.expanduser(model_path)
    print(
        "[perf-debug][load] "
        f"model_path={model_path}, use_lora={use_lora}, base_model={base_model}, merge_lora={merge_lora}",
        flush=True,
    )
    if _is_llava_video_qwen_model_id(model_path):
        if use_lora:
            raise ValueError("lmms-lab/LLaVA-Video-7B-Qwen2 暂不支持 LoRA 评测。")
        return load_llava_video_qwen_model_and_processor(model_path)
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
        if apply_pixel_limits:
            _apply_processor_pixel_limits(
                processor,
                base_id,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
            )
        _log_processor_pixel_config(processor)
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
        if apply_pixel_limits:
            _apply_processor_pixel_limits(
                processor,
                model_path,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
            )
        _log_processor_pixel_config(processor)
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
    if use_lora and base_model:
        pixel_limit_model_id = os.path.expanduser(base_model)
    elif use_lora:
        pixel_limit_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    else:
        pixel_limit_model_id = model_path
    if apply_pixel_limits:
        _apply_processor_pixel_limits(
            processor,
            pixel_limit_model_id,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
    _log_processor_pixel_config(processor)
    return model, processor


def load_llava_video_qwen_model_and_processor(model_path: str):
    """加载 lmms-lab/LLaVA-Video-7B-Qwen2，用法对齐 AKS evaluation/llava_vid.py。"""
    try:
        _patch_llava_transformers_compat()
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.model.language_model.llava_qwen import LlavaQwenConfig
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError(
            "lmms-lab/LLaVA-Video-7B-Qwen2 需要安装 LLaVA-NeXT："
            "pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
        ) from exc

    model_path = os.path.expanduser(model_path)
    AutoConfig.register("llava_qwen", LlavaQwenConfig)
    # HF snapshot 目录名不含 qwen，必须显式传 llava_qwen，否则 builder 走错分支。
    model_name = "llava_qwen" if "qwen" in model_path.lower() else get_model_name_from_path(model_path)
    device_map = _llava_load_device_map()
    overwrite_config = _build_llava_video_qwen_overwrite_config()
    attn_implementation = "sdpa" if torch.__version__ >= "2.1.2" else "eager"
    print(
        "[perf-debug][load-llava-video] "
        f"model_path={model_path}, model_name={model_name}, device_map={device_map}, "
        f"conv_template=chatml_direct, attn_implementation={attn_implementation}",
        flush=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map=device_map,
        torch_dtype="bfloat16",
        overwrite_config=overwrite_config,
        attn_implementation=attn_implementation,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643
    model = _finalize_llava_video_qwen_model(model)
    model.eval()
    processor = LlavaVideoQwenProcessor(tokenizer, image_processor)
    _ = conv_templates[processor.conv_template]
    _ = tokenizer_image_token
    _ = IMAGE_TOKEN_INDEX
    print(
        "[perf-debug][model] "
        f"dtype={model.dtype}, device={next(model.parameters()).device}, "
        f"processor={type(processor).__name__}",
        flush=True,
    )
    return model, processor


def load_text_model_and_processor(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.expanduser(model_path)
    print(f"[perf-debug][load-text] model_path={model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = _move_model_to_target_device(model)
    model.eval()
    attn_impl = getattr(model.config, "_attn_implementation", None)
    print(
        "[perf-debug][model] "
        f"dtype={model.dtype}, attn_impl={attn_impl}, device={next(model.parameters()).device}",
        flush=True,
    )
    print(f"[perf-debug][processor] processor={type(processor).__name__}", flush=True)
    return model, processor


def load_keyword_model_and_processor(model_path: str):
    """关键词提取：VL 模型走 VL loader，纯文本模型走 CausalLM loader。"""
    if _is_vl_model_id(model_path):
        return load_model_and_processor(model_path)
    return load_text_model_and_processor(model_path)


def _generate_response_llava_video_qwen(
    model,
    processor: LlavaVideoQwenProcessor,
    frames: list[Image.Image],
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, float, int, bool]:
    import time

    _patch_llava_transformers_compat()
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
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 151643
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

    # LLaVA-Qwen 的 generate 走 inputs_embeds，返回的 output_ids 通常只有新生成 token，
    # 不能按 prompt_len 切片（否则会得到空串）。与 AKS llava_vid.py 一致，直接 decode 全序列。
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

    t_proc_start = time.time()
    inputs, _ = prepare_vlm_inputs(processor, frames, prompt, model=model)
    inputs = inputs.to(model.device)
    processor_time = time.time() - t_proc_start
    _log_vision_inputs(processor, inputs, frames, model=model)

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

    if is_llava_next_video_inference(processor, model=model) or is_llava_video_qwen_inference(processor, model=model):
        raise RuntimeError("LLaVA 视频模型不支持 split embedding 推理路径，请使用 generate_response。")
    model_inputs, _ = prepare_vlm_inputs(processor, frames, prompt, model=model)
    model_inputs = model_inputs.to(model.device)

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


def generate_response_with_split_embedding_detailed(
    model,
    processor,
    frames: list[Image.Image],
    prompt: str,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    import time

    model_call_start = time.perf_counter()
    if is_llava_next_video_inference(processor, model=model) or is_llava_video_qwen_inference(processor, model=model):
        raise RuntimeError("LLaVA 视频模型不支持 split embedding 推理路径，请使用 generate_response。")

    processor_start = time.perf_counter()
    model_inputs, _ = prepare_vlm_inputs(processor, frames, prompt, model=model)
    model_inputs = model_inputs.to(model.device)
    processor_time = time.perf_counter() - processor_start

    embedding_start = time.perf_counter()
    fused_embeds = _build_image_fused_embeds(model, model_inputs)
    embedding_build_time = time.perf_counter() - embedding_start
    if fused_embeds is None:
        raise RuntimeError("模型不支持 embedding 分开输入，或构建图像 embedding 失败。")

    infer_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=fused_embeds,
            attention_mask=model_inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    inference_time = time.perf_counter() - infer_start

    decode_start = time.perf_counter()
    generated_ids = outputs.sequences
    response = _decode_new_tokens(processor, model_inputs["input_ids"], generated_ids)
    decode_time = time.perf_counter() - decode_start

    generated_token_count = int(len(outputs.scores))
    hit_max_tokens = generated_token_count >= max_new_tokens
    total_model_call_time = time.perf_counter() - model_call_start

    return {
        "response": response,
        "generated_token_count": generated_token_count,
        "hit_max_tokens": hit_max_tokens,
        "timings": {
            "processor": processor_time,
            "embedding_build": embedding_build_time,
            "inference": inference_time,
            "decode": decode_time,
            "total_model_call": total_model_call_time,
        },
    }
