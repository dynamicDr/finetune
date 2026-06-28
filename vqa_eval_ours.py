from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from data_loaders import (
    should_apply_vl_pixel_limits,
    get_data_loader,
    list_supported_datasets,
    resolve_dataset_root,
)
from data_loaders.base import VQASample, sample_matches_task_filter
from model_response_mode import load_model_response_mode_config, parse_response_by_mode, resolve_model_mode
from utils import (
    avg as _avg,
    build_user_text,
    build_user_text_with_subtitles,
    collect_video_frames_at_fps,
    format_labeled_options,
    KEYWORD_CACHE_DURATION_SUFFIXES,
    KEYWORD_EXTRACTOR_PROVIDERS,
    calculate_mra,
    collect_subtitles_for_frame_ids as _collect_subtitles_for_sample,
    compute_accuracy_from_results as _compute_accuracy_from_results,
    compute_score_counts_for_csv as _compute_score_counts_for_csv,
    dump_verbose_round,
    from_pretrained_local_first,
    init_verbose_run_dir,
    keyword_cache_dataset_key,
    keyword_cache_file,
    keyword_cache_run_dir,
    load_keyword_cache_entry,
    load_preprocessed_candidate_frames,
    log_ours_eval_to_csv,
    ours_eval_csv_columns,
    ours_eval_csv_row,
    pool_positions_at_fps,
    resolve_keyword_cache_root,
    DEFAULT_KEYWORD_CACHE_DIR,
    resolve_preprocessed_clip_dir,
    sanitize_cache_component,
    save_keyword_cache_entry,
    write_verbose_frame_selection_manifest,
)
from lmms_eval_bridge import generate_response, load_keyword_model_and_processor, load_model_and_processor, resolve_lmms_model_name
from lmms_eval_official import (
    aggregate_lvb,
    aggregate_videomme,
    build_lvb_prompt,
    build_videomme_prompt,
    is_official_lvb_dataset,
    is_official_videomme_dataset,
    resolve_lvb_max_new_tokens,
    resolve_videomme_max_new_tokens,
    score_lvb,
    score_videomme,
)

MODE_MAX_NEW_TOKENS = {"thinking": 4086, "instruct": 128}
PREPROCESSED_CLIP_COMPATIBLE_METHODS = {"ours"}
_VISUAL_ENCODER_CACHE: dict[str, "VisualEncoder"] = {}
_OPENAI_CLIENT_CACHE: dict[tuple[str, str], Any] = {}
_KEYWORD_LOCAL_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
VERBOSE = True
VERBOSE_OUTPUT_DIR = Path(__file__).resolve().parent / "verbose_eval_ours"
_VERBOSE_RUN_DIR: Path | None = None


# ==================== 基础工具 ====================
def _log(msg: str) -> None:
    print(f"[vqa_eval_ours] {msg}", flush=True)


def _to_feature_tensor(features: Any) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(features, attr, None)
        if isinstance(v, torch.Tensor):
            return v[:, 0, :] if attr == "last_hidden_state" and v.ndim >= 2 else v
    if isinstance(features, (tuple, list)) and features and isinstance(features[0], torch.Tensor):
        return features[0]
    raise TypeError(f"无法转换特征类型: {type(features)!r}")


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


@dataclass(frozen=True)
class VisualEncoder:
    backend: str  # "clip" | "blip_itc"
    model_id: str
    processor: Any
    model: Any
    device: str


def _is_blip_model(model_id: str) -> bool:
    return "blip-itm" in model_id.strip().lower()


# ==================== 视觉编码器（CLIP / BLIP ITC 双塔 embedding） ====================
def _load_visual_encoder(model_id: str, device: str | None) -> VisualEncoder:
    d = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = f"{model_id}::{d}"
    if key in _VISUAL_ENCODER_CACHE:
        return _VISUAL_ENCODER_CACHE[key]

    if _is_blip_model(model_id):
        try:
            from transformers import BlipForImageTextRetrieval, BlipProcessor
        except ImportError as exc:
            raise ImportError("BLIP ITC 依赖缺失，请安装 transformers。") from exc
        processor = from_pretrained_local_first(BlipProcessor.from_pretrained, model_id, log=_log)
        model = from_pretrained_local_first(
            BlipForImageTextRetrieval.from_pretrained, model_id, log=_log
        ).to(d).eval()
        ve = VisualEncoder(backend="blip_itc", model_id=model_id, processor=processor, model=model, device=d)
        _log(f"loaded BLIP ITC visual encoder: model_id={model_id}, device={d}")
    else:
        processor = from_pretrained_local_first(AutoProcessor.from_pretrained, model_id, log=_log)
        model = from_pretrained_local_first(AutoModel.from_pretrained, model_id, log=_log).to(d).eval()
        ve = VisualEncoder(backend="clip", model_id=model_id, processor=processor, model=model, device=d)
        _log(f"loaded CLIP visual encoder: model_id={model_id}, device={d}")

    _VISUAL_ENCODER_CACHE[key] = ve
    return ve


def _load_clip(model_id: str, device: str | None) -> tuple[Any, Any, str]:
    ve = _load_visual_encoder(model_id, device)
    return ve.processor, ve.model, ve.device


def _encode_clip_images(images: list[Image.Image], proc: Any, model: Any, device: str, bs: int) -> torch.Tensor:
    outs = []
    for i in range(0, len(images), bs):
        inputs = proc(images=images[i:i + bs], return_tensors="pt").to(device)
        with torch.no_grad():
            feats = _to_feature_tensor(model.get_image_features(**inputs))
        outs.append(_norm(feats))
    return torch.cat(outs, dim=0)


def _encode_clip_texts(texts: list[str], proc: Any, model: Any, device: str, bs: int) -> torch.Tensor:
    outs = []
    for i in range(0, len(texts), bs):
        inputs = proc(text=texts[i:i + bs], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = _to_feature_tensor(model.get_text_features(**inputs))
        outs.append(_norm(feats))
    return torch.cat(outs, dim=0)


def _encode_blip_images(images: list[Image.Image], proc: Any, model: Any, device: str, bs: int) -> torch.Tensor:
    """BLIP ITC：vision_model + vision_proj，与文本无关，可跨关键词复用。"""
    outs = []
    for i in range(0, len(images), bs):
        batch = images[i:i + bs]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)
            image_embeds = vision_outputs.last_hidden_state
            image_feat = model.vision_proj(image_embeds[:, 0, :])
        outs.append(_norm(image_feat))
    return torch.cat(outs, dim=0)


def _encode_blip_texts(texts: list[str], proc: Any, model: Any, device: str, bs: int) -> torch.Tensor:
    """BLIP ITC：text_encoder + text_proj，与图像无关。"""
    outs = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        inputs = proc(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_output = model.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
            )
            text_embeds = model.text_proj(text_output.last_hidden_state[:, 0, :])
        outs.append(_norm(text_embeds))
    return torch.cat(outs, dim=0)


def _encode_images_for_encoder(ve: VisualEncoder, images: list[Image.Image], bs: int) -> torch.Tensor:
    if ve.backend == "blip_itc":
        return _encode_blip_images(images, ve.processor, ve.model, ve.device, bs)
    return _encode_clip_images(images, ve.processor, ve.model, ve.device, bs)


def _encode_texts_for_encoder(ve: VisualEncoder, texts: list[str], bs: int) -> torch.Tensor:
    if ve.backend == "blip_itc":
        return _encode_blip_texts(texts, ve.processor, ve.model, ve.device, bs)
    return _encode_clip_texts(texts, ve.processor, ve.model, ve.device, bs)


def _encode_texts_for_dedup(ve: VisualEncoder, texts: list[str], bs: int) -> torch.Tensor:
    return _encode_texts_for_encoder(ve, texts, bs)


def _compute_kw_frame_sims(
    ve: VisualEncoder,
    images: list[Image.Image],
    texts: list[str],
    bs: int,
    img_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    if not texts or not images:
        return torch.empty((len(texts), len(images)), device=ve.device)
    if img_emb is None:
        img_emb = _encode_images_for_encoder(ve, images, bs)
    kw_emb = _encode_texts_for_encoder(ve, texts, bs)
    return kw_emb @ img_emb.T


def _encode_images(images: list[Image.Image], proc: Any, model: Any, device: str, bs: int) -> torch.Tensor:
    return _encode_clip_images(images, proc, model, device, bs)


def _encode_texts(texts: list[str], proc: Any, model: Any, device: str, bs: int) -> torch.Tensor:
    return _encode_clip_texts(texts, proc, model, device, bs)


# ==================== 关键词抽取（仅 LLM） ====================
def _parse_visual_keyword_phrases(raw_text: str) -> list[str]:
    text = (raw_text or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            out = []
            for x in parsed:
                s = str(x).strip().strip('"').strip("'")
                if s:
                    out.append(s)
            if out:
                return out
    except Exception:
        pass

    quoted = re.findall(r'"([^"\n]{2,120})"', text)
    if quoted:
        return [q.strip() for q in quoted if q.strip()]

    out = []
    for line in text.splitlines():
        s = re.sub(r"^\s*[-*0-9.)]+\s*", "", line).strip().strip('"').strip("'")
        if s:
            out.append(s)
    return out


def _build_keyword_extraction_prompt(
    question: str,
    options: list[str] | None,
    target_keywords: int,
    prompt_version: int,
) -> str:
    target_keywords = max(1, int(target_keywords))
    options_text = format_labeled_options(options or [])
    if int(prompt_version) == 0:
        return (
            "You are a visual element extractor for video question answering.\n"
            "Task: From the question and options, extract all VISUALLY OBSERVABLE elements "
            "that could appear in video frames.\n\n"
            "Rules:\n"
            "- Extract ONLY things that can be SEEN in a video frame: objects, scenes, "
            "actions, or on-screen text.\n"
            "- Do NOT extract abstract concepts such as reasons, causes, meanings, "
            "intentions, or feelings.\n"
            "- Each element must be a short DECLARATIVE phrase, e.g. \"a red cat\", "
            "\"a railway under construction\", \"an ancient tomb being excavated\".\n"
            "- Do NOT output single bare words (e.g. \"cat\") or questions.\n"
            f"- Output around {target_keywords} keywords.\n"
            "- Output STRICTLY a JSON array of strings. No explanation, no markdown, "
            "no extra text.\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{options_text}"
        )
    if int(prompt_version) == 1:
        return (
            "You extract visual phrases to help CLIP retrieve relevant FRAMES from a video "
            "for answering a multiple-choice question.\n\n"
            "## Rules\n"
            "- Each phrase describes something visible in a SINGLE FRAME (not a process).\n"
            "- Each phrase is a concrete noun phrase, 2-7 words, e.g. \"a man holding a knife\".\n"
            "- Start with a visual ANCHOR from the question (the persistent subject/scene), "
            "then combine the anchor with each option's visual differentiator.\n"
            "- SKIP options that are abstract (reasons, feelings, intentions)\n\n"
            "## Avoid (CLIP is bad at these)\n"
            "- Negations (\"no hat\"), counting (\"five birds\"), spatial directions (\"on the left\")\n"
            "- Subtle emotions, abstract intent, process verbs (\"building\", \"being excavated\")\n"
            "- Bare single words (\"cat\" → use \"a black cat\")\n\n"
            "## Output\n"
            f"Output STRICTLY a JSON array of strings. No explanation, no markdown. Output no more than {target_keywords} keywords. "
            "## Examples\n"
            "Q: What is the man doing in the kitchen?\n"
            "Options: A.cutting vegetables  B.washing dishes  C.reading  D.talking on phone\n"
            "Output: [\"a man in a kitchen\", \"a man cutting vegetables\", "
            "\"a man washing dishes\", \"a man reading a book\", \"a man holding a phone\"]\n\n"
            "Q: Why is the child crying?\n"
            "Options: A.dropped ice cream  B.afraid of dog  C.tired  D.hurt knee\n"
            "Output: [\"a crying child\", \"ice cream on the ground\", "
            "\"a child near a dog\", \"a child with a hurt knee\"]\n"
            "(\"tired\" skipped — not visually distinguishable)\n\n"
            f"Q: {question}\n"
            f"Options:\n{options_text}\n"
            "Output:"
        )
    if int(prompt_version) == 2:
        return (
            "You extract visual phrases for CLIP-based frame retrieval in video multiple-choice QA.\n"
            f"Output ONE JSON array with  no more than {target_keywords} elements combining two parts:\n"
            "  (A) phrases from the QUESTION only\n"
            "  (B) exactly ONE phrase per OPTION (when the option has visual content)\n"
            "## Part A — Question phrases\n"
            "From the QUESTION only (ignore options for this step), extract some short visual phrases:\n"
            "- Subject, scene, and core action/object/event named in the question\n"
            "- Concrete, single-frame visual cues\n\n"
            "Do NOT copy option text into Part A.\n"
            "## Part B — One phrase per option\n"
            "For EACH option, output exactly ONE phrase that captures its distinguishing visual content:\n"
            "- One option → one phrase. Never split one option into multiple phrases or sub-stages.\n"
            "- Prefer the core visible object or action in that option; tie to the same subject/scene from the question when helpful.\n"
            "- If an option has no stable visual content (for example bare numbers, yes/no), skip that option in Part B:\n\n"
            "## Phrase style\n"
            "- Single-frame, visual cue.\n"
            "- Use phrases of more than two words whenever possible\n"
            "- You can add extra synonyms for question(part A),  but NOT for the same option(part B).\n"
            "## Final output\n"
            "- Merge Part A then Part B into a single JSON array of strings\n"
            "- No markdown, no labels, no explanation\n"
            f"- Output no more than {target_keywords} key words.\n"
            "## Example 1 — what/which (object in question)\n"
            "Q: What did the woman take out of the refrigerator?\n"
            "Options:\n"
            "A. dragon fruit\n"
            "B. eggplant\n"
            "C. water\n"
            "D. beer\n"
            "Part A: [\"a woman in front of a refrigerator\", \"an open refrigerator door\", \"a woman holding food\"]\n"
            "Part B: [\"dragon fruit\", \"eggplant\", \"water bottle\", \"beer bottle\"]\n"
            "Output:\n"
            "[\"a woman in front of a refrigerator\", \"an open refrigerator door\", \"a woman holding food\", \"dragon fruit\", \"eggplant\", \"water bottle\", \"beer bottle\"]\n"
            "## Example 2 — counting (numeric options skipped in Part B)\n"
            "Q: Throughout this video, what is the total count of occurrences for the scene featuring the 'pole vault' action?\n"
            "Options:\n"
            "A. 1\n"
            "B. 2\n"
            "C. 3\n\n"
            "Part A: [\"pole vault athlete over bar\", \"pole vault runway\", \"person clearing a crossbar\"]\n"
            "Part B: (all options are numbers — skip)\n"
            "Output:\n"
            "[\"pole vault athlete over bar\", \"pole vault runway\", \"person clearing a crossbar\"]\n"
            "---\n"
            "Now give the JSON for this one:\n"
            f"Q: {question}\n"
            f"Options:\n{options_text}\n"
            "Output:"
        )
    if int(prompt_version) == 3:
        return (
            "Extract visually observable elements from the question and options for CLIP frame retrieval.\n\n"
            "Rules:\n"
            "- Only things visible in a single video frame: objects, people, actions, scenes, on-screen text.\n"
            "- Short noun phrases; skip abstract or non-visual content (reasons, numbers-only options, yes/no).\n"
            f"- Output at most {target_keywords} items.\n\n"
            "Output STRICTLY a JSON array of strings. No markdown, no explanation.\n\n"
            f"Q: {question}\n"
            f"Options:\n{options_text}"
        )
    raise ValueError(f"keyword_prompt_version 仅支持 0、1、2 或 3，当前: {prompt_version}")


def _dedup_keyword_phrases(kws_raw: list[str]) -> list[str]:
    out_kws, seen = [], set()
    for kw in kws_raw:
        s = kw.strip().lower()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out_kws.append(s)
    return out_kws


def _resolve_keyword_extractor(
    extractor_model: str,
    api_base_url: str,
    api_key_env: str,
) -> dict[str, str]:
    """解析 keyword_extractor_model，例如 local-Qwen/... / poe-gpt-5.2 / aio-gpt-5.2 / or-openai/gpt-4o。"""
    raw = str(extractor_model or "local").strip()
    if not raw or raw.lower() == "local":
        return {"mode": "local", "raw": raw, "local_model_path": ""}

    lower = raw.lower()
    for sep in ("-", "_", ":"):
        token = f"local{sep}"
        if lower.startswith(token):
            local_model_path = raw[len(token) :].strip()
            if not local_model_path:
                raise ValueError(
                    f"keyword_extractor_model={raw!r} 缺少模型名，例如 local-Qwen/Qwen3-VL-4B-Instruct"
                )
            return {"mode": "local", "raw": raw, "local_model_path": local_model_path}

    provider: str | None = None
    remote_model = raw
    for prefix in KEYWORD_EXTRACTOR_PROVIDERS:
        for sep in ("-", "_", ":"):
            token = f"{prefix}{sep}"
            if lower.startswith(token):
                provider = prefix
                remote_model = raw[len(token) :].strip()
                break
        if provider is not None:
            break

    if provider is not None:
        if not remote_model:
            raise ValueError(f"keyword_extractor_model={raw!r} 缺少模型名，例如 {provider}-gpt-5.2")
        spec = KEYWORD_EXTRACTOR_PROVIDERS[provider]
        return {
            "mode": "remote",
            "raw": raw,
            "provider": provider,
            "remote_model": remote_model,
            "base_url": spec["base_url"],
            "api_key_env": spec["api_key_env"],
            "api_style": spec["api_style"],
        }

    return {
        "mode": "remote",
        "raw": raw,
        "provider": "",
        "remote_model": raw,
        "base_url": str(api_base_url or "").strip() or KEYWORD_EXTRACTOR_PROVIDERS["aio"]["base_url"],
        "api_key_env": str(api_key_env or "").strip() or KEYWORD_EXTRACTOR_PROVIDERS["aio"]["api_key_env"],
        "api_style": "chat",
    }


def _read_api_key(api_key_env: str) -> str:
    fallbacks: tuple[str, ...]
    if api_key_env == "POE_API_KEY":
        fallbacks = ("POE_API_KEY", "OPENAI_API_KEY")
    elif api_key_env == "AIOHUB_API_KEY":
        fallbacks = ("AIOHUB_API_KEY", "OPENAI_API_KEY")
    elif api_key_env == "OPENROUTER_API_KEY":
        fallbacks = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
    else:
        fallbacks = (api_key_env, "AIOHUB_API_KEY", "POE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
    for env in fallbacks:
        val = os.getenv(env)
        if val:
            return val
    env_hint = "、".join(fallbacks)
    raise RuntimeError(f"缺少关键词提取 API key：请设置 {env_hint}")


def _openai_chat_response_text(response: Any) -> str:
    """从 OpenAI Chat Completions（或兼容网关）响应中取出 assistant 文本。"""
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if choices:
        first = choices[0]
        msg = getattr(first, "message", None)
        if msg is None and isinstance(first, dict):
            msg = first.get("message")
        if msg is not None:
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if content is not None:
                return str(content).strip()
    return str(response).strip()


_KEYWORD_API_NO_PROXY_HOSTS = (
    "api.aiohub.org,aiohub.org,api.poe.com,openrouter.ai,localhost,127.0.0.1"
)


def _ensure_keyword_api_no_proxy_env() -> None:
    """关键词远程 API 直连：避免集群 HTTP_PROXY 导致 ProxyError 503。"""
    for key in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(key, "")
        parts = [p.strip() for p in cur.split(",") if p.strip()]
        for host in _KEYWORD_API_NO_PROXY_HOSTS.split(","):
            if host not in parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)


def _openrouter_default_headers() -> dict[str, str]:
    """OpenRouter 可选 attribution 头。"""
    headers: dict[str, str] = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.getenv("OPENROUTER_APP_TITLE", "vqa_eval_ours").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def _get_openai_compatible_client(api_key_env: str, base_url: str) -> Any:
    api_key = _read_api_key(api_key_env)
    cache_key = (api_key_env, base_url)
    if cache_key not in _OPENAI_CLIENT_CACHE:
        import certifi
        import httpx
        import openai

        _ensure_keyword_api_no_proxy_env()
        http_client = httpx.Client(
            verify=certifi.where(),
            timeout=httpx.Timeout(120.0, connect=30.0),
            trust_env=False,
        )
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "http_client": http_client,
        }
        if "openrouter.ai" in base_url:
            default_headers = _openrouter_default_headers()
            if default_headers:
                client_kwargs["default_headers"] = default_headers
        _OPENAI_CLIENT_CACHE[cache_key] = openai.OpenAI(**client_kwargs)
    return _OPENAI_CLIENT_CACHE[cache_key]


def _get_local_keyword_model_and_processor(model_path: str) -> tuple[Any, Any]:
    path = os.path.expanduser(str(model_path).strip())
    if path not in _KEYWORD_LOCAL_MODEL_CACHE:
        _log(f"加载关键词提取本地模型: {path}")
        _KEYWORD_LOCAL_MODEL_CACHE[path] = load_keyword_model_and_processor(path)
    return _KEYWORD_LOCAL_MODEL_CACHE[path]


def _generate_local_keyword_text(kw_model: Any, kw_proc: Any, prompt_text: str, max_new_tokens: int) -> str:
    if getattr(kw_proc, "image_processor", None) is not None:
        content = [{"type": "text", "text": prompt_text}]
        text = kw_proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = kw_proc(text=[text], padding=True, return_tensors="pt").to(kw_model.device)
        decode_kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    else:
        text = kw_proc.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = kw_proc(text, return_tensors="pt").to(kw_model.device)
        decode_kwargs = {"skip_special_tokens": True}
    with torch.no_grad():
        out = kw_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    seq = out[0][len(inputs.input_ids[0]):]
    return kw_proc.decode(seq, **decode_kwargs).strip()


def _call_remote_keyword_llm(
    client: Any,
    *,
    api_style: str,
    remote_model: str,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    if api_style == "responses":
        # Poe 官方：https://api.poe.com/v1 + client.responses.create(..., input=...)
        response = client.responses.create(model=remote_model, input=prompt_text)
        output_text = getattr(response, "output_text", None)
        if output_text is not None:
            return str(output_text).strip()
        return _openai_chat_response_text(response)
    response = client.chat.completions.create(
        model=remote_model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
        max_tokens=max_new_tokens,
    )
    return _openai_chat_response_text(response)


def _extract_keywords_with_llm_text(
    model: Any,
    proc: Any,
    question: str,
    options: list[str] | None,
    max_new_tokens: int,
    target_keywords: int,
    prompt_version: int,
    extractor_model: str,
    api_base_url: str,
    api_key_env: str,
) -> tuple[list[str], list[str]]:
    prompt = _build_keyword_extraction_prompt(question, options, target_keywords, prompt_version)
    extractor_cfg = _resolve_keyword_extractor(extractor_model, api_base_url, api_key_env)

    def _ask_once(prompt_text: str) -> tuple[list[str], list[str]]:
        if extractor_cfg["mode"] == "local":
            local_model_path = str(extractor_cfg.get("local_model_path") or "").strip()
            if local_model_path:
                kw_model, kw_proc = _get_local_keyword_model_and_processor(local_model_path)
            else:
                kw_model, kw_proc = model, proc
            resp = _generate_local_keyword_text(kw_model, kw_proc, prompt_text, max_new_tokens)
        else:
            client = _get_openai_compatible_client(
                api_key_env=extractor_cfg["api_key_env"],
                base_url=extractor_cfg["base_url"],
            )
            resp = _call_remote_keyword_llm(
                client,
                api_style=extractor_cfg["api_style"],
                remote_model=extractor_cfg["remote_model"],
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
            )
        kws_raw_local = _parse_visual_keyword_phrases(resp)
        return kws_raw_local, _dedup_keyword_phrases(kws_raw_local)

    kws_raw, out_kws = _ask_once(prompt)
    if out_kws:
        return kws_raw, out_kws

    retry_prompt = (
        prompt
        + "\n\nYour previous answer was empty or not parseable. Retry now.\n"
        + "You MUST output a valid JSON array of visual phrases and include at least 1 keyword."
    )
    return _ask_once(retry_prompt)


def _extract_keywords_with_optional_cache(
    model: Any,
    proc: Any,
    *,
    sample_id: str,
    dataset: str,
    task_type: str | None = None,
    question: str,
    options: list[str] | None,
    max_new_tokens: int,
    target_keywords: int,
    prompt_version: int,
    extractor_model: str,
    api_base_url: str,
    api_key_env: str,
    use_cache: bool,
    cache_dir: str | None,
    cache_number: int = 0,
    cache_write_on_miss: bool = False,
) -> tuple[list[str], list[str]]:
    cache_dataset_key = keyword_cache_dataset_key(dataset, task_type)
    cache_root = resolve_keyword_cache_root(cache_dir)
    run_dir = keyword_cache_run_dir(
        cache_root,
        dataset=dataset,
        task_type=task_type,
        extractor_model=extractor_model,
        prompt_version=prompt_version,
        target_keywords=target_keywords,
        cache_number=cache_number,
    )
    cache_path = keyword_cache_file(run_dir, sample_id)

    if use_cache:
        cached = load_keyword_cache_entry(cache_path, sample_id=sample_id, log_fn=_log)
        if cached is not None:
            kws_raw, kws = cached
            _log(f"sample={sample_id} 关键词缓存命中 ({cache_path})")
            return kws_raw, kws
        if not cache_write_on_miss:
            raise RuntimeError(
                f"sample={sample_id} 关键词缓存未命中 ({cache_path})；"
                "已启用 --use_keyword_cache 且未开启 --write_keyword_cache，仅读缓存、不调用关键词 API。"
                "请检查 keyword_cache_dir、keyword_extractor_model、keyword_prompt_version、"
                "max_keywords、keyword_cache_number 是否与生成缓存时一致。"
            )

    kws_raw, kws = _extract_keywords_with_llm_text(
        model=model,
        proc=proc,
        question=question,
        options=options,
        max_new_tokens=max_new_tokens,
        target_keywords=target_keywords,
        prompt_version=prompt_version,
        extractor_model=extractor_model,
        api_base_url=api_base_url,
        api_key_env=api_key_env,
    )
    if use_cache and cache_write_on_miss and kws:
        save_keyword_cache_entry(
            cache_path,
            sample_id=sample_id,
            dataset=cache_dataset_key,
            extractor_model=extractor_model,
            prompt_version=prompt_version,
            target_keywords=target_keywords,
            kws_raw=kws_raw,
            kws=kws,
        )
        _log(f"sample={sample_id} 关键词已写入缓存 ({cache_path})")
    return kws_raw, kws


# ==================== 信息量计算与预算分配 ====================
def _merge_keywords(kws: list[str], emb: torch.Tensor, max_keywords: int) -> tuple[list[str], torch.Tensor, dict[str, Any]]:
    """未超过 max_keywords 时不删减；超过时删除与其它词 CLIP 相似度最高的 (n - max_keywords) 个。"""
    n = len(kws)
    if n == 0:
        ids = torch.empty((0,), device=emb.device, dtype=torch.long)
        return [], emb[ids], {
            "max_keywords": int(max_keywords),
            "input_count": 0,
            "output_count": 0,
            "input_keywords": [],
            "kept_keywords": [],
            "removed_keywords": [],
            "per_keyword_max_similarity": [],
        }
    max_keywords = max(1, int(max_keywords))
    per_keyword_scores: list[dict[str, Any]] = []
    if n == 1:
        per_keyword_scores.append(
            {"keyword": kws[0], "max_similarity_to_others": 0.0, "most_similar_keyword": ""}
        )
    else:
        sim = emb @ emb.T
        sim.fill_diagonal_(-1e9)
        argmax_idx = sim.argmax(dim=1)
        max_sim_all = sim.max(dim=1).values
        for i, kw in enumerate(kws):
            best_j = int(argmax_idx[i].item())
            per_keyword_scores.append(
                {
                    "keyword": kw,
                    "max_similarity_to_others": float(max_sim_all[i].item()),
                    "most_similar_keyword": kws[best_j],
                }
            )

    if n <= max_keywords:
        ids = torch.arange(n, device=emb.device, dtype=torch.long)
        return kws, emb[ids], {
            "max_keywords": int(max_keywords),
            "input_count": int(n),
            "output_count": int(n),
            "input_keywords": list(kws),
            "kept_keywords": list(kws),
            "removed_keywords": [],
            "per_keyword_max_similarity": per_keyword_scores,
        }

    ranked = sorted(range(n), key=lambda i: per_keyword_scores[i]["max_similarity_to_others"], reverse=True)
    remove_ids = set(ranked[: n - max_keywords])
    keep = [i for i in range(n) if i not in remove_ids]
    removed_rows = [
        {
            "keyword": kws[i],
            "most_similar_keyword": per_keyword_scores[i]["most_similar_keyword"],
            "max_similarity_to_others": per_keyword_scores[i]["max_similarity_to_others"],
        }
        for i in ranked[: n - max_keywords]
    ]
    ids = torch.tensor(keep, device=emb.device, dtype=torch.long)
    kept_keywords = [kws[i] for i in keep]
    return kept_keywords, emb[ids], {
        "max_keywords": int(max_keywords),
        "input_count": int(n),
        "output_count": int(len(kept_keywords)),
        "input_keywords": list(kws),
        "kept_keywords": kept_keywords,
        "removed_keywords": removed_rows,
        "per_keyword_max_similarity": per_keyword_scores,
    }


def _allocate_counts_by_weights(weights: torch.Tensor, total: int) -> torch.Tensor:
    if total <= 0 or weights.numel() == 0:
        return torch.zeros((weights.numel(),), dtype=torch.long, device=weights.device)
    w = weights.float().clamp(min=0.0)
    s = float(w.sum().item())
    if s <= 1e-12:
        w = torch.ones_like(w) / max(1, w.numel())
    else:
        w = w / s
    raw = w * float(total)
    base = torch.floor(raw).to(torch.long)
    remain = int(total - int(base.sum().item()))
    if remain > 0:
        frac = raw - base.float()
        order = torch.argsort(frac, descending=True)
        base[order[:remain]] += 1
    return base


def _local_evidence_score(x: torch.Tensor) -> float:
    """计算一维相似度曲线的局部证据度：减中位数基线后取 Hoyer 稀疏度。"""
    x = x.detach().float().flatten()
    n = int(x.numel())
    if n <= 1:
        return 0.0

    x = x.clamp(min=0.0)
    x = (x - torch.median(x)).clamp(min=0.0)
    l2 = torch.sqrt(torch.sum(x * x))
    if float(l2.item()) <= 0.0:
        return 0.0

    l1 = torch.sum(x)
    denom = math.sqrt(float(n)) - 1.0
    if denom <= 0.0:
        return 0.0
    score = (math.sqrt(float(n)) - float((l1 / l2).item())) / denom
    return float(min(1.0, max(0.0, score)))


def _compute_keyword_information(
    kws_rep: list[str],
    kw_frame_sims: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """合并后的关键词全部保留；用局部证据度 φ 经幂律 w_i=φ_i^λ/Σφ_j^λ 得到关键词权重。"""
    m = len(kws_rep)
    n = int(kw_frame_sims.shape[1]) if kw_frame_sims.ndim == 2 else 0
    if m == 0 or n == 0:
        return {
            "kws_use": [],
            "kw_weights": torch.empty((0,), device=kw_frame_sims.device),
            "rows": [],
        }

    sims = kw_frame_sims
    local_evidence = torch.tensor(
        [_local_evidence_score(sims[i]) for i in range(m)],
        dtype=kw_frame_sims.dtype,
        device=kw_frame_sims.device,
    ).clamp(min=0.0, max=1.0)

    kws_use = list(kws_rep)
    kw_info = local_evidence
    lam = max(0.0, float(args.keyword_weight_strength))
    phi = kw_info.clamp(min=0.0)
    powered = phi.pow(lam)
    ps = float(powered.sum().item())
    kw_weights = (
        powered / ps
        if ps > 1e-12
        else torch.ones_like(phi) / float(max(1, phi.numel()))
    )

    rows = []
    for i, kw in enumerate(kws_rep):
        rows.append(
            {
                "keyword": kw,
                "local_evidence_score": float(local_evidence[i].item()),
                "info": float(kw_info[i].item()),
                "weight": float(kw_weights[i].item()) if kw_weights.numel() > i else 0.0,
                "kept": True,
            }
        )

    return {
        "kws_use": kws_use,
        "kw_weights": kw_weights,
        "local_evidence": local_evidence,
        "rows": rows,
    }


# ==================== VLM 单轮推理与打分 ====================
def _run_vlm_once(
    model: Any,
    proc: Any,
    frames: list[Image.Image],
    prompt: str,
    max_new_tokens: int,
    model_mode: str,
    *,
    has_options: bool,
) -> dict[str, Any]:
    """与 vqa_eval.py 对齐：generate_response + parse_response_by_mode，不做 A-D logit fallback。"""
    response, inference_time, _generated_token_count, hit_max_tokens = generate_response(
        model,
        proc,
        frames,
        prompt,
        max_new_tokens=max_new_tokens,
    )
    _, _, pred_answer = parse_response_by_mode(
        response=response,
        has_options=has_options,
        model_mode=model_mode,
    )
    pred_u = str(pred_answer).strip().upper() if pred_answer is not None else ""
    return {
        "pred_answer": pred_u,
        "response": response,
        "option_probs": {},
        "entropy": 0.0,
        "inference_time": float(inference_time),
        "hit_max_tokens": int(hit_max_tokens),
    }


def _select_keyword_frames(
    kw_frame_sims: torch.Tensor,
    kw_weights: torch.Tensor,
    budget: int,
) -> list[int]:
    if budget <= 0:
        return []
    n_frames = int(kw_frame_sims.shape[1])
    return _quota_topk_select(
        kw_sims=kw_frame_sims,
        kw_w=kw_weights,
        budget=budget,
        candidate_idx=list(range(n_frames)),
    )


def _quota_topk_select(
    kw_sims: torch.Tensor,
    kw_w: torch.Tensor,
    budget: int,
    candidate_idx: list[int],
) -> list[int]:
    """按关键词权重分配帧数；重复帧则按关键词权重顺序继续取该词的下一个 top 帧。"""
    if kw_sims.ndim != 2 or budget <= 0:
        return []
    m = int(kw_sims.shape[0])
    n = int(kw_sims.shape[1])
    if m == 0 or n == 0:
        return []

    if kw_w.numel() != m:
        kw_w = torch.ones((m,), device=kw_sims.device)
    kw_w = kw_w.float().clamp(min=0.0)
    if float(kw_w.sum().item()) <= 1e-12:
        kw_w = torch.ones_like(kw_w)

    cand = sorted({int(i) for i in candidate_idx if 0 <= int(i) < n})
    if len(cand) < min(budget, n):
        cand = list(range(n))
    cand_set = set(cand)
    max_pick = min(int(budget), len(cand))
    quotas = _allocate_counts_by_weights(kw_w, max_pick)
    keyword_order = [int(i) for i in torch.argsort(kw_w, descending=True).tolist()]
    ranked_by_keyword: list[list[int]] = []
    next_pos = [0 for _ in range(m)]
    for j in range(m):
        ranked_by_keyword.append(
            [int(idx) for idx in torch.argsort(kw_sims[j], descending=True).tolist() if int(idx) in cand_set]
        )

    selected: list[int] = []
    selected_set: set[int] = set()

    def _take_next(j: int) -> int | None:
        while next_pos[j] < len(ranked_by_keyword[j]):
            idx = ranked_by_keyword[j][next_pos[j]]
            next_pos[j] += 1
            if idx in selected_set:
                continue
            return idx
        return None

    for j in keyword_order:
        q = int(quotas[j].item())
        if q <= 0:
            continue
        while q > 0 and len(selected) < max_pick:
            idx = _take_next(j)
            if idx is None:
                break
            selected.append(idx)
            selected_set.add(idx)
            q -= 1
        if len(selected) >= max_pick:
            break

    while len(selected) < max_pick:
        progressed = False
        for j in keyword_order:
            idx = _take_next(j)
            if idx is None:
                continue
            selected.append(idx)
            selected_set.add(idx)
            progressed = True
            if len(selected) >= max_pick:
                break
        if not progressed:
            break
    return selected


def _build_image_keyword_scores(
    selected_idx: list[int],
    frame_ids: list[int],
    kws_rep: list[str],
    kw_frame_sims: torch.Tensor,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not selected_idx:
        return out
    if not kws_rep or kw_frame_sims.shape[0] == 0:
        for idx in selected_idx:
            out.append({"index_in_pool": int(idx), "frame_id": int(frame_ids[idx]), "keyword_scores": {}})
        return out

    sel_tensor = torch.tensor(selected_idx, device=kw_frame_sims.device, dtype=torch.long)
    sims = kw_frame_sims[:, sel_tensor]
    sims_cpu = sims.detach().cpu().tolist()
    for j, idx in enumerate(selected_idx):
        kw_scores = {kw: float(sims_cpu[i][j]) for i, kw in enumerate(kws_rep)}
        out.append({"index_in_pool": int(idx), "frame_id": int(frame_ids[idx]), "keyword_scores": kw_scores})
    return out


def _build_question_options_visual_text(question: str, options: list[str] | None) -> str:
    if options:
        return f"Question: {question}\nOptions:\n{format_labeled_options(options)}"
    return f"Question: {question}"


# ==================== 单样本评估主流程 ====================
def _eval_one_sample(
    model: Any,
    proc: Any,
    visual_encoder: VisualEncoder,
    sample: VQASample,
    prompt: str,
    args: argparse.Namespace,
    max_new_tokens: int,
    model_mode: str,
    budget: int,
    preprocessed_clip_dir: str | None,
    use_subtitles: bool,
    subtitles_dir: str | None,
) -> dict[str, Any]:
    pool_fps = float(args.candidate_pool_fps)
    if pool_fps <= 0:
        raise ValueError(f"candidate_pool_fps 必须 > 0，当前: {pool_fps}")

    t_kw = time.perf_counter()
    kws_raw, kws = _extract_keywords_with_optional_cache(
        model=model,
        proc=proc,
        sample_id=sample.sample_id,
        dataset=str(args.dataset),
        task_type=str(sample.task_type),
        question=sample.question,
        options=sample.options,
        max_new_tokens=128,
        target_keywords=args.max_keywords,
        prompt_version=int(args.keyword_prompt_version),
        extractor_model=str(args.keyword_extractor_model),
        api_base_url=str(args.keyword_extractor_api_base_url),
        api_key_env=str(args.keyword_extractor_api_key_env),
        use_cache=bool(args.use_keyword_cache),
        cache_dir=str(args.keyword_cache_dir or ""),
        cache_number=int(args.keyword_cache_number),
        cache_write_on_miss=bool(args.write_keyword_cache),
    )
    if not kws:
        raise RuntimeError(f"LLM 关键词提取失败: sample={sample.sample_id}")
    kws_after_text_dedup = list(kws)
    kw_emb = _encode_texts_for_dedup(visual_encoder, kws_after_text_dedup, 32)
    kws_rep, kw_emb_rep, _ = _merge_keywords(kws_after_text_dedup, kw_emb, args.max_keywords)
    keyword_extract_time = time.perf_counter() - t_kw

    t0 = time.perf_counter()
    if args.use_preprocessed_clip_frames:
        frame_ids, imgs = load_preprocessed_candidate_frames(preprocessed_clip_dir or "", sample.sample_id)
        src_fps = float(args.preprocessed_clip_fps)
        keep = pool_positions_at_fps(len(imgs), src_fps, pool_fps)
        if keep:
            frame_ids, imgs = [frame_ids[i] for i in keep], [imgs[i] for i in keep]
    else:
        frame_ids, imgs = collect_video_frames_at_fps(sample.video_path, pool_fps)
    frame_sampling_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    img_emb = _encode_images_for_encoder(visual_encoder, imgs, args.ours_clip_batch_size)
    kw_frame_sims = _compute_kw_frame_sims(
        visual_encoder,
        imgs,
        kws_rep,
        args.ours_clip_batch_size,
        img_emb=img_emb,
    )
    info_pack = _compute_keyword_information(kws_rep, kw_frame_sims, args)
    kws_use: list[str] = info_pack["kws_use"]
    kw_weights: torch.Tensor = info_pack["kw_weights"]

    if VERBOSE:
        keep_weight_map: dict[str, float] = {}
        for i, kw in enumerate(kws_use):
            keep_weight_map[kw] = float(kw_weights[i].item()) if kw_weights.numel() > i else 0.0
        for r in info_pack["rows"]:
            _log(
                "sample={} kw='{}' info={:.4f} weight={:.4f} local_evidence={:.4f}".format(
                    sample.sample_id,
                    r["keyword"],
                    r["info"],
                    keep_weight_map.get(r["keyword"], 0.0),
                    r["local_evidence_score"],
                )
            )
    _log(f"sample={sample.sample_id} 关键词: llm={len(kws_after_text_dedup)} -> after_cap={len(kws_use)}")

    emb_time = time.perf_counter() - t1
    _log(
        f"sample={sample.sample_id} 候选={len(imgs)}, 关键词={len(kws_use)}/{len(kws_rep)}, "
        f"budget={budget}"
    )

    if kw_frame_sims.shape[0] == 0:
        raise RuntimeError(f"样本无可用关键词: sample={sample.sample_id}")

    selected_idx = _select_keyword_frames(
        kw_frame_sims=kw_frame_sims,
        kw_weights=kw_weights,
        budget=budget,
    )
    if not selected_idx:
        raise RuntimeError(f"样本未能选出有效帧: sample={sample.sample_id}")

    selected_idx = sorted(selected_idx, key=lambda x: frame_ids[x])
    _log(
        f"sample={sample.sample_id} 候选池={len(imgs)}, "
        f"选帧={len(selected_idx)}, method=quota_topk"
    )

    one_shot_frames = [imgs[i] for i in selected_idx]
    sampled_frame_ids = [int(frame_ids[i]) for i in selected_idx]
    subtitles_for_prompt, selected_frame_subtitles = (
        _collect_subtitles_for_sample(sample, sampled_frame_ids, subtitles_dir=subtitles_dir)
        if use_subtitles
        else ([], ["" for _ in sampled_frame_ids])
    )
    prompt_use = build_user_text_with_subtitles(sample.question, sample.options, subtitles_for_prompt) if use_subtitles else prompt
    one_shot_out = _run_vlm_once(
        model,
        proc,
        one_shot_frames,
        prompt_use,
        max_new_tokens,
        model_mode,
        has_options=sample.options is not None,
    )
    verbose_keywords = list(kws_use)
    verbose_keyword_rows = list(info_pack["rows"])
    verbose_kw_frame_sims = kw_frame_sims
    if VERBOSE:
        question_options_label = "question + options"
        question_options_text = _build_question_options_visual_text(sample.question, sample.options)
        qo_sims = _compute_kw_frame_sims(
            visual_encoder,
            imgs,
            [question_options_text],
            args.ours_clip_batch_size,
            img_emb=img_emb,
        )
        verbose_keywords.append(question_options_label)
        verbose_kw_frame_sims = torch.cat([kw_frame_sims, qo_sims], dim=0)
        verbose_keyword_rows.append(
            {
                "keyword": question_options_label,
                "local_evidence_score": 0.0,
                "info": 0.0,
                "weight": 0.0,
                "used_for_selection": False,
            }
        )
    all_frame_kw_scores = _build_image_keyword_scores(
        list(range(len(frame_ids))),
        frame_ids,
        verbose_keywords,
        verbose_kw_frame_sims,
    )
    dump_verbose_round(
        verbose=VERBOSE,
        verbose_run_dir=_VERBOSE_RUN_DIR,
        sample_id=sample.sample_id,
        stage="oneshot",
        round_id=0,
        question=sample.question,
        options=sample.options,
        gt_answer=sample.answer,
        all_keywords=verbose_keywords,
        frame_ids=frame_ids,
        selected_idx=selected_idx,
        imgs=imgs,
        image_keyword_scores=all_frame_kw_scores,
        keyword_info_scores=verbose_keyword_rows,
        vlm_out=one_shot_out,
        raw_keywords_before_dedup=kws_raw,
        selected_frame_subtitles=selected_frame_subtitles,
        selection_info={
            "num_frames_budget": int(budget),
            "candidate_pool_size": int(len(imgs)),
            "candidate_pool_fps": float(args.candidate_pool_fps),
            "max_keywords": int(args.max_keywords),
            "keyword_prompt_version": int(args.keyword_prompt_version),
            "keyword_extractor_model": str(args.keyword_extractor_model),
            "keyword_weight_strength": float(args.keyword_weight_strength),
            "frame_selection_method": "quota_topk",
            "keyword_extract_time_sec": float(keyword_extract_time),
        },
        task_type=str(sample.task_type),
    )
    _log(f"sample={sample.sample_id} one-shot pred={one_shot_out['pred_answer']}, frames={len(selected_idx)}")
    return {
        "pred_answer": str(one_shot_out["pred_answer"]),
        "response": str(one_shot_out["response"]),
        "inference_time": float(one_shot_out["inference_time"]),
        "frame_sampling_time": frame_sampling_time,
        "embedding_build_time": emb_time,
        "selected_frame_count": len(selected_idx),
        "over_limit_count": int(one_shot_out["hit_max_tokens"]),
    }


# ==================== 数据集级评估 ====================
def _build_eval_prompt(
    sample: VQASample,
    args: argparse.Namespace,
    lmms_model_name: str,
    *,
    use_subtitles: bool,
    num_frames: int,
    subtitles_dir: str | None = None,
) -> str:
    raw_doc = sample.metadata.get("raw_doc")
    if is_official_videomme_dataset(args.dataset) and isinstance(raw_doc, dict):
        return build_videomme_prompt(
            raw_doc,
            lmms_model_name,
            use_subtitles=use_subtitles,
            num_frames=num_frames,
        )
    if is_official_lvb_dataset(args.dataset) and isinstance(raw_doc, dict):
        return build_lvb_prompt(
            raw_doc,
            lmms_model_name,
            use_subtitles=use_subtitles,
            num_frames=num_frames,
            video_dir=resolve_dataset_root(args.dataset),
            subtitles_dir=subtitles_dir,
            subtitle_path=str(sample.metadata.get("subtitle_path", "") or "") or None,
        )
    return build_user_text(sample.question, sample.options)


def evaluate_vqa(
    model: Any,
    proc: Any,
    samples: list[VQASample],
    args: argparse.Namespace,
    max_new_tokens: int,
    model_mode: str,
    budget: int,
    preprocessed_clip_dir: str | None,
    use_subtitles: bool,
    subtitles_dir: str | None,
    lmms_model_name: str = "",
) -> dict[str, Any]:
    use_official_videomme = is_official_videomme_dataset(args.dataset)
    use_official_lvb = is_official_lvb_dataset(args.dataset)
    visual_encoder = _load_visual_encoder(args.ours_clip_model_id, args.ours_clip_device)
    if visual_encoder.backend == "blip_itc":
        _log(f"ours 使用 BLIP ITC 打分: model_id={visual_encoder.model_id}")

    res = {
        "correct": 0,
        "total": 0,
        "mra_sum": 0.0,
        "mra_count": 0,
        "inference_times": [],
        "frame_sampling_times": [],
        "embedding_build_times": [],
        "selected_frame_counts": [],
        "over_max_tokens_count": 0,
        "official_videomme_scores": [],
        "official_lvb_scores": [],
    }
    pbar = tqdm(samples, desc="评估进度(ours semantic refinement)")
    for s in pbar:
        if not sample_matches_task_filter(s, args.task_filter):
            continue
        prompt = _build_eval_prompt(
            s,
            args,
            lmms_model_name,
            use_subtitles=use_subtitles,
            num_frames=budget,
            subtitles_dir=subtitles_dir,
        )
        out = _eval_one_sample(
            model,
            proc,
            visual_encoder,
            s,
            prompt,
            args,
            max_new_tokens,
            model_mode,
            budget,
            preprocessed_clip_dir,
            use_subtitles=use_subtitles and not use_official_videomme and not use_official_lvb,
            subtitles_dir=subtitles_dir,
        )
        pred = out["pred_answer"]
        is_correct = False
        if use_official_videomme:
            raw_doc = s.metadata.get("raw_doc")
            if isinstance(raw_doc, dict):
                score_dict = score_videomme(raw_doc, str(out["response"]))
                res["official_videomme_scores"].append(score_dict)
                pred = score_dict.get("pred_answer", "")
                is_correct = float(score_dict.get("score", 0.0)) == 1.0
        elif use_official_lvb:
            raw_doc = s.metadata.get("raw_doc")
            if isinstance(raw_doc, dict):
                score_dict = score_lvb(raw_doc, str(out["response"]))
                res["official_lvb_scores"].append(score_dict)
                pred = score_dict.get("parsed_pred", "")
                is_correct = str(score_dict.get("answer", "")).strip().upper() == str(pred).strip().upper()
        res["inference_times"].append(float(out["inference_time"]))
        res["frame_sampling_times"].append(float(out["frame_sampling_time"]))
        res["embedding_build_times"].append(float(out["embedding_build_time"]))
        res["selected_frame_counts"].append(int(out["selected_frame_count"]))
        res["over_max_tokens_count"] += int(out["over_limit_count"])
        if s.options is not None:
            res["total"] += 1
            if not use_official_videomme and not use_official_lvb:
                is_correct = str(s.answer).strip().upper() == str(pred).strip().upper()
            if is_correct:
                res["correct"] += 1
        else:
            try:
                res["mra_sum"] += calculate_mra(float(pred) if pred else 0.0, float(s.answer))
            except (ValueError, TypeError):
                pass
            res["mra_count"] += 1
        acc, t = _compute_accuracy_from_results(res, args.task_filter)
        pbar.set_postfix(Acc=f"{acc:.2f}%", AvgTime=f"{t:.2f}s")
    if use_official_videomme and res["official_videomme_scores"]:
        res["official_videomme_overall"] = aggregate_videomme(res["official_videomme_scores"])
    if use_official_lvb and res["official_lvb_scores"]:
        res["official_lvb_overall"] = aggregate_lvb(res["official_lvb_scores"])
    res["visual_encoder_model"] = visual_encoder.model_id
    res["visual_encoder_backend"] = visual_encoder.backend
    return res


def parse_args():
    # ==================== 命令行参数 ====================
    p = argparse.ArgumentParser(description="VQA ours: 粗看+关键词驱动精看")
    p.add_argument("--dataset", type=str, default="vsibench", choices=list_supported_datasets())
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Thinking")
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--num_samples", type=str, default="10", help="评测样本数；all=全量 test")
    p.add_argument("--num_frames", type=int, default=16, help="单轮推理的总选帧预算（默认16）")
    p.add_argument("--frame_sampling_method", type=str, default="ours", choices=["ours"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--task_filter",
        type=str,
        default="all",
        help="all/mcq/numeric/generation，或数据集特定桶（如 videomme: short/medium/long；lvb: 15/60/600/3600 或 TOS/T2A/...；mlvu: plotQA/anomaly_reco/...）",
    )
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--ours_clip_model_id", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--ours_clip_device", type=str, default=None)
    p.add_argument("--ours_clip_batch_size", type=int, default=16)
    p.add_argument("--model_mode_config", type=str, default="config/model_response_modes.json")
    p.add_argument("--log_file", type=str, default="vqa_embedding_evaluation_log.csv")
    p.add_argument("--use_preprocessed_clip_frames", action="store_true")
    p.add_argument("--preprocessed_clip_fps", type=float, default=1.0)
    p.add_argument("--preprocessed_clip_dir", type=str, default="")
    p.add_argument("--use_subtitles", action="store_true", help="为 Video-MME 按采样帧时间对齐读取字幕并拼入 prompt")
    p.add_argument("--subtitles_dir", type=str, default="", help="字幕目录（可选）；为空时尝试在视频邻近目录下自动查找 subtitle/subtitles")
    p.add_argument(
        "--candidate_pool_fps",
        type=float,
        default=1.0,
        help="候选池粗采样目标 fps（默认 1.0，即一秒一帧）；视频按源 fps 换算帧索引，预处理目录按 preprocessed_clip_fps 重采样",
    )
    p.add_argument(
        "--max_keywords",
        type=int,
        default=5,
        help="最大关键词数：不超过则全保留；超过则删除与其它词 CLIP 相似度最高的 (n-max_keywords) 个",
    )
    p.add_argument(
        "--keyword_prompt_version",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="关键词抽取 prompt 版本：0=旧版，1=CLIP 帧检索导向，2=题干+每选项一词，3=极简视觉元素提取",
    )
    p.add_argument(
        "--keyword_extractor_model",
        type=str,
        default="local",
        help="关键词抽取模型：local=复用 --model_path；local-{model}=指定本地模型（如 local-Qwen/Qwen3-VL-4B-Instruct）；"
        "poe-gpt-5.2=Poe responses API；aio-gpt-5.2=aiohub chat API；"
        "or-openai/gpt-4o=OpenRouter chat API；无前缀时走 --keyword_extractor_api_*",
    )
    p.add_argument(
        "--keyword_extractor_api_base_url",
        type=str,
        default="https://api.aiohub.org/v1",
        help="无前缀远程模型时的 OpenAI Chat Completions base_url（poe-/aio-/or- 前缀会忽略此项）",
    )
    p.add_argument(
        "--keyword_extractor_api_key_env",
        type=str,
        default="AIOHUB_API_KEY",
        help="无前缀远程模型时的 API key 环境变量（poe-/aio-/or- 前缀会忽略此项）",
    )
    p.add_argument(
        "--keyword_weight_strength",
        type=float,
        default=1.0,
        help="关键词权重幂律指数 λ：w_i=φ_i^λ/Σφ_j^λ，φ 为局部证据度；λ=0 均分，λ=1 按 info 比例，λ>1 强化高信息量词",
    )
    p.add_argument(
        "--use_keyword_cache",
        action="store_true",
        help="仅使用关键词磁盘缓存（不调用关键词 API）：同一 dataset(+short/medium/long)+keyword_extractor_model"
        "+prompt+cache_number 下已跑过的样本直接读缓存；未命中则报错（默认目录 /userhome/cs3/duanty/vqa_keyword_cache）",
    )
    p.add_argument(
        "--write_keyword_cache",
        action="store_true",
        help="与 --use_keyword_cache 联用：缓存未命中时调用关键词 API 并写入缓存（需 API key）；"
        "未开启时仅读已有缓存，未命中则报错",
    )
    p.add_argument(
        "--keyword_cache_dir",
        type=str,
        default=str(DEFAULT_KEYWORD_CACHE_DIR),
        help="关键词缓存根目录（默认 /userhome/cs3/duanty/vqa_keyword_cache）",
    )
    p.add_argument(
        "--keyword_cache_number",
        type=int,
        default=0,
        help="关键词缓存组编号：同一 dataset+extractor+prompt 下可维护多组缓存，选最优一组评测",
    )
    return p.parse_args()


def main():
    # ==================== 主入口与实验记录 ====================
    exp_t0 = time.perf_counter()
    args = parse_args()
    global _VERBOSE_RUN_DIR
    _VERBOSE_RUN_DIR = init_verbose_run_dir(verbose=VERBOSE, output_dir=VERBOSE_OUTPUT_DIR, log_fn=_log)
    write_verbose_frame_selection_manifest(
        _VERBOSE_RUN_DIR,
        {
            "dataset": str(args.dataset),
            "task_filter": str(args.task_filter),
            "num_samples": str(args.num_samples),
            "num_frames": int(args.num_frames),
            "candidate_pool_fps": float(args.candidate_pool_fps),
            "max_keywords": int(args.max_keywords),
            "keyword_weight_strength": float(args.keyword_weight_strength),
            "keyword_extractor_model": str(args.keyword_extractor_model),
            "keyword_prompt_version": int(args.keyword_prompt_version),
            "index_file": "selected_frames_index.jsonl",
            "index_schema": {
                "sample_id": "样本 ID",
                "selected_frame_ids": "按时间排序的选中帧 ID",
                "selected_index_in_pool": "候选池中的帧下标",
                "frame_selection_method": "quota_topk",
                "verbose_detail_path": "单样本详细 info.json 相对路径",
            },
        },
    )
    eval_csv_dir = Path(__file__).resolve().parent / "eval_csv"
    eval_csv_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(eval_csv_dir / Path(args.log_file).name)
    preprocessed_clip_dir = resolve_preprocessed_clip_dir(
        args.dataset, args.preprocessed_clip_fps, args.preprocessed_clip_dir
    )
    if args.use_preprocessed_clip_frames and args.frame_sampling_method not in PREPROCESSED_CLIP_COMPATIBLE_METHODS:
        raise ValueError("use_preprocessed_clip_frames 仅支持 ours。")

    sample_count = None if args.num_samples.lower() == "all" else int(args.num_samples)
    loader = get_data_loader(args.dataset, seed=args.seed, task_filter=args.task_filter)
    samples = loader.get_eval_samples(sample_count=sample_count)

    resolved_model_path = os.path.expanduser(args.model_path)
    lora_path = ""
    if args.model_name:
        model_name = args.model_name
    elif args.use_lora:
        model_name = os.path.expanduser(args.base_model) if args.base_model else "Qwen/Qwen3-VL-4B-Thinking"
        lora_path = resolved_model_path
    else:
        model_name = os.path.basename(resolved_model_path.rstrip("/"))

    mode_cfg = load_model_response_mode_config(args.model_mode_config)
    candidates = [resolved_model_path] + ([model_name] if model_name else []) + ([os.path.expanduser(args.base_model)] if args.base_model else [])
    model_mode, last_err = "", None
    for c in candidates:
        try:
            model_mode, _, _ = resolve_model_mode(model_identifier=c, config=mode_cfg)
            break
        except (KeyError, ValueError) as e:
            last_err = e
    if not model_mode:
        raise RuntimeError(f"模型模式识别失败: {candidates}") from last_err
    if model_mode == "thinking":
        raise RuntimeError("本脚本仅支持 instruct 模式。请改用 instruct 模型或扩展实现。")
    effective_max_new_tokens = MODE_MAX_NEW_TOKENS[model_mode]
    lmms_model_name = resolve_lmms_model_name(resolved_model_path)
    if is_official_videomme_dataset(args.dataset):
        effective_max_new_tokens = resolve_videomme_max_new_tokens()
        print(
            "[vqa_eval_ours] Video-MME 使用官方 lmms-eval 数据 / prompt / 打分；"
            f"max_new_tokens={effective_max_new_tokens}, lmms_model={lmms_model_name}",
            flush=True,
        )
    elif is_official_lvb_dataset(args.dataset):
        effective_max_new_tokens = resolve_lvb_max_new_tokens()
        print(
            "[vqa_eval_ours] LongVideoBench validation 使用官方 lmms-eval prompt / 打分；"
            f"max_new_tokens={effective_max_new_tokens}, lmms_model={lmms_model_name}",
            flush=True,
        )

    budget = max(1, int(args.num_frames))
    if args.use_keyword_cache:
        cache_root = resolve_keyword_cache_root(args.keyword_cache_dir or None)
        cache_task = str(args.task_filter) if str(args.task_filter) in KEYWORD_CACHE_DURATION_SUFFIXES else None
        cache_run = keyword_cache_run_dir(
            cache_root,
            dataset=str(args.dataset),
            task_type=cache_task,
            extractor_model=str(args.keyword_extractor_model),
            prompt_version=int(args.keyword_prompt_version),
            target_keywords=int(args.max_keywords),
            cache_number=int(args.keyword_cache_number),
        )
        if cache_task:
            _log(f"关键词缓存已启用: {cache_run}")
        else:
            ds_key = sanitize_cache_component(str(args.dataset))
            _log(
                f"关键词缓存已启用: {cache_root}/{ds_key}-{{short|medium|long}}/... "
                f"(按样本 task_type 分目录)"
            )
    _log(
        f"配置: candidate_pool_fps={float(args.candidate_pool_fps):g}, "
        f"budget={budget}, frame_selection=quota_topk"
    )

    apply_pixel_limits = should_apply_vl_pixel_limits(
        resolved_model_path,
        args.dataset,
    )
    if apply_pixel_limits:
        print(
            f"[vqa_eval_ours] 启用 processor 像素限制（num_frames={budget}，防 OOM / context 溢出）",
            flush=True,
        )
    model, proc = load_model_and_processor(
        resolved_model_path,
        use_lora=args.use_lora,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
        apply_pixel_limits=apply_pixel_limits,
        num_frames=budget,
    )
    results = evaluate_vqa(
        model,
        proc,
        samples,
        args,
        effective_max_new_tokens,
        model_mode,
        budget,
        preprocessed_clip_dir,
        use_subtitles=bool(args.use_subtitles),
        subtitles_dir=(args.subtitles_dir.strip() if args.subtitles_dir.strip() else None),
        lmms_model_name=lmms_model_name,
    )

    avg_acc, avg_time = _compute_accuracy_from_results(results, args.task_filter)
    _, correct = _compute_score_counts_for_csv(results, args.task_filter)
    avg_fs = _avg(results["frame_sampling_times"])
    avg_emb = _avg(results["embedding_build_times"])
    avg_sel = _avg(results["selected_frame_counts"])
    avg_total_h = (time.perf_counter() - exp_t0) / 3600.0

    csv_columns = ours_eval_csv_columns()
    log_ours_eval_to_csv(
        log_file,
        csv_columns,
        ours_eval_csv_row(
            args,
            num_samples=len(samples),
            correct_count=correct,
            accuracy_percent=avg_acc,
            avg_inference_time=avg_time,
            avg_frame_sampling_time=avg_fs,
            avg_embedding_build_time=avg_emb,
            avg_selected_frame_count=avg_sel,
            avg_total_time_hours=avg_total_h,
            model_name=model_name,
            lora_path=lora_path,
            visual_encoder_model=str(results.get("visual_encoder_model", "")),
            visual_encoder_backend=str(results.get("visual_encoder_backend", "")),
        ),
        log_fn=_log,
    )
    _log(f"评估完成: samples={len(samples)}, acc={avg_acc:.2f}%, avg_infer={avg_time:.3f}s")
    if "official_videomme_overall" in results:
        print(
            f"官方 Video-MME Overall: {results['official_videomme_overall']:.1f}% "
            f"(n={len(results['official_videomme_scores'])})",
            flush=True,
        )
    if "official_lvb_overall" in results:
        print(
            f"官方 LongVideoBench Overall: {results['official_lvb_overall']:.1f}% "
            f"(n={len(results['official_lvb_scores'])})",
            flush=True,
        )


if __name__ == "__main__":
    main()
