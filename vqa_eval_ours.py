from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from data_loaders import (
    apply_dataset_cli_defaults,
    dataset_uses_vl_pixel_limits,
    get_data_loader,
    list_supported_datasets,
)
from data_loaders.base import VQASample, sample_matches_task_filter
from model_response_mode import load_model_response_mode_config, parse_response_by_mode, resolve_model_mode
from utils import (
    avg as _avg,
    build_user_text,
    build_user_text_with_subtitles,
    format_labeled_options,
    calculate_mra,
    collect_subtitles_for_frame_ids as _collect_subtitles_for_sample,
    compute_accuracy_from_results as _compute_accuracy_from_results,
    compute_score_counts_for_csv as _compute_score_counts_for_csv,
    dump_verbose_round,
    from_pretrained_local_first,
    init_verbose_run_dir,
    normalize_sample_id,
)
from vl_common import load_keyword_model_and_processor, load_model_and_processor, prepare_vlm_inputs

MODE_MAX_NEW_TOKENS = {"thinking": 4086, "instruct": 128}
PREPROCESSED_CLIP_COMPATIBLE_METHODS = {"ours"}
_CLIP_CACHE: dict[str, tuple[Any, Any, str]] = {}
_OPENAI_CLIENT_CACHE: dict[tuple[str, str], Any] = {}
_KEYWORD_LOCAL_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
VERBOSE = True
VERBOSE_OUTPUT_DIR = Path(__file__).resolve().parent / "verbose_eval_ours"
_DEFAULT_KEYWORD_CACHE_DIR = Path.home() / "vqa_keyword_cache"
_VERBOSE_RUN_DIR: Path | None = None


# ==================== 基础工具 ====================
def _log(msg: str) -> None:
    print(f"[vqa_eval_ours] {msg}", flush=True)


# ==================== 候选帧读取与采样 ====================
def _load_preprocessed_candidate_frames(preprocessed_clip_dir: str, sample_id: str) -> tuple[list[int], list[Image.Image]]:
    sample_dir = Path(preprocessed_clip_dir).expanduser().resolve() / normalize_sample_id(sample_id)
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"预处理帧目录不存在: {sample_dir}")
    meta_path = sample_dir / "metadata.json"
    frame_ids: list[int] = []
    image_paths: list[Path] = []
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        ids = meta.get("frame_ids", [])
        files = meta.get("files", [])
        if isinstance(ids, list) and isinstance(files, list) and len(ids) == len(files):
            for fid, rel_name in zip(ids, files):
                p = sample_dir / str(rel_name)
                if p.is_file():
                    frame_ids.append(int(fid))
                    image_paths.append(p)
    if not image_paths:
        for p in sorted(sample_dir.glob("frame_*.jpg")):
            m = re.match(r"frame_(\d+)\.jpg$", p.name)
            if m:
                frame_ids.append(int(m.group(1)))
                image_paths.append(p)
    if not image_paths:
        raise RuntimeError(f"预处理帧目录中没有可用图片: {sample_dir}")
    images: list[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return frame_ids, images


def _frame_indices_by_target_fps(total_frames: int, src_fps: float, target_fps: float) -> list[int]:
    if total_frames <= 0:
        return []
    if target_fps <= 0:
        raise ValueError(f"target_fps 必须 > 0，当前: {target_fps}")
    if not src_fps or src_fps <= 0:
        return list(range(total_frames))
    step = src_fps / target_fps
    indices: list[int] = []
    cursor = 0.0
    last = -1
    while True:
        idx = int(round(cursor))
        if idx >= total_frames:
            break
        if idx != last:
            indices.append(idx)
            last = idx
        cursor += step
    if indices and indices[0] != 0:
        indices.insert(0, 0)
    return indices


def _pool_positions_at_fps(n_items: int, src_fps: float, target_fps: float) -> list[int]:
    """在已有 n_items 个按 src_fps 采样的条目上，重采样到 target_fps。"""
    if n_items <= 0:
        return []
    if target_fps <= 0:
        raise ValueError(f"target_fps 必须 > 0，当前: {target_fps}")
    if not src_fps or src_fps <= 0:
        src_fps = float(target_fps)
    return _frame_indices_by_target_fps(n_items, src_fps, target_fps)


def _collect_video_frames_at_fps(
    video_path: str,
    target_fps: float,
) -> tuple[list[int], list[Image.Image]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"视频帧数无效: {video_path}")
    idxs = _frame_indices_by_target_fps(frame_count, src_fps, target_fps)
    frame_ids, images = [], []
    try:
        for fid in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ids.append(int(fid))
            images.append(Image.fromarray(rgb))
    finally:
        cap.release()
    if not images:
        raise RuntimeError(f"视频无可用候选帧: {video_path}")
    return frame_ids, images


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


# ==================== CLIP 编码 ====================
def _load_clip(model_id: str, device: str | None) -> tuple[Any, Any, str]:
    d = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = f"{model_id}::{d}"
    if key not in _CLIP_CACHE:
        proc = from_pretrained_local_first(AutoProcessor.from_pretrained, model_id, log=_log)
        model = from_pretrained_local_first(AutoModel.from_pretrained, model_id, log=_log).to(d).eval()
        _CLIP_CACHE[key] = (proc, model, d)
    return _CLIP_CACHE[key]


def _encode_images(images: list[Image.Image], proc: Any, model: Any, device: str, bs: int) -> torch.Tensor:
    outs = []
    for i in range(0, len(images), bs):
        inputs = proc(images=images[i:i + bs], return_tensors="pt").to(device)
        with torch.no_grad():
            feats = _to_feature_tensor(model.get_image_features(**inputs))
        outs.append(_norm(feats))
    return torch.cat(outs, dim=0)


def _encode_texts(texts: list[str], proc: Any, model: Any, device: str, bs: int) -> torch.Tensor:
    outs = []
    for i in range(0, len(texts), bs):
        inputs = proc(text=texts[i:i + bs], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = _to_feature_tensor(model.get_text_features(**inputs))
        outs.append(_norm(feats))
    return torch.cat(outs, dim=0)


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
    raise ValueError(f"keyword_prompt_version 仅支持 0 或 1，当前: {prompt_version}")


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


_KEYWORD_EXTRACTOR_PROVIDERS: dict[str, dict[str, str]] = {
    "poe": {
        "base_url": "https://api.poe.com/v1",
        "api_key_env": "POE_API_KEY",
        "api_style": "responses",
    },
    "aio": {
        "base_url": "https://api.aiohub.org/v1",
        "api_key_env": "AIOHUB_API_KEY",
        "api_style": "chat",
    },
    "or": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_style": "chat",
    },
}


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
    for prefix in _KEYWORD_EXTRACTOR_PROVIDERS:
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
        spec = _KEYWORD_EXTRACTOR_PROVIDERS[provider]
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
        "base_url": str(api_base_url or "").strip() or _KEYWORD_EXTRACTOR_PROVIDERS["aio"]["base_url"],
        "api_key_env": str(api_key_env or "").strip() or _KEYWORD_EXTRACTOR_PROVIDERS["aio"]["api_key_env"],
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
    """OpenRouter 可选 attribution 头（见 https://openrouter.ai/docs/quickstart）。"""
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


# ==================== 关键词磁盘缓存 ====================
_KEYWORD_CACHE_DURATION_SUFFIXES = frozenset({"short", "medium", "long"})


def _keyword_cache_dataset_key(dataset: str, task_type: str | None = None) -> str:
    """Video-MME 等按 duration 分桶时，缓存目录为 videomme-short / videomme-medium / videomme-long。"""
    ds = str(dataset or "dataset").strip()
    tt = str(task_type or "").strip().lower()
    if tt in _KEYWORD_CACHE_DURATION_SUFFIXES:
        return f"{ds}-{tt}"
    return ds


def _sanitize_cache_component(text: str, *, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(text or "").strip()).strip("_")
    if not s:
        s = "default"
    return s[:max_len]


def _resolve_keyword_cache_root(cache_dir: str | None) -> Path:
    raw = str(cache_dir or "").strip()
    if raw:
        return Path(os.path.expanduser(raw)).resolve()
    return _DEFAULT_KEYWORD_CACHE_DIR.resolve()


def _keyword_cache_run_dir(
    cache_root: Path,
    *,
    dataset: str,
    task_type: str | None = None,
    extractor_model: str,
    prompt_version: int,
    target_keywords: int,
    cache_number: int = 0,
) -> Path:
    """同一数据集(+duration) + 关键词抽取配置 + cache_number 共用一个缓存子目录。"""
    ext = _sanitize_cache_component(extractor_model or "local")
    ds = _sanitize_cache_component(_keyword_cache_dataset_key(dataset, task_type))
    return (
        cache_root
        / ds
        / ext
        / f"pv{int(prompt_version)}_tk{int(target_keywords)}"
        / f"cn{int(cache_number)}"
    )


def _keyword_cache_file(run_dir: Path, sample_id: str) -> Path:
    sid = normalize_sample_id(sample_id)
    return run_dir / f"{_sanitize_cache_component(sid, max_len=200)}.json"


def _load_keyword_cache_entry(path: Path, *, sample_id: str) -> tuple[list[str], list[str]] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _log(f"关键词缓存读取失败，将重新抽取: {path} ({e})")
        return None
    if not isinstance(data, dict):
        return None
    cached_sid = str(data.get("sample_id", "")).strip()
    if cached_sid and normalize_sample_id(cached_sid) != normalize_sample_id(sample_id):
        _log(f"关键词缓存 sample_id 不匹配，忽略: {path}")
        return None
    kws_raw = data.get("kws_raw")
    kws = data.get("kws")
    if not isinstance(kws_raw, list) or not isinstance(kws, list):
        return None
    kws_raw_out = [str(x).strip() for x in kws_raw if str(x).strip()]
    kws_out = [str(x).strip() for x in kws if str(x).strip()]
    if not kws_out:
        return None
    return kws_raw_out, kws_out


def _save_keyword_cache_entry(
    path: Path,
    *,
    sample_id: str,
    dataset: str,
    extractor_model: str,
    prompt_version: int,
    target_keywords: int,
    kws_raw: list[str],
    kws: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": normalize_sample_id(sample_id),
        "dataset": str(dataset),
        "extractor_model": str(extractor_model),
        "prompt_version": int(prompt_version),
        "target_keywords": int(target_keywords),
        "kws_raw": list(kws_raw),
        "kws": list(kws),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


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
) -> tuple[list[str], list[str]]:
    cache_dataset_key = _keyword_cache_dataset_key(dataset, task_type)
    cache_root = _resolve_keyword_cache_root(cache_dir)
    run_dir = _keyword_cache_run_dir(
        cache_root,
        dataset=dataset,
        task_type=task_type,
        extractor_model=extractor_model,
        prompt_version=prompt_version,
        target_keywords=target_keywords,
        cache_number=cache_number,
    )
    cache_path = _keyword_cache_file(run_dir, sample_id)

    if use_cache:
        cached = _load_keyword_cache_entry(cache_path, sample_id=sample_id)
        if cached is not None:
            kws_raw, kws = cached
            _log(f"sample={sample_id} 关键词缓存命中 ({cache_path})")
            return kws_raw, kws

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
    if use_cache and kws:
        _save_keyword_cache_entry(
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


def _allocate_adaptive_quotas(
    weights: torch.Tensor,
    local_evidence: torch.Tensor,
    budget: int,
    gamma: float,
) -> torch.Tensor:
    """按权重与 Hoyer LE 分配每关键词帧配额：尖峰词配额小，平缓词配额大。"""
    budget = int(budget)
    m = int(weights.numel())
    if budget <= 0 or m == 0:
        return torch.zeros((m,), dtype=torch.long, device=weights.device)

    w = weights.float().clamp(min=0.0)
    le = local_evidence.float().clamp(min=0.0, max=1.0)
    if le.numel() != m:
        le = torch.zeros_like(w)

    raw = w * (1.0 + float(gamma) * (1.0 - le))
    s = float(raw.sum().item())
    if s <= 1e-12:
        raw = torch.ones_like(w)
        s = float(m)

    if budget >= m:
        quotas = torch.ones((m,), dtype=torch.long, device=weights.device)
        extra = budget - m
        if extra > 0:
            quotas = quotas + _allocate_counts_by_weights(raw, extra)
        return quotas

    proportional = budget * raw / s
    quotas = torch.round(proportional).to(torch.long).clamp(min=1)
    diff = budget - int(quotas.sum().item())
    if diff == 0:
        return quotas

    frac = proportional - quotas.float()
    order = torch.argsort(frac, descending=(diff > 0))
    guard = 0
    while diff != 0 and guard < m * 4:
        i = int(order[guard % m].item())
        if diff > 0:
            quotas[i] += 1
            diff -= 1
        elif int(quotas[i].item()) > 1:
            quotas[i] -= 1
            diff += 1
        guard += 1
    if diff != 0:
        return _allocate_counts_by_weights(raw, budget)
    return quotas


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
    kw_emb_rep: torch.Tensor,
    img_emb: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """合并后的关键词全部保留；用局部证据度作为关键词信息量。"""
    m = int(kw_emb_rep.shape[0])
    n = int(img_emb.shape[0])
    if m == 0 or n == 0:
        return {
            "kws_use": [],
            "kw_emb_use": kw_emb_rep[:0],
            "kw_weights": torch.empty((0,), device=img_emb.device),
            "rows": [],
        }

    sims = kw_emb_rep @ img_emb.T  # (M, N)
    local_evidence = torch.tensor(
        [_local_evidence_score(sims[i]) for i in range(m)],
        dtype=kw_emb_rep.dtype,
        device=kw_emb_rep.device,
    ).clamp(min=0.0, max=1.0)

    kws_use = list(kws_rep)
    kw_emb_use = kw_emb_rep
    kw_info = local_evidence
    s = float(kw_info.sum().item())
    info_weights = (kw_info / s) if s > 1e-12 else torch.ones_like(kw_info) / float(max(1, kw_info.numel()))
    uniform_weights = torch.ones_like(info_weights) / float(max(1, info_weights.numel()))
    weight_strength = min(1.0, max(0.0, float(args.keyword_weight_strength)))
    kw_weights = ((1.0 - weight_strength) * uniform_weights + weight_strength * info_weights).clamp(min=0.0)
    ws = float(kw_weights.sum().item())
    kw_weights = (kw_weights / ws) if ws > 1e-12 else uniform_weights

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
        "kw_emb_use": kw_emb_use,
        "kw_weights": kw_weights,
        "local_evidence": local_evidence,
        "rows": rows,
    }


# ==================== VLM 单轮推理与打分 ====================
def _option_probs(proc: Any, logits: torch.Tensor) -> dict[str, float]:
    p = torch.softmax(logits, dim=-1)
    out = {}
    for o in ("A", "B", "C", "D"):
        # 选项首 token 可能有前导空格或大小写差异，统一做聚合。
        ids = set()
        for t in {o, o.lower(), f" {o}", f" {o.lower()}"}:
            toks = proc.tokenizer(t, add_special_tokens=False)["input_ids"]
            if toks:
                ids.add(int(toks[0]))
        out[o] = float(sum(float(p[i].item()) for i in ids)) if ids else 0.0
    return out


def _run_vlm_once(model: Any, proc: Any, frames: list[Image.Image], prompt: str, max_new_tokens: int, model_mode: str) -> dict[str, Any]:
    inputs, _ = prepare_vlm_inputs(proc, frames, prompt, model=model)
    inputs = inputs.to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
    infer_t = time.perf_counter() - t0
    seq = out.sequences
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, seq)]
    resp = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    _, _, pred = parse_response_by_mode(response=resp, has_options=True, model_mode=model_mode)
    logits = out.scores[-1][0] if out.scores else torch.zeros((model.config.vocab_size,), device=model.device)
    probs = _option_probs(proc, logits)
    pred_u = str(pred).strip().upper()
    if pred_u not in {"A", "B", "C", "D"}:
        pred_u = max(probs.items(), key=lambda x: x[1])[0]
    return {"pred_answer": pred_u, "response": resp, "option_probs": probs, "entropy": 0.0, "inference_time": infer_t, "hit_max_tokens": int(len(out.scores) >= max_new_tokens)}


def _build_quota_prescreen_candidates(
    kw_sims: torch.Tensor,
    kw_w: torch.Tensor,
    budget: int,
    alpha: int,
    quotas: torch.Tensor | None = None,
) -> list[int]:
    """按关键词配额做软预筛：每词取 top-(alpha*q_i)，并集去重。"""
    if kw_sims.ndim != 2 or kw_sims.shape[0] == 0 or budget <= 0:
        return []
    m = int(kw_sims.shape[0])
    n = int(kw_sims.shape[1])
    if kw_w.numel() != m:
        kw_w = torch.ones((m,), device=kw_sims.device)
    kw_w = kw_w.float().clamp(min=0.0)
    if quotas is None:
        quotas = _allocate_counts_by_weights(kw_w, budget)
    else:
        quotas = quotas.to(device=kw_sims.device, dtype=torch.long)
        if quotas.numel() != m:
            quotas = _allocate_counts_by_weights(kw_w, budget)
    alpha = max(1, int(alpha))
    picked: set[int] = set()
    for j in range(m):
        qj = int(quotas[j].item())
        topn = alpha * qj
        if topn <= 0:
            continue
        topn = min(n, topn)
        idxs = torch.argsort(kw_sims[j], descending=True)[:topn].tolist()
        picked.update(int(i) for i in idxs)
    return sorted(picked)


def _submodular_cover_greedy_select(
    kw_sims: torch.Tensor,
    kw_w: torch.Tensor,
    budget: int,
    candidate_idx: list[int],
) -> list[int]:
    """次模覆盖贪心：F(S)=sum_i w_i * max_{f in S} sim(f,k_i)。"""
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

    selected: list[int] = []
    selected_set: set[int] = set()
    covered = torch.zeros((m,), dtype=kw_sims.dtype, device=kw_sims.device)

    max_pick = min(int(budget), n)
    for _ in range(max_pick):
        avail = [i for i in cand if i not in selected_set]
        if not avail:
            break
        avail_tensor = torch.tensor(avail, device=kw_sims.device, dtype=torch.long)
        sim_avail = kw_sims[:, avail_tensor]  # (M, C)
        delta = torch.maximum(covered.unsqueeze(1), sim_avail) - covered.unsqueeze(1)
        gains = (delta * kw_w.unsqueeze(1)).sum(dim=0)  # (C,)
        best_local = int(torch.argmax(gains).item())
        if float(gains[best_local].item()) <= -1e8:
            break
        best_idx = int(avail[best_local])
        selected.append(best_idx)
        selected_set.add(best_idx)
        covered = torch.maximum(covered, kw_sims[:, best_idx])
    return selected


def _topq_mean_similarity(sims: list[float], q: int) -> float:
    if not sims or q <= 0:
        return 0.0
    k = min(int(q), len(sims))
    top = sorted(sims, reverse=True)[:k]
    return float(sum(top)) / k


def _sparsity_adaptive_quota_select(
    kw_sims: torch.Tensor,
    kw_w: torch.Tensor,
    quotas: torch.Tensor,
    budget: int,
    candidate_idx: list[int],
) -> list[int]:
    """Lazy greedy: F(S) = sum_i w_i * mean(top_{q_i} sims_i(S))."""
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

    if quotas.numel() != m:
        quotas = _allocate_counts_by_weights(kw_w, budget)
    quotas = quotas.long()

    cand = sorted({int(i) for i in candidate_idx if 0 <= int(i) < n})
    if len(cand) < min(budget, n):
        cand = list(range(n))

    max_pick = min(int(budget), n)
    selected: list[int] = []
    selected_set: set[int] = set()
    kw_selected_sims: list[list[float]] = [[] for _ in range(m)]

    for _ in range(max_pick):
        best_idx = -1
        best_gain = -1.0
        for f in cand:
            if f in selected_set:
                continue
            gain = 0.0
            sim_f = kw_sims[:, f]
            for i in range(m):
                qi = int(quotas[i].item())
                old_m = _topq_mean_similarity(kw_selected_sims[i], qi)
                new_m = _topq_mean_similarity(kw_selected_sims[i] + [float(sim_f[i].item())], qi)
                gain += float(kw_w[i].item()) * (new_m - old_m)
            if gain > best_gain:
                best_gain = gain
                best_idx = f
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        sim_best = kw_sims[:, best_idx]
        for i in range(m):
            kw_selected_sims[i].append(float(sim_best[i].item()))
    return selected


def _select_keyword_frames(
    kw_frame_sims: torch.Tensor,
    kw_weights: torch.Tensor,
    kw_local_evidence: torch.Tensor,
    budget: int,
    frame_selection_mode: int,
    args: argparse.Namespace,
) -> tuple[list[int], list[int], str]:
    if budget <= 0:
        return [], [], "skipped"
    if frame_selection_mode == 1:
        candidate_idx = _build_quota_prescreen_candidates(
            kw_sims=kw_frame_sims,
            kw_w=kw_weights,
            budget=budget,
            alpha=int(args.quota_prescreen_alpha),
        )
        selected_idx = _submodular_cover_greedy_select(
            kw_sims=kw_frame_sims,
            kw_w=kw_weights,
            budget=budget,
            candidate_idx=candidate_idx,
        )
        return selected_idx, candidate_idx, "coverage_greedy"
    if frame_selection_mode == 0:
        candidate_idx = list(range(int(kw_frame_sims.shape[1])))
        selected_idx = _quota_topk_select(
            kw_sims=kw_frame_sims,
            kw_w=kw_weights,
            budget=budget,
            candidate_idx=candidate_idx,
        )
        return selected_idx, candidate_idx, "quota_topk"
    if frame_selection_mode == 2:
        adaptive_quotas = _allocate_adaptive_quotas(
            kw_weights,
            kw_local_evidence,
            budget,
            float(args.quota_gamma),
        )
        candidate_idx = _build_quota_prescreen_candidates(
            kw_sims=kw_frame_sims,
            kw_w=kw_weights,
            budget=budget,
            alpha=int(args.quota_prescreen_alpha),
            quotas=adaptive_quotas,
        )
        selected_idx = _sparsity_adaptive_quota_select(
            kw_sims=kw_frame_sims,
            kw_w=kw_weights,
            quotas=adaptive_quotas,
            budget=budget,
            candidate_idx=candidate_idx,
        )
        return selected_idx, candidate_idx, "sparsity_adaptive_quota"
    raise ValueError(f"frame_selection_mode 仅支持 0/1/2，当前: {frame_selection_mode}")


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
    kw_emb_rep: torch.Tensor,
    img_emb: torch.Tensor,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not selected_idx:
        return out
    if not kws_rep or kw_emb_rep.shape[0] == 0:
        for idx in selected_idx:
            out.append({"index_in_pool": int(idx), "frame_id": int(frame_ids[idx]), "keyword_scores": {}})
        return out

    sel_tensor = torch.tensor(selected_idx, device=img_emb.device, dtype=torch.long)
    sims = kw_emb_rep @ img_emb[sel_tensor].T
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
    clip_proc: Any,
    clip_model: Any,
    clip_device: str,
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
    )
    if not kws:
        raise RuntimeError(f"LLM 关键词提取失败: sample={sample.sample_id}")
    kws_after_text_dedup = list(kws)
    kw_emb = _encode_texts(kws_after_text_dedup, clip_proc, clip_model, clip_device, 32)
    kws_rep, kw_emb_rep, _ = _merge_keywords(kws_after_text_dedup, kw_emb, args.max_keywords)
    keyword_extract_time = time.perf_counter() - t_kw

    t0 = time.perf_counter()
    if args.use_preprocessed_clip_frames:
        frame_ids, imgs = _load_preprocessed_candidate_frames(preprocessed_clip_dir or "", sample.sample_id)
        src_fps = float(args.preprocessed_clip_fps)
        keep = _pool_positions_at_fps(len(imgs), src_fps, pool_fps)
        if keep:
            frame_ids, imgs = [frame_ids[i] for i in keep], [imgs[i] for i in keep]
    else:
        frame_ids, imgs = _collect_video_frames_at_fps(sample.video_path, pool_fps)
    frame_sampling_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    img_emb = _encode_images(imgs, clip_proc, clip_model, clip_device, args.ours_clip_batch_size)
    info_pack = _compute_keyword_information(kws_rep, kw_emb_rep, img_emb, args)
    kws_use: list[str] = info_pack["kws_use"]
    kw_emb_use: torch.Tensor = info_pack["kw_emb_use"]
    kw_weights: torch.Tensor = info_pack["kw_weights"]
    kw_local_evidence: torch.Tensor = info_pack["local_evidence"]
    kw_frame_sims = (kw_emb_use @ img_emb.T) if kw_emb_use.shape[0] > 0 else torch.empty((0, img_emb.shape[0]), device=img_emb.device)

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

    if kw_emb_use.shape[0] == 0:
        raise RuntimeError(f"样本无可用关键词: sample={sample.sample_id}")

    frame_selection_mode = int(args.frame_selection_mode)
    selected_idx, candidate_idx, selection_method = _select_keyword_frames(
        kw_frame_sims=kw_frame_sims,
        kw_weights=kw_weights,
        kw_local_evidence=kw_local_evidence,
        budget=budget,
        frame_selection_mode=frame_selection_mode,
        args=args,
    )
    if not selected_idx:
        raise RuntimeError(f"样本未能选出有效帧: sample={sample.sample_id}")

    selected_idx = sorted(selected_idx, key=lambda x: frame_ids[x])
    if frame_selection_mode == 1:
        mode_log = f"alpha={int(args.quota_prescreen_alpha)}"
    elif frame_selection_mode == 2:
        mode_log = f"gamma={float(args.quota_gamma):g}, alpha={int(args.quota_prescreen_alpha)}"
    else:
        mode_log = "alpha=skipped"
    _log(
        f"sample={sample.sample_id} 候选池={len(candidate_idx) if candidate_idx else len(imgs)}, "
        f"选帧={len(selected_idx)}, mode={frame_selection_mode}/{selection_method}, "
        f"{mode_log}"
    )

    one_shot_frames = [imgs[i] for i in selected_idx]
    sampled_frame_ids = [int(frame_ids[i]) for i in selected_idx]
    subtitles_for_prompt, selected_frame_subtitles = (
        _collect_subtitles_for_sample(sample, sampled_frame_ids, subtitles_dir=subtitles_dir)
        if use_subtitles
        else ([], ["" for _ in sampled_frame_ids])
    )
    prompt_use = build_user_text_with_subtitles(sample.question, sample.options, subtitles_for_prompt) if use_subtitles else prompt
    one_shot_out = _run_vlm_once(model, proc, one_shot_frames, prompt_use, max_new_tokens, model_mode)
    verbose_keywords = list(kws_use)
    verbose_kw_emb = kw_emb_use
    verbose_keyword_rows = list(info_pack["rows"])
    if VERBOSE:
        question_options_label = "question + options"
        question_options_text = _build_question_options_visual_text(sample.question, sample.options)
        question_options_emb = _encode_texts([question_options_text], clip_proc, clip_model, clip_device, 1)
        verbose_keywords.append(question_options_label)
        verbose_kw_emb = torch.cat([verbose_kw_emb, question_options_emb], dim=0)
        verbose_keyword_rows.append(
            {
                "keyword": question_options_label,
                "local_evidence_score": 0.0,
                "info": 0.0,
                "weight": 0.0,
                "used_for_selection": False,
            }
        )
    all_frame_kw_scores = _build_image_keyword_scores(list(range(len(frame_ids))), frame_ids, verbose_keywords, verbose_kw_emb, img_emb)
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
            "quota_prescreen_alpha": int(args.quota_prescreen_alpha),
            "quota_gamma": float(args.quota_gamma),
            "max_keywords": int(args.max_keywords),
            "keyword_prompt_version": int(args.keyword_prompt_version),
            "keyword_extractor_model": str(args.keyword_extractor_model),
            "keyword_weight_strength": float(args.keyword_weight_strength),
            "frame_selection_mode": frame_selection_mode,
            "frame_selection_method": selection_method,
            "prescreen_candidate_count": int(len(candidate_idx) if candidate_idx else len(imgs)),
            "keyword_extract_time_sec": float(keyword_extract_time),
        },
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
) -> dict[str, Any]:
    clip_proc, clip_model, clip_device = _load_clip(args.ours_clip_model_id, args.ours_clip_device)

    res = {"correct": 0, "total": 0, "mra_sum": 0.0, "mra_count": 0, "inference_times": [], "frame_sampling_times": [], "embedding_build_times": [], "selected_frame_counts": [], "over_max_tokens_count": 0}
    pbar = tqdm(samples, desc="评估进度(ours semantic refinement)")
    for s in pbar:
        if not sample_matches_task_filter(s, args.task_filter):
            continue
        out = _eval_one_sample(
            model,
            proc,
            clip_proc,
            clip_model,
            clip_device,
            s,
            build_user_text(s.question, s.options),
            args,
            max_new_tokens,
            model_mode,
            budget,
            preprocessed_clip_dir,
            use_subtitles=use_subtitles,
            subtitles_dir=subtitles_dir,
        )
        pred = out["pred_answer"]
        res["inference_times"].append(float(out["inference_time"]))
        res["frame_sampling_times"].append(float(out["frame_sampling_time"]))
        res["embedding_build_times"].append(float(out["embedding_build_time"]))
        res["selected_frame_counts"].append(int(out["selected_frame_count"]))
        res["over_max_tokens_count"] += int(out["over_limit_count"])
        if s.options is not None:
            res["total"] += 1
            if str(s.answer).strip().upper() == str(pred).strip().upper():
                res["correct"] += 1
        else:
            try:
                res["mra_sum"] += calculate_mra(float(pred) if pred else 0.0, float(s.answer))
            except (ValueError, TypeError):
                pass
            res["mra_count"] += 1
        acc, t = _compute_accuracy_from_results(res, args.task_filter)
        pbar.set_postfix(Acc=f"{acc:.2f}%", AvgTime=f"{t:.2f}s")
    return res


def parse_args():
    # ==================== 命令行参数 ====================
    p = argparse.ArgumentParser(description="VQA ours: 粗看+关键词驱动精看")
    p.add_argument("--dataset", type=str, default="vsibench", choices=list_supported_datasets())
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--dataset_name", type=str, default="nyu-visionx/VSI-Bench")
    p.add_argument("--dataset_config", type=str, default="full")
    p.add_argument("--no_dataset_config", action="store_true")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Thinking")
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--video_dir", type=str, default="~/dataset/vsi_bench")
    p.add_argument("--num_samples", type=str, default="10")
    p.add_argument("--num_frames", type=int, default=16, help="单轮推理的总选帧预算（默认16）")
    p.add_argument("--frame_sampling_method", type=str, default="ours", choices=["ours"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--task_filter",
        type=str,
        default="all",
        help="all/mcq/numeric/generation，或数据集特定桶（如 videomme: short/medium/long；mlvu: plotQA/anomaly_reco/...）",
    )
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--ours_clip_model_id", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--ours_clip_device", type=str, default=None)
    p.add_argument("--ours_clip_batch_size", type=int, default=16)
    p.add_argument("--model_mode_config", type=str, default="config/model_response_modes.json")
    p.add_argument("--log_file", type=str, default="vqa_embedding_evaluation_log.csv")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--use_train_split", action="store_true")
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
    p.add_argument("--quota_prescreen_alpha", type=int, default=3, help="配额软预筛系数 alpha：每个关键词预筛 top-(alpha*q_i) 帧")
    p.add_argument(
        "--quota_gamma",
        type=float,
        default=1.0,
        help="mode2 稀疏自适应配额系数 gamma：raw_q_i = w_i * (1 + gamma * (1 - LE_i))",
    )
    p.add_argument(
        "--max_keywords",
        type=int,
        default=5,
        help="最大关键词数：不超过则全保留；超过则删除与其它词 CLIP 相似度最高的 (n-max_keywords) 个",
    )
    p.add_argument("--keyword_prompt_version", type=int, choices=[0, 1], default=0, help="关键词抽取prompt版本：0=旧版，1=CLIP帧检索导向新版")
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
    p.add_argument("--keyword_weight_strength", type=float, default=1.0, help="关键词权重强度：0为所有关键词均分，1为完全使用info权重")
    p.add_argument(
        "--use_keyword_cache",
        action="store_true",
        help="使用关键词磁盘缓存：同一 dataset(+short/medium/long)+keyword_extractor_model+prompt+cache_number "
        "下已跑过的样本直接读缓存，未命中则 LLM 抽取后写入（默认目录 ~/vqa_keyword_cache）",
    )
    p.add_argument(
        "--keyword_cache_dir",
        type=str,
        default="",
        help="关键词缓存根目录；为空时使用 ~/vqa_keyword_cache",
    )
    p.add_argument(
        "--keyword_cache_number",
        type=int,
        default=0,
        help="关键词缓存组编号：同一 dataset+extractor+prompt 下可维护多组缓存，选最优一组评测",
    )
    p.add_argument(
        "--frame_selection_mode",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="最终选帧模式：1=覆盖贪心，0=配额topk，2=稀疏自适应配额 sparsity_adaptive_quota",
    )
    return p.parse_args()


def _ours_csv_columns() -> list[str]:
    return [
        "timestamp",
        "dataset",
        "seed",
        "task_filter",
        "num_samples",
        "correct_count",
        "accuracy_percent",
        "num_frames",
        "avg_inference_time",
        "frame_sampling_method",
        "avg_frame_sampling_time",
        "avg_embedding_build_time",
        "avg_selected_frame_count",
        "avg_total_time_hours",
        "model_name",
        "lora_path",
        "candidate_pool_fps",
        "quota_prescreen_alpha",
        "quota_gamma",
        "max_keywords",
        "keyword_prompt_version",
        "keyword_extractor_model",
        "use_keyword_cache",
        "keyword_cache_number",
        "keyword_weight_strength",
        "frame_selection_mode",
        "use_preprocessed_clip_frames",
        "preprocessed_clip_fps",
        "use_subtitles",
    ]


def _ours_csv_row(
    args: argparse.Namespace,
    *,
    num_samples: int,
    correct_count: float,
    accuracy_percent: float,
    avg_inference_time: float,
    avg_frame_sampling_time: float,
    avg_embedding_build_time: float,
    avg_selected_frame_count: float,
    avg_total_time_hours: float,
    model_name: str,
    lora_path: str,
) -> list[Any]:
    return [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        args.dataset,
        args.seed,
        args.task_filter,
        num_samples,
        f"{correct_count:.6f}",
        f"{accuracy_percent:.2f}",
        args.num_frames,
        f"{avg_inference_time:.3f}",
        args.frame_sampling_method,
        f"{avg_frame_sampling_time:.6f}",
        f"{avg_embedding_build_time:.6f}",
        f"{avg_selected_frame_count:.6f}",
        f"{avg_total_time_hours:.6f}",
        model_name,
        lora_path,
        f"{float(args.candidate_pool_fps):g}",
        int(args.quota_prescreen_alpha),
        f"{float(args.quota_gamma):g}",
        int(args.max_keywords),
        int(args.keyword_prompt_version),
        str(args.keyword_extractor_model),
        bool(args.use_keyword_cache),
        int(args.keyword_cache_number),
        f"{float(args.keyword_weight_strength):g}",
        int(args.frame_selection_mode),
        bool(args.use_preprocessed_clip_frames),
        f"{float(args.preprocessed_clip_fps):g}",
        bool(args.use_subtitles),
    ]


def _log_ours_eval_to_csv(log_file: str, columns: list[str], row: list[Any]) -> None:
    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    if not write_header:
        with path.open("r", newline="", encoding="utf-8") as rf:
            existing_header = next(csv.reader(rf), None)
        if existing_header != columns:
            _log(f"CSV 表头与当前版本不一致，跳过写入: {path}")
            return
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(columns)
        w.writerow(row)


def main():
    # ==================== 主入口与实验记录 ====================
    exp_t0 = time.perf_counter()
    args = parse_args()
    apply_dataset_cli_defaults(args)
    global _VERBOSE_RUN_DIR
    _VERBOSE_RUN_DIR = init_verbose_run_dir(verbose=VERBOSE, output_dir=VERBOSE_OUTPUT_DIR, log_fn=_log)
    video_dir = os.path.expanduser(args.video_dir)
    eval_csv_dir = Path(__file__).resolve().parent / "eval_csv"
    eval_csv_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(eval_csv_dir / Path(args.log_file).name)
    default_pre = Path("/userhome/cs3/duanty/dataset_preposcess") / args.dataset / f"clip_{args.preprocessed_clip_fps:g}"
    preprocessed_clip_dir = os.path.expanduser(args.preprocessed_clip_dir) if args.preprocessed_clip_dir.strip() else str(default_pre)
    if args.use_preprocessed_clip_frames and args.frame_sampling_method not in PREPROCESSED_CLIP_COMPATIBLE_METHODS:
        raise ValueError("use_preprocessed_clip_frames 仅支持 ours。")

    sample_count = None if args.num_samples.lower() == "all" else int(args.num_samples)
    loader = get_data_loader(args.dataset, video_dir=video_dir, seed=args.seed, train_ratio=args.train_ratio, task_filter=args.task_filter, dataset_name=args.dataset_name, dataset_config=args.dataset_config, no_dataset_config=args.no_dataset_config)
    samples = loader.get_split_samples(split=args.dataset_split, use_train_split=args.use_train_split, sample_count=sample_count)

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

    budget = max(1, int(args.num_frames))
    if args.use_keyword_cache:
        cache_root = _resolve_keyword_cache_root(args.keyword_cache_dir or None)
        cache_task = str(args.task_filter) if str(args.task_filter) in _KEYWORD_CACHE_DURATION_SUFFIXES else None
        cache_run = _keyword_cache_run_dir(
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
            ds_key = _sanitize_cache_component(str(args.dataset))
            _log(
                f"关键词缓存已启用: {cache_root}/{ds_key}-{{short|medium|long}}/... "
                f"(按样本 task_type 分目录)"
            )
    _log(
        f"配置: candidate_pool_fps={float(args.candidate_pool_fps):g}, "
        f"budget={budget}, frame_selection_mode={int(args.frame_selection_mode)}, "
        f"quota_prescreen_alpha={int(args.quota_prescreen_alpha)}, "
        f"quota_gamma={float(args.quota_gamma):g}"
    )

    apply_pixel_limits = dataset_uses_vl_pixel_limits(
        args.dataset,
        args.dataset_split,
        args.dataset_name,
    )
    if apply_pixel_limits:
        print(
            "[vqa_eval_ours] MLVU-Test：启用 processor max_pixels 限制（防超高分辨率 OOM）",
            flush=True,
        )
    model, proc = load_model_and_processor(
        resolved_model_path,
        use_lora=args.use_lora,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
        apply_pixel_limits=apply_pixel_limits,
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
    )

    avg_acc, avg_time = _compute_accuracy_from_results(results, args.task_filter)
    _, correct = _compute_score_counts_for_csv(results, args.task_filter)
    avg_fs = _avg(results["frame_sampling_times"])
    avg_emb = _avg(results["embedding_build_times"])
    avg_sel = _avg(results["selected_frame_counts"])
    avg_total_h = (time.perf_counter() - exp_t0) / 3600.0

    csv_columns = _ours_csv_columns()
    _log_ours_eval_to_csv(
        log_file,
        csv_columns,
        _ours_csv_row(
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
        ),
    )
    _log(f"评估完成: samples={len(samples)}, acc={avg_acc:.2f}%, avg_infer={avg_time:.3f}s")


if __name__ == "__main__":
    main()
