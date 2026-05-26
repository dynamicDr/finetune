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

from data_loaders import get_data_loader, list_supported_datasets
from data_loaders.base import VQASample
from model_response_mode import load_model_response_mode_config, parse_response_by_mode, resolve_model_mode
from utils import dump_verbose_round, init_verbose_run_dir, normalize_sample_id
from vl_common import load_model_and_processor

MODE_MAX_NEW_TOKENS = {"thinking": 4086, "instruct": 128}
PREPROCESSED_CLIP_COMPATIBLE_METHODS = {"ours"}
_CLIP_CACHE: dict[str, tuple[Any, Any, str]] = {}
VERBOSE = True
VERBOSE_OUTPUT_DIR = Path(__file__).resolve().parent / "verbose_eval_ours"
_VERBOSE_RUN_DIR: Path | None = None


# ==================== 基础工具与通用统计 ====================
def _log(msg: str) -> None:
    print(f"[vqa_eval_ours] {msg}", flush=True)


def build_user_text(question: str, options: list[str] | None) -> str:
    if options:
        return f"{question}\n\nOptions:\n" + "\n".join(options) + "\n\nDirectly answer with the option letter only. Do not explain."
    return f"{question}\n\nPlease provide the numerical answer directly."


def build_user_text_with_subtitles(question: str, options: list[str] | None, subtitles: list[str] | None) -> str:
    options_text = "\n".join(options or [])
    subtitle_text = "\n".join((subtitles or [])).strip() or "No subtitles available"
    return (
        "This video's subtitles are listed below:\n"
        f"{subtitle_text}\n\n"
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        f"{question}\n{options_text}\n"
        "The best answer is:"
    )


def calculate_mra(pred: float, gt: float) -> float:
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, 1 - abs(pred - gt) / abs(gt))


def _compute_accuracy_from_results(results: dict, task_filter: str) -> tuple[float, float]:
    avg_accuracy = 0.0
    if task_filter in {"mcq", "short", "medium", "long"} and results["total"] > 0:
        avg_accuracy = results["correct"] / results["total"] * 100
    elif task_filter == "numeric" and results["mra_count"] > 0:
        avg_accuracy = results["mra_sum"] / results["mra_count"] * 100
    elif task_filter == "all":
        total_score = results["correct"] + results["mra_sum"]
        total_count = results["total"] + results["mra_count"]
        if total_count > 0:
            avg_accuracy = total_score / total_count * 100
    avg_inference_time = sum(results["inference_times"]) / len(results["inference_times"]) if results["inference_times"] else 0.0
    return avg_accuracy, avg_inference_time


def _compute_score_counts_for_csv(results: dict, task_filter: str) -> tuple[int, float]:
    if task_filter in {"mcq", "short", "medium", "long"}:
        return int(results["total"]), float(results["correct"])
    if task_filter == "numeric":
        return int(results["mra_count"]), float(results["mra_sum"])
    return int(results["total"] + results["mra_count"]), float(results["correct"] + results["mra_sum"])


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


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


def _collect_video_frames_at_fps(video_path: str, target_fps: float) -> tuple[list[int], list[Image.Image]]:
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


def _parse_subtitle_time(time_str: str) -> float:
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _load_srt_segments(subtitle_path: str) -> list[tuple[float, float, str]]:
    p = Path(subtitle_path)
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    segs: list[tuple[float, float, str]] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        time_line_idx = 0 if "-->" in lines[0] else (1 if len(lines) > 1 and "-->" in lines[1] else -1)
        if time_line_idx < 0:
            continue
        try:
            ts0, ts1 = [x.strip() for x in lines[time_line_idx].split("-->")]
            start_t = _parse_subtitle_time(ts0)
            end_t = _parse_subtitle_time(ts1)
        except Exception:
            continue
        content_lines = lines[time_line_idx + 1 :]
        if not content_lines:
            continue
        raw = " ".join(content_lines)
        cleaned = re.sub(r"<[^>]+>", "", raw).strip()
        if cleaned:
            segs.append((start_t, end_t, cleaned))
    return segs


def _resolve_subtitle_path(sample: VQASample, subtitles_dir: str | None) -> str | None:
    explicit_dir = os.path.expanduser(subtitles_dir) if subtitles_dir else ""
    video_path = Path(sample.video_path).expanduser().resolve()
    video_stem = video_path.stem
    meta_video_id = str(sample.metadata.get("videoID", "")).strip() if isinstance(sample.metadata, dict) else ""
    candidates: list[Path] = []
    subtitle_dir_names = ("subtitle", "subtitles", "subtitles/subtitle")

    def _append_stem_files(base_dir: Path) -> None:
        for stem in [meta_video_id, video_stem]:
            if stem:
                candidates.extend([base_dir / f"{stem}.srt", base_dir / f"{stem}.SRT"])

    if explicit_dir:
        base = Path(explicit_dir)
        if base.is_dir():
            _append_stem_files(base)
            for subname in subtitle_dir_names:
                _append_stem_files(base / subname)
    for root in [video_path.parent, video_path.parent.parent, video_path.parent.parent.parent]:
        for subname in subtitle_dir_names:
            _append_stem_files(root / subname)
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _collect_subtitles_for_sample(
    sample: VQASample,
    sampled_frame_ids: list[int],
    subtitles_dir: str | None,
) -> tuple[list[str], list[str]]:
    subtitle_path = _resolve_subtitle_path(sample, subtitles_dir=subtitles_dir)
    if not subtitle_path or not sampled_frame_ids:
        return [], ["" for _ in sampled_frame_ids]
    cap = cv2.VideoCapture(sample.video_path)
    if not cap.isOpened():
        return [], ["" for _ in sampled_frame_ids]
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1e-6:
        return [], ["" for _ in sampled_frame_ids]
    segments = _load_srt_segments(subtitle_path)
    if not segments:
        return [], ["" for _ in sampled_frame_ids]
    out: list[str] = []
    seen: set[str] = set()
    per_frame_subs: list[str] = []
    for fid in sampled_frame_ids:
        t = float(fid) / fps
        matched = ""
        for s0, s1, txt in segments:
            if s0 <= t < s1:
                matched = txt
                if txt not in seen:
                    seen.add(txt)
                    out.append(txt)
                break
        per_frame_subs.append(matched)
    return out, per_frame_subs


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
        _CLIP_CACHE[key] = (AutoProcessor.from_pretrained(model_id), AutoModel.from_pretrained(model_id).to(d).eval(), d)
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


def _extract_keywords_with_llm_text(
    model: Any,
    proc: Any,
    question: str,
    options: list[str] | None,
    max_new_tokens: int,
    target_keywords: int,
) -> tuple[list[str], list[str]]:
    target_keywords = max(1, int(target_keywords))
    low_keywords = max(1, target_keywords - 1)
    high_keywords = target_keywords + 1
    options_text = "\n".join(options or [])
    prompt = (
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
        f"- Output around {target_keywords} keywords (preferred range: {low_keywords}-{high_keywords}).\n"
        "- Output STRICTLY a JSON array of strings. No explanation, no markdown, "
        "no extra text.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}"
    )
    def _ask_once(prompt_text: str) -> tuple[list[str], list[str]]:
        content = [{"type": "text", "text": prompt_text}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        seq = out[0][len(inputs.input_ids[0]):]
        resp = proc.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        kws_raw_local = _parse_visual_keyword_phrases(resp)
        out_kws_local, seen_local = [], set()
        for kw in kws_raw_local:
            s = kw.strip().lower()
            if not s:
                continue
            if s in seen_local:
                continue
            seen_local.add(s)
            out_kws_local.append(s)
        return kws_raw_local, out_kws_local

    kws_raw, out_kws = _ask_once(prompt)
    if out_kws:
        return kws_raw, out_kws

    retry_prompt = (
        prompt
        + "\n\nYour previous answer was empty or not parseable. Retry now.\n"
        + "You MUST output a valid JSON array of visual phrases and include at least 1 keyword."
    )
    return _ask_once(retry_prompt)


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


def _compute_keyword_information(
    kws_rep: list[str],
    kw_emb_rep: torch.Tensor,
    img_emb: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """合并后的关键词全部保留；用 info_aggressiveness 控制配额权重。"""
    m = int(kw_emb_rep.shape[0])
    n = int(img_emb.shape[0])
    if m == 0 or n == 0:
        return {
            "kws_use": [],
            "kw_emb_use": kw_emb_rep[:0],
            "kw_weights": torch.empty((0,), device=img_emb.device),
            "rows": [],
            "kw_info_use": kw_emb_rep[:0],
        }

    sims = kw_emb_rep @ img_emb.T  # (M, N)
    peak = sims.max(dim=1).values
    mean = sims.mean(dim=1)
    std = sims.std(dim=1, unbiased=False).clamp(min=1e-6)
    prominence_z = (peak - mean) / std

    aggr = min(1.0, max(0.0, float(args.info_aggressiveness)))

    # 单一超参映射到原 6 个信息量控制项
    peak_floor = 0.10 + 0.20 * aggr
    peak_ceiling = 0.45 + 0.30 * aggr
    prominence_center = 1.0 + 1.2 * aggr
    prominence_scale = 1.0 - 0.6 * aggr
    entropy_temperature = 0.14 - 0.10 * aggr
    mix_prominence = 0.2 + 0.6 * aggr

    denom = max(1e-6, peak_ceiling - peak_floor)
    peak_term = ((peak - peak_floor) / denom).clamp(min=0.0, max=1.0)

    z_scale = max(1e-6, float(prominence_scale))
    prominence_term = torch.sigmoid((prominence_z - prominence_center) / z_scale)

    if n <= 1:
        concentration_term = torch.ones_like(peak)
    else:
        temp = max(1e-3, float(entropy_temperature))
        logits = (sims - mean.unsqueeze(1)) / temp
        p = torch.softmax(logits, dim=1).clamp(min=1e-12)
        ent = -(p * torch.log(p)).sum(dim=1)
        concentration_term = (1.0 - ent / math.log(float(n))).clamp(min=0.0, max=1.0)

    mix = float(mix_prominence)
    mix = min(1.0, max(0.0, mix))
    shape_term = mix * prominence_term + (1.0 - mix) * concentration_term
    info = (peak_term * shape_term).clamp(min=0.0)

    kws_use = list(kws_rep)
    kw_emb_use = kw_emb_rep
    if aggr <= 1e-12:
        # 用户要求：aggressiveness=0 时关键词均分
        kw_info = torch.ones((m,), device=kw_emb_rep.device)
        kw_weights = torch.ones((m,), device=kw_emb_rep.device) / float(m)
    else:
        kw_info = info
        s = float(kw_info.sum().item())
        kw_weights = (kw_info / s) if s > 1e-12 else torch.ones_like(kw_info) / float(max(1, kw_info.numel()))

    rows = []
    for i, kw in enumerate(kws_rep):
        rows.append(
            {
                "keyword": kw,
                "peak": float(peak[i].item()),
                "mean": float(mean[i].item()),
                "std": float(std[i].item()),
                "prominence_z": float(prominence_z[i].item()),
                "peak_term": float(peak_term[i].item()),
                "prominence_term": float(prominence_term[i].item()),
                "concentration_term": float(concentration_term[i].item()),
                "info": float(info[i].item()),
                "kept": True,
            }
        )

    return {
        "kws_use": kws_use,
        "kw_emb_use": kw_emb_use,
        "kw_weights": kw_weights,
        "kw_info_use": kw_info,
        "rows": rows,
        "mapped_params": {
            "info_peak_floor": float(peak_floor),
            "info_peak_ceiling": float(peak_ceiling),
            "info_prominence_center": float(prominence_center),
            "info_prominence_scale": float(prominence_scale),
            "info_entropy_temperature": float(entropy_temperature),
            "info_mix_prominence": float(mix_prominence),
        },
    }


def _mapped_info_params_from_aggr(aggr_raw: float) -> dict[str, float]:
    aggr = min(1.0, max(0.0, float(aggr_raw)))
    return {
        "info_peak_floor": float(0.10 + 0.20 * aggr),
        "info_peak_ceiling": float(0.45 + 0.30 * aggr),
        "info_prominence_center": float(1.0 + 1.2 * aggr),
        "info_prominence_scale": float(1.0 - 0.6 * aggr),
        "info_entropy_temperature": float(0.14 - 0.10 * aggr),
        "info_mix_prominence": float(0.2 + 0.6 * aggr),
    }


def _init_frames_uniform_plus_elements(
    frame_emb: torch.Tensor,
    elem_emb: torch.Tensor,
    elem_w: torch.Tensor,
    k0: int,
    uniform_ratio: float,
) -> list[int]:
    n = int(frame_emb.shape[0])
    if n <= 0 or k0 <= 0:
        return []
    k0 = min(int(k0), n)
    ur = min(1.0, max(0.0, float(uniform_ratio)))
    ku = int(round(k0 * ur))
    ke = k0 - ku

    # Step 2: uniform mid-point sampling over [0, N)
    su: set[int] = set()
    if ku > 0:
        for i in range(ku):
            idx = int(math.floor(n * (i + 0.5) / ku))
            idx = max(0, min(n - 1, idx))
            su.add(idx)

    # Step 3: element-guided sampling with per-element weighted quotas
    te: list[int] = []
    if ke > 0 and elem_emb.shape[0] > 0:
        sims = elem_emb @ frame_emb.T  # (M, N)
        if elem_w.numel() != elem_emb.shape[0]:
            elem_w = torch.ones((elem_emb.shape[0],), device=elem_emb.device)
        elem_w = elem_w.float().clamp(min=0.0)
        quotas = _allocate_counts_by_weights(elem_w, ke)
        picked: set[int] = set()
        for j in range(int(elem_emb.shape[0])):
            q = int(quotas[j].item())
            if q <= 0:
                continue
            for idx in torch.argsort(sims[j], descending=True).tolist():
                if idx in picked:
                    continue
                picked.add(int(idx))
                q -= 1
                if q <= 0:
                    break
        te = sorted(picked)
        if len(te) < ke:
            agg = (sims * elem_w.unsqueeze(1)).sum(dim=0)
            for idx in torch.argsort(agg, descending=True).tolist():
                if idx in picked:
                    continue
                picked.add(int(idx))
                te.append(int(idx))
                if len(te) >= ke:
                    break

    # Step 4: merge without forced backfill
    f0 = sorted(set(su).union(te))
    return f0


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
    content = [{"type": "image", "image": f} for f in frames] + [{"type": "text", "text": prompt}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=frames, padding=True, return_tensors="pt").to(model.device)
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
) -> list[int]:
    """按关键词配额做软预筛：每词取 top-(alpha*q_i)，并集去重。"""
    if kw_sims.ndim != 2 or kw_sims.shape[0] == 0 or budget <= 0:
        return []
    m = int(kw_sims.shape[0])
    n = int(kw_sims.shape[1])
    if kw_w.numel() != m:
        kw_w = torch.ones((m,), device=kw_sims.device)
    kw_w = kw_w.float().clamp(min=0.0)
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

    max_pick = min(int(budget), len(cand))
    for _ in range(max_pick):
        avail = [i for i in cand if i not in selected_set]
        if not avail:
            break
        avail_tensor = torch.tensor(avail, device=kw_sims.device, dtype=torch.long)
        sim_avail = kw_sims[:, avail_tensor]  # (M, C)
        delta = torch.maximum(covered.unsqueeze(1), sim_avail) - covered.unsqueeze(1)
        gains = (delta * kw_w.unsqueeze(1)).sum(dim=0)  # (C,)
        best_local = int(torch.argmax(gains).item())
        best_idx = int(avail[best_local])
        selected.append(best_idx)
        selected_set.add(best_idx)
        covered = torch.maximum(covered, kw_sims[:, best_idx])
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
    t0 = time.perf_counter()
    pool_fps = float(args.candidate_pool_fps)
    if pool_fps <= 0:
        raise ValueError(f"candidate_pool_fps 必须 > 0，当前: {pool_fps}")
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

    kws_raw, kws = _extract_keywords_with_llm_text(
        model=model,
        proc=proc,
        question=sample.question,
        options=sample.options,
        max_new_tokens=128,
        target_keywords=args.max_keywords,
    )
    if not kws:
        raise RuntimeError(f"LLM 关键词提取失败: sample={sample.sample_id}")
    kws_after_text_dedup = list(kws)
    kw_emb = _encode_texts(kws_after_text_dedup, clip_proc, clip_model, clip_device, 32)
    kws_rep, kw_emb_rep, _ = _merge_keywords(kws_after_text_dedup, kw_emb, args.max_keywords)
    info_pack = _compute_keyword_information(kws_rep, kw_emb_rep, img_emb, args)
    kws_use: list[str] = info_pack["kws_use"]
    kw_emb_use: torch.Tensor = info_pack["kw_emb_use"]
    kw_weights: torch.Tensor = info_pack["kw_weights"]
    kw_info_use: torch.Tensor = info_pack["kw_info_use"]
    kw_frame_sims = (kw_emb_use @ img_emb.T) if kw_emb_use.shape[0] > 0 else torch.empty((0, img_emb.shape[0]), device=img_emb.device)

    if VERBOSE:
        keep_weight_map: dict[str, float] = {}
        for i, kw in enumerate(kws_use):
            keep_weight_map[kw] = float(kw_weights[i].item()) if kw_weights.numel() > i else 0.0
        for r in info_pack["rows"]:
            _log(
                "sample={} kw='{}' info={:.4f} weight={:.4f} peak={:.4f} prom_z={:.3f} conc={:.3f}".format(
                    sample.sample_id,
                    r["keyword"],
                    r["info"],
                    keep_weight_map.get(r["keyword"], 0.0),
                    r["peak"],
                    r["prominence_z"],
                    r["concentration_term"],
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
    if not selected_idx:
        raise RuntimeError(f"样本未能选出有效帧: sample={sample.sample_id}")

    selected_idx = sorted(selected_idx, key=lambda x: frame_ids[x])
    _log(
        f"sample={sample.sample_id} 候选池={len(candidate_idx) if candidate_idx else len(imgs)}, "
        f"贪心选帧={len(selected_idx)}, alpha={int(args.quota_prescreen_alpha)}"
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
    one_shot_kw_scores = _build_image_keyword_scores(selected_idx, frame_ids, kws_use, kw_emb_use, img_emb)
    dump_verbose_round(
        verbose=VERBOSE,
        verbose_run_dir=_VERBOSE_RUN_DIR,
        sample_id=sample.sample_id,
        stage="oneshot",
        round_id=0,
        question=sample.question,
        options=sample.options,
        gt_answer=sample.answer,
        all_keywords=kws_use,
        frame_ids=frame_ids,
        selected_idx=selected_idx,
        imgs=imgs,
        image_keyword_scores=one_shot_kw_scores,
        keyword_info_scores=info_pack["rows"],
        vlm_out=one_shot_out,
        raw_keywords_before_dedup=kws_raw,
        keywords_after_info_filter=[{"keyword": kw, "info": float(kw_info_use[i].item())} for i, kw in enumerate(kws_use)],
        selected_frame_subtitles=selected_frame_subtitles,
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
        if args.task_filter != "all" and s.task_type != args.task_filter:
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
    p.add_argument("--task_filter", type=str, default="all", choices=["all", "mcq", "numeric", "short", "medium", "long"])
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
        "--max_keywords",
        type=int,
        default=5,
        help="最大关键词数：不超过则全保留；超过则删除与其它词 CLIP 相似度最高的 (n-max_keywords) 个",
    )
    p.add_argument(
        "--info_aggressiveness",
        type=float,
        default=0.6,
        help="信息量单一控制参数(0~1)：0时关键词均分配额，越大越偏向突出且集中的关键词",
    )
    p.add_argument("--coarse_uniform_ratio", type=float, default=0.0, help="coarse选帧中均匀抽样占比，其余预算按关键词信息量分配")
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
        "max_keywords",
        "info_aggressiveness",
        "mapped_info_peak_floor",
        "mapped_info_peak_ceiling",
        "mapped_info_prominence_center",
        "mapped_info_prominence_scale",
        "mapped_info_entropy_temperature",
        "mapped_info_mix_prominence",
        "use_preprocessed_clip_frames",
        "preprocessed_clip_fps",
        "use_subtitles",
    ]


def _ours_csv_row(
    args: argparse.Namespace,
    *,
    mapped_info_params: dict[str, float],
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
        int(args.max_keywords),
        f"{float(args.info_aggressiveness):g}",
        f"{float(mapped_info_params.get('info_peak_floor', 0.0)):g}",
        f"{float(mapped_info_params.get('info_peak_ceiling', 0.0)):g}",
        f"{float(mapped_info_params.get('info_prominence_center', 0.0)):g}",
        f"{float(mapped_info_params.get('info_prominence_scale', 0.0)):g}",
        f"{float(mapped_info_params.get('info_entropy_temperature', 0.0)):g}",
        f"{float(mapped_info_params.get('info_mix_prominence', 0.0)):g}",
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
    _log(
        f"配置: candidate_pool_fps={float(args.candidate_pool_fps):g}, "
        f"budget={budget}, quota_prescreen_alpha={int(args.quota_prescreen_alpha)}"
    )

    model, proc = load_model_and_processor(resolved_model_path, use_lora=args.use_lora, base_model=args.base_model, merge_lora=args.merge_lora)
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
            mapped_info_params=_mapped_info_params_from_aggr(float(args.info_aggressiveness)),
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
