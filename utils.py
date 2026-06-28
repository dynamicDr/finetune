from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
from PIL import Image, ImageDraw, ImageFont

_LOCAL_PRETRAINED_FALLBACK_EXC: tuple[type[BaseException], ...]
try:
    from huggingface_hub.errors import LocalEntryNotFoundError

    _LOCAL_PRETRAINED_FALLBACK_EXC = (OSError, LocalEntryNotFoundError)
except ImportError:
    _LOCAL_PRETRAINED_FALLBACK_EXC = (OSError,)


def from_pretrained_local_first(
    loader: Callable[..., Any],
    model_id: str,
    *,
    log: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> Any:
    """优先从本地 HuggingFace 缓存加载，仅在本地不可用时联网。"""
    local_kwargs = {**kwargs, "local_files_only": True}
    try:
        obj = loader(model_id, **local_kwargs)
        if log:
            log(f"从本地缓存加载: {model_id}")
        return obj
    except _LOCAL_PRETRAINED_FALLBACK_EXC as exc:
        if log:
            log(f"本地缓存不可用，改为联网加载: {model_id} ({exc})")
        net_kwargs = dict(kwargs)
        net_kwargs.pop("local_files_only", None)
        return loader(model_id, **net_kwargs)


_OPTION_LETTER_PREFIX_RE = re.compile(r"^([A-Za-z])[\.\)\:\-]\s*(.*)$", re.DOTALL)

KEYWORD_EXTRACTOR_PROVIDERS: dict[str, dict[str, str]] = {
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


def format_labeled_options(options: list[str]) -> str:
    """为 MCQ 选项加上 A. B. C. … 前缀，便于模型按字母作答。"""
    lines: list[str] = []
    for i, raw in enumerate(options):
        text = str(raw).strip()
        if not text:
            continue
        m = _OPTION_LETTER_PREFIX_RE.match(text)
        if m:
            text = m.group(2).strip() or text
        lines.append(f"{chr(ord('A') + i)}. {text}")
    return "\n".join(lines)


def build_user_text(question: str, options: list[str] | None) -> str:
    if options:
        return (
            f"{question}\n\nOptions:\n"
            f"{format_labeled_options(options)}\n\n"
            "Directly answer with the option letter only. Do not explain."
        )
    return f"{question}\n\nPlease provide the numerical answer directly."


def build_user_text_with_subtitles(question: str, options: list[str] | None, subtitles: list[str] | None) -> str:
    options_text = format_labeled_options(options or []) if options else ""
    subtitle_text = "\n".join((subtitles or [])).strip() or "No subtitles available"
    return (
        "This video's subtitles are listed below:\n"
        f"{subtitle_text}\n\n"
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter of the correct option.\n"
        f"{question}\n\nOptions:\n{options_text}\n"
        "The best answer is:"
    )


def calculate_mra(pred: float, gt: float) -> float:
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, 1 - abs(pred - gt) / abs(gt))


def compute_accuracy_from_results(results: dict, task_filter: str) -> tuple[float, float]:
    avg_accuracy = 0.0
    tf = str(task_filter).strip()
    if tf in {"mcq", "short", "medium", "long"} and results["total"] > 0:
        avg_accuracy = results["correct"] / results["total"] * 100
    elif tf not in {"all", "numeric", "generation"} and results["total"] > 0:
        avg_accuracy = results["correct"] / results["total"] * 100
    elif tf == "numeric" and results["mra_count"] > 0:
        avg_accuracy = results["mra_sum"] / results["mra_count"] * 100
    elif tf == "all":
        total_score = results["correct"] + results["mra_sum"]
        total_count = results["total"] + results["mra_count"]
        if total_count > 0:
            avg_accuracy = total_score / total_count * 100
    avg_inference_time = sum(results["inference_times"]) / len(results["inference_times"]) if results["inference_times"] else 0.0
    return avg_accuracy, avg_inference_time


def compute_score_counts_for_csv(results: dict, task_filter: str) -> tuple[int, float]:
    tf = str(task_filter).strip()
    if tf in {"mcq", "short", "medium", "long"}:
        return int(results["total"]), float(results["correct"])
    if tf not in {"all", "numeric", "generation"} and results["total"] > 0:
        return int(results["total"]), float(results["correct"])
    if tf == "numeric":
        return int(results["mra_count"]), float(results["mra_sum"])
    return int(results["total"] + results["mra_count"]), float(results["correct"] + results["mra_sum"])


def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def parse_subtitle_time(time_str: str) -> float:
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def load_srt_segments(subtitle_path: str) -> list[tuple[float, float, str]]:
    p = Path(subtitle_path)
    if not p.is_file():
        return []
    if p.suffix.lower() == ".json":
        return load_lvb_json_segments(subtitle_path)
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
            start_t = parse_subtitle_time(ts0)
            end_t = parse_subtitle_time(ts1)
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


def _lvb_timestamp_to_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        return parse_subtitle_time(text)
    return float(text)


def load_lvb_json_segments(
    subtitle_path: str,
    *,
    starting_timestamp: float = 0.0,
) -> list[tuple[float, float, str]]:
    p = Path(subtitle_path)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []

    segs: list[tuple[float, float, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", item.get("line", ""))).strip()
        if not text:
            continue
        if "timestamp" in item:
            ts = item["timestamp"]
            if not isinstance(ts, (list, tuple)) or len(ts) < 2:
                continue
            start = _lvb_timestamp_to_seconds(ts[0]) - starting_timestamp
            end = _lvb_timestamp_to_seconds(ts[1]) - starting_timestamp
        else:
            start = _lvb_timestamp_to_seconds(item.get("start", 0.0)) - starting_timestamp
            end = _lvb_timestamp_to_seconds(item.get("end", start)) - starting_timestamp
        if end < start:
            end = start
        segs.append((max(0.0, start), max(0.0, end), text))
    segs.sort(key=lambda x: x[0])
    return segs


def resolve_subtitle_path(sample: Any, subtitles_dir: str | None) -> str | None:
    if isinstance(getattr(sample, "metadata", None), dict):
        meta_path = str(sample.metadata.get("subtitle_path", "")).strip()
        if meta_path and Path(meta_path).is_file():
            return meta_path
        rel_path = str(sample.metadata.get("subtitle_path_rel", "")).strip()
        if rel_path:
            explicit_dir = os.path.expanduser(subtitles_dir) if subtitles_dir else ""
            rel_candidates: list[Path] = []
            if explicit_dir:
                rel_candidates.extend(
                    [
                        Path(explicit_dir) / rel_path,
                        Path(explicit_dir) / Path(rel_path).name,
                    ]
                )
            video_path = Path(sample.video_path).expanduser().resolve()
            for root in [video_path.parent, video_path.parent.parent, video_path.parent.parent.parent]:
                rel_candidates.extend(
                    [
                        root / rel_path,
                        root / "subtitles" / rel_path,
                        root / "subtitles" / Path(rel_path).name,
                    ]
                )
            for candidate in rel_candidates:
                if candidate.is_file():
                    return str(candidate)

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


def collect_unique_subtitles_for_sample(
    sample: Any,
    num_frames: int,
    frame_sampling_method: str,
    random_seed: int | None,
    subtitles_dir: str | None,
) -> list[str]:
    subtitle_path = resolve_subtitle_path(sample, subtitles_dir=subtitles_dir)
    if not subtitle_path:
        return []
    starting_timestamp = 0.0
    if isinstance(getattr(sample, "metadata", None), dict):
        try:
            starting_timestamp = float(sample.metadata.get("starting_timestamp_for_subtitles", 0.0) or 0.0)
        except (TypeError, ValueError):
            starting_timestamp = 0.0
    cap = cv2.VideoCapture(sample.video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if total_frames <= 0 or fps <= 1e-6:
        return []
    k = min(max(1, int(num_frames)), total_frames)
    if frame_sampling_method == "random":
        import random as _random

        rng = _random.Random(random_seed)
        frame_indices = sorted(rng.sample(range(total_frames), k))
    else:
        frame_indices = [int(i * total_frames / k) for i in range(k)]
    if Path(subtitle_path).suffix.lower() == ".json":
        segments = load_lvb_json_segments(subtitle_path, starting_timestamp=starting_timestamp)
    else:
        segments = load_srt_segments(subtitle_path)
    if not segments:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for fid in frame_indices:
        t = float(fid) / fps
        for s0, s1, txt in segments:
            if s0 <= t < s1:
                if txt not in seen:
                    seen.add(txt)
                    out.append(txt)
                break
    return out


def collect_subtitles_for_frame_ids(
    sample: Any,
    sampled_frame_ids: list[int],
    subtitles_dir: str | None,
) -> tuple[list[str], list[str]]:
    subtitle_path = resolve_subtitle_path(sample, subtitles_dir=subtitles_dir)
    if not subtitle_path or not sampled_frame_ids:
        return [], ["" for _ in sampled_frame_ids]
    starting_timestamp = 0.0
    if isinstance(getattr(sample, "metadata", None), dict):
        try:
            starting_timestamp = float(sample.metadata.get("starting_timestamp_for_subtitles", 0.0) or 0.0)
        except (TypeError, ValueError):
            starting_timestamp = 0.0
    cap = cv2.VideoCapture(sample.video_path)
    if not cap.isOpened():
        return [], ["" for _ in sampled_frame_ids]
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1e-6:
        return [], ["" for _ in sampled_frame_ids]
    if Path(subtitle_path).suffix.lower() == ".json":
        segments = load_lvb_json_segments(subtitle_path, starting_timestamp=starting_timestamp)
    else:
        segments = load_srt_segments(subtitle_path)
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


def init_verbose_run_dir(verbose: bool, output_dir: Path, log_fn: Callable[[str], None]) -> Path | None:
    if not verbose:
        return None
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / ts
    if run_dir.exists():
        suffix = 1
        while (output_dir / f"{ts}_{suffix}").exists():
            suffix += 1
        run_dir = output_dir / f"{ts}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_fn(f"verbose 已开启，日志目录: {run_dir}")
    log_fn(f"verbose 选帧汇总: {run_dir / 'selected_frames_index.jsonl'}")
    return run_dir


def write_verbose_frame_selection_manifest(verbose_run_dir: Path | None, manifest: dict[str, Any]) -> None:
    if verbose_run_dir is None:
        return
    path = verbose_run_dir / "selected_frames_index.meta.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def append_verbose_frame_selection_index(
    verbose_run_dir: Path | None,
    *,
    sample_id: str,
    task_type: str | None,
    question: str,
    options: list[str] | None,
    gt_answer: Any,
    pred_answer: str,
    is_correct: bool,
    frame_ids: list[int],
    selected_idx: list[int],
    keywords: list[str],
    selection_info: dict[str, Any] | None,
    detail_rel_path: str,
) -> None:
    """向 run 目录追加一条选帧记录，便于跨样本对比 frame selection 行为。"""
    if verbose_run_dir is None:
        return

    sel_info = dict(selection_info or {})
    prescreen_ids = [int(x) for x in sel_info.get("prescreen_frame_ids", [])]
    selected_frames = []
    for order, idx in enumerate(selected_idx):
        pool_idx = int(idx)
        fid = int(frame_ids[pool_idx])
        selected_frames.append(
            {
                "order": int(order),
                "index_in_pool": pool_idx,
                "frame_id": fid,
            }
        )

    record = {
        "sample_id": str(sample_id),
        "task_type": str(task_type or ""),
        "question": str(question),
        "options": list(options or []),
        "gt_answer": str(gt_answer),
        "pred_answer": str(pred_answer),
        "is_correct": bool(is_correct),
        "keywords": list(keywords),
        "candidate_pool_size": int(len(frame_ids)),
        "selected_count": int(len(selected_idx)),
        "selected_index_in_pool": [int(i) for i in selected_idx],
        "selected_frame_ids": [int(frame_ids[i]) for i in selected_idx],
        "selected_frames": selected_frames,
        "prescreen_frame_ids": prescreen_ids,
        "prescreen_candidate_count": int(sel_info.get("prescreen_candidate_count", len(prescreen_ids))),
        "frame_selection_mode": sel_info.get("frame_selection_mode"),
        "frame_selection_method": sel_info.get("frame_selection_method"),
        "selection_config": {
            k: sel_info.get(k)
            for k in (
                "num_frames_budget",
                "candidate_pool_fps",
                "quota_prescreen_alpha",
                "quota_gamma",
                "max_keywords",
                "keyword_prompt_version",
                "keyword_extractor_model",
                "keyword_weight_strength",
                "ensure_keyword_min_coverage",
            )
            if k in sel_info
        },
        "verbose_detail_path": str(detail_rel_path),
    }
    index_path = verbose_run_dir / "selected_frames_index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def dump_verbose_round(
    verbose: bool,
    verbose_run_dir: Path | None,
    sample_id: str,
    stage: str,
    round_id: int,
    question: str,
    options: list[str] | None,
    gt_answer: Any,
    all_keywords: list[str],
    frame_ids: list[int],
    selected_idx: list[int],
    imgs: list[Image.Image],
    image_keyword_scores: list[dict[str, Any]],
    keyword_info_scores: list[dict[str, Any]] | None,
    vlm_out: dict[str, Any],
    raw_keywords_before_dedup: list[str] | None = None,
    selected_frame_subtitles: list[str] | None = None,
    selection_info: dict[str, Any] | None = None,
    task_type: str | None = None,
) -> None:
    if not verbose or verbose_run_dir is None:
        return

    sample_dir = verbose_run_dir / normalize_sample_id(sample_id)
    round_dir = sample_dir / f"{stage}_round_{round_id:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    scores_by_idx = {int(item.get("index_in_pool", -1)): item.get("keyword_scores", {}) for item in image_keyword_scores}
    selected_images = []
    frame_subs = selected_frame_subtitles or []
    for order, idx in enumerate(selected_idx):
        fid = int(frame_ids[idx])
        name = f"{order:02d}_frame_{fid}.jpg"
        img_path = round_dir / name
        imgs[idx].save(img_path)
        kw_scores = scores_by_idx.get(int(idx), {})
        top_keywords = [
            {"keyword": str(kw), "clip_score": float(score)}
            for kw, score in sorted(kw_scores.items(), key=lambda item: float(item[1]), reverse=True)[:5]
        ]
        selected_images.append(
            {
                "order": int(order),
                "index_in_pool": int(idx),
                "frame_id": fid,
                "file": name,
                "subtitle": str(frame_subs[order]) if order < len(frame_subs) else "",
                "top_keywords": top_keywords,
            }
        )

    score_table_image = _render_keyword_frame_score_image(
        round_dir=round_dir,
        question=question,
        options=options or [],
        gt_answer=str(gt_answer),
        keywords=all_keywords,
        keyword_info_scores=keyword_info_scores or [],
        selected_idx=selected_idx,
        frame_ids=frame_ids,
        imgs=imgs,
        image_keyword_scores=image_keyword_scores,
        selected_frame_subtitles=frame_subs,
    )
    time_curve_image = _render_keyword_time_curve_image(
        round_dir=round_dir,
        keywords=all_keywords,
        keyword_info_scores=keyword_info_scores or [],
        frame_ids=frame_ids,
        image_keyword_scores=image_keyword_scores,
    )

    keyword_rows = []
    for item in keyword_info_scores or []:
        keyword_rows.append(
            {
                "text": str(item.get("keyword", "")),
                "local_evidence_score": float(item.get("local_evidence_score", 0.0)),
                "info_score": float(item.get("info", 0.0)),
                "weight": float(item.get("weight", 0.0)),
                "used_for_selection": bool(item.get("used_for_selection", True)),
            }
        )

    pred_answer = str(vlm_out.get("pred_answer", ""))
    gt_answer = str(gt_answer)
    selection_payload = dict(selection_info or {})
    selection_payload["keyword_frame_score_image"] = score_table_image
    selection_payload["keyword_time_curve_image"] = time_curve_image
    payload = {
        "sample": {
            "id": sample_id,
            "question": question,
            "options": options or [],
            "ground_truth": gt_answer,
        },
        "prediction": {
            "answer": pred_answer,
            "response": str(vlm_out.get("response", "")),
            "is_correct": pred_answer.strip().upper() == gt_answer.strip().upper(),
            "inference_time_sec": float(vlm_out.get("inference_time", 0.0)),
            "option_probs": _to_serializable_option_probs(vlm_out.get("option_probs", {})),
        },
        "keywords": {
            "raw_before_dedup": raw_keywords_before_dedup or [],
            "final": keyword_rows,
        },
        "frames": {
            "candidate_count": int(len(frame_ids)),
            "selected_count": len(selected_idx),
            "selected": selected_images,
        },
        "selection": selection_payload,
    }
    with (round_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    detail_rel_path = f"{normalize_sample_id(sample_id)}/{stage}_round_{round_id:02d}/info.json"
    selection_keywords = [str(item.get("keyword", "")) for item in (keyword_info_scores or []) if item.get("used_for_selection", True)]
    if not selection_keywords:
        selection_keywords = [str(kw) for kw in all_keywords if kw != "question + options"]
    append_verbose_frame_selection_index(
        verbose_run_dir,
        sample_id=sample_id,
        task_type=task_type,
        question=question,
        options=options,
        gt_answer=gt_answer,
        pred_answer=pred_answer,
        is_correct=pred_answer.strip().upper() == gt_answer.strip().upper(),
        frame_ids=frame_ids,
        selected_idx=selected_idx,
        keywords=selection_keywords,
        selection_info=selection_info,
        detail_rel_path=detail_rel_path,
    )


def _render_keyword_frame_score_image(
    round_dir: Path,
    question: str,
    options: list[str],
    gt_answer: str,
    keywords: list[str],
    keyword_info_scores: list[dict[str, Any]],
    selected_idx: list[int],
    frame_ids: list[int],
    imgs: list[Image.Image],
    image_keyword_scores: list[dict[str, Any]],
    selected_frame_subtitles: list[str] | None = None,
) -> str:
    out_name = "keyword_frame_clip_scores.png"
    if not keywords or not selected_idx:
        return ""

    idx_order = [int(x) for x in selected_idx]
    scores_by_idx = {int(item.get("index_in_pool", -1)): item.get("keyword_scores", {}) for item in image_keyword_scores}
    matrix: list[list[float]] = []
    for kw in keywords:
        row = []
        for idx in idx_order:
            kw_scores = scores_by_idx.get(int(idx), {})
            row.append(float(kw_scores.get(kw, 0.0)))
        matrix.append(row)

    flat = [v for row in matrix for v in row]
    vmin, vmax = (min(flat), max(flat)) if flat else (0.0, 1.0)
    span = max(1e-6, vmax - vmin)

    font = ImageFont.load_default()
    pad_x, pad_y = 10, 8
    row_h = 56
    score_fmt = "{:.3f}"
    max_kw_chars = max((len(k) for k in keywords), default=8)
    left_w = min(780, max(360, max_kw_chars * 8 + 260))
    thumb_w = 112
    thumb_h = 64
    cell_w = thumb_w + 8
    header_h = thumb_h + 40
    title_h = 28
    info_lines = [f"Question: {question}"] + [f"Option {chr(ord('A') + i)}: {opt}" for i, opt in enumerate(options[:8])] + [f"Ground Truth: {gt_answer}"]
    info_row_h = 16
    info_h = 12 + len(info_lines) * info_row_h

    n_rows = len(keywords)
    n_cols = len(idx_order)
    img_w = left_w + n_cols * cell_w + pad_x * 2
    img_h = info_h + title_h + header_h + n_rows * row_h + pad_y * 2
    canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((pad_x, pad_y, img_w - pad_x, pad_y + info_h - 4), outline=(180, 180, 180), fill=(248, 248, 248))
    for i, line in enumerate(info_lines):
        show = line if len(line) <= 180 else line[:177] + "..."
        draw.text((pad_x + 8, pad_y + 4 + i * info_row_h), show, fill=(20, 20, 20), font=font)

    draw.text((pad_x, pad_y + info_h + 2), "CLIP keyword-frame similarity table", fill=(0, 0, 0), font=font)

    x0 = pad_x
    y0 = pad_y + info_h + title_h
    draw.rectangle((x0, y0, x0 + left_w, y0 + header_h), outline=(180, 180, 180), fill=(245, 245, 245))
    draw.text((x0 + 8, y0 + 12), "keyword + info-scores / frame", fill=(30, 30, 30), font=font)

    info_map = {str(item.get("keyword", "")): item for item in keyword_info_scores}

    for c, idx in enumerate(idx_order):
        cx0 = x0 + left_w + c * cell_w
        cx1 = cx0 + cell_w
        draw.rectangle((cx0, y0, cx1, y0 + header_h), outline=(180, 180, 180), fill=(245, 245, 245))
        im = imgs[idx].copy().convert("RGB")
        im.thumbnail((thumb_w, thumb_h), Image.BICUBIC)
        tx = cx0 + (cell_w - im.width) // 2
        ty = y0 + 4 + (thumb_h - im.height) // 2
        canvas.paste(im, (tx, ty))
        draw.rectangle((tx - 1, ty - 1, tx + im.width + 1, ty + im.height + 1), outline=(120, 120, 120))
        draw.text((cx0 + 6, y0 + thumb_h + 6), f"frame_{int(frame_ids[idx])}", fill=(30, 30, 30), font=font)
        sub = ""
        if selected_frame_subtitles and c < len(selected_frame_subtitles):
            sub = str(selected_frame_subtitles[c]).strip()
        if len(sub) > 22:
            sub = sub[:19] + "..."
        draw.text((cx0 + 6, y0 + thumb_h + 18), f"sub: {sub}" if sub else "sub:", fill=(60, 60, 60), font=font)

    for r, kw in enumerate(keywords):
        ry0 = y0 + header_h + r * row_h
        ry1 = ry0 + row_h
        draw.rectangle((x0, ry0, x0 + left_w, ry1), outline=(210, 210, 210), fill=(252, 252, 252))
        kw_show = kw if len(kw) <= 72 else kw[:69] + "..."
        info_row = info_map.get(kw, {})
        local_evidence_score = float(info_row.get("local_evidence_score", 0.0))
        info_score = float(info_row.get("info", 0.0))
        draw.text((x0 + 8, ry0 + 6), kw_show, fill=(20, 20, 20), font=font)
        draw.text(
            (x0 + 8, ry0 + 30),
            f"local={local_evidence_score:.3f}  info={info_score:.3f}",
            fill=(40, 40, 40),
            font=font,
        )

        for c in range(n_cols):
            cx0 = x0 + left_w + c * cell_w
            cx1 = cx0 + cell_w
            score = matrix[r][c]
            norm = max(0.0, min(1.0, (score - vmin) / span))
            color = (
                int(245 - 70 * norm),
                int(250 - 120 * norm),
                int(255 - 150 * norm),
            )
            draw.rectangle((cx0, ry0, cx1, ry1), outline=(210, 210, 210), fill=color)
            draw.text((cx0 + 8, ry0 + 8), score_fmt.format(score), fill=(10, 10, 10), font=font)

    canvas.save(round_dir / out_name)
    return out_name


def _render_keyword_time_curve_image(
    round_dir: Path,
    keywords: list[str],
    keyword_info_scores: list[dict[str, Any]],
    frame_ids: list[int],
    image_keyword_scores: list[dict[str, Any]],
) -> str:
    out_name = "keyword_time_clip_curves.png"
    if not keywords or not frame_ids or not image_keyword_scores:
        return ""

    scores_by_idx = {int(item.get("index_in_pool", -1)): item.get("keyword_scores", {}) for item in image_keyword_scores}
    idx_order = sorted([idx for idx in scores_by_idx if 0 <= idx < len(frame_ids)], key=lambda idx: int(frame_ids[idx]))
    if not idx_order:
        return ""

    series: list[tuple[str, list[float]]] = []
    for kw in keywords:
        vals = [float(scores_by_idx.get(idx, {}).get(kw, 0.0)) for idx in idx_order]
        series.append((kw, vals))

    flat = [v for _, vals in series for v in vals]
    vmin, vmax = (min(flat), max(flat)) if flat else (0.0, 1.0)
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    pad = max(1e-6, (vmax - vmin) * 0.08)
    vmin -= pad
    vmax += pad
    span = max(1e-6, vmax - vmin)

    x_values = [int(frame_ids[idx]) for idx in idx_order]
    xmin, xmax = min(x_values), max(x_values)
    xspan = max(1, xmax - xmin)

    font = ImageFont.load_default()
    width = 1500
    plot_h = 620
    legend_rows = len(keywords)
    legend_h = 28 + legend_rows * 18
    margin_l, margin_r = 90, 340
    margin_t, margin_b = 54, 70
    height = margin_t + plot_h + margin_b + legend_h
    plot_x0 = margin_l
    plot_y0 = margin_t
    plot_x1 = width - margin_r
    plot_y1 = margin_t + plot_h

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_l, 18), "CLIP score over time by keyword", fill=(0, 0, 0), font=font)

    # Axes and grid.
    draw.rectangle((plot_x0, plot_y0, plot_x1, plot_y1), outline=(80, 80, 80), fill=(252, 252, 252))
    for i in range(6):
        y = plot_y1 - int(i * plot_h / 5)
        val = vmin + i * span / 5
        draw.line((plot_x0, y, plot_x1, y), fill=(225, 225, 225))
        draw.text((12, y - 7), f"{val:.3f}", fill=(60, 60, 60), font=font)
    for i in range(6):
        x = plot_x0 + int(i * (plot_x1 - plot_x0) / 5)
        fid = xmin + int(i * xspan / 5)
        draw.line((x, plot_y0, x, plot_y1), fill=(235, 235, 235))
        draw.text((x - 24, plot_y1 + 8), str(fid), fill=(60, 60, 60), font=font)

    draw.text((plot_x0 + (plot_x1 - plot_x0) // 2 - 40, plot_y1 + 34), "frame_id / time", fill=(30, 30, 30), font=font)
    draw.text((18, plot_y0 - 24), "CLIP score", fill=(30, 30, 30), font=font)

    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        (66, 133, 244),
        (219, 68, 55),
        (244, 180, 0),
        (15, 157, 88),
        (171, 71, 188),
    ]

    def _xy(fid: int, score: float) -> tuple[int, int]:
        x = plot_x0 + int((fid - xmin) / xspan * (plot_x1 - plot_x0))
        y = plot_y1 - int((score - vmin) / span * plot_h)
        return x, y

    info_map = {str(item.get("keyword", "")): item for item in keyword_info_scores}
    for sidx, (kw, vals) in enumerate(series):
        color = palette[sidx % len(palette)]
        points = [_xy(fid, score) for fid, score in zip(x_values, vals)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        else:
            draw.line(points, fill=color, width=2)

    legend_x = plot_x1 + 18
    legend_y = plot_y0
    draw.text((legend_x, legend_y), "Legend: keyword (info)", fill=(0, 0, 0), font=font)
    for sidx, kw in enumerate(keywords):
        color = palette[sidx % len(palette)]
        y = legend_y + 22 + sidx * 18
        info = float(info_map.get(kw, {}).get("info", 0.0))
        label = kw if len(kw) <= 34 else kw[:31] + "..."
        draw.line((legend_x, y + 6, legend_x + 24, y + 6), fill=color, width=3)
        draw.text((legend_x + 32, y), f"{label} ({info:.3f})", fill=(25, 25, 25), font=font)

    canvas.save(round_dir / out_name)
    return out_name


def _to_serializable_option_probs(probs: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in probs.items()}


def normalize_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id.strip()).strip("_")


PREPROCESSED_CLIP_BASE_DIR = "/userhome/cs3/duanty/dataset_preposcess"


def resolve_preprocessed_clip_dir(
    dataset: str,
    fps: float,
    override: str = "",
    *,
    base_dir: str = PREPROCESSED_CLIP_BASE_DIR,
) -> str:
    """返回预处理 clip 帧目录的绝对路径（供多用户共享读取）。"""
    if override.strip():
        return str(Path(override).expanduser().resolve())
    return str(Path(base_dir).resolve() / dataset.strip() / f"clip_{fps:g}")


# ==================== 候选帧读取与采样 ====================
def load_preprocessed_candidate_frames(
    preprocessed_clip_dir: str,
    sample_id: str,
) -> tuple[list[int], list[Image.Image]]:
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


def frame_indices_by_target_fps(total_frames: int, src_fps: float, target_fps: float) -> list[int]:
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


def pool_positions_at_fps(n_items: int, src_fps: float, target_fps: float) -> list[int]:
    """在已有 n_items 个按 src_fps 采样的条目上，重采样到 target_fps。"""
    if n_items <= 0:
        return []
    if target_fps <= 0:
        raise ValueError(f"target_fps 必须 > 0，当前: {target_fps}")
    if not src_fps or src_fps <= 0:
        src_fps = float(target_fps)
    return frame_indices_by_target_fps(n_items, src_fps, target_fps)


def collect_video_frames_at_fps(
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
    idxs = frame_indices_by_target_fps(frame_count, src_fps, target_fps)
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


# ==================== 关键词磁盘缓存 ====================
KEYWORD_CACHE_DURATION_SUFFIXES = frozenset({"short", "medium", "long"})
DEFAULT_KEYWORD_CACHE_DIR = Path("/userhome/cs3/duanty/vqa_keyword_cache")


def keyword_cache_dataset_key(dataset: str, task_type: str | None = None) -> str:
    """Video-MME 等按 duration 分桶时，缓存目录为 videomme-short / videomme-medium / videomme-long。"""
    ds = str(dataset or "dataset").strip()
    tt = str(task_type or "").strip().lower()
    if tt in KEYWORD_CACHE_DURATION_SUFFIXES:
        return f"{ds}-{tt}"
    return ds


def sanitize_cache_component(text: str, *, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(text or "").strip()).strip("_")
    if not s:
        s = "default"
    return s[:max_len]


def resolve_keyword_cache_root(cache_dir: str | None) -> Path:
    raw = str(cache_dir or "").strip()
    if raw:
        return Path(os.path.expanduser(raw)).resolve()
    return DEFAULT_KEYWORD_CACHE_DIR.resolve()


def keyword_cache_run_dir(
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
    ext = sanitize_cache_component(extractor_model or "local")
    ds = sanitize_cache_component(keyword_cache_dataset_key(dataset, task_type))
    return (
        cache_root
        / ds
        / ext
        / f"pv{int(prompt_version)}_tk{int(target_keywords)}"
        / f"cn{int(cache_number)}"
    )


def keyword_cache_file(run_dir: Path, sample_id: str) -> Path:
    sid = normalize_sample_id(sample_id)
    return run_dir / f"{sanitize_cache_component(sid, max_len=200)}.json"


def load_keyword_cache_entry(
    path: Path,
    *,
    sample_id: str,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[str], list[str]] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        if log_fn:
            log_fn(f"关键词缓存读取失败，将重新抽取: {path} ({e})")
        return None
    if not isinstance(data, dict):
        return None
    cached_sid = str(data.get("sample_id", "")).strip()
    if cached_sid and normalize_sample_id(cached_sid) != normalize_sample_id(sample_id):
        if log_fn:
            log_fn(f"关键词缓存 sample_id 不匹配，忽略: {path}")
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


def save_keyword_cache_entry(
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


# ==================== ours 评估 CSV 日志 ====================
def ours_eval_csv_columns() -> list[str]:
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
        "visual_encoder_backend",
        "visual_encoder_model",
        "lora_path",
        "candidate_pool_fps",
        "max_keywords",
        "keyword_prompt_version",
        "keyword_extractor_model",
        "use_keyword_cache",
        "keyword_cache_number",
        "keyword_weight_strength",
        "use_preprocessed_clip_frames",
        "preprocessed_clip_fps",
        "use_subtitles",
    ]


def ours_eval_csv_row(
    args: Any,
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
    visual_encoder_model: str = "",
    visual_encoder_backend: str = "",
) -> list[Any]:
    enc_model = visual_encoder_model or str(getattr(args, "ours_clip_model_id", ""))
    enc_backend = visual_encoder_backend or (
        "blip_itc" if "blip-itm" in enc_model.strip().lower() else "clip"
    )
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
        enc_backend,
        enc_model,
        lora_path,
        f"{float(args.candidate_pool_fps):g}",
        int(args.max_keywords),
        int(args.keyword_prompt_version),
        str(args.keyword_extractor_model),
        bool(args.use_keyword_cache),
        int(args.keyword_cache_number),
        f"{float(args.keyword_weight_strength):g}",
        bool(args.use_preprocessed_clip_frames),
        f"{float(args.preprocessed_clip_fps):g}",
        bool(args.use_subtitles),
    ]


def log_ours_eval_to_csv(
    log_file: str,
    columns: list[str],
    row: list[Any],
    *,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    if not write_header:
        with path.open("r", newline="", encoding="utf-8") as rf:
            existing_header = next(csv.reader(rf), None)
        if existing_header != columns:
            if log_fn:
                log_fn(f"CSV 表头与当前版本不一致，跳过写入: {path}")
            return
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(columns)
        w.writerow(row)

