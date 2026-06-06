from __future__ import annotations

import json
import math
import os
import re
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


def compute_accuracy_from_results(results: dict, task_filter: str) -> tuple[float, float]:
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


def compute_score_counts_for_csv(results: dict, task_filter: str) -> tuple[int, float]:
    if task_filter in {"mcq", "short", "medium", "long"}:
        return int(results["total"]), float(results["correct"])
    if task_filter == "numeric":
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


def resolve_subtitle_path(sample: Any, subtitles_dir: str | None) -> str | None:
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
    cap = cv2.VideoCapture(sample.video_path)
    if not cap.isOpened():
        return [], ["" for _ in sampled_frame_ids]
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1e-6:
        return [], ["" for _ in sampled_frame_ids]
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
    run_dir.mkdir(parents=True, exist_ok=True)
    log_fn(f"verbose 已开启，日志目录: {run_dir}")
    return run_dir


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

