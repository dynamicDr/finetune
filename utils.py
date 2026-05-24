from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont


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
    keywords_after_info_filter: list[dict[str, Any]] | None = None,
    selected_frame_subtitles: list[str] | None = None,
) -> None:
    if not verbose or verbose_run_dir is None:
        return

    sample_dir = verbose_run_dir / normalize_sample_id(sample_id)
    round_dir = sample_dir / f"{stage}_round_{round_id:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    selected_images = []
    frame_subs = selected_frame_subtitles or []
    for order, idx in enumerate(selected_idx):
        fid = int(frame_ids[idx])
        name = f"{order:02d}_frame_{fid}.jpg"
        img_path = round_dir / name
        imgs[idx].save(img_path)
        selected_images.append(
            {
                "index_in_pool": int(idx),
                "frame_id": fid,
                "file": name,
                "subtitle": str(frame_subs[order]) if order < len(frame_subs) else "",
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

    payload = {
        "sample_id": sample_id,
        "stage": stage,
        "round": int(round_id),
        "question": question,
        "options": options or [],
        "gt_answer": str(gt_answer),
        "raw_keywords_before_dedup": raw_keywords_before_dedup or [],
        "keywords_after_info_filter": keywords_after_info_filter or [],
        "all_keywords": all_keywords,
        "selected_count": len(selected_idx),
        "selected_images": selected_images,
        "image_keyword_clip_scores": image_keyword_scores,
        "pred_answer": str(vlm_out.get("pred_answer", "")),
        "option_probs": _to_serializable_option_probs(vlm_out.get("option_probs", {})),
        "keyword_frame_score_image": score_table_image,
        "response": str(vlm_out.get("response", "")),
        "inference_time": float(vlm_out.get("inference_time", 0.0)),
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
        peak_term = float(info_row.get("peak_term", 0.0))
        prom_term = float(info_row.get("prominence_term", 0.0))
        conc_term = float(info_row.get("concentration_term", 0.0))
        info_score = float(info_row.get("info", 0.0))
        draw.text((x0 + 8, ry0 + 6), kw_show, fill=(20, 20, 20), font=font)
        draw.text(
            (x0 + 8, ry0 + 30),
            f"peak={peak_term:.3f}  prom={prom_term:.3f}  conc={conc_term:.3f}  info={info_score:.3f}",
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


def _to_serializable_option_probs(probs: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in probs.items()}


def normalize_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id.strip()).strip("_")

