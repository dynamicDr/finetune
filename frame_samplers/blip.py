from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
import cv2
from PIL import Image

from frame_samplers.score_select import select_frame_positions_from_scores
from utils import from_pretrained_local_first

_VLM_CACHE: dict[str, Any] = {}


def _log(msg: str) -> None:
    print(f"[blip_sampler] {msg}", flush=True)


def _load_vlm(model_id: str, device: str | None):
    try:
        import torch
        from transformers import BlipForImageTextRetrieval, BlipProcessor
    except ImportError as exc:
        raise ImportError("选帧依赖缺失，请安装 torch 和 transformers。") from exc

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = f"{model_id}::{resolved_device}"
    if cache_key in _VLM_CACHE:
        _log(f"reuse cached model: model_id={model_id}, device={resolved_device}")
        return _VLM_CACHE[cache_key]

    _log(f"loading model: model_id={model_id}, device={resolved_device}")
    processor = from_pretrained_local_first(BlipProcessor.from_pretrained, model_id, log=_log)
    model = from_pretrained_local_first(
        BlipForImageTextRetrieval.from_pretrained, model_id, log=_log
    ).to(resolved_device).eval()
    _VLM_CACHE[cache_key] = (processor, model, resolved_device, torch)
    _log("model loaded and cached")
    return _VLM_CACHE[cache_key]


def _collect_candidate_frames(
    video_path: str,
    sample_every: int,
) -> tuple[list[int], list[Image.Image]]:
    if sample_every <= 0:
        raise ValueError(f"sample_every 必须 > 0，当前为 {sample_every}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    frame_ids: list[int] = []
    images: list[Image.Image] = []
    idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_ids.append(idx)
                images.append(Image.fromarray(rgb))
            idx += 1
    finally:
        cap.release()

    return frame_ids, images


def _normalize_sample_id(sample_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id.strip())
    return normalized.strip("_")


def _load_preprocessed_candidate_frames(
    preprocessed_clip_dir: str,
    sample_id: str,
) -> tuple[list[int], list[Image.Image]]:
    normalized_sample_id = _normalize_sample_id(sample_id)
    if not normalized_sample_id:
        raise ValueError(f"sample_id 非法，无法定位预处理帧目录: {sample_id!r}")
    sample_dir = Path(preprocessed_clip_dir).expanduser().resolve() / normalized_sample_id
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"预处理帧目录不存在: {sample_dir}")

    meta_path = sample_dir / "metadata.json"
    frame_ids: list[int] = []
    image_paths: list[Path] = []

    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        listed_ids = meta.get("frame_ids", [])
        listed_files = meta.get("files", [])
        if isinstance(listed_ids, list) and isinstance(listed_files, list) and len(listed_ids) == len(listed_files):
            for fid, rel_name in zip(listed_ids, listed_files):
                p = sample_dir / str(rel_name)
                if p.is_file():
                    frame_ids.append(int(fid))
                    image_paths.append(p)

    if not image_paths:
        for p in sorted(sample_dir.glob("frame_*.jpg")):
            m = re.match(r"frame_(\d+)\.jpg$", p.name)
            if not m:
                continue
            frame_ids.append(int(m.group(1)))
            image_paths.append(p)

    if not image_paths:
        raise RuntimeError(f"预处理帧目录中没有可用图片: {sample_dir}")

    images: list[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return frame_ids, images


def _format_question_and_options(question: str | None, options: list[str] | None) -> str:
    from utils import format_labeled_options

    q = (question or "").strip()
    if not q:
        raise ValueError("BLIP 选帧需要提供 question。")
    if not options:
        raise ValueError("BLIP 选帧需要提供 options，且每个选项必须包含具体内容。")
    if any(not str(opt).strip() for opt in options):
        raise ValueError("BLIP 选帧选项不能为空。")
    return f"{q}\nOptions:\n{format_labeled_options(options)}"


def _build_query(question: str | None, options: list[str] | None, answer: str | None) -> str:
    # 需求：自动忽略答案标签，仅使用 question+options 构造检索文本。
    _ = answer
    qa_text = _format_question_and_options(question=question, options=options)
    return f"a video frame relevant to the following question and options:\n{qa_text}"


def _extract_itm_scores(outputs, torch):
    logits = getattr(outputs, "itm_score", None)
    if logits is None:
        logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("BLIP 输出中缺少 itm_score/logits，无法计算打分。")
    probs = torch.nn.functional.softmax(logits, dim=-1)
    if probs.shape[-1] >= 2:
        return probs[:, 1]
    return probs.squeeze(-1)


def _score_images_batched(images, processor, model, torch, device, query, batch_size=16):
    scores = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        _log(f"scoring image batch: start={start}, end={start + len(batch) - 1}, batch_size={len(batch)}")
        inputs = processor(
            images=batch,
            text=[query] * len(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, use_itm_head=True)
        batch_scores = _extract_itm_scores(outputs, torch)
        scores.append(batch_scores)
    return torch.cat(scores, dim=0)


def sample_blip_frames(
    video_path: str,
    num_frames: int,
    random_seed: int | None = None,
    sample_id: str | None = None,
    question: str | None = None,
    options: list[str] | None = None,
    answer: str | None = None,
    model_id: str = "Salesforce/blip-itm-base-coco",
    sample_every: int = 15,
    device: str | None = None,
    batch_size: int = 16,
    min_frame_gap: int = 30,
    use_preprocessed_clip_frames: bool = False,
    preprocessed_clip_dir: str | None = None,
    use_segment_selection: bool = True,
) -> list[Image.Image]:
    t0 = time.time()
    _ = random_seed

    _log(
        "start sampling: "
        f"video_path={video_path}, num_frames={num_frames}, sample_every={sample_every}, "
        f"batch_size={batch_size}, min_frame_gap={min_frame_gap}, "
        f"use_segment_selection={use_segment_selection}"
    )

    if num_frames <= 0:
        _log("num_frames <= 0, return []")
        return []

    frame_ids: list[int]
    images: list[Image.Image]
    if use_preprocessed_clip_frames:
        if not preprocessed_clip_dir:
            raise ValueError("启用预处理 clip 帧时，必须提供 preprocessed_clip_dir。")
        if not sample_id:
            raise ValueError("启用预处理 clip 帧时，必须提供 sample_id。")
        frame_ids, images = _load_preprocessed_candidate_frames(
            preprocessed_clip_dir=preprocessed_clip_dir,
            sample_id=sample_id,
        )
        _log(
            "loaded preprocessed candidate frames: "
            f"sample_id={sample_id}, dir={os.path.expanduser(preprocessed_clip_dir)}, count={len(images)}"
        )
    else:
        frame_ids, images = _collect_candidate_frames(video_path, sample_every=sample_every)
    if not images:
        _log("no candidate frames collected, return []")
        return []
    _log(
        f"collected candidates: count={len(images)}, first_frame={frame_ids[0]}, "
        f"last_frame={frame_ids[-1]}"
    )

    processor, model, resolved_device, torch = _load_vlm(
        model_id=model_id,
        device=device,
    )
    _log(f"using device={resolved_device}")
    query = _build_query(question=question, options=options, answer=answer)
    if answer is not None and str(answer).strip():
        _log("answer label is ignored for blip sampling")
    preview_query = query if len(query) <= 160 else query[:157] + "..."
    _log(f"text query={preview_query}")

    scores = _score_images_batched(
        images=images,
        processor=processor,
        model=model,
        torch=torch,
        device=resolved_device,
        query=query,
        batch_size=batch_size,
    )
    _log(f"score shape={tuple(scores.shape)}")

    scores_cpu = scores.detach().cpu()
    _log(
        f"score stats: min={scores_cpu.min().item():.6f}, max={scores_cpu.max().item():.6f}, "
        f"mean={scores_cpu.mean().item():.6f}"
    )
    ranked = torch.argsort(scores, descending=True).tolist()
    topn = min(5, len(ranked))
    top_pairs = [(frame_ids[i], float(scores_cpu[i].item())) for i in ranked[:topn]]
    _log(f"top-{topn} candidate frames by score={top_pairs}")

    n_candidates = len(frame_ids)
    selected_positions = select_frame_positions_from_scores(
        scores=scores,
        frame_ids=frame_ids,
        num_frames=num_frames,
        use_segment_selection=use_segment_selection,
    )
    selected = [images[i] for i in selected_positions]
    selected_frame_ids = [frame_ids[i] for i in selected_positions]
    selection_mode = "segment" if use_segment_selection else "topk"
    _log(
        f"{selection_mode} selection: candidates={n_candidates}, "
        f"selected_positions={selected_positions}"
    )

    _log(
        f"selected frames: count={len(selected)}, frame_ids={selected_frame_ids}, "
        f"elapsed={time.time() - t0:.2f}s"
    )
    return selected
