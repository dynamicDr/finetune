from __future__ import annotations

import random
import warnings
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import BlipForImageTextRetrieval, BlipProcessor

try:
    import cv2  # type: ignore
    _CV2_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = e


_BLIP_CACHE: dict[tuple[str, str], tuple[BlipProcessor, BlipForImageTextRetrieval]] = {}


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_model_name(model_name: str) -> str:
    lowered = model_name.strip().lower()
    if lowered in {"base", "salesforce/blip-itm-base-coco"}:
        return "Salesforce/blip-itm-base-coco"
    if lowered in {"large", "salesforce/blip-itm-large-coco", "salesforce/blip-image-captioning-large"}:
        return "Salesforce/blip-itm-large-coco"
    return model_name


def _compose_query(question: str | None, answer: str | None) -> str:
    q = (question or "").strip()
    a = (answer or "").strip()
    if q and a:
        return f"Question: {q}\nAnswer: {a}"
    return q or a


def _read_frame_rgb(cap: Any, frame_idx: int) -> np.ndarray | None:
    if cv2 is None:
        raise ImportError("OpenCV(cv2) 导入失败，无法使用 focus 选帧。") from _CV2_IMPORT_ERROR
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _get_blip(model_name: str, device: str) -> tuple[BlipProcessor, BlipForImageTextRetrieval]:
    key = (model_name, device)
    if key not in _BLIP_CACHE:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForImageTextRetrieval.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        _BLIP_CACHE[key] = (processor, model)
    return _BLIP_CACHE[key]


def _blip_scores(
    frame_rgbs: list[np.ndarray],
    query: str,
    model_name: str,
    device: str,
    batch_size: int,
) -> list[float]:
    processor, model = _get_blip(model_name, device)
    similarities: list[float] = []
    with torch.no_grad():
        for i in range(0, len(frame_rgbs), batch_size):
            batch = frame_rgbs[i : i + batch_size]
            if not batch:
                continue
            images = [Image.fromarray(x) for x in batch]
            inputs = processor(images=images, text=[query] * len(images), return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs, use_itm_head=True)
            logits = outputs.itm_score if hasattr(outputs, "itm_score") else outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
            similarities.extend(probs.detach().cpu().tolist())
    return [float(v) for v in similarities]


def _uniform_indices(total_frames: int, n: int) -> list[int]:
    if total_frames <= 0 or n <= 0:
        return []
    n = min(n, total_frames)
    if n == total_frames:
        return list(range(total_frames))
    step = total_frames / n
    return sorted({min(total_frames - 1, int(i * step + step / 2)) for i in range(n)})


def sample_focus_frames(
    video_path: str,
    num_frames: int,
    question: str | None = None,
    answer: str | None = None,
    random_seed: int | None = None,
    blip_model_name: str = "Salesforce/blip-itm-large-coco",
    blip_device: str | None = None,
    blip_batch_size: int = 16,
) -> list[Image.Image]:
    if cv2 is None:
        raise ImportError(
            "OpenCV(cv2) 当前环境不可用，focus 选帧无法运行。"
            "请安装/修复 opencv-python 或 opencv-python-headless。"
        ) from _CV2_IMPORT_ERROR
    if not video_path:
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or num_frames <= 0:
        cap.release()
        return []

    k = min(num_frames, total_frames)
    device = _resolve_device(blip_device)
    model_name = _normalize_model_name(blip_model_name)
    query_text = _compose_query(question=question, answer=answer)
    if not query_text:
        warnings.warn("FOCUS 未收到查询文本，退化为默认问题。", RuntimeWarning, stacklevel=1)
        query_text = "What is happening in this video?"

    rng = random.Random(random_seed)

    # Stage-1: coarse scan
    coarse_count = min(32, total_frames)
    coarse_indices = _uniform_indices(total_frames, coarse_count)
    coarse_rgbs: list[np.ndarray] = []
    coarse_valid: list[int] = []
    for idx in coarse_indices:
        rgb = _read_frame_rgb(cap, idx)
        if rgb is not None:
            coarse_rgbs.append(rgb)
            coarse_valid.append(idx)
    if not coarse_valid:
        cap.release()
        return []
    coarse_scores = _blip_scores(coarse_rgbs, query_text, model_name, device, max(1, int(blip_batch_size)))
    coarse_rank = sorted(zip(coarse_valid, coarse_scores), key=lambda x: x[1], reverse=True)

    # Stage-2: refine around top anchors
    anchor_count = max(1, min(8, k))
    anchors = [idx for idx, _ in coarse_rank[:anchor_count]]
    half_window = max(4, total_frames // 32)
    refine_set: set[int] = set(coarse_valid)
    for center in anchors:
        left = max(0, center - half_window)
        right = min(total_frames - 1, center + half_window)
        local = _uniform_indices(right - left + 1, 8)
        refine_set.update(left + x for x in local)
        pool = [i for i in range(left, right + 1) if i not in refine_set]
        if pool:
            extra = rng.sample(pool, min(3, len(pool)))
            refine_set.update(extra)

    refine_indices = sorted(refine_set)
    refine_rgbs: list[np.ndarray] = []
    refine_valid: list[int] = []
    for idx in refine_indices:
        rgb = _read_frame_rgb(cap, idx)
        if rgb is not None:
            refine_rgbs.append(rgb)
            refine_valid.append(idx)
    cap.release()
    if not refine_valid:
        return []

    refine_scores = _blip_scores(refine_rgbs, query_text, model_name, device, max(1, int(blip_batch_size)))
    top = sorted(zip(refine_valid, refine_scores), key=lambda x: x[1], reverse=True)[:k]
    final_indices = sorted(idx for idx, _ in top)

    cap = cv2.VideoCapture(video_path)
    frames: list[Image.Image] = []
    for idx in final_indices:
        rgb = _read_frame_rgb(cap, idx)
        if rgb is not None:
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames