from __future__ import annotations

import os
import random
import warnings

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import BlipForImageTextRetrieval, BlipProcessor


DEFAULT_SEVILA_MODEL = "Salesforce/blip-itm-large-coco"
_SEVILA_CACHE: dict[tuple[str, str], tuple[BlipProcessor, BlipForImageTextRetrieval]] = {}


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _read_frame_rgb(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _uniform_indices(total_frames: int, n: int) -> list[int]:
    if total_frames <= 0 or n <= 0:
        return []
    n = min(n, total_frames)
    if n == total_frames:
        return list(range(total_frames))
    step = total_frames / n
    return sorted({min(total_frames - 1, int(i * step + step / 2)) for i in range(n)})


def _normalize_model_name(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"base", "salesforce/blip-itm-base-coco"}:
        return "Salesforce/blip-itm-base-coco"
    if lowered in {"large", "salesforce/blip-image-captioning-large", "salesforce/blip-itm-large-coco"}:
        return "Salesforce/blip-itm-large-coco"
    # 保持历史兼容：旧的 sevila checkpoint 路径传入时，改用 transformers 默认模型
    if lowered.endswith(".pth"):
        return DEFAULT_SEVILA_MODEL
    return name


def _compose_query(
    question: str | None,
    answer: str | None,
) -> str:
    q = (question or "").strip()
    a = (answer or "").strip()
    if q and a:
        return f"Question: {q}\nAnswer: {a}"
    return q or a


def _get_blip(model_name: str, device: str) -> tuple[BlipProcessor, BlipForImageTextRetrieval]:
    key = (model_name, device)
    if key not in _SEVILA_CACHE:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForImageTextRetrieval.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        _SEVILA_CACHE[key] = (processor, model)
    return _SEVILA_CACHE[key]


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


def sample_sevila_frames(
    video_path: str,
    num_frames: int,
    question: str | None = None,
    answer: str | None = None,
    random_seed: int | None = None,
    blip_model_name: str = DEFAULT_SEVILA_MODEL,
    blip_device: str | None = None,
    blip_batch_size: int = 16,
) -> list[Image.Image]:
    """
    Transformers-only SeViLA-style keyframe sampling:
    - coarse localization on uniform candidate frames
    - fine refinement around top anchors
    - rank by BLIP ITM score
    """
    if not os.path.isfile(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        return []
    k = max(1, min(num_frames, total_frames))

    device = _resolve_device(blip_device)
    model_name = _normalize_model_name(os.path.expanduser(blip_model_name))
    batch_size = max(1, int(blip_batch_size))
    rng = random.Random(random_seed)

    q = _compose_query(question=question, answer=answer)
    if not q:
        warnings.warn("SeViLA 未收到 query，已退化使用占位问题文本。", RuntimeWarning, stacklevel=1)
        q = "What is happening in this video?"

    cap = cv2.VideoCapture(video_path)

    def _score_indices(indices: list[int]) -> dict[int, float]:
        rgbs: list[np.ndarray] = []
        valid: list[int] = []
        for idx in indices:
            rgb = _read_frame_rgb(cap, idx)
            if rgb is None:
                continue
            rgbs.append(rgb)
            valid.append(idx)
        if not valid:
            return {}
        scores = _blip_scores(rgbs, query=q, model_name=model_name, device=device, batch_size=batch_size)
        return {idx: s for idx, s in zip(valid, scores)}

    # Stage-1: coarse localization
    coarse_indices = _uniform_indices(total_frames, min(32, total_frames))
    coarse_scores = _score_indices(coarse_indices)
    if not coarse_scores:
        cap.release()
        return []

    # Stage-2: refine around top anchors
    anchor_num = max(1, min(8, k))
    anchors = [idx for idx, _ in sorted(coarse_scores.items(), key=lambda x: x[1], reverse=True)[:anchor_num]]
    half_window = max(4, total_frames // 32)
    fine_candidates: set[int] = set(coarse_indices)
    for center in anchors:
        left = max(0, center - half_window)
        right = min(total_frames - 1, center + half_window)
        local_uniform = _uniform_indices(right - left + 1, min(8, right - left + 1))
        fine_candidates.update(left + x for x in local_uniform)
        pool = [i for i in range(left, right + 1) if i not in fine_candidates]
        if pool:
            extra = rng.sample(pool, min(3, len(pool)))
            fine_candidates.update(extra)

    fine_indices = sorted(fine_candidates)
    fine_scores = _score_indices(fine_indices)
    cap.release()
    if not fine_scores:
        return []

    selected_global = sorted(idx for idx, _ in sorted(fine_scores.items(), key=lambda x: x[1], reverse=True)[:k])

    cap = cv2.VideoCapture(video_path)
    frames: list[Image.Image] = []
    for idx in selected_global[:k]:
        rgb = _read_frame_rgb(cap, idx)
        if rgb is not None:
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames

