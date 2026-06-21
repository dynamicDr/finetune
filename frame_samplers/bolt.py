from __future__ import annotations

import os
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .clip import (
    _build_query as _build_clip_query,
    _encode_images_batched as _encode_clip_images_batched,
    _load_preprocessed_candidate_frames,
    _load_vlm,
    _to_feature_tensor,
)
from .siglip2 import _load_vlm as _load_siglip2

# BOLT 超参数（写死，不暴露到入口）
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
CANDIDATE_FPS = 1.0
BATCH_SIZE = 32
POWER = -1


def _log(msg: str) -> None:
    print(f"[bolt_sampler] {msg}", flush=True)


def _collect_candidate_frames_at_fps(
    video_path: str,
    fps: float,
) -> tuple[list[int], list[Image.Image]]:
    """复刻官方 extract_feature.py 的 1fps 候选帧抽取逻辑。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0 or video_fps <= 0:
        cap.release()
        return [], []

    new_total_frames = int(total_frames / video_fps * fps)
    if new_total_frames <= 0:
        cap.release()
        return [], []

    frame_idxs = np.arange(0, new_total_frames, 1, dtype=np.float64)
    frame_idxs = np.round(frame_idxs / fps * video_fps)
    frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).astype(int)

    frame_ids: list[int] = []
    images: list[Image.Image] = []
    try:
        for frame_idx in frame_idxs.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ids.append(int(frame_idx))
            images.append(Image.fromarray(rgb))
    finally:
        cap.release()
    return frame_ids, images


def _compute_clip_similarity(
    image_feats,
    text_feats,
    model: Any,
    torch,
) -> np.ndarray:
    """复刻官方 CLIP.compute_similarity：归一化余弦 + logit_scale。"""
    video_features = image_feats / image_feats.norm(p=2, dim=-1, keepdim=True)
    text_features = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
    logits_per_text = torch.matmul(text_features, video_features.t())
    if hasattr(model, "logit_scale"):
        logits_per_text = logits_per_text * model.logit_scale.exp().to(text_features.device)
    return logits_per_text.detach().cpu().squeeze(0).float().numpy()


def _compute_siglip2_similarity(image_feats, text_feats) -> np.ndarray:
    scores = (image_feats @ text_feats.T).squeeze(1)
    return scores.detach().cpu().float().numpy()


def _inverse_transform_sampling(score: np.ndarray, n: int, power: float = -1) -> np.ndarray:
    """复刻官方 inverse_transform_sampling。"""
    score = score - np.min(score)
    score = score / np.max(score)

    if power != -1:
        score = score**power

    probabilities = score / np.sum(score)
    cdf = np.cumsum(probabilities)
    uniform_sampling = np.linspace(1 / n, 1 - 1 / n, n)
    sampled_indices = np.searchsorted(cdf, uniform_sampling)
    return sampled_indices.astype(int)


def sample_bolt_frames(
    video_path: str,
    num_frames: int,
    random_seed: int | None = None,
    sample_id: str | None = None,
    question: str | None = None,
    options: list[str] | None = None,
    answer: str | None = None,
    extract_feature_model: str = "clip",
    use_preprocessed_clip_frames: bool = False,
    preprocessed_clip_dir: str | None = None,
) -> list[Image.Image]:
    t0 = time.time()
    _ = random_seed
    extract_feature_model = extract_feature_model.strip().lower()
    if extract_feature_model not in {"clip", "siglip2"}:
        raise ValueError(
            f"BOLT 不支持的 extract_feature_model: {extract_feature_model}，可选 clip/siglip2"
        )

    model_id = CLIP_MODEL_ID if extract_feature_model == "clip" else SIGLIP2_MODEL_ID
    _log(
        "start sampling: "
        f"video_path={video_path}, num_frames={num_frames}, fps={CANDIDATE_FPS}, "
        f"batch_size={BATCH_SIZE}, power={POWER}, extract_feature_model={extract_feature_model}, "
        f"model_id={model_id}"
    )

    if num_frames <= 0:
        _log("num_frames <= 0, return []")
        return []

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
        frame_ids, images = _collect_candidate_frames_at_fps(
            video_path=video_path,
            fps=CANDIDATE_FPS,
        )
    if not images:
        _log("no candidate frames collected, return []")
        return []

    _log(
        f"collected candidates: count={len(images)}, first_frame={frame_ids[0]}, "
        f"last_frame={frame_ids[-1]}"
    )

    if extract_feature_model == "clip":
        processor, model, resolved_device, torch = _load_vlm(
            model_id=CLIP_MODEL_ID,
            device=None,
        )
        query = _build_clip_query(question=question, options=options, answer=answer)
        preview_query = query if len(query) <= 160 else query[:157] + "..."
        _log(f"text query={preview_query}")

        image_feats = _encode_clip_images_batched(
            images=images,
            processor=processor,
            model=model,
            torch=torch,
            device=resolved_device,
            batch_size=BATCH_SIZE,
        )
        _log(f"image feature shape={tuple(image_feats.shape)}")

        text_inputs = processor(
            text=[query],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(resolved_device)
        with torch.no_grad():
            text_feats = _to_feature_tensor(model.get_text_features(**text_inputs), torch)
        _log(f"text feature shape={tuple(text_feats.shape)}")
        score = _compute_clip_similarity(image_feats, text_feats, model, torch)
    else:
        processor, model, resolved_device, torch = _load_siglip2(
            model_id=SIGLIP2_MODEL_ID,
            device=None,
        )
        query = _build_clip_query(question=question, options=options, answer=answer)
        preview_query = query if len(query) <= 160 else query[:157] + "..."
        _log(f"text query={preview_query}")

        image_feats = _encode_clip_images_batched(
            images=images,
            processor=processor,
            model=model,
            torch=torch,
            device=resolved_device,
            batch_size=BATCH_SIZE,
        )
        _log(f"image feature shape={tuple(image_feats.shape)}")

        text_inputs = processor(
            text=[query],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(resolved_device)
        with torch.no_grad():
            text_feats = _to_feature_tensor(model.get_text_features(**text_inputs), torch)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        _log(f"text feature shape={tuple(text_feats.shape)}")
        score = _compute_siglip2_similarity(image_feats, text_feats)

    _log(
        f"score stats: min={score.min():.6f}, max={score.max():.6f}, mean={score.mean():.6f}"
    )

    n_select = min(num_frames, len(images))
    sampled_positions = _inverse_transform_sampling(score, n_select, power=POWER)
    selected_positions = [int(i) for i in sampled_positions.tolist()]
    selected = [images[i] for i in selected_positions]
    selected_frame_ids = [frame_ids[i] for i in selected_positions]

    _log(
        f"selected frames: count={len(selected)}, frame_ids={selected_frame_ids}, "
        f"elapsed={time.time() - t0:.2f}s"
    )
    return selected
