#!/usr/bin/env python3
"""016: Video-MME baseline 选帧时间轴热力图（方法 × 时间，深色=选中帧）。"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_loaders import get_data_loader
from data_loaders.base import VQASample
from frame_samplers.aks import _extract_scores, _resolve_device, _select_frame_indices
from frame_samplers.bolt import (
    BATCH_SIZE as BOLT_BATCH_SIZE,
    CANDIDATE_FPS as BOLT_CANDIDATE_FPS,
    CLIP_MODEL_ID as BOLT_CLIP_MODEL_ID,
    POWER as BOLT_POWER,
    SIGLIP2_MODEL_ID as BOLT_SIGLIP2_MODEL_ID,
    _collect_candidate_frames_at_fps,
    _compute_clip_similarity,
    _compute_siglip2_similarity,
    _inverse_transform_sampling,
)
from frame_samplers.bolt import _load_preprocessed_candidate_frames as bolt_load_preprocessed
from frame_samplers.blip import (
    _build_query as blip_build_query,
    _collect_candidate_frames as blip_collect_candidate_frames,
    _load_preprocessed_candidate_frames as blip_load_preprocessed,
    _load_vlm as blip_load_vlm,
    _score_images_batched as blip_score_images_batched,
)
from frame_samplers.clip import (
    _build_query as clip_build_query,
    _collect_candidate_frames as clip_collect_candidate_frames,
    _encode_images_batched as clip_encode_images_batched,
    _load_preprocessed_candidate_frames as clip_load_preprocessed,
    _load_vlm as clip_load_vlm,
    _to_feature_tensor,
)
from frame_samplers.score_select import select_frame_positions_from_scores
from frame_samplers.siglip2 import (
    _build_query as siglip2_build_query,
    _collect_candidate_frames as siglip2_collect_candidate_frames,
    _encode_images_batched as siglip2_encode_images_batched,
    _load_preprocessed_candidate_frames as siglip2_load_preprocessed,
    _load_vlm as siglip2_load_vlm,
)
from utils import load_preprocessed_candidate_frames, resolve_preprocessed_clip_dir

SEED = 42
DATASET = "videomme"
NUM_SAMPLES = 10
NUM_FRAMES_GRID = [16, 32]
NUM_TIME_BINS = 200
USE_PREPROCESSED_CLIP_FRAMES = True
PREPROCESSED_CLIP_FPS = 1.0
FOCUS_BLIP_MODEL_NAME = "Salesforce/blip-itm-base-coco"
OUT_DIR = Path(__file__).resolve().parent / "016_outputs"

METHODS: list[tuple[str, str]] = [
    ("uniform", "uniform"),
    ("aks-blip", "aks-blip"),
    ("aks-clip", "aks-clip"),
    ("bolt-clip", "bolt-clip"),
    ("bolt-siglip2", "bolt-siglip2"),
    ("clip-new", "clip-new"),
    ("blip-new", "blip-new"),
    ("siglip2-new", "siglip2-new"),
]


def get_video_meta(video_path: str) -> tuple[int, float, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    duration_sec = (total_frames / fps) if fps > 0 else 0.0
    return total_frames, fps, duration_sec


def _uniform_frame_ids(
    *,
    video_path: str,
    sample_id: str,
    num_frames: int,
    preprocessed_clip_dir: str | None,
) -> list[int]:
    if USE_PREPROCESSED_CLIP_FRAMES:
        frame_ids, images = load_preprocessed_candidate_frames(preprocessed_clip_dir or "", sample_id)
        k = min(num_frames, len(images))
        if k <= 0:
            return []
        positions = [int(i * len(images) / k) for i in range(k)]
        return sorted({frame_ids[p] for p in positions})
    total_frames, _, _ = get_video_meta(video_path)
    if total_frames <= 0:
        return []
    return [int(i * total_frames / num_frames) for i in range(num_frames)]


def _score_based_frame_ids(
    *,
    method: str,
    video_path: str,
    sample_id: str,
    num_frames: int,
    question: str | None,
    options: list[str] | None,
    answer: str | None,
    preprocessed_clip_dir: str | None,
) -> list[int]:
    use_segment = method in {"clip", "blip", "siglip2"}
    if method in {"clip", "clip-new"}:
        load_preprocessed = clip_load_preprocessed
        collect_candidates = clip_collect_candidate_frames
        load_vlm = clip_load_vlm
        model_id = "openai/clip-vit-base-patch32"
        build_query = clip_build_query
        encode_images = clip_encode_images_batched
        use_segment = method == "clip"
    elif method in {"blip", "blip-new"}:
        load_preprocessed = blip_load_preprocessed
        collect_candidates = blip_collect_candidate_frames
        load_vlm = blip_load_vlm
        model_id = FOCUS_BLIP_MODEL_NAME
        use_segment = method == "blip"
    elif method in {"siglip2", "siglip2-new"}:
        load_preprocessed = siglip2_load_preprocessed
        collect_candidates = siglip2_collect_candidate_frames
        load_vlm = siglip2_load_vlm
        model_id = "google/siglip2-base-patch16-224"
        build_query = siglip2_build_query
        encode_images = siglip2_encode_images_batched
        use_segment = method == "siglip2"
    else:
        raise ValueError(f"unsupported score method: {method}")

    if USE_PREPROCESSED_CLIP_FRAMES:
        frame_ids, images = load_preprocessed(preprocessed_clip_dir or "", sample_id)
    else:
        frame_ids, images = collect_candidates(video_path, sample_every=15)
    if not images:
        return []

    if method in {"blip", "blip-new"}:
        processor, model, device, torch = load_vlm(model_id=model_id, device=None)
        query = blip_build_query(question=question, options=options, answer=answer)
        scores = blip_score_images_batched(
            images=images,
            processor=processor,
            model=model,
            torch=torch,
            device=device,
            query=query,
            batch_size=16,
        )
    else:
        processor, model, device, torch = load_vlm(model_id=model_id, device=None)
        query = build_query(question=question, options=options, answer=answer)
        image_feats = encode_images(
            images=images,
            processor=processor,
            model=model,
            torch=torch,
            device=device,
            batch_size=16,
        )
        text_inputs = processor(
            text=[query],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_feats = _to_feature_tensor(model.get_text_features(**text_inputs), torch)
        if method in {"clip", "clip-new"}:
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        else:
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        scores = image_feats @ text_feats.T
        scores = scores.squeeze(1)

    selected_positions = select_frame_positions_from_scores(
        scores=scores,
        frame_ids=frame_ids,
        num_frames=num_frames,
        use_segment_selection=use_segment,
    )
    return [frame_ids[i] for i in selected_positions]


def _bolt_frame_ids(
    *,
    extract_feature_model: str,
    video_path: str,
    sample_id: str,
    num_frames: int,
    question: str | None,
    options: list[str] | None,
    answer: str | None,
    preprocessed_clip_dir: str | None,
) -> list[int]:
    if USE_PREPROCESSED_CLIP_FRAMES:
        frame_ids, images = bolt_load_preprocessed(preprocessed_clip_dir or "", sample_id)
    else:
        frame_ids, images = _collect_candidate_frames_at_fps(video_path, fps=BOLT_CANDIDATE_FPS)
    if not images:
        return []

    if extract_feature_model == "clip":
        processor, model, device, torch = clip_load_vlm(model_id=BOLT_CLIP_MODEL_ID, device=None)
        query = clip_build_query(question=question, options=options, answer=answer)
        image_feats = clip_encode_images_batched(
            images=images,
            processor=processor,
            model=model,
            torch=torch,
            device=device,
            batch_size=BOLT_BATCH_SIZE,
        )
        text_inputs = processor(
            text=[query],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_feats = _to_feature_tensor(model.get_text_features(**text_inputs), torch)
        score = _compute_clip_similarity(image_feats, text_feats, model, torch)
    else:
        processor, model, device, torch = siglip2_load_vlm(model_id=BOLT_SIGLIP2_MODEL_ID, device=None)
        query = siglip2_build_query(question=question, options=options, answer=answer)
        image_feats = siglip2_encode_images_batched(
            images=images,
            processor=processor,
            model=model,
            torch=torch,
            device=device,
            batch_size=BOLT_BATCH_SIZE,
        )
        text_inputs = processor(
            text=[query],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_feats = _to_feature_tensor(model.get_text_features(**text_inputs), torch)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        score = _compute_siglip2_similarity(image_feats, text_feats)

    n_select = min(num_frames, len(images))
    sampled_positions = _inverse_transform_sampling(score, n_select, power=BOLT_POWER)
    return [frame_ids[int(i)] for i in sampled_positions.tolist()]


def _aks_frame_ids(
    *,
    extract_feature_model: str,
    video_path: str,
    sample_id: str,
    num_frames: int,
    question: str | None,
    options: list[str] | None,
    answer: str | None,
    preprocessed_clip_dir: str | None,
) -> list[int]:
    scores, frame_num = _extract_scores(
        video_path=video_path,
        sample_id=sample_id,
        question=question,
        options=options,
        answer=answer,
        extract_feature_model=extract_feature_model,
        device=_resolve_device(None),
        blip_model_id=FOCUS_BLIP_MODEL_NAME,
        use_preprocessed_clip_frames=USE_PREPROCESSED_CLIP_FRAMES,
        preprocessed_clip_dir=preprocessed_clip_dir,
    )
    if not scores or not frame_num:
        return []
    selected = _select_frame_indices(
        scores=scores,
        frame_num=frame_num,
        max_num_frames=num_frames,
        ratio=1,
        t1=0.8,
        t2=-100.0,
        all_depth=5,
    )
    return sorted(selected[:num_frames])


def select_frame_ids_for_method(
    method: str,
    sample: VQASample,
    num_frames: int,
    preprocessed_clip_dir: str | None,
) -> list[int]:
    sample_id = sample.resolve_preprocess_key()
    common = dict(
        video_path=sample.video_path,
        sample_id=sample_id,
        num_frames=num_frames,
        preprocessed_clip_dir=preprocessed_clip_dir,
    )
    qa = dict(
        question=sample.question,
        options=sample.options,
        answer=str(sample.answer),
    )
    if method == "uniform":
        return _uniform_frame_ids(**common)
    if method == "aks-blip":
        return _aks_frame_ids(extract_feature_model="blip", **common, **qa)
    if method == "aks-clip":
        return _aks_frame_ids(extract_feature_model="clip", **common, **qa)
    if method == "bolt-clip":
        return _bolt_frame_ids(extract_feature_model="clip", **common, **qa)
    if method == "bolt-siglip2":
        return _bolt_frame_ids(extract_feature_model="siglip2", **common, **qa)
    if method == "clip-new":
        return _score_based_frame_ids(method="clip-new", **common, **qa)
    if method == "blip-new":
        return _score_based_frame_ids(method="blip-new", **common, **qa)
    if method == "siglip2-new":
        return _score_based_frame_ids(method="siglip2-new", **common, **qa)
    raise ValueError(f"未知方法: {method}")


def frame_ids_to_time_bins(frame_ids: list[int], total_frames: int, num_bins: int) -> np.ndarray:
    mat = np.zeros(num_bins, dtype=np.float32)
    if total_frames <= 0 or not frame_ids:
        return mat
    denom = max(total_frames - 1, 1)
    for fid in frame_ids:
        fid_clamped = int(np.clip(fid, 0, total_frames - 1))
        bin_idx = int(round(fid_clamped / denom * (num_bins - 1)))
        mat[bin_idx] = 1.0
    return mat


def build_heatmap_matrix(
    selections: dict[str, list[int]],
    method_order: list[str],
    total_frames: int,
    num_bins: int,
) -> np.ndarray:
    mat = np.zeros((len(method_order), num_bins), dtype=np.float32)
    for row, method in enumerate(method_order):
        mat[row] = frame_ids_to_time_bins(selections.get(method, []), total_frames, num_bins)
    return mat


def plot_sample_heatmap(
    mat: np.ndarray,
    method_labels: list[str],
    duration_sec: float,
    sample: VQASample,
    num_frames: int,
    out_path: Path,
) -> None:
    fig_h = max(4.5, 0.55 * len(method_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="binary", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_yticks(range(len(method_labels)))
    ax.set_yticklabels(method_labels, fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Method")
    title_q = sample.question.strip().replace("\n", " ")
    if len(title_q) > 90:
        title_q = title_q[:87] + "..."
    ax.set_title(
        f"Frame selection heatmap | {sample.sample_id} | {num_frames}f\n"
        f"task={sample.task_type} | duration={duration_sec:.1f}s | {title_q}",
        fontsize=10,
    )
    if duration_sec > 0:
        tick_pos = np.linspace(0, mat.shape[1] - 1, 6)
        tick_labels = [f"{t * duration_sec / max(mat.shape[1] - 1, 1):.0f}" for t in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["not selected", "selected"])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_dir = OUT_DIR / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_clip_dir = resolve_preprocessed_clip_dir(
        DATASET, PREPROCESSED_CLIP_FPS, ""
    )
    loader = get_data_loader(DATASET, seed=SEED, task_filter="all")
    all_samples = loader._convert_all("test")
    if len(all_samples) < NUM_SAMPLES:
        raise RuntimeError(f"样本不足: {len(all_samples)} < {NUM_SAMPLES}")
    rng = random.Random(SEED)
    samples = rng.sample(all_samples, NUM_SAMPLES)
    samples_meta = [
        {
            "sample_id": s.sample_id,
            "task_type": s.task_type,
            "video_path": s.video_path,
            "question": s.question,
        }
        for s in samples
    ]
    (OUT_DIR / "selected_samples.json").write_text(
        json.dumps(samples_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    method_keys = [m for m, _ in METHODS]
    method_labels = [label for _, label in METHODS]
    selections_records: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="samples"):
        total_frames, fps, duration_sec = get_video_meta(sample.video_path)
        for num_frames in NUM_FRAMES_GRID:
            selections: dict[str, list[int]] = {}
            for method_key, method_label in tqdm(
                METHODS, desc=f"{sample.sample_id[:12]}@{num_frames}f", leave=False
            ):
                frame_ids = select_frame_ids_for_method(
                    method_key,
                    sample,
                    num_frames,
                    preprocessed_clip_dir,
                )
                selections[method_label] = frame_ids
                selections_records.append(
                    {
                        "sample_id": sample.sample_id,
                        "task_type": sample.task_type,
                        "num_frames": num_frames,
                        "method": method_label,
                        "selected_frame_ids": frame_ids,
                        "total_frames": total_frames,
                        "fps": fps,
                        "duration_sec": duration_sec,
                    }
                )
            mat = build_heatmap_matrix(selections, method_labels, total_frames, NUM_TIME_BINS)
            safe_id = sample.sample_id.replace("/", "_")
            out_path = heatmap_dir / f"{safe_id}_{num_frames}f.png"
            plot_sample_heatmap(mat, method_labels, duration_sec, sample, num_frames, out_path)

    with (OUT_DIR / "selections.jsonl").open("w", encoding="utf-8") as f:
        for rec in selections_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "seed": SEED,
        "dataset": DATASET,
        "num_samples": NUM_SAMPLES,
        "num_frames_grid": NUM_FRAMES_GRID,
        "methods": method_labels,
        "use_preprocessed_clip_frames": USE_PREPROCESSED_CLIP_FRAMES,
        "preprocessed_clip_dir": preprocessed_clip_dir,
        "num_heatmaps": len(list(heatmap_dir.glob("*.png"))),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"热力图目录: {heatmap_dir}")


if __name__ == "__main__":
    main()
