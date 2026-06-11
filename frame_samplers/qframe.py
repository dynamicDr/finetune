from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from utils import from_pretrained_local_first

_QFRAME_MODEL_CACHE: dict[str, Any] = {}


def _log(msg: str) -> None:
    print(f"[qframe_sampler] {msg}", flush=True)


def _to_feature_tensor(features, torch):
    if isinstance(features, torch.Tensor):
        return features
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(features, attr, None)
        if isinstance(val, torch.Tensor):
            if attr == "last_hidden_state" and val.ndim >= 2:
                return val[:, 0, :]
            return val
    if isinstance(features, (tuple, list)) and features:
        first = features[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"无法将特征输出转换为 Tensor，实际类型: {type(features)!r}")


def _load_clip(model_id: str, device: str | None):
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError("QFrame 依赖缺失，请安装 torch 和 transformers。") from exc

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = f"{model_id}::{resolved_device}"
    if cache_key in _QFRAME_MODEL_CACHE:
        _log(f"reuse cached model: model_id={model_id}, device={resolved_device}")
        return _QFRAME_MODEL_CACHE[cache_key]

    _log(f"loading model: model_id={model_id}, device={resolved_device}")
    processor = from_pretrained_local_first(AutoProcessor.from_pretrained, model_id, log=_log)
    model = from_pretrained_local_first(AutoModel.from_pretrained, model_id, log=_log).to(resolved_device).eval()
    _QFRAME_MODEL_CACHE[cache_key] = (processor, model, resolved_device, torch)
    _log("model loaded and cached")
    return _QFRAME_MODEL_CACHE[cache_key]


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
        if (
            isinstance(listed_ids, list)
            and isinstance(listed_files, list)
            and len(listed_ids) == len(listed_files)
        ):
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


def _uniform_positions(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if total <= count:
        return list(range(total))

    pos = [int(i * (total - 1) / (count - 1)) for i in range(count)]
    out: list[int] = []
    seen: set[int] = set()
    for p in pos:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _collect_uniform_candidates_from_video(
    video_path: str,
    candidate_count: int,
) -> tuple[list[int], list[Image.Image]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return [], []

    target_indices = _uniform_positions(total_frames, candidate_count)
    if not target_indices:
        cap.release()
        return [], []

    target_set = set(target_indices)
    frame_ids: list[int] = []
    images: list[Image.Image] = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in target_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_ids.append(idx)
                images.append(Image.fromarray(rgb))
                if len(frame_ids) >= len(target_indices):
                    break
            idx += 1
    finally:
        cap.release()
    return frame_ids, images


def _build_query(question: str | None, options: list[str] | None) -> str:
    q = (question or "").strip()
    if not q:
        return "Find frames that best support answering the video question."
    if not options:
        return q
    normalized_options = [str(opt).strip() for opt in options if str(opt).strip()]
    if not normalized_options:
        return q
    return f"{q}\nOptions:\n" + "\n".join(normalized_options)


def _encode_images_batched(images, processor, model, torch, device, batch_size=16):
    feats = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_feats = _to_feature_tensor(model.get_image_features(**inputs), torch)
        batch_feats = batch_feats / batch_feats.norm(dim=-1, keepdim=True)
        feats.append(batch_feats)
    return torch.cat(feats, dim=0)


def _resize_by_scale(image: Image.Image, scale: float) -> Image.Image:
    if math.isclose(scale, 1.0):
        return image
    w, h = image.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def _allocate_bucket_counts(
    target_total: int,
    high_ratio: int,
    mid_ratio: int,
    low_ratio: int,
) -> tuple[int, int, int]:
    if target_total <= 0:
        return 0, 0, 0

    ratios = [max(0, high_ratio), max(0, mid_ratio), max(0, low_ratio)]
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        return target_total, 0, 0

    raw = [(target_total * r) / ratio_sum for r in ratios]
    counts = [int(math.floor(v)) for v in raw]
    remain = target_total - sum(counts)
    fractional_order = sorted(
        range(3),
        key=lambda i: (raw[i] - counts[i], -i),
        reverse=True,
    )
    for i in range(remain):
        counts[fractional_order[i % 3]] += 1

    # 至少保留 1 张高分辨率帧，避免极小 num_frames 下高分桶为空。
    if counts[0] == 0 and target_total > 0:
        for donor in (2, 1):
            if counts[donor] > 0:
                counts[donor] -= 1
                counts[0] = 1
                break
        if counts[0] == 0:
            counts[0] = 1
            overflow = sum(counts) - target_total
            if overflow > 0:
                for donor in (2, 1, 0):
                    can_take = min(overflow, counts[donor] - (1 if donor == 0 else 0))
                    if can_take > 0:
                        counts[donor] -= can_take
                        overflow -= can_take
                    if overflow == 0:
                        break

    return counts[0], counts[1], counts[2]


def sample_qframe_frames(
    video_path: str,
    num_frames: int,
    random_seed: int | None = None,
    sample_id: str | None = None,
    question: str | None = None,
    options: list[str] | None = None,
    answer: str | None = None,
    model_id: str = "openai/clip-vit-base-patch32",
    candidate_pool_size: int = 128,
    temperature: float = 0.8,
    top_high_count: int = 4,
    top_mid_count: int = 8,
    top_low_count: int = 32,
    mid_scale: float = 0.5,
    low_scale: float = 0.25,
    device: str | None = None,
    batch_size: int = 16,
    use_preprocessed_clip_frames: bool = False,
    preprocessed_clip_dir: str | None = None,
) -> list[Image.Image]:
    t0 = time.time()
    _ = random_seed
    _ = answer
    _log(
        "start sampling: "
        f"video_path={video_path}, num_frames={num_frames}, candidate_pool_size={candidate_pool_size}, "
        f"bucket=[{top_high_count},{top_mid_count},{top_low_count}], "
        f"temperature={temperature}, batch_size={batch_size}"
    )

    if num_frames <= 0:
        _log("num_frames <= 0, return []")
        return []
    if candidate_pool_size <= 0:
        raise ValueError(f"candidate_pool_size 必须 > 0，当前为 {candidate_pool_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须 > 0，当前为 {batch_size}")
    if temperature <= 0:
        raise ValueError(f"temperature 必须 > 0，当前为 {temperature}")
    if min(top_high_count, top_mid_count, top_low_count) < 0:
        raise ValueError("分桶数量必须 >= 0")

    if use_preprocessed_clip_frames:
        if not preprocessed_clip_dir:
            raise ValueError("启用预处理 clip 帧时，必须提供 preprocessed_clip_dir。")
        if not sample_id:
            raise ValueError("启用预处理 clip 帧时，必须提供 sample_id。")
        all_frame_ids, all_images = _load_preprocessed_candidate_frames(
            preprocessed_clip_dir=preprocessed_clip_dir,
            sample_id=sample_id,
        )
        candidate_positions = _uniform_positions(len(all_images), candidate_pool_size)
        frame_ids = [all_frame_ids[i] for i in candidate_positions]
        images = [all_images[i] for i in candidate_positions]
        _log(
            "loaded preprocessed candidates: "
            f"sample_id={sample_id}, total={len(all_images)}, selected={len(images)}, "
            f"dir={os.path.expanduser(preprocessed_clip_dir)}"
        )
    else:
        frame_ids, images = _collect_uniform_candidates_from_video(
            video_path=video_path,
            candidate_pool_size=candidate_pool_size,
        )
    if not images:
        _log("no candidate frames, return []")
        return []

    processor, model, resolved_device, torch = _load_clip(
        model_id=model_id,
        device=device,
    )
    query = _build_query(question=question, options=options)
    preview_query = query if len(query) <= 160 else query[:157] + "..."
    _log(
        f"candidate_count={len(images)}, first_frame={frame_ids[0]}, "
        f"last_frame={frame_ids[-1]}, query={preview_query}"
    )

    image_feats = _encode_images_batched(
        images=images,
        processor=processor,
        model=model,
        torch=torch,
        device=resolved_device,
        batch_size=batch_size,
    )
    text_inputs = processor(
        text=[query],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(resolved_device)
    with torch.no_grad():
        text_feats = _to_feature_tensor(model.get_text_features(**text_inputs), torch)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    scores = (image_feats @ text_feats.T).squeeze(1)
    log_probs = torch.log_softmax(scores / temperature, dim=0)
    uniforms = torch.rand_like(log_probs).clamp_(1e-6, 1.0 - 1e-6)
    gumbel_noise = -torch.log(-torch.log(uniforms))
    perturbed_scores = log_probs + gumbel_noise
    ranked = torch.argsort(perturbed_scores, descending=True).tolist()

    target_total = min(num_frames, len(ranked))
    high_count, mid_count, low_count = _allocate_bucket_counts(
        target_total=target_total,
        high_ratio=top_high_count,
        mid_ratio=top_mid_count,
        low_ratio=top_low_count,
    )
    high_end = min(high_count, len(ranked))
    mid_end = min(high_end + mid_count, len(ranked))
    low_end = min(mid_end + low_count, len(ranked))
    high_pos = ranked[:high_end]
    mid_pos = ranked[high_end:mid_end]
    low_pos = ranked[mid_end:low_end]

    scale_by_position: dict[int, float] = {}
    for p in high_pos:
        scale_by_position[p] = 1.0
    for p in mid_pos:
        scale_by_position[p] = mid_scale
    for p in low_pos:
        scale_by_position[p] = low_scale

    ordered_positions = sorted(scale_by_position.keys(), key=lambda p: frame_ids[p])
    selected = [_resize_by_scale(images[p], scale_by_position[p]) for p in ordered_positions]
    selected_frame_ids = [frame_ids[p] for p in ordered_positions]

    _log(
        "bucket result: "
        f"high={len(high_pos)}, mid={len(mid_pos)}, low={len(low_pos)}, "
        f"selected={len(selected)}, target_total={target_total}, elapsed={time.time() - t0:.2f}s"
    )
    _log(f"time-ordered frame_ids={selected_frame_ids}")
    return selected
