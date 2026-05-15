from __future__ import annotations

import json
import random
import re
from pathlib import Path

import cv2
from PIL import Image


def _normalize_sample_id(sample_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id.strip())
    return normalized.strip("_")


def _load_preprocessed_candidate_frames(
    preprocessed_clip_dir: str,
    sample_id: str,
) -> list[Image.Image]:
    normalized_sample_id = _normalize_sample_id(sample_id)
    if not normalized_sample_id:
        raise ValueError(f"sample_id 非法，无法定位预处理帧目录: {sample_id!r}")
    sample_dir = Path(preprocessed_clip_dir).expanduser().resolve() / normalized_sample_id
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"预处理帧目录不存在: {sample_dir}")

    meta_path = sample_dir / "metadata.json"
    image_paths: list[Path] = []
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        listed_files = meta.get("files", [])
        if isinstance(listed_files, list):
            for rel_name in listed_files:
                p = sample_dir / str(rel_name)
                if p.is_file():
                    image_paths.append(p)

    if not image_paths:
        image_paths = sorted(sample_dir.glob("frame_*.jpg"))
    if not image_paths:
        raise RuntimeError(f"预处理帧目录中没有可用图片: {sample_dir}")

    images: list[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return images


def sample_random_frames(
    video_path: str,
    num_frames: int,
    random_seed: int | None = None,
    sample_id: str | None = None,
    question: str | None = None,
    answer: str | None = None,
    use_preprocessed_clip_frames: bool = False,
    preprocessed_clip_dir: str | None = None,
) -> list[Image.Image]:
    _ = question
    _ = answer
    if num_frames <= 0:
        return []
    if use_preprocessed_clip_frames:
        if not preprocessed_clip_dir:
            raise ValueError("启用预处理 clip 帧时，必须提供 preprocessed_clip_dir。")
        if not sample_id:
            raise ValueError("启用预处理 clip 帧时，必须提供 sample_id。")
        images = _load_preprocessed_candidate_frames(
            preprocessed_clip_dir=preprocessed_clip_dir,
            sample_id=sample_id,
        )
        k = min(num_frames, len(images))
        if k <= 0:
            return []
        rng = random.Random(random_seed)
        frame_indices = sorted(rng.sample(range(len(images)), k))
        return [images[i] for i in frame_indices]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    k = min(num_frames, total_frames)
    rng = random.Random(random_seed)
    frame_indices = sorted(rng.sample(range(total_frames), k))

    frames: list[Image.Image] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

