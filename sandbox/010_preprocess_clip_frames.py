from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
from tqdm import tqdm

from data_loaders import get_data_loader, list_supported_datasets
from utils import PREPROCESSED_CLIP_BASE_DIR


def _normalize_sample_id(sample_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id.strip())
    return normalized.strip("_")


def _frame_indices_by_target_fps(total_frames: int, src_fps: float, target_fps: float) -> list[int]:
    if total_frames <= 0:
        return []
    if target_fps <= 0:
        raise ValueError(f"target_fps 必须 > 0，当前: {target_fps}")

    if not src_fps or src_fps <= 0:
        # 无法获取原始 fps 时，退化为逐帧读取。
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
    if 0 not in indices:
        indices.insert(0, 0)
    return indices


def _save_preprocessed_frames(
    video_path: str,
    sample_id: str,
    output_root: Path,
    target_fps: float,
    overwrite: bool,
) -> tuple[bool, int]:
    normalized_id = _normalize_sample_id(sample_id)
    if not normalized_id:
        raise ValueError(f"sample_id 非法: {sample_id!r}")

    sample_dir = output_root / normalized_id
    metadata_path = sample_dir / "metadata.json"
    if metadata_path.is_file() and not overwrite:
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return True, int(len(meta.get("frame_ids", [])))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_indices = _frame_indices_by_target_fps(total_frames, src_fps, target_fps)

    sample_dir.mkdir(parents=True, exist_ok=True)
    # 覆盖模式下清理旧帧，避免历史残留。
    if overwrite:
        for old_file in sample_dir.glob("frame_*.jpg"):
            old_file.unlink(missing_ok=True)

    saved_names: list[str] = []
    valid_indices: list[int] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_name = f"frame_{frame_idx:08d}.jpg"
        out_path = sample_dir / out_name
        if cv2.imwrite(str(out_path), frame):
            saved_names.append(out_name)
            valid_indices.append(frame_idx)
    cap.release()

    metadata = {
        "sample_id": sample_id,
        "video_path": os.path.abspath(video_path),
        "source_fps": src_fps,
        "target_fps": target_fps,
        "frame_ids": valid_indices,
        "files": saved_names,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return True, len(valid_indices)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="离线预处理 CLIP 粗采样帧。")
    p.add_argument("--dataset", type=str, default="egoschema", choices=list_supported_datasets())
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--dataset_name", type=str, default="lmms-lab/EgoSchema")
    p.add_argument("--dataset_config", type=str, default="Subset")
    p.add_argument("--no_dataset_config", action="store_true")
    p.add_argument("--video_dir", type=str, default="~/dataset/egoschema")
    p.add_argument("--fps", type=float, default=1.0, help="目标粗采样 fps。")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train_ratio",
        type=float,
        default=0.0,
        help="默认 0.0 表示不做二次切分，按所选 split 全量处理。",
    )
    p.add_argument("--use_train_split", action="store_true")
    p.add_argument("--output_base", type=str, default=PREPROCESSED_CLIP_BASE_DIR)
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否覆盖已存在的预处理结果（默认开启，可用 --no-overwrite 关闭）。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError(f"--fps 必须 > 0，当前: {args.fps}")

    video_dir = os.path.expanduser(args.video_dir)
    output_base = Path(os.path.expanduser(args.output_base))
    output_root = (output_base / args.dataset / f"clip_{args.fps:g}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    loader = get_data_loader(
        args.dataset,
        video_dir=video_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        task_filter="all",
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        no_dataset_config=args.no_dataset_config,
    )
    samples = loader.get_split_samples(
        split=args.dataset_split,
        use_train_split=args.use_train_split,
        sample_count=None,
    )
    print(
        f"[preprocess_clip_frames] dataset={args.dataset}, split={args.dataset_split}, "
        f"samples={len(samples)}, fps={args.fps:g}, output={output_root}",
        flush=True,
    )

    ok_count = 0
    frame_count = 0
    failed_samples: list[str] = []
    for sample in tqdm(samples, desc="预处理进度"):
        success, n_frames = _save_preprocessed_frames(
            video_path=sample.video_path,
            sample_id=sample.resolve_preprocess_key(),
            output_root=output_root,
            target_fps=args.fps,
            overwrite=args.overwrite,
        )
        if success:
            ok_count += 1
            frame_count += n_frames
        else:
            failed_samples.append(sample.sample_id)

    print(
        f"[preprocess_clip_frames] 完成: success={ok_count}/{len(samples)}, "
        f"saved_frames={frame_count}, failed={len(failed_samples)}",
        flush=True,
    )
    if failed_samples:
        print(
            "[preprocess_clip_frames] failed sample ids (first 20): "
            + ", ".join(failed_samples[:20]),
            flush=True,
        )


if __name__ == "__main__":
    main()
