from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from tqdm import tqdm

from data_loaders import (
    apply_dataset_cli_defaults,
    dataset_uses_vl_pixel_limits,
    get_data_loader,
    list_supported_datasets,
)
from data_loaders.base import VQASample
from utils import normalize_sample_id
from vl_common import load_model_and_processor


def parse_args():
    p = argparse.ArgumentParser(description="按每秒1帧粗采样提取视频 embedding 并保存到本地")
    p.add_argument("--dataset", type=str, default="videomme", choices=list_supported_datasets())
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--dataset_name", type=str, default="lmms-lab/Video-MME")
    p.add_argument("--dataset_config", type=str, default="full")
    p.add_argument("--no_dataset_config", action="store_true")
    p.add_argument("--video_dir", type=str, default="~/dataset/Video-MME")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="embedding 输出目录；默认在 video_dir 下的 video_embeddings_1fps",
    )
    p.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_videos", type=int, default=None, help="仅提取前N个去重后视频，默认全量")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_train_split", action="store_true")
    p.add_argument(
        "--use_preprocessed_clip_frames",
        action="store_true",
        help="从离线预处理目录读取 1fps 粗采样帧（需先跑 sandbox/010_preprocess_clip_frames.py）",
    )
    p.add_argument(
        "--preprocessed_clip_fps",
        type=float,
        default=1.0,
        help="预处理目录命名中的 fps，默认对应 dataset_preposcess/{dataset}/clip_1",
    )
    p.add_argument(
        "--preprocessed_clip_dir",
        type=str,
        default="",
        help="预处理帧根目录；为空时自动使用 /userhome/cs3/duanty/dataset_preposcess/{dataset}/clip_{fps}",
    )
    return p.parse_args()


def _safe_name(video_path: str) -> str:
    stem = Path(video_path).stem
    stem = re.sub(r"[^0-9A-Za-z._-]", "_", stem)
    return stem or "video"


def _load_preprocessed_candidate_frames(
    preprocessed_clip_dir: str,
    sample_id: str,
) -> tuple[list[int], list[Image.Image], float, int]:
    normalized_sample_id = normalize_sample_id(sample_id)
    if not normalized_sample_id:
        raise ValueError(f"sample_id 非法，无法定位预处理帧目录: {sample_id!r}")
    sample_dir = Path(preprocessed_clip_dir).expanduser().resolve() / normalized_sample_id
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"预处理帧目录不存在: {sample_dir}")

    meta_path = sample_dir / "metadata.json"
    frame_ids: list[int] = []
    image_paths: list[Path] = []
    source_fps = 0.0
    total_frames = 0

    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        source_fps = float(meta.get("source_fps") or 0.0)
        listed_ids = meta.get("frame_ids", [])
        listed_files = meta.get("files", [])
        if isinstance(listed_ids, list) and isinstance(listed_files, list) and len(listed_ids) == len(listed_files):
            for fid, rel_name in zip(listed_ids, listed_files):
                p = sample_dir / str(rel_name)
                if p.is_file():
                    frame_ids.append(int(fid))
                    image_paths.append(p)
        if frame_ids:
            total_frames = int(frame_ids[-1]) + 1

    if not image_paths:
        for p in sorted(sample_dir.glob("frame_*.jpg")):
            m = re.match(r"frame_(\d+)\.jpg$", p.name)
            if not m:
                continue
            frame_ids.append(int(m.group(1)))
            image_paths.append(p)
        if frame_ids:
            total_frames = int(frame_ids[-1]) + 1

    if not image_paths:
        raise RuntimeError(f"预处理帧目录中没有可用图片: {sample_dir}")

    images: list[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return frame_ids, images, source_fps, total_frames


def sample_one_frame_per_second(video_path: str) -> tuple[list[Image.Image], list[int], float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], 0.0, 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 1.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return [], [], fps, total_frames

        duration_sec = int(total_frames / fps)
        indices = [int(round(i * fps)) for i in range(duration_sec + 1)]
        indices = [min(max(0, idx), total_frames - 1) for idx in indices]
        indices = sorted(set(indices))

        frames: list[Image.Image] = []
        valid_indices: list[int] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            valid_indices.append(idx)
        return frames, valid_indices, fps, total_frames
    finally:
        cap.release()


def _to_dtype(tensor: torch.Tensor, save_dtype: str) -> torch.Tensor:
    if save_dtype == "float16":
        return tensor.to(torch.float16)
    if save_dtype == "bfloat16":
        return tensor.to(torch.bfloat16)
    return tensor.to(torch.float32)


def extract_visual_embeddings(model, processor, frames: list[Image.Image]) -> torch.Tensor:
    content: list[dict[str, Any]] = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": "Describe this video briefly."})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to(model.device)

    image_grid_thw = model_inputs.get("image_grid_thw")
    with torch.no_grad():
        image_outputs = model.get_image_features(
            pixel_values=model_inputs["pixel_values"],
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )

    image_embed_seq = getattr(image_outputs, "pooler_output", None)
    if not isinstance(image_embed_seq, (list, tuple)) or len(image_embed_seq) == 0:
        raise RuntimeError("无法从模型输出中拿到 pooler_output，embedding 提取失败。")
    return torch.cat(image_embed_seq, dim=0)


def main():
    args = parse_args()
    apply_dataset_cli_defaults(args)
    video_dir = os.path.expanduser(args.video_dir)
    default_preprocessed_dir = (
        Path("/userhome/cs3/duanty/dataset_preposcess")
        / args.dataset
        / f"clip_{args.preprocessed_clip_fps:g}"
    )
    preprocessed_clip_dir = (
        Path(args.preprocessed_clip_dir).expanduser()
        if args.preprocessed_clip_dir.strip()
        else default_preprocessed_dir
    )
    if args.use_preprocessed_clip_frames:
        print(
            "[extract_emb] 使用预处理 clip 帧: "
            f"dir={preprocessed_clip_dir.resolve()}, fps={args.preprocessed_clip_fps:g}",
            flush=True,
        )

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif args.use_preprocessed_clip_frames:
        output_dir = (
            Path("/userhome/cs3/duanty/dataset_preposcess")
            / args.dataset
            / f"embeddings_{args.preprocessed_clip_fps:g}fps"
        ).resolve()
    else:
        output_dir = (Path(video_dir).expanduser() / "video_embeddings_1fps").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    extraction_items: list[tuple[str, VQASample | None]] = []
    seen_videos: set[str] = set()
    for sample in samples:
        path = os.path.expanduser(sample.video_path)
        if path in seen_videos:
            continue
        seen_videos.add(path)
        extraction_items.append((path, sample))
    if args.max_videos is not None:
        extraction_items = extraction_items[: args.max_videos]
    sampling_label = (
        f"preprocessed_clip_{args.preprocessed_clip_fps:g}fps"
        if args.use_preprocessed_clip_frames
        else "1fps"
    )

    apply_pixel_limits = dataset_uses_vl_pixel_limits(
        args.dataset,
        args.dataset_split,
        args.dataset_name,
    )
    model, processor = load_model_and_processor(
        args.model_path,
        apply_pixel_limits=apply_pixel_limits,
    )

    index_items: list[dict[str, Any]] = []
    success = 0
    failed = 0
    pbar = tqdm(extraction_items, desc="提取 embedding")
    for video_path, sample in pbar:
        if not args.use_preprocessed_clip_frames and not os.path.isfile(video_path):
            failed += 1
            continue
        try:
            if args.use_preprocessed_clip_frames:
                if sample is None:
                    failed += 1
                    continue
                frame_indices, frames, fps, total_frames = _load_preprocessed_candidate_frames(
                    str(preprocessed_clip_dir),
                    sample.resolve_preprocess_key(),
                )
            else:
                if not os.path.isfile(video_path):
                    failed += 1
                    continue
                frames, frame_indices, fps, total_frames = sample_one_frame_per_second(video_path)
            if not frames:
                failed += 1
                continue
            embeds = extract_visual_embeddings(model, processor, frames)
            embeds = _to_dtype(embeds.detach().cpu(), args.save_dtype)

            save_name = f"{_safe_name(video_path)}.pt"
            save_path = output_dir / save_name
            duplicate_idx = 1
            while save_path.exists():
                save_name = f"{_safe_name(video_path)}_{duplicate_idx}.pt"
                save_path = output_dir / save_name
                duplicate_idx += 1

            torch.save(
                {
                    "video_path": video_path,
                    "fps": fps,
                    "total_frames": total_frames,
                    "sampled_frame_indices": frame_indices,
                    "num_sampled_frames": len(frame_indices),
                    "embedding_shape": list(embeds.shape),
                    "embedding": embeds,
                    "model_path": args.model_path,
                    "sampling": sampling_label,
                    "preprocess_key": sample.resolve_preprocess_key() if sample is not None else None,
                    "preprocessed_clip_dir": str(preprocessed_clip_dir.resolve())
                    if args.use_preprocessed_clip_frames
                    else None,
                },
                save_path,
            )
            index_items.append(
                {
                    "video_path": video_path,
                    "sample_id": sample.sample_id if sample is not None else None,
                    "preprocess_key": sample.resolve_preprocess_key() if sample is not None else None,
                    "embedding_file": save_name,
                    "num_sampled_frames": len(frame_indices),
                    "fps": fps,
                    "total_frames": total_frames,
                    "embedding_shape": list(embeds.shape),
                }
            )
            success += 1
            pbar.set_postfix(success=success, failed=failed)
        except Exception:
            failed += 1
            pbar.set_postfix(success=success, failed=failed)

    summary = {
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "dataset_name": args.dataset_name,
        "video_dir": video_dir,
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "sampling": sampling_label,
        "use_preprocessed_clip_frames": bool(args.use_preprocessed_clip_frames),
        "preprocessed_clip_dir": str(preprocessed_clip_dir.resolve())
        if args.use_preprocessed_clip_frames
        else None,
        "save_dtype": args.save_dtype,
        "num_videos_total": len(extraction_items),
        "num_videos_success": success,
        "num_videos_failed": failed,
        "items": index_items,
    }
    with open(output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"完成：总视频={len(extraction_items)}, 成功={success}, 失败={failed}, "
        f"输出目录={output_dir}"
    )


if __name__ == "__main__":
    main()
