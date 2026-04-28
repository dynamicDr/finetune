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

from data_loaders import get_data_loader
from vl_common import load_model_and_processor


def parse_args():
    p = argparse.ArgumentParser(description="按每秒1帧粗采样提取视频 embedding 并保存到本地")
    p.add_argument("--dataset", type=str, default="videomme")
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
    return p.parse_args()


def _safe_name(video_path: str) -> str:
    stem = Path(video_path).stem
    stem = re.sub(r"[^0-9A-Za-z._-]", "_", stem)
    return stem or "video"


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
    video_dir = os.path.expanduser(args.video_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
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

    unique_video_paths: list[str] = []
    seen = set()
    for s in samples:
        path = os.path.expanduser(s.video_path)
        if path in seen:
            continue
        seen.add(path)
        unique_video_paths.append(path)
    if args.max_videos is not None:
        unique_video_paths = unique_video_paths[: args.max_videos]

    model, processor = load_model_and_processor(args.model_path)

    index_items: list[dict[str, Any]] = []
    success = 0
    failed = 0
    pbar = tqdm(unique_video_paths, desc="提取 embedding")
    for video_path in pbar:
        if not os.path.isfile(video_path):
            failed += 1
            continue
        try:
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
                    "sampling": "1fps",
                },
                save_path,
            )
            index_items.append(
                {
                    "video_path": video_path,
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
        "sampling": "1fps",
        "save_dtype": args.save_dtype,
        "num_videos_total": len(unique_video_paths),
        "num_videos_success": success,
        "num_videos_failed": failed,
        "items": index_items,
    }
    with open(output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"完成：总视频={len(unique_video_paths)}, 成功={success}, 失败={failed}, "
        f"输出目录={output_dir}"
    )


if __name__ == "__main__":
    main()
