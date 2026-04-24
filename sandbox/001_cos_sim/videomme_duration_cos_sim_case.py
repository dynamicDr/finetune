from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ====== 实验参数（按需直接修改这里）======
SEED = 42
VIDEO_DIR = "~/dataset/Video-MME"
DATASET_NAME = "lmms-lab/Video-MME"
META_PARQUET = ""  # 可选；空字符串表示自动探测本地 parquet
SAMPLE_FPS = 1.0
CLIP_MODEL = "openai/clip-vit-base-patch32"
OUTPUT_DIR = "sandbox/001_cos_sim/output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16


def load_videomme_rows(video_dir: Path, dataset_name: str, meta_parquet: str) -> list[dict]:
    candidate_parquets = []
    if meta_parquet:
        candidate_parquets.append(Path(os.path.expanduser(meta_parquet)))
    base = video_dir
    parent = base.parent
    candidate_parquets.extend(
        [
            base / "videomme" / "test-00000-of-00001.parquet",
            base / "test-00000-of-00001.parquet",
            parent / "videomme" / "test-00000-of-00001.parquet",
            parent / "test-00000-of-00001.parquet",
        ]
    )

    parquet_path = next((p for p in candidate_parquets if p.is_file()), None)
    if parquet_path is not None:
        import pandas as pd

        print(f"[INFO] 使用本地 parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        return df.to_dict(orient="records")

    print(f"[INFO] 未找到本地 parquet，回退到 HuggingFace: {dataset_name}")
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="test")
    return [ds[i] for i in range(len(ds))]


def build_video_index(video_dir: Path) -> dict[str, Path]:
    candidates = [
        video_dir / "videos" / "data",
        video_dir / "videos",
        video_dir / "data",
        video_dir,
        video_dir.parent / "videos" / "data",
        video_dir.parent / "videos",
        video_dir.parent,
    ]
    roots = [p for p in dict.fromkeys(candidates) if p.is_dir()]

    video_index: dict[str, Path] = {}
    for root in roots:
        for cur_root, _, files in os.walk(root):
            for fn in sorted(files):
                low = fn.lower()
                if not low.endswith((".mp4", ".webm", ".mkv", ".avi", ".mov")):
                    continue
                stem = Path(fn).stem
                if stem not in video_index:
                    video_index[stem] = Path(cur_root) / fn
    return video_index


def pick_one_video_per_duration(rows: list[dict], video_index: dict[str, Path], seed: int) -> dict[str, Path]:
    durations = ["short", "medium", "long"]
    candidates: dict[str, list[str]] = {d: [] for d in durations}
    seen: dict[str, set[str]] = {d: set() for d in durations}

    for row in rows:
        duration = str(row.get("duration", "")).strip().lower()
        if duration not in durations:
            continue
        vid = str(row.get("videoID", "")).strip() or str(row.get("video_id", "")).strip()
        if not vid or vid in seen[duration]:
            continue
        seen[duration].add(vid)
        candidates[duration].append(vid)

    rng = random.Random(seed)
    chosen: dict[str, Path] = {}
    for d in durations:
        available = sorted([vid for vid in candidates[d] if vid in video_index])
        if not available:
            continue
        picked_vid = rng.choice(available)
        chosen[d] = video_index[picked_vid]

    missing = [d for d in durations if d not in chosen]
    if missing:
        raise RuntimeError(f"这些时长没有找到可用视频文件: {missing}")
    return chosen


def sample_frames_1fps(video_path: Path, sample_fps: float) -> tuple[list[Image.Image], list[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps is None or orig_fps <= 0:
        orig_fps = 30.0
    interval = max(int(round(orig_fps / sample_fps)), 1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: list[Image.Image] = []
    times: list[float] = []
    i = 0

    bar = tqdm(total=total if total > 0 else None, desc=f"读取帧: {video_path.stem}", unit="frame")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i % interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            times.append(i / orig_fps)
        i += 1
        bar.update(1)
    bar.close()
    cap.release()

    if len(frames) < 2:
        raise RuntimeError(f"采样后帧数不足2帧，无法计算相邻相似度: {video_path}")
    return frames, times


def clip_adjacent_cos_sim(
    frames: list[Image.Image],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    batch_size: int,
) -> np.ndarray:
    all_features = []
    bar = tqdm(total=len(frames), desc="CLIP编码", unit="frame")
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            # 兼容不同实现：有些环境返回张量，有些返回带 pooler_output 的模型输出对象
            if not isinstance(feats, torch.Tensor):
                if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                    feats = feats.image_embeds
                elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                else:
                    raise TypeError(f"无法从 CLIP 输出中提取图像特征，返回类型: {type(feats)}")
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_features.append(feats.cpu())
        bar.update(len(batch))
    bar.close()

    features = torch.cat(all_features, dim=0)
    sims = (features[:-1] * features[1:]).sum(dim=-1).numpy()
    return sims


def save_curve_csv(csv_path: Path, x: list[float], y: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "cos_sim_adjacent"])
        for t, s in zip(x, y):
            w.writerow([f"{t:.6f}", f"{float(s):.6f}"])


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    video_dir = Path(os.path.expanduser(VIDEO_DIR))
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_videomme_rows(video_dir=video_dir, dataset_name=DATASET_NAME, meta_parquet=META_PARQUET)
    video_index = build_video_index(video_dir)
    picked = pick_one_video_per_duration(rows, video_index, seed=SEED)

    print("[INFO] 选中的3个视频：")
    for d in ["short", "medium", "long"]:
        print(f"  - {d}: {picked[d]}")

    device = DEVICE
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    results: dict[str, tuple[list[float], np.ndarray]] = {}
    outer = tqdm(["short", "medium", "long"], desc="总进度(3个视频)", unit="video")
    for duration in outer:
        video_path = picked[duration]
        outer.set_postfix_str(duration)
        print(f"[INFO] 当前处理文件: {video_path.name} ({duration})")

        frames, timestamps = sample_frames_1fps(video_path=video_path, sample_fps=SAMPLE_FPS)
        sims = clip_adjacent_cos_sim(
            frames=frames,
            processor=processor,
            model=model,
            device=device,
            batch_size=BATCH_SIZE,
        )

        x = timestamps[1:]
        results[duration] = (x, sims)

        save_curve_csv(output_dir / f"{duration}_adjacent_cos_sim.csv", x, sims)

        plt.figure(figsize=(10, 4))
        plt.plot(x, sims, marker="o", markersize=2, linewidth=1)
        plt.title(f"Video-MME {duration} - Adjacent Frame Cosine Similarity (1 fps)")
        plt.xlabel("Time (s)")
        plt.ylabel("Cosine Similarity")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{duration}_adjacent_cos_sim.png", dpi=150)
        plt.close()

    plt.figure(figsize=(11, 5))
    for duration in ["short", "medium", "long"]:
        x, sims = results[duration]
        plt.plot(x, sims, linewidth=1.2, label=duration)
    plt.title("Video-MME (short/medium/long) Adjacent Frame Cosine Similarity (1 fps)")
    plt.xlabel("Time (s)")
    plt.ylabel("Cosine Similarity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "all_durations_adjacent_cos_sim.png", dpi=160)
    plt.close()

    print(f"[DONE] 实验完成，结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

