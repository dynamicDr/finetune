"""
在 EgoSchema 上评估模型（默认多选题准确率）。
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from hf_dataset_loader import load_dataset
from vl_common import build_mcq_prompt, extract_video_frames, generate_response, load_model_and_processor, split_indices


def _normalize_options(sample: dict[str, Any]) -> list[str] | None:
    options = sample.get("options")
    if isinstance(options, list) and options:
        return [str(x).strip() for x in options]
    for key in ("option", "choices", "candidates"):
        cand = sample.get(key)
        if isinstance(cand, list) and cand:
            return [str(x).strip() for x in cand]

    collected: list[str] = []
    for i in range(8):
        for k in (f"option_{i}", f"option{i}", f"a{i}", f"choice_{i}"):
            if k in sample and str(sample[k]).strip():
                collected.append(str(sample[k]).strip())
                break
    return collected if collected else None


def _answer_to_letter(answer: Any, num_options: int) -> str | None:
    if answer is None:
        return None
    text = str(answer).strip().upper()
    if not text:
        return None
    if len(text) == 1 and "A" <= text <= "Z":
        return text
    m = re.search(r"-?\d+", text)
    if m:
        idx = int(m.group(0))
        if 0 <= idx < num_options:
            return chr(ord("A") + idx)
        if 1 <= idx <= num_options:
            return chr(ord("A") + idx - 1)
    return None


def _extract_answer_letter(response: str) -> str:
    response = response.strip()
    if "</think>" in response:
        response = response.split("</think>", 1)[-1].strip()
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        response = answer_match.group(1).strip()
    m = re.search(r"\b([A-Z])\b", response.upper())
    return m.group(1) if m else ""


def _resolve_video_path(video_dir: str, sample: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    for key in ("video_path", "video", "video_id", "video_uid", "video_idx", "sample_id", "uid", "q_uid"):
        val = sample.get(key)
        if val is not None and str(val).strip():
            candidates.append(str(val).strip())

    for c in candidates:
        if os.path.isfile(c):
            return c
        p = os.path.join(video_dir, c)
        if os.path.isfile(p):
            return p
        for ext in (".mp4", ".MP4", ".webm", ".mkv", ".avi"):
            p2 = os.path.join(video_dir, c + ext)
            if os.path.isfile(p2):
                return p2
    return None


def _log_to_csv(
    log_file: str,
    seed: int,
    num_samples: int,
    num_frames: int,
    accuracy: float,
    avg_inference_time: float,
    model_name: str,
    lora_path: str,
    train_ratio: float,
    use_train_split: bool,
):
    Path(log_file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.exists(log_file)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    split_name = "train" if use_train_split else "test"
    row = [
        timestamp,
        seed,
        num_samples,
        num_frames,
        f"{accuracy:.2f}",
        f"{avg_inference_time:.3f}",
        model_name,
        lora_path,
        train_ratio,
        split_name,
    ]
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(
                [
                    "timestamp",
                    "seed",
                    "num_samples",
                    "num_frames",
                    "accuracy",
                    "avg_inference_time",
                    "model_name",
                    "lora_path",
                    "train_ratio",
                    "eval_split",
                ]
            )
        w.writerow(row)
    print(f"结果已写入: {log_file}")


def parse_args():
    p = argparse.ArgumentParser(description="EgoSchema 评估")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--video_dir", type=str, default="~/dataset/egoschema/videos")
    p.add_argument("--dataset_name", type=str, default="lmms-lab/EgoSchema")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--num_samples", type=str, default="all")
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--use_train_split", action="store_true")
    p.add_argument("--log_file", type=str, default="egoschema_eval_log.csv")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    video_dir = os.path.expanduser(args.video_dir)

    eval_csv_dir = Path(__file__).resolve().parents[1] / "eval_csv"
    eval_csv_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(eval_csv_dir / Path(args.log_file).name)

    dataset = load_dataset(args.dataset_name, split=args.split)
    indices = list(range(len(dataset)))
    indices = split_indices(indices, args.seed, args.train_ratio, args.use_train_split)

    if args.num_samples.lower() != "all":
        n = int(args.num_samples)
        random.seed(args.seed + 1000)
        indices = random.sample(indices, min(n, len(indices)))

    resolved_model_path = os.path.expanduser(args.model_path)
    if args.use_lora:
        model_name = os.path.expanduser(args.base_model) if args.base_model else "Qwen/Qwen2.5-VL-3B-Instruct"
        lora_path = resolved_model_path
    else:
        model_name = os.path.basename(resolved_model_path.rstrip("/"))
        lora_path = ""

    model, processor = load_model_and_processor(
        resolved_model_path,
        use_lora=args.use_lora,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
    )

    total = 0
    correct = 0
    inference_times: list[float] = []
    pbar = tqdm(indices, desc="EgoSchema 评估")

    for idx in pbar:
        sample = dataset[idx]
        question = str(sample.get("question", "")).strip()
        options = _normalize_options(sample)
        if not question or not options:
            continue
        gt = _answer_to_letter(sample.get("answer", sample.get("ground_truth", sample.get("label"))), len(options))
        if not gt:
            continue

        video_path = _resolve_video_path(video_dir, sample)
        if not video_path:
            continue
        frames = extract_video_frames(video_path, num_frames=args.num_frames)
        if not frames:
            continue

        prompt = build_mcq_prompt(question, options)
        response, infer_t = generate_response(model, processor, frames, prompt)
        pred = _extract_answer_letter(response)
        is_correct = pred == gt

        total += 1
        correct += int(is_correct)
        inference_times.append(infer_t)
        pbar.set_postfix(acc=f"{100.0 * correct / max(total, 1):.2f}%")

    accuracy = 100.0 * correct / max(total, 1)
    avg_infer = sum(inference_times) / len(inference_times) if inference_times else 0.0
    print(f"样本数: {total}, Accuracy: {accuracy:.2f}%, Avg infer: {avg_infer:.3f}s")

    _log_to_csv(
        log_file=log_file,
        seed=args.seed,
        num_samples=total,
        num_frames=args.num_frames,
        accuracy=accuracy,
        avg_inference_time=avg_infer,
        model_name=model_name,
        lora_path=lora_path,
        train_ratio=args.train_ratio,
        use_train_split=args.use_train_split,
    )


if __name__ == "__main__":
    main()
