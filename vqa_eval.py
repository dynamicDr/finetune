from __future__ import annotations

import argparse
import csv
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from data_loaders import get_data_loader, list_supported_datasets
from data_loaders.base import VQASample
from frame_samplers import sample_video_frames
from vl_common import generate_response, load_model_and_processor


def build_user_text(question: str, options: list[str] | None) -> str:
    if options:
        return (
            f"{question}\n\nOptions:\n"
            + "\n".join(options)
            + "\n\nPlease answer with the option letter directly."
        )
    return f"{question}\n\nPlease provide the numerical answer directly."


def extract_answer(response: str, has_options: bool = False):
    response = response.strip()
    answer_portion = response.split("</think>", 1)[-1].strip() if "</think>" in response else response
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", answer_portion, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_portion = answer_match.group(1).strip()
    if has_options:
        match = re.search(r"\b([A-E])\b", answer_portion.upper())
        if match:
            return match.group(1)
    # 仅提取合法数字，避免 "." 这类无效串触发 float 转换报错
    numbers = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", answer_portion)
    for n in numbers:
        try:
            return float(n)
        except ValueError:
            continue
    return response


def calculate_mra(pred: float, gt: float) -> float:
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, 1 - abs(pred - gt) / abs(gt))


def _compute_accuracy_from_results(results: dict, task_filter: str) -> tuple[float, float]:
    avg_accuracy = 0.0
    if task_filter == "mcq" and results["total"] > 0:
        avg_accuracy = results["correct"] / results["total"] * 100
    elif task_filter == "numeric" and results["mra_count"] > 0:
        avg_accuracy = results["mra_sum"] / results["mra_count"] * 100
    elif task_filter == "all":
        total_score = 0.0
        total_count = 0
        if results["total"] > 0:
            total_score += results["correct"]
            total_count += results["total"]
        if results["mra_count"] > 0:
            total_score += results["mra_sum"]
            total_count += results["mra_count"]
        if total_count > 0:
            avg_accuracy = total_score / total_count * 100
    avg_inference_time = (
        sum(results["inference_times"]) / len(results["inference_times"]) if results["inference_times"] else 0.0
    )
    return avg_accuracy, avg_inference_time


def _compute_avg_frame_sampling_time(results: dict) -> float:
    times = results.get("frame_sampling_times", [])
    if not times:
        return 0.0
    return sum(times) / len(times)


def log_to_csv(
    log_file: str,
    dataset: str,
    seed: int,
    task_filter: str,
    num_samples: int,
    num_frames: int,
    avg_accuracy: float,
    avg_inference_time: float,
    frame_sampling_method: str,
    avg_frame_sampling_time: float,
    model_name: str,
    lora_path: str,
    train_ratio: float,
    use_train_split: bool,
):
    Path(log_file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.exists(log_file)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    split_name = "train" if use_train_split else "test"
    row_data = [
        timestamp,
        dataset,
        seed,
        task_filter,
        num_samples,
        num_frames,
        f"{avg_accuracy:.2f}",
        f"{avg_inference_time:.3f}",
        frame_sampling_method,
        f"{avg_frame_sampling_time:.6f}",
        model_name,
        lora_path,
        train_ratio,
        split_name,
    ]
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "dataset",
                    "seed",
                    "task_filter",
                    "num_samples",
                    "num_frames",
                    "avg_accuracy",
                    "avg_inference_time",
                    "frame_sampling_method",
                    "avg_frame_sampling_time",
                    "model_name",
                    "lora_path",
                    "train_ratio",
                    "eval_split",
                ]
            )
        writer.writerow(row_data)


def evaluate_vqa(
    model,
    processor,
    samples: list[VQASample],
    num_frames: int,
    task_filter: str,
    frame_sampling_method: str = "uniform",
    seed: int = 42,
    focus_blip_model_name: str = "Salesforce/blip-itm-base-coco",
    focus_blip_device: str | None = None,
    focus_blip_batch_size: int = 16,
) -> dict[str, Any]:
    results = {
        "correct": 0,
        "total": 0,
        "mra_sum": 0.0,
        "mra_count": 0,
        "inference_times": [],
        "frame_sampling_times": [],
    }

    pbar = tqdm(samples, desc="评估进度")
    for i, sample in enumerate(pbar):
        # 按 task_filter 显式过滤样本，避免不匹配类型产生无效计算
        if task_filter != "all" and sample.task_type != task_filter:
            continue

        random_seed = (seed + i) if frame_sampling_method == "random" else None
        t0 = time.perf_counter()
        frames = sample_video_frames(
            video_path=sample.video_path,
            num_frames=num_frames,
            method=frame_sampling_method,
            random_seed=random_seed,
            question=sample.question,
            answer=str(sample.answer),
            focus_blip_model_name=focus_blip_model_name,
            focus_blip_device=focus_blip_device,
            focus_blip_batch_size=focus_blip_batch_size,
        )
        frame_sampling_time = time.perf_counter() - t0
        results["frame_sampling_times"].append(frame_sampling_time)
        if not frames:
            warnings.warn(
                f"样本无可用帧，已跳过: sample_id={sample.sample_id}, video_path={sample.video_path}",
                RuntimeWarning,
                stacklevel=1,
            )
            continue

        prompt = build_user_text(sample.question, sample.options)
        response, inference_time = generate_response(model, processor, frames, prompt)
        pred_answer = extract_answer(response, has_options=bool(sample.options))
        results["inference_times"].append(inference_time)

        if sample.task_type == "mcq":
            is_correct = str(sample.answer).strip().upper() == str(pred_answer).strip().upper()
            results["total"] += 1
            if is_correct:
                results["correct"] += 1
        else:
            try:
                pred_num = float(pred_answer) if pred_answer else 0.0
                gt_num = float(sample.answer)
                results["mra_sum"] += calculate_mra(pred_num, gt_num)
                results["mra_count"] += 1
            except (ValueError, TypeError):
                pass

        partial_acc, partial_time = _compute_accuracy_from_results(results, task_filter)
        pbar.set_postfix(Acc=f"{partial_acc:.2f}%", AvgTime=f"{partial_time:.2f}s", n=i + 1)

    return results


def parse_args():
    p = argparse.ArgumentParser(description="通用 VQA 评估脚本")
    p.add_argument("--dataset", type=str, default="vsibench", choices=list_supported_datasets())
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--dataset_name", type=str, default="nyu-visionx/VSI-Bench")
    p.add_argument("--dataset_config", type=str, default="full")
    p.add_argument("--no_dataset_config", action="store_true")

    p.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--model_name", type=str, default=None, help="模型名称（可选，优先用于日志）")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--merge_lora", action="store_true")

    p.add_argument("--video_dir", type=str, default="~/dataset/vsi_bench")
    p.add_argument("--num_samples", type=str, default="10")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument(
        "--frame_sampling_method",
        type=str,
        default="uniform",
        choices=["uniform", "random", "focus", "sevila", "videoagent", "clip", "siglip2", "aks"],
    )
    p.add_argument("--focus_blip_model_name", type=str, default="Salesforce/blip-itm-base-coco")
    p.add_argument("--focus_blip_device", type=str, default=None)
    p.add_argument("--focus_blip_batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task_filter", type=str, default="all", choices=["all", "mcq", "numeric"])
    p.add_argument("--log_file", type=str, default="vqa_evaluation_log.csv")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--use_train_split", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    video_dir = os.path.expanduser(args.video_dir)
    eval_csv_dir = Path(__file__).resolve().parent / "eval_csv"
    eval_csv_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(eval_csv_dir / Path(args.log_file).name)

    sample_count = None if args.num_samples.lower() == "all" else int(args.num_samples)
    loader = get_data_loader(
        args.dataset,
        video_dir=video_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        task_filter=args.task_filter,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        no_dataset_config=args.no_dataset_config,
    )
    samples = loader.get_split_samples(
        split=args.dataset_split,
        use_train_split=args.use_train_split,
        sample_count=sample_count,
    )

    resolved_model_path = os.path.expanduser(args.model_path)
    lora_path = ""
    if args.model_name:
        model_name = args.model_name
    elif args.use_lora:
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
    results = evaluate_vqa(
        model=model,
        processor=processor,
        samples=samples,
        num_frames=args.num_frames,
        task_filter=args.task_filter,
        frame_sampling_method=args.frame_sampling_method,
        seed=args.seed,
        focus_blip_model_name=args.focus_blip_model_name,
        focus_blip_device=args.focus_blip_device,
        focus_blip_batch_size=args.focus_blip_batch_size,
    )
    avg_accuracy, avg_inference_time = _compute_accuracy_from_results(results, args.task_filter)
    avg_frame_sampling_time = _compute_avg_frame_sampling_time(results)
    log_to_csv(
        log_file=log_file,
        dataset=args.dataset,
        seed=args.seed,
        task_filter=args.task_filter,
        num_samples=len(samples),
        num_frames=args.num_frames,
        avg_accuracy=avg_accuracy,
        avg_inference_time=avg_inference_time,
        frame_sampling_method=args.frame_sampling_method,
        avg_frame_sampling_time=avg_frame_sampling_time,
        model_name=model_name,
        lora_path=lora_path,
        train_ratio=args.train_ratio,
        use_train_split=args.use_train_split,
    )
    print(
        f"评估完成：样本 {len(samples)}, Accuracy {avg_accuracy:.2f}%, "
        f"AvgInfer {avg_inference_time:.3f}s, AvgFrameSampling {avg_frame_sampling_time:.6f}s"
    )


if __name__ == "__main__":
    main()

