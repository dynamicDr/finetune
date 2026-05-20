from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from data_loaders import get_data_loader, list_supported_datasets
from data_loaders.base import VQASample
from data_loaders.ours import iterative_inference_with_cache, rank_frames_by_clip
from frame_samplers import sample_video_frames
from model_response_mode import load_model_response_mode_config, parse_response_by_mode, resolve_model_mode
from vl_common import generate_response_with_split_embedding, load_model_and_processor

MODE_MAX_NEW_TOKENS = {
    "thinking": 4086,
    "instruct": 128,
}
PREPROCESSED_CLIP_COMPATIBLE_METHODS = {
    "ours",
}


def build_user_text(question: str, options: list[str] | None) -> str:
    if options:
        return (
            f"{question}\n\nOptions:\n"
            + "\n".join(options)
            + "\n\nDirectly answer with the option letter only. Do not explain."
        )
    return f"{question}\n\nPlease provide the numerical answer directly."


def calculate_mra(pred: float, gt: float) -> float:
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, 1 - abs(pred - gt) / abs(gt))


def _compute_accuracy_from_results(results: dict, task_filter: str) -> tuple[float, float]:
    avg_accuracy = 0.0
    if task_filter in {"mcq", "short", "medium", "long"} and results["total"] > 0:
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


def _compute_avg_embedding_build_time(results: dict) -> float:
    times = results.get("embedding_build_times", [])
    if not times:
        return 0.0
    return sum(times) / len(times)


def _compute_avg_selected_frame_count(results: dict) -> float:
    counts = results.get("selected_frame_counts", [])
    if not counts:
        return 0.0
    return float(sum(counts) / len(counts))


def _compute_score_counts_for_csv(results: dict, task_filter: str) -> tuple[int, float]:
    if task_filter in {"mcq", "short", "medium", "long"}:
        return int(results["total"]), float(results["correct"])
    if task_filter == "numeric":
        return int(results["mra_count"]), float(results["mra_sum"])
    # all: 汇总离散正确数与数值题MRA分数
    return int(results["total"] + results["mra_count"]), float(results["correct"] + results["mra_sum"])


def log_to_csv(
    log_file: str,
    dataset: str,
    seed: int,
    task_filter: str,
    num_samples: int,
    evaluated_samples: int,
    correct_count: float,
    accuracy_percent: float,
    num_frames: int,
    avg_accuracy: float,
    avg_inference_time: float,
    frame_sampling_method: str,
    avg_frame_sampling_time: float,
    avg_embedding_build_time: float,
    avg_selected_frame_count: float,
    avg_total_time_hours: float,
    over_max_tokens_count: int,
    model_name: str,
    lora_path: str,
    train_ratio: float,
    use_train_split: bool,
    ours_clip_model_id: str,
    ours_clip_batch_size: int,
    use_preprocessed_clip_frames: bool,
    preprocessed_clip_fps: float,
    enable_early_stop: bool,
    early_stop_window: int,
    early_stop_conf_threshold: float,
    frame_increment: int,
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
        evaluated_samples,
        f"{correct_count:.6f}",
        f"{accuracy_percent:.2f}",
        num_frames,
        f"{avg_accuracy:.2f}",
        f"{avg_inference_time:.3f}",
        frame_sampling_method,
        f"{avg_frame_sampling_time:.6f}",
        f"{avg_embedding_build_time:.6f}",
        f"{avg_selected_frame_count:.6f}",
        f"{avg_total_time_hours:.6f}",
        over_max_tokens_count,
        model_name,
        lora_path,
        train_ratio,
        split_name,
        ours_clip_model_id,
        ours_clip_batch_size,
        use_preprocessed_clip_frames,
        preprocessed_clip_fps,
        enable_early_stop,
        early_stop_window,
        early_stop_conf_threshold,
        frame_increment,
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
                    "evaluated_samples",
                    "correct_count",
                    "accuracy_percent",
                    "num_frames",
                    "avg_accuracy",
                    "avg_inference_time",
                    "frame_sampling_method",
                    "avg_frame_sampling_time",
                    "avg_embedding_build_time",
                    "avg_selected_frame_count",
                    "avg_total_time_hours",
                    "over_max_tokens_count",
                    "model_name",
                    "lora_path",
                    "train_ratio",
                    "eval_split",
                    "ours_clip_model_id",
                    "ours_clip_batch_size",
                    "use_preprocessed_clip_frames",
                    "preprocessed_clip_fps",
                    "enable_early_stop",
                    "early_stop_window",
                    "early_stop_conf_threshold",
                    "frame_increment",
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
    max_new_tokens: int = 2048,
    ours_clip_model_id: str = "openai/clip-vit-base-patch32",
    ours_clip_device: str | None = None,
    ours_clip_batch_size: int = 16,
    model_mode: str = "thinking",
    require_think_end_for_scoring: bool = True,
    use_preprocessed_clip_frames: bool = False,
    preprocessed_clip_dir: str | None = None,
    enable_early_stop: bool = True,
    early_stop_window: int = 3,
    early_stop_conf_threshold: float = 0.9,
    frame_increment: int = 1,
) -> dict[str, Any]:
    results = {
        "correct": 0,
        "total": 0,
        "mra_sum": 0.0,
        "mra_count": 0,
        "inference_times": [],
        "frame_sampling_times": [],
        "embedding_build_times": [],
        "selected_frame_counts": [],
        "over_max_tokens_count": 0,
        "missing_think_end_count": 0,
    }

    pbar = tqdm(samples, desc="评估进度(embedding 分开输入)")
    for i, sample in enumerate(pbar):
        if task_filter != "all" and sample.task_type != task_filter:
            continue

        sample_total_start = time.perf_counter()
        step_times: dict[str, float] = {
            "prompt_build": 0.0,
            "frame_sampling": 0.0,
            "model_call": 0.0,
            "answer_extract": 0.0,
            "score_update": 0.0,
            "progress_update": 0.0,
        }

        print(f"问题: {sample.question}", flush=True)
        print(f"选项: {sample.options if sample.options is not None else '无'}", flush=True)
        print(f"正确答案: {sample.answer}", flush=True)

        t_step = time.perf_counter()
        prompt = build_user_text(sample.question, sample.options)
        step_times["prompt_build"] += time.perf_counter() - t_step

        response = ""
        pred_answer = None
        inference_time = 0.0
        embedding_build_time = 0.0
        generated_token_count = 0
        hit_max_tokens = False

        if frame_sampling_method == "ours":
            t_step = time.perf_counter()
            ranked_frames = rank_frames_by_clip(
                video_path=sample.video_path,
                sample_id=sample.sample_id,
                question=sample.question,
                options=sample.options,
                answer=str(sample.answer),
                clip_model_id=ours_clip_model_id,
                clip_device=ours_clip_device,
                clip_batch_size=ours_clip_batch_size,
                max_frames=num_frames,
                use_preprocessed_clip_frames=use_preprocessed_clip_frames,
                preprocessed_clip_dir=preprocessed_clip_dir,
            )
            frame_sampling_time = time.perf_counter() - t_step
            results["frame_sampling_times"].append(frame_sampling_time)
            step_times["frame_sampling"] += frame_sampling_time

            t_step = time.perf_counter()
            iterative_out = iterative_inference_with_cache(
                model=model,
                processor=processor,
                prompt=prompt,
                ranked_frames=ranked_frames,
                extract_answer_fn=lambda x, has_options: parse_response_by_mode(
                    response=x,
                    has_options=has_options,
                    model_mode=model_mode,
                )[2],
                has_options=bool(sample.options),
                max_new_tokens=max_new_tokens,
                model_mode=model_mode,
                enable_early_stop=enable_early_stop,
                early_stop_window=early_stop_window,
                early_stop_conf_threshold=early_stop_conf_threshold,
                frame_increment=frame_increment,
            )
            step_times["model_call"] += time.perf_counter() - t_step

            response = iterative_out["response"]
            pred_answer = iterative_out["pred_answer"]
            inference_time = iterative_out["inference_time"]
            embedding_build_time = iterative_out["embedding_build_time"]
            generated_token_count = int(iterative_out["generated_token_count"])
            hit_max_tokens = bool(iterative_out["hit_max_tokens"])
            round_details = iterative_out.get("round_details", [])
            if round_details:
                results["selected_frame_counts"].append(int(round_details[-1].get("frame_count", 0)))
            else:
                results["selected_frame_counts"].append(int(iterative_out.get("rounds_used", 0)))
        else:
            random_seed = (seed + i) if frame_sampling_method == "random" else None
            t_step = time.perf_counter()
            frames = sample_video_frames(
                video_path=sample.video_path,
                num_frames=num_frames,
                method=frame_sampling_method,
                random_seed=random_seed,
                question=sample.question,
                options=sample.options,
                answer=str(sample.answer),
                focus_blip_model_name=focus_blip_model_name,
                focus_blip_device=focus_blip_device,
                focus_blip_batch_size=focus_blip_batch_size,
            )
            frame_sampling_time = time.perf_counter() - t_step
            results["frame_sampling_times"].append(frame_sampling_time)
            step_times["frame_sampling"] += frame_sampling_time
            if not frames:
                warnings.warn(
                    f"样本无可用帧，已跳过: sample_id={sample.sample_id}, video_path={sample.video_path}",
                    RuntimeWarning,
                    stacklevel=1,
                )
                continue
            results["selected_frame_counts"].append(len(frames))
            try:
                t_step = time.perf_counter()
                response, inference_time, embedding_build_time, generated_token_count, hit_max_tokens = (
                    generate_response_with_split_embedding(
                        model=model,
                        processor=processor,
                        frames=frames,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                    )
                )
                step_times["model_call"] += time.perf_counter() - t_step
            except RuntimeError as e:
                warnings.warn(
                    f"embedding 分开输入失败，样本已跳过: sample_id={sample.sample_id}, error={e}",
                    RuntimeWarning,
                    stacklevel=1,
                )
                continue

        t_step = time.perf_counter()
        cot_text, ans_text, parsed_pred_answer = parse_response_by_mode(
            response=response,
            has_options=bool(sample.options),
            model_mode=model_mode,
        )
        pred_answer = parsed_pred_answer
        step_times["answer_extract"] += time.perf_counter() - t_step

        t_step = time.perf_counter()
        if hit_max_tokens:
            results["over_max_tokens_count"] += 1
        has_think_end = "</think>" in response
        if not has_think_end:
            results["missing_think_end_count"] += 1
        answer_usable = has_think_end or (not require_think_end_for_scoring)
        if not answer_usable:
            pred_answer = None

        results["inference_times"].append(inference_time)
        results["embedding_build_times"].append(embedding_build_time)

        if sample.options is not None:
            is_correct = answer_usable and (
                str(sample.answer).strip().upper() == str(pred_answer).strip().upper()
            )
            results["total"] += 1
            if is_correct:
                results["correct"] += 1
        else:
            if answer_usable:
                try:
                    pred_num = float(pred_answer) if pred_answer else 0.0
                    gt_num = float(sample.answer)
                    results["mra_sum"] += calculate_mra(pred_num, gt_num)
                    results["mra_count"] += 1
                except (ValueError, TypeError):
                    pass
            else:
                results["mra_count"] += 1
        step_times["score_update"] += time.perf_counter() - t_step

        t_step = time.perf_counter()
        partial_acc, partial_time = _compute_accuracy_from_results(results, task_filter)
        avg_embed_time = _compute_avg_embedding_build_time(results)
        pbar.set_postfix(
            Acc=f"{partial_acc:.2f}%",
            AvgTime=f"{partial_time:.2f}s",
            AvgEmbed=f"{avg_embed_time:.2f}s",
            n=i + 1,
        )
        step_times["progress_update"] += time.perf_counter() - t_step

        _ = sample_total_start
        _ = cot_text
        _ = ans_text

    return results


def parse_args():
    p = argparse.ArgumentParser(description="通用 VQA 评估脚本（embedding 分开输入）")
    p.add_argument("--dataset", type=str, default="vsibench", choices=list_supported_datasets())
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--dataset_name", type=str, default="nyu-visionx/VSI-Bench")
    p.add_argument("--dataset_config", type=str, default="full")
    p.add_argument("--no_dataset_config", action="store_true")

    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Thinking")
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
        choices=[
            "uniform",
            "random",
            "focus",
            "sevila",
            "videoagent",
            "clip",
            "siglip2",
            "aks",
            "aks-blip",
            "aks-clip",
            "ours",
        ],
    )
    p.add_argument("--focus_blip_model_name", type=str, default="Salesforce/blip-itm-base-coco")
    p.add_argument("--focus_blip_device", type=str, default=None)
    p.add_argument("--focus_blip_batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--task_filter",
        type=str,
        default="all",
        choices=["all", "mcq", "numeric", "short", "medium", "long"],
    )
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--ours_clip_model_id", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--ours_clip_device", type=str, default=None)
    p.add_argument("--ours_clip_batch_size", type=int, default=16)
    p.add_argument(
        "--enable_early_stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 ours 迭代早停机制（默认启用）。",
    )
    p.add_argument(
        "--early_stop_window",
        type=int,
        default=3,
        help="连续满足条件多少轮后触发早停。",
    )
    p.add_argument(
        "--early_stop_conf_threshold",
        type=float,
        default=0.9,
        help="早停置信度阈值：所选答案的 option_prob 必须大于该值。",
    )
    p.add_argument(
        "--frame_increment",
        type=int,
        default=1,
        help="ours 迭代时每轮新增帧数（默认1，即 top-1, top-2, ...）。",
    )
    p.add_argument(
        "--model_mode_config",
        type=str,
        default="config/model_response_modes.json",
        help="模型响应模式配置文件(JSON): 通过规则自动判断 thinking/instruct。",
    )
    p.add_argument("--log_file", type=str, default="vqa_embedding_evaluation_log.csv")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--use_train_split", action="store_true")
    p.add_argument(
        "--use_preprocessed_clip_frames",
        action="store_true",
        help="启用后，ours 在 CLIP 排序阶段优先读取离线预处理帧。",
    )
    p.add_argument(
        "--preprocessed_clip_fps",
        type=float,
        default=1.0,
        help="预处理目录命名中的 fps（默认对应 /userhome/cs3/duanty/dataset_preposcess/{dataset}/clip_1）。",
    )
    p.add_argument(
        "--preprocessed_clip_dir",
        type=str,
        default="",
        help="预处理帧根目录；为空时自动使用 /userhome/cs3/duanty/dataset_preposcess/{dataset}/clip_{fps}。",
    )
    return p.parse_args()


def main():
    experiment_start_time = time.perf_counter()
    args = parse_args()
    video_dir = os.path.expanduser(args.video_dir)
    eval_csv_dir = Path(__file__).resolve().parent / "eval_csv"
    eval_csv_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(eval_csv_dir / Path(args.log_file).name)
    default_preprocessed_dir = (
        Path("/userhome/cs3/duanty/dataset_preposcess")
        / args.dataset
        / f"clip_{args.preprocessed_clip_fps:g}"
    )
    preprocessed_clip_dir = (
        os.path.expanduser(args.preprocessed_clip_dir)
        if args.preprocessed_clip_dir.strip()
        else str(default_preprocessed_dir)
    )
    if args.use_preprocessed_clip_frames:
        if args.frame_sampling_method not in PREPROCESSED_CLIP_COMPATIBLE_METHODS:
            raise ValueError(
                "use_preprocessed_clip_frames 仅支持 ours，"
                f"当前 frame_sampling_method={args.frame_sampling_method}"
            )
        _ = preprocessed_clip_dir

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
        model_name = os.path.expanduser(args.base_model) if args.base_model else "Qwen/Qwen3-VL-4B-Thinking"
        lora_path = resolved_model_path
    else:
        model_name = os.path.basename(resolved_model_path.rstrip("/"))
        lora_path = ""

    mode_config = load_model_response_mode_config(args.model_mode_config)
    model_identifier_candidates = [resolved_model_path]
    if model_name:
        model_identifier_candidates.append(model_name)
    if args.base_model:
        model_identifier_candidates.append(os.path.expanduser(args.base_model))

    last_error: Exception | None = None
    model_mode = ""
    require_think_end_for_scoring = True
    matched_rule = ""
    for candidate in model_identifier_candidates:
        try:
            model_mode, require_think_end_for_scoring, matched_rule = resolve_model_mode(
                model_identifier=candidate,
                config=mode_config,
            )
            break
        except (KeyError, ValueError) as e:
            last_error = e
    if not model_mode:
        raise RuntimeError(
            "模型模式识别失败，请在 model_response_modes.json 中添加精确键。"
            f" tried={model_identifier_candidates}"
        ) from last_error
    _ = matched_rule
    effective_max_new_tokens = MODE_MAX_NEW_TOKENS[model_mode]

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
        max_new_tokens=effective_max_new_tokens,
        ours_clip_model_id=args.ours_clip_model_id,
        ours_clip_device=args.ours_clip_device,
        ours_clip_batch_size=args.ours_clip_batch_size,
        model_mode=model_mode,
        require_think_end_for_scoring=require_think_end_for_scoring,
        use_preprocessed_clip_frames=args.use_preprocessed_clip_frames,
        preprocessed_clip_dir=preprocessed_clip_dir,
        enable_early_stop=args.enable_early_stop,
        early_stop_window=args.early_stop_window,
        early_stop_conf_threshold=args.early_stop_conf_threshold,
        frame_increment=args.frame_increment,
    )
    avg_accuracy, avg_inference_time = _compute_accuracy_from_results(results, args.task_filter)
    evaluated_samples, correct_count = _compute_score_counts_for_csv(results, args.task_filter)
    avg_frame_sampling_time = _compute_avg_frame_sampling_time(results)
    avg_embedding_build_time = _compute_avg_embedding_build_time(results)
    avg_selected_frame_count = _compute_avg_selected_frame_count(results)
    # 保留历史字段名 avg_total_time_hours，但语义改为整次实验总耗时（wall-clock）
    avg_total_time_hours = (time.perf_counter() - experiment_start_time) / 3600.0
    log_to_csv(
        log_file=log_file,
        dataset=args.dataset,
        seed=args.seed,
        task_filter=args.task_filter,
        num_samples=len(samples),
        evaluated_samples=evaluated_samples,
        correct_count=correct_count,
        accuracy_percent=avg_accuracy,
        num_frames=args.num_frames,
        avg_accuracy=avg_accuracy,
        avg_inference_time=avg_inference_time,
        frame_sampling_method=args.frame_sampling_method,
        avg_frame_sampling_time=avg_frame_sampling_time,
        avg_embedding_build_time=avg_embedding_build_time,
        avg_selected_frame_count=avg_selected_frame_count,
        avg_total_time_hours=avg_total_time_hours,
        over_max_tokens_count=results["over_max_tokens_count"],
        model_name=model_name,
        lora_path=lora_path,
        train_ratio=args.train_ratio,
        use_train_split=args.use_train_split,
        ours_clip_model_id=args.ours_clip_model_id,
        ours_clip_batch_size=args.ours_clip_batch_size,
        use_preprocessed_clip_frames=args.use_preprocessed_clip_frames,
        preprocessed_clip_fps=args.preprocessed_clip_fps,
        enable_early_stop=args.enable_early_stop,
        early_stop_window=args.early_stop_window,
        early_stop_conf_threshold=args.early_stop_conf_threshold,
        frame_increment=args.frame_increment,
    )


if __name__ == "__main__":
    main()
