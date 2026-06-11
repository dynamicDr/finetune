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

from data_loaders import (
    apply_dataset_cli_defaults,
    dataset_uses_vl_pixel_limits,
    get_data_loader,
    list_supported_datasets,
)
from data_loaders.base import VQASample, sample_matches_task_filter
from frame_samplers import sample_video_frames
from model_response_mode import load_model_response_mode_config, parse_response_by_mode, resolve_model_mode
from utils import (
    build_user_text,
    build_user_text_with_subtitles,
    calculate_mra,
    collect_unique_subtitles_for_sample as _collect_subtitles_for_sample,
    compute_accuracy_from_results as _compute_accuracy_from_results,
    compute_score_counts_for_csv as _compute_score_counts_for_csv,
)
from vl_common import generate_response, load_model_and_processor

MODE_MAX_NEW_TOKENS = {
    "thinking": 4086,
    "instruct": 128,
}
PREPROCESSED_CLIP_COMPATIBLE_METHODS = {
    "clip",
    "clip-new",
    "aks",
    "aks-blip",
    "aks-clip",
    "uniform",
    "random",
    "siglip2",
    "siglip2-new",
    "qframe",
    "bolt-clip",
    "bolt-siglip2",
}

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
    evaluated_samples: int,
    correct_count: float,
    accuracy_percent: float,
    num_frames: int,
    avg_accuracy: float,
    avg_inference_time: float,
    frame_sampling_method: str,
    avg_frame_sampling_time: float,
    avg_embedding_build_time: float,
    avg_total_time_hours: float,
    over_max_tokens_count: int,
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
        evaluated_samples,
        f"{correct_count:.6f}",
        f"{accuracy_percent:.2f}",
        num_frames,
        f"{avg_accuracy:.2f}",
        f"{avg_inference_time:.3f}",
        frame_sampling_method,
        f"{avg_frame_sampling_time:.6f}",
        f"{avg_embedding_build_time:.6f}",
        f"{avg_total_time_hours:.6f}",
        over_max_tokens_count,
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
                    "evaluated_samples",
                    "correct_count",
                    "accuracy_percent",
                    "num_frames",
                    "avg_accuracy",
                    "avg_inference_time",
                    "frame_sampling_method",
                    "avg_frame_sampling_time",
                    "avg_embedding_build_time",
                    "avg_total_time_hours",
                    "over_max_tokens_count",
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
    max_new_tokens: int = 2048,
    model_mode: str = "thinking",
    require_think_end_for_scoring: bool = True,
    use_preprocessed_clip_frames: bool = False,
    preprocessed_clip_dir: str | None = None,
    use_subtitles: bool = False,
    subtitles_dir: str | None = None,
) -> dict[str, Any]:
    results = {
        "correct": 0,
        "total": 0,
        "mra_sum": 0.0,
        "mra_count": 0,
        "inference_times": [],
        "frame_sampling_times": [],
        "over_max_tokens_count": 0,
        "missing_think_end_count": 0,
    }

    pbar = tqdm(samples, desc="评估进度")
    for i, sample in enumerate(pbar):
        if not sample_matches_task_filter(sample, task_filter):
            continue

        random_seed = (seed + i) if frame_sampling_method == "random" else None
        t0 = time.perf_counter()
        try:
            frames = sample_video_frames(
                video_path=sample.video_path,
                num_frames=num_frames,
                method=frame_sampling_method,
                random_seed=random_seed,
                sample_id=sample.resolve_preprocess_key(),
                question=sample.question,
                options=sample.options,
                answer=str(sample.answer),
                focus_blip_model_name=focus_blip_model_name,
                focus_blip_device=focus_blip_device,
                focus_blip_batch_size=focus_blip_batch_size,
                use_preprocessed_clip_frames=use_preprocessed_clip_frames,
                preprocessed_clip_dir=preprocessed_clip_dir,
            )
        except Exception as e:
            if use_preprocessed_clip_frames:
                raise RuntimeError(
                    "预处理 clip 帧读取失败，实验已中断: "
                    f"sample_id={sample.sample_id}, video_path={sample.video_path}, "
                    f"preprocessed_dir={preprocessed_clip_dir}"
                ) from e
            raise
        frame_sampling_time = time.perf_counter() - t0
        results["frame_sampling_times"].append(frame_sampling_time)
        if not frames:
            if use_preprocessed_clip_frames:
                raise RuntimeError(
                    "启用预处理 clip 帧后仍未拿到任何帧，实验已中断: "
                    f"sample_id={sample.sample_id}, video_path={sample.video_path}, "
                    f"preprocessed_dir={preprocessed_clip_dir}"
                )
            warnings.warn(
                f"样本无可用帧，已跳过: sample_id={sample.sample_id}, video_path={sample.video_path}",
                RuntimeWarning,
                stacklevel=1,
            )
            continue

        subtitles_for_prompt = (
            _collect_subtitles_for_sample(
                sample=sample,
                num_frames=num_frames,
                frame_sampling_method=frame_sampling_method,
                random_seed=random_seed,
                subtitles_dir=subtitles_dir,
            )
            if use_subtitles
            else []
        )
        prompt = (
            build_user_text_with_subtitles(sample.question, sample.options, subtitles_for_prompt)
            if use_subtitles
            else build_user_text(sample.question, sample.options)
        )
        sample_t0 = time.time()
        sample_t1 = sample_t0 + frame_sampling_time
        response, inference_time, generated_token_count, hit_max_tokens = generate_response(
            model,
            processor,
            frames,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        sample_t4 = time.time()
        # processor / generate / decode timing are printed in generate_response.
        # Here we provide the requested frame+end-to-end line on the same sample.
        print(
            "[perf-debug][sample-timing] "
            f"frame_load={sample_t1 - sample_t0:.2f}s, "
            f"processor+generate+decode={sample_t4 - sample_t1:.2f}s",
            flush=True,
        )
        if hit_max_tokens:
            results["over_max_tokens_count"] += 1
        has_think_end = "</think>" in response
        if not has_think_end:
            results["missing_think_end_count"] += 1
        cot_text, ans_text, pred_answer = parse_response_by_mode(
            response=response,
            has_options=bool(sample.options),
            model_mode=model_mode,
        )
        print(f"[vqa_eval] sample_id={sample.sample_id} RAW:\n{response}", flush=True)
        print(f"[vqa_eval] sample_id={sample.sample_id} COT:\n{cot_text}", flush=True)
        print(f"[vqa_eval] sample_id={sample.sample_id} ANS:\n{ans_text}", flush=True)
        print(
            f"[vqa_eval] sample_id={sample.sample_id} TOKENS: "
            f"generated={generated_token_count}, limit={max_new_tokens}, hit_limit={hit_max_tokens}, "
            f"has_think_end={has_think_end}, model_mode={model_mode}, "
            f"require_think_end_for_scoring={require_think_end_for_scoring}",
            flush=True,
        )
        answer_usable = has_think_end or (not require_think_end_for_scoring)
        if not answer_usable:
            pred_answer = None
        results["inference_times"].append(inference_time)

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
        choices=[
            "uniform",
            "random",
            "focus",
            "sevila",
            "videoagent",
            "clip",
            "clip-new",
            "siglip2",
            "siglip2-new",
            "qframe",
            "bolt-clip",
            "bolt-siglip2",
            "aks",
            "aks-blip",
            "aks-clip",
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
        help="all/mcq/numeric/generation，或数据集特定桶（如 videomme: short/medium/long；mlvu: plotQA/anomaly_reco/...）",
    )
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument(
        "--model_mode_config",
        type=str,
        default="config/model_response_modes.json",
        help="模型响应模式配置文件(JSON): 通过规则自动判断 thinking/instruct。",
    )
    p.add_argument("--log_file", type=str, default="vqa_evaluation_log.csv")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--use_train_split", action="store_true")
    p.add_argument(
        "--use_preprocessed_clip_frames",
        action="store_true",
        help="启用后，clip/aks 会优先读取离线预处理帧。",
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
    p.add_argument("--use_subtitles", action="store_true", help="为 Video-MME 按采样帧时间对齐读取字幕并拼入 prompt")
    p.add_argument("--subtitles_dir", type=str, default="", help="字幕目录（可选）；为空时尝试在视频邻近目录下自动查找 subtitle/subtitles")
    return p.parse_args()


def main():
    experiment_start_time = time.perf_counter()
    args = parse_args()
    apply_dataset_cli_defaults(args)
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
                "use_preprocessed_clip_frames 仅支持 clip/clip-new/aks/aks-blip/aks-clip/uniform/random/siglip2/siglip2-new/qframe/bolt-clip/bolt-siglip2，"
                f"当前 frame_sampling_method={args.frame_sampling_method}"
            )
        print(
            "[vqa_eval] 启用预处理 clip 帧: "
            f"dir={preprocessed_clip_dir}, fps={args.preprocessed_clip_fps:g}",
            flush=True,
        )

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
    print(
        f"[vqa_eval] 模型模式识别: model_mode={model_mode}, "
        f"require_think_end_for_scoring={require_think_end_for_scoring}, "
        f"matched_rule={matched_rule}",
        flush=True,
    )
    effective_max_new_tokens = MODE_MAX_NEW_TOKENS[model_mode]
    print(
        f"[vqa_eval] max_new_tokens 已按模式固定: mode={model_mode}, "
        f"effective_max_new_tokens={effective_max_new_tokens}",
        flush=True,
    )

    apply_pixel_limits = dataset_uses_vl_pixel_limits(
        args.dataset,
        args.dataset_split,
        args.dataset_name,
    )
    if apply_pixel_limits:
        print(
            "[vqa_eval] MLVU-Test：启用 processor max_pixels 限制（防超高分辨率 OOM）",
            flush=True,
        )
    model, processor = load_model_and_processor(
        resolved_model_path,
        use_lora=args.use_lora,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
        apply_pixel_limits=apply_pixel_limits,
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
        model_mode=model_mode,
        require_think_end_for_scoring=require_think_end_for_scoring,
        use_preprocessed_clip_frames=args.use_preprocessed_clip_frames,
        preprocessed_clip_dir=preprocessed_clip_dir,
        use_subtitles=bool(args.use_subtitles),
        subtitles_dir=(args.subtitles_dir.strip() if args.subtitles_dir.strip() else None),
    )
    avg_accuracy, avg_inference_time = _compute_accuracy_from_results(results, args.task_filter)
    evaluated_samples, correct_count = _compute_score_counts_for_csv(results, args.task_filter)
    avg_frame_sampling_time = _compute_avg_frame_sampling_time(results)
    avg_embedding_build_time = 0.0
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
        avg_total_time_hours=avg_total_time_hours,
        over_max_tokens_count=results["over_max_tokens_count"],
        model_name=model_name,
        lora_path=lora_path,
        train_ratio=args.train_ratio,
        use_train_split=args.use_train_split,
    )
    print(
        f"评估完成：样本 {len(samples)}, Accuracy {avg_accuracy:.2f}%, "
        f"AvgInfer {avg_inference_time:.3f}s, AvgFrameSampling {avg_frame_sampling_time:.6f}s, "
        f"AvgEmbedBuild {avg_embedding_build_time:.6f}s, AvgTotal {avg_total_time_hours:.6f}h, "
        f"OverLimit {results['over_max_tokens_count']}, MissingThinkEnd {results['missing_think_end_count']}"
    )


if __name__ == "__main__":
    main()

