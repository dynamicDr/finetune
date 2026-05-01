from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import get_data_loader
from data_loaders.base import VQASample
from frame_samplers import sample_video_frames
from model_response_mode import load_model_response_mode_config, parse_response_by_mode, resolve_model_mode
from vl_common import generate_response, load_model_and_processor


MODE_MAX_NEW_TOKENS = {
    "thinking": 4086,
    "instruct": 128,
}


def build_user_text(question: str, options: list[str] | None) -> str:
    if options:
        return (
            f"{question}\n\nOptions:\n"
            + "\n".join(options)
            + "\n\nDirectly answer with the option letter only. Do not explain."
        )
    return f"{question}\n\nPlease provide the numerical answer directly."


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def resolve_model_path(base_path: str) -> str:
    base = Path(os.path.expanduser(base_path))
    snapshots = base / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"找不到 snapshots 目录: {snapshots}")
    snapshot = next(snapshots.iterdir(), None)
    if snapshot is None:
        raise FileNotFoundError(f"snapshots 目录为空: {snapshots}")
    return str(snapshot)


def _resolve_model_mode_for_eval(resolved_model_path: str, model_name: str) -> tuple[str, bool]:
    mode_config = load_model_response_mode_config(str(PROJECT_ROOT / "config/model_response_modes.json"))
    model_identifier_candidates = [resolved_model_path, model_name]

    last_error: Exception | None = None
    model_mode = ""
    require_think_end_for_scoring = True
    for candidate in model_identifier_candidates:
        try:
            model_mode, require_think_end_for_scoring, matched_rule = resolve_model_mode(
                model_identifier=candidate,
                config=mode_config,
            )
            print(
                f"[mode] model={model_name}, model_mode={model_mode}, "
                f"require_think_end_for_scoring={require_think_end_for_scoring}, matched_rule={matched_rule}",
                flush=True,
            )
            break
        except (KeyError, ValueError) as e:
            last_error = e

    if not model_mode:
        raise RuntimeError(
            "模型模式识别失败，请在 model_response_modes.json 中添加精确键。"
            f" tried={model_identifier_candidates}"
        ) from last_error

    return model_mode, require_think_end_for_scoring


def load_model_and_processor_with_accelerate_fallback(model_path: str):
    """
    优先复用项目原始加载逻辑；若当前环境缺 accelerate，则退化为不使用 device_map 的单设备加载。
    """
    try:
        return load_model_and_processor(
            model_path,
            use_lora=False,
            base_model=None,
            merge_lora=False,
        )
    except ValueError as e:
        msg = str(e)
        if "requires `accelerate`" not in msg:
            raise
        print(
            "[fallback] 检测到当前环境缺少 accelerate，切换为单设备加载（不使用 device_map='auto'）。",
            flush=True,
        )

    model_path_lower = model_path.lower()
    use_generic_loader = any(hint in model_path_lower for hint in ("qwen3-vl", "llava-onevision", "llava-next"))
    if use_generic_loader:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, processor


def evaluate_vqa_with_stage_prints(
    model,
    processor,
    samples: list[VQASample],
    num_frames: int,
    task_filter: str,
    frame_sampling_method: str,
    seed: int,
    max_new_tokens: int,
    model_mode: str,
    require_think_end_for_scoring: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "evaluated": 0,
        "correct": 0,
        "inference_times": [],
        "frame_sampling_times": [],
        "prompt_build_times": [],
        "parse_times": [],
        "sample_total_times": [],
        "over_max_tokens_count": 0,
        "missing_think_end_count": 0,
    }

    pbar = tqdm(samples, desc="评估进度")
    for i, sample in enumerate(pbar):
        if task_filter != "all" and sample.task_type != task_filter:
            continue

        sample_t0 = time.perf_counter()

        random_seed = (seed + i) if frame_sampling_method == "random" else None
        t0 = time.perf_counter()
        frames = sample_video_frames(
            video_path=sample.video_path,
            num_frames=num_frames,
            method=frame_sampling_method,
            random_seed=random_seed,
            question=sample.question,
            options=sample.options,
            answer=str(sample.answer),
        )
        frame_sampling_t = time.perf_counter() - t0
        results["frame_sampling_times"].append(frame_sampling_t)
        if not frames:
            warnings.warn(
                f"样本无可用帧，已跳过: sample_id={sample.sample_id}, video_path={sample.video_path}",
                RuntimeWarning,
                stacklevel=1,
            )
            continue

        t0 = time.perf_counter()
        prompt = build_user_text(sample.question, sample.options)
        prompt_build_t = time.perf_counter() - t0
        results["prompt_build_times"].append(prompt_build_t)

        # generate_response 内部会打印 processor/generate/decode 拆分耗时
        response, inference_time, generated_token_count, hit_max_tokens = generate_response(
            model,
            processor,
            frames,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        results["inference_times"].append(inference_time)
        if hit_max_tokens:
            results["over_max_tokens_count"] += 1

        t0 = time.perf_counter()
        has_think_end = "</think>" in response
        if not has_think_end:
            results["missing_think_end_count"] += 1
        _, _, pred_answer = parse_response_by_mode(
            response=response,
            has_options=bool(sample.options),
            model_mode=model_mode,
        )
        parse_t = time.perf_counter() - t0
        results["parse_times"].append(parse_t)

        answer_usable = has_think_end or (not require_think_end_for_scoring)
        is_correct = answer_usable and (
            str(sample.answer).strip().upper() == str(pred_answer).strip().upper()
        )
        results["evaluated"] += 1
        if is_correct:
            results["correct"] += 1

        sample_total_t = time.perf_counter() - sample_t0
        results["sample_total_times"].append(sample_total_t)

        print(
            f"[stage][sample] sample_id={sample.sample_id}, "
            f"frame_sampling={frame_sampling_t:.3f}s, "
            f"prompt_build={prompt_build_t:.3f}s, "
            f"inference(generate)={inference_time:.3f}s, "
            f"parse={parse_t:.3f}s, "
            f"sample_total={sample_total_t:.3f}s, "
            f"generated_tokens={generated_token_count}, "
            f"hit_max_tokens={hit_max_tokens}, "
            f"has_think_end={has_think_end}, "
            f"is_correct={is_correct}",
            flush=True,
        )

        acc = (results["correct"] / results["evaluated"] * 100.0) if results["evaluated"] else 0.0
        pbar.set_postfix(Acc=f"{acc:.2f}%", AvgTime=f"{_avg(results['inference_times']):.2f}s")

    return results


def print_model_summary(model_alias: str, model_name: str, model_path: str, results: dict[str, Any]) -> None:
    evaluated = int(results["evaluated"])
    correct = int(results["correct"])
    accuracy = (correct / evaluated * 100.0) if evaluated else 0.0
    print(f"\n[summary][{model_alias}] model_name={model_name}", flush=True)
    print(f"[summary][{model_alias}] model_path={model_path}", flush=True)
    print(
        f"[summary][{model_alias}] evaluated={evaluated}, correct={correct}, accuracy={accuracy:.2f}%",
        flush=True,
    )
    print(
        f"[summary][{model_alias}] avg_frame_sampling={_avg(results['frame_sampling_times']):.3f}s, "
        f"avg_prompt_build={_avg(results['prompt_build_times']):.3f}s, "
        f"avg_inference={_avg(results['inference_times']):.3f}s, "
        f"avg_parse={_avg(results['parse_times']):.3f}s, "
        f"avg_sample_total={_avg(results['sample_total_times']):.3f}s",
        flush=True,
    )
    print(
        f"[summary][{model_alias}] over_max_tokens_count={results['over_max_tokens_count']}, "
        f"missing_think_end_count={results['missing_think_end_count']}",
        flush=True,
    )


def print_cross_model_diff(lhs_alias: str, lhs: dict[str, Any], rhs_alias: str, rhs: dict[str, Any]) -> None:
    print(f"\n[diff] compare: lhs={lhs_alias}, rhs={rhs_alias} (rhs/lhs)", flush=True)
    keys = [
        ("avg_frame_sampling", "frame_sampling_times"),
        ("avg_prompt_build", "prompt_build_times"),
        ("avg_inference", "inference_times"),
        ("avg_parse", "parse_times"),
        ("avg_sample_total", "sample_total_times"),
    ]
    for label, key in keys:
        lv = _avg(lhs[key])
        rv = _avg(rhs[key])
        ratio = (rv / lv) if lv > 0 else float("inf")
        print(
            f"[diff] {label}: {lhs_alias}={lv:.3f}s, {rhs_alias}={rv:.3f}s, "
            f"delta={rv - lv:+.3f}s, ratio={ratio:.2f}x",
            flush=True,
        )


def main() -> None:
    # 固定参数（不使用命令行）
    dataset = "videomme"
    dataset_split = "test"
    dataset_name = "lmms-lab/Video-MME"
    no_dataset_config = True
    video_dir = os.path.expanduser("~/dataset/Video-MME")
    frame_sampling_method = "uniform"
    num_frames = 4
    task_filter = "short"
    train_ratio = 0.9
    seed = 42
    num_samples = 3

    model_settings = [
        {
            "alias": "llava_next_video_7b_hf",
            "model_name": "llava-hf/LLaVA-NeXT-Video-7B-hf",
            "cache_base": os.path.expanduser("~/.cache/huggingface/hub/models--llava-hf--LLaVA-NeXT-Video-7B-hf"),
        },
        {
            "alias": "qwen3_4b_instruct",
            "model_name": "Qwen/Qwen3-VL-4B-Instruct",
            "cache_base": os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct"),
        },
    ]

    print("=== 固定参数复现实验开始 ===", flush=True)
    print(
        f"[config] dataset={dataset}, split={dataset_split}, dataset_name={dataset_name}, "
        f"video_dir={video_dir}, task_filter={task_filter}, num_samples={num_samples}, "
        f"num_frames={num_frames}, frame_sampling_method={frame_sampling_method}, "
        f"train_ratio={train_ratio}, seed={seed}",
        flush=True,
    )

    t0 = time.perf_counter()
    loader = get_data_loader(
        dataset,
        video_dir=video_dir,
        seed=seed,
        train_ratio=train_ratio,
        task_filter=task_filter,
        dataset_name=dataset_name,
        dataset_config=None,
        no_dataset_config=no_dataset_config,
    )
    samples = loader.get_split_samples(
        split=dataset_split,
        use_train_split=False,
        sample_count=num_samples,
    )
    data_load_t = time.perf_counter() - t0
    print(f"[stage][global] data_load_and_sample={data_load_t:.3f}s, selected={len(samples)}", flush=True)

    all_results: dict[str, dict[str, Any]] = {}

    for ms in model_settings:
        alias = ms["alias"]
        model_name = ms["model_name"]
        cache_base = ms["cache_base"]

        print(f"\n========== model={alias} ==========", flush=True)
        resolved_model_path = resolve_model_path(cache_base)
        print(f"[stage][global] resolve_model_path={resolved_model_path}", flush=True)

        model_mode, require_think_end_for_scoring = _resolve_model_mode_for_eval(
            resolved_model_path=resolved_model_path,
            model_name=model_name,
        )
        max_new_tokens = MODE_MAX_NEW_TOKENS[model_mode]
        print(f"[config] model={alias}, mode={model_mode}, max_new_tokens={max_new_tokens}", flush=True)

        t0 = time.perf_counter()
        model, processor = load_model_and_processor_with_accelerate_fallback(resolved_model_path)
        model_load_t = time.perf_counter() - t0
        print(f"[stage][global] model_load={model_load_t:.3f}s", flush=True)

        t0 = time.perf_counter()
        results = evaluate_vqa_with_stage_prints(
            model=model,
            processor=processor,
            samples=samples,
            num_frames=num_frames,
            task_filter=task_filter,
            frame_sampling_method=frame_sampling_method,
            seed=seed,
            max_new_tokens=max_new_tokens,
            model_mode=model_mode,
            require_think_end_for_scoring=require_think_end_for_scoring,
        )
        eval_t = time.perf_counter() - t0
        print(f"[stage][global] eval_loop_total={eval_t:.3f}s", flush=True)

        results["model_load_time"] = model_load_t
        all_results[alias] = results
        print_model_summary(alias, model_name, resolved_model_path, results)

    if "llava_next_video_7b_hf" in all_results and "qwen3_4b_instruct" in all_results:
        llava_res = all_results["llava_next_video_7b_hf"]
        qwen_res = all_results["qwen3_4b_instruct"]
        lv = float(llava_res["model_load_time"])
        rv = float(qwen_res["model_load_time"])
        ratio = (rv / lv) if lv > 0 else float("inf")
        print(
            f"[diff] model_load: llava_next_video_7b_hf={lv:.3f}s, "
            f"qwen3_4b_instruct={rv:.3f}s, delta={rv - lv:+.3f}s, ratio={ratio:.2f}x",
            flush=True,
        )
        print_cross_model_diff(
            lhs_alias="llava_next_video_7b_hf",
            lhs=llava_res,
            rhs_alias="qwen3_4b_instruct",
            rhs=qwen_res,
        )


if __name__ == "__main__":
    main()
