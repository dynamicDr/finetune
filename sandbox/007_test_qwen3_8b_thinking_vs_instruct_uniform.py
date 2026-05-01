from __future__ import annotations

import csv
import os
import re
import sys
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import get_data_loader
from data_loaders.base import VQASample
from frame_samplers import sample_video_frames
from vl_common import generate_response_with_split_embedding, load_model_and_processor


def build_user_text(question: str, options: list[str] | None) -> str:
    if options:
        return (
            f"{question}\n\nOptions:\n"
            + "\n".join(options)
            + "\n\nDirectly answer with the option letter only. Do not explain."
        )
    return f"{question}\n\nPlease provide the numerical answer directly."


def _extract_answer_core(text: str, has_options: bool = False):
    text = text.strip()
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        text = answer_match.group(1).strip()
    if has_options:
        match = re.search(r"\b([A-E])\b", text.upper())
        if match:
            return match.group(1)
    return text


def extract_answer_for_thinking(response: str, has_options: bool = False):
    response = response.strip()
    if "</think>" in response:
        answer_part = response.split("</think>", 1)[-1].strip()
    else:
        answer_part = response
    return _extract_answer_core(answer_part, has_options=has_options)


def extract_answer_for_instruct(response: str, has_options: bool = False):
    response = response.strip()
    if "</think>" in response:
        answer_part = response.split("</think>", 1)[-1].strip()
    else:
        answer_part = response
    return _extract_answer_core(answer_part, has_options=has_options)


def _init_stats() -> dict[str, float | int]:
    return {
        "count": 0,
        "correct": 0,
        "missing_think_end": 0,
        "over_max_tokens": 0,
        "total": 0.0,
        "prompt_build": 0.0,
        "frame_sampling": 0.0,
        "model_call_total": 0.0,
        "embedding_build": 0.0,
        "inference": 0.0,
        "model_other": 0.0,
        "answer_extract": 0.0,
        "score_update": 0.0,
    }


def _pick_samples(samples: list[VQASample], tasks: list[str], n_each: int) -> dict[str, list[VQASample]]:
    buckets: dict[str, list[VQASample]] = {t: [] for t in tasks}
    for s in samples:
        if s.task_type in buckets:
            buckets[s.task_type].append(s)
    selected: dict[str, list[VQASample]] = {}
    for t in tasks:
        selected[t] = buckets[t][:n_each]
    return selected


MODE_MAX_NEW_TOKENS = {
    "thinking": 4086,
    "instruct": 128,
}


def run() -> None:
    # ====== 全部硬编码配置 ======
    DATASET = "videomme"
    DATASET_SPLIT = "test"
    DATASET_NAME = "lmms-lab/Video-MME"
    DATASET_CONFIG = None
    NO_DATASET_CONFIG = True
    VIDEO_DIR = os.path.expanduser("~/dataset/videomme")

    MODEL_CONFIGS = [
        {
            "model_key": "thinking",
            "model_path": "Qwen/Qwen3-VL-8B-Thinking",
            "extract_answer_fn": extract_answer_for_thinking,
            "require_think_end_for_correct": True,
        },
        {
            "model_key": "instruct",
            "model_path": "Qwen/Qwen3-VL-8B-Instruct",
            "extract_answer_fn": extract_answer_for_instruct,
            "require_think_end_for_correct": False,
        },
    ]
    USE_LORA = False
    BASE_MODEL = None
    MERGE_LORA = False

    NUM_FRAMES = 8
    FRAME_SAMPLING_METHOD = "uniform"

    SEED = 42
    TRAIN_RATIO = 0.8
    USE_TRAIN_SPLIT = False

    TASKS = ["short", "medium"]
    N_EACH = 3
    # ==========================

    print("加载数据...", flush=True)
    loader = get_data_loader(
        DATASET,
        video_dir=VIDEO_DIR,
        seed=SEED,
        train_ratio=TRAIN_RATIO,
        task_filter="all",
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        no_dataset_config=NO_DATASET_CONFIG,
    )
    all_samples = loader.get_split_samples(
        split=DATASET_SPLIT,
        use_train_split=USE_TRAIN_SPLIT,
        sample_count=None,
    )
    selected = _pick_samples(all_samples, TASKS, N_EACH)
    for task in TASKS:
        print(f"{task}: 可用={len([s for s in all_samples if s.task_type == task])}, 取样={len(selected[task])}")

    # 为保证 Thinking/Instruct 输入一致，这里先对每个样本做一次 uniform 取帧并缓存。
    frame_cache: dict[str, dict[str, object]] = {}
    for task in TASKS:
        for sample in selected[task]:
            t0 = time.perf_counter()
            frames = sample_video_frames(
                video_path=sample.video_path,
                num_frames=NUM_FRAMES,
                method=FRAME_SAMPLING_METHOD,
                random_seed=None,
                question=sample.question,
                options=sample.options,
                answer=str(sample.answer),
            )
            frame_sampling_time = time.perf_counter() - t0
            if not frames:
                warnings.warn(
                    f"样本无可用帧，后续将跳过: sample_id={sample.sample_id}, video_path={sample.video_path}",
                    RuntimeWarning,
                    stacklevel=1,
                )
            frame_cache[str(sample.sample_id)] = {
                "frames": frames,
                "frame_sampling_time": frame_sampling_time,
            }
    print(f"uniform帧缓存完成: {len(frame_cache)} 个样本", flush=True)

    detail_rows: list[dict[str, str | int | float]] = []
    stats_by_model_task: dict[str, dict[str, dict[str, float | int]]] = {}

    for cfg in MODEL_CONFIGS:
        model_key = cfg["model_key"]
        model_path = cfg["model_path"]
        extract_answer_fn = cfg["extract_answer_fn"]
        require_think_end_for_correct = bool(cfg["require_think_end_for_correct"])
        effective_max_new_tokens = MODE_MAX_NEW_TOKENS[model_key]

        print(f"\n================ 加载模型: {model_key} ({model_path}) ================", flush=True)
        print(
            f"[config] max_new_tokens 已按模式固定: model_key={model_key}, "
            f"effective_max_new_tokens={effective_max_new_tokens}",
            flush=True,
        )
        model, processor = load_model_and_processor(
            model_path,
            use_lora=USE_LORA,
            base_model=BASE_MODEL,
            merge_lora=MERGE_LORA,
        )
        stats_by_model_task[model_key] = {t: _init_stats() for t in TASKS}

        for task in TASKS:
            samples = selected[task]
            print(
                f"\n===== 开始测试 model={model_key}, task={task}, num_samples={len(samples)} =====",
                flush=True,
            )
            pbar = tqdm(samples, desc=f"{model_key}-{task}-uniform")
            for idx, sample in enumerate(pbar, start=1):
                sample_start = time.perf_counter()
                print(f"==============={sample.sample_id}=============", flush=True)
                print(f"ground truth: {sample.answer}", flush=True)

                t0 = time.perf_counter()
                prompt = build_user_text(sample.question, sample.options)
                prompt_t = time.perf_counter() - t0

                cache_entry = frame_cache[str(sample.sample_id)]
                frames = cache_entry["frames"]
                frame_t = float(cache_entry["frame_sampling_time"])
                if not frames:
                    print(f"sample_id={sample.sample_id} 无可用帧，跳过。", flush=True)
                    continue

                t0 = time.perf_counter()
                response, infer_t, embedding_t, generated_token_count, hit_max_tokens = (
                    generate_response_with_split_embedding(
                        model=model,
                        processor=processor,
                        frames=frames,
                        prompt=prompt,
                        max_new_tokens=effective_max_new_tokens,
                    )
                )
                model_call_t = time.perf_counter() - t0
                model_other_t = max(0.0, model_call_t - embedding_t - infer_t)

                t0 = time.perf_counter()
                pred_answer = extract_answer_fn(response, has_options=bool(sample.options))
                has_think_end = "</think>" in response
                answer_extract_t = time.perf_counter() - t0

                t0 = time.perf_counter()
                basic_match = (
                    str(sample.answer).strip().upper() == str(pred_answer).strip().upper()
                )
                if require_think_end_for_correct:
                    is_correct = has_think_end and basic_match
                else:
                    is_correct = basic_match
                score_update_t = time.perf_counter() - t0

                sample_total_t = time.perf_counter() - sample_start

                st = stats_by_model_task[model_key][task]
                st["count"] += 1
                st["correct"] += int(is_correct)
                st["missing_think_end"] += int(not has_think_end)
                st["over_max_tokens"] += int(hit_max_tokens)
                st["total"] += sample_total_t
                st["prompt_build"] += prompt_t
                st["frame_sampling"] += frame_t
                st["model_call_total"] += model_call_t
                st["embedding_build"] += embedding_t
                st["inference"] += infer_t
                st["model_other"] += model_other_t
                st["answer_extract"] += answer_extract_t
                st["score_update"] += score_update_t

                pbar.set_postfix(
                    acc=f"{(st['correct'] / st['count'] * 100) if st['count'] else 0.0:.2f}%",
                    avg_total=f"{(st['total'] / st['count']) if st['count'] else 0.0:.2f}s",
                    n=st["count"],
                )

                detail_rows.append(
                    {
                        "model_key": model_key,
                        "model_path": model_path,
                        "task": task,
                        "sample_idx": idx,
                        "sample_id": sample.sample_id,
                        "video_path": sample.video_path,
                        "gt_answer": str(sample.answer),
                        "pred_answer": str(pred_answer),
                        "is_correct": int(is_correct),
                        "basic_match": int(basic_match),
                        "require_think_end_for_correct": int(require_think_end_for_correct),
                        "has_think_end": int(has_think_end),
                        "frame_sampling_method": FRAME_SAMPLING_METHOD,
                        "num_frames": NUM_FRAMES,
                        "selected_frame_count": len(frames),
                        "embedding_build_time_s": float(embedding_t),
                        "inference_time_s": float(infer_t),
                        "model_call_total_s": float(model_call_t),
                        "model_other_time_s": float(model_other_t),
                        "prompt_build_time_s": float(prompt_t),
                        "frame_sampling_time_s": float(frame_t),
                        "answer_extract_time_s": float(answer_extract_t),
                        "score_update_time_s": float(score_update_t),
                        "sample_total_time_s": float(sample_total_t),
                        "generated_token_count": int(generated_token_count),
                        "hit_max_tokens": int(bool(hit_max_tokens)),
                    }
                )

                print(
                    f"[timing][sample] model={model_key}, task={task}, idx={idx}, sample_id={sample.sample_id}, "
                    f"total={sample_total_t:.4f}s, prompt_build={prompt_t:.4f}s, frame_sampling={frame_t:.4f}s, "
                    f"model_call_total={model_call_t:.4f}s, embedding_build={embedding_t:.4f}s, "
                    f"inference={infer_t:.4f}s, model_other={model_other_t:.4f}s, "
                    f"answer_extract={answer_extract_t:.4f}s, score_update={score_update_t:.4f}s, "
                    f"is_correct={is_correct}, basic_match={basic_match}, has_think_end={has_think_end}, "
                    f"generated_tokens={generated_token_count}, hit_max_tokens={hit_max_tokens}",
                    flush=True,
                )

    print("\n================ 最终统计（按模型 + short/medium） ================", flush=True)
    for model_key in stats_by_model_task:
        print(f"\n----- model={model_key} -----", flush=True)
        for task in TASKS:
            st = stats_by_model_task[model_key][task]
            cnt = int(st["count"])
            if cnt == 0:
                print(f"task={task}: 无样本", flush=True)
                continue
            acc = float(st["correct"]) / cnt * 100.0
            print(
                f"task={task}, count={cnt}, acc={acc:.2f}%, missing_think_end={int(st['missing_think_end'])}, "
                f"over_max_tokens={int(st['over_max_tokens'])}",
                flush=True,
            )
            print(
                f"  avg_total={float(st['total']) / cnt:.4f}s, "
                f"avg_prompt_build={float(st['prompt_build']) / cnt:.4f}s, "
                f"avg_frame_sampling={float(st['frame_sampling']) / cnt:.4f}s, "
                f"avg_model_call_total={float(st['model_call_total']) / cnt:.4f}s, "
                f"avg_embedding_build={float(st['embedding_build']) / cnt:.4f}s, "
                f"avg_inference={float(st['inference']) / cnt:.4f}s, "
                f"avg_model_other={float(st['model_other']) / cnt:.4f}s, "
                f"avg_answer_extract={float(st['answer_extract']) / cnt:.4f}s, "
                f"avg_score_update={float(st['score_update']) / cnt:.4f}s",
                flush=True,
            )

    detail_csv = Path(__file__).resolve().parent / (
        "006_test_qwen3_8b_thinking_vs_instruct_uniform_details_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".csv"
    )
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model_key",
            "model_path",
            "task",
            "sample_idx",
            "sample_id",
            "video_path",
            "gt_answer",
            "pred_answer",
            "is_correct",
            "basic_match",
            "require_think_end_for_correct",
            "has_think_end",
            "frame_sampling_method",
            "num_frames",
            "selected_frame_count",
            "embedding_build_time_s",
            "inference_time_s",
            "model_call_total_s",
            "model_other_time_s",
            "prompt_build_time_s",
            "frame_sampling_time_s",
            "answer_extract_time_s",
            "score_update_time_s",
            "sample_total_time_s",
            "generated_token_count",
            "hit_max_tokens",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"明细表已写入: {detail_csv}", flush=True)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(__file__).resolve().parent / (
        f"006_test_qwen3_8b_thinking_vs_instruct_uniform_{ts}.log"
    )
    with open(log_file, "w", encoding="utf-8") as f:
        with redirect_stdout(f), redirect_stderr(f):
            run()


if __name__ == "__main__":
    main()
