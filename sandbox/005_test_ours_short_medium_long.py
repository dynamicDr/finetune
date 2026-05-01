from __future__ import annotations

import csv
import os
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from data_loaders import get_data_loader
from data_loaders.base import VQASample
from data_loaders.ours import iterative_inference_with_cache, rank_frames_by_clip
from vl_common import load_model_and_processor


def build_user_text(question: str, options: list[str] | None) -> str:
    if options:
        return (
            f"{question}\n\nOptions:\n"
            + "\n".join(options)
            + "\n\nDirectly answer with the option letter only. Do not explain."
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
    return answer_portion


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


THINKING_MAX_NEW_TOKENS = 4086


def run() -> None:
    # ====== 全部硬编码配置 ======
    DATASET = "videomme"
    DATASET_SPLIT = "test"
    DATASET_NAME = "lmms-lab/Video-MME"
    DATASET_CONFIG = None
    NO_DATASET_CONFIG = True
    VIDEO_DIR = os.path.expanduser("~/dataset/videomme")

    MODEL_PATH = "Qwen/Qwen3-VL-4B-Thinking"
    USE_LORA = False
    BASE_MODEL = None
    MERGE_LORA = False

    NUM_FRAMES = 8
    OURS_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    OURS_CLIP_DEVICE = None
    OURS_CLIP_BATCH_SIZE = 16

    SEED = 42
    TRAIN_RATIO = 0.8
    USE_TRAIN_SPLIT = False

    TASKS = ["short", "medium", "long"]
    N_EACH = 3
    # ==========================

    print("加载数据与模型...", flush=True)
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

    model, processor = load_model_and_processor(
        MODEL_PATH,
        use_lora=USE_LORA,
        base_model=BASE_MODEL,
        merge_lora=MERGE_LORA,
    )
    print(
        f"[config] max_new_tokens 已按模式固定: model_key=thinking, "
        f"effective_max_new_tokens={THINKING_MAX_NEW_TOKENS}",
        flush=True,
    )

    stats_by_task: dict[str, dict[str, float | int]] = {t: _init_stats() for t in TASKS}
    round_table_rows: list[dict[str, str | int | float]] = []

    for task in TASKS:
        samples = selected[task]
        print(f"\n===== 开始测试 task={task}, num_samples={len(samples)} =====", flush=True)
        pbar = tqdm(samples, desc=f"ours-{task}")
        for idx, sample in enumerate(pbar, start=1):
            sample_start = time.perf_counter()
            print(f"==============={sample.sample_id}=============", flush=True)
            print(f"ground truth: {sample.answer}", flush=True)

            t0 = time.perf_counter()
            prompt = build_user_text(sample.question, sample.options)
            prompt_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            ranked_frames = rank_frames_by_clip(
                video_path=sample.video_path,
                question=sample.question,
                options=sample.options,
                answer=str(sample.answer),
                clip_model_id=OURS_CLIP_MODEL_ID,
                clip_device=OURS_CLIP_DEVICE,
                clip_batch_size=OURS_CLIP_BATCH_SIZE,
                max_frames=NUM_FRAMES,
            )
            frame_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            iterative_out = iterative_inference_with_cache(
                model=model,
                processor=processor,
                prompt=prompt,
                ranked_frames=ranked_frames,
                extract_answer_fn=extract_answer,
                has_options=bool(sample.options),
                max_new_tokens=THINKING_MAX_NEW_TOKENS,
            )
            model_call_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            response = iterative_out["response"]
            pred_answer = iterative_out["pred_answer"]
            has_think_end = "</think>" in response
            answer_extract_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            is_correct = has_think_end and (
                str(sample.answer).strip().upper() == str(pred_answer).strip().upper()
            )
            score_update_t = time.perf_counter() - t0

            embedding_t = float(iterative_out["embedding_build_time"])
            infer_t = float(iterative_out["inference_time"])
            model_other_t = max(0.0, model_call_t - embedding_t - infer_t)
            sample_total_t = time.perf_counter() - sample_start
            hit_max_tokens = bool(iterative_out["hit_max_tokens"])
            round_details = iterative_out.get("round_details", [])

            for rd in round_details:
                option_probs = rd.get("option_probs", {})
                frame_ids = rd.get("selected_frame_ids", [])
                clip_scores = rd.get("selected_clip_scores", [])
                token_lps = rd.get("answer_token_logprobs", [])
                round_table_rows.append(
                    {
                        "task": task,
                        "sample_idx": idx,
                        "sample_id": sample.sample_id,
                        "round_idx": int(rd.get("round_idx", 0)),
                        "video_path": sample.video_path,
                        "gt_answer": str(sample.answer),
                        "pred_answer": str(rd.get("pred_answer", "")),
                        "is_correct": int(
                            str(sample.answer).strip().upper()
                            == str(rd.get("pred_answer", "")).strip().upper()
                        ),
                        "has_think_end": int(bool(rd.get("has_think_end", False))),
                        "selected_frame_count": len(frame_ids),
                        "selected_frame_ids": "|".join(str(x) for x in frame_ids),
                        "selected_clip_scores": "|".join(f"{float(x):.6f}" for x in clip_scores),
                        "option_prob_A": float(option_probs.get("A", 0.0)),
                        "option_prob_B": float(option_probs.get("B", 0.0)),
                        "option_prob_C": float(option_probs.get("C", 0.0)),
                        "option_prob_D": float(option_probs.get("D", 0.0)),
                        "answer_prob": float(rd.get("answer_prob", 0.0)),
                        "answer_logprob": float(rd.get("answer_logprob", float("-inf"))),
                        "answer_token_logprobs": "|".join(f"{float(x):.6f}" for x in token_lps),
                        "embedding_build_time_s": float(rd.get("embedding_build_time", 0.0)),
                        "inference_time_s": float(rd.get("inference_time", 0.0)),
                        "round_total_time_s": float(rd.get("embedding_build_time", 0.0))
                        + float(rd.get("inference_time", 0.0)),
                        "generated_token_count": int(rd.get("generated_token_count", 0)),
                        "hit_max_tokens": int(bool(rd.get("hit_max_tokens", False))),
                    }
                )

            st = stats_by_task[task]
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

            print(
                f"[timing][sample] task={task}, idx={idx}, sample_id={sample.sample_id}, "
                f"total={sample_total_t:.4f}s, prompt_build={prompt_t:.4f}s, frame_sampling={frame_t:.4f}s, "
                f"model_call_total={model_call_t:.4f}s, embedding_build={embedding_t:.4f}s, "
                f"inference={infer_t:.4f}s, model_other={model_other_t:.4f}s, "
                f"answer_extract={answer_extract_t:.4f}s, score_update={score_update_t:.4f}s, "
                f"is_correct={is_correct}, hit_max_tokens={hit_max_tokens}",
                flush=True,
            )

    print("\n================ 最终统计（按 short/medium/long） ================", flush=True)
    for task in TASKS:
        st = stats_by_task[task]
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
        "005_test_ours_short_medium_long_round_details_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".csv"
    )
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "task",
            "sample_idx",
            "sample_id",
            "round_idx",
            "video_path",
            "gt_answer",
            "pred_answer",
            "is_correct",
            "has_think_end",
            "selected_frame_count",
            "selected_frame_ids",
            "selected_clip_scores",
            "option_prob_A",
            "option_prob_B",
            "option_prob_C",
            "option_prob_D",
            "answer_prob",
            "answer_logprob",
            "answer_token_logprobs",
            "embedding_build_time_s",
            "inference_time_s",
            "round_total_time_s",
            "generated_token_count",
            "hit_max_tokens",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(round_table_rows)
    print(f"逐轮明细表已写入: {detail_csv}", flush=True)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(__file__).resolve().parent / f"005_test_ours_short_medium_long_{ts}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        with redirect_stdout(f), redirect_stderr(f):
            run()


if __name__ == "__main__":
    main()
