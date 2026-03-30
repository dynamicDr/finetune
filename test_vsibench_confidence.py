"""
test_vsibench_confidence.py

在 VSI-Bench 上评估：
- 原答案准确率
- 每个样本的 confidence score
- 高信心(>threshold) 与 低信心(<=threshold) 子集准确率
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from tqdm import tqdm

from train_vsibench import build_user_text
from test_vsibench import (
    calculate_mra,
    extract_answer,
    extract_video_frames,
    load_model_and_processor,
    should_include_sample,
    split_indices,
)


class ConfidenceHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)
        self.act = nn.Sigmoid()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 统一到线性层权重 dtype，避免 bf16/fp32 矩阵乘法报错
        hidden_state = hidden_state.to(dtype=self.fc.weight.dtype)
        return self.act(self.fc(hidden_state))


def get_model_hidden_size(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("模型缺少 config，无法推断 hidden_size")
    if hasattr(cfg, "hidden_size") and cfg.hidden_size is not None:
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and getattr(cfg.text_config, "hidden_size", None) is not None:
        return int(cfg.text_config.hidden_size)
    raise ValueError("无法从模型配置中读取 hidden_size")


def _last_token_hidden_from_generate_output(gen_out) -> torch.Tensor | None:
    hidden_states = getattr(gen_out, "hidden_states", None)
    if not hidden_states:
        return None

    step_state = hidden_states[-1]
    if isinstance(step_state, (tuple, list)):
        layer_state = step_state[-1]
    else:
        layer_state = step_state

    if not torch.is_tensor(layer_state):
        return None
    if layer_state.ndim == 3:
        return layer_state[:, -1, :]
    if layer_state.ndim == 2:
        return layer_state
    return None


def generate_response_and_hidden(
    model,
    processor,
    frames: list,
    question: str,
    max_new_tokens: int = 256,
) -> tuple[str, torch.Tensor]:
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    seq = gen_out.sequences
    prompt_len = inputs.input_ids.shape[1]
    gen_ids = seq[:, prompt_len:]
    response = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    last_hidden = _last_token_hidden_from_generate_output(gen_out)
    if last_hidden is None:
        with torch.no_grad():
            fw = model(input_ids=seq, output_hidden_states=True, return_dict=True)
        last_hidden = fw.hidden_states[-1][:, -1, :]

    return response, last_hidden


def append_summary_csv(
    log_file: str,
    seed: int,
    task_filter: str,
    num_samples: str,
    num_frames: int,
    threshold: float,
    overall_acc: float,
    high_acc: float,
    low_acc: float,
    high_cnt: int,
    low_cnt: int,
    model_name: str,
):
    Path(log_file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.exists(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(
                [
                    "seed",
                    "task_filter",
                    "num_samples",
                    "num_frames",
                    "threshold",
                    "overall_acc",
                    "high_conf_acc",
                    "low_conf_acc",
                    "high_conf_count",
                    "low_conf_count",
                    "model_name",
                ]
            )
        w.writerow(
            [
                seed,
                task_filter,
                num_samples,
                num_frames,
                threshold,
                f"{overall_acc:.2f}",
                f"{high_acc:.2f}",
                f"{low_acc:.2f}",
                high_cnt,
                low_cnt,
                model_name,
            ]
        )


def parse_args():
    p = argparse.ArgumentParser(description="评估 VSI-Bench + Confidence Head")
    p.add_argument("--model_path", type=str, required=True, help="模型路径或 HF ID；use_lora=true 时为 adapter 路径")
    p.add_argument("--video_dir", type=str, default="~/dataset/vsi_bench")
    p.add_argument("--num_samples", type=str, default="all")
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task_filter", type=str, default="all", choices=["all", "mcq", "numeric"])
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--threshold", type=float, default=0.7, help="高置信阈值")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--log_file", type=str, default="confidence_eval_summary.csv")
    p.add_argument("--detail_file", type=str, default="confidence_eval_details.csv")

    p.add_argument("--confidence_head_path", type=str, required=True, help="训练得到的 confidence_head.pt")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--merge_lora", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_dir = os.path.expanduser(args.video_dir)
    eval_dir = Path(__file__).resolve().parent / "eval_csv"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_file = str(eval_dir / Path(args.log_file).name)
    detail_file = str(eval_dir / Path(args.detail_file).name)

    dataset = load_dataset("nyu-visionx/VSI-Bench", split="test")
    filtered_indices = [i for i in range(len(dataset)) if should_include_sample(dataset[i], args.task_filter)]
    test_indices = split_indices(filtered_indices, args.seed, args.train_ratio, use_train_split=False)
    if args.num_samples.lower() != "all":
        n = int(args.num_samples)
        random.seed(args.seed + 1000)
        test_indices = random.sample(test_indices, min(n, len(test_indices)))

    model, processor = load_model_and_processor(
        args.model_path,
        use_lora=args.use_lora,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
    )
    model.eval()

    conf_head = ConfidenceHead(get_model_hidden_size(model)).to(model.device)
    conf_head.load_state_dict(torch.load(os.path.expanduser(args.confidence_head_path), map_location=model.device))
    conf_head.eval()

    total = 0
    correct = 0
    high_total = 0
    high_correct = 0
    low_total = 0
    low_correct = 0
    detail_rows: list[dict] = []

    pbar = tqdm(test_indices, desc="评估(含置信度)")
    for idx in pbar:
        sample = dataset[idx]
        scene_name = sample.get("scene_name", "")
        video_path = os.path.join(video_dir, f"{scene_name}.mp4")
        if not os.path.exists(video_path):
            continue

        frames = extract_video_frames(video_path, num_frames=args.num_frames)
        if not frames:
            continue

        question = sample.get("question", "")
        options = sample.get("options", None)
        gt_answer = sample.get("ground_truth", "")
        prompt = build_user_text(question, options)

        response, last_hidden = generate_response_and_hidden(
            model,
            processor,
            frames,
            prompt,
            max_new_tokens=args.max_new_tokens,
        )
        pred_answer = extract_answer(response, has_options=bool(options))
        conf = float(conf_head(last_hidden).item())

        if options:
            is_correct = str(gt_answer).strip().upper() == str(pred_answer).strip().upper()
        else:
            try:
                pred_num = float(pred_answer) if pred_answer else 0.0
                gt_num = float(gt_answer)
                is_correct = calculate_mra(pred_num, gt_num) > 0.5
            except (TypeError, ValueError):
                is_correct = False

        total += 1
        correct += int(is_correct)
        if conf > args.threshold:
            high_total += 1
            high_correct += int(is_correct)
        else:
            low_total += 1
            low_correct += int(is_correct)

        detail_rows.append(
            {
                "index": idx,
                "question_type": "mcq" if options else "numeric",
                "ground_truth": gt_answer,
                "prediction": pred_answer,
                "is_correct": int(is_correct),
                "confidence": f"{conf:.6f}",
            }
        )
        pbar.set_postfix(acc=f"{100.0 * correct / max(total, 1):.2f}%")

    overall_acc = 100.0 * correct / max(total, 1)
    high_acc = 100.0 * high_correct / max(high_total, 1)
    low_acc = 100.0 * low_correct / max(low_total, 1)

    model_name = args.base_model if args.use_lora and args.base_model else args.model_path
    append_summary_csv(
        summary_file,
        args.seed,
        args.task_filter,
        args.num_samples,
        args.num_frames,
        args.threshold,
        overall_acc,
        high_acc,
        low_acc,
        high_total,
        low_total,
        model_name,
    )

    with open(detail_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["index", "question_type", "ground_truth", "prediction", "is_correct", "confidence"],
        )
        w.writeheader()
        w.writerows(detail_rows)

    summary = {
        "overall_acc": round(overall_acc, 2),
        "high_conf_acc": round(high_acc, 2),
        "low_conf_acc": round(low_acc, 2),
        "high_conf_count": high_total,
        "low_conf_count": low_total,
        "threshold": args.threshold,
        "summary_csv": summary_file,
        "detail_csv": detail_file,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

