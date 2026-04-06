"""
train_vsibench_confidence.py

在现有 VSI-Bench 评测流程上训练 Confidence Head：
- 冻结 Qwen（含 LoRA/基座）全部参数
- 只训练 ConfidenceHead（Linear + Sigmoid）
- 用“答案是否正确”作为 confidence 标签（1/0）
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from train_vsibench import build_user_text
from hf_dataset_loader import load_dataset
from test_vsibench import (
    calculate_mra,
    extract_answer,
    extract_video_frames,
    load_model_and_processor,
    should_include_sample,
    split_indices,
)


class ConfidenceHead(nn.Module):
    """使用最后一个 token 的 hidden state 预测置信度。"""

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
    """尽可能从 generate(return_dict_in_generate=True, output_hidden_states=True) 中取最后 token 的 hidden。"""
    hidden_states = getattr(gen_out, "hidden_states", None)
    if not hidden_states:
        return None

    # 常见结构：tuple(steps) -> tuple(layers) -> tensor[B, T, H]
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
        # 回退：无法从 generate 提取时，用完整序列前向取最后 token hidden。
        with torch.no_grad():
            fw = model(
                input_ids=seq,
                output_hidden_states=True,
                return_dict=True,
            )
        last_hidden = fw.hidden_states[-1][:, -1, :]

    return response, last_hidden


def parse_args():
    p = argparse.ArgumentParser(description="训练 VSI-Bench Confidence Head（冻结 Qwen，仅训练置信头）")
    p.add_argument("--model_path", type=str, required=True, help="模型路径或 HuggingFace ID")
    p.add_argument("--video_dir", type=str, default="~/dataset/vsi_bench")
    p.add_argument("--output_dir", type=str, default="./outputs/vsibench_confidence_head")
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--task_filter", type=str, default="all", choices=["all", "mcq", "numeric"])
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=9999999999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--max_new_tokens", type=int, default=128)

    # 兼容现有 LoRA 评测路径
    p.add_argument("--use_lora", action="store_true", help="model_path 是否为 LoRA adapter 目录")
    p.add_argument("--base_model", type=str, default=None, help="use_lora=true 时必填基座模型")
    p.add_argument("--merge_lora", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_dir = os.path.expanduser(args.video_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("nyu-visionx/VSI-Bench", split="test")
    filtered_indices = [i for i in range(len(dataset)) if should_include_sample(dataset[i], args.task_filter)]
    train_indices = split_indices(filtered_indices, args.seed, args.train_ratio, use_train_split=True)
    if args.max_samples is not None:
        train_indices = train_indices[: args.max_samples]

    model, processor = load_model_and_processor(
        args.model_path,
        use_lora=args.use_lora,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = get_model_hidden_size(model)
    conf_head = ConfidenceHead(hidden_size).to(model.device)
    optimizer = torch.optim.AdamW(conf_head.parameters(), lr=args.learning_rate)
    bce_loss = nn.BCELoss()

    global_step = 0
    for epoch in range(args.num_train_epochs):
        conf_head.train()
        correct_count = 0
        seen_count = 0
        running_loss = 0.0

        pbar = tqdm(train_indices, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
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

            if options:
                is_correct = str(gt_answer).strip().upper() == str(pred_answer).strip().upper()
            else:
                try:
                    pred_num = float(pred_answer) if pred_answer else 0.0
                    gt_num = float(gt_answer)
                    is_correct = calculate_mra(pred_num, gt_num) > 0.5
                except (TypeError, ValueError):
                    is_correct = False

            target = torch.tensor([[1.0 if is_correct else 0.0]], device=model.device)
            conf_pred = conf_head(last_hidden.detach())
            loss = bce_loss(conf_pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            seen_count += 1
            correct_count += int(is_correct)
            running_loss += float(loss.item())
            pbar.set_postfix(
                conf_loss=f"{running_loss / max(seen_count, 1):.4f}",
                ans_acc=f"{100.0 * correct_count / max(seen_count, 1):.2f}%",
            )

    ckpt_path = output_dir / "confidence_head.pt"
    torch.save(conf_head.state_dict(), ckpt_path)

    meta = {
        "hidden_size": hidden_size,
        "threshold_default": 0.7,
        "model_path": args.model_path,
        "use_lora": args.use_lora,
        "base_model": args.base_model,
        "task_filter": args.task_filter,
        "num_frames": args.num_frames,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
    }
    with open(output_dir / "confidence_head_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Confidence Head 已保存: {ckpt_path}")
    print(f"配置已保存: {output_dir / 'confidence_head_config.json'}")


if __name__ == "__main__":
    main()

