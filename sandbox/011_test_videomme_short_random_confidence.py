from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_loaders import get_data_loader
from frame_samplers import sample_video_frames
from model_response_mode import parse_response_by_mode
from vl_common import load_model_and_processor


# =========================
# 固定配置（无 args）
# =========================
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
DATASET = "videomme"
DATASET_SPLIT = "test"
TASK_FILTERS = ["short","medium","long"]  # 例: ["short", "medium"]
VIDEO_DIR = "/userhome/cs3/duanty/dataset/Video-MME"
NUM_FRAMES = 16
FRAME_SAMPLING_METHOD = "random"
SEED = 42
MAX_NEW_TOKENS = 128
MODEL_MODE = "instruct"

OUT_DIR = Path("outputs/videomme_short_random_confidence").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("加载模型...")
model, processor = load_model_and_processor(MODEL_PATH)

print("加载数据集...")
loader = get_data_loader(
    DATASET,
    video_dir=VIDEO_DIR,
    seed=SEED,
    train_ratio=0.8,
    task_filter="all",
)
samples = loader._convert_all(DATASET_SPLIT)
allowed_task_types = {str(x).strip().lower() for x in TASK_FILTERS if str(x).strip()}
if not allowed_task_types:
    raise ValueError("TASK_FILTERS 不能为空。")
samples = [s for s in samples if str(s.task_type).strip().lower() in allowed_task_types]
print(f"样本数（{sorted(allowed_task_types)}, 全量）: {len(samples)}")

rows: list[dict] = []
rng = np.random.default_rng(SEED)

for i, sample in enumerate(tqdm(samples, desc="评估中")):
    if not sample.options:
        continue

    prompt = (
        f"{sample.question}\n\nOptions:\n"
        + "\n".join([f"{chr(ord('A') + j)}. {opt}" for j, opt in enumerate(sample.options)])
        + "\n\nDirectly answer with the option letter only. Do not explain."
    )

    frames = sample_video_frames(
        video_path=sample.video_path,
        num_frames=NUM_FRAMES,
        method=FRAME_SAMPLING_METHOD,
        random_seed=SEED + i,
        sample_id=sample.sample_id,
        question=sample.question,
        options=sample.options,
        answer=str(sample.answer),
    )
    if not frames:
        continue

    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences
    generated_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    _, _, pred_answer = parse_response_by_mode(
        response=response,
        has_options=True,
        model_mode=MODEL_MODE,
    )
    pred_letter = str(pred_answer).strip().upper()
    gt_letter = str(sample.answer).strip().upper()
    is_correct = int(pred_letter == gt_letter)

    option_probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

    if len(outputs.scores) > 0 and pred_letter in {"A", "B", "C", "D"}:
        gen_token_ids = generated_ids[:, -len(outputs.scores):]
        gen_ids_list = [int(x) for x in gen_token_ids[0].tolist()]
        answer_ids = processor.tokenizer(pred_letter, add_special_tokens=False)["input_ids"]

        answer_pos = -1
        if answer_ids:
            m = len(answer_ids)
            for start in range(len(gen_ids_list) - m, -1, -1):
                if gen_ids_list[start:start + m] == answer_ids:
                    answer_pos = start
                    break

        if answer_pos >= 0:
            probs = torch.softmax(outputs.scores[answer_pos][0], dim=-1)
            for opt in ("A", "B", "C", "D"):
                candidates = {
                    opt,
                    opt.lower(),
                    f" {opt}",
                    f" {opt.lower()}",
                }
                token_ids: set[int] = set()
                for c in candidates:
                    ids = processor.tokenizer(c, add_special_tokens=False)["input_ids"]
                    if ids:
                        token_ids.add(int(ids[0]))
                p = 0.0
                for tid in token_ids:
                    if 0 <= tid < probs.shape[-1]:
                        p += float(probs[tid].item())
                option_probs[opt] = p

    pred_option_prob = float(option_probs.get(pred_letter, 0.0))
    gt_option_prob = float(option_probs.get(gt_letter, 0.0))
    max_option_prob = float(max(option_probs.values()))

    rows.append(
        {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "gt_answer": gt_letter,
            "pred_answer": pred_letter,
            "is_correct": is_correct,
            "pred_option_prob": pred_option_prob,
            "gt_option_prob": gt_option_prob,
            "max_option_prob": max_option_prob,
            "option_prob_A": float(option_probs["A"]),
            "option_prob_B": float(option_probs["B"]),
            "option_prob_C": float(option_probs["C"]),
            "option_prob_D": float(option_probs["D"]),
        }
    )

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("没有得到有效样本结果，请检查数据路径、模型与推理配置。")

df["is_correct_float"] = df["is_correct"].astype(float)
corr_pred_conf = float(df["pred_option_prob"].corr(df["is_correct_float"]))
corr_max_conf = float(df["max_option_prob"].corr(df["is_correct_float"]))
overall_acc = float(df["is_correct_float"].mean())

bins = np.linspace(0.0, 1.0, 11)
df["pred_conf_bin"] = pd.cut(df["pred_option_prob"], bins=bins, include_lowest=True)
bin_df = (
    df.groupby("pred_conf_bin", observed=False)
    .agg(
        n=("is_correct", "count"),
        accuracy=("is_correct_float", "mean"),
        pred_option_prob_mean=("pred_option_prob", "mean"),
        gt_option_prob_mean=("gt_option_prob", "mean"),
        max_option_prob_mean=("max_option_prob", "mean"),
    )
    .reset_index()
)

sample_csv = OUT_DIR / "sample_level_results.csv"
bin_csv = OUT_DIR / "confidence_bin_summary.csv"
summary_txt = OUT_DIR / "summary.txt"

df.drop(columns=["is_correct_float"]).to_csv(sample_csv, index=False, encoding="utf-8")
bin_df.to_csv(bin_csv, index=False, encoding="utf-8")

with summary_txt.open("w", encoding="utf-8") as f:
    f.write(f"num_samples={len(df)}\n")
    f.write(f"overall_accuracy={overall_acc:.6f}\n")
    f.write(f"corr(is_correct, pred_option_prob)={corr_pred_conf:.6f}\n")
    f.write(f"corr(is_correct, max_option_prob)={corr_max_conf:.6f}\n")

# 图1：散点图（加微小抖动方便观察）
jitter = rng.uniform(-0.03, 0.03, size=len(df))
plt.figure(figsize=(8, 5))
plt.scatter(
    df["pred_option_prob"].values,
    df["is_correct_float"].values + jitter,
    s=12,
    alpha=0.35,
)
plt.yticks([0, 1], ["Wrong", "Correct"])
plt.ylim(-0.15, 1.15)
plt.xlabel("Predicted Option Probability")
plt.ylabel("Correctness (jittered)")
plt.title("Correctness vs Predicted Option Probability")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_correctness_vs_pred_prob.png", dpi=160)
plt.close()

# 图2：置信度分桶准确率（reliability 风格）
plot_bin = bin_df.copy()
plot_bin["bin_mid"] = plot_bin["pred_conf_bin"].astype(str).str.extract(r"\((.*), (.*)\]").astype(float).mean(axis=1)
plt.figure(figsize=(8, 5))
plt.plot(plot_bin["bin_mid"], plot_bin["accuracy"], marker="o")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, alpha=0.5)
plt.xlabel("Predicted Option Probability (bin midpoint)")
plt.ylabel("Accuracy in Bin")
plt.title("Binned Accuracy vs Predicted Probability")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUT_DIR / "binned_accuracy_vs_pred_prob.png", dpi=160)
plt.close()

print("完成。输出文件：")
print(sample_csv)
print(bin_csv)
print(summary_txt)
print(OUT_DIR / "scatter_correctness_vs_pred_prob.png")
print(OUT_DIR / "binned_accuracy_vs_pred_prob.png")
