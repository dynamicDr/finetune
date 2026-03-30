"""
test_vsibench.py - 使用 VSI-Bench 数据集评估 Qwen2.5-VL-3B 模型

支持 --train_ratio 和 --seed 参数，与 train_vsibench.py 配合使用，
确保训练和测试使用相同的数据划分。
"""
import argparse
import os
import re
import time
import random
import csv
from datetime import datetime
from pathlib import Path

import cv2
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration


def load_model_and_processor(
    model_path: str,
    use_lora: bool = False,
    base_model: str | None = None,
    merge_lora: bool = False,
):
    """加载模型和处理器"""
    model_path = os.path.expanduser(model_path)
    if use_lora and base_model:
        from peft import PeftModel

        base_id = os.path.expanduser(base_model)
        if "Qwen3-VL" in base_id:
            base = AutoModelForImageTextToText.from_pretrained(
                base_id,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, model_path)
            if merge_lora:
                model = model.merge_and_unload()
            proc_src = (
                model_path
                if os.path.isfile(os.path.join(model_path, "preprocessor_config.json"))
                or os.path.isfile(os.path.join(model_path, "tokenizer_config.json"))
                else base_id
            )
            processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)
        else:
            base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_id,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, model_path)
            if merge_lora:
                model = model.merge_and_unload()
            processor = AutoProcessor.from_pretrained(base_id)
        model.eval()
        return model, processor

    if "Qwen3-VL" in model_path:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        return model, processor

    if use_lora:
        from peft import PeftModel

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    model.eval()
    return model, processor


def extract_video_frames(video_path: str, num_frames: int = 8) -> list:
    """从视频中均匀提取指定数量的帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def generate_response(model, processor, frames: list, question: str) -> tuple[str, float]:
    """生成模型回复，返回回复和推理时间"""
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    inference_time = time.time() - start_time

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response, inference_time


def extract_answer(response: str, has_options: bool = False):
    """从模型回复中提取答案。"""
    response = response.strip()

    answer_portion = response
    if "</think>" in response:
        answer_portion = response.split("</think>", 1)[-1].strip()
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", answer_portion, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_portion = answer_match.group(1).strip()

    if has_options:
        match = re.search(r"\b([A-D])\b", answer_portion.upper())
        if match:
            return match.group(1)

    numbers = re.findall(r"[\d.]+", answer_portion)
    return float(numbers[0]) if numbers else response


def calculate_mra(pred: float, gt: float) -> float:
    """计算 Mean Relative Accuracy (MRA)"""
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0, 1 - abs(pred - gt) / abs(gt))


def should_include_sample(sample, task_filter: str) -> bool:
    """判断样本是否应该被包含在评估中"""
    return task_filter == "all" or (sample.get("options") is not None) == (task_filter == "mcq")


def split_indices(
    indices: list[int],
    seed: int,
    train_ratio: float,
    use_train_split: bool,
) -> list[int]:
    """
    根据 seed 和 train_ratio 划分索引，返回训练集或测试集部分。
    必须保证 train 和 test 脚本使用相同的 seed 和 train_ratio。
    """
    indices_copy = indices.copy()
    random.seed(seed)
    random.shuffle(indices_copy)
    
    split_point = int(len(indices_copy) * train_ratio)
    if use_train_split:
        return indices_copy[:split_point]
    else:
        return indices_copy[split_point:]


def _compute_accuracy_from_results(results: dict, task_filter: str) -> tuple[float, float]:
    """根据当前 results 字典计算已评估部分的准确率和平均推理时间。"""
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
        sum(results["inference_times"]) / len(results["inference_times"])
        if results["inference_times"]
        else 0.0
    )
    return avg_accuracy, avg_inference_time


def _write_progress_file(
    progress_file: str,
    n_done: int,
    n_total: int,
    partial_accuracy: float,
    elapsed_sec: float,
):
    """覆盖写入进度文件，便于超时或断点后查看已跑多少、当前准确率。"""
    with open(progress_file, "w", encoding="utf-8") as f:
        f.write(f"n_done={n_done}\n")
        f.write(f"n_total={n_total}\n")
        f.write(f"partial_accuracy={partial_accuracy:.2f}\n")
        f.write(f"elapsed_sec={elapsed_sec:.1f}\n")


def log_to_csv(
    log_file: str,
    seed: int,
    task_filter: str,
    num_samples: int,
    num_frames: int,
    avg_accuracy: float,
    avg_inference_time: float,
    model_name: str,
    lora_path: str,
    train_ratio: float,
    use_train_split: bool,
):
    """将评估结果记录到 CSV 文件并打印追加的行"""
    Path(log_file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.exists(log_file)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    split_name = "train" if use_train_split else "test"
    row_data = [
        timestamp,
        seed,
        task_filter,
        num_samples,
        num_frames,
        f"{avg_accuracy:.2f}",
        f"{avg_inference_time:.3f}",
        model_name,
        lora_path,
        train_ratio,
        split_name,
    ]

    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                'timestamp',
                'seed',
                'task_filter',
                'num_samples',
                'num_frames',
                'avg_accuracy',
                'avg_inference_time',
                'model_name',
                'lora_path',
                'train_ratio',
                'eval_split',
            ])

        writer.writerow(row_data)

    print(f"\n记录已添加到 {log_file}:")
    print(f"{','.join(map(str, row_data))}")


def evaluate_vsibench(
    model,
    processor,
    video_dir: str,
    num_samples = 10,
    num_frames: int = 8,
    seed: int = 42,
    task_filter: str = "all",
    log_file: str = "vsibench_evaluation_log.csv",
    model_name: str = "",
    lora_path: str = "",
    progress_interval: int = 10,
    train_ratio: float = 0.8,
    use_train_split: bool = False,
):
    """在 VSI-Bench 数据集上评估模型

    Args:
        num_samples: 评估的样本数量,可以是整数或 "all" (测试全部样本)
        task_filter: 'all' (所有题目), 'mcq' (只做多选题), 'numeric' (只做数值题)
        log_file: CSV 日志文件路径
        progress_interval: 每评估多少个样本写一次进度文件（0 表示不写）
        train_ratio: 训练集比例（与 train_vsibench.py 保持一致）
        use_train_split: True 评估训练集，False 评估测试集（默认）
    """
    dataset = load_dataset("nyu-visionx/VSI-Bench", split="test")

    # 先筛选符合条件的样本
    filtered_indices = [
        i for i in range(len(dataset))
        if should_include_sample(dataset[i], task_filter)
    ]

    # 使用与训练相同的逻辑划分数据
    filtered_indices = split_indices(filtered_indices, seed, train_ratio, use_train_split)
    
    split_name = "训练集" if use_train_split else "测试集"
    print(f"评估 {split_name}，共 {len(filtered_indices)} 个样本 (train_ratio={train_ratio}, seed={seed})")

    # 从划分后的样本中随机抽取或使用全部
    if num_samples == "all":
        sample_indices = filtered_indices
        actual_num_samples = len(filtered_indices)
    else:
        # 使用不同的 seed 进行抽样，避免与划分冲突
        random.seed(seed + 1000)
        sample_indices = random.sample(
            filtered_indices,
            min(num_samples, len(filtered_indices))
        )
        actual_num_samples = len(sample_indices)

    samples = dataset.select(sample_indices)

    results = {
        "correct": 0,
        "total": 0,
        "mra_sum": 0.0,
        "mra_count": 0,
        "details": [],
        "inference_times": [],
        "mcq_count": 0,
        "numeric_count": 0,
    }

    start_time = time.time()
    n_total = len(samples)

    pbar = tqdm(samples, desc="评估进度")

    for i, sample in enumerate(pbar):
        try:
            question = sample.get("question", "")
            answer = sample.get("ground_truth", "")
            scene_name = sample.get("scene_name", "") + ".mp4"
            options = sample.get("options", None)
            task_type = sample.get("task_type", sample.get("question_type", "unknown"))

            video_path = os.path.join(video_dir, scene_name)
            if not os.path.exists(video_path):
                continue

            frames = extract_video_frames(video_path, num_frames=num_frames)
            if not frames:
                continue

            if options:
                prompt = f"{question}\n\nOptions:\n" + "\n".join(options)
                prompt += "\n\nPlease answer with the option letter directly (A, B, C, or D)."
                results["mcq_count"] += 1
            else:
                prompt = f"{question}\n\nPlease provide the numerical answer directly."
                results["numeric_count"] += 1

            response, inference_time = generate_response(model, processor, frames, prompt)
            results["inference_times"].append(inference_time)

            pred_answer = extract_answer(response, has_options=bool(options))

            is_correct = False
            mra_score = None

            if options:
                is_correct = str(answer).strip().upper() == str(pred_answer).strip().upper()
                results["total"] += 1
                if is_correct:
                    results["correct"] += 1
            else:
                try:
                    pred_num = float(pred_answer) if pred_answer else 0
                    gt_num = float(answer)
                    mra_score = calculate_mra(pred_num, gt_num)
                    results["mra_sum"] += mra_score
                    results["mra_count"] += 1
                    is_correct = mra_score > 0.5
                except (ValueError, TypeError):
                    pass

            results["details"].append(
                {
                    "index": sample_indices[i],
                    "task_type": task_type,
                    "question_type": "MCQ" if options else "Numeric",
                    "question": question,
                    "ground_truth": answer,
                    "prediction": response,
                    "is_correct": is_correct,
                    "mra_score": mra_score,
                    "inference_time": inference_time,
                }
            )

            if len(results["details"]) > 0:
                partial_acc, partial_time = _compute_accuracy_from_results(results, task_filter)
                pbar.set_postfix({
                    'Acc': f'{partial_acc:.2f}%',
                    'AvgTime': f'{partial_time:.2f}s'
                })

            n_done = len(results["details"])
            # 不输出 progress.txt：评测仅写 CSV

        except Exception as e:
            continue

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

    avg_inference_time = sum(results["inference_times"]) / len(results["inference_times"]) if results["inference_times"] else 0.0

    log_to_csv(
        log_file,
        seed,
        task_filter,
        actual_num_samples,
        num_frames,
        avg_accuracy,
        avg_inference_time,
        model_name,
        lora_path,
        train_ratio,
        use_train_split,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="使用 VSI-Bench 评估 Qwen2.5-VL-3B 模型")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="模型路径或 HuggingFace 模型 ID；与 --use_lora 连用时为 adapter 目录（train_vsibench 的 --output_dir）")
    parser.add_argument("--use_lora", action="store_true", help="是否加载 LoRA 权重")
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="LoRA 基座（HF id 或路径）。Qwen3-VL 微调评测需指定；可由 run.py 从 model.name 传入",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="加载 LoRA 后 merge_and_unload（更占显存峰值、启动更慢；默认保持 adapter 形态）",
    )
    parser.add_argument("--num_samples", type=str, default="10", help="评估的样本数量,可以是数字或 'all' (测试全部样本)")
    parser.add_argument("--num_frames", type=int, default=8, help="每个视频提取的帧数")
    parser.add_argument("--video_dir", type=str, default="~/dataset/vsi_bench", help="视频文件目录路径")
    parser.add_argument("--seed", type=int, default=42, help="随机抽样的种子（需与训练时一致）")
    parser.add_argument("--task_filter", type=str, default="all", choices=["all", "mcq", "numeric"], help="题目类型筛选: all (所有题目), mcq (只做多选题), numeric (只做数值题)")
    parser.add_argument("--log_file", type=str, default="vsibench_evaluation_log.csv", help="CSV 日志文件路径")
    
    # 数据划分参数
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例（需与训练时一致，默认0.8）")
    parser.add_argument("--use_train_split", action="store_true", help="评估训练集（默认评估测试集）")
    
    args = parser.parse_args()
    video_dir = os.path.expanduser(args.video_dir)

    # 所有 CSV 统一输出到仓库根目录的 eval_csv/ 下
    eval_csv_dir = Path(__file__).resolve().parent / "eval_csv"
    eval_csv_dir.mkdir(parents=True, exist_ok=True)
    args.log_file = str(eval_csv_dir / Path(args.log_file).name)

    if args.num_samples.lower() == "all":
        num_samples = "all"
    else:
        num_samples = int(args.num_samples)

    resolved_model_path = os.path.expanduser(args.model_path)
    if args.use_lora:
        if args.base_model:
            model_name = os.path.expanduser(args.base_model)
        else:
            model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        lora_path = resolved_model_path
    else:
        path_str = resolved_model_path.rstrip("/")
        last_dashdash = path_str.rfind("--")
        if last_dashdash != -1:
            start = last_dashdash + 2
            slash_pos = path_str.find("/", start)
            if slash_pos == -1:
                model_name = path_str[start:]
            else:
                model_name = path_str[start:slash_pos]
        else:
            model_name = os.path.basename(path_str)
        lora_path = ""

    model, processor = load_model_and_processor(
        resolved_model_path,
        use_lora=args.use_lora,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
    )
    evaluate_vsibench(
        model,
        processor,
        video_dir=video_dir,
        num_samples=num_samples,
        num_frames=args.num_frames,
        seed=args.seed,
        task_filter=args.task_filter,
        log_file=args.log_file,
        model_name=model_name,
        lora_path=lora_path,
        train_ratio=args.train_ratio,
        use_train_split=args.use_train_split,
    )


if __name__ == "__main__":
    main()