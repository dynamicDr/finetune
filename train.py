"""
test_vsibench.py - 使用 VSI-Bench 数据集评估 Qwen2.5-VL-3B 模型

用法:
    python test_vsibench.py --video_dir /path/to/videos
    python test_vsibench.py --video_dir /path/to/videos --task_filter mcq  # 只做多选题
    python test_vsibench.py --video_dir /path/to/videos --task_filter numeric  # 只做数值题
    python test_vsibench.py --video_dir /path/to/videos --task_filter all  # 都做
"""

import argparse
import os
import re
import time
import random

import cv2
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_model_and_processor(model_path: str, use_lora: bool = False):
    """加载模型和处理器"""
    print(f"正在加载模型: {model_path}")

    if use_lora:
        from peft import PeftModel

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    model.eval()
    print("模型加载完成!")
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
    """从模型回复中提取答案。
    对于 Thinking 模型（<think>...</think>），答案在 </think> 之后，需从该部分提取，否则会误取推理过程中的中间值。
    """
    response = response.strip()

    # Thinking 模型：答案在 </think> 之后（或 <answer>...</answer> 内）
    answer_portion = response
    if "</think>" in response:
        answer_portion = response.split("</think>", 1)[-1].strip()
    # 若有 <answer>...</answer>，优先从中提取
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", answer_portion, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_portion = answer_match.group(1).strip()

    if has_options:
        match = re.search(r"\b([A-D])\b", answer_portion.upper())
        if match:
            return match.group(1)

    numbers = re.findall(r"[\d.]+", answer_portion)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass

    return response


def calculate_mra(pred: float, gt: float) -> float:
    """计算 Mean Relative Accuracy (MRA)"""
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0, 1 - abs(pred - gt) / abs(gt))


def should_include_sample(sample, task_filter: str) -> bool:
    """判断样本是否应该被包含在评估中"""
    if task_filter == "all":
        return True

    has_options = sample.get("options", None) is not None

    if task_filter == "mcq":
        return has_options
    elif task_filter == "numeric":
        return not has_options

    return True


def evaluate_vsibench(
    model,
    processor,
    video_dir: str,
    num_samples: int = 10,
    num_frames: int = 8,
    seed: int = 42,
    task_filter: str = "all"
):
    """在 VSI-Bench 数据集上评估模型

    Args:
        task_filter: 'all' (所有题目), 'mcq' (只做多选题), 'numeric' (只做数值题)
    """
    dataset = load_dataset("nyu-visionx/VSI-Bench", split="test")

    # 先筛选符合条件的样本
    filtered_indices = [
        i for i in range(len(dataset))
        if should_include_sample(dataset[i], task_filter)
    ]

    print(f"数据集总大小: {len(dataset)}")
    print(f"任务筛选: {task_filter}")
    print(f"符合条件的样本数: {len(filtered_indices)}")
    print(f"每个视频抽取帧数: {num_frames}")

    # 从筛选后的样本中随机抽取
    random.seed(seed)
    sample_indices = random.sample(
        filtered_indices,
        min(num_samples, len(filtered_indices))
    )
    samples = dataset.select(sample_indices)

    print(f"随机抽取样本数: {len(samples)}, 随机种子: {seed}")

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

    for i, sample in enumerate(tqdm(samples, desc="评估进度")):
        try:
            question = sample.get("question", "")
            answer = sample.get("ground_truth", "")
            scene_name = sample.get("scene_name", "") + ".mp4"
            options = sample.get("options", None)
            task_type = sample.get("task_type", sample.get("question_type", "unknown"))

            video_path = os.path.join(video_dir, scene_name)
            if not os.path.exists(video_path):
                print(f"视频不存在: {video_path}")
                continue

            frames = extract_video_frames(video_path, num_frames=num_frames)
            if not frames:
                print(f"无法提取帧: {video_path}")
                continue

            # 构建提示
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

            # 评估结果
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

        except Exception as e:
            print(f"样本 {i} 出错: {e}")
            continue

    # 输出结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)

    print(f"题目类型统计:")
    print(f"  多选题数量: {results['mcq_count']}")
    print(f"  数值题数量: {results['numeric_count']}")
    print()

    if results["total"] > 0:
        accuracy = results["correct"] / results["total"] * 100
        print(f"多选题准确率: {results['correct']}/{results['total']} = {accuracy:.2f}%")

    if results["mra_count"] > 0:
        avg_mra = results["mra_sum"] / results["mra_count"] * 100
        print(f"数值题平均 MRA: {avg_mra:.2f}%")

    # 推理时间统计
    if results["inference_times"]:
        avg_time = sum(results["inference_times"]) / len(results["inference_times"])
        min_time = min(results["inference_times"])
        max_time = max(results["inference_times"])
        print(f"\n推理时间统计:")
        print(f"  平均推理时间: {avg_time:.3f} 秒")
        print(f"  最小推理时间: {min_time:.3f} 秒")
        print(f"  最大推理时间: {max_time:.3f} 秒")
        print(f"  总推理时间: {sum(results['inference_times']):.3f} 秒")

    # 按任务类型统计
    task_results = {}
    for detail in results["details"]:
        task = detail["task_type"]
        if task not in task_results:
            task_results[task] = {"correct": 0, "total": 0}
        task_results[task]["total"] += 1
        if detail["is_correct"]:
            task_results[task]["correct"] += 1

    print("\n按任务类型:")
    for task, stats in task_results.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {task}: {stats['correct']}/{stats['total']} = {acc:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="使用 VSI-Bench 评估 Qwen2.5-VL-3B 模型")

    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="模型路径或 HuggingFace 模型 ID")

    parser.add_argument("--use_lora", action="store_true", help="是否加载 LoRA 权重")

    parser.add_argument("--num_samples", type=int, default=10, help="评估的样本数量")

    parser.add_argument("--num_frames", type=int, default=8, help="每个视频提取的帧数")

    parser.add_argument("--video_dir", type=str, default="~/dataset/vsi_bench", help="视频文件目录路径")

    parser.add_argument("--seed", type=int, default=42, help="随机抽样的种子")

    parser.add_argument("--task_filter", type=str, default="all", choices=["all", "mcq", "numeric"], help="题目类型筛选: all (所有题目), mcq (只做多选题), numeric (只做数值题)")

    args = parser.parse_args()
    video_dir = os.path.expanduser(args.video_dir)

    model, processor = load_model_and_processor(args.model_path, args.use_lora)
    evaluate_vsibench(
        model,
        processor,
        video_dir=video_dir,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        seed=args.seed,
        task_filter=args.task_filter,
    )

    print("\n评估完成!")


if __name__ == "__main__":
    main()