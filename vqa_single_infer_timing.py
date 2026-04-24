from __future__ import annotations

import argparse
import os
import time

import torch
from tqdm import tqdm

from data_loaders import get_data_loader, list_supported_datasets
from frame_samplers import sample_video_frames
from vqa_eval import build_user_text, calculate_mra, extract_answer
from vl_common import load_model_and_processor


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args():
    p = argparse.ArgumentParser(description="VQA 推理细粒度耗时测试（默认 10 样本）")
    p.add_argument("--dataset", type=str, default="egoschema", choices=list_supported_datasets())
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--dataset_name", type=str, default="lmms-lab/EgoSchema")
    p.add_argument("--dataset_config", type=str, default="Subset")
    p.add_argument("--no_dataset_config", action="store_true")
    p.add_argument("--video_dir", type=str, default="~/dataset/egoschema/videos/videos")
    p.add_argument("--start_index", type=int, default=0, help="从抽样后列表的哪个下标开始测")
    p.add_argument("--test_samples", type=int, default=10, help="实际测试多少个样本")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument(
        "--frame_sampling_method",
        type=str,
        default="uniform",
        choices=["uniform"],
    )
    p.add_argument("--focus_blip_model_name", type=str, default="Salesforce/blip-itm-base-coco")
    p.add_argument("--focus_blip_device", type=str, default=None)
    p.add_argument("--focus_blip_batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task_filter", type=str, default="mcq", choices=["all", "mcq", "numeric"])
    p.add_argument("--train_ratio", type=float, default=0.0)
    p.add_argument("--use_train_split", action="store_true")
    p.add_argument("--num_samples", type=str, default="all", help="先从数据集中抽样多少条")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Thinking")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    stage_order = [
        "sample_video_frames",
        "build_prompt",
        "apply_chat_template",
        "processor_build_inputs",
        "move_inputs_to_device",
        "model_prefill_forward",
        "model_prefill_argmax",
        "model_decode_forward",
        "model_decode_argmax_eos",
        "model_decode_concat_tokens",
        "decode_response",
        "extract_answer",
        "compute_metric",
    ]
    stage_sums = {k: 0.0 for k in stage_order}
    timings: list[tuple[str, float]] = []
    t_total_0 = time.perf_counter()

    t0 = time.perf_counter()
    video_dir = os.path.expanduser(args.video_dir)
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
    timings.append(("load_dataset", time.perf_counter() - t0))

    if not samples:
        raise RuntimeError("没有可用样本，请检查数据集参数")
    if args.start_index < 0 or args.start_index >= len(samples):
        raise IndexError(f"start_index={args.start_index} 越界，当前样本数={len(samples)}")

    t0 = time.perf_counter()
    model, processor = load_model_and_processor(
        os.path.expanduser(args.model_path),
        use_lora=args.use_lora,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
    )
    timings.append(("load_model_and_processor", time.perf_counter() - t0))
    device = model.device if isinstance(model.device, torch.device) else torch.device(model.device)

    end_index = min(len(samples), args.start_index + args.test_samples)
    target_samples = samples[args.start_index:end_index]
    processed = 0
    skipped_no_frames = 0
    mcq_total = 0
    mcq_correct = 0
    numeric_total = 0
    numeric_mra_sum = 0.0
    generated_tokens_total = 0
    last_response = ""
    last_pred_answer = ""
    last_sample_id = ""

    pbar = tqdm(target_samples, desc="Timing Progress", unit="sample")
    for i, sample in enumerate(pbar):
        random_seed = (args.seed + args.start_index + i) if args.frame_sampling_method == "random" else None

        t0 = time.perf_counter()
        frames = sample_video_frames(
            video_path=sample.video_path,
            num_frames=args.num_frames,
            method=args.frame_sampling_method,
            random_seed=random_seed,
            question=sample.question,
            options=sample.options,
            answer=str(sample.answer),
            focus_blip_model_name=args.focus_blip_model_name,
            focus_blip_device=args.focus_blip_device,
            focus_blip_batch_size=args.focus_blip_batch_size,
        )
        stage_sums["sample_video_frames"] += time.perf_counter() - t0
        if not frames:
            skipped_no_frames += 1
            pbar.set_postfix(processed=processed, skipped=skipped_no_frames)
            continue

        t0 = time.perf_counter()
        prompt = build_user_text(sample.question, sample.options)
        stage_sums["build_prompt"] += time.perf_counter() - t0

        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        t0 = time.perf_counter()
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        stage_sums["apply_chat_template"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt")
        stage_sums["processor_build_inputs"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        inputs = inputs.to(device)
        sync_if_cuda(device)
        stage_sums["move_inputs_to_device"] += time.perf_counter() - t0

        eos_ids: list[int] = []
        eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos_ids.append(int(eos_token_id))
        if not eos_ids:
            eos_ids = [151645]

        sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        sync_if_cuda(device)
        stage_sums["model_prefill_forward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        logits = out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        past_key_values = out.past_key_values
        generated_token_list = [next_token]
        sync_if_cuda(device)
        stage_sums["model_prefill_argmax"] += time.perf_counter() - t0

        decode_forward_time = 0.0
        decode_argmax_time = 0.0
        with torch.no_grad():
            for _ in range(args.max_new_tokens - 1):
                sync_if_cuda(device)
                t1 = time.perf_counter()
                out = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                sync_if_cuda(device)
                decode_forward_time += time.perf_counter() - t1

                t1 = time.perf_counter()
                logits = out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                past_key_values = out.past_key_values
                generated_token_list.append(next_token)
                decode_argmax_time += time.perf_counter() - t1
                if int(next_token[0, 0].item()) in eos_ids:
                    break
        stage_sums["model_decode_forward"] += decode_forward_time
        stage_sums["model_decode_argmax_eos"] += decode_argmax_time

        t0 = time.perf_counter()
        generated_ids_trimmed = torch.cat(generated_token_list, dim=1)
        sync_if_cuda(device)
        stage_sums["model_decode_concat_tokens"] += time.perf_counter() - t0
        generated_tokens_total += int(generated_ids_trimmed.shape[1])

        t0 = time.perf_counter()
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        stage_sums["decode_response"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_answer = extract_answer(response, has_options=bool(sample.options))
        stage_sums["extract_answer"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if sample.task_type == "mcq":
            mcq_total += 1
            if str(sample.answer).strip().upper() == str(pred_answer).strip().upper():
                mcq_correct += 1
        else:
            numeric_total += 1
            try:
                numeric_mra_sum += calculate_mra(float(pred_answer), float(sample.answer))
            except (TypeError, ValueError):
                numeric_mra_sum += 0.0
        stage_sums["compute_metric"] += time.perf_counter() - t0

        processed += 1
        last_response = response
        last_pred_answer = str(pred_answer)
        last_sample_id = sample.sample_id
        avg_sample_sec = stage_sums["sample_video_frames"] / processed if processed > 0 else 0.0
        avg_prefill_sec = (
            (stage_sums["model_prefill_forward"] + stage_sums["model_prefill_argmax"]) / processed
            if processed > 0
            else 0.0
        )
        avg_decode_sec = (
            (stage_sums["model_decode_forward"] + stage_sums["model_decode_argmax_eos"]) / processed
            if processed > 0
            else 0.0
        )
        avg_tokens = (generated_tokens_total / processed) if processed > 0 else 0.0
        pbar.set_postfix(
            processed=processed,
            skipped=skipped_no_frames,
            avg_sample=f"{avg_sample_sec:.3f}s",
            avg_prefill=f"{avg_prefill_sec:.3f}s",
            avg_decode=f"{avg_decode_sec:.3f}s",
            avg_tokens=f"{avg_tokens:.1f}",
        )

    total = time.perf_counter() - t_total_0
    measured_sum = sum(sec for _, sec in timings) + sum(stage_sums.values())

    print("=" * 88)
    print("VQA Inference Timing over Multiple Samples (non-overlap)")
    print(f"dataset={args.dataset}/{args.dataset_split} | requested={len(target_samples)} | processed={processed} | skipped_no_frames={skipped_no_frames}")
    print("-" * 88)
    print(f"{'stage':40s} {'seconds':>12s} {'percent_of_total':>18s}")
    for stage, sec in timings:
        pct = (sec / total * 100.0) if total > 0 else 0.0
        print(f"{stage:40s} {sec:12.6f} {pct:18.2f}%")
    for stage in stage_order:
        sec = stage_sums[stage]
        pct = (sec / total * 100.0) if total > 0 else 0.0
        avg = (sec / processed) if processed > 0 else 0.0
        print(f"{stage + ' (sum)':40s} {sec:12.6f} {pct:18.2f}%")
        print(f"{stage + ' (avg/sample)':40s} {avg:12.6f} {(avg / total * 100.0):18.2f}%")
    print("-" * 88)
    print(f"{'measured_sum':40s} {measured_sum:12.6f} {(measured_sum / total * 100.0):18.2f}%")
    print(f"{'total_end_to_end':40s} {total:12.6f} {100.00:18.2f}%")
    print("=" * 88)

    mcq_acc = (mcq_correct / mcq_total) if mcq_total > 0 else 0.0
    numeric_mra = (numeric_mra_sum / numeric_total) if numeric_total > 0 else 0.0
    print("\nMetrics")
    print(f"mcq_acc: {mcq_acc:.4f} ({mcq_correct}/{mcq_total})")
    print(f"numeric_mra: {numeric_mra:.4f} ({numeric_total} samples)")
    if processed > 0:
        print(f"avg_generated_tokens: {generated_tokens_total / processed:.2f}")
    if processed > 0:
        print(f"\nlast_sample_id: {last_sample_id}")
        print(f"last_pred_answer: {last_pred_answer}")
        print(f"last_raw_response:\n{last_response}")


if __name__ == "__main__":
    main()
