#!/usr/bin/env python3
"""
在 VSI-Bench 上统计 Qwen3-VL 视觉编码与语言侧耗时（对齐 test_vsibench 数据与构图）。

指标（近似）：
- vision_ms：get_image_features
- prefill_ms：单次 model(**inputs)
- lm_prefill_approx_ms：prefill - vision
- generate_ms：generate(max_new_tokens=K)
- decode_approx_ms：generate - prefill

默认 sweep：4B/8B × Thinking/Instruct，帧数 4/8/16，每配置最多 50 个有视频的样本。

用法：
  python benchmark_vl_vision_vs_lm_time.py
  python benchmark_vl_vision_vs_lm_time.py --legacy --num_samples 5 --num_frames 4 --only 4b
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from data_loaders.base import load_dataset
from frame_samplers import sample_video_frames
from visbench.train_vsibench import build_user_text
from visbench.eval_vsibench import should_include_sample
from vl_common import split_indices


HUB_ROOT = Path.home() / ".cache/huggingface/hub"

MODEL_PRESETS: list[tuple[str, str, str]] = [
    ("4B-T", "Qwen3-VL-4B-Thinking", "models--Qwen--Qwen3-VL-4B-Thinking"),
    ("8B-T", "Qwen3-VL-8B-Thinking", "models--Qwen--Qwen3-VL-8B-Thinking"),
    ("4B-I", "Qwen3-VL-4B-Instruct", "models--Qwen--Qwen3-VL-4B-Instruct"),
    ("8B-I", "Qwen3-VL-8B-Instruct", "models--Qwen--Qwen3-VL-8B-Instruct"),
]


def resolve_hub_snapshot(models_hub_dir: Path) -> Path:
    snapshots = models_hub_dir / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(f"找不到 snapshots: {snapshots}")
    cand = next(snapshots.iterdir(), None)
    if cand is None:
        raise FileNotFoundError(f"snapshots 为空: {snapshots}")
    return cand


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_inputs_vsi(processor, frames: list, prompt: str, device: torch.device):
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt")
    return inputs.to(device)


def measure_once(model, inputs, max_new_tokens: int) -> tuple[float, float, float]:
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    pv_typed = pixel_values.type(model.visual.dtype)

    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.get_image_features(pv_typed, image_grid_thw)
    cuda_sync()
    t_vision = time.perf_counter() - t0

    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
    cuda_sync()
    t_prefill = time.perf_counter() - t0

    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    cuda_sync()
    t_gen = time.perf_counter() - t0

    return t_vision, t_prefill, t_gen


def pick_vsi_rows_with_video(
    video_dir: str,
    num_samples: int,
    seed: int,
    train_ratio: float,
    task_filter: str,
) -> list[dict]:
    dataset = load_dataset("nyu-visionx/VSI-Bench", split="test")
    filtered = [i for i in range(len(dataset)) if should_include_sample(dataset[i], task_filter)]
    test_indices = split_indices(filtered, seed, train_ratio, use_train_split=False)
    random.seed(seed + 1000)
    order = list(test_indices)
    random.shuffle(order)
    rows: list[dict] = []
    for idx in order:
        if len(rows) >= num_samples:
            break
        row = dataset[idx]
        scene = row.get("scene_name", "")
        video_path = os.path.join(video_dir, f"{scene}.mp4")
        if not os.path.isfile(video_path):
            continue
        rows.append(row)
    return rows


def collect_vsi_samples(
    video_dir: str,
    num_samples: int,
    num_frames: int,
    seed: int,
    train_ratio: float,
    task_filter: str,
):
    rows = pick_vsi_rows_with_video(video_dir, num_samples, seed, train_ratio, task_filter)
    samples: list[tuple[dict, list]] = []
    for row in rows:
        scene = row.get("scene_name", "")
        video_path = os.path.join(video_dir, f"{scene}.mp4")
        frames = sample_video_frames(video_path=video_path, num_frames=num_frames, method="uniform")
        if not frames:
            continue
        prompt = build_user_text(row.get("question", ""), row.get("options", None))
        samples.append((row, frames))
    return samples


def ms_stats(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def benchmark_one_config(
    model,
    processor,
    device: torch.device,
    rows: list[dict],
    video_dir: str,
    num_frames: int,
    warmup: int,
    max_new_tokens: int,
    desc: str,
) -> dict:
    samples_meta: list[tuple[dict, list]] = []
    for row in rows:
        scene = row.get("scene_name", "")
        video_path = os.path.join(video_dir, f"{scene}.mp4")
        frames = sample_video_frames(video_path=video_path, num_frames=num_frames, method="uniform")
        if not frames:
            continue
        samples_meta.append((row, frames))

    if not samples_meta:
        return {
            "n": 0,
            "vision_mean": 0.0,
            "vision_std": 0.0,
            "prefill_mean": 0.0,
            "prefill_std": 0.0,
            "lm_prefill_approx_mean": 0.0,
            "generate_mean": 0.0,
            "generate_std": 0.0,
            "decode_approx_mean": 0.0,
            "vision_over_generate_pct": 0.0,
            "vision_over_prefill_pct": 0.0,
        }

    _, first_frames = samples_meta[0]
    first_prompt = build_user_text(
        samples_meta[0][0].get("question", ""),
        samples_meta[0][0].get("options", None),
    )
    warm_inputs = build_inputs_vsi(processor, first_frames, first_prompt, device)
    for _ in range(warmup):
        measure_once(model, warm_inputs, max_new_tokens)

    tv_list, tp_list, tg_list = [], [], []
    for row, frames in tqdm(samples_meta, desc=desc):
        prompt = build_user_text(row.get("question", ""), row.get("options", None))
        inputs = build_inputs_vsi(processor, frames, prompt, device)
        tv, tp, tg = measure_once(model, inputs, max_new_tokens)
        tv_list.append(tv * 1000.0)
        tp_list.append(tp * 1000.0)
        tg_list.append(tg * 1000.0)

    v_m, v_s = ms_stats(tv_list)
    p_m, p_s = ms_stats(tp_list)
    g_m, g_s = ms_stats(tg_list)
    lm_approx = p_m - v_m
    dec_approx = max(g_m - p_m, 0.0)
    vg = (v_m / g_m * 100.0) if g_m > 0 else 0.0
    vp = (v_m / p_m * 100.0) if p_m > 0 else 0.0

    return {
        "n": len(tv_list),
        "vision_mean": v_m,
        "vision_std": v_s,
        "prefill_mean": p_m,
        "prefill_std": p_s,
        "lm_prefill_approx_mean": lm_approx,
        "generate_mean": g_m,
        "generate_std": g_s,
        "decode_approx_mean": dec_approx,
        "vision_over_generate_pct": vg,
        "vision_over_prefill_pct": vp,
    }


def parse_models_arg(s: str) -> list[tuple[str, str, Path]]:
    want = {x.strip().upper() for x in s.split(",") if x.strip()}
    if "ALL" in want or not want:
        chosen = MODEL_PRESETS
    else:
        chosen = [m for m in MODEL_PRESETS if m[0].upper() in want]
        if not chosen:
            raise ValueError(
                f"--models 无匹配: {s}；可选 ALL 或 {','.join(m[0] for m in MODEL_PRESETS)}"
            )
    out = []
    for short, full, hub_name in chosen:
        hub_parent = HUB_ROOT / hub_name
        if not hub_parent.is_dir():
            raise FileNotFoundError(f"未找到模型目录: {hub_parent}")
        out.append((short, full, hub_parent))
    return out


def parse_frame_counts(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy", action="store_true", help="仅 4B/8B Thinking，单 --num_frames，默认 5 样本")
    p.add_argument(
        "--model_4b",
        type=str,
        default=str(HUB_ROOT / "models--Qwen--Qwen3-VL-4B-Thinking"),
    )
    p.add_argument(
        "--model_8b",
        type=str,
        default=str(HUB_ROOT / "models--Qwen--Qwen3-VL-8B-Thinking"),
    )
    p.add_argument("--models", type=str, default="ALL", help="ALL 或 4B-T,8B-T,4B-I,8B-I")
    p.add_argument("--frame_counts", type=str, default="4,8,16")
    p.add_argument("--video_dir", type=str, default=os.path.expanduser("~/dataset/vsi_bench"))
    p.add_argument("--num_frames", type=int, default=4, help="仅 --legacy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--task_filter", type=str, default="mcq", choices=["all", "mcq", "numeric"])
    p.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="默认 sweep=50、legacy=5",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--only", type=str, default="", choices=["", "4b", "8b"], help="仅 --legacy")
    p.add_argument("--output_csv", type=str, default="")
    p.add_argument("--no_csv", action="store_true")
    args = p.parse_args()

    video_dir = os.path.expanduser(args.video_dir)
    if not torch.cuda.is_available():
        print("未检测到 CUDA，将使用 CPU（极慢）。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    num_samples_sweep = args.num_samples if args.num_samples is not None else 10

    if args.legacy:
        num_samples_legacy = args.num_samples if args.num_samples is not None else 5
        _main_legacy(args, video_dir, device, dtype, num_samples_legacy)
        return

    frame_counts = parse_frame_counts(args.frame_counts)
    model_entries = parse_models_arg(args.models)

    rows = pick_vsi_rows_with_video(
        video_dir=video_dir,
        num_samples=num_samples_sweep,
        seed=args.seed,
        train_ratio=args.train_ratio,
        task_filter=args.task_filter,
    )
    if not rows:
        raise RuntimeError(
            f"没有可用样本：确认 video_dir={video_dir} 下有 scene_name 对应 mp4。"
        )

    print(f"device={device}, dtype={dtype}")
    print(f"video_dir={video_dir}, 固定样本行数={len(rows)}, frame_counts={frame_counts}")
    print(f"models={[m[1] for m in model_entries]}, task_filter={args.task_filter}, seed={args.seed}")
    print()

    csv_rows: list[dict] = []
    for short, full_name, hub_parent in model_entries:
        snap = resolve_hub_snapshot(hub_parent)
        print("=" * 60)
        print(f"{short} | {full_name}")
        print("snapshot:", snap)

        model = AutoModelForImageTextToText.from_pretrained(
            str(snap),
            dtype=dtype,
            device_map=None,
            trust_remote_code=True,
        )
        model.eval()
        model.to(device)

        if not hasattr(model, "get_image_features"):
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError("模型需支持 get_image_features（Qwen3-VL）。")

        processor = AutoProcessor.from_pretrained(str(snap), trust_remote_code=True)

        for nf in frame_counts:
            stats = benchmark_one_config(
                model=model,
                processor=processor,
                device=device,
                rows=rows,
                video_dir=video_dir,
                num_frames=nf,
                warmup=args.warmup,
                max_new_tokens=args.max_new_tokens,
                desc=f"{short} {nf}帧",
            )
            n = stats["n"]
            print(f"  --- num_frames={nf} 有效 n={n} ---")
            print(f"  vision_mean            : {stats['vision_mean']:8.2f} ms  (±{stats['vision_std']:.2f})")
            print(f"  prefill_mean           : {stats['prefill_mean']:8.2f} ms  (±{stats['prefill_std']:.2f})")
            print(f"  lm_prefill_approx_mean : {stats['lm_prefill_approx_mean']:8.2f} ms")
            print(f"  generate_mean          : {stats['generate_mean']:8.2f} ms  (±{stats['generate_std']:.2f})")
            print(f"  decode_approx_mean     : {stats['decode_approx_mean']:8.2f} ms")
            print(f"  vision / generate      : {stats['vision_over_generate_pct']:.1f}%")
            print(f"  vision / prefill       : {stats['vision_over_prefill_pct']:.1f}%")
            print()

            csv_rows.append(
                {
                    "short": short,
                    "model": full_name,
                    "num_frames": nf,
                    "n_samples": n,
                    "vision_ms_mean": round(stats["vision_mean"], 3),
                    "vision_ms_std": round(stats["vision_std"], 3),
                    "prefill_ms_mean": round(stats["prefill_mean"], 3),
                    "prefill_ms_std": round(stats["prefill_std"], 3),
                    "lm_prefill_approx_ms_mean": round(stats["lm_prefill_approx_mean"], 3),
                    "generate_ms_mean": round(stats["generate_mean"], 3),
                    "generate_ms_std": round(stats["generate_std"], 3),
                    "decode_approx_ms_mean": round(stats["decode_approx_mean"], 3),
                    "vision_over_generate_pct": round(stats["vision_over_generate_pct"], 2),
                    "vision_over_prefill_pct": round(stats["vision_over_prefill_pct"], 2),
                }
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("=" * 60)
    print("汇总（v/g %, v/p %）")
    hdr = f"{'model':<10} {'frames':>6} {'n':>4} {'vis_ms':>8} {'pref_ms':>8} {'gen_ms':>8} {'v/g%':>6} {'v/p%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in csv_rows:
        print(
            f"{r['short']:<10} {r['num_frames']:>6} {r['n_samples']:>4} "
            f"{r['vision_ms_mean']:>8.1f} {r['prefill_ms_mean']:>8.1f} {r['generate_ms_mean']:>8.1f} "
            f"{r['vision_over_generate_pct']:>6.1f} {r['vision_over_prefill_pct']:>6.1f}"
        )

    if not args.no_csv and csv_rows:
        out = args.output_csv.strip()
        if not out:
            os.makedirs("eval_csv", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join("eval_csv", f"vl_vision_lm_benchmark_{ts}.csv")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print()
        print(f"已写 CSV: {out}")


def _main_legacy(args, video_dir: str, device: torch.device, dtype, num_samples: int):
    samples_meta = collect_vsi_samples(
        video_dir=video_dir,
        num_samples=num_samples,
        num_frames=args.num_frames,
        seed=args.seed,
        train_ratio=args.train_ratio,
        task_filter=args.task_filter,
    )
    if not samples_meta:
        raise RuntimeError(
            f"没有可用样本：确认 video_dir={video_dir} 下有对应 mp4。"
        )

    specs: list[tuple[str, Path]] = []
    hub_4b = Path(os.path.expanduser(args.model_4b))
    hub_8b = Path(os.path.expanduser(args.model_8b))
    if args.only in ("", "4b"):
        specs.append(("Qwen3-VL-4B-Thinking", resolve_hub_snapshot(hub_4b)))
    if args.only in ("", "8b"):
        specs.append(("Qwen3-VL-8B-Thinking", resolve_hub_snapshot(hub_8b)))

    print(f"device={device}, dtype={dtype} [legacy]")
    print(f"video_dir={video_dir}, num_frames={args.num_frames}, 样本数={len(samples_meta)}")
    print()

    for name, snap in specs:
        print("=" * 60)
        print(name)
        print("snapshot:", snap)

        model = AutoModelForImageTextToText.from_pretrained(
            str(snap),
            dtype=dtype,
            device_map=None,
            trust_remote_code=True,
        )
        model.eval()
        model.to(device)

        if not hasattr(model, "get_image_features"):
            raise RuntimeError("模型需支持 get_image_features（Qwen3-VL）。")

        processor = AutoProcessor.from_pretrained(str(snap), trust_remote_code=True)

        _, first_frames = samples_meta[0]
        first_prompt = build_user_text(
            samples_meta[0][0].get("question", ""),
            samples_meta[0][0].get("options", None),
        )
        warm_inputs = build_inputs_vsi(processor, first_frames, first_prompt, device)
        for _ in range(args.warmup):
            measure_once(model, warm_inputs, args.max_new_tokens)

        tv_list, tp_list, tg_list = [], [], []
        for row, frames in tqdm(samples_meta, desc=f"测时 {name}"):
            prompt = build_user_text(row.get("question", ""), row.get("options", None))
            inputs = build_inputs_vsi(processor, frames, prompt, device)
            tv, tp, tg = measure_once(model, inputs, args.max_new_tokens)
            tv_list.append(tv * 1000.0)
            tp_list.append(tp * 1000.0)
            tg_list.append(tg * 1000.0)

        v_m, v_s = ms_stats(tv_list)
        p_m, p_s = ms_stats(tp_list)
        g_m, g_s = ms_stats(tg_list)
        lm_approx = p_m - v_m
        dec_approx = max(g_m - p_m, 0.0)

        print(f"  vision_mean            : {v_m:8.2f} ms  (±{v_s:.2f})")
        print(f"  prefill_mean           : {p_m:8.2f} ms  (±{p_s:.2f})")
        print(f"  lm_prefill_approx_mean : {lm_approx:8.2f} ms")
        print(f"  generate_mean          : {g_m:8.2f} ms  (±{g_s:.2f})")
        print(f"  decode_approx_mean     : {dec_approx:8.2f} ms")
        if g_m > 0:
            print(f"  vision / generate      : {v_m / g_m * 100:.1f}%")
        if p_m > 0:
            print(f"  vision / prefill       : {v_m / p_m * 100:.1f}%")
        print()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
