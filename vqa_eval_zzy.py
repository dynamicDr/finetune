from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from data_loaders import should_apply_vl_pixel_limits, get_data_loader, list_supported_datasets
from data_loaders.base import VQASample, sample_matches_task_filter
from model_response_mode import load_model_response_mode_config, parse_response_by_mode, resolve_model_mode
from utils import build_user_text, build_user_text_with_subtitles, from_pretrained_local_first
from vl_common import load_model_and_processor, prepare_vlm_inputs

MODE_MAX_NEW_TOKENS = {"thinking": 4086, "instruct": 128}
PREPROCESSED_CLIP_COMPATIBLE_METHODS = {"zzy"}
_CLIP_CACHE: dict[str, tuple[Any, Any, str]] = {}


def _log(msg: str) -> None:
    print(f"[vqa_eval_zzy] {msg}", flush=True)


def _parse_subtitle_time(time_str: str) -> float:
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _load_srt_segments(subtitle_path: str) -> list[tuple[float, float, str]]:
    p = Path(subtitle_path)
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    segs: list[tuple[float, float, str]] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        time_line_idx = 0 if "-->" in lines[0] else (1 if len(lines) > 1 and "-->" in lines[1] else -1)
        if time_line_idx < 0:
            continue
        try:
            ts0, ts1 = [x.strip() for x in lines[time_line_idx].split("-->")]
            start_t = _parse_subtitle_time(ts0)
            end_t = _parse_subtitle_time(ts1)
        except Exception:
            continue
        content_lines = lines[time_line_idx + 1 :]
        if not content_lines:
            continue
        raw = " ".join(content_lines)
        cleaned = re.sub(r"<[^>]+>", "", raw).strip()
        if cleaned:
            segs.append((start_t, end_t, cleaned))
    return segs


def _resolve_subtitle_path(sample: VQASample, subtitles_dir: str | None) -> str | None:
    explicit_dir = os.path.expanduser(subtitles_dir) if subtitles_dir else ""
    video_path = Path(sample.video_path).expanduser().resolve()
    video_stem = video_path.stem
    meta_video_id = str(sample.metadata.get("videoID", "")).strip() if isinstance(sample.metadata, dict) else ""
    candidates: list[Path] = []
    subtitle_dir_names = ("subtitle", "subtitles", "subtitles/subtitle")

    def _append_stem_files(base_dir: Path) -> None:
        for stem in [meta_video_id, video_stem]:
            if stem:
                candidates.extend([base_dir / f"{stem}.srt", base_dir / f"{stem}.SRT"])

    if explicit_dir:
        base = Path(explicit_dir)
        if base.is_dir():
            _append_stem_files(base)
            for subname in subtitle_dir_names:
                _append_stem_files(base / subname)
    for root in [video_path.parent, video_path.parent.parent, video_path.parent.parent.parent]:
        for subname in subtitle_dir_names:
            _append_stem_files(root / subname)
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _collect_subtitles_for_sample(
    sample: VQASample,
    sampled_frame_ids: list[int],
    subtitles_dir: str | None,
) -> tuple[list[str], list[str]]:
    subtitle_path = _resolve_subtitle_path(sample, subtitles_dir=subtitles_dir)
    if not subtitle_path or not sampled_frame_ids:
        return [], ["" for _ in sampled_frame_ids]
    cap = cv2.VideoCapture(sample.video_path)
    if not cap.isOpened():
        return [], ["" for _ in sampled_frame_ids]
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1e-6:
        return [], ["" for _ in sampled_frame_ids]
    segments = _load_srt_segments(subtitle_path)
    if not segments:
        return [], ["" for _ in sampled_frame_ids]
    out: list[str] = []
    seen: set[str] = set()
    per_frame_subs: list[str] = []
    for fid in sampled_frame_ids:
        t = float(fid) / fps
        matched = ""
        for s0, s1, txt in segments:
            if s0 <= t < s1:
                matched = txt
                if txt not in seen:
                    seen.add(txt)
                    out.append(txt)
                break
        per_frame_subs.append(matched)
    return out, per_frame_subs


def _sample_uniform_positions(total: int, target: int) -> list[int]:
    if target >= total:
        return list(range(total))
    return sorted(set(int(x) for x in torch.linspace(0, total - 1, target).round().tolist()))


def _collect_video_frames_uniform(video_path: str, target_frames: int) -> tuple[list[int], list[Image.Image]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"视频帧数无效: {video_path}")
    idxs = _sample_uniform_positions(frame_count, target_frames)
    frame_ids, images = [], []
    try:
        for fid in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ids.append(int(fid))
            images.append(Image.fromarray(rgb))
    finally:
        cap.release()
    if not images:
        raise RuntimeError(f"视频无可用候选帧: {video_path}")
    return frame_ids, images


def _load_preprocessed_candidate_frames(preprocessed_clip_dir: str, sample_id: str) -> tuple[list[int], list[Image.Image]]:
    sample_dir = Path(preprocessed_clip_dir).expanduser().resolve() / re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id.strip()).strip("_")
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"预处理帧目录不存在: {sample_dir}")
    meta_path = sample_dir / "metadata.json"
    frame_ids: list[int] = []
    image_paths: list[Path] = []
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        ids = meta.get("frame_ids", [])
        files = meta.get("files", [])
        if isinstance(ids, list) and isinstance(files, list) and len(ids) == len(files):
            for fid, rel_name in zip(ids, files):
                p = sample_dir / str(rel_name)
                if p.is_file():
                    frame_ids.append(int(fid))
                    image_paths.append(p)
    if not image_paths:
        for p in sorted(sample_dir.glob("frame_*.jpg")):
            m = re.match(r"frame_(\d+)\.jpg$", p.name)
            if m:
                frame_ids.append(int(m.group(1)))
                image_paths.append(p)
    if not image_paths:
        raise RuntimeError(f"预处理帧目录中没有可用图片: {sample_dir}")
    images: list[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return frame_ids, images


def _to_feature_tensor(features: Any) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(features, attr, None)
        if isinstance(v, torch.Tensor):
            return v[:, 0, :] if attr == "last_hidden_state" and v.ndim >= 2 else v
    if isinstance(features, (tuple, list)) and features and isinstance(features[0], torch.Tensor):
        return features[0]
    raise TypeError(f"无法转换特征类型: {type(features)!r}")


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _load_clip(model_id: str, device: str | None) -> tuple[Any, Any, str]:
    d = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = f"{model_id}::{d}"
    if key not in _CLIP_CACHE:
        proc = from_pretrained_local_first(AutoProcessor.from_pretrained, model_id, log=_log)
        model = from_pretrained_local_first(AutoModel.from_pretrained, model_id, log=_log).to(d).eval()
        _CLIP_CACHE[key] = (proc, model, d)
    return _CLIP_CACHE[key]


def _encode_images(images: list[Image.Image], proc: Any, model: Any, device: str, bs: int) -> torch.Tensor:
    outs = []
    for i in range(0, len(images), bs):
        inputs = proc(images=images[i:i + bs], return_tensors="pt").to(device)
        with torch.no_grad():
            feats = _to_feature_tensor(model.get_image_features(**inputs))
        outs.append(_norm(feats))
    return torch.cat(outs, dim=0)


def _encode_text(text: str, proc: Any, model: Any, device: str) -> torch.Tensor:
    inputs = proc(text=[text], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = _to_feature_tensor(model.get_text_features(**inputs))
    return _norm(feats)[0]


def _run_vlm_once(model: Any, proc: Any, frames: list[Image.Image], prompt: str, max_new_tokens: int, model_mode: str) -> dict[str, Any]:
    inputs, _ = prepare_vlm_inputs(proc, frames, prompt, model=model)
    inputs = inputs.to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
    infer_t = time.perf_counter() - t0
    seq = out.sequences
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, seq)]
    resp = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    _, _, pred = parse_response_by_mode(response=resp, has_options=True, model_mode=model_mode)
    pred_u = str(pred).strip().upper()
    if pred_u not in {"A", "B", "C", "D"}:
        pred_u = "A"
    return {"pred_answer": pred_u, "response": resp, "inference_time": infer_t, "hit_max_tokens": int(len(out.scores) >= max_new_tokens)}


def _clip_topk_select(
    frame_ids: list[int],
    imgs: list[Image.Image],
    question: str,
    options: list[str] | None,
    budget: int,
    clip_proc: Any,
    clip_model: Any,
    clip_device: str,
    clip_batch_size: int,
) -> list[int]:
    n = len(imgs)
    if n == 0:
        return []
    k = min(max(1, int(budget)), n)
    prompt = build_user_text(question, options)
    img_emb = _encode_images(imgs, clip_proc, clip_model, clip_device, clip_batch_size)
    txt_emb = _encode_text(prompt, clip_proc, clip_model, clip_device)
    scores = img_emb @ txt_emb.unsqueeze(1)
    top_idx = torch.argsort(scores.squeeze(1), descending=True)[:k].tolist()
    return sorted([int(i) for i in top_idx], key=lambda x: frame_ids[x])


def calculate_mra(pred: float, gt: float) -> float:
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, 1 - abs(pred - gt) / abs(gt))


def _compute_accuracy_from_results(results: dict, task_filter: str) -> tuple[float, float]:
    avg_accuracy = 0.0
    if task_filter in {"mcq", "short", "medium", "long"} and results["total"] > 0:
        avg_accuracy = results["correct"] / results["total"] * 100
    elif task_filter == "numeric" and results["mra_count"] > 0:
        avg_accuracy = results["mra_sum"] / results["mra_count"] * 100
    elif task_filter == "all":
        total_score = results["correct"] + results["mra_sum"]
        total_count = results["total"] + results["mra_count"]
        if total_count > 0:
            avg_accuracy = total_score / total_count * 100
    avg_inference_time = sum(results["inference_times"]) / len(results["inference_times"]) if results["inference_times"] else 0.0
    return avg_accuracy, avg_inference_time


def _compute_score_counts_for_csv(results: dict, task_filter: str) -> tuple[int, float]:
    if task_filter in {"mcq", "short", "medium", "long"}:
        return int(results["total"]), float(results["correct"])
    if task_filter == "numeric":
        return int(results["mra_count"]), float(results["mra_sum"])
    return int(results["total"] + results["mra_count"]), float(results["correct"] + results["mra_sum"])


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _eval_one_sample(
    model: Any,
    proc: Any,
    clip_proc: Any,
    clip_model: Any,
    clip_device: str,
    sample: VQASample,
    args: argparse.Namespace,
    max_new_tokens: int,
    model_mode: str,
    preprocessed_clip_dir: str | None,
    use_subtitles: bool,
    subtitles_dir: str | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if args.use_preprocessed_clip_frames:
        frame_ids, imgs = _load_preprocessed_candidate_frames(preprocessed_clip_dir or "", sample.sample_id)
    else:
        # zzy 占位：先均匀采样较大池，再用 clip top-k 精筛
        frame_ids, imgs = _collect_video_frames_uniform(sample.video_path, target_frames=max(64, args.num_frames * 4))
    frame_sampling_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    selected_idx = _clip_topk_select(
        frame_ids=frame_ids,
        imgs=imgs,
        question=sample.question,
        options=sample.options,
        budget=args.num_frames,
        clip_proc=clip_proc,
        clip_model=clip_model,
        clip_device=clip_device,
        clip_batch_size=args.ours_clip_batch_size,
    )
    if not selected_idx:
        raise RuntimeError(f"zzy 未选出有效帧: sample={sample.sample_id}")
    one_shot_frames = [imgs[i] for i in selected_idx]
    sampled_frame_ids = [int(frame_ids[i]) for i in selected_idx]

    subtitles_for_prompt, _ = (
        _collect_subtitles_for_sample(sample, sampled_frame_ids, subtitles_dir=subtitles_dir)
        if use_subtitles
        else ([], ["" for _ in sampled_frame_ids])
    )
    prompt = build_user_text_with_subtitles(sample.question, sample.options, subtitles_for_prompt) if use_subtitles else build_user_text(sample.question, sample.options)
    out = _run_vlm_once(model, proc, one_shot_frames, prompt, max_new_tokens, model_mode)
    emb_time = time.perf_counter() - t1

    return {
        "pred_answer": str(out["pred_answer"]),
        "response": str(out["response"]),
        "inference_time": float(out["inference_time"]),
        "frame_sampling_time": float(frame_sampling_time),
        "embedding_build_time": float(emb_time),
        "selected_frame_count": len(selected_idx),
        "over_limit_count": int(out["hit_max_tokens"]),
    }


def evaluate_vqa(
    model: Any,
    proc: Any,
    samples: list[VQASample],
    args: argparse.Namespace,
    max_new_tokens: int,
    model_mode: str,
    preprocessed_clip_dir: str | None,
    use_subtitles: bool,
    subtitles_dir: str | None,
) -> dict[str, Any]:
    clip_proc, clip_model, clip_device = _load_clip(args.ours_clip_model_id, args.ours_clip_device)
    res = {"correct": 0, "total": 0, "mra_sum": 0.0, "mra_count": 0, "inference_times": [], "frame_sampling_times": [], "embedding_build_times": [], "selected_frame_counts": [], "over_max_tokens_count": 0}
    pbar = tqdm(samples, desc="评估进度(zzy clip top-k)")
    for s in pbar:
        if not sample_matches_task_filter(s, args.task_filter):
            continue
        out = _eval_one_sample(
            model=model,
            proc=proc,
            clip_proc=clip_proc,
            clip_model=clip_model,
            clip_device=clip_device,
            sample=s,
            args=args,
            max_new_tokens=max_new_tokens,
            model_mode=model_mode,
            preprocessed_clip_dir=preprocessed_clip_dir,
            use_subtitles=use_subtitles,
            subtitles_dir=subtitles_dir,
        )
        pred = out["pred_answer"]
        res["inference_times"].append(float(out["inference_time"]))
        res["frame_sampling_times"].append(float(out["frame_sampling_time"]))
        res["embedding_build_times"].append(float(out["embedding_build_time"]))
        res["selected_frame_counts"].append(int(out["selected_frame_count"]))
        res["over_max_tokens_count"] += int(out["over_limit_count"])
        if s.options is not None:
            res["total"] += 1
            if str(s.answer).strip().upper() == str(pred).strip().upper():
                res["correct"] += 1
        else:
            try:
                res["mra_sum"] += calculate_mra(float(pred) if pred else 0.0, float(s.answer))
            except (ValueError, TypeError):
                pass
            res["mra_count"] += 1
        acc, t = _compute_accuracy_from_results(res, args.task_filter)
        pbar.set_postfix(Acc=f"{acc:.2f}%", AvgTime=f"{t:.2f}s")
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VQA zzy: clip top-k 占位方法")
    p.add_argument("--dataset", type=str, default="vsibench", choices=list_supported_datasets())
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--dataset_name", type=str, default="nyu-visionx/VSI-Bench")
    p.add_argument("--dataset_config", type=str, default="full")
    p.add_argument("--no_dataset_config", action="store_true")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Thinking")
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--video_dir", type=str, default="~/dataset/vsi_bench")
    p.add_argument("--num_samples", type=str, default="10")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--frame_sampling_method", type=str, default="zzy", choices=["zzy"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task_filter", type=str, default="all", choices=["all", "mcq", "numeric", "short", "medium", "long"])
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--ours_clip_model_id", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--ours_clip_device", type=str, default=None)
    p.add_argument("--ours_clip_batch_size", type=int, default=16)
    p.add_argument("--model_mode_config", type=str, default="config/model_response_modes.json")
    p.add_argument("--log_file", type=str, default="vqa_embedding_evaluation_log.csv")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--use_train_split", action="store_true")
    p.add_argument("--use_preprocessed_clip_frames", action="store_true")
    p.add_argument("--preprocessed_clip_fps", type=float, default=1.0)
    p.add_argument("--preprocessed_clip_dir", type=str, default="")
    p.add_argument("--use_subtitles", action="store_true")
    p.add_argument("--subtitles_dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    exp_t0 = time.perf_counter()
    args = parse_args()
    video_dir = os.path.expanduser(args.video_dir)
    eval_csv_dir = Path(__file__).resolve().parent / "eval_csv"
    eval_csv_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(eval_csv_dir / Path(args.log_file).name)
    default_pre = Path("/userhome/cs3/duanty/dataset_preposcess") / args.dataset / f"clip_{args.preprocessed_clip_fps:g}"
    preprocessed_clip_dir = os.path.expanduser(args.preprocessed_clip_dir) if args.preprocessed_clip_dir.strip() else str(default_pre)
    if args.use_preprocessed_clip_frames and args.frame_sampling_method not in PREPROCESSED_CLIP_COMPATIBLE_METHODS:
        raise ValueError("use_preprocessed_clip_frames 仅支持 zzy。")

    sample_count = None if args.num_samples.lower() == "all" else int(args.num_samples)
    loader = get_data_loader(args.dataset, video_dir=video_dir, seed=args.seed, train_ratio=args.train_ratio, task_filter=args.task_filter, dataset_name=args.dataset_name, dataset_config=args.dataset_config, no_dataset_config=args.no_dataset_config)
    samples = loader.get_split_samples(split=args.dataset_split, use_train_split=args.use_train_split, sample_count=sample_count)

    resolved_model_path = os.path.expanduser(args.model_path)
    lora_path = ""
    if args.model_name:
        model_name = args.model_name
    elif args.use_lora:
        model_name = os.path.expanduser(args.base_model) if args.base_model else "Qwen/Qwen3-VL-4B-Thinking"
        lora_path = resolved_model_path
    else:
        model_name = os.path.basename(resolved_model_path.rstrip("/"))

    mode_cfg = load_model_response_mode_config(args.model_mode_config)
    candidates = [resolved_model_path] + ([model_name] if model_name else []) + ([os.path.expanduser(args.base_model)] if args.base_model else [])
    model_mode, last_err = "", None
    for c in candidates:
        try:
            model_mode, _, _ = resolve_model_mode(model_identifier=c, config=mode_cfg)
            break
        except (KeyError, ValueError) as e:
            last_err = e
    if not model_mode:
        raise RuntimeError(f"模型模式识别失败: {candidates}") from last_err
    if model_mode == "thinking":
        raise RuntimeError("本脚本仅支持 instruct 模式。请改用 instruct 模型或扩展实现。")
    effective_max_new_tokens = MODE_MAX_NEW_TOKENS[model_mode]

    apply_pixel_limits = should_apply_vl_pixel_limits(
        resolved_model_path,
        args.dataset,
        args.dataset_split,
        args.dataset_name,
    )
    if apply_pixel_limits:
        print(
            f"[vqa_eval_zzy] 启用 processor 像素限制（num_frames={args.num_frames}，防 OOM / context 溢出）",
            flush=True,
        )
    model, proc = load_model_and_processor(
        resolved_model_path,
        use_lora=args.use_lora,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
        apply_pixel_limits=apply_pixel_limits,
        num_frames=args.num_frames,
    )
    results = evaluate_vqa(
        model=model,
        proc=proc,
        samples=samples,
        args=args,
        max_new_tokens=effective_max_new_tokens,
        model_mode=model_mode,
        preprocessed_clip_dir=preprocessed_clip_dir,
        use_subtitles=bool(args.use_subtitles),
        subtitles_dir=(args.subtitles_dir.strip() if args.subtitles_dir.strip() else None),
    )

    avg_acc, avg_time = _compute_accuracy_from_results(results, args.task_filter)
    eval_n, correct = _compute_score_counts_for_csv(results, args.task_filter)
    avg_fs = _avg(results["frame_sampling_times"])
    avg_emb = _avg(results["embedding_build_times"])
    avg_sel = _avg(results["selected_frame_counts"])
    avg_total_h = (time.perf_counter() - exp_t0) / 3600.0

    header = [
        "timestamp",
        "dataset",
        "seed",
        "task_filter",
        "num_samples",
        "evaluated_samples",
        "correct_count",
        "accuracy_percent",
        "num_frames",
        "avg_inference_time",
        "frame_sampling_method",
        "avg_frame_sampling_time",
        "avg_embedding_build_time",
        "avg_selected_frame_count",
        "avg_total_time_hours",
        "over_max_tokens_count",
        "model_name",
        "lora_path",
        "train_ratio",
        "eval_split",
    ]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        args.dataset,
        args.seed,
        args.task_filter,
        len(samples),
        eval_n,
        f"{correct:.6f}",
        f"{avg_acc:.2f}",
        args.num_frames,
        f"{avg_time:.3f}",
        args.frame_sampling_method,
        f"{avg_fs:.6f}",
        f"{avg_emb:.6f}",
        f"{avg_sel:.6f}",
        f"{avg_total_h:.6f}",
        int(results["over_max_tokens_count"]),
        model_name,
        lora_path,
        args.train_ratio,
        "train" if args.use_train_split else "test",
    ]
    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)

    _log(f"评估完成: samples={len(samples)}, acc={avg_acc:.2f}%, avg_infer={avg_time:.3f}s")


if __name__ == "__main__":
    main()
