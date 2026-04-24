from __future__ import annotations

import heapq
import importlib.util
import time
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

_AKS_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_AKS_PATCHED_MODULES: set[str] = set()


def _log(msg: str) -> None:
    print(f"[aks_sampler] {msg}", flush=True)


def _to_feature_tensor(features, torch):
    if isinstance(features, torch.Tensor):
        return features
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(features, attr, None)
        if isinstance(val, torch.Tensor):
            if attr == "last_hidden_state" and val.ndim >= 2:
                return val[:, 0, :]
            return val
    if isinstance(features, (tuple, list)) and features:
        first = features[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"无法将特征输出转换为 Tensor，实际类型: {type(features)!r}")


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch
    except ImportError as exc:
        raise ImportError("AKS 依赖缺失：需要安装 torch。") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_module_from_file(module_name: str, file_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {module_name}，文件: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _ensure_aks_lavis_patches(*, include_blip: bool = False, include_sevila: bool = False) -> None:
    base_dir = Path(__file__).resolve().parent / "AKS"
    patch_targets: list[tuple[str, Path]] = []
    if include_blip:
        patch_targets.append(
            (
                "lavis.models.blip_models.blip_image_text_matching",
                base_dir / "blip_image_text_matching.py",
            )
        )
    if include_sevila:
        patch_targets.append(
            (
                "lavis.models.sevila_models.sevila",
                base_dir / "sevila.py",
            )
        )

    for module_name, file_path in patch_targets:
        if module_name in _AKS_PATCHED_MODULES:
            continue
        if not file_path.exists():
            raise FileNotFoundError(f"AKS 适配文件不存在: {file_path}")
        loaded = sys.modules.get(module_name)
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file and Path(loaded_file).resolve() == file_path.resolve():
            _AKS_PATCHED_MODULES.add(module_name)
            continue
        _load_module_from_file(module_name, file_path)
        _AKS_PATCHED_MODULES.add(module_name)
        _log(f"patched lavis module: {module_name} <- {file_path}")


def _load_clip(device: str):
    key = ("clip", device)
    if key in _AKS_MODEL_CACHE:
        _log(f"reuse cached CLIP model on device={device}")
        return _AKS_MODEL_CACHE[key]
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("AKS-CLIP 依赖缺失：需要安装 transformers。") from exc
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _AKS_MODEL_CACHE[key] = (model, processor)
    _log(f"loaded CLIP model on device={device}")
    return _AKS_MODEL_CACHE[key]


def _load_blip(device: str):
    key = ("blip", device)
    if key in _AKS_MODEL_CACHE:
        _log(f"reuse cached BLIP model on device={device}")
        return _AKS_MODEL_CACHE[key]
    try:
        from transformers import BlipForImageTextRetrieval, BlipProcessor
    except ImportError as exc:
        raise ImportError("AKS-BLIP 依赖缺失：需要安装 transformers。") from exc
    model_id = "Salesforce/blip-itm-base-coco"
    model = BlipForImageTextRetrieval.from_pretrained(model_id).to(device).eval()
    processor = BlipProcessor.from_pretrained(model_id)
    _AKS_MODEL_CACHE[key] = (model, processor)
    _log(f"loaded BLIP model({model_id}) on device={device}")
    return _AKS_MODEL_CACHE[key]


def _load_blip2(device: str):
    key = ("blip2", device)
    if key in _AKS_MODEL_CACHE:
        _log(f"reuse cached BLIP2 model on device={device}")
        return _AKS_MODEL_CACHE[key]
    try:
        import torch
        from transformers import Blip2Model, Blip2Processor
    except ImportError as exc:
        raise ImportError("AKS-BLIP2 依赖缺失：需要安装 transformers。") from exc
    model_id = "Salesforce/blip2-opt-2.7b"
    model_kwargs: dict[str, Any] = {}
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16
    model = Blip2Model.from_pretrained(model_id, **model_kwargs).to(device).eval()
    processor = Blip2Processor.from_pretrained(model_id)
    _AKS_MODEL_CACHE[key] = (model, processor)
    _log(f"loaded BLIP2 model({model_id}) on device={device}")
    return _AKS_MODEL_CACHE[key]


def _load_sevila(device: str):
    key = ("sevila", device)
    if key in _AKS_MODEL_CACHE:
        _log(f"reuse cached SeViLA model on device={device}")
        return _AKS_MODEL_CACHE[key]
    try:
        _ensure_aks_lavis_patches(include_sevila=True)
        from lavis.models import load_model_and_preprocess
    except ImportError as exc:
        raise ImportError("AKS-SeViLA 依赖缺失：需要安装 lavis。") from exc
    model, vis_processors, text_processors = load_model_and_preprocess(
        name="sevila",
        model_type="pretrain_flant5xl",
        is_eval=True,
        device=device,
    )
    _AKS_MODEL_CACHE[key] = (model, vis_processors, text_processors)
    _log(f"loaded SeViLA model on device={device}")
    return _AKS_MODEL_CACHE[key]


def _read_frame_rgb(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def meanstd(
    len_scores: int,
    dic_scores: list[dict[str, Any]],
    n: int,
    fns: list[list[int]],
    t1: float,
    t2: float,
    all_depth: int,
):
    _ = len_scores
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score["score"]
        depth = dic_score["depth"]
        mean = np.mean(score)
        std = np.std(score)

        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = np.mean(top_score) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            score1 = score[: len(score) // 2]
            score2 = score[len(score) // 2 :]
            fn1 = fn[: len(score) // 2]
            fn2 = fn[len(score) // 2 :]
            split_scores.append(dict(score=score1, depth=depth + 1))
            split_scores.append(dict(score=score2, depth=depth + 1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score = []
        all_split_fn = []
    all_split_score = no_split_scores + all_split_score
    all_split_fn = no_split_fn + all_split_fn
    return all_split_score, all_split_fn


def _format_question_and_options(question: str | None, options: list[str] | None) -> str:
    _ = options
    q = (question or "").strip()
    if not q:
        raise ValueError("AKS 需要提供 question。")
    return q


def _compose_query(
    question: str | None,
    options: list[str] | None,
    answer: str | None,
    extract_feature_model: str,
) -> str:
    _ = answer
    qa_text = _format_question_and_options(question=question, options=options)
    if extract_feature_model == "sevila":
        return f"Question: {qa_text}. Is this a good frame can answer the question?"
    return qa_text


def _extract_blip_itm_score(outputs, torch) -> float:
    logits = getattr(outputs, "itm_score", None)
    if logits is None:
        logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("BLIP 输出中缺少 itm_score/logits，无法计算打分。")
    probs = torch.nn.functional.softmax(logits, dim=-1)
    if probs.shape[-1] >= 2:
        return float(probs[..., 1].item())
    return float(probs.squeeze().item())


def _extract_scores(
    video_path: str,
    question: str | None,
    options: list[str] | None,
    answer: str | None,
    extract_feature_model: str,
    device: str,
) -> tuple[list[float], list[int]]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("AKS 依赖缺失：需要安装 torch。") from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps) if fps and fps > 0 else 1)
    frame_nums = int(total_frames / step)
    _log(
        f"video stats: total_frames={total_frames}, fps={fps:.4f}, "
        f"sampling_step={step}, candidate_count={frame_nums}"
    )

    query = _compose_query(
        question=question,
        options=options,
        answer=answer,
        extract_feature_model=extract_feature_model,
    )
    preview_query = query if len(query) <= 160 else query[:157] + "..."
    _log(f"extract_feature_model={extract_feature_model}, query={preview_query}")
    scores: list[float] = []
    frame_num: list[int] = []

    if extract_feature_model == "blip":
        model, processor = _load_blip(device=device)
        for j in range(frame_nums):
            idx = j * step
            raw = _read_frame_rgb(cap, idx)
            if raw is None:
                continue
            image = Image.fromarray(raw)
            inputs = processor(
                images=image,
                text=query,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            with torch.no_grad():
                blip_output = model(**inputs, use_itm_head=True)
            scores.append(_extract_blip_itm_score(blip_output, torch))
            frame_num.append(idx)
    elif extract_feature_model == "blip2":
        model, processor = _load_blip2(device=device)
        text_inputs = processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            text_features = _to_feature_tensor(model.get_text_features(**text_inputs), torch)
        for j in range(frame_nums):
            idx = j * step
            raw = _read_frame_rgb(cap, idx)
            if raw is None:
                continue
            image = Image.fromarray(raw)
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = _to_feature_tensor(model.get_image_features(**image_inputs), torch)
            blip2_score = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
            scores.append(float(blip2_score.item()))
            frame_num.append(idx)
    elif extract_feature_model == "clip":
        model, processor = _load_clip(device=device)
        inputs_text = processor(text=query, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = _to_feature_tensor(model.get_text_features(**inputs_text), torch)
        for j in range(frame_nums):
            idx = j * step
            raw = _read_frame_rgb(cap, idx)
            if raw is None:
                continue
            image = Image.fromarray(raw)
            inputs_image = processor(images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = _to_feature_tensor(model.get_image_features(**inputs_image), torch)
            clip_score = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
            scores.append(float(clip_score.item()))
            frame_num.append(idx)
    elif extract_feature_model == "sevila":
        model, vis_processors, text_processors = _load_sevila(device=device)
        txt = text_processors["eval"](query)
        for j in range(frame_nums):
            idx = j * step
            raw = _read_frame_rgb(cap, idx)
            if raw is None:
                continue
            image = Image.fromarray(raw)
            img = vis_processors["eval"](image).unsqueeze(0).unsqueeze(0).to(device)
            samples = {"video": img, "loc_input": txt}
            with torch.no_grad():
                sevila_score = model.generate_score(samples)
            scores.append(float(sevila_score.squeeze(0).squeeze(0)))
            frame_num.append(idx)
    else:
        raise ValueError(
            f"AKS 不支持的 extract_feature_model: {extract_feature_model}，可选 blip/blip2/clip/sevila"
        )

    cap.release()
    if scores:
        arr = np.asarray(scores, dtype=np.float32)
        _log(
            f"score extraction done: valid_frames={len(frame_num)}, "
            f"min={float(arr.min()):.6f}, max={float(arr.max()):.6f}, mean={float(arr.mean()):.6f}"
        )
    else:
        _log("score extraction done: no valid score")
    return scores, frame_num


def _select_frame_indices(
    scores: list[float],
    frame_num: list[int],
    max_num_frames: int,
    ratio: int,
    t1: float,
    t2: float,
    all_depth: int,
) -> list[int]:
    _log(
        f"start selecting indices: total_scores={len(scores)}, ratio={ratio}, "
        f"max_num_frames={max_num_frames}, t1={t1}, t2={t2}, all_depth={all_depth}"
    )
    nums = int(len(scores) / ratio)
    if nums <= 0:
        _log("after ratio subsampling, nums <= 0, return []")
        return []
    new_score = [scores[num * ratio] for num in range(nums)]
    new_fnum = [frame_num[num * ratio] for num in range(nums)]
    score = new_score
    fn = new_fnum
    num = max_num_frames
    # 若深度过大，官方 f_num=int(num/2**depth) 在小 num 场景会退化为 0，导致整段无帧。
    # 这里仅限制有效 depth，不改变官方 f_num 公式本身。
    effective_depth = min(all_depth, int(math.log2(max(1, num))))
    if len(score) >= num:
        score_np = np.asarray(score, dtype=np.float32)
        normalized_data = (score_np - np.min(score_np)) / (np.max(score_np) - np.min(score_np))
        a, b = meanstd(len(score), [dict(score=normalized_data, depth=0)], num, [fn], t1, t2, effective_depth)
        depth_hist: dict[int, int] = {}
        for seg in a:
            d = int(seg["depth"])
            depth_hist[d] = depth_hist.get(d, 0) + 1
        _log(
            f"meanstd segmentation done: segments={len(a)}, depth_hist={depth_hist}, "
            f"effective_depth={effective_depth}"
        )
        out = []
        if len(score) >= num:
            for s, f in zip(a, b):
                f_num = int(num / 2 ** (s["depth"]))
                topk = heapq.nlargest(f_num, range(len(s["score"])), s["score"].__getitem__)
                f_nums = [f[t] for t in topk]
                out.extend(f_nums)
        out.sort()
        _log(f"selected candidate indices from segments: count={len(out)}, head={out[:10]}")
        return out
    _log(f"score count < num_frames, use all sampled frame indices: count={len(fn)}")
    return fn


def sample_aks_frames(
    video_path: str,
    num_frames: int,
    random_seed: int | None = None,
    question: str | None = None,
    options: list[str] | None = None,
    answer: str | None = None,
    extract_feature_model: str = "blip",
    device: str | None = None,
    ratio: int = 1,
    t1: float = 0.8,
    t2: float = -100.0,
    all_depth: int = 5,
) -> list[Image.Image]:
    t0 = time.time()
    _ = random_seed
    _log(
        "start sampling: "
        f"video_path={video_path}, num_frames={num_frames}, extract_feature_model={extract_feature_model}, "
        f"ratio={ratio}, t1={t1}, t2={t2}, all_depth={all_depth}"
    )
    if num_frames <= 0:
        _log("num_frames <= 0, return []")
        return []
    if ratio <= 0:
        raise ValueError(f"AKS 参数错误：ratio 必须 > 0，当前值 {ratio}")

    resolved_device = _resolve_device(device)
    _log(f"resolved device={resolved_device}")
    scores, frame_num = _extract_scores(
        video_path=video_path,
        question=question,
        options=options,
        answer=answer,
        extract_feature_model=extract_feature_model,
        device=resolved_device,
    )

    if not scores or not frame_num:
        _log("scores/frame_num empty, return []")
        return []

    selected_indices = _select_frame_indices(
        scores=scores,
        frame_num=frame_num,
        max_num_frames=num_frames,
        ratio=ratio,
        t1=t1,
        t2=t2,
        all_depth=all_depth,
    )
    _log(
        f"post-selection indices: count={len(selected_indices)}, "
        f"head={selected_indices[:10]}"
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    frames: list[Image.Image] = []
    try:
        for idx in selected_indices[:num_frames]:
            rgb = _read_frame_rgb(cap, idx)
            if rgb is not None:
                frames.append(Image.fromarray(rgb))
    finally:
        cap.release()
    _log(
        f"final selected frames: count={len(frames)}, "
        f"requested={num_frames}, elapsed={time.time() - t0:.2f}s"
    )
    return frames
