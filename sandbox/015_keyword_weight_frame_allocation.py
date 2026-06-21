#!/usr/bin/env python3
"""统计 keyword_weight_strength 消融下每个关键词分到的帧数，并画图。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_loaders import get_data_loader
from vqa_eval_ours import (
    _allocate_counts_by_weights,
    _encode_images,
    _encode_texts,
    _keyword_cache_file,
    _keyword_cache_run_dir,
    _load_clip,
    _load_keyword_cache_entry,
    _load_preprocessed_candidate_frames,
    _local_evidence_score,
    _merge_keywords,
    _pool_positions_at_fps,
    _resolve_keyword_cache_root,
)

SEED = 42
DATASET = "videomme"
TASK_FILTER = "short"
NUM_SAMPLES = 10
MAX_KEYWORDS = 10
NUM_FRAMES_GRID = [16, 32]
WEIGHT_STRENGTH_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
KEYWORD_PROMPT_VERSION = 0
KEYWORD_EXTRACTOR_MODEL = "poe-gpt-5.2"
KEYWORD_CACHE_NUMBER = 0
CANDIDATE_POOL_FPS = 1.0
PREPROCESSED_CLIP_FPS = 1.0
VISUAL_ENCODER_MODEL = "openai/clip-vit-base-patch32"
OURS_CLIP_BATCH_SIZE = 16
PREPROCESSED_CLIP_DIR = Path("/userhome/cs3/duanty/dataset_preposcess") / DATASET / "clip_1"
OUT_DIR = Path(__file__).resolve().parent / "015_outputs"


def compute_keyword_weights(
    kw_emb_rep: torch.Tensor,
    img_emb: torch.Tensor,
    keyword_weight_strength: float,
) -> torch.Tensor:
    sims = kw_emb_rep @ img_emb.T
    m = int(kw_emb_rep.shape[0])
    local_evidence = torch.tensor(
        [_local_evidence_score(sims[i]) for i in range(m)],
        dtype=kw_emb_rep.dtype,
        device=kw_emb_rep.device,
    ).clamp(min=0.0, max=1.0)
    s = float(local_evidence.sum().item())
    info_weights = (
        local_evidence / s
        if s > 1e-12
        else torch.ones_like(local_evidence) / float(max(1, m))
    )
    uniform_weights = torch.ones_like(info_weights) / float(max(1, m))
    strength = max(0.0, float(keyword_weight_strength))
    kw_weights = ((1.0 - strength) * uniform_weights + strength * info_weights).clamp(min=0.0)
    ws = float(kw_weights.sum().item())
    if ws <= 1e-12:
        return uniform_weights
    return kw_weights / ws


def quota_topk_select_with_counts(
    kw_sims: torch.Tensor,
    kw_w: torch.Tensor,
    budget: int,
    candidate_idx: list[int],
) -> tuple[list[int], list[int], torch.Tensor]:
    """与 vqa_eval_ours._quota_topk_select 一致，额外返回每关键词帧数与 quota。"""
    if kw_sims.ndim != 2 or budget <= 0:
        return [], [0] * int(kw_sims.shape[0]), torch.zeros((kw_sims.shape[0],), dtype=torch.long)
    m = int(kw_sims.shape[0])
    n = int(kw_sims.shape[1])
    if m == 0 or n == 0:
        return [], [], torch.zeros((0,), dtype=torch.long)

    if kw_w.numel() != m:
        kw_w = torch.ones((m,), device=kw_sims.device)
    kw_w = kw_w.float().clamp(min=0.0)
    if float(kw_w.sum().item()) <= 1e-12:
        kw_w = torch.ones_like(kw_w)

    cand = sorted({int(i) for i in candidate_idx if 0 <= int(i) < n})
    if len(cand) < min(budget, n):
        cand = list(range(n))
    cand_set = set(cand)
    max_pick = min(int(budget), len(cand))
    quotas = _allocate_counts_by_weights(kw_w, max_pick)
    keyword_order = [int(i) for i in torch.argsort(kw_w, descending=True).tolist()]
    ranked_by_keyword: list[list[int]] = []
    next_pos = [0 for _ in range(m)]
    for j in range(m):
        ranked_by_keyword.append(
            [int(idx) for idx in torch.argsort(kw_sims[j], descending=True).tolist() if int(idx) in cand_set]
        )

    selected: list[int] = []
    selected_set: set[int] = set()
    keyword_counts = [0 for _ in range(m)]

    def _take_next(j: int) -> int | None:
        while next_pos[j] < len(ranked_by_keyword[j]):
            idx = ranked_by_keyword[j][next_pos[j]]
            next_pos[j] += 1
            if idx in selected_set:
                continue
            return idx
        return None

    for j in keyword_order:
        q = int(quotas[j].item())
        if q <= 0:
            continue
        while q > 0 and len(selected) < max_pick:
            idx = _take_next(j)
            if idx is None:
                break
            selected.append(idx)
            selected_set.add(idx)
            keyword_counts[j] += 1
            q -= 1
        if len(selected) >= max_pick:
            break

    while len(selected) < max_pick:
        progressed = False
        for j in keyword_order:
            idx = _take_next(j)
            if idx is None:
                continue
            selected.append(idx)
            selected_set.add(idx)
            keyword_counts[j] += 1
            progressed = True
            if len(selected) >= max_pick:
                break
        if not progressed:
            break
    return selected, keyword_counts, quotas


def keyword_cache_path(sample_id: str, task_type: str) -> Path:
    cache_root = _resolve_keyword_cache_root(None)
    run_dir = _keyword_cache_run_dir(
        cache_root,
        dataset=DATASET,
        task_type=task_type,
        extractor_model=KEYWORD_EXTRACTOR_MODEL,
        prompt_version=KEYWORD_PROMPT_VERSION,
        target_keywords=MAX_KEYWORDS,
        cache_number=KEYWORD_CACHE_NUMBER,
    )
    return _keyword_cache_file(run_dir, sample_id)


def load_cached_keywords(sample_id: str, task_type: str) -> tuple[list[str], list[str]]:
    cache_path = keyword_cache_path(sample_id, task_type)
    cached = _load_keyword_cache_entry(cache_path, sample_id=sample_id)
    if cached is None:
        raise RuntimeError(f"关键词缓存未命中: sample={sample_id}, path={cache_path}")
    return cached


def plot_experiment(
    *,
    num_frames: int,
    strength: float,
    rank_labels: list[str],
    mean_frames: list[float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(rank_labels) * 0.55), 5))
    x = np.arange(len(rank_labels))
    bars = ax.bar(x, mean_frames, color="#4C78A8", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(rank_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frames allocated (mean over samples)")
    ax.set_xlabel("Keywords sorted by weight (high → low)")
    ax.set_title(
        f"Keyword frame allocation | num_frames={num_frames}, "
        f"keyword_weight_strength={strength:g}, max_keywords={MAX_KEYWORDS}, n={NUM_SAMPLES}"
    )
    ax.set_ylim(0, max(mean_frames + [1]) * 1.15)
    for bar, val in zip(bars, mean_frames):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[015] 输出目录: {OUT_DIR}")

    loader = get_data_loader(DATASET, seed=SEED, task_filter=TASK_FILTER)
    samples = loader.get_eval_samples(sample_count=NUM_SAMPLES)
    if len(samples) < NUM_SAMPLES:
        raise RuntimeError(f"样本不足 {NUM_SAMPLES}，当前 {len(samples)}")

    print("[015] 加载 CLIP...")
    clip_proc, clip_model, clip_device = _load_clip(VISUAL_ENCODER_MODEL, None)

    bundles: dict[str, dict] = {}
    for sample in tqdm(samples, desc="015 预计算"):
        sid = str(sample.sample_id)
        kws_raw, kws = load_cached_keywords(sid, str(sample.task_type))
        frame_ids, imgs = _load_preprocessed_candidate_frames(str(PREPROCESSED_CLIP_DIR), sid)
        keep = _pool_positions_at_fps(len(imgs), PREPROCESSED_CLIP_FPS, CANDIDATE_POOL_FPS)
        if keep:
            frame_ids, imgs = [frame_ids[i] for i in keep], [imgs[i] for i in keep]
        kw_emb = _encode_texts(kws, clip_proc, clip_model, clip_device, 32)
        kws_rep, kw_emb_rep, _ = _merge_keywords(kws, kw_emb, MAX_KEYWORDS)
        img_emb = _encode_images(imgs, clip_proc, clip_model, clip_device, OURS_CLIP_BATCH_SIZE)
        bundles[sid] = {
            "kws_rep": kws_rep,
            "kw_emb_rep": kw_emb_rep,
            "img_emb": img_emb,
        }

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for num_frames in NUM_FRAMES_GRID:
        for strength in WEIGHT_STRENGTH_GRID:
            tag = f"nf{num_frames}_ws{strength:g}".replace(".", "p")
            rank_to_counts: dict[int, list[int]] = {}

            for sid, bundle in bundles.items():
                kws_rep = bundle["kws_rep"]
                kw_emb_rep = bundle["kw_emb_rep"]
                img_emb = bundle["img_emb"]
                kw_weights = compute_keyword_weights(kw_emb_rep, img_emb, strength)
                kw_sims = kw_emb_rep @ img_emb.T
                _, keyword_counts, quotas = quota_topk_select_with_counts(
                    kw_sims=kw_sims,
                    kw_w=kw_weights,
                    budget=num_frames,
                    candidate_idx=list(range(int(kw_sims.shape[1]))),
                )
                order = sorted(range(len(kws_rep)), key=lambda i: float(kw_weights[i].item()), reverse=True)
                for rank, j in enumerate(order, start=1):
                    rank_to_counts.setdefault(rank, []).append(int(keyword_counts[j]))
                    detail_rows.append(
                        {
                            "num_frames": num_frames,
                            "keyword_weight_strength": strength,
                            "sample_id": sid,
                            "keyword_rank": rank,
                            "keyword": kws_rep[j],
                            "weight": float(kw_weights[j].item()),
                            "quota": int(quotas[j].item()),
                            "frames_allocated": int(keyword_counts[j]),
                        }
                    )

            max_rank = max(rank_to_counts) if rank_to_counts else 0
            rank_labels = [f"#{r}" for r in range(1, max_rank + 1)]
            mean_frames = [
                float(np.mean(rank_to_counts[r])) if rank_to_counts.get(r) else 0.0 for r in range(1, max_rank + 1)
            ]
            plot_path = OUT_DIR / f"alloc_{tag}.png"
            plot_experiment(
                num_frames=num_frames,
                strength=strength,
                rank_labels=rank_labels,
                mean_frames=mean_frames,
                out_path=plot_path,
            )
            summary_rows.append(
                {
                    "num_frames": num_frames,
                    "keyword_weight_strength": strength,
                    "max_keyword_rank": max_rank,
                    "mean_frames_by_rank": mean_frames,
                    "plot": str(plot_path),
                }
            )
            print(f"[015] 完成 {tag}: {plot_path}")

    detail_path = OUT_DIR / "allocation_detail.jsonl"
    with detail_path.open("w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = OUT_DIR / "allocation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"[015] 明细: {detail_path}")
    print(f"[015] 汇总: {summary_path}")


if __name__ == "__main__":
    main()
