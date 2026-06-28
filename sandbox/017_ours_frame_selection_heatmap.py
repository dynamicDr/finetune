#!/usr/bin/env python3
"""017: Ours 方法选帧时间轴热力图（参数行 × 时间，深色=选中帧）。

入口函数 ``run_ours_frame_selection_heatmap`` 可通过 ``OursHeatmapConfig`` 调整
keyword_weight_strength、num_frames、max_keywords 等超参。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
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
from data_loaders.base import VQASample
from utils import (
    keyword_cache_file,
    keyword_cache_run_dir,
    load_keyword_cache_entry,
    load_preprocessed_candidate_frames,
    pool_positions_at_fps,
    resolve_keyword_cache_root,
    resolve_preprocessed_clip_dir,
)
from vqa_eval_ours import (
    VisualEncoder,
    _allocate_counts_by_weights,
    _compute_keyword_information,
    _compute_kw_frame_sims,
    _encode_texts_for_dedup,
    _load_visual_encoder,
    _merge_keywords,
)

OUT_DIR = Path(__file__).resolve().parent / "017_outputs"


@dataclass
class OursHeatmapConfig:
    """Ours 选帧热力图实验配置。"""

    seed: int = 42
    dataset: str = "videomme"
    task_filter: str = "all"
    num_samples: int = 10
    num_frames_grid: list[int] = field(default_factory=lambda: [16, 32])
    keyword_weight_strength_grid: list[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 4.0, 8.0])
    max_keywords: int = 10
    keyword_prompt_version: int = 0
    keyword_extractor_model: str = "poe-gpt-5.2"
    keyword_cache_number: int = 0
    keyword_cache_dir: str = ""
    candidate_pool_fps: float = 1.0
    preprocessed_clip_fps: float = 1.0
    visual_encoder_model: str = "openai/clip-vit-base-patch32"
    ours_clip_batch_size: int = 16
    use_preprocessed_clip_frames: bool = True
    num_time_bins: int = 200
    include_uniform_baseline: bool = True
    out_dir: Path = field(default_factory=lambda: OUT_DIR)


@dataclass
class SampleBundle:
    frame_ids: list[int]
    kw_frame_sims: torch.Tensor
    kws_rep: list[str]


def get_video_meta(video_path: str) -> tuple[int, float, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    duration_sec = (total_frames / fps) if fps > 0 else 0.0
    return total_frames, fps, duration_sec


def _keyword_cache_path(cfg: OursHeatmapConfig, sample_id: str, task_type: str) -> Path:
    cache_root = resolve_keyword_cache_root(cfg.keyword_cache_dir or None)
    run_dir = keyword_cache_run_dir(
        cache_root,
        dataset=cfg.dataset,
        task_type=task_type,
        extractor_model=cfg.keyword_extractor_model,
        prompt_version=cfg.keyword_prompt_version,
        target_keywords=cfg.max_keywords,
        cache_number=cfg.keyword_cache_number,
    )
    return keyword_cache_file(run_dir, sample_id)


def load_cached_keywords(
    cfg: OursHeatmapConfig,
    sample_id: str,
    task_type: str,
) -> tuple[list[str], list[str]]:
    cache_path = _keyword_cache_path(cfg, sample_id, task_type)
    cached = load_keyword_cache_entry(cache_path, sample_id=sample_id)
    if cached is None:
        raise RuntimeError(
            f"关键词缓存未命中: sample={sample_id}, path={cache_path}。"
            "请先对目标样本运行 vqa_eval_ours 并开启 --use_keyword_cache。"
        )
    return cached


def _uniform_frame_ids(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    return [int(i * total_frames / num_frames) for i in range(num_frames)]


def _build_ours_args(cfg: OursHeatmapConfig, keyword_weight_strength: float) -> SimpleNamespace:
    return SimpleNamespace(
        keyword_weight_strength=float(keyword_weight_strength),
        max_keywords=int(cfg.max_keywords),
    )


def build_sample_bundle(
    sample: VQASample,
    cfg: OursHeatmapConfig,
    visual_encoder: VisualEncoder,
    preprocessed_clip_dir: str | None,
) -> SampleBundle:
    sample_id = sample.resolve_preprocess_key()
    _kws_raw, kws = load_cached_keywords(cfg, sample_id, str(sample.task_type))
    if not kws:
        raise RuntimeError(f"关键词缓存为空: sample={sample_id}")

    if cfg.use_preprocessed_clip_frames:
        frame_ids, imgs = load_preprocessed_candidate_frames(preprocessed_clip_dir or "", sample_id)
        keep = pool_positions_at_fps(len(imgs), cfg.preprocessed_clip_fps, cfg.candidate_pool_fps)
        if keep:
            frame_ids = [frame_ids[i] for i in keep]
            imgs = [imgs[i] for i in keep]
    else:
        from utils import collect_video_frames_at_fps

        frame_ids, imgs = collect_video_frames_at_fps(sample.video_path, cfg.candidate_pool_fps)

    if not imgs:
        raise RuntimeError(f"候选帧为空: sample={sample_id}")

    kw_emb = _encode_texts_for_dedup(visual_encoder, kws, 32)
    kws_rep, _kw_emb_rep, _ = _merge_keywords(kws, kw_emb, cfg.max_keywords)
    kw_frame_sims = _compute_kw_frame_sims(
        visual_encoder,
        imgs,
        kws_rep,
        cfg.ours_clip_batch_size,
    )
    if kw_frame_sims.shape[0] == 0:
        raise RuntimeError(f"关键词-帧相似度为空: sample={sample_id}")

    return SampleBundle(frame_ids=frame_ids, kw_frame_sims=kw_frame_sims, kws_rep=kws_rep)


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


def format_allocation_list(values: list[float]) -> str:
    parts: list[str] = []
    for v in values:
        if abs(v - round(v)) < 1e-6:
            parts.append(str(int(round(v))))
        else:
            parts.append(f"{v:.1f}")
    return "[" + ", ".join(parts) + "]"


def mean_allocation_by_rank(samples_counts: list[list[int]]) -> list[float]:
    if not samples_counts:
        return []
    max_rank = max(len(row) for row in samples_counts)
    out: list[float] = []
    for r in range(max_rank):
        vals = [row[r] for row in samples_counts if r < len(row)]
        out.append(float(np.mean(vals)) if vals else 0.0)
    return out


def build_allocation_lines(mean_by_ws: dict[float, list[float]]) -> list[str]:
    lines: list[str] = []
    for ws in sorted(mean_by_ws.keys()):
        lines.append(f"λ={ws:g}: {format_allocation_list(mean_by_ws[ws])}")
    return lines


def select_ours_selection(
    bundle: SampleBundle,
    cfg: OursHeatmapConfig,
    *,
    num_frames: int,
    keyword_weight_strength: float,
) -> tuple[list[int], list[int], list[str]]:
    """返回 (选中 frame_id, 按权重降序的每词帧数, 按权重降序的关键词)。"""
    args = _build_ours_args(cfg, keyword_weight_strength)
    info_pack = _compute_keyword_information(bundle.kws_rep, bundle.kw_frame_sims, args)
    kw_weights: torch.Tensor = info_pack["kw_weights"]
    n_frames = int(bundle.kw_frame_sims.shape[1])
    selected_idx, keyword_counts, _quotas = quota_topk_select_with_counts(
        kw_sims=bundle.kw_frame_sims,
        kw_w=kw_weights,
        budget=int(num_frames),
        candidate_idx=list(range(n_frames)),
    )
    if not selected_idx:
        return [], [], []
    order = sorted(
        range(len(bundle.kws_rep)),
        key=lambda i: float(kw_weights[i].item()),
        reverse=True,
    )
    counts_by_rank = [int(keyword_counts[j]) for j in order]
    kws_by_rank = [bundle.kws_rep[j] for j in order]
    selected_idx = sorted(selected_idx, key=lambda x: bundle.frame_ids[x])
    frame_ids = [int(bundle.frame_ids[i]) for i in selected_idx]
    return frame_ids, counts_by_rank, kws_by_rank


def select_ours_frame_ids(
    bundle: SampleBundle,
    cfg: OursHeatmapConfig,
    *,
    num_frames: int,
    keyword_weight_strength: float,
) -> list[int]:
    frame_ids, _counts, _kws = select_ours_selection(
        bundle,
        cfg,
        num_frames=num_frames,
        keyword_weight_strength=keyword_weight_strength,
    )
    return frame_ids


def frame_ids_to_time_bins(frame_ids: list[int], total_frames: int, num_bins: int) -> np.ndarray:
    mat = np.zeros(num_bins, dtype=np.float32)
    if total_frames <= 0 or not frame_ids:
        return mat
    denom = max(total_frames - 1, 1)
    for fid in frame_ids:
        fid_clamped = int(np.clip(fid, 0, total_frames - 1))
        bin_idx = int(round(fid_clamped / denom * (num_bins - 1)))
        mat[bin_idx] = 1.0
    return mat


def build_heatmap_matrix(
    selections: dict[str, list[int]],
    row_order: list[str],
    total_frames: int,
    num_bins: int,
) -> np.ndarray:
    mat = np.zeros((len(row_order), num_bins), dtype=np.float32)
    for row, label in enumerate(row_order):
        mat[row] = frame_ids_to_time_bins(selections.get(label, []), total_frames, num_bins)
    return mat


def plot_sample_heatmap(
    mat: np.ndarray,
    row_labels: list[str],
    duration_sec: float,
    sample: VQASample,
    num_frames: int,
    cfg: OursHeatmapConfig,
    out_path: Path,
    *,
    allocation_lines: list[str] | None = None,
) -> None:
    n_alloc_lines = len(allocation_lines or [])
    alloc_h = 0.22 * n_alloc_lines if n_alloc_lines else 0.0
    fig_h = max(4.5, 0.55 * len(row_labels) + 1.5 + alloc_h)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="binary", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Config")
    title_q = sample.question.strip().replace("\n", " ")
    if len(title_q) > 90:
        title_q = title_q[:87] + "..."
    ws_grid = ",".join(f"{x:g}" for x in cfg.keyword_weight_strength_grid)
    ax.set_title(
        f"Ours frame selection heatmap | {sample.sample_id} | {num_frames}f\n"
        f"task={sample.task_type} | duration={duration_sec:.1f}s | "
        f"max_kw={cfg.max_keywords} | λ∈[{ws_grid}]\n{title_q}",
        fontsize=10,
    )
    if duration_sec > 0:
        tick_pos = np.linspace(0, mat.shape[1] - 1, 6)
        tick_labels = [f"{t * duration_sec / max(mat.shape[1] - 1, 1):.0f}" for t in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["not selected", "selected"])
    if allocation_lines:
        alloc_text = "Mean frames/keyword (weight rank →):\n" + "\n".join(allocation_lines)
        fig.text(
            0.02,
            0.02,
            alloc_text,
            ha="left",
            va="bottom",
            fontsize=7.5,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bottom = min(0.12 + 0.035 * n_alloc_lines, 0.45) if allocation_lines else 0.08
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _build_row_specs(cfg: OursHeatmapConfig) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if cfg.include_uniform_baseline:
        rows.append(("uniform", "uniform"))
    for ws in cfg.keyword_weight_strength_grid:
        key = f"ours_ws={ws:g}"
        rows.append((key, f"ours λ={ws:g}"))
    return rows


def run_ours_frame_selection_heatmap(cfg: OursHeatmapConfig) -> dict[str, Any]:
    """入口：按配置生成 Ours 选帧热力图与 selections 记录。"""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = cfg.out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_clip_dir = None
    if cfg.use_preprocessed_clip_frames:
        preprocessed_clip_dir = resolve_preprocessed_clip_dir(
            cfg.dataset, cfg.preprocessed_clip_fps, ""
        )

    loader = get_data_loader(cfg.dataset, seed=cfg.seed, task_filter=cfg.task_filter)
    all_samples = loader._convert_all("test")
    if len(all_samples) < cfg.num_samples:
        raise RuntimeError(f"样本不足: {len(all_samples)} < {cfg.num_samples}")
    rng = random.Random(cfg.seed)
    samples = rng.sample(all_samples, cfg.num_samples)

    samples_meta = [
        {
            "sample_id": s.sample_id,
            "task_type": s.task_type,
            "video_path": s.video_path,
            "question": s.question,
        }
        for s in samples
    ]
    (cfg.out_dir / "selected_samples.json").write_text(
        json.dumps(samples_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[017] 加载视觉编码器...")
    visual_encoder = _load_visual_encoder(cfg.visual_encoder_model, None)

    bundles: dict[str, SampleBundle] = {}
    for sample in tqdm(samples, desc="017 预计算"):
        sid = str(sample.sample_id)
        bundles[sid] = build_sample_bundle(sample, cfg, visual_encoder, preprocessed_clip_dir)

    row_specs = _build_row_specs(cfg)
    row_labels = [label for _, label in row_specs]
    selections_records: list[dict[str, Any]] = []
    alloc_by_nf_ws: dict[int, dict[float, list[list[int]]]] = {
        nf: {float(ws): [] for ws in cfg.keyword_weight_strength_grid}
        for nf in cfg.num_frames_grid
    }
    plot_jobs: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="017 选帧"):
        sid = str(sample.sample_id)
        bundle = bundles[sid]
        total_frames, fps, duration_sec = get_video_meta(sample.video_path)

        for num_frames in cfg.num_frames_grid:
            selections: dict[str, list[int]] = {}
            for row_key, row_label in row_specs:
                if row_key == "uniform":
                    frame_ids: list[int] = _uniform_frame_ids(total_frames, num_frames)
                    counts_by_rank: list[int] | None = None
                    kws_by_rank: list[str] | None = None
                    ws_val = None
                else:
                    ws_val = float(row_key.split("=", 1)[1])
                    frame_ids, counts_by_rank, kws_by_rank = select_ours_selection(
                        bundle,
                        cfg,
                        num_frames=num_frames,
                        keyword_weight_strength=ws_val,
                    )
                    alloc_by_nf_ws[num_frames][ws_val].append(counts_by_rank)
                selections[row_label] = frame_ids
                selections_records.append(
                    {
                        "sample_id": sample.sample_id,
                        "task_type": sample.task_type,
                        "num_frames": num_frames,
                        "method": row_label,
                        "row_key": row_key,
                        "keyword_weight_strength": ws_val,
                        "selected_frame_ids": frame_ids,
                        "keyword_frames_by_rank": counts_by_rank,
                        "keywords_by_rank": kws_by_rank,
                        "total_frames": total_frames,
                        "fps": fps,
                        "duration_sec": duration_sec,
                    }
                )

            plot_jobs.append(
                {
                    "sample": sample,
                    "num_frames": num_frames,
                    "selections": selections,
                    "total_frames": total_frames,
                    "duration_sec": duration_sec,
                }
            )

    allocation_summary: dict[str, Any] = {}
    allocation_lines_by_nf: dict[int, list[str]] = {}
    for num_frames in cfg.num_frames_grid:
        mean_by_ws = {
            ws: mean_allocation_by_rank(alloc_by_nf_ws[num_frames][ws])
            for ws in cfg.keyword_weight_strength_grid
        }
        lines = build_allocation_lines(mean_by_ws)
        allocation_lines_by_nf[num_frames] = lines
        nf_key = f"{num_frames}f"
        allocation_summary[nf_key] = {
            "num_frames": num_frames,
            "mean_frames_by_weight_rank": {
                f"{ws:g}": mean_by_ws[ws] for ws in cfg.keyword_weight_strength_grid
            },
            "text_lines": lines,
        }
        txt_path = cfg.out_dir / f"allocation_summary_{nf_key}.txt"
        txt_path.write_text(
            f"# Mean frames per keyword (sorted by weight, high → low), n={cfg.num_samples}\n"
            + "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )

    for job in tqdm(plot_jobs, desc="017 绘图"):
        sample = job["sample"]
        num_frames = int(job["num_frames"])
        mat = build_heatmap_matrix(
            job["selections"],
            row_labels,
            int(job["total_frames"]),
            cfg.num_time_bins,
        )
        safe_id = str(sample.sample_id).replace("/", "_")
        out_path = heatmap_dir / f"{safe_id}_{num_frames}f.png"
        plot_sample_heatmap(
            mat,
            row_labels,
            float(job["duration_sec"]),
            sample,
            num_frames,
            cfg,
            out_path,
            allocation_lines=allocation_lines_by_nf.get(num_frames),
        )

    with (cfg.out_dir / "selections.jsonl").open("w", encoding="utf-8") as f:
        for rec in selections_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        **asdict(cfg),
        "out_dir": str(cfg.out_dir),
        "preprocessed_clip_dir": preprocessed_clip_dir,
        "row_labels": row_labels,
        "allocation_summary": allocation_summary,
        "num_heatmaps": len(list(heatmap_dir.glob("*.png"))),
    }
    (cfg.out_dir / "allocation_summary.json").write_text(
        json.dumps(allocation_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cfg.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"[017] 热力图目录: {heatmap_dir}")
    return summary


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_cli() -> OursHeatmapConfig:
    p = argparse.ArgumentParser(description="017: Ours 选帧时间轴热力图")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=str, default="videomme")
    p.add_argument("--task_filter", type=str, default="all")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--num_frames_grid", type=str, default="16,32", help="逗号分隔，如 16,32")
    p.add_argument(
        "--keyword_weight_strength_grid",
        type=str,
        default="0,1,2,4,8",
        help="逗号分隔的 λ 网格，对应 w_i=φ_i^λ/Σφ_j^λ",
    )
    p.add_argument("--max_keywords", type=int, default=10)
    p.add_argument("--keyword_prompt_version", type=int, default=0)
    p.add_argument("--keyword_extractor_model", type=str, default="poe-gpt-5.2")
    p.add_argument("--keyword_cache_number", type=int, default=0)
    p.add_argument("--keyword_cache_dir", type=str, default="")
    p.add_argument("--candidate_pool_fps", type=float, default=1.0)
    p.add_argument("--preprocessed_clip_fps", type=float, default=1.0)
    p.add_argument("--visual_encoder_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--ours_clip_batch_size", type=int, default=16)
    p.add_argument("--num_time_bins", type=int, default=200)
    p.add_argument("--no_uniform_baseline", action="store_true", help="不在热力图中加入 uniform 对照行")
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = p.parse_args()
    return OursHeatmapConfig(
        seed=args.seed,
        dataset=args.dataset,
        task_filter=args.task_filter,
        num_samples=args.num_samples,
        num_frames_grid=_parse_int_list(args.num_frames_grid),
        keyword_weight_strength_grid=_parse_float_list(args.keyword_weight_strength_grid),
        max_keywords=args.max_keywords,
        keyword_prompt_version=args.keyword_prompt_version,
        keyword_extractor_model=args.keyword_extractor_model,
        keyword_cache_number=args.keyword_cache_number,
        keyword_cache_dir=args.keyword_cache_dir,
        candidate_pool_fps=args.candidate_pool_fps,
        preprocessed_clip_fps=args.preprocessed_clip_fps,
        visual_encoder_model=args.visual_encoder_model,
        ours_clip_batch_size=args.ours_clip_batch_size,
        num_time_bins=args.num_time_bins,
        include_uniform_baseline=not args.no_uniform_baseline,
        out_dir=Path(args.out_dir),
    )


def main() -> None:
    cfg = parse_cli()
    run_ours_frame_selection_heatmap(cfg)


if __name__ == "__main__":
    main()
