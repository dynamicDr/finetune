from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_loaders import apply_dataset_cli_defaults, get_data_loader
from model_response_mode import load_model_response_mode_config, resolve_model_mode
from utils import build_user_text
from vl_common import load_model_and_processor
from vqa_eval_ours import (
    MODE_MAX_NEW_TOKENS,
    _build_quota_prescreen_candidates,
    _collect_video_frames_at_fps,
    _encode_images,
    _encode_texts,
    _keyword_cache_file,
    _keyword_cache_run_dir,
    _load_clip,
    _load_keyword_cache_entry,
    _local_evidence_score,
    _merge_keywords,
    _quota_topk_select,
    _resolve_keyword_cache_root,
    _run_vlm_once,
    _submodular_cover_greedy_select,
)


# =========================
# 固定配置（无 args）
# =========================
SEED = 42
DATASET = "videomme"
DATASET_SPLIT = "test"
TASK_FILTER = "all"
VIDEO_DIR = "/userhome/cs3/duanty/dataset/Video-MME"
NUM_SAMPLES = 10

MODEL_PATH = "Qwen/Qwen3-VL-4B-Thinking"
MODEL_MODE_CONFIG = "config/model_response_modes.json"
OURS_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
OURS_CLIP_DEVICE = None
OURS_CLIP_BATCH_SIZE = 16

NUM_FRAMES_BUDGET = 16
CANDIDATE_POOL_FPS = 1.0
KEYWORD_PROMPT_VERSION = 0
KEYWORD_EXTRACTOR_MODEL = "poe-gpt-5.2"
KEYWORD_CACHE_DIR = ""
KEYWORD_CACHE_NUMBER = 0

# 3 x 3 消融网格（max_keywords 固定为 10）
MAX_KEYWORDS = 10
WEIGHT_STRENGTH_GRID = [2.0, 4.0, 8.0]
QUOTA_PRESCREEN_ALPHA_GRID = [3, 2, 1]

OUT_DIR = Path("/userhome/cs3/duanty/finetune/sandbox/013_outputs").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_keyword_weights(
    kw_emb_rep: torch.Tensor,
    img_emb: torch.Tensor,
    keyword_weight_strength: float,
) -> torch.Tensor:
    """与 eval_ours 同形公式，但不把 strength 截断到 1.0（支持 2/4/8 消融）。"""
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
    strength = float(keyword_weight_strength)
    kw_weights = ((1.0 - strength) * uniform_weights + strength * info_weights).clamp(min=0.0)
    ws = float(kw_weights.sum().item())
    if ws <= 1e-12:
        return uniform_weights
    return kw_weights / ws


def same_frame_stats(cov_ids: list[int], topk_ids: list[int]) -> dict[str, float | int]:
    cov_set = set(cov_ids)
    topk_set = set(topk_ids)
    inter = cov_set & topk_set
    uni = cov_set | topk_set
    overlap_count = len(inter)
    denom = max(len(cov_set), len(topk_set), 1)
    return {
        "overlap_count": overlap_count,
        "same_frame_percent": float(overlap_count / denom * 100.0),
        "jaccard": float(overlap_count / len(uni)) if uni else 0.0,
        "cov_only_count": len(cov_set - topk_set),
        "topk_only_count": len(topk_set - cov_set),
    }


def is_correct(pred: str, gt: str) -> int:
    return int(str(pred).strip().upper() == str(gt).strip().upper())


def keyword_cache_path(sample_id: str, task_type: str) -> Path:
    cache_root = _resolve_keyword_cache_root(KEYWORD_CACHE_DIR)
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


def load_cached_keywords_only(sample_id: str, task_type: str) -> tuple[list[str], list[str]]:
    cache_path = keyword_cache_path(sample_id, task_type)
    cached = _load_keyword_cache_entry(cache_path, sample_id=sample_id)
    if cached is None:
        raise RuntimeError(
            "关键词缓存未命中，本脚本禁止调用 API/LLM 抽词。"
            f" sample={sample_id}, cache={cache_path}"
        )
    return cached


random.seed(SEED)

loader_args = SimpleNamespace(
    dataset=DATASET,
    dataset_split=DATASET_SPLIT,
    dataset_name="nyu-visionx/VSI-Bench",
    dataset_config="full",
    no_dataset_config=False,
    video_dir=VIDEO_DIR,
)
apply_dataset_cli_defaults(loader_args)

print("[013] 加载数据集...")
loader = get_data_loader(
    loader_args.dataset,
    video_dir=loader_args.video_dir,
    seed=SEED,
    train_ratio=0.8,
    task_filter=TASK_FILTER,
    dataset_name=loader_args.dataset_name,
    dataset_config=loader_args.dataset_config,
    no_dataset_config=loader_args.no_dataset_config,
)
samples = loader.get_split_samples(
    split=loader_args.dataset_split,
    use_train_split=False,
    sample_count=None,
)
if len(samples) < NUM_SAMPLES:
    raise RuntimeError(f"数据集样本不足 {NUM_SAMPLES}，当前仅 {len(samples)}")
samples = samples[:NUM_SAMPLES]
print(f"[013] 样本数: {len(samples)}")

print("[013] 检查关键词缓存（仅缓存，禁止 API）...")
missing_cache: list[str] = []
for sample in samples:
    sid = str(sample.sample_id)
    cache_path = keyword_cache_path(sid, str(sample.task_type))
    if _load_keyword_cache_entry(cache_path, sample_id=sid) is None:
        missing_cache.append(f"  {sid} -> {cache_path}")
if missing_cache:
    raise RuntimeError(
        "以下样本缺少关键词缓存，请先跑 eval_ours 生成缓存后再执行本脚本:\n"
        + "\n".join(missing_cache)
    )
print(f"[013] 关键词缓存检查通过: {len(samples)}/{len(samples)}")

print("[013] 加载 VLM...")
vlm_model, vlm_proc = load_model_and_processor(MODEL_PATH)
mode_cfg = load_model_response_mode_config(MODEL_MODE_CONFIG)
model_mode = ""
for mode_id in [MODEL_PATH, Path(MODEL_PATH).name]:
    try:
        model_mode, _, _ = resolve_model_mode(model_identifier=mode_id, config=mode_cfg)
        break
    except (KeyError, ValueError):
        continue
if not model_mode:
    raise RuntimeError(f"无法在 model_response_modes.json 中解析模型模式: {MODEL_PATH}")
max_new_tokens = int(MODE_MAX_NEW_TOKENS.get(model_mode, 128))
print(f"[013] model_mode={model_mode}, max_new_tokens={max_new_tokens}")

print("[013] 加载 CLIP...")
clip_proc, clip_model, clip_device = _load_clip(OURS_CLIP_MODEL_ID, OURS_CLIP_DEVICE)

# 预计算帧 / 嵌入 / 关键词（避免 9 组配置重复读视频）
sample_bundles: dict[str, dict] = {}
for sample in tqdm(samples, desc="013 预计算关键词"):
    sid = str(sample.sample_id)
    frame_ids, imgs = _collect_video_frames_at_fps(sample.video_path, CANDIDATE_POOL_FPS)
    if not imgs:
        raise RuntimeError(f"样本无可用帧: sample={sid}")

    kws_raw, kws = load_cached_keywords_only(sid, str(sample.task_type))
    if not kws:
        raise RuntimeError(f"关键词缓存为空: sample={sid}")

    kw_emb = _encode_texts(kws, clip_proc, clip_model, clip_device, 32)
    kws_rep, kw_emb_rep, _ = _merge_keywords(kws, kw_emb, MAX_KEYWORDS)
    if not kws_rep:
        raise RuntimeError(f"关键词合并后为空: sample={sid}")

    img_emb = _encode_images(imgs, clip_proc, clip_model, clip_device, OURS_CLIP_BATCH_SIZE)
    sample_bundles[sid] = {
        "sample": sample,
        "frame_ids": frame_ids,
        "imgs": imgs,
        "kws_raw": kws_raw,
        "kws_rep": kws_rep,
        "kw_emb_rep": kw_emb_rep,
        "img_emb": img_emb,
        "prompt": build_user_text(sample.question, sample.options),
        "gt_answer": str(sample.answer),
    }

valid_ids = set(sample_bundles.keys())
if not valid_ids:
    raise RuntimeError("没有有效样本，请检查数据路径与关键词提取配置。")
if len(valid_ids) < len(samples):
    raise RuntimeError(
        f"仅 {len(valid_ids)}/{len(samples)} 个样本预计算成功，"
        "请检查视频路径与帧读取是否正常。"
    )

vlm_cache: dict[tuple[str, tuple[int, ...]], str] = {}

def predict_with_frames(sid: str, imgs: list, selected_idx: list[int], prompt: str) -> str:
    key = (sid, tuple(sorted(int(i) for i in selected_idx)))
    if key in vlm_cache:
        return vlm_cache[key]
    frames = [imgs[i] for i in selected_idx]
    out = _run_vlm_once(vlm_model, vlm_proc, frames, prompt, max_new_tokens, model_mode)
    pred = str(out["pred_answer"])
    vlm_cache[key] = pred
    return pred

summary_rows: list[dict] = []
detail_rows: list[dict] = []
config_list = [
    (ws, alpha)
    for ws in WEIGHT_STRENGTH_GRID
    for alpha in QUOTA_PRESCREEN_ALPHA_GRID
]

ablation_bar = tqdm(config_list, desc="013 消融 3x3", unit="cfg")
for keyword_weight_strength, quota_prescreen_alpha in ablation_bar:
    tag = f"ws={keyword_weight_strength:g}_mk={MAX_KEYWORDS}_a={quota_prescreen_alpha}"
    ablation_bar.set_postfix(cfg=tag)

    same_percents: list[float] = []
    jaccards: list[float] = []
    cov_correct = 0
    topk_correct = 0
    n_eval = 0

    for sid in sorted(valid_ids):
        bundle = sample_bundles[sid]
        sample = bundle["sample"]
        imgs = bundle["imgs"]
        frame_ids = bundle["frame_ids"]
        kw_emb_rep = bundle["kw_emb_rep"]
        img_emb = bundle["img_emb"]
        prompt = bundle["prompt"]
        gt = bundle["gt_answer"]

        kw_weights = compute_keyword_weights(kw_emb_rep, img_emb, keyword_weight_strength)
        kw_sims = kw_emb_rep @ img_emb.T

        candidate_idx_cov = _build_quota_prescreen_candidates(
            kw_sims=kw_sims,
            kw_w=kw_weights,
            budget=NUM_FRAMES_BUDGET,
            alpha=quota_prescreen_alpha,
        )
        selected_cov_idx = _submodular_cover_greedy_select(
            kw_sims=kw_sims,
            kw_w=kw_weights,
            budget=NUM_FRAMES_BUDGET,
            candidate_idx=candidate_idx_cov,
        )
        selected_topk_idx = _quota_topk_select(
            kw_sims=kw_sims,
            kw_w=kw_weights,
            budget=NUM_FRAMES_BUDGET,
            candidate_idx=list(range(int(kw_sims.shape[1]))),
        )

        cov_ids = sorted(int(frame_ids[i]) for i in selected_cov_idx)
        topk_ids = sorted(int(frame_ids[i]) for i in selected_topk_idx)
        stats = same_frame_stats(cov_ids, topk_ids)
        same_percents.append(float(stats["same_frame_percent"]))
        jaccards.append(float(stats["jaccard"]))

        pred_cov = predict_with_frames(sid, imgs, selected_cov_idx, prompt)
        pred_topk = predict_with_frames(sid, imgs, selected_topk_idx, prompt)
        cov_ok = is_correct(pred_cov, gt)
        topk_ok = is_correct(pred_topk, gt)
        cov_correct += cov_ok
        topk_correct += topk_ok
        n_eval += 1

        detail_rows.append(
            {
                "keyword_weight_strength": keyword_weight_strength,
                "max_keywords": MAX_KEYWORDS,
                "quota_prescreen_alpha": quota_prescreen_alpha,
                "sample_id": sid,
                "task_type": str(sample.task_type),
                "same_frame_percent": stats["same_frame_percent"],
                "jaccard": stats["jaccard"],
                "overlap_count": stats["overlap_count"],
                "coverage_pred": pred_cov,
                "topk_pred": pred_topk,
                "gt_answer": gt,
                "coverage_correct": cov_ok,
                "topk_correct": topk_ok,
            }
        )

    avg_same = float(sum(same_percents) / max(1, len(same_percents)))
    avg_jaccard = float(sum(jaccards) / max(1, len(jaccards)))
    cov_acc = float(cov_correct / max(1, n_eval) * 100.0)
    topk_acc = float(topk_correct / max(1, n_eval) * 100.0)

    summary_rows.append(
        {
            "keyword_weight_strength": keyword_weight_strength,
            "max_keywords": MAX_KEYWORDS,
            "quota_prescreen_alpha": quota_prescreen_alpha,
            "num_samples": n_eval,
            "topk_alignment_percent": avg_same,
            "avg_jaccard": avg_jaccard,
            "coverage_accuracy_percent": cov_acc,
            "topk_accuracy_percent": topk_acc,
            "coverage_correct": cov_correct,
            "topk_correct": topk_correct,
        }
    )
    tqdm.write(
        f"[013] {tag} | topk对齐={avg_same:.1f}% | coverage_acc={cov_acc:.1f}% | topk_acc={topk_acc:.1f}%"
    )

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["keyword_weight_strength", "max_keywords", "quota_prescreen_alpha"]
)
detail_df = pd.DataFrame(detail_rows)

summary_csv = OUT_DIR / "ablation_summary.csv"
detail_csv = OUT_DIR / "ablation_detail.csv"
summary_txt = OUT_DIR / "ablation_summary.txt"

summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
detail_df.to_csv(detail_csv, index=False, encoding="utf-8")

lines = [
    "013 Coverage vs TopK 消融 (3x3, max_keywords=10)",
    f"num_samples={len(valid_ids)}",
    f"max_keywords={MAX_KEYWORDS}",
    f"weight_strength_grid={WEIGHT_STRENGTH_GRID}",
    f"quota_prescreen_alpha_grid={QUOTA_PRESCREEN_ALPHA_GRID}",
    "",
    "keyword_weight_strength | max_keywords | alpha | topk对齐度(%) | coverage准确率(%) | topk准确率(%) | jaccard",
    "-" * 95,
]
for _, row in summary_df.iterrows():
    lines.append(
        f"{row['keyword_weight_strength']:>6g} | {int(row['max_keywords']):>11} | "
        f"{int(row['quota_prescreen_alpha']):>5} | {row['topk_alignment_percent']:>11.1f} | "
        f"{row['coverage_accuracy_percent']:>15.1f} | {row['topk_accuracy_percent']:>13.1f} | "
        f"{row['avg_jaccard']:>7.3f}"
    )
summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n[013] ========== 消融结果汇总 ==========")
print(summary_txt.read_text(encoding="utf-8"))
print("[013] 输出文件:")
print(summary_csv)
print(detail_csv)
print(summary_txt)
