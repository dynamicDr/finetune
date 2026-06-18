"""014: topk 消融实验选帧重合度 8×8 热力图（轻量化，只读 verbose 索引）。"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
VERBOSE_GLOB = "20260617_*"
OUT_DIR = Path(__file__).resolve().parent / "014_outputs"
NUM_FRAMES = 16
NUM_SAMPLES: int | None = None  # None = 全量（各配置公共样本）
SEED = 42

# note.txt topk 消融: ws=0,1,2,4 × ensure_keyword_min_coverage=false,true
WS_GRID = [0, 1, 2, 4]
COV_GRID = [False, True]


def config_key(ws: float, cov: bool) -> tuple[float, bool]:
    return (float(ws), bool(cov))


def config_label(ws: float, cov: bool) -> str:
    ws_s = str(int(ws)) if float(ws).is_integer() else str(ws)
    return f"ws={ws_s}\ncov={'T' if cov else 'F'}"


def overlap_ratio(ids_a: set[int], ids_b: set[int]) -> float:
    if not ids_a and not ids_b:
        return 1.0
    return len(ids_a & ids_b) / float(NUM_FRAMES)


def load_selections(verbose_root: Path) -> dict[tuple[float, bool], dict[str, set[int]]]:
    """返回 {(ws, cov): {sample_id: frame_id_set}}。"""
    by_cfg: dict[tuple[float, bool], dict[str, set[int]]] = defaultdict(dict)
    run_dirs = sorted(verbose_root.glob(VERBOSE_GLOB))
    if not run_dirs:
        raise FileNotFoundError(f"未找到 verbose 目录: {verbose_root / VERBOSE_GLOB}")

    for run_dir in run_dirs:
        index_path = run_dir / "selected_frames_index.jsonl"
        if not index_path.is_file():
            continue
        with index_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if int(rec.get("frame_selection_mode", -1)) != 0:
                    continue
                sc = rec.get("selection_config") or {}
                ws = sc.get("keyword_weight_strength")
                cov = sc.get("ensure_keyword_min_coverage")
                if ws is None or cov is None:
                    meta_path = run_dir / "selected_frames_index.meta.json"
                    if meta_path.is_file():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        ws = meta.get("keyword_weight_strength") if ws is None else ws
                        cov = meta.get("ensure_keyword_min_coverage") if cov is None else cov
                if ws is None or cov is None:
                    continue
                key = config_key(float(ws), bool(cov))
                sample_id = str(rec["sample_id"])
                frame_ids = {int(x) for x in rec.get("selected_frame_ids", [])}
                by_cfg[key][sample_id] = frame_ids
    return dict(by_cfg)


def ordered_configs(by_cfg: dict[tuple[float, bool], dict[str, set[int]]]) -> list[tuple[float, bool]]:
    expected = [config_key(ws, cov) for ws in WS_GRID for cov in COV_GRID]
    missing = [k for k in expected if k not in by_cfg]
    if missing:
        miss_s = ", ".join(f"ws={ws},cov={cov}" for ws, cov in missing)
        raise RuntimeError(f"缺少完整配置: {miss_s}")
    for key in expected:
        n = len(by_cfg[key])
        if n < 900:
            print(f"[warn] {key} 仅 {n} 条样本（期望 900）")
    return expected


def pick_sample_ids(
    by_cfg: dict[tuple[float, bool], dict[str, set[int]]],
    configs: list[tuple[float, bool]],
    n: int | None,
    seed: int,
) -> list[str]:
    common = set.intersection(*(set(by_cfg[c].keys()) for c in configs))
    if not common:
        raise RuntimeError("各配置无公共样本")
    if n is None:
        return sorted(common)
    if len(common) < n:
        raise RuntimeError(f"公共样本仅 {len(common)} 个，不足 {n}")
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(common), n))


def build_overlap_matrix(
    by_cfg: dict[tuple[float, bool], dict[str, set[int]]],
    configs: list[tuple[float, bool]],
    sample_ids: list[str],
) -> np.ndarray:
    n_cfg = len(configs)
    mat = np.zeros((n_cfg, n_cfg), dtype=np.float64)
    for i, ci in enumerate(configs):
        for j, cj in enumerate(configs):
            vals = [
                overlap_ratio(by_cfg[ci][sid], by_cfg[cj][sid])
                for sid in sample_ids
            ]
            mat[i, j] = float(np.mean(vals))
    return mat


def plot_heatmap(mat: np.ndarray, labels: list[str], out_path: Path, sample_ids: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="YlOrRd")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Config")
    ax.set_ylabel("Config")
    ax.set_title(
        f"TopK frame overlap (n={len(sample_ids)} avg, |cap|/{NUM_FRAMES})\n"
        f"mode=0, Video-MME medium, ws x min_kw_cover ablation"
    )
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean overlap ratio")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    # 附带记录所选 20 题
    (out_path.parent / "selected_sample_ids.txt").write_text(
        "\n".join(sample_ids) + "\n", encoding="utf-8"
    )


def main() -> None:
    verbose_root = ROOT_DIR / "verbose_eval_ours"
    by_cfg = load_selections(verbose_root)
    configs = ordered_configs(by_cfg)
    labels = [config_label(ws, cov) for ws, cov in configs]
    sample_ids = pick_sample_ids(by_cfg, configs, NUM_SAMPLES, SEED)
    mat = build_overlap_matrix(by_cfg, configs, sample_ids)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT_DIR / "overlap_matrix.csv", mat, delimiter=",", fmt="%.4f")
    with (OUT_DIR / "overlap_matrix_labels.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": labels, "sample_ids": sample_ids}, f, ensure_ascii=False, indent=2)

    plot_heatmap(mat, labels, OUT_DIR / "frame_overlap_heatmap.png", sample_ids)

    print(f"配置数: {len(configs)}")
    print(f"样本数: {len(sample_ids)}（全量公共集）")
    print(f"矩阵均值(非对角): {mat[~np.eye(len(configs), dtype=bool)].mean():.3f}")
    print(f"输出: {OUT_DIR / 'frame_overlap_heatmap.png'}")


if __name__ == "__main__":
    main()
