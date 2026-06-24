from __future__ import annotations

import os
from typing import Any

from .base import BaseDataLoader, VQASample
from .egoschema import EgoSchemaLoader
from .mlvu import MLVULoader
from .nextqa import NextQALoader
from .shared_paths import SHARED_DATASET_ROOT
from .videomme import VideoMMELoader
from .vsibench import VSIBenchLoader


LOADER_REGISTRY: dict[str, type[BaseDataLoader]] = {
    "egoschema": EgoSchemaLoader,
    "mlvu": MLVULoader,
    "nextqa": NextQALoader,
    "videomme": VideoMMELoader,
    "vsibench": VSIBenchLoader,
}

# 各数据集默认根目录（共享 duanty 数据，可用 FINETUNE_SHARED_DATASET_ROOT 覆盖）
DEFAULT_DATASET_ROOTS: dict[str, str] = {
    "egoschema": str(SHARED_DATASET_ROOT / "egoschema"),
    "mlvu": str(SHARED_DATASET_ROOT / "mlvu_test"),
    "nextqa": str(SHARED_DATASET_ROOT / "nextqa"),
    "videomme": str(SHARED_DATASET_ROOT / "Video-MME"),
    "vsibench": str(SHARED_DATASET_ROOT / "vsi_bench"),
}


def resolve_dataset_root(dataset: str) -> str:
    key = dataset.strip().lower()
    if key not in DEFAULT_DATASET_ROOTS:
        raise ValueError(
            f"未配置数据集根目录: {dataset}。可选: {', '.join(sorted(DEFAULT_DATASET_ROOTS))}"
        )
    return os.path.expanduser(DEFAULT_DATASET_ROOTS[key])


def get_data_loader(dataset: str, **kwargs) -> BaseDataLoader:
    key = dataset.strip().lower()
    if key not in LOADER_REGISTRY:
        raise ValueError(f"不支持的数据集: {dataset}。可选: {', '.join(sorted(LOADER_REGISTRY.keys()))}")
    kwargs.setdefault("video_dir", resolve_dataset_root(key))
    return LOADER_REGISTRY[key](**kwargs)


def list_supported_datasets() -> list[str]:
    return sorted(LOADER_REGISTRY.keys())


def dataset_uses_vl_pixel_limits(dataset: str) -> bool:
    """MLVU-Test 含超高分辨率 outlier 视频，需限制 processor 单帧像素防 OOM。"""
    return dataset.strip().lower() == "mlvu"


def should_apply_vl_pixel_limits(model_id: str, dataset: str = "", **_: Any) -> bool:
    """数据集或模型任一需要像素限制时启用（如 MLVU outlier、LLaVA anyres 多帧超 context）。"""
    from lmms_eval_bridge import model_uses_vl_pixel_limits

    return dataset_uses_vl_pixel_limits(dataset) or model_uses_vl_pixel_limits(model_id)


def apply_dataset_cli_defaults(args: Any) -> None:
    """兼容旧脚本：video_dir / dataset_name 等已由各 Loader 内置，此处不再改写。"""
    return

