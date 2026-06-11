from __future__ import annotations

from typing import Any

from .base import BaseDataLoader, VQASample
from .egoschema import EgoSchemaLoader
from .mlvu import MLVULoader
from .nextqa import NextQALoader
from .videomme import VideoMMELoader
from .vsibench import VSIBenchLoader


LOADER_REGISTRY: dict[str, type[BaseDataLoader]] = {
    "egoschema": EgoSchemaLoader,
    "mlvu": MLVULoader,
    "nextqa": NextQALoader,
    "videomme": VideoMMELoader,
    "vsibench": VSIBenchLoader,
}


def get_data_loader(dataset: str, **kwargs) -> BaseDataLoader:
    key = dataset.strip().lower()
    if key not in LOADER_REGISTRY:
        raise ValueError(f"不支持的数据集: {dataset}。可选: {', '.join(sorted(LOADER_REGISTRY.keys()))}")
    return LOADER_REGISTRY[key](**kwargs)


def list_supported_datasets() -> list[str]:
    return sorted(LOADER_REGISTRY.keys())


def dataset_uses_vl_pixel_limits(
    dataset: str,
    dataset_split: str = "",
    dataset_name: str = "",
) -> bool:
    """MLVU-Test 含超高分辨率 outlier 视频，需限制 processor 单帧像素防 OOM。"""
    key = dataset.strip().lower()
    split = dataset_split.strip().lower()
    name = dataset_name.strip().lower()
    if key == "mlvu" and split == "test":
        return True
    return "mlvu_test" in name.replace("-", "_")


def apply_dataset_cli_defaults(args: Any) -> None:
    """当用户只改 --dataset 时，自动补齐常用 video_dir / dataset_name / split。"""
    key = str(getattr(args, "dataset", "")).strip().lower()
    if key == "mlvu":
        if getattr(args, "video_dir", "") in {"~/dataset/vsi_bench", "~/dataset/Video-MME"}:
            args.video_dir = "~/dataset/mlvu_test"
        if getattr(args, "dataset_name", "") in {"nyu-visionx/VSI-Bench", "lmms-lab/Video-MME"}:
            args.dataset_name = "MLVU/MLVU_Test"
        if getattr(args, "dataset_split", "") == "test" or not getattr(args, "dataset_split", ""):
            args.dataset_split = "test"
        args.no_dataset_config = True
    elif key == "videomme":
        if getattr(args, "video_dir", "") == "~/dataset/vsi_bench":
            args.video_dir = "~/dataset/videomme"
        if getattr(args, "dataset_name", "") == "nyu-visionx/VSI-Bench":
            args.dataset_name = "lmms-lab/Video-MME"
        args.no_dataset_config = True

