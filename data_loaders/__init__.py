from __future__ import annotations

from .base import BaseDataLoader, VQASample
from .egoschema import EgoSchemaLoader
from .nextqa import NextQALoader
from .vsibench import VSIBenchLoader


LOADER_REGISTRY: dict[str, type[BaseDataLoader]] = {
    "egoschema": EgoSchemaLoader,
    "nextqa": NextQALoader,
    "vsibench": VSIBenchLoader,
}


def get_data_loader(dataset: str, **kwargs) -> BaseDataLoader:
    key = dataset.strip().lower()
    if key not in LOADER_REGISTRY:
        raise ValueError(f"不支持的数据集: {dataset}。可选: {', '.join(sorted(LOADER_REGISTRY.keys()))}")
    return LOADER_REGISTRY[key](**kwargs)


def list_supported_datasets() -> list[str]:
    return sorted(LOADER_REGISTRY.keys())

