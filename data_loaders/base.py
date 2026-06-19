from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
import sys
from pathlib import Path
from typing import Any

from lmms_eval_bridge import split_indices


def load_dataset(*args, **kwargs):
    """
    Robust HuggingFace datasets loader.

    Avoid importing any local `datasets/` folder in this repo by temporarily
    removing repo root from `sys.path` before importing HuggingFace `datasets`.
    """
    repo_root = str(Path(__file__).resolve().parents[1])
    removed = False
    if repo_root in sys.path:
        sys.path.remove(repo_root)
        removed = True
    try:
        from datasets import load_dataset as hf_load_dataset
    finally:
        if removed:
            sys.path.insert(0, repo_root)
    return hf_load_dataset(*args, **kwargs)


@dataclass
class VQASample:
    sample_id: str
    video_path: str
    question: str
    answer: str
    options: list[str] | None = None
    task_type: str = "mcq"
    metadata: dict[str, Any] = field(default_factory=dict)
    preprocess_key: str | None = None

    def resolve_preprocess_key(self) -> str:
        return self.preprocess_key or self.sample_id


def sample_matches_task_filter(sample: VQASample, task_filter: str) -> bool:
    """与各 DataLoader._include_by_task 一致：mcq=有选项，generation/numeric=无选项，其余按 task_type。"""
    tf = str(task_filter).strip()
    if tf == "all":
        return True
    if tf == "mcq":
        return sample.options is not None
    if tf in {"generation", "numeric"}:
        return sample.options is None
    return sample.task_type == tf


class BaseDataLoader(ABC):
    def __init__(
        self,
        video_dir: str,
        seed: int = 42,
        train_ratio: float = 0.8,
        task_filter: str = "all",
    ):
        self.video_dir = video_dir
        self.seed = seed
        self.train_ratio = train_ratio
        self.task_filter = task_filter

    @abstractmethod
    def load_raw_dataset(self, split: str):
        """Return a raw dataset object for a given split."""

    @abstractmethod
    def to_vqa_sample(self, raw_sample: Any, index: int) -> VQASample | None:
        """Convert one raw row to unified VQASample; return None to skip."""

    def _include_by_task(self, sample: VQASample) -> bool:
        return sample_matches_task_filter(sample, self.task_filter)

    def _convert_all(self, split: str) -> list[VQASample]:
        dataset = self.load_raw_dataset(split)
        out: list[VQASample] = []
        for i in range(len(dataset)):
            sample = self.to_vqa_sample(dataset[i], i)
            if sample is None:
                continue
            if not self._include_by_task(sample):
                continue
            out.append(sample)
        return out

    def get_eval_samples(
        self,
        sample_count: int | None = None,
        sample_seed_offset: int = 1000,
    ) -> list[VQASample]:
        """加载 test split；sample_count 为 None 时取全部，否则随机抽 num_samples 条。"""
        samples = self._convert_all("test")
        if sample_count is not None:
            random.seed(self.seed + sample_seed_offset)
            k = min(sample_count, len(samples))
            samples = random.sample(samples, k)
        return samples

    def get_split_samples(
        self,
        split: str,
        use_train_split: bool,
        max_samples: int | None = None,
        sample_count: int | None = None,
        sample_seed_offset: int = 1000,
    ) -> list[VQASample]:
        samples = self._convert_all(split)
        indices = split_indices(
            list(range(len(samples))),
            seed=self.seed,
            train_ratio=self.train_ratio,
            use_train_split=use_train_split,
        )
        selected = [samples[i] for i in indices]

        if max_samples is not None:
            selected = selected[:max_samples]

        if sample_count is not None:
            random.seed(self.seed + sample_seed_offset)
            k = min(sample_count, len(selected))
            selected = random.sample(selected, k)
        return selected
