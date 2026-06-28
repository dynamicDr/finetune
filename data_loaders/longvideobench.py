from __future__ import annotations

import json
import os
import random
import re
import warnings
from pathlib import Path
from typing import Any

from .base import BaseDataLoader, VQASample, load_dataset, sample_matches_task_filter

LVB_HF_DATASET = "longvideobench/LongVideoBench"
LVB_EVAL_SPLIT = "validation"
LVB_DURATION_GROUPS = frozenset({"15", "60", "600", "3600"})
LVB_QUESTION_CATEGORIES = frozenset(
    {
        "TOS",
        "T2A",
        "T2O",
        "O2E",
        "S2E",
        "T3E",
        "TAA",
        "E3E",
        "T3O",
        "SSS",
        "SOS",
        "E2O",
        "S2A",
        "SAA",
        "O3O",
        "S2O",
        "T2E",
    }
)


class LongVideoBenchLoader(BaseDataLoader):
    """LongVideoBench validation 加载器（仅 validation，约 1.34k 条）。"""

    def __init__(
        self,
        video_dir: str,
        seed: int = 42,
        train_ratio: float = 0.8,
        task_filter: str = "all",
        dataset_name: str = LVB_HF_DATASET,
    ):
        super().__init__(video_dir=video_dir, seed=seed, train_ratio=train_ratio, task_filter=task_filter)
        self.dataset_name = dataset_name
        root = Path(os.path.expanduser(video_dir))
        self.video_roots = [
            str(p)
            for p in (
                root / "videos",
                root,
                root.parent / "videos",
            )
            if p.is_dir()
        ]
        self.subtitle_roots = [
            str(p)
            for p in (
                root / "subtitles",
                root,
                root.parent / "subtitles",
            )
            if p.is_dir()
        ]
        self._video_index: dict[str, str] | None = None

    @staticmethod
    def _raw_doc(raw_sample: Any) -> dict[str, Any]:
        if hasattr(raw_sample, "items"):
            return dict(raw_sample)
        return dict(raw_sample)

    def _local_annotation_candidates(self) -> list[Path]:
        root = Path(os.path.expanduser(self.video_dir))
        return [
            root / "lvb_val.json",
            root / "validation-00000-of-00001.parquet",
        ]

    def _load_local_records(self) -> list[dict[str, Any]] | None:
        root = Path(os.path.expanduser(self.video_dir))
        json_path = root / "lvb_val.json"
        if json_path.is_file():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [dict(row) for row in data]

        parquet_path = root / "validation-00000-of-00001.parquet"
        if parquet_path.is_file():
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError("读取本地 LongVideoBench parquet 需要 pandas。") from exc
            df = pd.read_parquet(parquet_path)
            return df.to_dict(orient="records")
        return None

    def load_raw_dataset(self, split: str) -> list[dict[str, Any]] | Any:
        split_key = str(split).strip().lower()
        if split_key not in {LVB_EVAL_SPLIT, "val", "validation"}:
            raise ValueError(
                f"LongVideoBench 仅支持 validation split（约 1.34k 条），收到 split={split!r}。"
            )

        local_records = self._load_local_records()
        if local_records is not None:
            return local_records
        return load_dataset(self.dataset_name, split=LVB_EVAL_SPLIT)

    def _build_video_index(self) -> None:
        if self._video_index is not None:
            return
        index: dict[str, str] = {}
        for root in self.video_roots:
            for cur_root, _, files in os.walk(root):
                for fn in files:
                    low = fn.lower()
                    if not low.endswith((".mp4", ".webm", ".mkv", ".avi", ".mov")):
                        continue
                    stem, _ = os.path.splitext(fn)
                    path = os.path.join(cur_root, fn)
                    index.setdefault(fn, path)
                    index.setdefault(stem, path)
        self._video_index = index

    def _resolve_video_path(self, doc: dict[str, Any]) -> str | None:
        rel = str(doc.get("video_path", "")).strip()
        if not rel:
            return None
        root = Path(os.path.expanduser(self.video_dir))
        direct_candidates = [
            root / rel,
            root / "videos" / rel,
            root / "videos" / Path(rel).name,
        ]
        for path in direct_candidates:
            if path.is_file():
                return str(path)

        self._build_video_index()
        if self._video_index is None:
            return None
        name = Path(rel).name
        stem = Path(rel).stem
        return self._video_index.get(name) or self._video_index.get(stem)

    def _resolve_subtitle_path(self, doc: dict[str, Any]) -> str | None:
        rel = str(doc.get("subtitle_path", "")).strip()
        if not rel:
            return None
        root = Path(os.path.expanduser(self.video_dir))
        direct_candidates = [
            root / rel,
            root / "subtitles" / rel,
            root / "subtitles" / Path(rel).name,
        ]
        for path in direct_candidates:
            if path.is_file():
                return str(path)
        for sub_root in self.subtitle_roots:
            candidate = Path(sub_root) / Path(rel).name
            if candidate.is_file():
                return str(candidate)
        return None

    @staticmethod
    def _extract_options(doc: dict[str, Any]) -> list[str] | None:
        options: list[str] = []
        for i in range(5):
            opt = doc.get(f"option{i}")
            if opt is None:
                continue
            text = str(opt).strip()
            if not text or text.upper() == "N/A":
                continue
            options.append(text)
        return options or None

    @staticmethod
    def _choice_to_letter(choice: Any) -> str:
        try:
            idx = int(choice)
        except (TypeError, ValueError):
            s = str(choice).strip().upper()
            m = re.search(r"\b([A-E])\b", s)
            return m.group(1) if m else s
        if 0 <= idx <= 25:
            return chr(ord("A") + idx)
        return str(choice).strip().upper()

    def _include_by_task(self, sample: VQASample) -> bool:
        if not sample_matches_task_filter(sample, self.task_filter):
            return False
        tf = str(self.task_filter).strip()
        if tf in {"all", "mcq", "numeric", "generation"}:
            return True
        meta = sample.metadata or {}
        duration_group = str(meta.get("duration_group", "")).strip()
        question_category = str(meta.get("question_category", "")).strip()
        if tf in LVB_DURATION_GROUPS:
            return duration_group == tf
        if tf in LVB_QUESTION_CATEGORIES:
            return question_category == tf
        return sample.task_type == tf

    def to_vqa_sample(self, raw_sample: dict[str, Any], index: int) -> VQASample | None:
        doc = self._raw_doc(raw_sample)
        question = str(doc.get("question", "")).strip()
        if not question:
            return None

        options = self._extract_options(doc)
        if not options:
            return None

        answer = self._choice_to_letter(doc.get("correct_choice"))
        if not answer:
            return None

        video_path = self._resolve_video_path(doc)
        if not video_path:
            return None

        sample_id = str(doc.get("id", "")).strip() or f"lvb_{index}"
        duration_group = str(doc.get("duration_group", "")).strip()
        question_category = str(doc.get("question_category", "")).strip()
        subtitle_path = self._resolve_subtitle_path(doc)

        return VQASample(
            sample_id=sample_id,
            video_path=video_path,
            question=question,
            answer=answer,
            options=options,
            task_type=duration_group or question_category or "mcq",
            preprocess_key=str(doc.get("video_id", "")).strip() or sample_id,
            metadata={
                "source_index": index,
                "raw_doc": doc,
                "video_id": doc.get("video_id", ""),
                "video_path_rel": doc.get("video_path", ""),
                "subtitle_path": subtitle_path or doc.get("subtitle_path", ""),
                "subtitle_path_rel": doc.get("subtitle_path", ""),
                "duration": doc.get("duration", ""),
                "duration_group": duration_group,
                "question_category": question_category,
                "topic_category": doc.get("topic_category", ""),
                "starting_timestamp_for_subtitles": doc.get("starting_timestamp_for_subtitles", 0.0),
                "correct_choice": doc.get("correct_choice"),
                "dataset_name": self.dataset_name,
                "split": LVB_EVAL_SPLIT,
            },
        )

    def get_eval_samples(
        self,
        sample_count: int | None = None,
        sample_seed_offset: int = 1000,
    ) -> list[VQASample]:
        if self.train_ratio > 0:
            warnings.warn(
                "LongVideoBench 使用官方 validation split，已忽略 train_ratio。",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.get_split_samples(
            split=LVB_EVAL_SPLIT,
            use_train_split=False,
            sample_count=sample_count,
            sample_seed_offset=sample_seed_offset,
        )

    def get_split_samples(
        self,
        split: str,
        use_train_split: bool,
        max_samples: int | None = None,
        sample_count: int | None = None,
        sample_seed_offset: int = 1000,
    ) -> list[VQASample]:
        if use_train_split:
            warnings.warn(
                "LongVideoBench 无 train split，已忽略 use_train_split。",
                RuntimeWarning,
                stacklevel=2,
            )

        dataset = self.load_raw_dataset(split)
        samples: list[VQASample] = []
        need = sample_count if sample_count is not None else max_samples
        for i in range(len(dataset)):
            sample = self.to_vqa_sample(dataset[i], i)
            if sample is None or not self._include_by_task(sample):
                continue
            samples.append(sample)
            if need is not None and len(samples) >= need:
                break

        if sample_count is not None and len(samples) > sample_count:
            random.seed(self.seed + sample_seed_offset)
            samples = random.sample(samples, sample_count)
        elif max_samples is not None:
            samples = samples[:max_samples]
        return samples


# 供 task_filter / 准确率统计复用
def sample_matches_lvb_bucket_filter(sample: VQASample, task_filter: str) -> bool:
    tf = str(task_filter).strip()
    if tf in {"all", "mcq", "numeric", "generation"}:
        return True
    meta = sample.metadata or {}
    if tf in LVB_DURATION_GROUPS:
        return str(meta.get("duration_group", "")).strip() == tf
    if tf in LVB_QUESTION_CATEGORIES:
        return str(meta.get("question_category", "")).strip() == tf
    return sample.task_type == tf


def lvb_uses_mcq_accuracy(task_filter: str) -> bool:
    tf = str(task_filter).strip()
    if tf in {"mcq", "short", "medium", "long"}:
        return True
    if tf in {"all", "numeric", "generation"}:
        return False
    return True
