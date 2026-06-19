from __future__ import annotations

import os
import random
import re
import warnings
from typing import Any

from lmms_eval_official import resolve_videomme_video_path

from .base import BaseDataLoader, VQASample, load_dataset


class VideoMMELoader(BaseDataLoader):
    def __init__(
        self,
        video_dir: str,
        seed: int = 42,
        train_ratio: float = 0.8,
        task_filter: str = "all",
        dataset_name: str = "lmms-lab/Video-MME",
        dataset_config: str | None = None,
        no_dataset_config: bool = False,
        use_official_eval: bool = True,
    ):
        super().__init__(video_dir=video_dir, seed=seed, train_ratio=train_ratio, task_filter=task_filter)
        self.dataset_name = dataset_name
        self.dataset_config = None if no_dataset_config else dataset_config
        self.use_official_eval = use_official_eval

        base_video_dir = os.path.expanduser(video_dir)
        parent_video_dir = os.path.dirname(base_video_dir.rstrip("/"))
        candidates = [
            os.path.join(base_video_dir, "videos", "data"),
            os.path.join(base_video_dir, "videos"),
            os.path.join(base_video_dir, "data"),
            base_video_dir,
            os.path.join(parent_video_dir, "videos", "data"),
            os.path.join(parent_video_dir, "videos"),
            parent_video_dir,
        ]
        self.video_roots = [p for p in dict.fromkeys(candidates) if os.path.isdir(p)]
        self._video_index: dict[str, str] | None = None

    def _local_parquet_candidates(self) -> list[str]:
        base_video_dir = os.path.expanduser(self.video_dir)
        parent_video_dir = os.path.dirname(base_video_dir.rstrip("/"))
        return [
            os.path.join(base_video_dir, "videomme", "test-00000-of-00001.parquet"),
            os.path.join(base_video_dir, "test-00000-of-00001.parquet"),
            os.path.join(parent_video_dir, "videomme", "test-00000-of-00001.parquet"),
            os.path.join(parent_video_dir, "test-00000-of-00001.parquet"),
        ]

    def _load_local_parquet_records(self) -> list[dict[str, Any]] | None:
        local_parquet = next((p for p in self._local_parquet_candidates() if os.path.isfile(p)), None)
        if not local_parquet:
            return None
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("读取本地 Video-MME parquet 需要 pandas。") from exc
        df = pd.read_parquet(local_parquet)
        return df.to_dict(orient="records")

    def load_raw_dataset(self, split: str):
        local_records = self._load_local_parquet_records()
        if local_records is not None:
            return local_records

        kwargs: dict[str, Any] = {}
        if self.dataset_config:
            kwargs["name"] = self.dataset_config
        return load_dataset(self.dataset_name, split=split, **kwargs)

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
                    if stem not in index:
                        index[stem] = path
        self._video_index = index

    def _resolve_video_path(self, raw_sample: dict[str, Any]) -> str | None:
        self._build_video_index()
        if self._video_index is None:
            return None

        ids = [
            str(raw_sample.get("videoID", "")).strip(),
            str(raw_sample.get("video_id", "")).strip(),
            str(raw_sample.get("video", "")).strip(),
        ]
        for vid in ids:
            if vid and vid in self._video_index:
                return self._video_index[vid]
        return None

    @staticmethod
    def _extract_options(raw_sample: dict[str, Any]) -> list[str] | None:
        raw_options = raw_sample.get("options", None)
        if raw_options is None:
            return None

        if hasattr(raw_options, "tolist"):
            raw_options = raw_options.tolist()
        if not isinstance(raw_options, list):
            return None

        options = [str(x).strip() for x in raw_options if str(x).strip()]
        return options if options else None

    @staticmethod
    def _normalize_answer(raw_answer: Any) -> str:
        s = str(raw_answer).strip().upper()
        if not s:
            return ""
        m = re.search(r"\b([A-E])\b", s)
        if m:
            return m.group(1)
        return s

    @staticmethod
    def _raw_doc(raw_sample: Any) -> dict[str, Any]:
        if hasattr(raw_sample, "items"):
            return dict(raw_sample)
        return dict(raw_sample)

    def to_vqa_sample(self, raw_sample: dict[str, Any], index: int) -> VQASample | None:
        doc = self._raw_doc(raw_sample)
        question = str(doc.get("question", "")).strip()
        if not question:
            return None

        options = self._extract_options(doc)
        if not options:
            return None

        if self.use_official_eval:
            answer = str(doc.get("answer", "")).strip()
            video_path = resolve_videomme_video_path(doc, video_dir=self.video_dir)
            if not video_path:
                video_path = self._resolve_video_path(doc)
        else:
            answer = self._normalize_answer(doc.get("answer", ""))
            video_path = self._resolve_video_path(doc)

        if not answer:
            return None
        if not video_path:
            return None

        sample_id = str(doc.get("question_id", "")).strip() or f"videomme_{index}"
        duration = str(doc.get("duration", "")).strip().lower()
        if duration not in {"short", "medium", "long"}:
            duration = "short"
        return VQASample(
            sample_id=sample_id,
            video_path=video_path,
            question=question,
            answer=answer,
            options=options,
            task_type=duration,
            metadata={
                "source_index": index,
                "raw_doc": doc,
                "videoID": doc.get("videoID", ""),
                "video_id": doc.get("video_id", ""),
                "duration": doc.get("duration", ""),
                "domain": doc.get("domain", ""),
                "sub_category": doc.get("sub_category", ""),
                "task_type": doc.get("task_type", ""),
                "dataset_name": self.dataset_name,
                "dataset_config": self.dataset_config,
            },
        )

    def get_eval_samples(
        self,
        sample_count: int | None = None,
        sample_seed_offset: int = 1000,
    ) -> list[VQASample]:
        if not self.use_official_eval:
            return super().get_eval_samples(
                sample_count=sample_count,
                sample_seed_offset=sample_seed_offset,
            )

        dataset = self.load_raw_dataset("test")
        samples: list[VQASample] = []
        need = sample_count
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
        return samples

    def get_split_samples(
        self,
        split: str,
        use_train_split: bool,
        max_samples: int | None = None,
        sample_count: int | None = None,
        sample_seed_offset: int = 1000,
    ) -> list[VQASample]:
        if not self.use_official_eval:
            return super().get_split_samples(
                split=split,
                use_train_split=use_train_split,
                max_samples=max_samples,
                sample_count=sample_count,
                sample_seed_offset=sample_seed_offset,
            )

        if use_train_split or self.train_ratio > 0:
            warnings.warn(
                "官方 Video-MME 评测使用完整 test split，已忽略 train_ratio / use_train_split。",
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
