from __future__ import annotations

import os
import re
from typing import Any

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
    ):
        super().__init__(video_dir=video_dir, seed=seed, train_ratio=train_ratio, task_filter=task_filter)
        self.dataset_name = dataset_name
        self.dataset_config = None if no_dataset_config else dataset_config

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

    def _include_by_task(self, sample: VQASample) -> bool:
        if self.task_filter in {"all", "mcq"}:
            return True
        return sample.task_type == self.task_filter

    def _local_parquet_candidates(self) -> list[str]:
        base_video_dir = os.path.expanduser(self.video_dir)
        parent_video_dir = os.path.dirname(base_video_dir.rstrip("/"))
        return [
            os.path.join(base_video_dir, "videomme", "test-00000-of-00001.parquet"),
            os.path.join(base_video_dir, "test-00000-of-00001.parquet"),
            os.path.join(parent_video_dir, "videomme", "test-00000-of-00001.parquet"),
            os.path.join(parent_video_dir, "test-00000-of-00001.parquet"),
        ]

    def load_raw_dataset(self, split: str):
        local_parquet = next((p for p in self._local_parquet_candidates() if os.path.isfile(p)), None)
        if local_parquet:
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError("读取本地 Video-MME parquet 需要 pandas。") from exc
            df = pd.read_parquet(local_parquet)
            return df.to_dict(orient="records")

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

    def to_vqa_sample(self, raw_sample: dict[str, Any], index: int) -> VQASample | None:
        question = str(raw_sample.get("question", "")).strip()
        if not question:
            return None

        options = self._extract_options(raw_sample)
        if not options:
            return None

        answer = self._normalize_answer(raw_sample.get("answer", ""))
        if not answer:
            return None

        video_path = self._resolve_video_path(raw_sample)
        if not video_path:
            return None

        sample_id = str(raw_sample.get("question_id", "")).strip() or f"videomme_{index}"
        duration = str(raw_sample.get("duration", "")).strip().lower()
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
                "videoID": raw_sample.get("videoID", ""),
                "video_id": raw_sample.get("video_id", ""),
                "duration": raw_sample.get("duration", ""),
                "domain": raw_sample.get("domain", ""),
                "sub_category": raw_sample.get("sub_category", ""),
                "task_type": raw_sample.get("task_type", ""),
                "dataset_name": self.dataset_name,
                "dataset_config": self.dataset_config,
            },
        )
