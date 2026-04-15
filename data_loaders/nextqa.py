from __future__ import annotations

import os
from typing import Any

from .base import BaseDataLoader, VQASample, load_dataset


class NextQALoader(BaseDataLoader):
    def __init__(
        self,
        video_dir: str,
        seed: int = 42,
        train_ratio: float = 0.8,
        task_filter: str = "all",
        dataset_name: str = "lmms-lab/NExTQA",
        dataset_config: str | None = "MC",
        no_dataset_config: bool = False,
    ):
        super().__init__(video_dir=video_dir, seed=seed, train_ratio=train_ratio, task_filter=task_filter)
        self.dataset_name = dataset_name
        self.dataset_config = None if no_dataset_config else dataset_config

        base_video_dir = os.path.expanduser(video_dir)
        candidates = [
            os.path.join(base_video_dir, "NExTVideo"),
            base_video_dir,
        ]
        # 去重并保序，只保留存在的目录
        self.video_roots = [p for p in dict.fromkeys(candidates) if os.path.isdir(p)]
        self._video_index: dict[str, str] | None = None

    def load_raw_dataset(self, split: str):
        kwargs: dict[str, Any] = {}
        if self.dataset_config:
            kwargs["name"] = self.dataset_config
        return load_dataset(self.dataset_name, split=split, **kwargs)

    @staticmethod
    def _normalize_answer(answer: Any) -> str:
        if isinstance(answer, int) and 0 <= answer <= 4:
            return "ABCDE"[answer]
        if isinstance(answer, str):
            s = answer.strip()
            u = s.upper()
            if u in {"A", "B", "C", "D", "E"}:
                return u
            if s.isdigit():
                i = int(s)
                if 0 <= i <= 4:
                    return "ABCDE"[i]
                if 1 <= i <= 5:
                    return "ABCDE"[i - 1]
            return s
        return str(answer).strip()

    def _build_video_index(self) -> None:
        if self._video_index is not None:
            return
        index: dict[str, str] = {}
        for root in self.video_roots:
            for cur_root, _, files in os.walk(root):
                for fn in files:
                    low = fn.lower()
                    if not low.endswith((".mp4", ".webm", ".mkv", ".avi")):
                        continue
                    stem, _ = os.path.splitext(fn)
                    path = os.path.join(cur_root, fn)
                    if stem not in index:
                        index[stem] = path
        self._video_index = index

    def _resolve_video_path(self, video: Any) -> str | None:
        self._build_video_index()
        if self._video_index is None:
            return None
        vid = str(video).strip()
        if not vid:
            return None
        if vid in self._video_index:
            return self._video_index[vid]
        # 某些数据可能把 id 写成浮点样式，兜底转整数字符串
        try:
            vid_int = str(int(float(vid)))
            return self._video_index.get(vid_int)
        except ValueError:
            return None

    def to_vqa_sample(self, raw_sample: dict[str, Any], index: int) -> VQASample | None:
        question = str(raw_sample.get("question", "")).strip()
        if not question:
            return None

        options = [str(raw_sample.get(f"a{i}", "")).strip() for i in range(5)]
        if any(not opt for opt in options):
            return None

        answer = self._normalize_answer(raw_sample.get("answer", ""))
        if not answer:
            return None

        video_path = self._resolve_video_path(raw_sample.get("video", ""))
        if not video_path:
            return None

        qid = raw_sample.get("qid", index)
        return VQASample(
            sample_id=f"nextqa_{qid}",
            video_path=video_path,
            question=question,
            answer=answer,
            options=options,
            task_type="mcq",
            metadata={
                "source_index": index,
                "qid": qid,
                "video": raw_sample.get("video", ""),
                "type": raw_sample.get("type", ""),
                "dataset_name": self.dataset_name,
                "dataset_config": self.dataset_config,
            },
        )
