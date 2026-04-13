from __future__ import annotations

import os
from typing import Any

from .base import BaseDataLoader, VQASample, load_dataset


class VSIBenchLoader(BaseDataLoader):
    def __init__(
        self,
        video_dir: str,
        seed: int = 42,
        train_ratio: float = 0.8,
        task_filter: str = "all",
        dataset_name: str = "nyu-visionx/VSI-Bench",
        dataset_config: str | None = "full",
        no_dataset_config: bool = False,
    ):
        super().__init__(video_dir=video_dir, seed=seed, train_ratio=train_ratio, task_filter=task_filter)
        self.dataset_name = dataset_name
        self.dataset_config = None if no_dataset_config else dataset_config

    def load_raw_dataset(self, split: str):
        kwargs: dict[str, Any] = {}
        if self.dataset_config:
            kwargs["name"] = self.dataset_config
        return load_dataset(self.dataset_name, split=split, **kwargs)

    def to_vqa_sample(self, raw_sample: dict[str, Any], index: int) -> VQASample | None:
        scene = str(raw_sample.get("scene_name", "")).strip()
        if not scene:
            return None
        video_path = os.path.join(self.video_dir, f"{scene}.mp4")
        if not os.path.isfile(video_path):
            return None

        question = str(raw_sample.get("question", "")).strip()
        answer = str(raw_sample.get("ground_truth", "")).strip()
        if not question or not answer:
            return None

        options = raw_sample.get("options", None)
        if not isinstance(options, list) or len(options) == 0:
            options = None
            task_type = "numeric"
        else:
            options = [str(x) for x in options]
            task_type = "mcq"

        return VQASample(
            sample_id=f"vsibench_{index}",
            video_path=video_path,
            question=question,
            answer=answer,
            options=options,
            task_type=task_type,
            metadata={
                "source_index": index,
                "scene_name": scene,
                "task_type_raw": raw_sample.get("task_type", raw_sample.get("question_type", "unknown")),
            },
        )
