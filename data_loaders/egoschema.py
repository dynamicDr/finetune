from __future__ import annotations

import os
from typing import Any

from .base import BaseDataLoader, VQASample, load_dataset


class EgoSchemaLoader(BaseDataLoader):
    def __init__(
        self,
        video_dir: str,
        seed: int = 42,
        train_ratio: float = 0.8,
        task_filter: str = "all",
        dataset_name: str = "lmms-lab/EgoSchema",
        dataset_config: str | None = "Subset",
        no_dataset_config: bool = False,
    ):
        super().__init__(video_dir=video_dir, seed=seed, train_ratio=train_ratio, task_filter=task_filter)
        self.dataset_name = dataset_name
        self.dataset_config = None if no_dataset_config else dataset_config
        base_video_dir = os.path.expanduser(video_dir)
        # 按更深目录优先，兼容 /egoschema 和 /egoschema/videos 作为输入。
        candidates = [
            os.path.join(base_video_dir, "videos", "videos", "videos"),
            os.path.join(base_video_dir, "videos", "videos"),
            os.path.join(base_video_dir, "videos"),
            base_video_dir,
        ]
        # 去重并保序
        self.video_roots = list(dict.fromkeys(candidates))

    def load_raw_dataset(self, split: str):
        kwargs: dict[str, Any] = {}
        if self.dataset_config:
            kwargs["name"] = self.dataset_config
        return load_dataset(self.dataset_name, split=split, **kwargs)

    @staticmethod
    def _pick_video_id(raw_sample: dict[str, Any]) -> str:
        for k in ("video_id", "video_uid", "video_idx", "q_uid", "uid", "id"):
            v = raw_sample.get(k, None)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    @staticmethod
    def _extract_options(raw_sample: dict[str, Any]) -> list[str] | None:
        if isinstance(raw_sample.get("options"), list) and len(raw_sample["options"]) > 0:
            return [str(x).strip() for x in raw_sample["options"]]
        if isinstance(raw_sample.get("option"), list) and len(raw_sample["option"]) > 0:
            return [str(x).strip() for x in raw_sample["option"]]
        option_keys = ("option0", "option1", "option2", "option3", "option4")
        vals = [str(raw_sample.get(k, "")).strip() for k in option_keys]
        vals = [v for v in vals if v]
        return vals if vals else None

    @staticmethod
    def _normalize_mcq_answer(raw_answer: Any) -> str:
        # 支持 A-E、0-4、1-5 三类标注格式
        if isinstance(raw_answer, str):
            s = raw_answer.strip()
            upper = s.upper()
            if upper in {"A", "B", "C", "D", "E"}:
                return upper
            if s.isdigit():
                raw_answer = int(s)
            else:
                return s

        if isinstance(raw_answer, int):
            if 0 <= raw_answer <= 4:
                return "ABCDE"[raw_answer]
            if 1 <= raw_answer <= 5:
                return "ABCDE"[raw_answer - 1]

        return str(raw_answer).strip()

    def _resolve_video_path(self, video_id: str) -> str | None:
        if not video_id:
            return None

        candidate_names = [video_id]
        # 若 video_id 不带扩展名，补全常见视频后缀
        if "." not in os.path.basename(video_id):
            candidate_names.extend([f"{video_id}.mp4", f"{video_id}.webm", f"{video_id}.mkv", f"{video_id}.avi"])

        for root in self.video_roots:
            for name in candidate_names:
                p = os.path.join(root, name)
                if os.path.isfile(p):
                    return p
        return None

    def to_vqa_sample(self, raw_sample: dict[str, Any], index: int) -> VQASample | None:
        question = str(raw_sample.get("question", "")).strip()
        if not question:
            return None

        video_id = self._pick_video_id(raw_sample)
        video_path = self._resolve_video_path(video_id)
        if not video_path:
            return None

        options = self._extract_options(raw_sample)
        raw_answer = raw_sample.get("answer", raw_sample.get("ground_truth", ""))
        answer = self._normalize_mcq_answer(raw_answer)
        if not answer:
            return None

        task_type = "mcq" if options else "numeric"
        return VQASample(
            sample_id=f"egoschema_{index}",
            video_path=video_path,
            question=question,
            answer=answer,
            options=options,
            task_type=task_type,
            metadata={
                "source_index": index,
                "video_id": video_id,
                "dataset_name": self.dataset_name,
                "dataset_config": self.dataset_config,
            },
        )
