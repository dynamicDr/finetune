from __future__ import annotations

import os
from typing import Any

from .base import BaseDataLoader, VQASample, load_dataset


def _pick_first_non_empty(raw: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        if k in raw and raw[k] is not None:
            v = str(raw[k]).strip()
            if v:
                return v
    return ""


def _extract_options(raw: dict[str, Any]) -> list[str] | None:
    # style-1: options: [...]
    opts = raw.get("options")
    if isinstance(opts, list) and opts:
        out = [str(x).strip() for x in opts if str(x).strip()]
        return out or None

    # style-2: option_0..option_4 / option0..option4
    bucket: list[str] = []
    for i in range(8):
        for key in (f"option_{i}", f"option{i}", f"candidate_{i}", f"candidate{i}"):
            if key in raw and raw[key] is not None:
                text = str(raw[key]).strip()
                if text:
                    bucket.append(text)
                    break
    return bucket or None


def _normalize_mcq_answer(answer_raw: Any, options: list[str] | None) -> str:
    if answer_raw is None:
        return ""
    text = str(answer_raw).strip()
    if not text:
        return ""

    # If answer is index-like and options exist, convert to option letter.
    if options is not None:
        try:
            idx = int(float(text))
            if 0 <= idx < len(options):
                return chr(ord("A") + idx)
        except Exception:
            pass

    # Keep letter answer if provided.
    if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return text.upper()
    return text


class EgoSchemaLoader(BaseDataLoader):
    def __init__(
        self,
        video_dir: str,
        seed: int = 42,
        train_ratio: float = 0.8,
        task_filter: str = "all",
        dataset_name: str = "lmms-lab/EgoSchema",
        dataset_config: str | None = None,
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

    def _resolve_video_path(self, video_id: str) -> str | None:
        if not video_id:
            return None
        # Support both ids without extension and full filename.
        candidates = [video_id]
        if "." not in os.path.basename(video_id):
            candidates.extend([f"{video_id}.mp4", f"{video_id}.webm", f"{video_id}.mkv", f"{video_id}.avi"])
        for name in candidates:
            path = os.path.join(self.video_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def to_vqa_sample(self, raw_sample: dict[str, Any], index: int) -> VQASample | None:
        video_id = _pick_first_non_empty(
            raw_sample,
            ["video_id", "video_uid", "video_idx", "q_uid", "uid", "id"],
        )
        video_path = self._resolve_video_path(video_id)
        if not video_path:
            return None

        question = _pick_first_non_empty(raw_sample, ["question", "query", "prompt"])
        if not question:
            return None

        options = _extract_options(raw_sample)
        answer_raw = raw_sample.get("answer", raw_sample.get("label", raw_sample.get("ground_truth")))
        answer = _normalize_mcq_answer(answer_raw, options)
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
            },
        )
