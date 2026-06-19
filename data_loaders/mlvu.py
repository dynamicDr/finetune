from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import random

from .base import BaseDataLoader, VQASample
from lmms_eval_bridge import split_indices


# MLVU-Test 多选题任务（11 类中的 9 类 MCQ）
MLVU_MCQ_TASK_TYPES = frozenset(
    {
        "plotQA",
        "needleQA",
        "ego",
        "count",
        "order",
        "anomaly_reco",
        "topic_reasoning",
        "sportsQA",
        "tutorialQA",
    }
)

# MLVU 生成式任务
MLVU_GENERATION_TASK_TYPES = frozenset({"summary", "subsummary", "sub_scene"})


class MLVULoader(BaseDataLoader):
    """MLVU 本地 JSON + 视频加载器（支持 Dev: ~/dataset/mlvu，Test: ~/dataset/mlvu_test）。"""

    def __init__(
        self,
        video_dir: str,
        seed: int = 42,
        train_ratio: float = 0.8,
        task_filter: str = "all",
        dataset_name: str = "MLVU/MLVU_Test",
        dataset_config: str | None = None,
        no_dataset_config: bool = False,
    ):
        super().__init__(video_dir=video_dir, seed=seed, train_ratio=train_ratio, task_filter=task_filter)
        self.dataset_name = dataset_name
        self.dataset_config = None if no_dataset_config else dataset_config
        self._video_index: dict[str, str] | None = None

    def _dataset_root(self) -> Path:
        return Path(os.path.expanduser(self.video_dir))

    def _video_search_roots(self) -> list[str]:
        root = self._dataset_root()
        candidates = [
            root / "MLVU_Test" / "video" / "video",
            root / "MLVU_Test" / "video",
            root / "MLVU" / "video",
            root,
        ]
        # Dev 集按任务分子目录：MLVU/video/1_plotQA 等
        dev_video = root / "MLVU" / "video"
        if dev_video.is_dir():
            for child in sorted(dev_video.iterdir()):
                if child.is_dir():
                    candidates.append(child)
        return [str(p) for p in candidates if p.is_dir()]

    def _build_video_index(self) -> None:
        if self._video_index is not None:
            return
        index: dict[str, str] = {}
        for root in self._video_search_roots():
            for cur_root, _, files in os.walk(root):
                for fn in files:
                    if not fn.lower().endswith((".mp4", ".webm", ".mkv", ".avi", ".mov")):
                        continue
                    stem, _ = os.path.splitext(fn)
                    path = os.path.join(cur_root, fn)
                    index.setdefault(fn, path)
                    index.setdefault(stem, path)
        self._video_index = index

    def _resolve_video_path(self, video_name: str) -> str | None:
        self._build_video_index()
        if self._video_index is None:
            return None
        name = str(video_name).strip()
        if not name:
            return None
        if name in self._video_index:
            return self._video_index[name]
        stem, _ = os.path.splitext(name)
        return self._video_index.get(stem)

    @staticmethod
    def _normalize_question_type(raw: Any) -> str:
        return str(raw or "").strip()

    @staticmethod
    def _extract_options(raw_sample: dict[str, Any]) -> list[str] | None:
        raw = raw_sample.get("candidates", raw_sample.get("options"))
        if raw is None:
            return None
        if not isinstance(raw, list):
            return None
        options = [str(x).strip() for x in raw if str(x).strip()]
        return options or None

    @staticmethod
    def _answer_to_letter(answer: str, options: list[str] | None) -> str:
        ans = str(answer).strip()
        if not ans:
            return ""
        if len(ans) == 1 and ans.upper() in "ABCDEF":
            return ans.upper()
        m = re.search(r"\b([A-F])\b", ans.upper())
        if m:
            return m.group(1)
        if options:
            for i, opt in enumerate(options):
                if opt.strip().lower() == ans.lower():
                    return chr(ord("A") + i)
        return ans

    def _load_dev_records(self) -> list[dict[str, Any]]:
        json_dir = self._dataset_root() / "MLVU" / "json"
        if not json_dir.is_dir():
            raise FileNotFoundError(f"未找到 MLVU Dev 标注目录: {json_dir}")
        records: list[dict[str, Any]] = []
        for path in sorted(json_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for i, row in enumerate(data):
                item = dict(row)
                item.setdefault("question_id", f"dev_{path.stem}_{i}")
                item["_split"] = "dev"
                records.append(item)
        return records

    def _load_test_records(self) -> list[dict[str, Any]]:
        root = self._dataset_root()
        records: list[dict[str, Any]] = []

        mcq_gt = root / "test-ground-truth" / "test_mcq_gt.json"
        if mcq_gt.is_file():
            data = json.loads(mcq_gt.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for row in data:
                    item = dict(row)
                    item["_split"] = "test"
                    item["_format"] = "mcq"
                    records.append(item)

        if self.task_filter in {"all", "generation", "numeric"}:
            for rel in (
                ("test-ground-truth", "test_ssc_gt.json"),
                ("test-ground-truth", "test_vs_gt.json"),
            ):
                path = root / rel[0] / rel[1]
                if not path.is_file():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    continue
                for row in data:
                    item = dict(row)
                    item["_split"] = "test"
                    item["_format"] = "generation"
                    records.append(item)

        if not records:
            raise FileNotFoundError(
                f"未找到 MLVU Test 标注（期望 test-ground-truth/*.json）: {root}"
            )
        return records

    def load_raw_dataset(self, split: str) -> list[dict[str, Any]]:
        split_key = str(split).strip().lower()
        if split_key in {"dev", "train", "validation", "val"}:
            return self._load_dev_records()
        if split_key in {"test", "testset"}:
            return self._load_test_records()
        raise ValueError(
            f"MLVU 不支持 split={split!r}。请使用 test（MLVU-Test）或 dev（MLVU-Dev）。"
        )

    def to_vqa_sample(self, raw_sample: dict[str, Any], index: int) -> VQASample | None:
        question = str(raw_sample.get("question", "")).strip()
        if not question:
            return None

        question_type = self._normalize_question_type(
            raw_sample.get("question_type", raw_sample.get("task_type", ""))
        )
        options = self._extract_options(raw_sample)
        is_mcq = options is not None and len(options) > 0

        if is_mcq:
            answer = self._answer_to_letter(str(raw_sample.get("answer", "")), options)
            if not answer:
                return None
            task_type = question_type or "mcq"
        else:
            answer = str(raw_sample.get("answer", "")).strip()
            if not answer:
                return None
            task_type = question_type or "generation"
            options = None

        video_path = self._resolve_video_path(str(raw_sample.get("video", "")))
        if not video_path:
            return None

        sample_id = str(raw_sample.get("question_id", "")).strip() or f"mlvu_{index}"
        return VQASample(
            sample_id=sample_id,
            video_path=video_path,
            question=question,
            answer=answer,
            options=options,
            task_type=task_type,
            preprocess_key=sample_id,
            metadata={
                "source_index": index,
                "video": raw_sample.get("video", ""),
                "duration": raw_sample.get("duration", ""),
                "question_type": question_type,
                "split": raw_sample.get("_split", ""),
                "format": raw_sample.get("_format", "mcq" if is_mcq else "generation"),
                "scoring_points": raw_sample.get("scoring_points", None),
                "dataset_name": self.dataset_name,
            },
        )

    def get_eval_samples(
        self,
        sample_count: int | None = None,
        sample_seed_offset: int = 1000,
    ) -> list[VQASample]:
        return self.get_split_samples(
            split="test",
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
        """MLVU 已有官方 dev/test，默认不再做 train_ratio 二次划分。"""
        samples = self._convert_all(split)
        split_key = str(split).strip().lower()
        if split_key in {"test", "testset", "dev", "validation", "val", "train"}:
            selected = samples
        else:
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
