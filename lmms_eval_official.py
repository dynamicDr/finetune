"""Official lmms-eval task helpers: Video-MME data paths, prompts, and metrics."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_LMMS_VIDEOMME_UTILS = "lmms_eval.tasks.videomme.utils"


def is_official_videomme_dataset(dataset: str) -> bool:
    return str(dataset).strip().lower() == "videomme"


def _videomme_yaml_path() -> Path:
    import importlib

    mod = importlib.import_module(_LMMS_VIDEOMME_UTILS)
    return Path(mod.__file__).resolve().parent / "videomme.yaml"


def load_videomme_task_config() -> dict[str, Any]:
    with open(_videomme_yaml_path(), "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    safe_lines = [line for line in raw_lines if "!function" not in line]
    return yaml.safe_load("".join(safe_lines))


def resolve_videomme_prompt_kwargs(lmms_model_name: str) -> dict[str, Any]:
    cfg = load_videomme_task_config()
    kwargs_map = cfg.get("lmms_eval_specific_kwargs") or {}
    merged = dict(kwargs_map.get("default") or {})
    if lmms_model_name in kwargs_map:
        merged.update(kwargs_map[lmms_model_name])
    return merged


def resolve_videomme_generation_kwargs() -> dict[str, Any]:
    cfg = load_videomme_task_config()
    return dict(cfg.get("generation_kwargs") or {"max_new_tokens": 16})


def resolve_videomme_max_new_tokens() -> int:
    return int(resolve_videomme_generation_kwargs().get("max_new_tokens", 16))


def _official_videomme_cache_data_dir() -> str:
    import importlib

    mod = importlib.import_module(_LMMS_VIDEOMME_UTILS)
    return os.path.join(mod.base_cache_dir, mod.cache_name, "data")


def _candidate_video_paths(video_id: str, data_dirs: list[str]) -> list[str]:
    stem = str(video_id).strip()
    if not stem:
        return []
    suffixes = (".mp4", ".MP4", ".mkv", ".webm")
    paths: list[str] = []
    for data_dir in data_dirs:
        for suffix in suffixes:
            paths.append(os.path.join(data_dir, stem + suffix))
    return paths


def resolve_videomme_video_path(doc: dict[str, Any], *, video_dir: str | None = None) -> str | None:
    video_id = str(doc.get("videoID", "")).strip()
    if not video_id:
        return None

    data_dirs = [_official_videomme_cache_data_dir()]
    if video_dir:
        base = os.path.expanduser(video_dir)
        parent = os.path.dirname(base.rstrip("/"))
        data_dirs.extend(
            [
                os.path.join(base, "data"),
                os.path.join(base, "videos", "data"),
                os.path.join(base, "videos"),
                base,
                os.path.join(parent, "videos", "data"),
                os.path.join(parent, "videos"),
                parent,
            ]
        )

    seen: set[str] = set()
    for path in _candidate_video_paths(video_id, data_dirs):
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None


def build_videomme_prompt(
    doc: dict[str, Any],
    lmms_model_name: str,
    *,
    use_subtitles: bool = False,
    num_frames: int | None = None,
) -> str:
    import importlib

    mod = importlib.import_module(_LMMS_VIDEOMME_UTILS)
    kwargs = resolve_videomme_prompt_kwargs(lmms_model_name)
    if use_subtitles:
        subtitle_kwargs = dict(kwargs)
        if num_frames is not None:
            subtitle_kwargs["frame_num"] = num_frames
        return mod.videomme_doc_to_text_subtitle(doc, subtitle_kwargs)
    return mod.videomme_doc_to_text(doc, kwargs)


def score_videomme(doc: dict[str, Any], response: str) -> dict[str, Any]:
    import importlib

    mod = importlib.import_module(_LMMS_VIDEOMME_UTILS)
    return mod.videomme_process_results(doc, [response])["videomme_perception_score"]


def aggregate_videomme(scores: list[dict[str, Any]]) -> float:
    import importlib

    mod = importlib.import_module(_LMMS_VIDEOMME_UTILS)
    return float(mod.videomme_aggregate_results(scores))
