"""Official lmms-eval task helpers: Video-MME / LongVideoBench data paths, prompts, and metrics."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

_LMMS_VIDEOMME_UTILS = "lmms_eval.tasks.videomme.utils"
_LMMS_LVB_UTILS = "lmms_eval.tasks.longvideobench.utils"


def is_official_videomme_dataset(dataset: str) -> bool:
    return str(dataset).strip().lower() == "videomme"


def is_official_lvb_dataset(dataset: str) -> bool:
    return str(dataset).strip().lower() in {"lvb", "longvideobench"}


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


def _lvb_yaml_path() -> Path:
    import importlib

    mod = importlib.import_module(_LMMS_LVB_UTILS)
    return Path(mod.__file__).resolve().parent / "longvideobench_val_i.yaml"


def load_lvb_task_config() -> dict[str, Any]:
    with open(_lvb_yaml_path(), "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    safe_lines = [line for line in raw_lines if "!function" not in line]
    return yaml.safe_load("".join(safe_lines))


def resolve_lvb_prompt_kwargs(lmms_model_name: str) -> dict[str, Any]:
    cfg = load_lvb_task_config()
    kwargs_map = cfg.get("lmms_eval_specific_kwargs") or {}
    merged = dict(kwargs_map.get("default") or {})
    if lmms_model_name in kwargs_map:
        merged.update(kwargs_map[lmms_model_name])
    return merged


def resolve_lvb_generation_kwargs() -> dict[str, Any]:
    cfg = load_lvb_task_config()
    return dict(cfg.get("generation_kwargs") or {"max_new_tokens": 32})


def resolve_lvb_max_new_tokens() -> int:
    return int(resolve_lvb_generation_kwargs().get("max_new_tokens", 32))


def resolve_lvb_subtitle_path(
    doc: dict[str, Any],
    *,
    video_dir: str | None = None,
    subtitles_dir: str | None = None,
) -> str | None:
    rel = str(doc.get("subtitle_path", "")).strip()
    if not rel:
        return None

    candidates: list[Path] = []
    if subtitles_dir:
        base = Path(os.path.expanduser(subtitles_dir))
        candidates.extend([base / rel, base / Path(rel).name])
    if video_dir:
        root = Path(os.path.expanduser(video_dir))
        candidates.extend(
            [
                root / rel,
                root / "subtitles" / rel,
                root / "subtitles" / Path(rel).name,
            ]
        )
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def build_lvb_prompt(
    doc: dict[str, Any],
    lmms_model_name: str,
    *,
    use_subtitles: bool = False,
    num_frames: int | None = None,
    video_dir: str | None = None,
    subtitles_dir: str | None = None,
    subtitle_path: str | None = None,
) -> str:
    import importlib

    mod = importlib.import_module(_LMMS_LVB_UTILS)
    kwargs = resolve_lvb_prompt_kwargs(lmms_model_name)

    candidates: list[str] = []
    for i in range(5):
        candidate = doc.get(f"option{i}")
        if candidate is not None and str(candidate).strip().upper() != "N/A":
            candidates.append(str(candidate).strip())
    question = doc["question"] + "\n" + "\n".join(
        [f"{chr(ord('A') + i)}. {candidate}" for i, candidate in enumerate(candidates)]
    )
    pre_prompt = str(kwargs.get("pre_prompt", ""))
    post_prompt = str(kwargs.get("post_prompt", ""))

    if use_subtitles and kwargs.get("insert_interleave_subtitles", False):
        resolved_subtitle = subtitle_path or resolve_lvb_subtitle_path(
            doc,
            video_dir=video_dir,
            subtitles_dir=subtitles_dir,
        )
        if resolved_subtitle:
            with open(resolved_subtitle, encoding="utf-8") as f:
                subtitles = json.load(f)
            max_num_frames = int(num_frames or kwargs.get("max_num_frames", 16) or 16)
            frame_timestamps = mod.compute_frame_timestamps(float(doc["duration"]), max_num_frames)
            interleaved_prefix = mod.insert_subtitles_into_frames(
                frame_timestamps,
                subtitles,
                float(doc.get("starting_timestamp_for_subtitles", 0.0) or 0.0),
                float(doc["duration"]),
            )
            return f"{pre_prompt}{interleaved_prefix}\n{question}\n{post_prompt}"

    return f"{pre_prompt}{question}\n{post_prompt}"


def score_lvb(doc: dict[str, Any], response: str) -> dict[str, Any]:
    import importlib

    mod = importlib.import_module(_LMMS_LVB_UTILS)
    return mod.longvideobench_process_results(doc, [response])["lvb_acc"]


def aggregate_lvb(scores: list[dict[str, Any]]) -> float:
    import importlib

    mod = importlib.import_module(_LMMS_LVB_UTILS)
    return float(mod.longvideobench_aggregate_results(scores)) * 100.0
