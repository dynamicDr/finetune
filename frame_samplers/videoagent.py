from __future__ import annotations

import json
import os
import random
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .uniform import sample_uniform_frames


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _videoagent_dir() -> Path:
    return Path(__file__).resolve().parent / "VideoAgent"


def _resolve_caption_file() -> Path | None:
    env_caption = os.environ.get("VIDEOAGENT_CAPTION_FILE", "").strip()
    if env_caption:
        p = Path(env_caption).expanduser()
        return p if p.is_file() else None

    candidates = [
        _videoagent_dir() / "download" / "lavila_subset.json",
        _videoagent_dir() / "download" / "lavila_fullset_merged.json",
        _videoagent_dir() / "lavila_subset.json",
        _videoagent_dir() / "lavila_fullset_merged.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _resolve_feature_source() -> Path | None:
    env_features = os.environ.get("VIDEOAGENT_FEATURE_DIR", "").strip()
    if env_features:
        p = Path(env_features).expanduser()
        if p.is_dir() or (p.is_file() and p.suffix.lower() == ".zip"):
            return p
        zip_hint = p.with_suffix(".zip")
        return zip_hint if zip_hint.is_file() else None

    candidates = [
        _videoagent_dir() / "download" / "ego_features_448",
        _videoagent_dir() / "ego_features_448",
        _videoagent_dir() / "download" / "ego_features_448.zip",
        _videoagent_dir() / "ego_features_448.zip",
    ]
    for p in candidates:
        if p.is_dir() or (p.is_file() and p.suffix.lower() == ".zip"):
            return p
    return None


@lru_cache(maxsize=2)
def _load_captions(caption_file: str) -> dict[str, list[str]]:
    p = Path(caption_file).expanduser()
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
    return out


def parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_pattern = r"\{.*?\}|\[.*?\]"
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return None


def parse_text_find_number(text: str) -> int:
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        if match in range(-1, 5):
            return match
        return random.randint(0, 4)
    except Exception:
        return -1


def parse_text_find_confidence(text: str) -> int:
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        return random.randint(1, 3)
    except Exception:
        return 1


def get_llm_response(
    system_prompt: str,
    prompt: str,
    json_format: bool = True,
    model: str = "gpt-4-1106-preview",
) -> str:
    from openai import OpenAI

    try:
        from .VideoAgent.utils_general import get_from_cache, save_to_cache
    except Exception:
        get_from_cache = None  # type: ignore[assignment]
        save_to_cache = None  # type: ignore[assignment]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    key = json.dumps([model, messages], ensure_ascii=False)
    if get_from_cache is not None:
        cached_value = get_from_cache(key)
        if cached_value is not None:
            return cached_value

    client = OpenAI()
    for _ in range(3):
        try:
            if json_format:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(model=model, messages=messages)
            response = completion.choices[0].message.content or ""
            if save_to_cache is not None:
                save_to_cache(key, response)
            return response
        except Exception:
            continue
    return "GPT Error"


def generate_final_answer(question: str, caption: dict[str, str], num_frames: int, model: str) -> str:
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question:
    ```
    {question}
    ```
    Please think carefully and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question, and you must select one answer index from the candidates.
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    return get_llm_response(system_prompt, prompt, json_format=True, model=model)


def generate_description_step(
    question: str,
    caption: dict[str, str],
    num_frames: int,
    segment_des: dict[int, str],
    model: str,
) -> str:
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "3", "duration": "xxx - xxx", "description": "frame of xxx"},
        ]
    }
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question:
    ```
    {question}
    ```
    However, the information in the initial frames is not suffient.
    Objective:
    Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights.
    Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    return get_llm_response(system_prompt, prompt, json_format=True, model=model)


def self_eval(previous_prompt: str, answer: str, model: str) -> str:
    confidence_format = {"confidence": "xxx"}
    prompt = f"""Please assess the confidence level in the decision-making process.
    The provided information is as as follows,
    {previous_prompt}
    The decision making process is as follows,
    {answer}
    Criteria for Evaluation:
    Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
    Partial Information (Confidence Level: 2): If information partially supports an informed guess.
    Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
    Assessment Focus:
    Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the decision-making context.
    Please generate the confidence with JSON format {confidence_format}
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    return get_llm_response(system_prompt, prompt, json_format=True, model=model)


def ask_gpt_caption(question: str, caption: dict[str, str], num_frames: int, model: str):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of five uniformly sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question:
    ```
    {question}
    ```
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False, model=model)
    return prompt, response


def ask_gpt_caption_step(question: str, caption: dict[str, str], num_frames: int, model: str):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question:
    ```
    {question}
    ```
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False, model=model)
    return prompt, response


def read_caption(captions: list[str], sample_idx: list[int]) -> dict[str, str]:
    video_caption = {}
    for idx in sample_idx:
        if 1 <= idx <= len(captions):
            video_caption[f"frame {idx}"] = captions[idx - 1]
    return video_caption


def _to_raw_frame_idx(cap_idx_1based: int, total_caps: int, total_frames: int) -> int:
    if total_caps <= 1:
        return 0
    ratio = (cap_idx_1based - 1) / max(total_caps - 1, 1)
    return int(round(ratio * max(total_frames - 1, 0)))


def _read_frames_by_indices(video_path: str, cap_indices_1based: list[int], total_caps: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    frames: list[Image.Image] = []
    for cap_idx in sorted(set(cap_indices_1based)):
        raw_idx = _to_raw_frame_idx(cap_idx, total_caps=total_caps, total_frames=total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def _build_formatted_question(question: str | None) -> str:
    # 当前框架只传入问题文本，不传 options；保持兼容并尽量贴近原流程
    return (question or "").strip()


def sample_videoagent_frames(
    video_path: str,
    num_frames: int,
    question: str | None = None,
    answer: str | None = None,
    random_seed: int | None = None,
) -> list[Image.Image]:
    _ = random_seed
    _ = answer
    if not video_path or num_frames <= 0:
        return []

    caption_file = _resolve_caption_file()
    feature_source = _resolve_feature_source()
    llm_model = os.environ.get("VIDEOAGENT_LLM_MODEL", "gpt-4-1106-preview")

    if not caption_file or not feature_source or not os.getenv("OPENAI_API_KEY"):
        warnings.warn(
            "VideoAgent 依赖缺失（caption/features/OPENAI_API_KEY），回退为 uniform 选帧。",
            RuntimeWarning,
            stacklevel=1,
        )
        return sample_uniform_frames(video_path, num_frames, question=question, answer=answer)

    try:
        from .VideoAgent.utils_clip import frame_retrieval_seg_ego
    except Exception:
        warnings.warn("加载 VideoAgent.utils_clip 失败，回退为 uniform 选帧。", RuntimeWarning, stacklevel=1)
        return sample_uniform_frames(video_path, num_frames, question=question, answer=answer)

    video_id = Path(video_path).stem
    all_caps = _load_captions(str(caption_file))
    caps = all_caps.get(video_id, [])
    if len(caps) < 2:
        warnings.warn("未找到视频对应 captions，回退为 uniform 选帧。", RuntimeWarning, stacklevel=1)
        return sample_uniform_frames(video_path, num_frames, question=question, answer=answer)

    formatted_question = _build_formatted_question(question)
    num_caps_frames = len(caps)

    sample_idx = np.linspace(1, num_caps_frames, num=5, dtype=int).tolist()
    sample_idx = sorted(list(set(int(x) for x in sample_idx)))
    sampled_caps = read_caption(caps, sample_idx)

    previous_prompt, answer_str = ask_gpt_caption(formatted_question, sampled_caps, num_caps_frames, llm_model)
    pred_answer = parse_text_find_number(answer_str)
    confidence_str = self_eval(previous_prompt, answer_str, llm_model)
    confidence = parse_text_find_confidence(confidence_str)

    if confidence < 3:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}" for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question, sampled_caps, num_caps_frames, segment_des, llm_model
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_descriptions = (
                parsed_candiate_descriptions.get("frame_descriptions", [])
                if isinstance(parsed_candiate_descriptions, dict)
                else []
            )
            frame_idx = frame_retrieval_seg_ego(frame_descriptions, video_id, sample_idx, feature_dir=str(feature_source))
            sample_idx += [int(x) for x in frame_idx if isinstance(x, int)]
            sample_idx = sorted(list(set(sample_idx)))
            sample_idx = [x for x in sample_idx if 1 <= x <= num_caps_frames]
            sampled_caps = read_caption(caps, sample_idx)
            previous_prompt, answer_str = ask_gpt_caption_step(formatted_question, sampled_caps, num_caps_frames, llm_model)
            pred_answer = parse_text_find_number(answer_str)
            confidence_str = self_eval(previous_prompt, answer_str, llm_model)
            confidence = parse_text_find_confidence(confidence_str)
        except Exception:
            answer_str = generate_final_answer(formatted_question, sampled_caps, num_caps_frames, llm_model)
            pred_answer = parse_text_find_number(answer_str)

    if confidence < 3:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}" for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question, sampled_caps, num_caps_frames, segment_des, llm_model
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_descriptions = (
                parsed_candiate_descriptions.get("frame_descriptions", [])
                if isinstance(parsed_candiate_descriptions, dict)
                else []
            )
            frame_idx = frame_retrieval_seg_ego(frame_descriptions, video_id, sample_idx, feature_dir=str(feature_source))
            sample_idx += [int(x) for x in frame_idx if isinstance(x, int)]
            sample_idx = sorted(list(set(sample_idx)))
            sample_idx = [x for x in sample_idx if 1 <= x <= num_caps_frames]
            sampled_caps = read_caption(caps, sample_idx)
            answer_str = generate_final_answer(formatted_question, sampled_caps, num_caps_frames, llm_model)
            pred_answer = parse_text_find_number(answer_str)
        except Exception:
            answer_str = generate_final_answer(formatted_question, sampled_caps, num_caps_frames, llm_model)
            pred_answer = parse_text_find_number(answer_str)

    if pred_answer == -1:
        pred_answer = random.randint(0, 4)

    if len(sample_idx) > num_frames:
        pick = np.linspace(0, len(sample_idx) - 1, num=num_frames, dtype=int).tolist()
        sample_idx = [sample_idx[i] for i in pick]
    return _read_frames_by_indices(video_path, sample_idx, total_caps=num_caps_frames)
