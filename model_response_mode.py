from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "model_response_modes.json"


def load_model_response_mode_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"模型响应模式配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("模型响应模式配置必须是 JSON 对象。")
    if not cfg:
        raise ValueError(f"模型响应模式配置为空: {path}")
    return cfg


def resolve_model_mode(
    model_identifier: str,
    config: dict[str, Any],
) -> tuple[str, bool, str]:
    # 简单映射: {"model_name": "thinking|instruct"}
    # 要求精确匹配；未命中直接报错，避免静默落到错误模式。
    mode_raw = config.get(model_identifier)
    if mode_raw is None:
        keys = ", ".join(sorted(config.keys()))
        raise KeyError(
            "模型未在 model_response_modes.json 中配置。"
            f" model_identifier={model_identifier!r}, available=[{keys}]"
        )
    mode = str(mode_raw).strip().lower()
    if mode not in {"thinking", "instruct"}:
        raise ValueError(f"非法模式: {mode!r}，仅支持 thinking 或 instruct。")
    require_think_end = mode == "thinking"
    return mode, require_think_end, f"exact:{model_identifier}"


def parse_response_by_mode(
    response: str,
    has_options: bool,
    model_mode: str,
) -> tuple[str, str, str | float]:
    text = response.strip()
    mode = model_mode.strip().lower()

    if "</think>" in text:
        cot_text, answer_portion = text.split("</think>", 1)
        cot_text = cot_text.strip()
        answer_portion = answer_portion.strip()
    elif mode == "thinking":
        cot_text = text
        answer_portion = text
    else:
        cot_text = ""
        answer_portion = text

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", answer_portion, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_portion = answer_match.group(1).strip()

    if has_options:
        match = re.search(r"\b([A-E])\b", answer_portion.upper())
        if match:
            return cot_text, answer_portion, match.group(1)

    numbers = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", answer_portion)
    for n in numbers:
        try:
            return cot_text, answer_portion, float(n)
        except ValueError:
            continue

    return cot_text, answer_portion, text


def extract_answer_by_mode(response: str, has_options: bool, model_mode: str):
    _, _, pred_answer = parse_response_by_mode(
        response=response,
        has_options=has_options,
        model_mode=model_mode,
    )
    return pred_answer
