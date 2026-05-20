from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


_CLIP_CACHE: dict[str, Any] = {}


def _log(msg: str) -> None:
    print(f"[ours] {msg}", flush=True)


def _to_feature_tensor(features: Any) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(features, attr, None)
        if isinstance(val, torch.Tensor):
            if attr == "last_hidden_state" and val.ndim >= 2:
                return val[:, 0, :]
            return val
    if isinstance(features, (tuple, list)) and features and isinstance(features[0], torch.Tensor):
        return features[0]
    raise TypeError(f"无法将特征输出转换为 Tensor，实际类型: {type(features)!r}")


def _format_question_and_options(question: str | None, options: list[str] | None) -> str:
    q = (question or "").strip()
    if not q:
        raise ValueError("ours 迭代方法需要 question。")
    if not options:
        raise ValueError("ours 迭代方法需要 options，且每个选项必须包含内容。")
    normalized_options: list[str] = []
    for i, raw_option in enumerate(options):
        option_text = str(raw_option).strip()
        if not option_text:
            raise ValueError(f"ours 迭代方法选项不能为空：第 {i + 1} 个选项为空。")
        if re.fullmatch(r"[A-Ea-e](?:[\.\)\:\-])?", option_text):
            raise ValueError(f"ours 迭代方法选项不能只写字母：{option_text}")
        prefixed = re.match(r"^([A-Ea-e])[\.\)\:\-]\s*(.*)$", option_text)
        if prefixed:
            content = prefixed.group(2).strip()
            if not content:
                raise ValueError(f"ours 迭代方法选项不能只写字母：{option_text}")
            normalized_options.append(f"{prefixed.group(1).upper()}. {content}")
            continue
        normalized_options.append(f"{chr(ord('A') + i)}. {option_text}")
    return f"{q}\nOptions:\n" + "\n".join(normalized_options)


def _build_clip_query(question: str | None, options: list[str] | None, answer: str | None) -> str:
    _ = answer
    qa_text = _format_question_and_options(question, options)
    return f"a video frame relevant to the following question and options:\n{qa_text}"


def _load_clip(model_id: str, device: str | None):
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = f"{model_id}::{resolved_device}"
    if cache_key in _CLIP_CACHE:
        return _CLIP_CACHE[cache_key]
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(resolved_device).eval()
    _CLIP_CACHE[cache_key] = (processor, model, resolved_device)
    return _CLIP_CACHE[cache_key]


def _collect_candidate_frames(video_path: str, sample_every: int) -> tuple[list[int], list[Image.Image]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    if sample_every <= 0:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if video_fps <= 0:
            cap.release()
            raise RuntimeError(f"无法读取视频 FPS，不能按 1fps 粗采样: {video_path}")
        resolved_sample_every = max(1, int(round(video_fps)))
    else:
        resolved_sample_every = sample_every
    frame_ids: list[int] = []
    images: list[Image.Image] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % resolved_sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_ids.append(idx)
                images.append(Image.fromarray(rgb))
            idx += 1
    finally:
        cap.release()
    if not images:
        raise RuntimeError(f"视频无可用候选帧: {video_path}")
    return frame_ids, images


def _normalize_sample_id(sample_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id.strip())
    return normalized.strip("_")


def _load_preprocessed_candidate_frames(
    preprocessed_clip_dir: str,
    sample_id: str,
) -> tuple[list[int], list[Image.Image]]:
    normalized_sample_id = _normalize_sample_id(sample_id)
    if not normalized_sample_id:
        raise ValueError(f"sample_id 非法，无法定位预处理帧目录: {sample_id!r}")
    sample_dir = Path(preprocessed_clip_dir).expanduser().resolve() / normalized_sample_id
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"预处理帧目录不存在: {sample_dir}")

    meta_path = sample_dir / "metadata.json"
    frame_ids: list[int] = []
    image_paths: list[Path] = []
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        listed_ids = meta.get("frame_ids", [])
        listed_files = meta.get("files", [])
        if isinstance(listed_ids, list) and isinstance(listed_files, list) and len(listed_ids) == len(listed_files):
            for fid, rel_name in zip(listed_ids, listed_files):
                p = sample_dir / str(rel_name)
                if p.is_file():
                    frame_ids.append(int(fid))
                    image_paths.append(p)
    if not image_paths:
        for p in sorted(sample_dir.glob("frame_*.jpg")):
            m = re.match(r"frame_(\d+)\.jpg$", p.name)
            if not m:
                continue
            frame_ids.append(int(m.group(1)))
            image_paths.append(p)
    if not image_paths:
        raise RuntimeError(f"预处理帧目录中没有可用图片: {sample_dir}")

    images: list[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return frame_ids, images


def _encode_images_batched(
    images: list[Image.Image],
    processor,
    model,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_feats = _to_feature_tensor(model.get_image_features(**inputs))
        batch_feats = batch_feats / batch_feats.norm(dim=-1, keepdim=True)
        feats.append(batch_feats)
    return torch.cat(feats, dim=0)


@dataclass
class RankedFrame:
    frame_id: int
    image: Image.Image
    score: float


def rank_frames_by_clip(
    video_path: str,
    question: str,
    options: list[str] | None,
    answer: str | None,
    clip_model_id: str = "openai/clip-vit-base-patch32",
    clip_device: str | None = None,
    clip_batch_size: int = 16,
    coarse_sample_every: int = 0,
    max_frames: int | None = None,
    sample_id: str | None = None,
    use_preprocessed_clip_frames: bool = False,
    preprocessed_clip_dir: str | None = None,
) -> list[RankedFrame]:
    t0 = time.perf_counter()
    if use_preprocessed_clip_frames:
        if not preprocessed_clip_dir:
            raise ValueError("启用预处理 clip 帧时，必须提供 preprocessed_clip_dir。")
        if not sample_id:
            raise ValueError("启用预处理 clip 帧时，必须提供 sample_id。")
        frame_ids, images = _load_preprocessed_candidate_frames(
            preprocessed_clip_dir=preprocessed_clip_dir,
            sample_id=sample_id,
        )
    else:
        frame_ids, images = _collect_candidate_frames(video_path, sample_every=coarse_sample_every)
    processor, model, resolved_device = _load_clip(clip_model_id, clip_device)
    query = _build_clip_query(question=question, options=options, answer=answer)

    image_feats = _encode_images_batched(
        images=images,
        processor=processor,
        model=model,
        device=resolved_device,
        batch_size=clip_batch_size,
    )
    text_inputs = processor(
        text=[query],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(resolved_device)
    with torch.no_grad():
        text_feats = _to_feature_tensor(model.get_text_features(**text_inputs))
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    scores = (image_feats @ text_feats.T).squeeze(1)
    ranked_indices = torch.argsort(scores, descending=True).tolist()

    ranked_frames: list[RankedFrame] = []
    for idx in ranked_indices:
        ranked_frames.append(
            RankedFrame(
                frame_id=frame_ids[idx],
                image=images[idx],
                score=float(scores[idx].detach().cpu().item()),
            )
        )
    if max_frames is not None:
        ranked_frames = ranked_frames[:max_frames]

    top_preview = [(x.frame_id, round(x.score, 6)) for x in ranked_frames[: min(10, len(ranked_frames))]]
    _ = top_preview
    _ = t0
    return ranked_frames


def _extract_single_frame_embed(model, processor, frame: Image.Image) -> torch.Tensor:
    # Qwen3-VL 的 processor 在只给 images、不传 text 时会在内部访问 text[i] 触发 TypeError。
    # 这里显式提供一个空文本占位，确保能稳定拿到 pixel_values / image_grid_thw。
    frame_inputs = processor(text=[""], images=[frame], return_tensors="pt").to(model.device)
    if "pixel_values" not in frame_inputs:
        raise RuntimeError("单帧视觉编码失败：processor 输出缺少 pixel_values。")
    with torch.no_grad():
        image_outputs = model.get_image_features(
            pixel_values=frame_inputs["pixel_values"],
            image_grid_thw=frame_inputs.get("image_grid_thw"),
            return_dict=True,
        )
    pooled = getattr(image_outputs, "pooler_output", None)
    if not isinstance(pooled, (list, tuple)) or not pooled:
        raise RuntimeError("无法从视觉编码器输出中提取 pooler_output。")
    one = pooled[0]
    if not isinstance(one, torch.Tensor):
        raise RuntimeError("pooler_output[0] 不是 Tensor。")
    return one.detach().to(model.device)


def _build_prompt_inputs(processor, frames: list[Image.Image], prompt: str, device: torch.device):
    content: list[dict[str, Any]] = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to(device)
    return model_inputs


def _find_subsequence(seq: list[int], sub: list[int]) -> tuple[int, int]:
    if not sub:
        raise RuntimeError("答案 token 序列为空，无法计算答案 token 的置信度。")
    n = len(seq)
    m = len(sub)
    for i in range(0, n - m + 1):
        if seq[i:i + m] == sub:
            return i, i + m
    raise RuntimeError(f"在生成 token 中未找到答案 token 序列: {sub}")


def _find_subsequence_last(seq: list[int], sub: list[int]) -> tuple[int, int]:
    if not sub:
        raise RuntimeError("答案 token 序列为空，无法计算答案 token 的置信度。")
    n = len(seq)
    m = len(sub)
    for i in range(n - m, -1, -1):
        if seq[i:i + m] == sub:
            return i, i + m
    raise RuntimeError(f"在生成 token 中未找到答案 token 序列(倒序匹配): {sub}")


def _extract_generated_token_ids_for_scores(
    generated_ids: torch.Tensor,
    scores: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    score_len = len(scores)
    if score_len <= 0:
        raise RuntimeError("scores 为空，无法计算答案置信度。")
    if generated_ids.shape[1] < score_len:
        raise RuntimeError(
            f"生成 token 长度小于 scores 长度: generated={generated_ids.shape[1]}, scores={score_len}"
        )
    gen_token_ids = generated_ids[:, -score_len:]
    if gen_token_ids.shape[0] != 1:
        raise RuntimeError(f"只支持 batch=1，当前 batch={gen_token_ids.shape[0]}")
    if gen_token_ids.shape[1] == 0:
        raise RuntimeError("生成 token 为空，无法计算答案置信度。")
    return gen_token_ids


def _option_token_ids(processor, option: str) -> set[int]:
    candidates = {
        option,
        option.lower(),
        f" {option}",
        f" {option.lower()}",
    }
    token_ids: set[int] = set()
    for text in candidates:
        ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            token_ids.add(int(ids[0]))
    return token_ids


def _option_probs_from_step_logits(processor, logits: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    out: dict[str, float] = {}
    for option in ("A", "B", "C", "D"):
        token_ids = _option_token_ids(processor, option)
        if not token_ids:
            out[option] = 0.0
            continue
        p = 0.0
        for tid in token_ids:
            p += float(probs[tid].item())
        out[option] = p
    return out


def _answer_confidence_from_scores(
    processor,
    generated_ids: torch.Tensor,
    prompt_len: int,
    scores: tuple[torch.Tensor, ...],
    answer_text: str,
) -> tuple[float, float, list[float]]:
    # 仅使用与 scores 对齐的“新生成 token”来计算概率，避免把思维链前缀/提示词切片规则耦合进来。
    # generate 的不同返回形态下，generated_ids 可能是 [prompt + generated] 或 [generated]。
    gen_token_ids = _extract_generated_token_ids_for_scores(generated_ids, scores)

    step_logprobs: list[float] = []
    for t in range(gen_token_ids.shape[1]):
        logits_t = scores[t][0]
        token_id = int(gen_token_ids[0, t].item())
        lp = float(torch.log_softmax(logits_t, dim=-1)[token_id].item())
        step_logprobs.append(lp)

    answer_ids = processor.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    gen_ids_list = [int(x) for x in gen_token_ids[0].tolist()]
    try:
        # 答案 token 可能在思维链里被提前提及；取“最后一次出现”更贴近最终答案段。
        s, e = _find_subsequence_last(gen_ids_list, answer_ids)
    except RuntimeError:
        decoded_gen = processor.tokenizer.decode(gen_ids_list, skip_special_tokens=False)
        think_text, answer_text_decoded = _split_thinking_and_answer_text(decoded_gen)
        decoded_answer = processor.tokenizer.decode(answer_ids, skip_special_tokens=False)
        _ = decoded_answer
        _ = gen_ids_list
        _ = think_text
        _ = answer_text_decoded
        _ = decoded_gen
        raise
    answer_token_logprobs = step_logprobs[s:e]
    avg_answer_logprob = float(sum(answer_token_logprobs) / len(answer_token_logprobs))
    answer_prob = float(torch.exp(torch.tensor(avg_answer_logprob)).item())
    return avg_answer_logprob, answer_prob, answer_token_logprobs


def _decode_response(processor, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    prompt_len = int(input_ids.shape[1])
    token_ids = generated_ids[:, prompt_len:] if generated_ids.shape[1] > prompt_len else generated_ids
    text = processor.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def _split_thinking_and_answer_text(response: str, model_mode: str = "thinking") -> tuple[str, str]:
    if model_mode == "thinking" and "</think>" in response:
        think, answer = response.split("</think>", 1)
        return think.strip(), answer.strip()
    if model_mode == "instruct":
        return "", response.strip()
    return response.strip(), ""


def iterative_inference_with_cache(
    model,
    processor,
    ranked_frames: list[RankedFrame],
    prompt: str,
    extract_answer_fn,
    has_options: bool,
    max_new_tokens: int = 1024,
    model_mode: str = "thinking",
    enable_early_stop: bool = True,
    early_stop_window: int = 3,
    early_stop_conf_threshold: float = 0.9,
    frame_increment: int = 1,
) -> dict[str, Any]:
    if not ranked_frames:
        raise RuntimeError("ranked_frames 为空，无法进行迭代推理。")
    if frame_increment <= 0:
        raise ValueError(f"frame_increment 必须 > 0，当前为 {frame_increment}")

    visual_cache: dict[int, torch.Tensor] = {}
    cache_hits = 0
    cache_misses = 0
    total_build_time = 0.0
    total_infer_time = 0.0
    rounds = 0
    final_answer: str | float | None = None
    final_response = ""
    final_prob = 0.0
    final_logprob = float("-inf")
    final_generated_token_count = 0
    final_hit_max_tokens = False
    round_details: list[dict[str, Any]] = []
    stable_answer_for_early_stop: str | None = None
    stable_round_count = 0


    total_candidates = len(ranked_frames)
    round_frame_counts = list(range(frame_increment, total_candidates + 1, frame_increment))
    if not round_frame_counts or round_frame_counts[-1] != total_candidates:
        round_frame_counts.append(total_candidates)

    for round_idx, frame_count in enumerate(round_frame_counts, start=1):
        rounds = round_idx
        selected_by_score = ranked_frames[:frame_count]
        # top-k 按分数选出后，构建视觉输入时按时间顺序排列，保持视频时序一致。
        selected = sorted(selected_by_score, key=lambda x: x.frame_id)
        selected_frames = [x.image for x in selected]
        selected_ids = [x.frame_id for x in selected]
        selected_scores = [x.score for x in selected]
        frame_score_pairs = [(x.frame_id, round(x.score, 6)) for x in selected]
        _log(f"第{round_idx}轮选帧(frame_id, clip_score): {frame_score_pairs}")

        build_start = time.perf_counter()
        model_inputs = _build_prompt_inputs(processor, selected_frames, prompt, model.device)
        token_embeds = model.get_input_embeddings()(model_inputs["input_ids"])

        image_chunks: list[torch.Tensor] = []
        for frame in selected:
            if frame.frame_id in visual_cache:
                cache_hits += 1
                image_chunks.append(visual_cache[frame.frame_id])
            else:
                cache_misses += 1
                embed = _extract_single_frame_embed(model, processor, frame.image)
                visual_cache[frame.frame_id] = embed
                image_chunks.append(embed)

        image_embeds = torch.cat(image_chunks, dim=0).to(model.device)
        image_mask, _ = model.model.get_placeholder_mask(
            model_inputs["input_ids"],
            inputs_embeds=token_embeds,
            image_features=image_embeds,
        )
        mask_elements = int(image_mask.sum().item())
        image_embed_elements = int(image_embeds.numel())
        if mask_elements != image_embed_elements:
            raise RuntimeError(
                "视觉占位元素数量不匹配: "
                f"mask_elements={mask_elements}, image_embed_elements={image_embed_elements}, "
                f"image_mask_shape={tuple(image_mask.shape)}, image_embeds_shape={tuple(image_embeds.shape)}"
            )
        fused_embeds = token_embeds.masked_scatter(image_mask, image_embeds)
        build_time = time.perf_counter() - build_start
        total_build_time += build_time

        infer_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=fused_embeds,
                attention_mask=model_inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        infer_time = time.perf_counter() - infer_start
        total_infer_time += infer_time

        generated_ids = outputs.sequences
        prompt_len = int(model_inputs["input_ids"].shape[1])
        generated_token_count = int(len(outputs.scores))
        hit_max_tokens = generated_token_count >= max_new_tokens
        response = _decode_response(processor, model_inputs["input_ids"], generated_ids)
        think_text, answer_text = _split_thinking_and_answer_text(
            response=response,
            model_mode=model_mode,
        )
        has_think_end = "</think>" in response
        pred_answer = extract_answer_fn(response, has_options=has_options)
        should_compute_confidence = (model_mode == "instruct") or has_think_end
        if should_compute_confidence:
            try:
                avg_logprob, answer_prob, token_lps = _answer_confidence_from_scores(
                    processor=processor,
                    generated_ids=generated_ids,
                    prompt_len=prompt_len,
                    scores=outputs.scores,
                    answer_text=str(pred_answer),
                )
                gen_token_ids = _extract_generated_token_ids_for_scores(generated_ids, outputs.scores)
                answer_ids = processor.tokenizer(str(pred_answer), add_special_tokens=False)["input_ids"]
                s, _ = _find_subsequence_last([int(x) for x in gen_token_ids[0].tolist()], answer_ids)
                option_probs = _option_probs_from_step_logits(processor, outputs.scores[s][0])
            except RuntimeError as e:
                _ = e
                avg_logprob = float("-inf")
                answer_prob = 0.0
                token_lps = []
                option_probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        else:
            avg_logprob = float("-inf")
            answer_prob = 0.0
            token_lps = []
            option_probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        _log(
            f"第{round_idx}轮结果: infer_time={infer_time:.4f}s, pred_answer={pred_answer}, "
            f"answer_avg_logprob={avg_logprob:.6f}, answer_prob={answer_prob:.6f}, "
            f"option_probs(A/B/C/D)={[round(option_probs[x], 6) for x in ('A', 'B', 'C', 'D')]}, "
            f"answer_token_logprobs={[round(x, 6) for x in token_lps]}, "
            f"generated_tokens={generated_token_count}, limit={max_new_tokens}, hit_limit={hit_max_tokens}"
        )

        pred_answer_str = str(pred_answer).strip().upper() if pred_answer is not None else ""
        selected_option_prob = (
            float(option_probs.get(pred_answer_str, 0.0))
            if (has_options and pred_answer_str in {"A", "B", "C", "D"})
            else 0.0
        )
        if (
            enable_early_stop
            and has_options
            and pred_answer_str in {"A", "B", "C", "D"}
            and selected_option_prob > early_stop_conf_threshold
        ):
            if pred_answer_str == stable_answer_for_early_stop:
                stable_round_count += 1
            else:
                stable_answer_for_early_stop = pred_answer_str
                stable_round_count = 1
        else:
            stable_answer_for_early_stop = None
            stable_round_count = 0

        if pred_answer is not None:
            final_answer = pred_answer
        final_response = response
        final_prob = answer_prob
        final_logprob = avg_logprob
        final_generated_token_count = generated_token_count
        final_hit_max_tokens = hit_max_tokens
        round_details.append(
            {
                "round_idx": round_idx,
                "frame_count": frame_count,
                "selected_frame_ids": selected_ids,
                "selected_clip_scores": selected_scores,
                "embedding_build_time": build_time,
                "inference_time": infer_time,
                "pred_answer": pred_answer,
                "answer_prob": answer_prob,
                "answer_logprob": avg_logprob,
                "answer_token_logprobs": token_lps,
                "option_probs": option_probs,
                "generated_token_count": generated_token_count,
                "hit_max_tokens": hit_max_tokens,
                "has_think_end": has_think_end,
                "selected_option_prob": selected_option_prob,
            }
        )
        if enable_early_stop and stable_round_count >= early_stop_window:
            _log(
                f"第{round_idx}轮触发早停: 连续{stable_round_count}轮答案={stable_answer_for_early_stop}，"
                f"且所选选项概率>{early_stop_conf_threshold}"
            )
            break

    if final_answer is None:
        final_answer = extract_answer_fn(final_response, has_options=has_options)
        if final_answer is None:
            final_answer = ""

    return {
        "response": final_response,
        "pred_answer": final_answer,
        "answer_confidence": final_prob,
        "answer_logprob": final_logprob,
        "rounds_used": rounds,
        "cache_size": len(visual_cache),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "embedding_build_time": total_build_time,
        "inference_time": total_infer_time,
        "generated_token_count": final_generated_token_count,
        "hit_max_tokens": final_hit_max_tokens,
        "round_details": round_details,
    }
