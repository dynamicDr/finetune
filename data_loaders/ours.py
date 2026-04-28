from __future__ import annotations

import re
import time
from dataclasses import dataclass
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
        _log(f"复用 CLIP 缓存: model={model_id}, device={resolved_device}")
        return _CLIP_CACHE[cache_key]
    _log(f"加载 CLIP: model={model_id}, device={resolved_device}")
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
        _log(
            "sample_every<=0，启用默认1fps粗采样: "
            f"video_fps={video_fps:.4f}, resolved_sample_every={resolved_sample_every}"
        )
    else:
        resolved_sample_every = sample_every
        _log(f"使用指定粗采样步长: sample_every={resolved_sample_every}")
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
    _log(f"粗采样完成: 候选帧数={len(images)}, sample_every={resolved_sample_every}")
    if not images:
        raise RuntimeError(f"视频无可用候选帧: {video_path}")
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
        _log(f"CLIP 图像编码 batch: start={start}, size={len(batch)}")
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
) -> list[RankedFrame]:
    t0 = time.perf_counter()
    frame_ids, images = _collect_candidate_frames(video_path, sample_every=coarse_sample_every)
    processor, model, resolved_device = _load_clip(clip_model_id, clip_device)
    query = _build_clip_query(question=question, options=options, answer=answer)
    _log(f"CLIP query: {(query[:200] + '...') if len(query) > 200 else query}")

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
    _log(
        f"CLIP 排序完成: 总帧数={len(ranked_frames)}, top预览={top_preview}, "
        f"耗时={time.perf_counter() - t0:.3f}s"
    )
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


def _answer_confidence_from_scores(
    processor,
    generated_ids: torch.Tensor,
    prompt_len: int,
    scores: tuple[torch.Tensor, ...],
    answer_text: str,
) -> tuple[float, float, list[float]]:
    if generated_ids.shape[1] > prompt_len:
        gen_token_ids = generated_ids[:, prompt_len:]
    else:
        gen_token_ids = generated_ids
    if gen_token_ids.shape[0] != 1:
        raise RuntimeError(f"只支持 batch=1，当前 batch={gen_token_ids.shape[0]}")
    if gen_token_ids.shape[1] == 0:
        raise RuntimeError("生成 token 为空，无法计算答案置信度。")
    if len(scores) != gen_token_ids.shape[1]:
        raise RuntimeError(
            f"scores 长度与生成 token 数不一致: scores={len(scores)}, tokens={gen_token_ids.shape[1]}"
        )

    step_logprobs: list[float] = []
    for t in range(gen_token_ids.shape[1]):
        logits_t = scores[t][0]
        token_id = int(gen_token_ids[0, t].item())
        lp = float(torch.log_softmax(logits_t, dim=-1)[token_id].item())
        step_logprobs.append(lp)

    answer_ids = processor.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    gen_ids_list = [int(x) for x in gen_token_ids[0].tolist()]
    try:
        s, e = _find_subsequence(gen_ids_list, answer_ids)
    except RuntimeError:
        decoded_gen = processor.tokenizer.decode(gen_ids_list, skip_special_tokens=False)
        think_text, answer_text_decoded = _split_thinking_and_answer_text(decoded_gen)
        decoded_answer = processor.tokenizer.decode(answer_ids, skip_special_tokens=False)
        _log("答案 token 对齐失败，下面是调试信息：")
        _log(f"answer_text(raw)={answer_text!r}")
        _log(f"answer_ids={answer_ids}")
        _log(f"answer_ids_decode={decoded_answer!r}")
        _log(f"generated_ids={gen_ids_list}")
        _log(f"generated_think_text={think_text!r}")
        _log(f"generated_final_answer_text={answer_text_decoded!r}")
        _log(f"generated_decode={decoded_gen!r}")
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


def _split_thinking_and_answer_text(response: str) -> tuple[str, str]:
    if "</think>" in response:
        think, answer = response.split("</think>", 1)
        return think.strip(), answer.strip()
    return response.strip(), ""


def iterative_inference_with_cache(
    model,
    processor,
    ranked_frames: list[RankedFrame],
    prompt: str,
    extract_answer_fn,
    has_options: bool,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    if not ranked_frames:
        raise RuntimeError("ranked_frames 为空，无法进行迭代推理。")

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

    _log(f"开始迭代推理: 总候选帧={len(ranked_frames)}，固定跑满top-1到top-{len(ranked_frames)}")

    for k in range(1, len(ranked_frames) + 1):
        rounds = k
        selected = ranked_frames[:k]
        selected_frames = [x.image for x in selected]
        selected_ids = [x.frame_id for x in selected]
        selected_scores = [x.score for x in selected]
        _log(
            f"第{k}轮: 使用top-{k}帧, frame_ids={selected_ids}, "
            f"clip_scores={[round(s, 6) for s in selected_scores]}"
        )

        build_start = time.perf_counter()
        model_inputs = _build_prompt_inputs(processor, selected_frames, prompt, model.device)
        token_embeds = model.get_input_embeddings()(model_inputs["input_ids"])

        image_chunks: list[torch.Tensor] = []
        for frame in selected:
            if frame.frame_id in visual_cache:
                cache_hits += 1
                _log(f"视觉缓存命中: frame_id={frame.frame_id}")
                image_chunks.append(visual_cache[frame.frame_id])
            else:
                cache_misses += 1
                _log(f"视觉缓存未命中，开始编码: frame_id={frame.frame_id}")
                embed = _extract_single_frame_embed(model, processor, frame.image)
                visual_cache[frame.frame_id] = embed
                _log(f"视觉编码完成并写入缓存: frame_id={frame.frame_id}, shape={tuple(embed.shape)}")
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
        _log(
            f"第{k}轮 embedding构建完成: build_time={build_time:.4f}s, "
            f"cache_size={len(visual_cache)}, hit={cache_hits}, miss={cache_misses}, "
            f"image_mask_shape={tuple(image_mask.shape)}, image_embeds_shape={tuple(image_embeds.shape)}"
        )

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
        generated_token_count = int(generated_ids.shape[1] - prompt_len)
        hit_max_tokens = generated_token_count >= max_new_tokens
        response = _decode_response(processor, model_inputs["input_ids"], generated_ids)
        think_text, answer_text = _split_thinking_and_answer_text(response)
        has_think_end = "</think>" in response
        if has_think_end:
            pred_answer = extract_answer_fn(response, has_options=has_options)
            avg_logprob, answer_prob, token_lps = _answer_confidence_from_scores(
                processor=processor,
                generated_ids=generated_ids,
                prompt_len=prompt_len,
                scores=outputs.scores,
                answer_text=str(pred_answer),
            )
        else:
            pred_answer = None
            avg_logprob = float("-inf")
            answer_prob = 0.0
            token_lps = []
            _log("第{k}轮未检测到 </think> 结束符，按无效答案处理。".format(k=k))
        _log(
            f"第{k}轮结果: infer_time={infer_time:.4f}s, pred_answer={pred_answer}, "
            f"answer_avg_logprob={avg_logprob:.6f}, answer_prob={answer_prob:.6f}, "
            f"answer_token_logprobs={[round(x, 6) for x in token_lps]}, "
            f"generated_tokens={generated_token_count}, limit={max_new_tokens}, hit_limit={hit_max_tokens}"
        )
        _log(f"第{k}轮思维链输出:\n{think_text}")
        _log(f"第{k}轮最终回复输出:\n{answer_text}")
        _log(f"第{k}轮大模型完整输出:\n{response}")

        final_answer = pred_answer
        final_response = response
        final_prob = answer_prob
        final_logprob = avg_logprob
        final_generated_token_count = generated_token_count
        final_hit_max_tokens = hit_max_tokens
        _log(f"第{k}轮结束：不启用早停，继续下一轮。")

    if final_answer is None:
        raise RuntimeError("迭代推理未产生任何答案。")

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
    }
