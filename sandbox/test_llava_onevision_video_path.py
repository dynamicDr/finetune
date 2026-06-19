#!/usr/bin/env python3
"""验证 LLaVA-OneVision 走官方 video 路径（~196 visual token/帧）。"""

from __future__ import annotations

import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vl_common import (
    collect_visual_token_ids,
    is_llava_hf_video_inference,
    is_llava_onevision_video_inference,
    prepare_vlm_inputs,
)

MODEL_ID = os.environ.get(
    "LLAVA_OV_MODEL",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
)
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "8"))
EXPECTED_TOKENS_PER_FRAME = 196


def main() -> None:
    from transformers import AutoProcessor

    print(f"[test] loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    assert is_llava_onevision_video_inference(processor), (
        "应识别为 LLaVA-OneVision video 推理路径"
    )
    assert is_llava_hf_video_inference(processor)

    frames = [
        Image.new("RGB", (1920, 1080), color=(i * 7 % 255, 20, 40))
        for i in range(NUM_FRAMES)
    ]
    prompt = "Describe this video briefly."

    inputs, _ = prepare_vlm_inputs(processor, frames, prompt)
    visual_token_ids = collect_visual_token_ids(processor)
    visual_token_count = int(
        sum((inputs["input_ids"] == tid).sum().item() for tid in visual_token_ids)
    )

    assert "pixel_values_videos" in inputs, "应使用 pixel_values_videos（video 路径）"
    assert "pixel_values" not in inputs, "不应走多图 pixel_values 路径"
    pv = inputs["pixel_values_videos"]
    assert tuple(pv.shape[1:3]) == (NUM_FRAMES, 3), f"视频帧数不匹配: {pv.shape}"

    expected = NUM_FRAMES * EXPECTED_TOKENS_PER_FRAME
    tolerance = NUM_FRAMES  # 允许每帧 ±1 的舍入
    print(
        f"[test] frames={NUM_FRAMES}, visual_tokens={visual_token_count}, "
        f"expected≈{expected}, pixel_values_videos.shape={tuple(pv.shape)}, "
        f"seq_len={inputs['input_ids'].shape[1]}"
    )
    assert abs(visual_token_count - expected) <= tolerance, (
        f"visual token 数异常: got {visual_token_count}, expected ~{expected}"
    )
    print("[test] PASS: LLaVA-OneVision 官方 video 路径工作正常")


if __name__ == "__main__":
    main()
