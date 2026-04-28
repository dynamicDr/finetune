from __future__ import annotations

import argparse
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def decode_new_tokens(processor, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    prompt_len = int(input_ids.shape[1])
    token_ids = generated_ids[:, prompt_len:] if generated_ids.shape[1] > prompt_len else generated_ids
    text = processor.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def pick_tensor(x: object) -> torch.Tensor | None:
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "image_embeds") and isinstance(x.image_embeds, torch.Tensor):
        return x.image_embeds
    if hasattr(x, "last_hidden_state") and isinstance(x.last_hidden_state, torch.Tensor):
        return x.last_hidden_state
    if hasattr(x, "__dict__"):
        for value in x.__dict__.values():
            if isinstance(value, torch.Tensor):
                return value
    return None


def build_image_fused_embeds(model, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if "pixel_values" not in model_inputs:
        return None

    image_grid_thw = model_inputs.get("image_grid_thw")
    with torch.no_grad():
        image_outputs = model.get_image_features(
            pixel_values=model_inputs["pixel_values"],
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )

    image_embed_seq = image_outputs.pooler_output
    if not isinstance(image_embed_seq, (list, tuple)) or len(image_embed_seq) == 0:
        return None

    image_embeds = torch.cat(image_embed_seq, dim=0).to(model.device)
    token_embeds = model.get_input_embeddings()(model_inputs["input_ids"])
    image_mask, _ = model.model.get_placeholder_mask(
        model_inputs["input_ids"],
        inputs_embeds=token_embeds,
        image_features=image_embeds,
    )
    fused_embeds = token_embeds.masked_scatter(image_mask, image_embeds)
    return fused_embeds


def sync_cuda_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="只测两条路径：图片+指令、embedding+指令")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Thinking")
    parser.add_argument("--image_path", type=str, default="sandbox/dataset/image/plane2.png")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device: {device}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()
    if device == "cpu":
        model.to(device)

    image = Image.open(args.image_path).convert("RGB")
    prompt = "请用一句简短中文描述这张图片的主要内容。"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 1) 标准输入：图片 + 指令
    standard_inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        standard_ids = model.generate(
            **standard_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    standard_text = decode_new_tokens(processor, standard_inputs["input_ids"], standard_ids)
    print("\n[测试1] 标准输入（图片 + 指令）")
    print(standard_text)

    # 2) 先图片 -> embedding，再输入：embedding + 指令
    print("\n[测试2] 先提图像 embedding，再喂（embedding + 指令）")
    if not hasattr(model, "get_image_features") or not hasattr(model, "model"):
        print("模型没有 get_image_features，无法测试 embedding 路径。")
        return

    test2_start = time.perf_counter()
    sync_cuda_if_needed(str(model.device))
    embed_build_start = time.perf_counter()
    fused_embeds = build_image_fused_embeds(model, standard_inputs)
    sync_cuda_if_needed(str(model.device))
    embed_build_cost = time.perf_counter() - embed_build_start
    if fused_embeds is None:
        print("图像 embedding 构造失败。")
        return

    sync_cuda_if_needed(str(model.device))
    infer_start = time.perf_counter()
    with torch.no_grad():
        embed_ids = model.generate(
            inputs_embeds=fused_embeds,
            attention_mask=standard_inputs.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    sync_cuda_if_needed(str(model.device))
    infer_cost = time.perf_counter() - infer_start
    embed_text = decode_new_tokens(processor, standard_inputs["input_ids"], embed_ids)
    test2_total_cost = time.perf_counter() - test2_start

    print(f"图片 -> embedding 耗时: {embed_build_cost:.3f}s")
    print(f"embedding + 指令 推理耗时: {infer_cost:.3f}s")
    print(f"测试2总耗时: {test2_total_cost:.3f}s")
    print(embed_text)


if __name__ == "__main__":
    main()
