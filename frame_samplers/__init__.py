from __future__ import annotations

from PIL import Image

from .random import sample_random_frames
from .uniform import sample_uniform_frames


SUPPORTED_FRAME_SAMPLERS = ("uniform", "random", "focus", "sevila", "videoagent", "clip", "siglip2", "aks")


def sample_video_frames(
    video_path: str,
    num_frames: int = 8,
    method: str = "uniform",
    random_seed: int | None = None,
    question: str | None = None,
    answer: str | None = None,
    focus_blip_model_name: str = "Salesforce/blip-itm-large-coco",
    focus_blip_device: str | None = None,
    focus_blip_batch_size: int = 16,
) -> list[Image.Image]:
    method = method.strip().lower()
    if method == "uniform":
        return sample_uniform_frames(
            video_path,
            num_frames,
            random_seed=random_seed,
            question=question,
            answer=answer,
        )
    if method == "random":
        return sample_random_frames(
            video_path,
            num_frames,
            random_seed=random_seed,
            question=question,
            answer=answer,
        )
    if method == "focus":
        from .focus import sample_focus_frames

        return sample_focus_frames(
            video_path,
            num_frames,
            question=question,
            answer=answer,
            random_seed=random_seed,
            blip_model_name=focus_blip_model_name,
            blip_device=focus_blip_device,
            blip_batch_size=focus_blip_batch_size,
        )
    if method == "sevila":
        from .sevila import sample_sevila_frames

        return sample_sevila_frames(
            video_path,
            num_frames,
            question=question,
            answer=answer,
            random_seed=random_seed,
            blip_model_name=focus_blip_model_name,
            blip_device=focus_blip_device,
            blip_batch_size=focus_blip_batch_size,
        )
    if method == "videoagent":
        from .videoagent import sample_videoagent_frames

        return sample_videoagent_frames(
            video_path,
            num_frames,
            question=question,
            answer=answer,
            random_seed=random_seed,
        )
    if method == "clip":
        from .clip import sample_clip_frames

        return sample_clip_frames(
            video_path,
            num_frames,
            question=question,
            answer=answer,
            random_seed=random_seed,
        )
    if method == "siglip2":
        from .siglip2 import sample_siglip2_frames

        return sample_siglip2_frames(
            video_path,
            num_frames,
            question=question,
            answer=answer,
            random_seed=random_seed,
        )
    if method == "aks":
        from .aks import sample_aks_frames

        return sample_aks_frames(
            video_path,
            num_frames,
            question=question,
            answer=answer,
            random_seed=random_seed,
        )
    raise ValueError(f"不支持的选帧方法: {method}，可选: {', '.join(SUPPORTED_FRAME_SAMPLERS)}")

