from __future__ import annotations

from PIL import Image

from .random import sample_random_frames
from .uniform import sample_uniform_frames


SUPPORTED_FRAME_SAMPLERS = (
    "uniform",
    "random",
    "focus",
    "sevila",
    "videoagent",
    "clip",
    "siglip2",
    "qframe",
    "bolt-clip",
    "bolt-siglip2",
    "aks",
    "aks-blip",
    "aks-clip",
)


def sample_video_frames(
    video_path: str,
    num_frames: int = 8,
    method: str = "uniform",
    random_seed: int | None = None,
    sample_id: str | None = None,
    question: str | None = None,
    options: list[str] | None = None,
    answer: str | None = None,
    focus_blip_model_name: str = "Salesforce/blip-itm-large-coco",
    focus_blip_device: str | None = None,
    focus_blip_batch_size: int = 16,
    use_preprocessed_clip_frames: bool = False,
    preprocessed_clip_dir: str | None = None,
) -> list[Image.Image]:
    method = method.strip().lower()
    if method == "uniform":
        return sample_uniform_frames(
            video_path,
            num_frames,
            random_seed=random_seed,
            sample_id=sample_id,
            question=question,
            answer=answer,
            use_preprocessed_clip_frames=use_preprocessed_clip_frames,
            preprocessed_clip_dir=preprocessed_clip_dir,
        )
    if method == "random":
        return sample_random_frames(
            video_path,
            num_frames,
            random_seed=random_seed,
            sample_id=sample_id,
            question=question,
            answer=answer,
            use_preprocessed_clip_frames=use_preprocessed_clip_frames,
            preprocessed_clip_dir=preprocessed_clip_dir,
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
            sample_id=sample_id,
            question=question,
            options=options,
            answer=answer,
            random_seed=random_seed,
            use_preprocessed_clip_frames=use_preprocessed_clip_frames,
            preprocessed_clip_dir=preprocessed_clip_dir,
        )
    if method == "siglip2":
        from .siglip2 import sample_siglip2_frames

        return sample_siglip2_frames(
            video_path,
            num_frames,
            sample_id=sample_id,
            question=question,
            options=options,
            answer=answer,
            random_seed=random_seed,
            use_preprocessed_clip_frames=use_preprocessed_clip_frames,
            preprocessed_clip_dir=preprocessed_clip_dir,
        )
    if method == "qframe":
        from .qframe import sample_qframe_frames

        return sample_qframe_frames(
            video_path,
            num_frames,
            sample_id=sample_id,
            question=question,
            options=options,
            answer=answer,
            random_seed=random_seed,
            use_preprocessed_clip_frames=use_preprocessed_clip_frames,
            preprocessed_clip_dir=preprocessed_clip_dir,
        )
    if method in {"bolt-clip", "bolt-siglip2"}:
        from .bolt import sample_bolt_frames

        bolt_feature_model = "clip" if method == "bolt-clip" else "siglip2"
        return sample_bolt_frames(
            video_path,
            num_frames,
            sample_id=sample_id,
            question=question,
            options=options,
            answer=answer,
            random_seed=random_seed,
            extract_feature_model=bolt_feature_model,
            use_preprocessed_clip_frames=use_preprocessed_clip_frames,
            preprocessed_clip_dir=preprocessed_clip_dir,
        )
    if method in {"aks", "aks-blip", "aks-clip"}:
        from .aks import sample_aks_frames

        aks_feature_model = "blip" if method in {"aks", "aks-blip"} else "clip"
        return sample_aks_frames(
            video_path,
            num_frames,
            sample_id=sample_id,
            question=question,
            options=options,
            answer=answer,
            random_seed=random_seed,
            extract_feature_model=aks_feature_model,
            use_preprocessed_clip_frames=use_preprocessed_clip_frames,
            preprocessed_clip_dir=preprocessed_clip_dir,
        )
    raise ValueError(f"不支持的选帧方法: {method}，可选: {', '.join(SUPPORTED_FRAME_SAMPLERS)}")

