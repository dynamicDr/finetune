from __future__ import annotations

import cv2
from PIL import Image


def sample_uniform_frames(
    video_path: str,
    num_frames: int,
    random_seed: int | None = None,
    question: str | None = None,
    answer: str | None = None,
) -> list[Image.Image]:
    _ = random_seed
    _ = question
    _ = answer
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames: list[Image.Image] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

