from __future__ import annotations


def select_frame_positions_from_scores(
    scores,
    frame_ids: list[int],
    num_frames: int,
    use_segment_selection: bool,
) -> list[int]:
    """按 CLIP/SigLIP2 分数选帧：分段每段 1 帧，或全局 top-k。"""
    n_candidates = len(frame_ids)
    if n_candidates <= 0 or num_frames <= 0:
        return []

    if use_segment_selection:
        n_segments = min(num_frames, n_candidates)
        selected_positions: list[int] = []
        for seg_idx in range(n_segments):
            seg_start = int(seg_idx * n_candidates / n_segments)
            seg_end = int((seg_idx + 1) * n_candidates / n_segments)
            if seg_start >= seg_end:
                continue
            seg_scores = scores[seg_start:seg_end]
            rel_best = int(seg_scores.argmax().item())
            selected_positions.append(seg_start + rel_best)
    else:
        max_pick = min(num_frames, n_candidates)
        ranked = scores.argsort(descending=True).tolist()
        selected_positions = ranked[:max_pick]

    return sorted(set(selected_positions), key=lambda i: frame_ids[i])
