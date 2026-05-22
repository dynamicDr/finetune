from __future__ import annotations

import torch

from vqa_eval_ours import (
    assert_metadata_complete,
    assert_prefix_property,
    build_layered_sequence,
)


def test_prefix_property_and_metadata() -> None:
    torch.manual_seed(1234)
    patch_features = torch.randn(6, 4, 4, 16)
    scores = torch.randn(6, 4, 4)
    frame_ids = [i * 5 for i in range(6)]
    sequence = build_layered_sequence(
        patch_features=patch_features,
        scores=scores,
        frame_ids=frame_ids,
        pool_h=2,
        pool_w=2,
        enable_layer0=True,
        enable_layer1=True,
        layer1_mode="global_patch_sort",
    )
    assert len(sequence) > 0
    assert_prefix_property(sequence, budget_small=16, budget_large=64)
    assert_metadata_complete(sequence)
