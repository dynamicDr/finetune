"""Compatibility shims for LLaVA-NeXT against transformers 5.x.

The installed ``llava`` package still imports helpers from
``transformers.modeling_utils`` that were moved or removed in transformers 5.x.
Apply these patches before importing ``llava`` or lmms-eval's ``llava_vid``.
"""
from __future__ import annotations

import transformers.modeling_utils as modeling_utils
from transformers import pytorch_utils

_APPLIED = False


def _find_pruneable_heads_and_indices(
    heads: list[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
):
    import torch

    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def _ensure_config_rope_parameters(config) -> None:
    if config.rope_parameters is not None:
        return
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        config.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}


def _patch_llava_language_model_inits() -> None:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    from llava.model.language_model import llava_qwen

    if not getattr(llava_qwen, "_finetune_transformers5_compat", False):
        import torch.nn as nn

        def _patched_qwen_init(self, config):
            Qwen2ForCausalLM.__init__(self, config)
            config.model_type = "llava_qwen"
            _ensure_config_rope_parameters(config)
            self.model = llava_qwen.LlavaQwenModel(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        llava_qwen.LlavaQwenForCausalLM.__init__ = _patched_qwen_init
        llava_qwen._finetune_transformers5_compat = True

    try:
        from llava.model.language_model import llava_qwen_moe
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM

        if not getattr(llava_qwen_moe, "_finetune_transformers5_compat", False):
            import torch.nn as nn

            def _patched_moe_init(self, config):
                Qwen2MoeForCausalLM.__init__(self, config)
                config.model_type = "llava_qwen_moe"
                _ensure_config_rope_parameters(config)
                self.model = llava_qwen_moe.LlavaQwenMoeModel(config)
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                self.post_init()

            llava_qwen_moe.LlavaQwenMoeForCausalLM.__init__ = _patched_moe_init
            llava_qwen_moe._finetune_transformers5_compat = True
    except Exception:
        pass


def apply_llava_transformers_compat() -> None:
    global _APPLIED
    if _APPLIED:
        return

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = pytorch_utils.apply_chunking_to_forward
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = pytorch_utils.prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        modeling_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

    _patch_llava_language_model_inits()

    _APPLIED = True


apply_llava_transformers_compat()
