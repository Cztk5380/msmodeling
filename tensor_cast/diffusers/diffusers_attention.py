from contextlib import contextmanager

import diffusers
import torch
import torch.nn.functional as F
from aenum import extend_enum

from diffusers.models.attention_dispatch import _AttentionBackendRegistry


extend_enum(
    diffusers.models.attention_dispatch.AttentionBackendName,
    "TENSOR_CAST",
    "tensor_cast",
)


@_AttentionBackendRegistry.register("tensor_cast")
def _attention(query, key, value, **kwargs):
    return torch.ops.tensor_cast.multimodal_attention(
        query, key, value, None, None, None, None, None
    )


@contextmanager
def use_custom_sdpa():
    original_sdpa = F.scaled_dot_product_attention

    def _custom_sdpa(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        return torch.ops.tensor_cast.multimodal_attention(
            q, k, v, attn_mask, None, None, None, None
        )

    F.scaled_dot_product_attention = _custom_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa
