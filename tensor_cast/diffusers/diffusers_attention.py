from aenum import extend_enum
import torch
import diffusers

from diffusers.models.attention_dispatch import _AttentionBackendRegistry, AttentionBackendName


extend_enum(diffusers.models.attention_dispatch.AttentionBackendName, "TENSOR_CAST", "tensor_cast")


@_AttentionBackendRegistry.register("tensor_cast")
def _attention(
    query, key, value, **kwargs
):
    return torch.ops.tensor_cast.attention(query, key, value, None, None, None, None, None)
