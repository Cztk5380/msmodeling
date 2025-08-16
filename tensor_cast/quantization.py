from typing import Optional
import torch
from .utils import register_tensor_cast_op


@register_tensor_cast_op("quantize")
def _quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor]=None,
    out_dtype: torch.dtype=torch.int8
) -> torch.Tensor:
    """`out = clamp(round(x / scale) + offset)`"""
    return torch.empty_like(x, dtype=out_dtype)