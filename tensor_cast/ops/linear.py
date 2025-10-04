from typing import Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("static_quant_linear")
@register_tensor_cast_op("static_quant_linear_int4")
def _static_quant_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w_offset: Optional[torch.Tensor] = None,
    x_scale: Optional[torch.Tensor] = None,
    x_offset: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Semantically equivalent to
    `out = (dequant(x) @ dequant(w) + bias).to_dtype(out_dtype)` if x_scale is not None
    `out = (x @ dequant(w) + bias).to_dtype(out_dtype)` otherwise

    Args:
        x: (M, K)
        x_scale: scalar, (1, M)
        x_offset: scalar, (1, M)
        w: (K, N), for int4 (K/2, N) or (K, N/2)
        w_scale: scalar, (1, N) or (K_group, N)
        w_offset: scalar, (1, N) or (K_group, N)
        bias: scalar, (1, N)
    """
    if out_dtype is None:
        out_dtype = x.dtype
    return torch.empty((x.shape[0], w.shape[1]), dtype=out_dtype, device="meta")


@register_tensor_cast_op("fp8_linear")
def _fp8_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    FP8 linear operation. Both x and w are in FP8 format.
    Semantically equivalent to:
    `out = (dequant_fp8(x) @ dequant_fp8(w) + bias).to_dtype(out_dtype)`

    Args:
        x: (M, K) in FP8 format
        w: (K, N) in FP8 format
        x_scale: scalar or (M,) for activation scale
        w_scale: scalar or (N,) for weight scale
        bias: scalar or (N,) bias tensor
        out_dtype: output data type
    """
    if out_dtype is None:
        out_dtype = torch.float16  # Default output for FP8
    return torch.empty((x.shape[0], w.shape[1]), dtype=out_dtype, device="meta")
