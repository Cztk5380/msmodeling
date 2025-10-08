from typing import Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("static_quant_linear")
@register_tensor_cast_op("static_quant_linear_int4")
def _static_quant_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w_offset: Optional[torch.Tensor],
    x_scale: Optional[torch.Tensor],
    x_offset: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """
    Semantically equivalent to
    `out = (dequant(x) @ dequant(w) + bias).to_dtype(out_dtype)` if x_scale is not None
    `out = (x @ dequant(w) + bias).to_dtype(out_dtype)` otherwise

    Args:
        x: (M, K)
        x_scale: scalar, (M,)
        x_offset: scalar, (M,)
        w: (K, N), for int4 (K/2, N) or (K, N/2)
        w_scale: scalar, (N,) or (K_group, N)
        w_offset: scalar, (N,) or (K_group, N)
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
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
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


@register_tensor_cast_op("mxfp4_linear")
def _mxfp4_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """
    MXFP4 linear operation. Both x and w are in MXFP4 format with logical
    dimension as (M, K) and (K, N). The MXFP4 format is logically represented
    with torch.int4.

    Semantically equivalent to:
    `out = (dequant_mxfp4(x) @ dequant_mxfp4(w) + bias).to_dtype(out_dtype)`

    Args:
        x: (M, K) in MXFP4 format
        w: (K, N) in MXFP4 format
        x_scale: (Kg,) for activation scale in torch.float8_e8m0fnu
        w_scale: (Kg,) for weight scale in torch.float8_e8m0fnu
        bias: scalar or (N,) bias tensor
        out_dtype: output data type
    """
    if out_dtype is None:
        out_dtype = torch.float16  # Default output for MXFP4
    return torch.empty((x.shape[0], w.shape[1]), dtype=out_dtype, device="meta")
