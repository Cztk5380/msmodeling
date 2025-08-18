import torch
from typing import Optional
from ..utils import register_tensor_cast_op


@register_tensor_cast_op("dynamic_quant_linear")
@register_tensor_cast_op("dynamic_quant_linear_int4")
def _dynamic_quant_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w_offset: Optional[torch.Tensor]=None,
    bias: Optional[torch.Tensor]=None,
    out_dtype: Optional[torch.dtype]=None
) -> torch.Tensor:
    """
    Semantically equivalent to
    `out = (fake_quant(x) @ dequant(w) + bias).to_dtype(out_dtype)`

    Args:
        x: (M, K)
        w: (K, N), for int4 (K/2, N) or (K, N/2)
        w_scale: scalar, (1, N) or (K_group, N)
        w_offset: scalar, (1, N)
        bias: scalar, (1, N)
    """
    if out_dtype is None:
        out_dtype = x.dtype
    return torch.empty((x.shape[0], w.shape[1]), dtype=out_dtype, device="meta")


@register_tensor_cast_op("static_quant_linear")
@register_tensor_cast_op("static_quant_linear_int4")
def _static_quant_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w_offset: Optional[torch.Tensor]=None,
    x_scale: Optional[torch.Tensor]=None,
    x_offset: Optional[torch.Tensor]=None,
    bias: Optional[torch.Tensor]=None,
    out_dtype: Optional[torch.dtype]=None
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
