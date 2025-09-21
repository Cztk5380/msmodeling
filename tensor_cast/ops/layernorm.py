from typing import Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("rms_norm")
def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x).contiguous()


@register_tensor_cast_op("rms_norm_quant")
def _rms_norm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=out_dtype).contiguous()


@register_tensor_cast_op("add_rms_norm")
def _add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x).contiguous()


@register_tensor_cast_op("add_rms_norm2")
def _add_rms_norm2(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x).contiguous(), torch.empty_like(x).contiguous()


@register_tensor_cast_op("add_rms_norm_quant")
def _add_rms_norm_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=out_dtype).contiguous()


@register_tensor_cast_op("add_rms_norm_quant2")
def _add_rms_norm_quant2(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x, dtype=out_dtype).contiguous(), torch.empty_like(
        x
    ).contiguous()
