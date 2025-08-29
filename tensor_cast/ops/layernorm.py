import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("rms_norm")
def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x).contiguous()


@register_tensor_cast_op("add_rms_norm")
def _add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x).contiguous()


@register_tensor_cast_op("add_rms_norm_quant")
def _add_rms_norm_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x).contiguous()
