from typing import Optional, Sequence, Tuple
import torch
from ..utils import register_tensor_cast_op
import math

@register_tensor_cast_op("convolution")
def _convolution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Optional[Sequence[int]] = None,
    padding: Optional[Sequence[int]] = None,
    dilation: Optional[Sequence[int]] = None,
    groups: Optional[int] = None,
) -> torch.Tensor:
    B = input.shape[0]
    C_out = weight.shape[0]
    input_shape = input.shape
    weight_shape = weight.shape

    dim = input.dim()-2
    if stride is None:
        stride = [1] * dim
    if padding is None:
        padding = [0] * dim
    if dilation is None:
        dilation = [1] * dim
    if groups is None:
        groups = 1

    # ensure tuple/list
    stride = list(stride) if isinstance(stride, (list, tuple)) else [stride] * dim
    padding = list(padding) if isinstance(padding, (list, tuple)) else [padding] * dim
    dilation = list(dilation) if isinstance(dilation, (list, tuple)) else [dilation] * dim

    if input.dim()==3:
        _, _, L_in = input_shape
        _, _, K_l = weight_shape

        s_l, = stride
        p_l, = padding
        d_l, = dilation

        L_out = math.floor((L_in + 2 * p_l - d_l * (K_l - 1) - 1) / s_l + 1)
        output_shape = (B, C_out, L_out)
    elif input.dim()==4:
        _, _, H_in, W_in = input_shape
        _, _, K_h, K_w = weight_shape
        s_h, s_w = stride
        p_h, p_w = padding
        d_h, d_w = dilation
        H_out = math.floor((H_in + 2 * p_h - d_h * (K_h - 1) - 1) / s_h + 1)
        W_out = math.floor((W_in + 2 * p_w - d_w * (K_w - 1) - 1) / s_w + 1)
        output_shape = (B, C_out, H_out, W_out)
    elif input.dim()==5:
        _, _, D_in, H_in, W_in = input_shape
        _, _, K_d, K_h, K_w = weight_shape
        s_d, s_h, s_w = stride
        p_d, p_h, p_w = padding
        d_d, d_h, d_w = dilation
        D_out = math.floor((D_in + 2 * p_d - d_d * (K_d - 1) - 1) / s_d + 1)
        H_out = math.floor((H_in + 2 * p_h - d_h * (K_h - 1) - 1) / s_h + 1)
        W_out = math.floor((W_in + 2 * p_w - d_w * (K_w - 1) - 1) / s_w + 1)
        output_shape = (B, C_out, D_out, H_out, W_out)
    else:
        raise ValueError(f"Unsupported convolution dimension: {input.dim()}")
    return torch.empty(output_shape, dtype=input.dtype, device="meta").contiguous()
