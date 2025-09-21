import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op(
    "_internal_repeat_marker_begin"
)
def _internal_repeat_marker_begin(
    x: torch.Tensor,
    id: int,
    repeats: int,
) -> torch.Tensor:
    return x


@register_tensor_cast_op(
    "_internal_repeat_marker_end"
)
def _internal_repeat_marker_end(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    return x
