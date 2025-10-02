import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("_internal_mark_region_begin")
def _internal_mark_region_begin(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Mark the beginning of a region of execution."""
    return x


@register_tensor_cast_op("_internal_mark_region_end")
def _internal_mark_region_end(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Mark the end of a region of execution."""
    return x


@register_tensor_cast_op("_internal_copy_region")
def _internal_copy_region(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Copy a region of execution marked previously."""
    return x
