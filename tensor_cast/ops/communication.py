import torch

from ..utils import exact_division, register_tensor_cast_op


@register_tensor_cast_op("all_reduce")
def _all_reduce(x: torch.Tensor, group_name: str) -> torch.Tensor:
    return torch.empty_like(x)


@register_tensor_cast_op("reduce_scatter")
def _reduce_scatter(
    x: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    # TODO: get world_size from group_name
    new_shape = list(x.shape)
    new_shape[dim] = exact_division(new_shape[dim], world_size)
    return torch.empty(new_shape, dtype=x.dtype, device=x.device)


@register_tensor_cast_op("all_gather")
def _all_gather(
    x: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    # TODO: get world_size from group_name
    new_shape = list(x.shape)
    new_shape[dim] = new_shape[dim] * world_size
    return torch.empty(new_shape, dtype=x.dtype, device=x.device)
