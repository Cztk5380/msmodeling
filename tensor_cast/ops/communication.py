from typing import List

import torch

from ..utils import exact_division, register_tensor_cast_op


@register_tensor_cast_op("all_to_all")
def _all_to_all(
    x: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    rank: int,
    rank_group: List[int],
) -> torch.Tensor:
    output_num = sum(output_split_sizes)
    return torch.empty((output_num,) + x.shape[1:], dtype=x.dtype, device=x.device)


@register_tensor_cast_op("all_reduce")
def _all_reduce(x: torch.Tensor, rank: int, rank_group: List[int]) -> torch.Tensor:
    return torch.empty_like(x)


@register_tensor_cast_op("reduce_scatter")
def _reduce_scatter(
    x: torch.Tensor, dim: int, rank: int, rank_group: List[int]
) -> torch.Tensor:
    world_size = len(rank_group)
    new_shape = list(x.shape)
    new_shape[dim] = exact_division(new_shape[dim], world_size)
    return torch.empty(new_shape, dtype=x.dtype, device=x.device)


@register_tensor_cast_op("all_gather")
def _all_gather(
    x: torch.Tensor, dim: int, rank: int, rank_group: List[int]
) -> torch.Tensor:
    world_size = len(rank_group)
    new_shape = list(x.shape)
    new_shape[dim] = new_shape[dim] * world_size
    return torch.empty(new_shape, dtype=x.dtype, device=x.device)
