import math
from typing import Optional

import torch


def get_sharded_shape(shape: torch.Tensor, dim: int, block_size: int):
    sharded_shape = list(shape)
    sharded_shape[dim] = block_size
    return sharded_shape


def get_partial_sharded(tensor: torch.Tensor, world_size: int, rank: int, dim: int = 0):
    assert dim in [0, -1, tensor.dim() - 1]

    size = tensor.shape[dim]
    block_size = math.ceil(size / world_size)

    start = rank * block_size
    stop = (rank + 1) * block_size

    if dim == 0:
        tensor = tensor[start:stop]
    else:
        tensor = tensor[..., start:stop]

    sharded_shape = get_sharded_shape(tensor.shape, dim, block_size)
    tensor_zeros = torch.zeros(
        size=sharded_shape, dtype=tensor.dtype, device=tensor.device
    )
    if dim == 0:
        tensor_zeros[: tensor.shape[0]] = tensor
    else:
        tensor_zeros[..., : tensor.shape[-1]] = tensor

    return tensor_zeros


class ModelWrapperBase(torch.nn.Module):
    def __init__(self, wrapped: Optional[torch.nn.Module]):
        super().__init__()
        self._inner = wrapped

    def unwrap(self) -> torch.nn.Module:
        wrapped = self
        while isinstance(wrapped, ModelWrapperBase):
            wrapped = wrapped._inner
        return wrapped

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            if hasattr(self._inner, item):
                return getattr(self._inner, item)
            raise
