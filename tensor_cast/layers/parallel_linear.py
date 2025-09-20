import math

import torch
from torch import nn

from ..parallel_group import ParallelGroup
from .utils import ModelWrapperBase


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


class ParallelLinearBase(ModelWrapperBase):
    """
    A parallel linear layer that replaces a standard torch.nn.Linear layer.
    It handles different tensor parallel types.
    """

    def __init__(self, linear_layer: torch.nn.Linear):
        super().__init__(linear_layer)
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

    def create_weights(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RowParallelLinear(ParallelLinearBase):
    def __init__(
        self,
        linear_layer: torch.nn.Linear,
        tp_group: ParallelGroup,
        dp_group: ParallelGroup,
        global_dp_group: ParallelGroup,
        slice_input_by_last_dim: bool = False,
        reduce_output: bool = True,
    ):
        super().__init__(linear_layer)
        self.tp_group = tp_group
        self.tp_size = self.tp_group.world_size
        self.tp_rank = self.tp_group.rank_in_group
        self.in_features_per_partition = math.ceil(self.in_features / self.tp_size)
        self.out_features_per_partition = self.out_features
        self.create_weights()
        self.dp_group = dp_group
        self.global_dp_group = global_dp_group
        self.gather_slice_output = (
            self.global_dp_group.world_size != self.dp_group.world_size
        )
        self.slice_input_by_last_dim = slice_input_by_last_dim
        self.reduce_output = reduce_output

    def create_weights(self):
        shard_weight = get_partial_sharded(
            self._inner.weight, self.tp_size, self.tp_rank, dim=1
        )
        self._inner.weight = nn.Parameter(shard_weight.contiguous())
        if self._inner.bias is not None:  # noqa: SIM102
            # need to check
            if self.tp_rank != 0:
                self._inner.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.slice_input_by_last_dim:
            x = get_partial_sharded(x, self.tp_size, self.tp_rank, dim=-1).contiguous()

        output = self._inner(x)
        if self.reduce_output:
            output = self.tp_group.all_reduce(output)

        if self.gather_slice_output:
            output = self.dp_group.all_gather(output, dim=0)
            output = self.global_dp_group.slice(output, dim=0).contiguous()
        return output


class ColumnParallelLinear(ParallelLinearBase):
    def __init__(
        self,
        linear_layer: torch.nn.Linear,
        tp_group: ParallelGroup,
        dp_group: ParallelGroup,
        global_dp_group: ParallelGroup,
        gather_output: bool = False,
    ):
        super().__init__(linear_layer)
        self.tp_group = tp_group
        self.tp_size = self.tp_group.world_size
        self.tp_rank = self.tp_group.rank_in_group
        self.in_features_per_partition = self.in_features
        self.out_features_per_partition = math.ceil(self.out_features / self.tp_size)
        self.create_weights()
        self.dp_group = dp_group
        self.global_dp_group = global_dp_group
        self.gather_slice_input = (
            self.global_dp_group.world_size != self.dp_group.world_size
        )
        self.gather_output = gather_output

    def create_weights(self):
        shard_weight = get_partial_sharded(
            self._inner.weight, self.tp_size, self.tp_rank, dim=0
        )
        self._inner.weight = nn.Parameter(shard_weight.contiguous())
        if self._inner.bias is not None:
            shard_bias = get_partial_sharded(
                self._inner.bias, self.tp_size, self.tp_rank
            )
            self._inner.bias = nn.Parameter(shard_bias.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gather_slice_input:
            x = self.global_dp_group.all_gather(x, dim=0)
            x = self.dp_group.slice(x, dim=0).contiguous()

        output = self._inner(x)
        if self.gather_output:
            output = self.tp_group.all_gather(output)
            output = output[..., : self.out_features].contiguous()

        return output


COLWISE_LINEAR = "colwise"
ROWWISE_LINEAR = "rowwise"

PARALLEL_MODULE_CLS = {
    COLWISE_LINEAR: ColumnParallelLinear,
    ROWWISE_LINEAR: RowParallelLinear,
}
