from typing import List, Optional

import numpy as np

import torch

from .model_config import ParallelConfig
from .utils import exact_division


class ParallelGroup:
    """
    ParallelGroup handles all communication operations of its process group.
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        rank_groups: list[list[int]],
    ):
        """
        Initialize an instance of class ParallelGroup.

        Args:
            rank:
                The global rank.
            local_rank:
                The local rank of this process on this device.
            rank_groups:
                All the groups divided according to the current parallel strategy. We need to find the group
                that contains the given global rank.
                For instance, when tp_size is 2 and world_size is 8, the input rank_groups would be
                [[0, 1], [2, 3], [4, 5], [6, 7]] —— these represents all the tp_groups. If the given global rank is 5,
                the corresponding group we need is [4, 5], which means the attribute `ranks` will be set to [4, 5].
        """
        self.rank = rank
        self.local_rank = local_rank
        self.rank_group = None
        for ranks in rank_groups:
            if self.rank in ranks:
                self.rank_group = ranks
                self.rank_group.sort()
                self.rank_in_group = self.rank_group.index(self.rank)
                break
        self.world_size = len(self.rank_group)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        return torch.ops.tensor_cast.all_reduce(input_, self.rank, self.rank_group)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        return torch.ops.tensor_cast.reduce_scatter(
            input_, dim, self.rank, self.rank_group
        )

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        return torch.ops.tensor_cast.all_gather(input_, dim, self.rank, self.rank_group)

    def all_to_all(
        self,
        input_: List[torch.Tensor],
        output_split_sizes: List[int],
        input_split_sizes: List[int],
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        return torch.ops.tensor_cast.all_to_all(
            input_, output_split_sizes, input_split_sizes, self.rank, self.rank_group
        )

    def slice(self, input_: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        split_size = exact_division(input_.size()[dim], self.world_size)
        start_pos = self.rank_in_group * split_size
        return torch.narrow(input_, dim=dim, start=start_pos, length=split_size)


_DEFAULT_PG = ParallelGroup(0, 0, [[0]])


class ParallelGroupManager:
    tp_group: Optional[ParallelGroup] = None
    dp_group: Optional[ParallelGroup] = None

    mlp_tp_group: Optional[ParallelGroup] = None
    mlp_dp_group: Optional[ParallelGroup] = None

    lmhead_tp_group: Optional[ParallelGroup] = None
    lmhead_dp_group: Optional[ParallelGroup] = None

    def __init__(self, parallel_config: ParallelConfig):
        self.parallel_config = parallel_config
        self.initialize_model_parallel()

    def initialize_model_parallel(self):
        world_size = self.parallel_config.world_size
        rank = self.parallel_config.rank
        local_rank = self.parallel_config.local_rank

        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size
        data_parallel_size = self.parallel_config.data_parallel_size

        all_ranks = np.arange(world_size)

        def initialize_parallel(
            init_tensor_parallel, tensor_parallel_size, data_parallel_size
        ):
            if init_tensor_parallel:
                rank_groups = all_ranks.reshape(-1, tensor_parallel_size)
            else:
                rank_groups = (
                    all_ranks.reshape(
                        -1,
                        data_parallel_size,
                        pipeline_parallel_size,
                        tensor_parallel_size,
                    )
                    .swapaxes(1, 3)
                    .reshape(-1, data_parallel_size)
                )

            _ParallelGroup = ParallelGroup(
                rank=rank,
                local_rank=local_rank,
                rank_groups=[x.tolist() for x in rank_groups],
            )
            return _ParallelGroup

        self.tp_group = initialize_parallel(
            True, tensor_parallel_size, data_parallel_size
        )

        self.dp_group = initialize_parallel(
            False, tensor_parallel_size, data_parallel_size
        )

        self.mlp_tp_group = initialize_parallel(
            True,
            self.parallel_config.mlp_tensor_parallel_size,
            self.parallel_config.mlp_data_parallel_size,
        )
        self.mlp_dp_group = initialize_parallel(
            False,
            self.parallel_config.mlp_tensor_parallel_size,
            self.parallel_config.mlp_data_parallel_size,
        )

        self.lmhead_tp_group = initialize_parallel(
            True,
            self.parallel_config.lmhead_tensor_parallel_size,
            self.parallel_config.lmhead_data_parallel_size,
        )
        self.lmhead_dp_group = initialize_parallel(
            False,
            self.parallel_config.lmhead_tensor_parallel_size,
            self.parallel_config.lmhead_data_parallel_size,
        )

        if self.parallel_config.has_ep():
            self.ep_group = initialize_parallel(False, 1, world_size)
        else:
            self.ep_group = initialize_parallel(False, world_size, 1)
