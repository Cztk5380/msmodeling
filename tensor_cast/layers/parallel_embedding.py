import torch
from torch import nn

from ..parallel_group import ParallelGroup
from .utils import get_partial_sharded, ModelWrapperBase


class ParallelEmbedding(ModelWrapperBase):
    """
    A parallel embedding layer that replaces a standard torch.nn.Embedding layer.
    """

    def __init__(self, embedding: torch.nn.Embedding, tp_group: ParallelGroup):
        super().__init__(embedding)
        self.embedding_dim = embedding
        self.tp_group = tp_group
        self.tp_size = tp_group.world_size
        self.tp_rank = tp_group.rank_in_group
        self.create_weights()

    def create_weights(self):
        if not self.tp_size > 1:
            return
        shard_weight = get_partial_sharded(
            self._inner.weight, self.tp_size, self.tp_rank, dim=1
        )
        self._inner.weight = nn.Parameter(shard_weight.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._inner(x)
        x = self.tp_group.all_gather(x, dim=-1)
        x = x[..., : self.embedding_dim]
        return x
