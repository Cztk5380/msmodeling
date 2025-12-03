from typing import Optional

import torch

from ..patch_torch import support_autocast_for_meta


class CachingRotaryEmb(torch.nn.Module):
    """
    Cache the position embeddings so that we can do quick index_select without
    computing them again and again in each forward.
    """

    def __init__(
        self,
        rotary_emb: torch.nn.Module,
        act_dtype: torch.dtype,
        max_position_embeddings: int,
        is_vl_model:bool = False
    ):
        super().__init__()
        self.act_dtype = act_dtype
        self.is_vl_model = is_vl_model
        x = torch.empty(
            max_position_embeddings, device="meta", dtype=act_dtype
        ).unsqueeze(0)
        position_ids = torch.arange(
            0, max_position_embeddings, device="meta", dtype=torch.long
        ).unsqueeze(0)
        with support_autocast_for_meta():
            position_embeddings = rotary_emb(x, position_ids)
        self.cos_sin_cache: Optional[torch.Tensor]
        if (
            isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) == 2
        ):
            if is_vl_model:
                position_embeddings = torch.cat(position_embeddings, dim=-1)
                self.register_buffer("cos_sin_cache", position_embeddings, persistent=False)
            else:
                position_embeddings = torch.cat(position_embeddings, dim=-1).squeeze()
                self.register_buffer("cos_sin_cache", position_embeddings, persistent=False)
        else:
            self.cos_sin_cache = None
            self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if self.is_vl_model and self.cos_sin_cache is not None:
            bs, seq_len = position_ids.shape[-2:]
            pos_emb = self.cos_sin_cache[:, :seq_len, :]
            pos_emb = pos_emb.expand(bs, seq_len, -1)
            return pos_emb.chunk(2, dim=-1)
        if self.cos_sin_cache is not None and x.dtype == self.act_dtype:
            return (
                self.cos_sin_cache.index_select(0, position_ids.flatten())
                .reshape(position_ids.size(0), -1, self.cos_sin_cache.size(-1))
                .chunk(2, dim=-1)
            )
        else:
            return self.rotary_emb(x, position_ids)
