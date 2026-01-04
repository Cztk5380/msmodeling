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
    ):
        super().__init__()
        self.act_dtype = act_dtype
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
            position_embeddings = torch.cat(position_embeddings, dim=-1).squeeze()
            self.register_buffer("cos_sin_cache", position_embeddings, persistent=False)
        else:
            self.cos_sin_cache = None
            self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if self.cos_sin_cache is not None and x.dtype == self.act_dtype:
            if position_ids.ndim == 3:
                # Determine whether the input is text-only or multimodal based on tensor dimensions.
                # If it is multimodal, use the text shape (B, S).
                # position_ids is (3, batch, seq_len), where 3--> (T/H/W)
                position_ids = position_ids[0]
            return (
                self.cos_sin_cache.index_select(0, position_ids.flatten())
                .reshape(position_ids.size(0), -1, self.cos_sin_cache.size(-1))
                .chunk(2, dim=-1)
            )
        else:
            return self.rotary_emb(x, position_ids)
