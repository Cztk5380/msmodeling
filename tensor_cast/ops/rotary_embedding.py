from typing import Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("apply_rope")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed, k_embed = torch.empty_like(query), torch.empty_like(key)
    q_embed = q_embed.transpose(1, 2)
    k_embed = k_embed.transpose(1, 2)
    return q_embed.contiguous(), k_embed.contiguous()
