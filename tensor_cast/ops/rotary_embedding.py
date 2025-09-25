from typing import Tuple

import torch

from ..utils import register_tensor_cast_op

@register_tensor_cast_op("apply_rope")
def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(query).contiguous(), torch.empty_like(key).contiguous()