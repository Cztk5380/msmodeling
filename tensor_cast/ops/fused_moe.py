from typing import List, Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("fused_moe")
def _fused_moe(
    x: torch.Tensor,
    experts_gate: List[torch.Tensor],
    experts_up: List[torch.Tensor],
    experts_down: List[torch.Tensor],
    shared_experts_gate: Optional[torch.Tensor],
    shared_experts_up: Optional[torch.Tensor],
    shared_experts_down: Optional[torch.Tensor],
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_act: str,
) -> torch.Tensor:
    return torch.empty_like(x)
