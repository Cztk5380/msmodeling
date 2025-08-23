from typing import List, Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("dispatch_tokens")
def _dispatch_tokens(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    num_tokens = topk_indices.numel()
    num_tokens_per_expert = num_tokens // num_experts
    num_tokens_last_expert = num_tokens - num_tokens_per_expert * (num_experts - 1)
    dispatched_x = [
        torch.empty(
            (num_tokens_per_expert, x.shape[-1]), dtype=x.dtype, device=x.device
        )
        for _ in range(num_experts - 1)
    ]
    dispatched_x.append(
        torch.empty(
            (num_tokens_last_expert, x.shape[-1]), dtype=x.dtype, device=x.device
        )
    )
    dispatched_indices = [
        torch.empty((num_tokens_per_expert,), dtype=torch.long, device=x.device)
        for _ in range(num_experts - 1)
    ]
    dispatched_indices.append(
        torch.empty((num_tokens_last_expert,), dtype=torch.long, device=x.device)
    )
    dispatched_weights = [
        torch.empty((num_tokens_per_expert,), dtype=x.dtype, device=x.device)
        for _ in range(num_experts - 1)
    ]
    dispatched_weights.append(
        torch.empty((num_tokens_last_expert,), dtype=torch.long, device=x.device)
    )
    return dispatched_x, dispatched_indices, dispatched_weights
