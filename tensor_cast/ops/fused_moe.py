import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("permute_tokens")
def _permute_tokens(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Repeat the input tokens top-k times, and rearrange them according to the order of the experts
    selected by the tokens.

    Args:
        x: (bsz, seq_len, hidden_size), the tokens
        topk_indices: (bsz, seq_len, top_k), the top-k experts selected by each token

    Returns:
        permuted_x: (bsz * seq_len * top_k, hidden_size)
    """
    num_tokens = topk_indices.numel()
    return torch.empty((num_tokens, x.shape[-1]), dtype=x.dtype, device=x.device)


@register_tensor_cast_op("unpermute_tokens")
def _unpermute_tokens(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Rearrange the input tokens (initially sorted by their selected experts) by token indices.

    Args:
        x: (bsz * seq_len * top_k, hidden_size), the tokens
        topk_indices: (bsz, seq_len, top_k), the top-k experts selected by each token

    Returns:
        unpermuted_x: (bsz, seq_len, top_k, hidden_size)
    """
    return torch.empty_like(x).reshape(topk_indices.shape + (x.shape[-1],))
