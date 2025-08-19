import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("reshape_and_cache", mutates_args=("kv_cache",))
def _reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    pass


@register_tensor_cast_op("attention")
def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


@register_tensor_cast_op("attention_quant")
def _attention_quant(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_scale: torch.Tensor,
    query_offset: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_offset: torch.Tensor,
    attention_prob_scale: torch.Tensor,
    attention_prob_offset: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()
