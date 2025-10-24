from typing import Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("concat_and_cache_mla", mutates_args=("kv_cache",))
def _concat_and_cache_mla(
    kv_c_normed: torch.Tensor,
    k_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    concat `kv_c_normed` and `k_rot` with into `kv_cache` according to `slot_mapping`.

    Args:
        kv_c_normed: (num_tokens, kv_lora_rank)
        k_rot: (num_tokens, qk_rope_head_dim)
        kv_cache: (total_num_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
        slot_mapping: see `AttentionMetadataBase`
    """


@register_tensor_cast_op("multihead_latent_attention")
def _multihead_latent_attention(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: Optional[torch.Tensor],
    W_UK_T: Optional[torch.Tensor],
    W_UV: Optional[torch.Tensor],
    kv_b_proj: Optional[torch.Tensor],
    v_head_dim: int,
) -> torch.Tensor:
    """
    This op computes multi-head latent attention (MLA). It is supposed to use different
    algorithms for prefill and decode shapes while the input sequences could fuse prefill
    and decode sequences and should be handled separately with different algorithms.

    We judge the prefill or decode phase according to the query length per `query_start_loc`.
    If the query length is

    For prefill (non-strict math/code):
        k_nope, v = (kv_c_normed @ kv_b_proj).view(-1, num_heads, qk_nope_head_dim + v_head_dim).split(dim=-1)
        softmax(q @ (k_nope, k_rot)) @ v

    For decode (non-strict math/code):
        softmax(q @ W_UK_T @ k_cache) @ v_cache @ W_UV

    Args:
        q: (num_tokens, num_heads, qk_nope_head_dim+qk_rope_head_dim)
            The query states after compression and decompression.
        kv_c_normed: (num_tokens, kv_lora_rank)
            The normalized and compressed key-value states.
        k_rot: (num_tokens, qk_rope_head_dim)
            The slice of key after applying rotation embedding.
        kv_cache: (total_num_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
            The cached key-value states with current KV states already updated.
        block_table/query_start_loc/seq_lens: see `AttentionMetadataBase`
        W_UK_T, W_UV: (num_heads, qk_nope_head_dim, kv_lora_rank), (num_heads, kv_lora_rank, v_head_dim)
            used in the decode phase, None if only prefill sequences are provided.
        kv_b_proj: (kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
            used in the prefill phase, None if only decode sequences are provided.

    Returns:
        (num_tokens, num_heads, v_head_dim)
    """
    return torch.empty(q.shape[0], q.shape[1], v_head_dim, dtype=q.dtype, device="meta")


@register_tensor_cast_op("multihead_latent_attention_quant")
def _multihead_latent_attention_quant(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: Optional[torch.Tensor],
    W_UK_T: Optional[torch.Tensor],
    W_UV: Optional[torch.Tensor],
    kv_b_proj: Optional[torch.Tensor],
    v_head_dim: int,
    query_scale: torch.Tensor,
    query_offset: Optional[torch.Tensor],
    kv_scale: torch.Tensor,
    kv_offset: Optional[torch.Tensor],
    kv_projected_scale: torch.Tensor,
    kv_projected_offset: Optional[torch.Tensor],
    qk_scale: torch.Tensor,
    qk_offset: Optional[torch.Tensor],
    v_scale: torch.Tensor,
    v_offset: Optional[torch.Tensor],
    attention_prob_scale: torch.Tensor,
    attention_prob_offset: Optional[torch.Tensor],
    kv_b_proj_scale: torch.Tensor,
    kv_b_proj_offset: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    out_offset: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """
    Similar to `multihead_latent_attention` but with quantization support.
    For prefill (non-strict math/code):
        quant_kv_proj = quant(kv_c_normed @ kv_b_proj, kv_projected_scale, kv_projected_offset)
        k_nope, v = quant_kv_proj.view(-1, num_heads, qk_nope_head_dim + v_head_dim).split(dim=-1)
        out_fp = quant(softmax(q @ (k_nope, k_rot)), attention_prob_scale, attention_prob_offset) @ v
        out = quant(out_fp, out_scale, out_offset) # optional

    For decode (non-strict math/code):
        quant_qk = quant(q @ W_UK_T, qk_scale, qk_offset)
        quant_scores = quant(softmax(quant_qk @ k_cache), attention_prob_scale, attention_prob_offset)
        out_fp = quant(quant_scores @ v_cache, v_scale, v_offset) @ W_UV
        out = quant(out_fp, out_scale, out_offset) # optional

    Returns:
        (num_tokens, num_heads, v_head_dim)
    """
    if out_dtype is None:
        out_dtype = q.dtype
    return torch.empty(
        q.shape[0], q.shape[1], v_head_dim, dtype=out_dtype, device="meta"
    )
