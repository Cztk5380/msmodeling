from typing import Optional
import torch
import dataclasses


# adapted from vLLM but trimmed to avoid redundancy
@dataclasses.dataclass
class AttentionMetadataBase:
    """Per-layer attention metadata"""
    query_start_loc: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""

    block_table_tensor: Optional[torch.Tensor] = None
    """(batch_size, max_blocks_per_seq)"""
    slot_mapping: Optional[torch.Tensor] = None
    """(num_tokens,) The indices of the token slots that input tokens will be
    stored into."""

    causal: bool = True


class AttentionBase(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward().")


# adapted from vLLM
def flash_attention_forward(
        # Transformers args
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs):
    attention_by_layers: Optional[dict[int, AttentionBase]] = kwargs.pop("attention_by_layers", None)
    assert attention_by_layers is not None, "Expect attention_by_layers to be provided."

    kv_cache_by_layers: Optional[dict[int, torch.Tensor]] = kwargs.pop("kv_cache_by_layers", None)
    attention_meta: AttentionMetadataBase = kwargs.pop("attention_meta", None)
    attention_meta_by_layers: Optional[dict[int, AttentionMetadataBase]] = kwargs.pop("attention_meta_by_layers", None)
    assert attention_meta is None or attention_meta_by_layers is None, "Only one of attention_meta and attention_meta_by_layers can be provided."

    self_attn = attention_by_layers[module.layer_idx]
    kv_cache = kv_cache_by_layers[module.layer_idx] if kv_cache_by_layers else None
    attention_meta = attention_meta_by_layers[module.layer_idx] if attention_meta_by_layers else attention_meta
    # TODO: understand why we need these shape manipulation
    hidden = query.shape[-2]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    # return (attn_output, attn_weights) while we ignore attn_weights
    return self_attn.forward(query, key, value, attention_mask, kv_cache=kv_cache, attention_meta=attention_meta, **kwargs), None


class AttentionMetadataTensorCast(AttentionMetadataBase):
    pass


class AttentionTensorCast(AttentionBase):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> torch.Tensor:
        attention_meta = kwargs.pop("attention_meta", None)
        query_start_loc = attention_meta.query_start_loc if attention_meta else None
        seq_lens = attention_meta.seq_lens if attention_meta else None
        return torch.ops.tensor_cast.attention_forward(
            query,
            key,
            value,
            query_start_loc,
            seq_lens,
        )


@torch.library.custom_op("tensor_cast::attention_forward", mutates_args=())
def _attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    # TODO: estimate the compute and memory cost
    if query_start_loc and seq_lens:
        start_points = query_start_loc[:-1]
        end_points = query_start_loc[1:]
        query_lens = end_points - start_points
        assert query_lens >= 0, "Query lengths must be non-negative."
        assert query_lens <= seq_lens, "Query lengths must be less than or equal to sequence lengths."
        
    return torch.empty_like(query).contiguous()

def _attention_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


_attention_forward.register_fake(_attention_forward_fake)