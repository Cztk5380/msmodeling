import threading
from typing import Optional

import diffusers
import torch
from aenum import extend_enum

from diffusers.models.attention_dispatch import _AttentionBackendRegistry

from ..parallel_group import ParallelGroup


extend_enum(
    diffusers.models.attention_dispatch.AttentionBackendName,
    "TENSOR_CAST",
    "tensor_cast",
)

_thread_local = threading.local()


def set_sp_group(sp_group: Optional[ParallelGroup]):
    _thread_local.sp_group = sp_group


def get_sp_group() -> Optional[ParallelGroup]:
    return getattr(_thread_local, "sp_group", None)


@_AttentionBackendRegistry.register("tensor_cast")
def _attention(query, key, value, **kwargs):
    sp_group = get_sp_group()
    if sp_group is None:
        return torch.ops.tensor_cast.attention(
            query, key, value, None, None, None, None, None
        )

    ulysses_size = sp_group.world_size

    # all-to-all: (b, s, h, w) -> (b, s * p, h, w / p)
    # In cross attention, query shape is not equal to key, value shape
    batch_size, seq_per_rank, num_heads, head_dim = query.shape
    batch_size_kv, seq_per_rank_kv, num_heads_kv, head_dim_kv = key.shape
    input_tensor_q = torch.ones(
        (batch_size, seq_per_rank, num_heads, head_dim // ulysses_size),
        dtype=query.dtype,
        device=query.device,
    )
    input_tensor_kv = torch.ones(
        (batch_size_kv, seq_per_rank_kv, num_heads_kv, head_dim_kv // ulysses_size),
        dtype=query.dtype,
        device=query.device,
    )
    input_split_sizes = [1 for _ in range(ulysses_size - 1)]
    output_split_sizes = [1 for _ in range(ulysses_size - 1)]

    _ = sp_group.all_to_all(
        input_tensor_q,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )
    _ = sp_group.all_to_all(
        input_tensor_kv,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )
    _ = sp_group.all_to_all(
        input_tensor_kv,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )
    query = query.view(
        batch_size, seq_per_rank * ulysses_size, num_heads, head_dim // ulysses_size
    )
    key = key.view(
        batch_size_kv,
        seq_per_rank_kv * ulysses_size,
        num_heads_kv,
        head_dim_kv // ulysses_size,
    )
    value = value.view(
        batch_size_kv,
        seq_per_rank_kv * ulysses_size,
        num_heads_kv,
        head_dim_kv // ulysses_size,
    )
    out = torch.ops.tensor_cast.attention(
        query, key, value, None, None, None, None, None
    )

    _ = sp_group.all_to_all(
        input_tensor_q,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )
    out = out.view(batch_size, seq_per_rank, num_heads, head_dim)
    return out
