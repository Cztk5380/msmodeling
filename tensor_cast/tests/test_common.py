import random

import torch

from ..layers.attention import AttentionMetadataTensorCast


def create_attn_metadata_and_kv_cache(model, model_config):
    batch_size = 2
    query_len_1 = 55
    query_len_2 = 45
    seq_len_1 = 2000
    seq_len_2 = 1500
    num_blocks = 10000
    block_size = 128
    max_seq_len = max(seq_len_1, seq_len_2)
    query_start_loc = torch.tensor(
        [0, query_len_1, query_len_1 + query_len_2], dtype=torch.long
    )
    seq_lens = torch.tensor([seq_len_1, seq_len_2], dtype=torch.long)
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(batch_size):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_table_tensor = torch.tensor(block_tables, dtype=torch.long)
    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table_tensor=block_table_tensor,
    )

    num_tokens = query_len_1 + query_len_2
    kv_cache_by_layers = {}
    for i in range(model.num_hidden_layers):
        kv_cache_by_layers[i] = torch.empty(
            [
                2,
                num_blocks,
                block_size,
                model.text_config.num_key_value_heads,
                model.text_config.head_dim,
            ],
            dtype=model_config.dtype,
            device="meta",
        )
    return attn_meta, kv_cache_by_layers, num_tokens


def create_mla_metadata_and_kv_cache(model, model_config):
    batch_size = 2
    query_len_1 = 55
    query_len_2 = 45
    seq_len_1 = 2000
    seq_len_2 = 1500
    num_blocks = 10000
    block_size = 128
    max_seq_len = max(seq_len_1, seq_len_2)
    query_start_loc = torch.tensor(
        [0, query_len_1, query_len_1 + query_len_2], dtype=torch.long
    )
    seq_lens = torch.tensor([seq_len_1, seq_len_2], dtype=torch.long)
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(batch_size):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_table_tensor = torch.tensor(block_tables, dtype=torch.long)
    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table_tensor=block_table_tensor,
    )

    num_tokens = query_len_1 + query_len_2
    kv_cache_by_layers = {}
    for i in range(model.num_hidden_layers):
        kv_cache_by_layers[i] = torch.empty(
            [
                num_blocks,
                block_size,
                model.text_config.kv_lora_rank + model.text_config.qk_rope_head_dim,
            ],
            dtype=model_config.dtype,
            device="meta",
        )
    return attn_meta, kv_cache_by_layers, num_tokens


def has_submodule_with_cls_name(module, cls_name):
    return any(
        type(sub_module).__name__ == cls_name
        for _, sub_module in module.named_modules()
    )
