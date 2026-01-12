# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
"""
input_generation
"""

from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from ..layers.attention import AttentionMetadataTensorCast
from ..layers.sampler import SamplingMetadata
from ..performance_model import bytes_of_tensor
from ..transformers.utils import get_attention_quant_config, logger
from ..utils import exact_division


@dataclass
class RequestInfo:
    query_len: int
    seq_len: int
    is_decode: bool = True
    num_input_tokens: int = None
    num_output_tokens: int = None
    concurrency: int = 1
    image_batch_size: int = None
    image_height: int = None
    image_width: int = None


def generate_inputs(model, requests: List[RequestInfo], block_size: int = 128):
    # TODO merge generate_inputs and generate_inputs_varlen
    # for now, unify the function signatures, Firstly.
    request = requests[0]
    concurrency = request.concurrency
    seq_len = request.seq_len
    query_len = request.query_len
    is_decode = request.is_decode
    image_kwargs = {}
    if model.is_vl_model:
        image_kwargs = generate_image_inputs(
            model,
            request.image_batch_size,
            request.image_height,
            request.image_width,
            concurrency,
        )
        num_image_tokens = image_kwargs.pop("num_image_tokens", 0)
        if is_decode:
            # In the decode phase, the image input is removed, but the image token needs to be added to content_length
            image_kwargs = {}
            seq_len += num_image_tokens
        else:
            query_len += num_image_tokens
    else:
        if (
            request.image_batch_size is not None
            or request.image_height is not None
            or request.image_width is not None
        ):
            logger.warning(
                "For non-VL models, the parameter input of the image is ignored"
            )
    model_config = model.model_config
    num_mtp_tokens = (
        model_config.mtp_config.num_mtp_layers if model_config.mtp_config else 0
    )
    parallel_config = model_config.parallel_config
    batch_size = (
        concurrency + parallel_config.data_parallel_size - 1
    ) // parallel_config.data_parallel_size

    max_context_length = seq_len + num_mtp_tokens + 1

    # Paged attention parameters (can be adjusted)
    num_blocks = (
        max_context_length * batch_size + block_size - 1
    ) // block_size  # Total number of blocks available in the KV cache

    # Prepare Attention Metadata for Paged Attention
    # `query_start_loc` indicates the start of each query in the concatenated input tensor.
    # Shape: [num_queries + 1] -> e.g., [0, 50, 100, 150] for 3 queries of length 50.
    query_start_loc = torch.arange(
        0, (batch_size + 1) * query_len, query_len, dtype=torch.long
    )

    # `seq_lens` is the total length (context + new tokens) for each sequence in the batch.
    seq_lens = torch.empty(batch_size, dtype=torch.long)
    seq_lens.fill_(seq_len)

    query_lens = torch.empty(batch_size, dtype=torch.long)
    query_lens.fill_(query_len)

    # `block_tables` map logical sequence blocks to physical blocks in the KV cache.
    max_num_blocks_per_seq = (seq_len + block_size - 1) // block_size

    block_table_tensor = torch.empty(
        (batch_size, max_num_blocks_per_seq), dtype=torch.long, device="meta"
    )

    slot_mapping = torch.empty(
        (batch_size * query_len,), dtype=torch.long, device="meta"
    )

    # We use padding to ensure that the number of tokens in each DP domain is divisible by tp_size.
    # This allows the data to be evenly distributed across each device if needed,
    # thereby enabling arbitrary conversion of DP domains.
    padding_tokens = 0
    if batch_size * query_len % parallel_config.tensor_parallel_size != 0:
        padding_tokens = parallel_config.tensor_parallel_size - (
            batch_size * query_len % parallel_config.tensor_parallel_size
        )

    query_start_loc[-1] = query_start_loc[-1] + padding_tokens

    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_lens=query_lens,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )

    # The total number of new tokens to be processed in this batch, concatenated.
    num_tokens = batch_size * query_len + padding_tokens
    input_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    kv_cache_by_layers, kv_cache_per_token = _get_kv_cache_info(
        model, num_blocks, block_size
    )
    sampling_metadata = SamplingMetadata(
        query_start_loc=attn_meta.query_start_loc,
    )
    if is_decode:
        # do not prune logits
        sampling_metadata.selected_token_indices = None

    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_meta": attn_meta,
        "kv_cache_by_layers": kv_cache_by_layers,
        "kv_cache_per_token": kv_cache_per_token,
        "sampling_metadata": sampling_metadata,
    }
    if model.model_config.hf_config.model_type == "qwen3_next":
        kwargs["cache_position"] = torch.arange(
            0, num_tokens, dtype=torch.long, device="cpu"
        )
    kwargs.update(image_kwargs)
    return kwargs


def generate_image_inputs(
    model, image_batch_size, image_height, image_width, concurrency
):
    if image_batch_size is None or image_height is None or image_width is None:
        print("For vision-language models,without image input")
        return {}
    vision_config = model.model_config.hf_config.vision_config
    patch_size = vision_config.patch_size
    merge_size = (
        vision_config.spatial_merge_size if vision_config.spatial_merge_size else 2
    )
    # Rescales the image
    resized_height, resized_width = smart_resize(
        image_height, image_width, factor=patch_size * merge_size
    )
    # For images, the value of grid_t is 1.
    grid_t = 1
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long).expand(
        image_batch_size, 3
    )
    channel = vision_config.in_channels if vision_config.in_channels else 3
    temporal_patch_size = (
        vision_config.temporal_patch_size if vision_config.temporal_patch_size else 2
    )
    hidden_dim = channel * temporal_patch_size * patch_size * patch_size
    tokens = grid_t * grid_h * grid_w
    pixel_values = torch.empty(
        image_batch_size * tokens,
        hidden_dim,
        dtype=model.model_config.dtype,
        device="meta",
    )
    # Calculate the token embedded in the text.
    merge_length = merge_size**2
    num_image_tokens = image_batch_size * (tokens // merge_length + 2)
    parallel_config = model.model_config.parallel_config
    batch_size = (
        concurrency + parallel_config.data_parallel_size - 1
    ) // parallel_config.data_parallel_size
    pixel_values = pixel_values.repeat(batch_size, 1)
    image_grid_thw = image_grid_thw.repeat(batch_size, 1)
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "num_image_tokens": num_image_tokens,
    }


def _get_kv_cache_info(
    model, num_blocks: int, block_size: int
) -> Tuple[dict[Any, Any], int]:
    model_config = model.model_config
    parallel_config = model.model_config.parallel_config
    # Initialize the KV cache structure (also on 'meta' device).
    kv_cache_per_token = 0
    kv_cache_by_layers = {}
    for i in range(model.num_hidden_layers):
        kvcache_dtype = model_config.dtype
        if (attention_config := get_attention_quant_config(model, i)) is not None:
            kvcache_dtype = attention_config.get_quant_dtype()

        if model_config.mla_config is not None:
            # Shape: [num_blocks, block_size, kv_lora_head_dim + qk_rope_head_dim]
            kv_cache_by_layers[i] = torch.empty(
                [
                    num_blocks,
                    block_size,
                    model.text_config.kv_lora_rank + model.text_config.qk_rope_head_dim,
                ],
                dtype=kvcache_dtype,
                device="meta",
            )
        else:
            # Shape: [2 (K/V), num_blocks, block_size, num_heads, head_dim]
            if (
                model.text_config.num_key_value_heads
                >= parallel_config.tensor_parallel_size
            ):
                kv_heads = exact_division(
                    model.text_config.num_key_value_heads,
                    parallel_config.tensor_parallel_size,
                )
            else:
                assert (
                    parallel_config.tensor_parallel_size
                    % model.text_config.num_key_value_heads
                    == 0
                )
                kv_heads = 1

            kv_cache_by_layers[i] = torch.empty(
                [
                    2,
                    num_blocks,
                    block_size,
                    kv_heads,
                    model.head_dim,
                ],
                dtype=kvcache_dtype,
                device="meta",
            )
        kv_cache_per_token += bytes_of_tensor(kv_cache_by_layers[i]) / (
            num_blocks * block_size
        )
    return kv_cache_by_layers, kv_cache_per_token


def get_kv_cache_info(model, num_blocks, block_size):
    model_config = model.model_config
    tp_size = model_config.parallel_config.tensor_parallel_size
    kv_cache_by_layers = {}
    kv_cache_per_token = 0
    for i in range(model.num_hidden_layers):
        kvcache_dtype = model_config.dtype
        attention_config = get_attention_quant_config(model, i)
        if attention_config is not None:
            kvcache_dtype = attention_config.get_quant_dtype()

        if model_config.mla_config is not None:
            kv_cache_by_layers[i] = torch.empty(
                (
                    num_blocks,
                    block_size,
                    model.text_config.kv_lora_rank + model.text_config.qk_rope_head_dim,
                ),
                dtype=kvcache_dtype,
                device="meta",
            )
        else:
            assert model.text_config.num_key_value_heads % tp_size == 0
            kv_cache_by_layers[i] = torch.empty(
                (
                    2,
                    num_blocks,
                    block_size,
                    model.text_config.num_key_value_heads // tp_size,
                    model.head_dim,
                ),
                dtype=kvcache_dtype,
                device="meta",
            )
        kv_cache_per_token += bytes_of_tensor(kv_cache_by_layers[i]) / (
            num_blocks * block_size
        )

    return kv_cache_by_layers, kv_cache_per_token


def generate_inputs_varlen(model, requests: List[RequestInfo], block_size):
    """
    requests: List[RequestInfo], each dict represents a request, containing keys: query_len, seq_len, is_decode
    """
    model_config = model.model_config
    parallel_config = model_config.parallel_config
    mtp = getattr(model_config, "mtp_config", None)
    num_mtp_tokens = mtp.num_mtp_layers if mtp else 0

    batch_size = len(requests)
    if batch_size == 0:
        return {}

    query_lens = [r.query_len for r in requests]
    seq_lens = [r.seq_len for r in requests]
    is_decode_list = [r.is_decode for r in requests]
    num_tokens = sum(query_lens)

    # padding query to make sure total num_tokens is divisible by tp_size in each dp domain
    padding_nums = parallel_config.tensor_parallel_size - (
        num_tokens % parallel_config.tensor_parallel_size
    )
    num_tokens += padding_nums

    query_start_loc = [0]
    for ql in query_lens:
        query_start_loc.append(query_start_loc[-1] + ql)
    query_start_loc = torch.tensor(query_start_loc, dtype=torch.long)
    query_start_loc[-1] += padding_nums

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long)
    query_len_t = torch.tensor(query_lens, dtype=torch.long)

    num_blocks = (
        sum(seq_lens) + batch_size * (num_mtp_tokens + 1) + block_size - 1
    ) // block_size
    max_num_blocks_per_seq = (max(seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.empty(
        (batch_size, max_num_blocks_per_seq), dtype=torch.long, device="meta"
    )
    slot_mapping = torch.empty((num_tokens,), dtype=torch.long, device="meta")

    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        query_lens=query_len_t,
        seq_lens=seq_lens_t,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )

    input_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

    kv_cache_by_layers, kv_cache_per_token = get_kv_cache_info(
        model, num_blocks, block_size
    )

    sampling_meta = SamplingMetadata(query_start_loc=query_start_loc)
    selected_token_indices = []

    pos = 0
    for ql, decode in zip(query_lens, is_decode_list):
        if decode:
            selected_token_indices.extend(range(pos, pos + ql))
        else:
            selected_token_indices.append(pos + ql - 1)
        pos += ql
    sampling_meta.selected_token_indices = torch.tensor(
        selected_token_indices, dtype=torch.long, device="meta"
    )

    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_meta": attn_meta,
        "kv_cache_by_layers": kv_cache_by_layers,
        "sampling_metadata": sampling_meta,
        "kv_cache_per_token": kv_cache_per_token,
    }

    if model.model_config.hf_config.model_type == "qwen3_next":
        kwargs["cache_position"] = torch.arange(
            num_tokens, dtype=torch.long, device="cpu"
        )

    return kwargs


def get_inputs_num_bytes(model, requests: List[RequestInfo], block_size: int) -> int:
    """
    Get the number of bytes of the input tensors.
    """
    input_kwargs = generate_inputs_varlen(model, requests, block_size)
    inputs_num_bytes = 0
    inputs_num_bytes += bytes_of_tensor(input_kwargs["input_ids"])
    inputs_num_bytes += bytes_of_tensor(input_kwargs["position_ids"])
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].query_start_loc)
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].seq_lens)
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].query_lens)
    inputs_num_bytes += bytes_of_tensor(
        input_kwargs["attention_meta"].block_table_tensor
    )
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].slot_mapping)
    return inputs_num_bytes
