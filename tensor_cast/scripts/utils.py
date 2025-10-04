from enum import StrEnum

import torch

from .. import device_profiles  # noqa: F401

from ..compilation import get_backend
from ..layers.attention import AttentionMetadataTensorCast, AttentionTensorCast
from ..layers.mla import MultiheadLatentAttentionTensorCast
from ..layers.quant_linear import TensorCastQuantLinear
from ..layers.sampler import SamplingMetadata
from ..model_config import (
    LinearQuantConfig,
    LinearQuantType,
    MlaConfig,
    ModelConfig,
    MtpConfig,
    ParallelConfig,
    QuantConfig,
)
from ..transformers.model import TransformerModel
from ..transformers.utils import (
    model_id_to_json,
    model_id_to_mla_module_name,
    model_id_to_mtp_block_module_name,
)

from ..utils import DTYPE_FP8


class QuantLinearAction(StrEnum):
    DISABLED = "DISABLED"
    W8A16_STATIC = "W8A16_STATIC"
    W8A8_STATIC = "W8A8_STATIC"
    W4A8_STATIC = "W4A8_STATIC"
    W8A16_DYNAMIC = "W8A16_DYNAMIC"
    W8A8_DYNAMIC = "W8A8_DYNAMIC"
    W4A8_DYNAMIC = "W4A8_DYNAMIC"
    FP8 = "FP8"


def get_available_memory_gb(device_profile, runtime, reserved_memory_size_gb=0):
    """
    Get available memory on the device during executing models under the runtime. It is the minimum
    available memory, not the one after model execution.

    :param device_profile: The device configuration
    :param runtime: The runtime under which the models have been executed
    :param reserved_memory_size_gb: The reserved memory size on top of the consumption of the models.
    :return: The minimum available memory during execution.
    """
    total_device_memory_gb = device_profile.memory_size_bytes / 1024**3
    peak_memory_usage_gb = runtime.memory_tracker.peak_mem_usage() / 1024**3
    device_memory_available_gb = (
        total_device_memory_gb - peak_memory_usage_gb - reserved_memory_size_gb
    )
    return device_memory_available_gb


def get_parallel_config(world_size: int, tp_size: int = 1, ep: bool = False):
    return ParallelConfig(
        world_size=world_size, tensor_parallel_size=tp_size, expert_parallel=ep
    )


def get_linear_quant_config(quant_action: QuantLinearAction, **kwargs):
    # TODO: support per-channel/per-group setting
    # TODO: support asymmetric quant setting

    if quant_action in ("W8A16_STATIC", "W8A16_DYNAMIC"):
        quant_type = LinearQuantType.W8A16
    elif quant_action in ("W8A8_STATIC", "W8A8_DYNAMIC"):
        quant_type = LinearQuantType.W8A8
    elif quant_action == "FP8":
        quant_type = LinearQuantType.FP8
    elif quant_action in ("W4A8_STATIC", "W4A8_DYNAMIC"):
        quant_type = LinearQuantType.W4A8
    else:
        raise ValueError(f"Unsupported quantization action {quant_action}")

    config_args = {
        "weight_scale": torch.tensor(1.0),
        "quant_type": quant_type,
    }
    if quant_action in ("W8A16_STATIC", "W8A8_STATIC", "W4A8_STATIC"):
        config_args["activation_scale"] = torch.tensor(1.0)
    if quant_type == LinearQuantType.FP8:
        config_args["dynamic_quant_dtype"] = DTYPE_FP8
    config_args.update(kwargs)
    return LinearQuantConfig(**config_args)


def get_quant_config(
    quant_action: QuantLinearAction, quant_lmhead: bool = False, **kwargs
):
    quant_config = QuantConfig()
    if quant_action == QuantLinearAction.DISABLED:
        return quant_config
    quant_config.linear_configs["layers.*"] = get_linear_quant_config(
        quant_action, **kwargs
    )
    quant_config.linear_configs["*.layers.*"] = get_linear_quant_config(
        quant_action, **kwargs
    )
    if quant_lmhead:
        quant_config.linear_configs["lm_head"] = get_linear_quant_config(
            quant_action, **kwargs
        )
        quant_config.linear_configs["*.lm_head"] = get_linear_quant_config(
            quant_action, **kwargs
        )
    return quant_config


def build_model(
    model_id: str,
    parallel_config: ParallelConfig,
    quant_config: QuantConfig,
    enable_lmhead: bool = True,
    num_mtp_tokens: int = 0,
    compile: bool = False,
    allow_graph_break: bool = True,
    enable_repetition=True,
    num_hidden_layers_override=0,
) -> TransformerModel:
    """
    Build a transformer model based on the given args

    :param model_id: Transformer model id.
    :param parallel_config: Parallel configuration.
    :param quant_config: Quantization configuration.
    :param num_mtp_tokens: Number of multi-token-prediction tokens.
    :param compile: Whether we want to do `torch.compile` on the model. Useful for optimal perf estimation.
    :param allow_graph_break: Allow graph break during `torch.compile`. Keep it on for better compatibility.
    :param enable_repetition: Enable TensorCast to recognize repetitive patterns in the model for better model
                              loading and execution performance.
    :return: The loaded (and possibly compiled) Transformer model.
    """
    model_config = ModelConfig(
        parallel_config,
        quant_config,
        attention_cls=AttentionTensorCast,
        quant_linear_cls=TensorCastQuantLinear,
        hf_config_json=model_id_to_json(model_id),
        enable_lmhead=enable_lmhead,
    )
    mla_module_name = model_id_to_mla_module_name(model_id)
    if mla_module_name is not None:
        mla_config = MlaConfig(
            module_name=mla_module_name,
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
    if (
        num_mtp_tokens > 0
        and (mtp_block_module_name := model_id_to_mtp_block_module_name(model_id))
        is not None
    ):
        mtp_config = MtpConfig(
            num_mtp_layers=num_mtp_tokens,
            mtp_block_module_name=mtp_block_module_name,
        )
        model_config.mtp_config = mtp_config
    model_config.enable_repetition = enable_repetition
    if num_hidden_layers_override > 0:
        model_config.num_hidden_layers_override = num_hidden_layers_override
    model = TransformerModel(model_id, model_config)
    if compile:
        model = torch.compile(
            model, backend=get_backend(), dynamic=True, fullgraph=not allow_graph_break
        )
    return model


def generate_inputs(model, query_len, seq_len, concurrency, is_decode=True):
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
    block_size = 128
    num_blocks = (
        max_context_length * batch_size + block_size
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

    # `block_tables` map logical sequence blocks to physical blocks in the KV cache.
    max_num_blocks_per_seq = (seq_len + block_size - 1) // block_size

    block_table_tensor = torch.empty(
        (batch_size, max_num_blocks_per_seq), dtype=torch.long, device="meta"
    )

    slot_mapping = torch.empty(
        (batch_size * query_len,), dtype=torch.long, device="meta"
    )

    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )

    # The total number of new tokens to be processed in this batch, concatenated.
    num_tokens = batch_size * query_len
    input_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    # Initialize the KV cache structure (also on 'meta' device).
    kv_cache_by_layers = {}
    for i in range(model.num_hidden_layers):
        if model_config.mla_config is not None:
            # Shape: [num_blocks, block_size, kv_lora_head_dim + qk_rope_head_dim]
            kv_cache_by_layers[i] = torch.empty(
                [
                    num_blocks,
                    block_size,
                    model.text_config.kv_lora_rank + model.text_config.qk_rope_head_dim,
                ],
                dtype=model_config.dtype,
                device="meta",
            )
        else:
            # Shape: [2 (K/V), num_blocks, block_size, num_heads, head_dim]
            assert (
                model.text_config.num_key_value_heads
                % parallel_config.tensor_parallel_size
                == 0
            )
            kv_cache_by_layers[i] = torch.empty(
                [
                    2,
                    num_blocks,
                    block_size,
                    model.text_config.num_key_value_heads
                    // parallel_config.tensor_parallel_size,
                    model.head_dim,
                ],
                dtype=model_config.dtype,
                device="meta",
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
        "sampling_metadata": sampling_metadata,
    }
    if model.model_id.startswith("Qwen/Qwen3-Next"):
        kwargs["cache_position"] = torch.arange(
            0, num_tokens, dtype=torch.long, device="cpu"
        )
    return kwargs
