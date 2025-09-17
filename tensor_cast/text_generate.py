import argparse
import logging
import random
import time
from enum import StrEnum
from typing import Optional

import torch

from . import config
from .compilation import get_backend
from .device import DeviceProfile

from .layers.attention import AttentionMetadataTensorCast, AttentionTensorCast
from .layers.mla import MultiheadLatentAttentionTensorCast
from .layers.quant_linear import TensorCastQuantLinear

from .layers.sampler import SamplingMetadata
from .model_config import (
    LinearQuantConfig,
    LinearQuantType,
    MlaConfig,
    ModelConfig,
    MtpConfig,
    ParallelConfig,
    QuantConfig,
)
from .performance_model.analytic import AnalyticPerformanceModel
from .performance_model.memory_tracker import MemoryTracker
from .runtime import Runtime
from .transformers.model import TransformerModel
from .transformers.utils import (
    model_id_to_json,
    model_id_to_mla_module_name,
    model_id_to_mtp_block_module_name,
)


class MachineAction(StrEnum):
    A2 = "A2"


class QuantLinearAction(StrEnum):
    W8A16_STATIC = ("W8A16_STATIC",)
    W8A8_STATIC = ("W8A8_STATIC",)
    W4A8_STATIC = ("W4A8_STATIC",)
    W8A16_DYNAMIC = ("W8A16_DYNAMIC",)
    W8A8_DYNAMIC = ("W8A8_DYNAMIC",)
    W4A8_DYNAMIC = ("W4A8_DYNAMIC",)


def get_linear_quant_config(quant_action: QuantLinearAction):
    # TODO: support per-channel/per-group setting
    # TODO: support asymmetric quant setting

    if quant_action in ("W8A16_STATIC", "W8A16_DYNAMIC"):
        quant_type = LinearQuantType.W8A16
    elif quant_action in ("W8A8_STATIC", "W8A8_DYNAMIC"):
        quant_type = LinearQuantType.W8A8
    else:
        quant_type = LinearQuantType.W4A8

    config_args = {
        "weight_scale": torch.max(torch.abs(torch.randn(1))) / 127.0,
        "quant_type": quant_type,
    }
    if quant_action in ("W8A16_STATIC", "W8A8_STATIC", "W4A8_STATIC"):
        config_args["activation_scale"] = torch.max(torch.abs(torch.randn(1))) / 127.0
    return LinearQuantConfig(**config_args)


def get_quant_config(quant_action: QuantLinearAction):
    quant_config = QuantConfig()
    quant_config.linear_configs["*"] = get_linear_quant_config(quant_action)
    return quant_config


def run_inference(
    machine: MachineAction,
    model_id: str,
    num_queries: int,
    input_length: int,
    context_length: int,
    max_context_length: int,
    do_compile: bool,
    dump_input_shapes: bool = False,
    chrome_trace: Optional[str] = None,
    quantize_linear_action: Optional[QuantLinearAction] = None,
    num_mtp_layers: int = 0,
    num_hidden_layers_override: int = 0,
    is_decode: bool = False,
):
    """
    Sets up and runs a simulated LLM inference pass.
    """
    if str(machine) not in DeviceProfile.all_machines:
        raise ValueError(f"Machine '{machine}' not recognized.")
    device_profile = DeviceProfile.all_machines[str(machine)]

    print("--- Configuration ---")
    print(f"machine: {device_profile}")
    print(f"Model ID: {model_id}")
    print(f"Number of Queries (Batch Size): {num_queries}")
    print(f"Input Length (per query): {input_length}")
    print(f"Context Length (per query): {context_length}")
    print(f"Max Context Length (per query): {max_context_length}")
    print(f"Decode: {is_decode}")
    if num_mtp_layers > 0:
        print(f"Number of MTP layers: {num_mtp_layers}")
    print(f"Use torch.compile: {do_compile}")
    print(f"Group table averages by input shapes: {dump_input_shapes}")
    if chrome_trace:
        print(f"Chrome trace output file: {chrome_trace}")
    print("---------------------\n")

    # Derived parameters
    batch_size = num_queries
    query_len = input_length
    seq_len = context_length + query_len  # Total sequence length for each query

    if seq_len + num_mtp_layers + 1 > max_context_length:
        raise ValueError(
            f"max context length {max_context_length} too small to support query {query_len}, context {context_length}"
        )

    # Paged attention parameters (can be adjusted)
    block_size = 128
    num_blocks = (
        max_context_length + block_size
    ) // block_size  # Total number of blocks available in the KV cache

    # Prepare Attention Metadata for Paged Attention
    # `query_start_loc` indicates the start of each query in the concatenated input tensor.
    # Shape: [num_queries + 1] -> e.g., [0, 50, 100, 150] for 3 queries of length 50.
    query_start_loc = torch.arange(
        0, (batch_size + 1) * query_len, query_len, dtype=torch.long
    )

    # `seq_lens` is the total length (context + new tokens) for each sequence in the batch.
    seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.long)

    # `block_tables` map logical sequence blocks to physical blocks in the KV cache.
    max_num_blocks_per_seq = (seq_len + block_size - 1) // block_size

    block_tables = []
    for _ in range(batch_size):
        # Assign random physical blocks to each sequence's logical blocks.
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)

    block_table_tensor = torch.tensor(block_tables, dtype=torch.long)

    slot_mapping = torch.empty(
        (batch_size * query_len,), dtype=torch.long, device="meta"
    )

    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )

    # Initialize Model
    print("Initializing model on 'meta' device...")
    perf_model = AnalyticPerformanceModel(device_profile)
    quant_config = QuantConfig()
    if quantize_linear_action:
        quant_config = get_quant_config(quantize_linear_action)
    model_config = ModelConfig(
        ParallelConfig(),
        quant_config,
        attention_cls=AttentionTensorCast,
        quant_linear_cls=TensorCastQuantLinear,
        hf_config_json=model_id_to_json(model_id),
    )
    if num_hidden_layers_override > 0:
        model_config.num_hidden_layers_override = num_hidden_layers_override
    mla_module_name = model_id_to_mla_module_name(model_id)
    if mla_module_name is not None:
        mla_config = MlaConfig(
            module_name=mla_module_name,
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config

    if num_mtp_layers > 0:
        mtp_block_module_name = model_id_to_mtp_block_module_name(model_id)
        if not mtp_block_module_name:
            raise ValueError(
                f"Could not find mtp block module name for {model_id}. Check if the model supports MTP."
            )
        mtp_config = MtpConfig(
            num_mtp_layers=num_mtp_layers,
            mtp_block_module_name=mtp_block_module_name,
        )
        model_config.mtp_config = mtp_config
    hf_config_json = model_id_to_json(model_id)
    if hf_config_json:
        model_config.hf_config_json = hf_config_json
    model = TransformerModel(model_id, model_config)
    if do_compile:
        print("   Compiling model with torch.compile...")
        model = torch.compile(
            model, backend=get_backend(), dynamic=True, fullgraph=True
        )
        print("   ...compilation complete.")

    print("Preparing dummy input tensors...")
    # The total number of new tokens to be processed in this batch, concatenated.
    num_tokens = batch_size * query_len
    inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    # Initialize the KV cache structure (also on 'meta' device).
    kv_cache_by_layers = {}
    for i in range(model.num_hidden_layers):
        if mla_module_name is not None:
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
    sampling_metadata = SamplingMetadata(
        query_start_loc=attn_meta.query_start_loc,
    )
    if is_decode:
        # do not prune logits
        sampling_metadata.selected_token_indices = None
    print("Running simulated inference...")
    run_start = time.perf_counter()
    with (
        Runtime(
            perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
        ) as runtime,
        torch.no_grad(),
    ):
        _ = model.forward(
            inputs,
            position_ids,
            attention_meta=attn_meta,
            kv_cache_by_layers=kv_cache_by_layers,
            sampling_metadata=sampling_metadata,
        )
    run_end = time.perf_counter()
    print()
    print(f"Model compilation and execution time: {run_end - run_start}s")
    result = runtime.table_averages(group_by_input_shapes=dump_input_shapes)
    print(result)
    print(
        f"Peak memory usage: "
        f"{max([mem_profile.usage_before_call_bytes for mem_profile in runtime.memory_tracker.get_profile()]) / 1e9} GB"
    )
    if chrome_trace:
        runtime.export_chrome_trace(chrome_trace)


def main():
    """
    Main function to parse arguments and run the inference simulation.
    """
    # TODO: add parallel configuration
    # TODO: add quantization configuration
    parser = argparse.ArgumentParser(
        description="Run a simulated LLM inference pass and dump the perf result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--machine",
        type=MachineAction,
        choices=list(MachineAction),
        default=MachineAction.A2,
        help="The machine type for simulation.",
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID from Hugging Face (e.g., 'meta-llama/Llama-2-7b-hf').",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        required=True,
        help="Number of inference queries to run in a batch.",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        required=True,
        help="The length of the new input tokens for each query.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=0,
        help="The context length for each query. Defaults to 0.",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=128 * 1024,
        help="Max supported context length for each query.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="If set, invoke torch.compile() on the model before inference.",
    )
    parser.add_argument(
        "--dump-input-shapes",
        action="store_true",
        help="If set, group the table average by input shapes",
    )
    parser.add_argument(
        "--chrome-trace",
        type=str,
        default=None,
        help="Generate chrome trace file",
    )
    parser.add_argument(
        "--quantize-linear-action",
        type=QuantLinearAction,
        choices=list(QuantLinearAction),
        default=None,
        help="Quantize all linear layers in the model from choices (currently only support symmetric quant)",
    )
    parser.add_argument(
        "--graph-log-url",
        type=str,
        default=None,
        help="For debug: the path for dumping the compiled graphs if compile is on",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Whether we are doing decode",
    )
    parser.add_argument(
        "--num-mtp-layers",
        type=int,
        default=0,
        help="Number of MTP layers, 0 means disabled - only support models having MTP like DeepSeek",
    )
    parser.add_argument(
        "--num-hidden-layers-override",
        type=int,
        default=0,
        help="Override the number of hidden layers, for debugging only",
    )

    args = parser.parse_args()

    if args.log_level:
        logging.basicConfig(level=args.log_level.upper())

    if args.graph_log_url:
        config.compilation.debug.graph_log_url = args.graph_log_url

    run_inference(
        machine=args.machine,
        model_id=args.model_id,
        num_queries=args.num_queries,
        input_length=args.input_length,
        context_length=args.context_length,
        max_context_length=args.max_context_length,
        do_compile=args.compile,
        dump_input_shapes=args.dump_input_shapes,
        chrome_trace=args.chrome_trace,
        quantize_linear_action=args.quantize_linear_action,
        num_mtp_layers=args.num_mtp_layers,
        num_hidden_layers_override=args.num_hidden_layers_override,
        is_decode=args.decode,
    )


if __name__ == "__main__":
    main()
