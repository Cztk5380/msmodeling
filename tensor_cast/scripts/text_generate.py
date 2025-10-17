import argparse
import logging
import time
from typing import Optional

import torch

from .. import config, device_profiles  # noqa: F401
from ..device import DeviceProfile

from ..model_config import ParallelConfig, QuantConfig, QuantGranularity
from ..performance_model.analytic import AnalyticPerformanceModel
from ..performance_model.memory_tracker import MemoryTracker

from ..performance_model.utils import bytes_of_tensor
from ..runtime import Runtime

from .utils import (
    build_model,
    create_quant_config,
    generate_inputs,
    QuantizeAttentionAction,
    QuantizeLinearAction,
)


def run_inference(
    device: str,
    model_id: str,
    num_queries: int,
    query_len: int,
    context_length: int,
    do_compile: bool,
    allow_graph_break: bool,
    dump_input_shapes: bool = False,
    chrome_trace: Optional[str] = None,
    quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.W8A8_DYNAMIC,
    quantize_lmhead: bool = False,
    mxfp4_group_size: int = 32,
    quantize_attention_action: QuantizeAttentionAction = QuantizeAttentionAction.DISABLED,
    num_mtp_tokens: int = 0,
    num_hidden_layers_override: int = 0,
    is_decode: bool = False,
    disable_repetition: bool = False,
    reserved_memory_gb=0,
    world_size: int = 1,
    tp_size: int = 1,
    dp_size: Optional[int] = None,
    mlp_tp_size: Optional[int] = None,
    mlp_dp_size: Optional[int] = None,
    lmhead_tp_size: Optional[int] = None,
    lmhead_dp_size: Optional[int] = None,
    ep: bool = False,
):
    """
    Sets up and runs a simulated LLM inference pass.
    """
    if device not in DeviceProfile.all_device_profiles:
        raise ValueError(f"Device '{device}' not recognized.")
    device_profile = DeviceProfile.all_device_profiles[device]

    parallel_config = ParallelConfig(
        world_size=world_size,
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        mlp_tensor_parallel_size=mlp_tp_size,
        mlp_data_parallel_size=mlp_dp_size,
        lmhead_tensor_parallel_size=lmhead_tp_size,
        lmhead_data_parallel_size=lmhead_dp_size,
        expert_parallel=ep,
    )
    batch_size = (
        num_queries + parallel_config.data_parallel_size - 1
    ) // parallel_config.data_parallel_size
    seq_len = context_length + query_len  # Total sequence length for each query

    print("--- Configuration ---")
    print(f"Device: {device}")
    print(f"Model ID: {model_id}")
    print(f"Number of Queries: {num_queries}")
    print(f"Number of Queries per DP rank: {batch_size}")
    print(f"Input Length (per query): {query_len}")
    print(f"Context Length (per query): {context_length}")
    print(f"Decode: {is_decode}")
    if quantize_linear_action != QuantizeLinearAction.DISABLED:
        print(
            f"Quantization Linear: {quantize_linear_action}, quantize LM Head: {quantize_lmhead}"
        )
        if quantize_linear_action == QuantizeLinearAction.MXFP4:
            print(f"  MXFP4 group size: {mxfp4_group_size}")
    else:
        print("Quantization Linear: Disabled")
    if quantize_attention_action != QuantizeAttentionAction.DISABLED:
        print(f"Quantization Attention: {quantize_attention_action}")
    else:
        print("Quantization Attention: Disabled")
    print(f"Enable repetition: {not disable_repetition}")
    if num_mtp_tokens > 0:
        print(f"Number of MTP layers: {num_mtp_tokens}")
    print(f"Use torch.compile: {do_compile}")
    if do_compile:
        print(f"  allow graph break: {allow_graph_break}")
    print(f"Group table averages by input shapes: {dump_input_shapes}")
    if chrome_trace:
        print(f"Chrome trace output file: {chrome_trace}")
    print("---------------------\n")

    # Initialize Model
    print("Initializing model on 'meta' device...")
    perf_model = AnalyticPerformanceModel(device_profile)
    quant_config = QuantConfig()
    if (
        quantize_linear_action != QuantizeLinearAction.DISABLED
        or quantize_attention_action != QuantizeAttentionAction.DISABLED
    ):
        extra_kwargs = {}
        if quantize_linear_action == QuantizeLinearAction.MXFP4:
            extra_kwargs.update(
                weight_group_size=mxfp4_group_size,
                weight_quant_granularity=QuantGranularity.PER_GROUP,
            )
        quant_config = create_quant_config(
            quantize_linear_action,
            quantize_lmhead=quantize_lmhead,
            quantize_attention_action=quantize_attention_action,
            **extra_kwargs,
        )
    model = build_model(
        model_id,
        parallel_config,
        quant_config,
        enable_lmhead=True,
        num_mtp_tokens=num_mtp_tokens,
        compile=do_compile,
        allow_graph_break=allow_graph_break,
        enable_repetition=not disable_repetition,
        num_hidden_layers_override=num_hidden_layers_override,
    )
    print("Preparing dummy input tensors...")
    input_kwargs = generate_inputs(model, query_len, seq_len, num_queries)
    print("Running simulated inference...")
    run_start = time.perf_counter()
    with (
        Runtime(
            perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
        ) as runtime,
        torch.no_grad(),
    ):
        _ = model.forward(**input_kwargs)
    run_end = time.perf_counter()
    execution_time_s = runtime.total_execution_time_s()[perf_model.name]
    print()
    print(f"Model compilation and execution time: {run_end - run_start}s")
    result = runtime.table_averages(group_by_input_shapes=dump_input_shapes)
    print(result)
    # Print memory usage
    total_device_memory_gb = device_profile.memory_size_bytes / 1024**3
    model_weight_size_gb = model.weight_size / 1024**3
    peak_memory_usage_gb = runtime.memory_tracker.peak_mem_usage() / 1024**3
    total_kv_cache_size_gb = (
        sum(
            bytes_of_tensor(kv_cache)
            for kv_cache in input_kwargs["kv_cache_by_layers"].values()
        )
        / 1024**3
    )
    model_activation_size_gb = (
        peak_memory_usage_gb - total_kv_cache_size_gb - model_weight_size_gb
    )
    device_memory_available_gb = (
        total_device_memory_gb - peak_memory_usage_gb - reserved_memory_gb
    )
    print(f"Total device memory: {total_device_memory_gb:.3f} GB")
    print(f"  Model weight size: {model_weight_size_gb:.3f} GB")
    print(f"  KV cache: {total_kv_cache_size_gb:.3f} GB")
    print(f"  Model activation size: {model_activation_size_gb:.3f} GB")
    print(f"  Reserved memory: {reserved_memory_gb} GB")
    print(f"  Memory available: {device_memory_available_gb} GB")

    print("Stats breakdowns:")
    for breakdown_name, breakdown in runtime.get_breakdowns().items():
        total = sum(breakdown.values())
        if total == 0:
            continue
        percentage_breakdown = [value * 100 / total for value in breakdown.values()]
        print(f"  {breakdown_name}: ", end="")
        print(
            [
                f"{key}: {percentage:.2f}"
                for key, percentage in zip(breakdown.keys(), percentage_breakdown)
            ]
        )
    if chrome_trace:
        runtime.export_chrome_trace(chrome_trace)

    # Return metrics for validation
    return {
        "total_device_memory_gb": total_device_memory_gb,
        "model_weight_size_gb": model_weight_size_gb,
        "peak_memory_usage_gb": peak_memory_usage_gb,
        "kv_cache_size_gb": total_kv_cache_size_gb,
        "model_activation_size_gb": model_activation_size_gb,
        "device_memory_available_gb": device_memory_available_gb,
        "execution_time_s": execution_time_s,
        "table_result": result,
        "breakdowns": runtime.get_breakdowns(),
    }


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
        "--device",
        type=str,
        choices=list(DeviceProfile.all_device_profiles.keys()),
        default="TEST_DEVICE",
        help="The device type for simulation.",
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
        "--query-length",
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
        "--compile",
        action="store_true",
        help="If set, invoke torch.compile() on the model before inference.",
    )
    parser.add_argument(
        "--compile-allow-graph-break",
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
        type=QuantizeLinearAction,
        choices=list(QuantizeLinearAction),
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        help="Quantize all linear layers in the model from choices (currently only support symmetric quant)",
    )
    parser.add_argument(
        "--quantize-lmhead",
        action="store_true",
        help="Whether to quantize LM Head, off by default since quantizing LM Head usually impact accuracy a lot",
    )
    parser.add_argument(
        "--mxfp4-group-size",
        type=int,
        default=32,
        help="Group size for MXFP4 quantization",
    )
    parser.add_argument(
        "--quantize-attention-action",
        type=QuantizeAttentionAction,
        choices=list(QuantizeAttentionAction),
        default=QuantizeAttentionAction.DISABLED,
        help="Quantize the KV cache with the given action",
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
        "--num-mtp-tokens",
        type=int,
        default=0,
        help="Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek",
    )
    parser.add_argument(
        "--num-hidden-layers-override",
        type=int,
        default=0,
        help="Override the number of hidden layers, for debugging only",
    )
    parser.add_argument(
        "--disable-repetition",
        action="store_true",
        help="Preserve the original behavior of the transformer models. Do not leverage the repetition "
        "pattern of the transformer models to save runtime cost",
    )
    parser.add_argument(
        "--reserved-memory-gb",
        default=0,
        help="Size of reserved device memory (in GB) that we cannot use from applications.",
    )
    # ========== ParallelConfig Parameters ==========
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="The total number of processes",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The tp size for the whole model",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=None,
        help="The dp size for the whole model",
    )
    parser.add_argument(
        "--mlp-tp-size",
        type=int,
        default=None,
        help="The tp size fo mlp layer, can override tp-size for mlp layer",
    )
    parser.add_argument(
        "--mlp-dp-size",
        type=int,
        default=None,
        help="The dp size fo mlp layer, can override dp-size for mlp layer",
    )
    parser.add_argument(
        "--lmhead-tp-size",
        type=int,
        default=None,
        help="The tp size fo lm head, can override tp-size for lm head",
    )
    parser.add_argument(
        "--lmhead-dp-size",
        type=int,
        default=None,
        help="The dp size fo lm head, can override dp-size for lm head",
    )
    parser.add_argument(
        "--ep",
        action="store_true",
        help="Whether or not to implement expert parallel",
    )

    args = parser.parse_args()

    if args.log_level:
        logging.basicConfig(level=args.log_level.upper())

    if args.graph_log_url:
        config.compilation.debug.graph_log_url = args.graph_log_url

    run_inference(
        device=args.device,
        model_id=args.model_id,
        num_queries=args.num_queries,
        query_len=args.query_length,
        context_length=args.context_length,
        do_compile=args.compile,
        allow_graph_break=args.compile_allow_graph_break,
        dump_input_shapes=args.dump_input_shapes,
        chrome_trace=args.chrome_trace,
        quantize_linear_action=args.quantize_linear_action,
        quantize_lmhead=args.quantize_lmhead,
        mxfp4_group_size=args.mxfp4_group_size,
        quantize_attention_action=args.quantize_attention_action,
        num_mtp_tokens=args.num_mtp_tokens,
        num_hidden_layers_override=args.num_hidden_layers_override,
        is_decode=args.decode,
        disable_repetition=args.disable_repetition,
        reserved_memory_gb=args.reserved_memory_gb,
        world_size=args.world_size,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        mlp_tp_size=args.mlp_tp_size,
        mlp_dp_size=args.mlp_dp_size,
        lmhead_tp_size=args.lmhead_tp_size,
        lmhead_dp_size=args.lmhead_dp_size,
        ep=args.ep,
    )


if __name__ == "__main__":
    main()
