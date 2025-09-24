import argparse
import logging
import time
from enum import StrEnum
from typing import Optional

import torch

from . import config, device_profiles  # noqa: F401
from .compilation import get_backend
from .device import DeviceProfile

from .layers.attention import AttentionTensorCast
from .layers.mla import MultiheadLatentAttentionTensorCast
from .layers.quant_linear import TensorCastQuantLinear

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

from .scripts.utils import generate_inputs
from .transformers.model import TransformerModel
from .transformers.utils import (
    default_repetition_config,
    model_id_to_json,
    model_id_to_mla_module_name,
    model_id_to_mtp_block_module_name,
    model_id_to_repetition_config,
)


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
    device: str,
    model_id: str,
    num_queries: int,
    query_len: int,
    context_length: int,
    do_compile: bool,
    allow_graph_break: bool,
    dump_input_shapes: bool = False,
    chrome_trace: Optional[str] = None,
    quantize_linear_action: Optional[QuantLinearAction] = None,
    num_mtp_layers: int = 0,
    num_hidden_layers_override: int = 0,
    is_decode: bool = False,
    enable_repetition: bool = False,
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
    print(f"Enable repetition: {enable_repetition}")
    if num_mtp_layers > 0:
        print(f"Number of MTP layers: {num_mtp_layers}")
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
    if quantize_linear_action:
        quant_config = get_quant_config(quantize_linear_action)
    model_config = ModelConfig(
        parallel_config,
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
    if enable_repetition:
        repetition_config = model_id_to_repetition_config(model_id)
        if not repetition_config:
            repetition_config = default_repetition_config()
        model_config.repetitive_layer_config = repetition_config
    model = TransformerModel(model_id, model_config)
    if do_compile:
        print("   Compiling model with torch.compile...")
        compile_backend = get_backend()
        model = torch.compile(
            model, backend=compile_backend, dynamic=True, fullgraph=not allow_graph_break
        )
        print("   ...compilation complete.")

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
            kv_cache.nelement() * kv_cache.element_size()
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
    parser.add_argument(
        "--enable-repetition",
        action="store_true",
        help="Leverage the repetition pattern of the transformer models to save runtime cost",
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
        query_len=args.input_length,
        context_length=args.context_length,
        do_compile=args.compile,
        allow_graph_break=args.compile_allow_graph_break,
        dump_input_shapes=args.dump_input_shapes,
        chrome_trace=args.chrome_trace,
        quantize_linear_action=args.quantize_linear_action,
        num_mtp_layers=args.num_mtp_layers,
        num_hidden_layers_override=args.num_hidden_layers_override,
        is_decode=args.decode,
        enable_repetition=args.enable_repetition,
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
