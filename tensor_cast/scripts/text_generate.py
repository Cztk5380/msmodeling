import argparse
import logging
import time

import torch

from .. import config, device_profiles  # noqa: F401
from ..device import DeviceProfile
from ..performance_model.analytic import AnalyticPerformanceModel
from ..performance_model.memory_tracker import MemoryTracker
from ..performance_model.utils import bytes_of_tensor
from ..runtime import Runtime
from .utils import (
    build_model,
    generate_inputs,
    QuantizeAttentionAction,
    QuantizeLinearAction,
    UserInputConfig,
)


def run_inference(user_input: UserInputConfig):
    """
    Sets up and runs a simulated LLM inference pass.
    """

    device_profile = DeviceProfile.all_device_profiles[user_input.device]

    # Initialize Model
    print("Initializing model on 'meta' device...")
    perf_model = AnalyticPerformanceModel(device_profile)
    model = build_model(user_input=user_input).eval()
    print("Preparing dummy input tensors...")
    input_kwargs = generate_inputs(
        model, user_input.query_len, user_input.seq_len, user_input.num_queries
    )
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
    result = runtime.table_averages(group_by_input_shapes=user_input.dump_input_shapes)
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
        total_device_memory_gb - peak_memory_usage_gb - user_input.reserved_memory_gb
    )
    print(f"Total device memory: {total_device_memory_gb:.3f} GB")
    print(f"  Model weight size: {model_weight_size_gb:.3f} GB")
    print(f"  KV cache: {total_kv_cache_size_gb:.3f} GB")
    print(f"  Model activation size: {model_activation_size_gb:.3f} GB")
    print(f"  Reserved memory: {user_input.reserved_memory_gb} GB")
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
    if user_input.chrome_trace:
        runtime.export_chrome_trace(user_input.chrome_trace)

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
        "--o-proj-tp-size",
        type=int,
        default=None,
        help="The tp size for attn o_proj layer",
    )
    parser.add_argument(
        "--o-proj-dp-size",
        type=int,
        default=None,
        help="The dp size for attn o_proj layer",
    )
    parser.add_argument(
        "--mlp-tp-size",
        type=int,
        default=None,
        help="The tp size for mlp layer, can override tp-size for mlp layer",
    )
    parser.add_argument(
        "--mlp-dp-size",
        type=int,
        default=None,
        help="The dp size for mlp layer, can override dp-size for mlp layer",
    )
    parser.add_argument(
        "--lmhead-tp-size",
        type=int,
        default=None,
        help="The tp size for lm head, can override tp-size for lm head",
    )
    parser.add_argument(
        "--lmhead-dp-size",
        type=int,
        default=None,
        help="The dp size for lm head, can override dp-size for lm head",
    )
    parser.add_argument(
        "--ep",
        action="store_true",
        help="Whether or not to implement expert parallel",
    )
    parser.add_argument(
        "--word-embedding-tp",
        action="store_true",
        help="Whether or not to implement word embedding tensor parallel",
    )
    parser.add_argument(
        "--enable-redundant-experts",
        action="store_true",
        help="Whether or not to use redundant experts. When this flag is True: "
        "if the externalization of shared experts is not enabled at this time, "
        "each device will add one redundant expert. If the externalization of shared experts is enabled "
        "and the number of routing experts on each device is the same, "
        "then each device hosting the routing experts will also add one redundant expert.",
    )
    parser.add_argument(
        "--enable-external-shared-experts",
        action="store_true",
        help="Whether or not to implement external shared experts",
    )
    parser.add_argument(
        "--host-external-shared-experts",
        action="store_true",
        help="Whether to have the current device host the external shared experts",
    )

    args = parser.parse_args()

    if args.log_level:
        logging.basicConfig(level=args.log_level.upper())

    if args.graph_log_url:
        config.compilation.debug.graph_log_url = args.graph_log_url

    user_input = UserInputConfig.from_args(args)
    run_inference(user_input)


if __name__ == "__main__":
    main()
