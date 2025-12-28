import argparse
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from ..core.utils import (
    build_model,
    generate_inputs,
    get_available_memory_gb,
    QuantizeAttentionAction,
    QuantizeLinearAction,
    RequestInfo,
    UserInputConfig,
)
from ..device import DeviceProfile
from ..performance_model.analytic import AnalyticPerformanceModel
from ..performance_model.memory_tracker import MemoryTracker
from ..runtime import Runtime
from ..transformers.model import TransformerModel

logger = logging.getLogger(__name__)


def get_benchmark_query_and_seq_length(
    input_length, output_length, is_decode=True, num_mtp_tokens=0, context_length=0
):
    if is_decode:
        query_len = num_mtp_tokens + 1
        seq_len = (
            input_length + output_length // 2 + context_length + query_len
        )  # Assume the average TPOT happens in the middle of the auto-regressive loop at `output_lenght // 2`
    else:
        query_len = input_length
        seq_len = context_length + query_len
    return query_len, seq_len


def find_best_throughput(
    model: TransformerModel,
    device_profile: DeviceProfile,
    input_length: int,
    output_length: int,
    slo_limit: float,
    is_decode: bool,
    mtp_acceptance_rate: Optional[List[float]] = None,
    reserved_memory_size_gb: float = 10,  # assume 10GB reserved memory
    serving_overhead_s: float = 0.002,  # assume 2ms serving cost by default
) -> Tuple[
    float, int, Dict[str, float], Optional[str]
]:  # (latency, concurrency, breakdown, error message)
    slo_name = "TPOT" if is_decode else "TTFT"

    if mtp_acceptance_rate is None:
        mtp_acceptance_rate = [0.9, 0.6, 0.4, 0.2]

    model_config = model.model_config
    num_mtp_tokens = (
        model_config.mtp_config.num_mtp_layers
        if model_config.mtp_config is not None
        else 0
    )
    assert num_mtp_tokens <= len(mtp_acceptance_rate)

    def error(latency, available_memory_gb, *args, **kwargs):
        if latency <= 0:
            return "Runtime Error"
        elif latency > slo_limit:
            return f"{slo_name} {latency * 1000:.3f} exceeds limit"
        elif available_memory_gb <= 0:
            return f"OOM: {available_memory_gb}"
        return ""

    def run(concurrency):
        try:
            query_len, seq_len = get_benchmark_query_and_seq_length(
                input_length,
                output_length,
                is_decode=is_decode,
                num_mtp_tokens=num_mtp_tokens,
            )

            inputs = generate_inputs(
                model,
                [
                    RequestInfo(
                        query_len=query_len,
                        seq_len=seq_len,
                        concurrency=concurrency,
                        is_decode=is_decode,
                    )
                ],
            )
            perf_model = AnalyticPerformanceModel(device_profile)
            with (
                Runtime(
                    perf_model,
                    device_profile,
                    memory_tracker=MemoryTracker(device_profile),
                ) as runtime,
                torch.no_grad(),
            ):
                _ = model.forward(**inputs)
            latency = (
                runtime.total_execution_time_s()[perf_model.name] + serving_overhead_s
            )
            if is_decode:
                average_tokens = sum(mtp_acceptance_rate[:num_mtp_tokens]) + 1
                latency /= average_tokens
            available_memory_gb = get_available_memory_gb(
                device_profile, runtime, reserved_memory_size_gb
            )
            return (
                latency,
                available_memory_gb,
                next(iter(runtime.get_breakdowns().values())),
            )
        except (
            RuntimeError,
            AssertionError,
            torch._dynamo.exc.Unsupported,
        ):  # TODO(jgong5): catch assertion due to limited support of TP+EP, need to fix
            return 0, math.inf, {}

    # 1. Exponentially search to find an upper bound quickly.
    min_concurrency = model_config.parallel_config.data_parallel_size
    concurrency = min_concurrency
    max_concurrency = 0
    while True:
        latency, available_memory_gb, breakdown = run(concurrency)
        error_msg = error(latency, available_memory_gb)
        if not error_msg:
            max_concurrency = concurrency
            concurrency *= 2
        else:
            break
    if max_concurrency == 0:
        return latency, concurrency, breakdown, error_msg

    # 2. Binary search between the last known good value and the first failed one.
    low = max_concurrency
    high = concurrency
    best_concurrency = max_concurrency

    while low <= high:
        mid = (low + high) // 2
        if mid <= best_concurrency:
            # If mid is not greater than our current best, no need to test.
            # This also prevents infinite loops when low = mid.
            low = mid + 1
            continue

        latency, available_memory_gb, _ = run(mid)
        if not error(latency, available_memory_gb):
            # 'mid' is a better candidate. Update our best and search for higher values.
            best_concurrency = mid
            low = mid + 1
        else:
            # 'mid' failed. The optimal value must be lower.
            high = mid - 1

    # 3. Return the final latency and the best concurrency found.
    final_latency, _, breakdown = run(best_concurrency)
    return final_latency, best_concurrency, breakdown, ""


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM inference on given devices and models to search for best throughput under "
        "given input/output sequence length and SLO limitations",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input-length",
        type=int,
        required=True,
        help="The input length of the prompt.",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        required=True,
        help="The expected output length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=list(DeviceProfile.all_device_profiles.keys()),
        default="",
        help="The device type for benchmarking.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Model ID from Hugging Face (e.g., 'meta-llama/Llama-2-7b-hf').",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=1,
        help="Number of devices",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="",
        help="""
Configuration file for device list, model list and number of devices, overriding --device, --model-id, --num-devices
Example config.yaml format:
---------------------------
devices:
  - "ATLAS_800_A2_280T_64G"
  - "H20"

models:
  "Qwen/Qwen3-32B": [1, 2, 4]  # number of devices to test
  "zai-org/GLM-4.5": [4, 8]

""",
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
        "--num-mtp-tokens",
        type=int,
        default=0,
        help="Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek",
    )
    parser.add_argument(
        "--ttft-limits",
        type=float,
        default=[1],
        nargs="+",
        help="A list of TTFT constraints under which to search for the best throughput.",
    )
    parser.add_argument(
        "--mtp-acceptance-rate",
        type=float,
        default=[0.9, 0.6, 0.4, 0.2],
        nargs="+",
        help="Acceptance rate list for MTP",
    )
    parser.add_argument(
        "--tpot-limits",
        type=float,
        default=[0.05],
        nargs="+",
        help="A list of TPOT constraints under which to search for the best throughput.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["decode", "prefill", "both"],
        help="Inference mode",
    )
    parser.add_argument(
        "--quantize-linear-action",
        type=QuantizeLinearAction,
        choices=list(QuantizeLinearAction),
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        help="Quantize all linear layers in the model from choices (currently only support symmetric quant)",
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
        "--log-level",
        type=str,
        default=None,
        help="Logging level",
    )
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            config_data = yaml.safe_load(f)
            device_name_list = config_data.get("devices", [])
            device_list = [
                DeviceProfile.all_device_profiles[device] for device in device_name_list
            ]
            model_id_to_decode_device_num = config_data.get("models", {})
    else:
        if not args.device or not args.model_id:
            raise ValueError(
                "You should either specify --device, --model-id or a configuration file via --config"
            )
        device_list = [DeviceProfile.all_device_profiles[args.device]]
        model_id_to_decode_device_num = {args.model_id: [args.num_devices]}
    user_input = UserInputConfig.from_args(args)
    if user_input.do_compile:
        torch._dynamo.config.recompile_limit = 1000000
        torch._dynamo.config.accumulated_recompile_limit = 1000000

    input_length = args.input_length
    output_length = args.output_length
    ttft_limits = args.ttft_limits
    tpot_limits = args.tpot_limits
    inference_modes = [True, False]
    if args.mode == "decode":
        inference_modes = [True]
    elif args.mode == "prefill":
        inference_modes = [False]
    if args.log_level:
        logging.basicConfig(level=args.log_level.upper())

    for is_decode in inference_modes:
        slo_name = "TPOT" if is_decode else "TTFT"
        slo_limits = tpot_limits if is_decode else ttft_limits
        print(
            f"Device Type, Number of Devices, Input Length, Output Length, Model, "
            f"Linear Quant Type, Attn Quant Type, TP Size, Use EP, MTP Tokens, "
            f"{slo_name} Target(ms), Concurrency, {slo_name}(ms), "
            f"Total TPS, TPS/Device, Mem, Comm, Cube, Vec, Error Message"
        )
        for model_id, device_num_list in model_id_to_decode_device_num.items():
            for device_profile in device_list:
                for num_devices in device_num_list:
                    if device_profile.comm_grid.grid.nelement() < num_devices:
                        continue
                    user_input.model_id = model_id
                    tp_size_list = [1 << i for i in range(num_devices.bit_length())]
                    for tp_size in tp_size_list:
                        torch.compiler.reset()
                        user_input.tp_size = tp_size
                        model = build_model(user_input).eval()

                        num_mtp_tokens = (
                            model.model_config.mtp_config.num_mtp_layers
                            if model.model_config.mtp_config is not None
                            else 0
                        )

                        # When Attention is not MLA,
                        # it is not supported when the kv heads can not be evenly distributed across a tp group
                        # and uniformly replicated within a tp group.
                        if (
                            model.model_config.mla_config is None
                            and model.text_config.num_key_value_heads % tp_size != 0
                            and tp_size % model.text_config.num_key_value_heads != 0
                        ):
                            continue
                        for slo_limit in slo_limits:
                            latency, concurrency, breakdown, err_msg = (
                                find_best_throughput(
                                    model,
                                    device_profile,
                                    input_length,
                                    output_length,
                                    slo_limit,
                                    is_decode,
                                    mtp_acceptance_rate=user_input.mtp_acceptance_rate,
                                )
                            )
                            TPS = concurrency / latency if latency != 0 else 0
                            if not is_decode:
                                TPS *= input_length
                            total = sum(breakdown.values())
                            if total == 0:
                                continue
                            percentage_breakdown = [
                                f"{value * 100 / total:.2f}"
                                for value in breakdown.values()
                            ]
                            print(
                                f"{device_profile.name}, {num_devices}, {input_length}, {output_length}, {model_id}, "
                                f"{user_input.quantize_linear_action}, {user_input.quantize_attention_action}, "
                                f"{tp_size}, {model.model_config.parallel_config.expert_parallel}, "
                                f"{num_mtp_tokens}, {slo_limit * 1000:.3f}, {concurrency}, {latency * 1000:.3f}, "
                                f"{TPS:.1f}, {TPS / num_devices:.1f}, {','.join(percentage_breakdown)}, {err_msg}",
                                flush=True,
                            )


if __name__ == "__main__":
    main()
