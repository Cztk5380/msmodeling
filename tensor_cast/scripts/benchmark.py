import argparse
import bisect
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from ..core.input_generator import generate_inputs, RequestInfo
from ..core.model_builder import build_model
from ..core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from ..core.user_config import UserInputConfig
from ..core.utils import get_available_memory_gb
from .. import device_profiles # noqa: F401
from ..device import DeviceProfile
from ..performance_model.analytic import AnalyticPerformanceModel
from ..performance_model.memory_tracker import MemoryTracker
from ..runtime import Runtime
from ..transformers.model import TransformerModel
from .utils import check_positive_integer

logger = logging.getLogger(__name__)


class ConcurrencyRangeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) not in (1, 2):
            raise argparse.ArgumentTypeError(
                f"{option_string} expects [min max] or [max], got {values}"
            )
        if any(v <= 0 for v in values):
            raise argparse.ArgumentTypeError(
                f"{option_string} values must be > 0, got {values}"
            )
        setattr(namespace, self.dest, values)


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
    concurrency_range: Optional[List[int]] = None,  # [min, max] for concurrency search
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

    def run_and_check(concurrency):
        """Run benchmark and return (latency, available_memory_gb, breakdown, error_msg)."""
        latency, available_memory_gb, breakdown = run(concurrency)
        error_msg = error(latency, available_memory_gb)
        return latency, available_memory_gb, breakdown, error_msg

    def is_feasible(concurrency):
        """Check if concurrency is feasible (no error)."""
        _, _, _, err = run_and_check(concurrency)
        return not err

    def binary_search_max_feasible(low, high):
        """Use bisect to find maximum feasible concurrency in [low, high]."""
        if low > high:
            return low
        candidates = range(low, high + 1)
        idx = bisect.bisect_left(candidates, True, key=lambda x: not is_feasible(x))
        return candidates[idx - 1] if idx > 0 else low

    def exponential_search_bounds(start, upper_limit=None):
        """Find bounds for binary search using exponential growth."""
        low = start
        high = start
        if upper_limit is None:
            while True:
                high *= 2
                if is_feasible(high):
                    low = high
                else:
                    return low, high

        upper_limit = max(upper_limit, start)
        while high < upper_limit:
            next_high = min(high * 2, upper_limit)
            if is_feasible(next_high):
                low = next_high
                high = next_high
            else:
                return low, next_high
        return low, high

    concurrency_min, concurrency_max = None, None

    if concurrency_range is not None:
        if len(concurrency_range) == 1:
            concurrency_max = concurrency_range[0]
        else:
            concurrency_min, concurrency_max = sorted(concurrency_range)

    if concurrency_min is not None:
        search_min = concurrency_min
    else:
        search_min = 1

    latency, _, breakdown, error_msg = run_and_check(search_min)
    if error_msg:
        return latency, search_min, breakdown, error_msg

    if concurrency_max is not None:
        _, upper_bound = exponential_search_bounds(search_min, concurrency_max)
        best_concurrency = binary_search_max_feasible(search_min, upper_bound)
    else:
        lower_bound, upper_bound = exponential_search_bounds(search_min)
        best_concurrency = binary_search_max_feasible(lower_bound, upper_bound)

    final_latency, _, breakdown, _ = run_and_check(best_concurrency)
    return final_latency, best_concurrency, breakdown, ""


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM inference on given devices and models to search for best throughput under "
        "given input/output sequence length and SLO limitations",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input-length",
        type=check_positive_integer,
        required=True,
        help="The input length of the prompt.",
    )
    parser.add_argument(
        "--output-length",
        type=check_positive_integer,
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
        type=check_positive_integer,
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
        type=check_positive_integer,
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
    parser.add_argument(
        "--tp-sizes",
        type=int,
        nargs="+",
        default=None,
        help="TP sizes to search (default: powers of 2 up to num_devices)",
    )
    parser.add_argument(
        "--concurrency-range",
        type=int,
        nargs="+",
        action=ConcurrencyRangeAction,
        default=None,
        help="Concurrency range: [min max] or [max] (default: 1 for min, no limit for max)",
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
                    if args.tp_sizes:
                        tp_size_list = [tp for tp in args.tp_sizes if tp <= num_devices]
                        if not tp_size_list:
                            raise ValueError(
                                f"All specified TP sizes {args.tp_sizes} exceed num_devices ({num_devices})"
                            )
                    else:
                        tp_size_list = [1 << i for i in range(num_devices.bit_length())]
                    for tp_size in tp_size_list:
                        torch.compiler.reset()
                        user_input.tp_size = tp_size
                        # if the moe_config is None, ep will be set False in update_parallel_config
                        # so set it True here, moe models can enable ep parallel correctly
                        user_input.ep = True
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
                                    concurrency_range=args.concurrency_range,
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
