import logging
import math
from typing import Dict, List, Optional, Tuple

import torch

from .. import device_profiles  # noqa: F401

from ..device import ATLAS_800, DeviceProfile
from ..model_config import LinearQuantType
from ..performance_model.analytic import AnalyticPerformanceModel
from ..performance_model.memory_tracker import MemoryTracker
from ..runtime import Runtime
from ..transformers.model import TransformerModel
from ..transformers.utils import model_id_to_moe_config

from .utils import (
    build_model,
    generate_inputs,
    get_available_memory_gb,
    get_parallel_config,
    get_quant_config,
)

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
) -> Tuple[float, int, Dict[str, float], Optional[str]]:  # (latency, concurrency)
    slo_name = "TPOT" if is_decode else "TTFT"

    if mtp_acceptance_rate is None:
        mtp_acceptance_rate = [0.9, 0.6, 0.4, 0.2]

    model_config = model.model_config
    num_mtp_layers = (
        model_config.mtp_config.num_mtp_layers
        if model_config.mtp_config is not None
        else 0
    )
    assert num_mtp_layers <= len(mtp_acceptance_rate)

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
                num_mtp_tokens=num_mtp_layers,
            )
            inputs = generate_inputs(
                model,
                query_len,
                seq_len,
                concurrency,
                is_decode=is_decode,
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
                average_tokens = sum(mtp_acceptance_rate[:num_mtp_layers]) + 1
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
            AssertionError
        ):  # TODO(jgong5): catch assertion due to limited support of TP+EP, need to fix
            return 0, math.inf, {}

    # 1. Exponentially search to find an upper bound quickly.
    min_concurrency = model_config.parallel_config.data_parallel_size
    concurrency = min_concurrency
    max_concurrency = 0
    while True:
        error_msg = error(*run(concurrency))
        if not error_msg:
            max_concurrency = concurrency
            concurrency *= 2
        else:
            break
    if max_concurrency == 0:
        return 0, concurrency, {}, error_msg

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


# device_list = list(DeviceProfile.all_device_profiles)
# device_list.remove(TEST_DEVICE)
device_list = [
    ATLAS_800.A3_752T_128G_DIE,
    ATLAS_800.A3_560T_128G_DIE,
    ATLAS_800.A2_376T_64G,
    ATLAS_800.A2_313T_64G,
    ATLAS_800.A2_280T_64G,
    ATLAS_800.A2_280T_64G_PCIE,
    ATLAS_800.A2_280T_32G_PCIE,
    # NVIDIA.H20,
    # NVIDIA.RTX_5090D,
]

model_id_to_decode_device_num = {
    "Qwen/Qwen3-32B": [16],
    "Qwen/Qwen2.5-72B": [1, 2, 4, 8],
    "Qwen/Qwen3-14B": [1, 2, 4, 8],
    "Qwen/Qwen3-8B": [1, 2, 4, 8],
    "Qwen/Qwen3-30B-A3B": [1, 2, 4, 8],
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": [4, 8, 16, 32, 64],
    "Qwen/Qwen3-235B-A22B": [4, 8, 16, 32, 64],
    "zai-org/GLM-4.5": [4, 8, 16, 32, 64],
    "deepseek-ai/DeepSeek-V3.1": [8, 16, 32, 64],
    "moonshotai/Kimi-K2-Base": [16, 32, 64],
    "Qwen/Qwen3-Next-80B-A3B-Instruct": [16, 32, 64],
}

torch._dynamo.config.recompile_limit = 10000
torch._dynamo.config.accumulated_recompile_limit = 10000

compile = True
mtp = 1
linear_quant_type = LinearQuantType.W8A8
input_length = 3500
output_length = 1500
ttft_limits = [1]
tpot_limits = [0.05]

logging.basicConfig(level="INFO")

for is_decode in [True, False]:
    slo_name = "TPOT" if is_decode else "TTFT"
    slo_limits = tpot_limits if is_decode else ttft_limits
    print(
        f"Device Type, Number of Devices, Input Length, Output Length, Model, "
        f"Quant Type, TP Size, Use EP, Use MTP, {slo_name} Target(ms), Concurrency, {slo_name}(ms), "
        f"Total TPS, TPS/Device, Mem, Comm, Cube, Vec, Error Message"
    )
    for model_id, device_num_list in model_id_to_decode_device_num.items():
        for device_profile in device_list:
            for num_devices in device_num_list:
                if device_profile.comm_grid.grid.nelement() < num_devices:
                    continue
                moe_config = model_id_to_moe_config(model_id)
                tp_size_list = [1 << i for i in range(num_devices.bit_length())]
                ep = moe_config is not None
                for tp_size in tp_size_list:
                    torch.compiler.reset()
                    parallel_config = get_parallel_config(
                        world_size=num_devices,
                        tp_size=tp_size,
                        ep=ep,
                    )
                    quant_config = get_quant_config(
                        linear_quant_type
                    )  # use W8A8 by default
                    model = build_model(
                        model_id,
                        parallel_config,
                        quant_config,
                        num_mtp_tokens=mtp,
                        compile=compile,
                    )
                    num_mtp_layers = (
                        model.model_config.mtp_config.num_mtp_layers
                        if model.model_config.mtp_config is not None
                        else 0
                    )
                    if (
                        model.model_config.mla_config is None
                        and model.text_config.num_key_value_heads % tp_size != 0
                    ):
                        continue
                    for slo_limit in slo_limits:
                        latency, concurrency, breakdown, err_msg = find_best_throughput(
                            model,
                            device_profile,
                            input_length,
                            output_length,
                            slo_limit,
                            is_decode,
                        )
                        TPS = concurrency / latency if latency != 0 else 0
                        if not is_decode:
                            TPS *= input_length
                        total = sum(breakdown.values())
                        if total == 0:
                            continue
                        percentage_breakdown = [
                            f"{value * 100 / total:.2f}" for value in breakdown.values()
                        ]
                        print(
                            f"{device_profile.name}, {num_devices}, {input_length}, {output_length}, {model_id}, "
                            f"{linear_quant_type.name}, {tp_size}, {ep}, {num_mtp_layers > 0}, {slo_limit * 1000:.3f}, "
                            f"{concurrency}, {latency * 1000:.3f}, {TPS:.1f}, "
                            f"{TPS / num_devices:.1f}, {','.join(percentage_breakdown)}, {err_msg}",
                            flush=True,
                        )
