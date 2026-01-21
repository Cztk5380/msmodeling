import argparse
import logging
import time
from typing import Optional

import torch

from ..core.quantization.config import create_quant_config
from ..core.quantization.datatypes import QuantizeLinearAction
from ..device import DeviceProfile
from .. import device_profiles # noqa: F401
from ..diffusers.diffusers_attention import set_sp_group, use_custom_sdpa
from ..diffusers.diffusers_model import build_diffusers_transformer_model
from ..diffusers.diffusers_utils import (
    get_ulysses_split_dim,
    model_class_to_input,
    model_class_to_vae_stride,
)
from ..model_config import ParallelConfig, QuantConfig
from ..parallel_group import ParallelGroup
from ..performance_model.analytic import AnalyticPerformanceModel
from ..performance_model.memory_tracker import MemoryTracker
from ..quantize_utils import QuantGranularity
from ..runtime import Runtime
from ..utils import str_to_dtype
from .utils import check_positive_integer


def generate_diffusers_inputs(
    batch_size, height, width, frame_num, seq_lens, model_config
):
    kwargs = {
        "hidden_states": generate_diffusers_pixel_input(
            batch_size, height, width, frame_num, model_config
        ),
        "encoder_hidden_states": generate_diffusers_text_input(
            batch_size, seq_lens, model_config
        ),
        "timestep": generate_diffusers_timestamp_input(model_config),
    }
    extra_args = generate_extra_input(batch_size, seq_lens, model_config)
    kwargs.update(extra_args)
    return kwargs


def generate_diffusers_pixel_input(batch_size, height, width, frame_num, model_config):
    vae_stride = model_class_to_vae_stride(
        model_config.transformer_config.model_config.get("_class_name")
    )
    channels = model_config.transformer_config.model_config.get("in_channels")
    size = [
        batch_size,
        channels,
        (frame_num - 1) // vae_stride[0] + 1,
        height // vae_stride[1],
        width // vae_stride[1],
    ]

    noise = torch.zeros(
        size=size,
        device=torch.device("meta"),
        dtype=model_config.transformer_config.dtype,
    )

    return noise


def generate_diffusers_text_input(batch_size, seq_lens, model_config):
    hidden_size = model_config.transformer_config.model_config.get("text_dim")  # Wan
    hidden_size = hidden_size or model_config.transformer_config.model_config.get(
        "text_embed_dim"
    )  # Hunyuan
    if hidden_size is None:
        raise ValueError("Get hidden_size from config failed.")
    size = [batch_size, seq_lens, hidden_size]
    encoder_hidden_states = torch.zeros(
        size=size,
        device=torch.device("meta"),
        dtype=model_config.transformer_config.dtype,
    )
    return encoder_hidden_states


def generate_extra_input(batch_size, seq_lens, model_config):
    res = {}

    if (
        model_config.transformer_config.model_config.get("pooled_projection_dim")
        is not None
    ):
        pooled_projections = torch.zeros(
            [
                batch_size,
                model_config.transformer_config.model_config.get(
                    "pooled_projection_dim"
                ),
            ],
            device=torch.device("meta"),
            dtype=model_config.transformer_config.dtype,
        )
        res["pooled_projections"] = pooled_projections

    if model_config.transformer_config.model_config.get("guidance_embeds"):
        guidance = torch.zeros(
            [1],
            device=torch.device("meta"),
            dtype=model_config.transformer_config.dtype,
        )
        res["guidance"] = guidance

    res.update(
        model_class_to_input(
            model_config.transformer_config.model_config.get("_class_name")
        )(
            batch_size=batch_size,
            seq_lens=seq_lens,
            dtype=model_config.transformer_config.dtype,
            **model_config.transformer_config.model_config,
        )
    )

    return res


def generate_diffusers_timestamp_input(model_config):
    return torch.zeros(
        [1], device=torch.device("meta"), dtype=model_config.transformer_config.dtype
    )


def process_input(input_kwargs, model_config):
    ulysses_size = model_config.transformer_config.parallel_config.ulysses_size
    rank = model_config.transformer_config.parallel_config.rank
    if ulysses_size == 1:
        return input_kwargs, None

    hidden_states = input_kwargs.get("hidden_states")
    split_dim = get_ulysses_split_dim(hidden_states, ulysses_size)

    hidden_states = hidden_states.chunk(ulysses_size, dim=split_dim)
    hidden_states = hidden_states[rank]
    input_kwargs["hidden_states"] = hidden_states

    return input_kwargs, split_dim


def run_inference(
    device: str,
    model_id: str,
    batch_size: int,
    seq_len: int,
    chrome_trace: Optional[str] = None,
    height: int = 832,
    width: int = 400,
    frame_num: int = 81,
    sample_step: int = 50,
    profiler: bool = False,
    dtype: str = "float16",
    quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.W8A8_DYNAMIC,
    mxfp4_group_size: int = 32,
    use_cfg: bool = False,
    world_size: int = 1,
    ulysses_size: int = 1,
    cfg_parallel: bool = False,
):
    if device not in DeviceProfile.all_device_profiles:
        raise ValueError(f"Device '{device}' not recognized.")
    device_profile = DeviceProfile.all_device_profiles[device]
    perf_model = AnalyticPerformanceModel(device_profile)

    parallel_config = ParallelConfig(
        world_size=world_size,
        ulysses_size=ulysses_size,
    )
    quant_config = QuantConfig()
    if quantize_linear_action != QuantizeLinearAction.DISABLED:
        extra_kwargs = {}
        if quantize_linear_action == QuantizeLinearAction.MXFP4:
            extra_kwargs.update(
                weight_group_size=mxfp4_group_size,
                weight_quant_granularity=QuantGranularity.PER_GROUP,
            )
        quant_config = create_quant_config(
            quantize_linear_action,
            **extra_kwargs,
        )
    dtype = str_to_dtype(dtype)

    model, model_config = build_diffusers_transformer_model(
        model_id,
        parallel_config,
        quant_config,
        dtype,
    )
    if ulysses_size > 1:
        set_sp_group(model.sp_group)
    if use_cfg and cfg_parallel:
        cfg_parallel_group = ParallelGroup(
            0, [[0, 1]], world_size
        )  # cfg parallel group can only be size 2
    else:
        cfg_parallel_group = None

    print("Preparing dummy input tensors...")
    input_kwargs = generate_diffusers_inputs(
        batch_size, height, width, frame_num, seq_len, model_config
    )
    input_kwargs, split_dim = process_input(input_kwargs, model_config)

    print(input_kwargs)
    print("Running simulated inference...")
    run_start = time.perf_counter()

    with (
        Runtime(
            perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
        ) as runtime,
        torch.no_grad(),
        use_custom_sdpa(),
    ):
        for _ in range(sample_step):
            out = model.forward(**input_kwargs)
            if (
                use_cfg and not cfg_parallel
            ):  # use cfg but not use cfg parallel, do one extra forward
                _ = model.forward(**input_kwargs)
            if ulysses_size > 1:
                out = model.sp_group.all_gather(out, dim=split_dim)
            if (
                use_cfg and cfg_parallel
            ):  # use cfg and use cfg parallel, do all-gather after each step of DiT forward
                out = cfg_parallel_group.all_gather(out, dim=0)

    run_end = time.perf_counter()
    print()

    print(f"Model compilation and execution time: {run_end - run_start}s")
    result = runtime.table_averages(group_by_input_shapes=False)
    print(result)

    if profiler:
        runtime.export_chrome_trace("tensor_cast_tracer.json")


def main():
    # TODO add parallel config
    # TODO add quant config
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
        help="Model config json path.",
    )
    parser.add_argument(
        "--batch-size",
        type=check_positive_integer,
        required=True,
    )
    parser.add_argument(
        "--seq-len",
        type=check_positive_integer,
        required=True,
        help="Number of inference queries to run in a batch.",
    )
    parser.add_argument(
        "--chrome-trace",
        type=str,
        default=None,
        help="Generate chrome trace file",
    )
    parser.add_argument(
        "--height",
        type=check_positive_integer,
        default=400,
    )
    parser.add_argument(
        "--width",
        type=check_positive_integer,
        default=832,
    )
    parser.add_argument(
        "--frame-num",
        type=check_positive_integer,
        default=81,
    )
    parser.add_argument(
        "--sample-step",
        type=check_positive_integer,
        default=1,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--enable-profiler",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--quantize-linear-action",
        type=QuantizeLinearAction,
        choices=list(QuantizeLinearAction),
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        help="Quantize all linear layers in the model from choices (currently only support symmetric quant)",
    )
    parser.add_argument(
        "--use-cfg",
        action="store_true",
        default=False,
    )

    parallel_group = parser.add_argument_group("Parallel Options")
    parallel_group.add_argument(
        "--world-size",
        type=check_positive_integer,
        default=1,
        help="Number of devices.",
    )
    parallel_group.add_argument(
        "--ulysses-size",
        type=check_positive_integer,
        default=1,
        help="Ulysses size.",
    )
    parallel_group.add_argument(
        "--cfg-parallel",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if args.log_level:
        logging.basicConfig(level=args.log_level.upper())

    if args.world_size % args.ulysses_size != 0:
        raise ValueError(
            f"World size {args.world_size!r} must be divisible by ulysses size {args.ulysses_size!r}."
        )

    run_inference(
        device=args.device,
        model_id=args.model_id,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        chrome_trace=args.chrome_trace,
        height=args.height,
        width=args.width,
        frame_num=args.frame_num,
        sample_step=args.sample_step,
        profiler=args.enable_profiler,
        dtype=args.dtype,
        use_cfg=args.use_cfg,
        world_size=args.world_size,
        ulysses_size=args.ulysses_size,
        quantize_linear_action=args.quantize_linear_action,
        cfg_parallel=args.cfg_parallel,
    )


if __name__ == "__main__":
    main()
