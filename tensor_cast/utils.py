import argparse
import fnmatch
import logging
import re
from typing import List, Optional

import torch
from transformers.utils.quantization_config import (
    CompressedTensorsConfig,
    FineGrainedFP8Config,
    QuantizationConfigMixin,
)

# placeholder for FP8, don't hard-code specific fp8 format
DTYPE_FP8 = torch.float8_e5m2
# use int4 placeholder for FP4
DTYPE_FP4 = torch.int4

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}


def set_log_level(level="info"):
    if level.lower() in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS.get(level.lower()))
    else:
        logger.warning("log level %r not found, set to info", level)


def set_logger(logger_: logging.Logger):
    logger_.propagate = False
    logger_.setLevel(logging.INFO)
    if not logger_.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger_.addHandler(console_handler)


logger = logging.getLogger("msmodeling_logger")
set_logger(logger)


def check_positive_integer(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid integer value: %r", value) from None
    if value <= 0:
        raise argparse.ArgumentTypeError("%r is not a positive integer", value)
    if value > 1e6:
        raise argparse.ArgumentTypeError("%r is too large", value)
    return value


def register_tensor_cast_op(name, mutates_args=(), **kwargs):
    """
    Register tensor_cast custom op with `name` under tensor_cast namespace.
    We only support meta tensor in the tensor_cast ops so the fake implementation
    is the same as the normal implementation.
    """

    def decorator(func):
        custom_op = torch.library.custom_op(
            f"tensor_cast::{name}", mutates_args=mutates_args, **kwargs
        )(func)
        custom_op.register_fake(func)
        return func

    return decorator


def exact_division(numerator, denominator):
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )
    return numerator // denominator


def pattern_match(name: str, pattern_list: List[Optional[str]]) -> bool:
    """
    three ways to match:fnmatch/re/real_name
    example of names:
    # ['lm_head', 're:.*self_attn.*', 're:.*shared_experts.*', 're:.*mlp\\.(gate|up|gate_up|down)_proj.*']
    # ["gate","e_score_correction_bias","lm_head"]
    """
    matched = False
    if not pattern_list:
        return matched
    for pattern in pattern_list:
        if pattern.startswith("re:"):
            pattern = pattern.replace("re:", "")
            matched = bool(re.match(pattern, name))
        elif pattern in name:
            matched = True
        else:
            matched = fnmatch.fnmatch(name, pattern)
        if matched:
            break
    return matched


def get_modules_to_not_convert(
    quant_config: QuantizationConfigMixin,
) -> List[Optional[str]]:
    modules_to_not_convert = []
    if isinstance(quant_config, FineGrainedFP8Config):
        modules_to_not_convert = quant_config.modules_to_not_convert
    elif isinstance(quant_config, CompressedTensorsConfig):
        modules_to_not_convert = quant_config.quantization_config.ignore
    return modules_to_not_convert


_str_to_dtype = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def str_to_dtype(string: str) -> torch.dtype:
    res = _str_to_dtype.get(string)
    if res is None:
        raise ValueError(f"Unsupported type for model: {string}")
    return res
