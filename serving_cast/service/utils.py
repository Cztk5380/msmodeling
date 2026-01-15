# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

import argparse
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

import torch

from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.performance_model.utils import bytes_of_tensor
from tensor_cast.runtime import Runtime


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}
LIMIT_TIME = 1e6
BYTES_TO_GB = 1024**3
MAX_ITER_NUMS = 10

COMMON_COLUMNS = [
    "model_name",
    "input_length",
    "output_length",
    "concurrency",
    "ttft",
    "tpot",
    "total_devices",
    "backend",
    "device_name",
]

AGG_COLUMNS = COMMON_COLUMNS + ["token/s", "token/s/device", "parallel", "batch_size"]


@dataclass
class DataConfig:
    input_length: int = None
    output_length: int = None
    batch_size: int = None
    ttft_limits: float = None
    tpot_limits: float = None
    max_prefill_tokens: int = None
    device_nums: int = None
    device_profile: object = None


class BackendName(Enum):
    MindIE = "MindIE"


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


def run_static(model, input_kwargs, device_profile, reserved_memory_gb=0.0):
    perf_model = AnalyticPerformanceModel(device_profile)

    run_start = time.perf_counter()
    with (
        Runtime(
            perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
        ) as runtime,
        torch.no_grad(),
    ):
        _ = model.forward(**input_kwargs)
    run_end = time.perf_counter()
    logger.debug("Model compilation and execution time: %.2f s", run_end - run_start)
    execution_time_s = runtime.total_execution_time_s()[perf_model.name]
    total_device_memory_gb = device_profile.memory_size_bytes / BYTES_TO_GB
    model_weight_size_gb = model.weight_size / BYTES_TO_GB
    peak_memory_usage_gb = runtime.memory_tracker.peak_mem_usage() / BYTES_TO_GB
    total_kv_cache_size_gb = (
        sum(
            bytes_of_tensor(kv_cache)
            for kv_cache in input_kwargs["kv_cache_by_layers"].values()
        )
        / BYTES_TO_GB
    )
    model_activation_size_gb = (
        peak_memory_usage_gb - total_kv_cache_size_gb - model_weight_size_gb
    )

    device_memory_available_gb = (
        total_device_memory_gb - peak_memory_usage_gb - reserved_memory_gb
    )

    return {
        "execution_time_s": execution_time_s * 1000,
        "device_memory_available_gb": device_memory_available_gb,
        "model_activation_size_gb": model_activation_size_gb,
    }


def check_string_valid(string: str, max_len=256):
    if len(string) > max_len:
        raise argparse.ArgumentTypeError(
            "String length exceeds %d characters: %r", max_len, string
        )
    if not re.match(r"^[a-zA-Z0-9_/.-]+$", string):
        raise argparse.ArgumentTypeError(
            "String contains invalid characters: %r", string
        )
    return string


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


def check_positive_float(value):
    if value.lower() == "inf":
        return float("inf")
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid float value: %r", value) from None
    if value <= 0:
        raise argparse.ArgumentTypeError("%r is not a positive number", value)
    return value
