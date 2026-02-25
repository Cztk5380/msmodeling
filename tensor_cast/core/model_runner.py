# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# _*_coding:utf-8_*_
"""
ModelRuner
"""

from __future__ import annotations

import logging

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import torch

from ..device import DeviceProfile
from ..layers.sampler import Sampler
from ..performance_model.analytic import AnalyticPerformanceModel
from ..performance_model.memory_tracker import MemoryTracker
from ..performance_model.utils import bytes_of_tensor
from ..runtime import Runtime
from .input_generator import (
    generate_inputs_varlen,
    get_inputs_num_bytes,
    get_kv_cache_info,
    RequestInfo,
)
from .model_builder import build_model

if TYPE_CHECKING:
    from .user_config import UserInputConfig


logger = logging.getLogger(__name__)


class ModelRunner:
    """
    corresponding to one data-parallel partition ('dp_rank')
    """

    def __init__(self, user_input: UserInputConfig):
        self.user_input = user_input

        # ---------- 1. init device profile and performance_model ----------
        if user_input.device not in DeviceProfile.all_device_profiles:
            logger.error(
                "Unsupported device: %s. Available devices: %s",
                user_input.device,
                list(DeviceProfile.all_device_profiles.keys()),
            )
            raise ValueError(f"Device '{user_input.device}' not recognized.")

        logger.info("Loading device profile")
        self.device_profile = DeviceProfile.all_device_profiles[user_input.device]
        logger.debug("Device profile loaded: %s", self.device_profile)

        logger.info("Initializing performance model")
        self.perf_model = AnalyticPerformanceModel(self.device_profile)
        logger.debug("Performance model initialized: %s", self.perf_model)

        #  ---------- 2. generate default request from user config----------
        logger.info("Generating request information")
        if user_input.num_queries != 0:
            self.request_info_default = [user_input.get_request_info()]
            logger.debug("Request configured: %s", self.request_info_default)
        else:
            logger.debug("No default requests configured (num_queries = 0)")
            self.request_info_default = None

        # ---------- 3. build model ----------
        logger.info("Building model architecture")
        self.model = build_model(user_input).eval()
        logger.debug("Model built:_%s", self.model)

        # ---------- 4. static_memory ----------
        self.total_device_memory_gb = self.device_profile.memory_size_bytes / 1024**3
        self.model_weight_size_gb = self.model.weight_size / 1024**3

        logger.info("Initializing Sampler")
        self.sampler = Sampler()
        logger.debug("Sampler initialized: %s", self.sampler)

    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------
    def run_inference(
        self,
        requests: Optional[List[RequestInfo]] = None,
        generate_inputs_func: Callable = generate_inputs_varlen,
        with_sampler: bool = False,
    ) -> ModelRunnerMetrics:
        def calculate_single_card_tps(self, execution_time_s: float) -> Optional[float]:
            if not execution_time_s or execution_time_s <= 0:
                raise ValueError("execution_time_s must be positive")
            tps = (self.user_input.num_queries * self.user_input.query_len) / (
                execution_time_s * self.user_input.world_size
            )
            return tps

        data_parallel_size = self.model.model_config.parallel_config.data_parallel_size
        logger.debug("data_parallel_size: %s", data_parallel_size)

        batch_size = (
            self.user_input.num_queries + data_parallel_size - 1
        ) // data_parallel_size
        logger.debug("batch_size: %s", batch_size)

        if requests is None:
            requests = self.request_info_default
        logger.debug("requests: %s", requests)

        input_kwargs = generate_inputs_func(
            self.model,
            requests,
            block_size=self.user_input.block_size,
        )

        run_start = time.perf_counter()

        with (
            Runtime(
                self.perf_model,
                self.device_profile,
                memory_tracker=MemoryTracker(self.device_profile),
            ) as runtime,
            torch.no_grad(),
        ):
            logits = self.model.forward(**input_kwargs)
            if with_sampler:
                _ = self.sampler(logits, input_kwargs["sampling_metadata"])
        run_end = time.perf_counter()
        execution_time_s = runtime.total_execution_time_s()[self.perf_model.name]
        run_time_s = run_end - run_start

        table_result = runtime.table_averages(
            group_by_input_shapes=self.user_input.dump_input_shapes
        )

        tps_value = calculate_single_card_tps(self, execution_time_s=execution_time_s)

        peak_memory_usage_gb = runtime.memory_tracker.peak_mem_usage() / 1024**3

        kv_cache_size_gb = (
            sum(
                bytes_of_tensor(kv_cache)
                for kv_cache in input_kwargs["kv_cache_by_layers"].values()
            )
            / 1024**3
        )
        kv_cache_per_token_gb = input_kwargs["kv_cache_per_token"] / 1024**3
        if self.model.get_visual() and input_kwargs.get("pixel_values") is None:
            # If there is no image input, the visual part does not participate
            # in the calculation and needs to be removed
            visual_weight_size_gb = (
                self.model.get_weight_size_nested([self.model.get_visual()]) / 1024**3
            )
            self.model_weight_size_gb = (
                self.model_weight_size_gb - visual_weight_size_gb
            )

        model_activation_size_gb = (
            peak_memory_usage_gb - kv_cache_size_gb - self.model_weight_size_gb
        )
        device_memory_available_gb = (
            self.total_device_memory_gb
            - peak_memory_usage_gb
            - self.user_input.reserved_memory_gb
        )

        if self.user_input.chrome_trace:
            runtime.export_chrome_trace(self.user_input.chrome_trace)

        return ModelRunnerMetrics(
            total_device_memory_gb=self.total_device_memory_gb,
            model_weight_size_gb=self.model_weight_size_gb,
            peak_memory_usage_gb=peak_memory_usage_gb,
            kv_cache_size_gb=kv_cache_size_gb,
            kv_cache_per_token_gb=kv_cache_per_token_gb,
            model_activation_size_gb=model_activation_size_gb,
            reserved_memory_gb=self.user_input.reserved_memory_gb,
            device_memory_available_gb=device_memory_available_gb,
            single_card_tps=tps_value,
            execution_time_s=execution_time_s,
            run_time_s=run_time_s,
            batch_size=batch_size,
            table_result=table_result,
            breakdowns=runtime.get_breakdowns(),
        )

    def get_inputs_num_bytes(self, requests: List[Request]) -> int:  # noqa: F821
        return get_inputs_num_bytes(self.model, requests, self.user_input.block_size)

    def get_kv_cache_num_bytes(self, num_tokens: int) -> int:
        return get_kv_cache_info(self.model, 1, 1) * num_tokens


@dataclass
class ModelRunnerMetrics:
    total_device_memory_gb: float
    model_weight_size_gb: float
    peak_memory_usage_gb: float
    kv_cache_size_gb: float
    kv_cache_per_token_gb: float
    model_activation_size_gb: float
    reserved_memory_gb: float
    device_memory_available_gb: float
    single_card_tps: float
    execution_time_s: float
    run_time_s: float
    batch_size: int
    table_result: str = ""
    breakdowns: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def print_info(self):
        print(f"Number of Queries per DP rank: {self.batch_size}")
        print(f"Model compilation and execution time: {self.run_time_s:.3f} s")
        print(self.table_result)
        print(f"TPS/Device: {self.single_card_tps:.4g} token/s")

        print(f"Total device memory: {self.total_device_memory_gb:.3f} GB")
        print(f"  Model weight size: {self.model_weight_size_gb:.3f} GB")
        print(f"  KV cache: {self.kv_cache_size_gb:.3f} GB")
        print(f"  Model activation size: {self.model_activation_size_gb:.3f} GB")
        print(f"  Reserved memory: {self.reserved_memory_gb:.3f} GB")
        print(f"  Memory available: {self.device_memory_available_gb:.3f} GB")

        print("Stats breakdowns:")
        for breakdown_name, breakdown in self.breakdowns.items():
            total = sum(breakdown.values())
            if total == 0:
                continue
            formatted = ", ".join(
                f"{key}: {val * 100 / total:.2f}" for key, val in breakdown.items()
            )
            print(f"  {breakdown_name}: {formatted}")
