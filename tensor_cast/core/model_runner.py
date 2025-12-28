# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#!/usr/bin/env python
# _*_coding:utf-8_*_
"""
ModelRuner v2
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import torch

from ..device import DeviceProfile
from ..layers.sampler import Sampler
from ..performance_model.analytic import AnalyticPerformanceModel
from ..performance_model.memory_tracker import MemoryTracker
from ..performance_model.utils import bytes_of_tensor
from ..runtime import Runtime
from .utils import (
    build_model,
    generate_inputs_varlen,
    get_inputs_num_bytes,
    get_kv_cache_info,
    ModelRunnerMetrics,
    RequestInfo,
    UserInputConfig,
)


class ModelRunner:
    """
    corresponding to one data-parallel partition ('dp_rank')
    """

    def __init__(self, user_input: UserInputConfig):
        self.user_input = user_input
        # ---------- 1. init device profile and performance_model ----------
        print("Initializing model on 'meta' device...")
        if user_input.device not in DeviceProfile.all_device_profiles:
            raise ValueError(f"Device '{user_input.device}' not recognized.")
        self.device_profile = DeviceProfile.all_device_profiles[user_input.device]
        self.perf_model = AnalyticPerformanceModel(self.device_profile)

        #  ---------- 2. generate default request from user config----------
        self.request_info_default = [user_input.get_request_info()]

        # ---------- 3. build model ----------
        self.model = build_model(user_input).eval()

        # ---------- 4. static_memory ----------
        self.total_device_memory_gb = self.device_profile.memory_size_bytes / 1024**3
        self.model_weight_size_gb = self.model.weight_size / 1024**3

        self.sampler = Sampler()

    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------
    def run_inference(
        self,
        requests: Optional[List[RequestInfo]] = None,
        generate_inputs_func: Callable = generate_inputs_varlen,
        with_sampler: bool = False,
    ) -> ModelRunnerMetrics:
        batch_size = (
            self.user_input.num_queries
            + self.model.model_config.parallel_config.data_parallel_size
            - 1
        ) // self.model.model_config.parallel_config.data_parallel_size
        print(f"Number of Queries per DP rank: {batch_size}")

        print("Preparing dummy input tensors...")
        if requests is None:
            requests = self.request_info_default
        input_kwargs = generate_inputs_func(
            self.model,
            requests,
            block_size=self.user_input.block_size,
        )

        print("Running simulated inference...")
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
        print(f"Model compilation and execution time: {run_end - run_start}s")
        table_result = runtime.table_averages(
            group_by_input_shapes=self.user_input.dump_input_shapes
        )
        print(table_result)
        peak_memory_usage_gb = runtime.memory_tracker.peak_mem_usage() / 1024**3

        kv_cache_size_gb = (
            sum(
                bytes_of_tensor(kv_cache)
                for kv_cache in input_kwargs["kv_cache_by_layers"].values()
            )
            / 1024**3
        )
        kv_cache_per_token_gb = input_kwargs["kv_cache_per_token"] / 1024**3
        model_activation_size_gb = (
            peak_memory_usage_gb - kv_cache_size_gb - self.model_weight_size_gb
        )
        device_memory_available_gb = (
            self.total_device_memory_gb
            - peak_memory_usage_gb
            - self.user_input.reserved_memory_gb
        )

        print(f"Total device memory: {self.total_device_memory_gb:.3f} GB")
        print(f"  Model weight size: {self.model_weight_size_gb:.3f} GB")
        print(f"  KV cache: {kv_cache_size_gb:.3f} GB")
        print(f"  Model activation size: {model_activation_size_gb:.3f} GB")
        print(f"  Reserved memory: {self.user_input.reserved_memory_gb} GB")
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
            execution_time_s=execution_time_s,
            table_result=table_result,
            breakdowns=runtime.get_breakdowns(),
        )

    def get_inputs_num_bytes(self, requests: List[Request]) -> int:  # noqa: F821
        return get_inputs_num_bytes(self.model, requests, self.user_input.block_size)

    def get_kv_cache_num_bytes(self, num_tokens: int) -> int:
        return get_kv_cache_info(self.model, 1, 1) * num_tokens
