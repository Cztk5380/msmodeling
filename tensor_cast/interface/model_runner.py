# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from tensor_cast.device import DeviceProfile
from tensor_cast.layers.sampler import Sampler
from tensor_cast.model_config import ParallelConfig, QuantConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.performance_model.utils import bytes_of_tensor
from tensor_cast.quantize_utils import QuantGranularity
from tensor_cast.runtime import Runtime
from tensor_cast.scripts.utils import (
    build_model,
    create_quant_config,
    generate_inputs_varlen,
    get_inputs_num_bytes,
    get_kv_cache_info,
    QuantizeAttentionAction,
    QuantizeLinearAction,
    RequestInfo,
)


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
    execution_time_s: float
    table_result: str = ""
    breakdowns: Dict[str, Dict[str, float]] = field(default_factory=dict)


class ModelRunner:
    """
    corresponding to one data-parallel partition ('dp_rank')
    """

    def __init__(
        self,
        *,
        device: str,
        model_id: str,
        do_compile: bool = False,
        allow_graph_break: bool = False,
        dump_input_shapes: bool = False,
        chrome_trace: Optional[str] = None,
        quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.W8A8_DYNAMIC,
        quantize_lmhead: bool = False,
        mxfp4_group_size: int = 32,
        quantize_attention_action: QuantizeAttentionAction = QuantizeAttentionAction.DISABLED,
        num_mtp_tokens: int = 0,
        num_hidden_layers_override: int = 0,
        world_size: int = 1,
        tp_size: int = 1,
        dp_size: Optional[int] = None,
        mlp_tp_size: Optional[int] = None,
        mlp_dp_size: Optional[int] = None,
        lmhead_tp_size: Optional[int] = None,
        lmhead_dp_size: Optional[int] = None,
        ep: bool = False,
        reserved_memory_gb: float = 0.0,
        block_size: int = 128,
    ):
        # ---------- 1. device ----------
        if device not in DeviceProfile.all_device_profiles:
            raise ValueError(f"Device '{device}' not recognized.")
        self.device_profile = DeviceProfile.all_device_profiles[device]

        # ---------- 2. parallel / quant ----------
        self.parallel_config = ParallelConfig(
            world_size=world_size,
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            mlp_tensor_parallel_size=mlp_tp_size,
            mlp_data_parallel_size=mlp_dp_size,
            lmhead_tensor_parallel_size=lmhead_tp_size,
            lmhead_data_parallel_size=lmhead_dp_size,
            expert_parallel=ep,
        )
        self.quant_config = self._make_quant_config(
            quantize_linear_action,
            quantize_lmhead,
            quantize_attention_action,
            mxfp4_group_size,
        )

        # ---------- 3. build model ----------
        self.model = build_model(
            model_id,
            self.parallel_config,
            self.quant_config,
            num_mtp_tokens=num_mtp_tokens,
            compile=do_compile,
            allow_graph_break=allow_graph_break,
            enable_repetition=True,
            num_hidden_layers_override=num_hidden_layers_override,
        )

        # ---------- 4. performance / runtime ----------
        self.perf_model = AnalyticPerformanceModel(self.device_profile)
        self.do_compile = do_compile
        self.allow_graph_break = allow_graph_break
        self.dump_input_shapes = dump_input_shapes
        self.chrome_trace = chrome_trace

        self.total_device_memory_gb = self.device_profile.memory_size_bytes / 1024**3
        self.model_weight_size_gb = self.model.weight_size / 1024**3
        self.reserved_memory_gb = reserved_memory_gb
        self.block_size = block_size

        self.sampler = Sampler()

    # -----------------------------------------------------
    # internal helpers
    # -----------------------------------------------------
    @staticmethod
    def _make_quant_config(
        linear_action: QuantizeLinearAction,
        lmhead: bool,
        attn_action: QuantizeAttentionAction,
        group_size: int,
    ) -> QuantConfig:
        if (
            linear_action == QuantizeLinearAction.DISABLED
            and attn_action == QuantizeAttentionAction.DISABLED
        ):
            return QuantConfig()
        extra = {}
        if linear_action == QuantizeLinearAction.MXFP4:
            extra.update(
                weight_group_size=group_size,
                weight_quant_granularity=QuantGranularity.PER_GROUP,
            )
        return create_quant_config(
            linear_action,
            quantize_lmhead=lmhead,
            quantize_attention_action=attn_action,
            **extra,
        )

    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------
    def run_inference(self, requests: List[RequestInfo]) -> InferenceMetrics:  # noqa: F821
        input_kwargs = generate_inputs_varlen(
            self.model,
            requests,
            block_size=self.block_size,
        )

        with (
            Runtime(
                self.perf_model,
                self.device_profile,
                memory_tracker=MemoryTracker(self.device_profile),
            ) as runtime,
            torch.no_grad(),
        ):
            logits = self.model.forward(**input_kwargs)
            _ = self.sampler(logits, input_kwargs["sampling_metadata"])
        execution_time_s = runtime.total_execution_time_s()[self.perf_model.name]

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
            self.total_device_memory_gb - peak_memory_usage_gb - self.reserved_memory_gb
        )

        table_result = runtime.table_averages(
            group_by_input_shapes=self.dump_input_shapes
        )
        if self.chrome_trace:
            runtime.export_chrome_trace(self.chrome_trace)

        return ModelRunnerMetrics(
            total_device_memory_gb=self.total_device_memory_gb,
            model_weight_size_gb=self.model_weight_size_gb,
            peak_memory_usage_gb=peak_memory_usage_gb,
            kv_cache_size_gb=kv_cache_size_gb,
            kv_cache_per_token_gb=kv_cache_per_token_gb,
            model_activation_size_gb=model_activation_size_gb,
            reserved_memory_gb=self.reserved_memory_gb,
            device_memory_available_gb=device_memory_available_gb,
            execution_time_s=execution_time_s,
            table_result=table_result,
            breakdowns=runtime.get_breakdowns(),
        )

    def get_inputs_num_bytes(self, requests: List[Request]) -> int:  # noqa: F821
        return get_inputs_num_bytes(self.model, requests, self.block_size)

    def get_kv_cache_num_bytes(self, num_tokens: int) -> int:
        return get_kv_cache_info(self.model, 1, 1) * num_tokens
