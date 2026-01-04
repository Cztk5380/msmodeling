#!/usr/bin/env python
# _*_coding:utf-8_*_
"""
user_config
"""

from dataclasses import dataclass, field, fields
from typing import List, Optional

from ..core.input_generator import RequestInfo
from ..core.quantization.config import create_quant_config
from ..core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from ..device import DeviceProfile
from ..model_config import ParallelConfig, QuantConfig, RemoteSource


@dataclass
class UserInputConfig:
    device: str = "TEST_DEVICE"
    model_id: str = ""
    num_queries: int = 0
    query_len: int = 0
    context_length: int = 0
    do_compile: bool = False
    allow_graph_break: bool = False
    dump_input_shapes: bool = False
    chrome_trace: Optional[str] = None
    graph_log_url: Optional[str] = None
    log_level: Optional[str] = None
    quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.W8A8_DYNAMIC
    quantize_lmhead: bool = False
    mxfp4_group_size: int = 32
    quantize_attention_action: QuantizeAttentionAction = (
        QuantizeAttentionAction.DISABLED
    )
    decode: bool = False
    num_mtp_tokens: int = 0
    mtp_acceptance_rate: List[float] = field(
        default_factory=lambda: [0.9, 0.6, 0.4, 0.2]
    )
    num_hidden_layers_override: int = 0
    disable_repetition: bool = False
    reserved_memory_gb: float = 0
    world_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    dp_size: Optional[int] = None
    o_proj_tp_size: Optional[int] = None
    o_proj_dp_size: Optional[int] = None
    mlp_tp_size: Optional[int] = None
    mlp_dp_size: Optional[int] = None
    lmhead_tp_size: Optional[int] = None
    lmhead_dp_size: Optional[int] = None
    ep: bool = False
    word_embedding_tp: bool = False
    enable_redundant_experts: bool = False
    enable_external_shared_experts: bool = False
    host_external_shared_experts: bool = False
    block_size: int = 128
    remote_source: str = RemoteSource.huggingface

    def __post_init__(self):
        self._validate_device()
        self._validate_quantize_action()

    def _validate_device(self):
        if self.device not in DeviceProfile.all_device_profiles:
            raise ValueError(f"Device '{self.device}' not recognized.")

    def _validate_quantize_action(self):
        if self.quantize_linear_action != QuantizeLinearAction.DISABLED:
            print(
                f"Quantization Linear: {self.quantize_linear_action}, quantize LM Head: {self.quantize_lmhead}"
            )
            if self.quantize_linear_action == QuantizeLinearAction.MXFP4:
                print(f"  MXFP4 group size: {self.mxfp4_group_size}")
        else:
            print("Quantization Linear: Disabled")
        if self.quantize_attention_action != QuantizeAttentionAction.DISABLED:
            print(f"Quantization Attention: {self.quantize_attention_action}")
        else:
            print("Quantization Attention: Disabled")

    def _print_info(self):
        print("--- Configuration ---")
        print(f"Device: {self.device}")
        print(f"Model ID: {self.model_id}")
        print(f"Number of Queries: {self.num_queries}")
        print(f"Input Length (per query): {self.query_len}")
        print(f"Context Length (per query): {self.context_length}")

        print(f"Enable repetition: {not self.disable_repetition}")
        if self.num_mtp_tokens > 0:
            print(f"Number of MTP layers: {self.num_mtp_tokens}")
        print(f"Use torch.compile: {self.do_compile}")
        if self.do_compile:
            print(f"  allow graph break: {self.allow_graph_break}")
        print(f"Group table averages by input shapes: {self.dump_input_shapes}")
        if self.chrome_trace:
            print(f"Chrome trace output file: {self.chrome_trace}")
        print("---------------------\n")

    def get_parallel_config(self) -> ParallelConfig:
        return ParallelConfig(
            world_size=self.world_size,
            tensor_parallel_size=self.tp_size,
            data_parallel_size=self.dp_size,
            o_proj_tensor_parallel_size=self.o_proj_tp_size,
            o_proj_data_parallel_size=self.o_proj_dp_size,
            mlp_tensor_parallel_size=self.mlp_tp_size,
            mlp_data_parallel_size=self.mlp_dp_size,
            lmhead_tensor_parallel_size=self.lmhead_tp_size,
            lmhead_data_parallel_size=self.lmhead_dp_size,
            expert_parallel=self.ep,
            embedding_parallel=self.word_embedding_tp,
            pipeline_parallel_size=self.pp_size,
        )

    def get_quant_config(self) -> QuantConfig:
        if (
            self.quantize_linear_action == QuantizeLinearAction.DISABLED
            and self.quantize_attention_action == QuantizeAttentionAction.DISABLED
        ):
            return QuantConfig()
        extra_kwargs = {}
        if self.quantize_linear_action == QuantizeLinearAction.MXFP4:
            from ..quantize_utils import QuantGranularity

            extra_kwargs.update(
                weight_group_size=self.mxfp4_group_size,
                weight_quant_granularity=QuantGranularity.PER_GROUP,
            )
        return create_quant_config(
            self.quantize_linear_action,
            quantize_lmhead=self.quantize_lmhead,
            quantize_attention_action=self.quantize_attention_action,
            **extra_kwargs,
        )

    def get_request_info(self) -> RequestInfo:
        return RequestInfo(
            query_len=self.query_len,
            seq_len=self.context_length + self.query_len,
            concurrency=self.num_queries,
        )

    @classmethod
    def from_args(cls, args) -> "UserInputConfig":
        # get all names of cls
        field_names = {_field.name for _field in fields(cls)}

        # Extract only the fields that exist in the cls from args.

        # Handle the special case where the input arguments differ
        # from the command-line arguments by implementing backward compatibility first.
        # input_key:target_key
        special_input_key_map = {
            "compile": "do_compile",
            "compile_allow_graph_break": "allow_graph_break",
            "query_length": "query_len",
            "num_devices": "world_size",
        }
        filtered_kwargs = {}
        for field_name, field_value in vars(args).items():
            if field_name in special_input_key_map:
                filtered_kwargs[special_input_key_map[field_name]] = field_value
            elif field_name in field_names:
                filtered_kwargs[field_name] = field_value
        return cls(**filtered_kwargs)
