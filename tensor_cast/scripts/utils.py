try:
    # Native in Python 3.11+
    from enum import StrEnum
except ImportError:
    # Fallback for Python 3.10
    from strenum import StrEnum
from dataclasses import dataclass
from typing import List, Optional

import torch

from .. import device_profiles  # noqa: F401
from ..compilation import get_backend
from ..device import DeviceProfile
from ..layers.attention import AttentionMetadataTensorCast, AttentionTensorCast
from ..layers.mla import MultiheadLatentAttentionTensorCast
from ..layers.quant_linear import TensorCastQuantLinear
from ..layers.sampler import SamplingMetadata
from ..model_config import (
    LinearQuantConfig,
    MlaConfig,
    ModelConfig,
    MtpConfig,
    MultiheadLatentAttentionQuantConfig,
    ParallelConfig,
    QuantConfig,
)
from ..performance_model.utils import bytes_of_tensor
from ..quantize_utils import AttentionQuantType, LinearQuantType
from ..transformers.model import TransformerModel
from ..transformers.utils import (
    AutoModelConfigLoader,
    get_attention_quant_config,
    get_mla_module_name,
    get_moe_config,
    get_mtp_block_module_name,
)
from ..utils import exact_division


class QuantizeLinearAction(StrEnum):
    DISABLED = "DISABLED"
    W8A16_STATIC = "W8A16_STATIC"
    W8A8_STATIC = "W8A8_STATIC"
    W4A8_STATIC = "W4A8_STATIC"
    W8A16_DYNAMIC = "W8A16_DYNAMIC"
    W8A8_DYNAMIC = "W8A8_DYNAMIC"
    W4A8_DYNAMIC = "W4A8_DYNAMIC"
    FP8 = "FP8"
    MXFP4 = "MXFP4"


class QuantizeAttentionAction(StrEnum):
    # TODO(jgong5): support FP8 quantization
    DISABLED = "DISABLED"
    INT8 = "INT8"


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

    batch_size: int = 1
    seq_len: int = 1

    def __post_init__(self):
        self._init_dp_size()
        self._init_batch_size_and_seq_len()
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
        print(f"Number of Queries per DP rank: {self.batch_size}")
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

    def _init_dp_size(self):
        if self.dp_size is None:
            self.dp_size = self.world_size // self.tp_size // self.pp_size

    def _init_batch_size_and_seq_len(self):
        self.batch_size = (self.num_queries + self.dp_size - 1) // self.dp_size
        self.seq_len = (
            self.context_length + self.query_len
        )  # Total sequence length for each query

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

    @classmethod
    def from_args(cls, args) -> "UserInputConfig":
        return cls(
            device=args.device,
            model_id=args.model_id,
            num_queries=args.num_queries,
            query_len=args.query_length,
            context_length=args.context_length,
            do_compile=args.compile,
            allow_graph_break=args.compile_allow_graph_break,
            dump_input_shapes=args.dump_input_shapes,
            chrome_trace=args.chrome_trace,
            graph_log_url=args.graph_log_url,
            log_level=args.log_level,
            quantize_linear_action=args.quantize_linear_action,
            quantize_lmhead=args.quantize_lmhead,
            mxfp4_group_size=args.mxfp4_group_size,
            quantize_attention_action=args.quantize_attention_action,
            decode=args.decode,
            num_mtp_tokens=args.num_mtp_tokens,
            num_hidden_layers_override=args.num_hidden_layers_override,
            disable_repetition=args.disable_repetition,
            reserved_memory_gb=args.reserved_memory_gb,
            world_size=args.world_size,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
            o_proj_tp_size=args.o_proj_tp_size,
            o_proj_dp_size=args.o_proj_dp_size,
            mlp_tp_size=args.mlp_tp_size,
            mlp_dp_size=args.mlp_dp_size,
            lmhead_tp_size=args.lmhead_tp_size,
            lmhead_dp_size=args.lmhead_dp_size,
            ep=args.ep,
            word_embedding_tp=args.word_embedding_tp,
            enable_redundant_experts=args.enable_redundant_experts,
            enable_external_shared_experts=args.enable_external_shared_experts,
            host_external_shared_experts=args.host_external_shared_experts,
        )


def get_available_memory_gb(device_profile, runtime, reserved_memory_size_gb=0):
    """
    Get available memory on the device during executing models under the runtime. It is the minimum
    available memory, not the one after model execution.

    :param device_profile: The device configuration
    :param runtime: The runtime under which the models have been executed
    :param reserved_memory_size_gb: The reserved memory size on top of the consumption of the models.
    :return: The minimum available memory during execution.
    """
    total_device_memory_gb = device_profile.memory_size_bytes / 1024**3
    peak_memory_usage_gb = runtime.memory_tracker.peak_mem_usage() / 1024**3
    device_memory_available_gb = (
        total_device_memory_gb - peak_memory_usage_gb - reserved_memory_size_gb
    )
    return device_memory_available_gb


def get_parallel_config(world_size: int, tp_size: int = 1, ep: bool = False):
    return ParallelConfig(
        world_size=world_size, tensor_parallel_size=tp_size, expert_parallel=ep
    )


def create_linear_quant_config(quantize_linear_action: QuantizeLinearAction, **kwargs):
    # TODO: support per-channel/per-group setting
    # TODO: support asymmetric quant setting

    if quantize_linear_action in ("W8A16_STATIC", "W8A16_DYNAMIC"):
        quant_type = LinearQuantType.W8A16
    elif quantize_linear_action in ("W8A8_STATIC", "W8A8_DYNAMIC"):
        quant_type = LinearQuantType.W8A8
    elif quantize_linear_action == "FP8":
        quant_type = LinearQuantType.FP8
    elif quantize_linear_action == "MXFP4":
        quant_type = LinearQuantType.MXFP4
        if "weight_group_size" not in kwargs:
            raise ValueError(
                "weight_group_size must be provided for MXFP4 quantization"
            )
    elif quantize_linear_action in ("W4A8_STATIC", "W4A8_DYNAMIC"):
        quant_type = LinearQuantType.W4A8
    else:
        raise ValueError(f"Unsupported quantization action {quantize_linear_action}")

    config_args = {
        "quant_type": quant_type,
    }

    if "weight_scale" not in kwargs and quant_type != LinearQuantType.MXFP4:
        # For MXFP4, weight_scale is created from the weight tensor during model initialization
        config_args["weight_scale"] = torch.tensor(1.0)

    if quantize_linear_action in ("W8A16_STATIC", "W8A8_STATIC", "W4A8_STATIC"):
        config_args["activation_scale"] = torch.tensor(1.0)
    config_args.update(kwargs)
    return LinearQuantConfig(**config_args)


def create_attention_quant_config(quantize_attention_action: QuantizeAttentionAction):
    if quantize_attention_action == QuantizeAttentionAction.INT8:
        # default to symmetric quant with dummy scales
        # for simplicity, we use MLA quant config for both MLA and regular attention
        return MultiheadLatentAttentionQuantConfig(
            quant_type=AttentionQuantType.INT8,
            query_scale=torch.tensor(1.0),
            kv_scale=torch.tensor(1.0),
            attention_prob_scale=torch.tensor(1.0),
            kv_projected_scale=torch.tensor(1.0),
            qk_scale=torch.tensor(1.0),
            v_scale=torch.tensor(1.0),
            out_scale=torch.tensor(1.0),
        )
    else:
        raise ValueError(f"Unsupported quantization action {quantize_attention_action}")


def create_quant_config(
    quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.DISABLED,
    quantize_lmhead: bool = False,
    quantize_attention_action: QuantizeAttentionAction = QuantizeAttentionAction.DISABLED,
    **kwargs,
):
    quant_config = QuantConfig()
    if quantize_linear_action != QuantizeLinearAction.DISABLED:
        quant_config.linear_configs["layers.*"] = create_linear_quant_config(
            quantize_linear_action, **kwargs
        )
        quant_config.linear_configs["*.layers.*"] = create_linear_quant_config(
            quantize_linear_action, **kwargs
        )
        if quantize_lmhead:
            quant_config.linear_configs["lm_head"] = create_linear_quant_config(
                quantize_linear_action, **kwargs
            )
            quant_config.linear_configs["*.lm_head"] = create_linear_quant_config(
                quantize_linear_action, **kwargs
            )
    if quantize_attention_action != QuantizeAttentionAction.DISABLED:
        # default to symmetric quant with dummy scales
        quant_config.attention_configs[-1] = create_attention_quant_config(
            quantize_attention_action
        )

    return quant_config


def build_model(
    model_id: str = "",
    parallel_config: ParallelConfig = None,
    quant_config: QuantConfig = None,
    num_mtp_tokens: int = 0,
    compile: bool = False,
    allow_graph_break: bool = True,
    enable_repetition: bool = True,
    num_hidden_layers_override: int = 0,
    enable_redundant_experts: bool = False,
    enable_external_shared_experts: bool = False,
    host_external_shared_experts: bool = False,
    user_input: UserInputConfig = None,
) -> TransformerModel:
    """
    Build a transformer model based on the given args

    :param model_id: Transformer model id.
    :param parallel_config: Parallel configuration.
    :param quant_config: Quantization configuration.
    :param num_mtp_tokens: Number of multi-token-prediction tokens.
    :param compile: Whether we want to do `torch.compile` on the model. Useful for optimal perf estimation.
    :param allow_graph_break: Allow graph break during `torch.compile`. Keep it on for better compatibility.
    :param enable_repetition: Enable TensorCast to recognize repetitive patterns in the model for better model
                              loading and execution performance.
    :return: The loaded (and possibly compiled) Transformer model.
    """

    config_resolver = ConfigResolver(
        model_id=model_id,
        user_input=user_input,
        quant_config=quant_config,
        parallel_config=parallel_config,
    )

    model_config = config_resolver.resolve()

    model = TransformerModel(model_id, model_config)
    use_full_graph = not user_input.allow_graph_break or (not allow_graph_break)
    if compile or user_input.do_compile:
        model = torch.compile(
            model, backend=get_backend(), dynamic=True, fullgraph=use_full_graph
        )
    return model


class ConfigResolver:
    def __init__(
        self,
        model_id: str = "",
        user_input: UserInputConfig = None,
        parallel_config: ParallelConfig = None,
        quant_config: QuantConfig = None,
    ):
        self.model_id = model_id or user_input.model_id
        auto_loader = AutoModelConfigLoader()
        self.hf_config = auto_loader.load_config(self.model_id)
        self.user_input = user_input

        if user_input is not None:
            quant_config = user_input.get_quant_config()
            parallel_config = user_input.get_parallel_config()
        else:
            if parallel_config is None or quant_config is None:
                raise ValueError(
                    "When the user input is None,quant_config and parallel_config can not be None"
                )

        self.model_config = ModelConfig(
            parallel_config,
            quant_config,
            attention_cls=AttentionTensorCast,
            quant_linear_cls=TensorCastQuantLinear,
        )
        self.model_config.hf_config = self.hf_config
        self.model_config.trust_remote_code = (
            not auto_loader.is_transformers_natively_supported
        )

    def resolve(self) -> ModelConfig:
        self.update_hf_config(
            enable_repetition=not self.user_input.disable_repetition,
            num_hidden_layers_override=self.user_input.num_hidden_layers_override,
        )
        self.update_moe_config(
            enable_redundant_experts=self.user_input.enable_redundant_experts,
            enable_external_shared_experts=self.user_input.enable_external_shared_experts,
            host_external_shared_experts=self.user_input.host_external_shared_experts,
        )
        self.update_mla_config()
        self.update_mtp_config(num_mtp_tokens=self.user_input.num_mtp_tokens)
        return self.model_config

    def update_moe_config(
        self,
        model_type: str = "",
        enable_redundant_experts: bool = False,
        enable_external_shared_experts: bool = False,
        host_external_shared_experts: bool = False,
    ):
        if not model_type:
            model_type = self.hf_config.model_type
        moe_config = get_moe_config(model_type)
        if moe_config is not None:
            moe_config.enable_redundant_experts = enable_redundant_experts
            moe_config.enable_external_shared_experts = enable_external_shared_experts
            moe_config.host_external_shared_experts = host_external_shared_experts
        self.model_config.moe_config = moe_config

    def update_mla_config(self, model_type: str = ""):
        if not model_type:
            model_type = self.hf_config.model_type
        mla_module_name = get_mla_module_name(model_type)
        if mla_module_name is not None:
            mla_config = MlaConfig(
                module_name=mla_module_name,
                mla_cls=MultiheadLatentAttentionTensorCast,
            )
            self.model_config.mla_config = mla_config

    def update_mtp_config(self, model_type: str = "", num_mtp_tokens: int = 0):
        if not model_type:
            model_type = self.hf_config.model_type
        mtp_block_module_name = get_mtp_block_module_name(model_type)
        if num_mtp_tokens > 0 and mtp_block_module_name:
            mtp_config = MtpConfig(
                num_mtp_layers=num_mtp_tokens,
                mtp_block_module_name=mtp_block_module_name,
            )
            self.model_config.mtp_config = mtp_config

    def update_hf_config(
        self, enable_repetition: bool = False, num_hidden_layers_override: int = 0
    ):
        self.model_config.enable_repetition = enable_repetition
        self.model_config.num_hidden_layers_override = num_hidden_layers_override


def generate_inputs(model, query_len, seq_len, concurrency, is_decode=True):
    model_config = model.model_config
    num_mtp_tokens = (
        model_config.mtp_config.num_mtp_layers if model_config.mtp_config else 0
    )
    parallel_config = model_config.parallel_config
    batch_size = (
        concurrency + parallel_config.data_parallel_size - 1
    ) // parallel_config.data_parallel_size

    max_context_length = seq_len + num_mtp_tokens + 1

    # Paged attention parameters (can be adjusted)
    block_size = 128
    num_blocks = (
        max_context_length * batch_size + block_size - 1
    ) // block_size  # Total number of blocks available in the KV cache

    # Prepare Attention Metadata for Paged Attention
    # `query_start_loc` indicates the start of each query in the concatenated input tensor.
    # Shape: [num_queries + 1] -> e.g., [0, 50, 100, 150] for 3 queries of length 50.
    query_start_loc = torch.arange(
        0, (batch_size + 1) * query_len, query_len, dtype=torch.long
    )

    # `seq_lens` is the total length (context + new tokens) for each sequence in the batch.
    seq_lens = torch.empty(batch_size, dtype=torch.long)
    seq_lens.fill_(seq_len)

    query_lens = torch.empty(batch_size, dtype=torch.long)
    query_lens.fill_(query_len)

    # `block_tables` map logical sequence blocks to physical blocks in the KV cache.
    max_num_blocks_per_seq = (seq_len + block_size - 1) // block_size

    block_table_tensor = torch.empty(
        (batch_size, max_num_blocks_per_seq), dtype=torch.long, device="meta"
    )

    slot_mapping = torch.empty(
        (batch_size * query_len,), dtype=torch.long, device="meta"
    )

    # We use padding to ensure that the number of tokens in each DP domain is divisible by tp_size.
    # This allows the data to be evenly distributed across each device if needed,
    # thereby enabling arbitrary conversion of DP domains.
    padding_tokens = 0
    if batch_size * query_len % parallel_config.tensor_parallel_size != 0:
        padding_tokens = parallel_config.tensor_parallel_size - (
            batch_size * query_len % parallel_config.tensor_parallel_size
        )

    query_start_loc[-1] = query_start_loc[-1] + padding_tokens

    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_lens=query_lens,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )

    # The total number of new tokens to be processed in this batch, concatenated.
    num_tokens = batch_size * query_len + padding_tokens
    input_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    # Initialize the KV cache structure (also on 'meta' device).
    kv_cache_by_layers = {}
    for i in range(model.num_hidden_layers):
        kvcache_dtype = model_config.dtype
        if (attention_config := get_attention_quant_config(model, i)) is not None:
            kvcache_dtype = attention_config.get_quant_dtype()

        if model_config.mla_config is not None:
            # Shape: [num_blocks, block_size, kv_lora_head_dim + qk_rope_head_dim]
            kv_cache_by_layers[i] = torch.empty(
                [
                    num_blocks,
                    block_size,
                    model.text_config.kv_lora_rank + model.text_config.qk_rope_head_dim,
                ],
                dtype=kvcache_dtype,
                device="meta",
            )
        else:
            # Shape: [2 (K/V), num_blocks, block_size, num_heads, head_dim]
            if (
                model.text_config.num_key_value_heads
                >= parallel_config.tensor_parallel_size
            ):
                kv_heads = exact_division(
                    model.text_config.num_key_value_heads,
                    parallel_config.tensor_parallel_size,
                )
            else:
                assert (
                    parallel_config.tensor_parallel_size
                    % model.text_config.num_key_value_heads
                    == 0
                )
                kv_heads = 1

            kv_cache_by_layers[i] = torch.empty(
                [
                    2,
                    num_blocks,
                    block_size,
                    kv_heads,
                    model.head_dim,
                ],
                dtype=kvcache_dtype,
                device="meta",
            )
    sampling_metadata = SamplingMetadata(
        query_start_loc=attn_meta.query_start_loc,
    )
    if is_decode:
        # do not prune logits
        sampling_metadata.selected_token_indices = None

    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_meta": attn_meta,
        "kv_cache_by_layers": kv_cache_by_layers,
        "sampling_metadata": sampling_metadata,
    }
    if model.model_id.startswith("Qwen/Qwen3-Next"):
        kwargs["cache_position"] = torch.arange(
            0, num_tokens, dtype=torch.long, device="cpu"
        )
    return kwargs


@dataclass
class RequestInfo:
    query_len: int
    seq_len: int
    is_decode: bool
    num_input_tokens: int = None
    num_output_tokens: int = None


def get_kv_cache_info(model, num_blocks, block_size):
    model_config = model.model_config
    tp_size = model_config.parallel_config.tensor_parallel_size
    kv_cache_by_layers = {}
    kv_cache_per_token = 0
    for i in range(model.num_hidden_layers):
        kvcache_dtype = model_config.dtype
        attention_config = get_attention_quant_config(model, i)
        if attention_config is not None:
            kvcache_dtype = attention_config.get_quant_dtype()

        if model_config.mla_config is not None:
            kv_cache_by_layers[i] = torch.empty(
                (
                    num_blocks,
                    block_size,
                    model.text_config.kv_lora_rank + model.text_config.qk_rope_head_dim,
                ),
                dtype=kvcache_dtype,
                device="meta",
            )
        else:
            assert model.text_config.num_key_value_heads % tp_size == 0
            kv_cache_by_layers[i] = torch.empty(
                (
                    2,
                    num_blocks,
                    block_size,
                    model.text_config.num_key_value_heads // tp_size,
                    model.head_dim,
                ),
                dtype=kvcache_dtype,
                device="meta",
            )
        kv_cache_per_token += bytes_of_tensor(kv_cache_by_layers[i]) / (
            num_blocks * block_size
        )

    return kv_cache_by_layers, kv_cache_per_token


def generate_inputs_varlen(model, requests: List[RequestInfo], block_size):
    """
    requests: List[RequestInfo], each dict represents a request, containing keys: query_len, seq_len, is_decode
    """
    model_config = model.model_config
    mtp = getattr(model_config, "mtp_config", None)
    num_mtp_tokens = mtp.num_mtp_layers if mtp else 0

    batch_size = len(requests)
    if batch_size == 0:
        return {}

    query_lens = [r.query_len for r in requests]
    seq_lens = [r.seq_len for r in requests]
    is_decode_list = [r.is_decode for r in requests]
    num_tokens = sum(query_lens)

    query_start_loc = [0]
    for ql in query_lens:
        query_start_loc.append(query_start_loc[-1] + ql)
    query_start_loc = torch.tensor(query_start_loc, dtype=torch.long)

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long)
    query_len_t = torch.tensor(query_lens, dtype=torch.long)

    num_blocks = (
        sum(seq_lens) + batch_size * (num_mtp_tokens + 1) + block_size - 1
    ) // block_size
    max_num_blocks_per_seq = (max(seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.empty(
        (batch_size, max_num_blocks_per_seq), dtype=torch.long, device="meta"
    )
    slot_mapping = torch.empty((num_tokens,), dtype=torch.long, device="meta")

    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        query_lens=query_len_t,
        seq_lens=seq_lens_t,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )

    input_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

    kv_cache_by_layers, kv_cache_per_token = get_kv_cache_info(
        model, num_blocks, block_size
    )

    sampling_meta = SamplingMetadata(query_start_loc=query_start_loc)
    selected_token_indices = []

    pos = 0
    for ql, decode in zip(query_lens, is_decode_list):
        if decode:
            selected_token_indices.extend(range(pos, pos + ql))
        else:
            selected_token_indices.append(pos + ql - 1)
        pos += ql
    sampling_meta.selected_token_indices = torch.tensor(
        selected_token_indices, dtype=torch.long, device="meta"
    )

    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_meta": attn_meta,
        "kv_cache_by_layers": kv_cache_by_layers,
        "sampling_metadata": sampling_meta,
        "kv_cache_per_token": kv_cache_per_token,
    }

    if model.model_id.startswith("Qwen/Qwen3-Next"):
        kwargs["cache_position"] = torch.arange(
            num_tokens, dtype=torch.long, device="cpu"
        )

    return kwargs


def get_inputs_num_bytes(model, requests: List[RequestInfo], block_size: int) -> int:
    """
    Get the number of bytes of the input tensors.
    """
    input_kwargs = generate_inputs_varlen(model, requests, block_size)
    inputs_num_bytes = 0
    inputs_num_bytes += bytes_of_tensor(input_kwargs["input_ids"])
    inputs_num_bytes += bytes_of_tensor(input_kwargs["position_ids"])
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].query_start_loc)
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].seq_lens)
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].query_lens)
    inputs_num_bytes += bytes_of_tensor(
        input_kwargs["attention_meta"].block_table_tensor
    )
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].slot_mapping)
    return inputs_num_bytes
