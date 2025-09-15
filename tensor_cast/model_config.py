import dataclasses
from enum import auto, Enum
from typing import Dict, Optional, Type

import torch


class LinearQuantType(Enum):
    W8A16 = auto()  # Weight in int8, activation in bfloat16 or half
    W8A8 = auto()  # Weight in int8, activation in int8
    W4A8 = auto()  # Weight in int4, activation in int8


class QuantGranularity(Enum):
    PER_TENSOR = auto()  # use a single quant param for the entire tensor
    PER_SAMPLE = (
        auto()
    )  # use quant param per sample in the batch (e.g. per-token for LLM)


class QuantScheme(Enum):
    SYMMETRIC = auto()
    ASYMMETRIC = auto()


@dataclasses.dataclass
class LinearQuantConfig:
    """
    Quantization configuration for PyTorch Linear op.

    The shape of the scale/offset decides the granularity, i.e.
    per-tensor, per-channel or per-group.

    For symmetric quantization, the offset tensor is None.

    For dynamic quantization, the activation scale/offset is None.
    """

    # weight config
    weight_scale: torch.Tensor
    weight_offset: Optional[torch.Tensor] = None
    weight_transposed: bool = False
    """
    Weight's shape is always in (N, K) following PyTorch's Linear semantics.
    This field sets the contiguous layout: True: (N, K) contiguous, False: (K, N) contiguous
    """
    weight_int4_pack_dim: int = 1
    """The dim to pack two int4 into an int8"""

    quant_type: LinearQuantType = LinearQuantType.W8A16

    # activation config
    dynamic_quant_dtype: torch.dtype = torch.int8
    dynamic_quant_granularity: QuantGranularity = QuantGranularity.PER_TENSOR
    dynamic_quant_scheme: QuantScheme = QuantScheme.SYMMETRIC
    activation_scale: Optional[torch.Tensor] = None
    activation_offset: Optional[torch.Tensor] = None

    # output config
    out_dtype: Optional[torch.dtype] = None


@dataclasses.dataclass
class AttentionQuantConfig:
    """
    Quantization configuration for an attention layer, specifying
    how KV cache is quantized and how the intermediate activation
    tensors are quantized and computed for attention scoring, normalization
    and aggregation.

    For a normal attention implementation, we would have something like below,
    where Q and KV cache are quantized and quant dtype of Q and attention prob
    is aligned with that of KV:

    `out = dequant(quant(softmax(dequant(quant(Q) @ K^T)), attention_prob_scale/offset) @ V)`

    TODO: support dynamic quant of query, kv, attention_prob?
    TODO: support different quant dtype of Q and attention_prob from KV
    TODO: support int4 quant
    """

    kv_scale: torch.Tensor
    query_scale: torch.Tensor
    attention_prob_scale: torch.Tensor

    query_offset: Optional[torch.Tensor] = None
    kv_offset: Optional[torch.Tensor] = None
    attention_prob_offset: Optional[torch.Tensor] = None


@dataclasses.dataclass
class QuantConfig:
    linear_configs: Dict[str, LinearQuantConfig] = dataclasses.field(
        default_factory=dict
    )
    """Per-layer configs: full module path -> LinearQuantConfig"""

    attention_configs: Dict[int, AttentionQuantConfig] = dataclasses.field(
        default_factory=dict
    )
    """Per-layer configs: attn_layer_id -> AttentionQuantConfig"""


@dataclasses.dataclass
class ParallelConfig:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    tensor_parallel_size: int = 1
    data_parallel_size: Optional[int] = None
    pipeline_parallel_size: int = 1
    mlp_tensor_parallel_size: Optional[int] = None
    mlp_data_parallel_size: Optional[int] = None
    lmhead_parallel: bool = False
    lmhead_tensor_parallel_size: Optional[int] = None
    lmhead_data_parallel_size: Optional[int] = None
    embedding_parallel: bool = False

    def has_attn_tp(self) -> bool:
        return self.tensor_parallel_size > 1

    def has_mlp_tp(self) -> bool:
        return self.mlp_tensor_parallel_size > 1

    def has_lmhead_tp(self) -> bool:
        return self.lmhead_tensor_parallel_size > 1

    def __post_init__(self) -> None:
        if self.data_parallel_size is None:
            self.data_parallel_size = (
                self.world_size
                // self.tensor_parallel_size
                // self.pipeline_parallel_size
            )

        if (
            self.tensor_parallel_size
            * self.data_parallel_size
            * self.pipeline_parallel_size
            != self.world_size
        ):
            raise ValueError(
                f"tensor_parallel_size ({self.tensor_parallel_size}) * "
                f"data_parallel_size ({self.data_parallel_size}) * "
                f"pipeline_parallel_size ({self.pipeline_parallel_size}) "
                f"must equal world_size ({self.world_size})"
            )

        if self.mlp_tensor_parallel_size is None:
            self.mlp_tensor_parallel_size = self.tensor_parallel_size
        if self.mlp_data_parallel_size is None:
            self.mlp_data_parallel_size = self.data_parallel_size
        if (
            self.mlp_tensor_parallel_size
            * self.mlp_data_parallel_size
            * self.pipeline_parallel_size
            != self.world_size
        ):
            raise ValueError(
                f"mlp_tensor_parallel_size ({self.mlp_tensor_parallel_size}) * "
                f"mlp_data_parallel_size ({self.mlp_data_parallel_size}) * "
                f"pipeline_parallel_size ({self.pipeline_parallel_size}) "
                f"must equal world_size ({self.world_size})"
            )

        if not self.lmhead_parallel:
            self.lmhead_tensor_parallel_size = 1
            self.lmhead_data_parallel_size = self.world_size
        else:
            if self.lmhead_tensor_parallel_size is None:
                self.lmhead_tensor_parallel_size = self.tensor_parallel_size
            if self.lmhead_data_parallel_size is None:
                self.lmhead_data_parallel_size = self.data_parallel_size
            if (
                self.lmhead_tensor_parallel_size
                * self.lmhead_data_parallel_size
                * self.pipeline_parallel_size
                != self.world_size
            ):
                raise ValueError(
                    f"lmhead_tensor_parallel_size ({self.lmhead_tensor_parallel_size}) * "
                    f"lmhead_data_parallel_size ({self.lmhead_data_parallel_size}) * "
                    f"pipeline_parallel_size ({self.pipeline_parallel_size}) "
                    f"must equal world_size ({self.world_size})"
                )


@dataclasses.dataclass(frozen=True)
class MoEFieldNames:
    gate: str = "gate"
    experts: str = "experts"
    shared_experts: Optional[str] = "shared_experts"
    top_k: Optional[str] = "top_k"
    norm_topk_prob: Optional[str] = "norm_topk_prob"


@dataclasses.dataclass
class MoEConfig:
    module_name: str
    fused_moe_cls: Optional[Type["FusedMoEBase"]] = None  # noqa: F821
    field_names: MoEFieldNames = MoEFieldNames()
    gate_returns_raw_logits: bool = False
    """whether the gate module returns raw logits or (topk_indices, topk_weights) tuple"""
    # TODO: add expert-parallel configuration


@dataclasses.dataclass(frozen=True)
class MlaFieldNames:
    config: str = "config"
    layer_idx: str = "layer_idx"
    num_heads: str = "num_heads"
    q_lora_rank: str = "q_lora_rank"
    qk_nope_head_dim: str = "qk_nope_head_dim"
    qk_rope_head_dim: str = "qk_rope_head_dim"
    qk_head_dim: str = "qk_head_dim"
    kv_lora_rank: str = "kv_lora_rank"
    v_head_dim: str = "v_head_dim"
    q_proj: Optional[str] = "q_proj"
    q_a_proj: Optional[str] = "q_a_proj"
    q_b_proj: Optional[str] = "q_b_proj"
    kv_a_proj_with_mqa: str = "kv_a_proj_with_mqa"
    kv_b_proj: str = "kv_b_proj"
    o_proj: str = "o_proj"
    q_a_layernorm: Optional[str] = "q_a_layernorm"
    kv_a_layernorm: str = "kv_a_layernorm"

    def __post_init__(self):
        if self.q_proj is None and (
            self.q_a_proj is None or self.q_b_proj is None or self.q_a_layernorm is None
        ):
            raise ValueError(
                "Either q_proj or all of q_a_proj/q_b_proj/q_a_layernorm must be specified"
            )


@dataclasses.dataclass
class MlaConfig:
    module_name: str
    mla_cls: Optional[Type["MultiheadLatentAttentionBase"]] = None  # noqa: F821
    field_names: MlaFieldNames = MlaFieldNames()


@dataclasses.dataclass
class MtpConfig:
    num_mtp_layers: int
    mtp_block_module_name: str


@dataclasses.dataclass
class ModelConfig:
    parallel_config: ParallelConfig
    quant_config: QuantConfig
    dtype: torch.dtype = torch.half
    cache_rotary_embedding: bool = True
    moe_config: Optional[MoEConfig] = None
    mla_config: Optional[MlaConfig] = None
    mtp_config: Optional[MtpConfig] = None
    attention_cls: Optional[Type["AttentionBase"]] = None  # noqa: F821
    quant_linear_cls: Optional[Type["QuantLinearBase"]] = None  # noqa: F821
    trust_remote_code: bool = True
    hf_config_json: Optional[str] = None
    """load transformer configuration from a local json file. When this is specified,
    `disable_auto_map` is assumed to be True."""
    disable_auto_map: Optional[bool] = None
    """set it to True if we want to use a local model definition, not
    loading it from remote. Useful when the local transformer model is preferred."""
    enable_lmhead: Optional[bool] = None
    num_hidden_layers_override: Optional[int] = None
    """Override hf_config.num_hidden_layers, useful for speeding up sanity tests
    with small overrides for very large models."""
