import dataclasses
from enum import auto, Enum
from typing import Dict, Optional, Type

import torch


class LinearQuantType(Enum):
    W8A16 = auto()  # Weight in int8, activation in bfloat16 or half
    W8A8 = auto()  # Weight in int8, activation in int8
    W4A8 = auto()  # Weight in int4, activation in int8
    FP8 = auto()  # Weight in float8, activation in float8
    MXFP4 = auto()  # both weight and activation in MXFP4


class AttentionQuantType(Enum):
    INT8 = auto()
    # TODO(jgong5): support FP8


class QuantGranularity(Enum):
    PER_TENSOR = auto()  # use a single quant param for the entire tensor
    PER_SAMPLE = (
        auto()
    )  # use quant param per sample in the batch (e.g. per-token for LLM)
    PER_GROUP = auto()  # use quant param per channel group


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
    # The scale and offset are computed according to `weight_quant_granularity`
    # and `weight_quant_scheme` if they are not explicitly provided.
    weight_scale: Optional[torch.Tensor] = None
    weight_offset: Optional[torch.Tensor] = None
    weight_transposed: bool = False
    """
    Weight's shape is always in (N, K) following PyTorch's Linear semantics.
    This field sets the contiguous layout: True: (N, K) contiguous, False: (K, N) contiguous
    """
    weight_int4_pack_dim: int = 1
    """The dim to pack two int4 into an int8"""
    weight_group_size: Optional[int] = None
    """Group size for weight quantization along k-dim. For MXFP4, it also implies
    the channel group size for activation quantization. The shape of weight_scale
    should be aligned with this setting."""
    weight_quant_granularity: Optional[QuantGranularity] = None
    """Quantization granularity for weight. If None, it is inferred from the shape
    of weight_scale and weight_offset."""
    weight_quant_scheme: QuantScheme = QuantScheme.SYMMETRIC
    """Quantization scheme for weight. If None, it is inferred from whether
    weight_offset is None or not."""

    quant_type: LinearQuantType = LinearQuantType.W8A16

    # activation config
    dynamic_quant_granularity: Optional[QuantGranularity] = None
    dynamic_quant_scheme: QuantScheme = QuantScheme.SYMMETRIC
    activation_scale: Optional[torch.Tensor] = None
    """Scale for static quantization, None for dynamic quantization"""
    activation_offset: Optional[torch.Tensor] = None
    """Offset for static quantization, None for symmetric quantization or dynamic quantization"""

    # output config
    out_dtype: Optional[torch.dtype] = None
    """We deliberately not support output scales and only support out_dtype as
    high-precision dtype (fp16, bf16, fp32) for simplicity. Use-case for quantized
    output is not common."""

    def __post_init__(self):
        if (
            self.weight_quant_granularity is not None
            and self.dynamic_quant_granularity is None
        ):
            self.dynamic_quant_granularity = self.weight_quant_granularity
        if self.weight_scale is None and self.weight_offset is not None:
            raise ValueError(
                "weight_offset is provided but weight_scale is None, which is invalid"
            )
        if self.weight_scale is None:
            if self.weight_quant_granularity is None:
                self.weight_quant_granularity = QuantGranularity.PER_TENSOR
            if self.weight_quant_scheme is None:
                self.weight_quant_scheme = QuantScheme.SYMMETRIC
        if (
            self.weight_scale is None
            and self.weight_quant_granularity == QuantGranularity.PER_GROUP
            and self.weight_group_size is None
        ):
            raise ValueError(
                "weight_group_size must be provided when weight_quant_granularity is PER_GROUP and "
                "weight_scale is not provided"
            )
        if self.activation_scale is None:
            if self.dynamic_quant_granularity is None:
                self.dynamic_quant_granularity = QuantGranularity.PER_TENSOR
            if self.dynamic_quant_scheme is None:
                self.dynamic_quant_scheme = QuantScheme.SYMMETRIC
        # Validate FP8 configuration
        if self.quant_type == LinearQuantType.FP8:
            if (
                self.dynamic_quant_scheme is not None
                and self.dynamic_quant_scheme != QuantScheme.SYMMETRIC
            ):
                raise ValueError(
                    "FP8 quantization only supports symmetric scheme for activations"
                )
            if self.activation_scale is not None or self.activation_offset is not None:
                raise ValueError(
                    "FP8 quantization does not support static activation quantization"
                )

        # Validate MXFP4 configuration
        if self.quant_type == LinearQuantType.MXFP4:
            if self.dynamic_quant_granularity != QuantGranularity.PER_GROUP:
                raise ValueError(
                    "MXFP4 quantization only supports PER_GROUP granularity"
                )
            if (
                self.dynamic_quant_scheme is not None
                and self.dynamic_quant_scheme != QuantScheme.SYMMETRIC
            ):
                raise ValueError("MXFP4 quantization only supports symmetric scheme")
            if self.activation_scale is not None or self.activation_offset is not None:
                raise ValueError(
                    "MXFP4 quantization does not support static activation quantization"
                )


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

    `out = dequant(quant(softmax(dequant(Q @ K^T)), attention_prob_scale/offset) @ V)`

    TODO: support dynamic quant of query, kv, attention_prob?
    TODO: support different quant dtype of Q and attention_prob from KV
    TODO: support int4 quant
    """

    quant_type: AttentionQuantType = AttentionQuantType.INT8
    kv_scale: Optional[torch.Tensor] = None
    query_scale: Optional[torch.Tensor] = None
    attention_prob_scale: Optional[torch.Tensor] = None

    query_offset: Optional[torch.Tensor] = None
    kv_offset: Optional[torch.Tensor] = None
    attention_prob_offset: Optional[torch.Tensor] = None

    def get_quant_dtype(self) -> torch.dtype:
        if self.quant_type == AttentionQuantType.INT8:
            return torch.int8
        else:
            raise ValueError(f"Unsupported attention quant type {self.quant_type}")


@dataclasses.dataclass
class MultiheadLatentAttentionQuantConfig(AttentionQuantConfig):
    """
    Quantization configuration for multihead latent attention (MLA) layer.

    Similar to `AttentionQuantConfig`, but with additional quant params
    for the kv projection.

    Check `tensor_cast.multihead_latent_attention_quant` op for more details.
    """

    kv_projected_scale: Optional[torch.Tensor] = None
    kv_projected_offset: Optional[torch.Tensor] = None
    qk_scale: Optional[torch.Tensor] = None
    qk_offset: Optional[torch.Tensor] = None
    v_scale: Optional[torch.Tensor] = None
    v_offset: Optional[torch.Tensor] = None
    out_scale: Optional[torch.Tensor] = None
    out_offset: Optional[torch.Tensor] = None


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
    lmhead_tensor_parallel_size: Optional[int] = None
    lmhead_data_parallel_size: Optional[int] = None
    embedding_parallel: bool = False
    # TODO: use expert_parallel_size instead of expert_parallel
    expert_parallel: bool = False

    def has_attn_tp(self) -> bool:
        return self.tensor_parallel_size > 1

    def has_mlp_tp(self) -> bool:
        return self.mlp_tensor_parallel_size > 1

    def has_lmhead_tp(self) -> bool:
        return self.lmhead_tensor_parallel_size > 1

    def has_ep(self) -> bool:
        return self.expert_parallel and self.world_size > 1

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
    shared_experts_gate: Optional[str] = "shared_experts_gate"
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
    enable_repetition: bool = False
    """Transformer models have repetitive patterns. This configuration flag tells TensorCast
    whether to automatically detect and leverage the repetition patterns to reduce the
    performance estimation cost. This is especially helpful for large models."""
