import dataclasses
from enum import Enum, auto
from typing import Dict, Optional, Type
import torch


class LinearQuantType(Enum):
    W8A16 = auto()  # Weight in int8, activation in bfloat16 or half
    W8A8 = auto()   # Weight in int8, activation in int8
    W4A8 = auto()   # Weight in int4, activation in int8


class QuantGranularity(Enum):
    PER_TENSOR = auto()  # use a single quant param for the entire tensor
    PER_SAMPLE = auto()  # use quant param per sample in the batch (e.g. per-token for LLM)


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
    linear_configs: Dict[str, LinearQuantConfig] = dataclasses.field(default_factory=dict)
    """Per-layer configs: full module path -> LinearQuantConfig"""

    attention_configs: Dict[int, AttentionQuantConfig] = dataclasses.field(default_factory=dict)
    """Per-layer configs: attn_layer_id -> AttentionQuantConfig"""


@dataclasses.dataclass
class ParallelConfig:
    # TODO
    pass


@dataclasses.dataclass
class ModelConfig:
    parallel_config: ParallelConfig
    quant_config: QuantConfig
    attention_cls: Optional[Type["AttentionBase"]] = None
    quant_linear_cls: Optional[Type["QuantLinearBase"]] = None
    dtype: torch.dtype = torch.half
    trust_remote_code: bool = True

