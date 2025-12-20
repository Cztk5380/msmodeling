#!/usr/bin/env python
# _*_coding:utf-8_*_

from enum import auto, Enum
from typing import Optional

import torch

from tensor_cast.utils import DTYPE_FP4, DTYPE_FP8


class LinearQuantType(Enum):
    W8A16 = auto()  # Weight in int8, activation in bfloat16 or half
    W8A8 = auto()  # Weight in int8, activation in int8
    W4A8 = auto()  # Weight in int4, activation in int8
    FP8 = auto()  # Weight in float8, activation in float8
    MXFP4 = auto()  # both weight and activation in MXFP4


def quant_type_to_dynamic_quant_dtype(
    quant_type: LinearQuantType,
) -> Optional[torch.dtype]:
    if quant_type in (LinearQuantType.W8A8, LinearQuantType.W4A8):
        return torch.int8
    elif quant_type == LinearQuantType.FP8:
        return DTYPE_FP8
    elif quant_type == LinearQuantType.MXFP4:
        return DTYPE_FP4
    elif quant_type == LinearQuantType.W8A16:
        return None
    else:
        raise ValueError(f"Unsupported quant_type for dynamic quant: {quant_type}")


def quant_type_to_weight_dtype(quant_type: LinearQuantType) -> torch.dtype:
    if quant_type in (
        LinearQuantType.W8A8,
        LinearQuantType.W4A8,
        LinearQuantType.W8A16,
    ):
        return torch.int8
    elif quant_type == LinearQuantType.FP8:
        return DTYPE_FP8
    elif quant_type == LinearQuantType.MXFP4:
        return DTYPE_FP4
    else:
        raise ValueError(f"Unsupported quant_type for weight quant: {quant_type}")


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
