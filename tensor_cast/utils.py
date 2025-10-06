from typing import Optional

import torch

from .model_config import LinearQuantType

# placeholder for FP8, don't hard-code specific fp8 format
DTYPE_FP8 = torch.float8_e5m2
# use int4 placeholder for FP4
DTYPE_FP4 = torch.int4


def register_tensor_cast_op(name, mutates_args=(), **kwargs):
    """
    Register tensor_cast custom op with `name` under tensor_cast namespace.
    We only support meta tensor in the tensor_cast ops so the fake implementation
    is the same as the normal implementation.
    """

    def decorator(func):
        custom_op = torch.library.custom_op(
            f"tensor_cast::{name}", mutates_args=mutates_args, **kwargs
        )(func)
        custom_op.register_fake(func)
        return func

    return decorator


def exact_division(numerator, denominator):
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )
    return numerator // denominator


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
