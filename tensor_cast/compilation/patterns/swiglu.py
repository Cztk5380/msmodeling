import logging

import torch
import torch._prims as prims

from ... import config

_SWIGLU_DTYPE_LIST = [torch.float16, torch.bfloat16]

logger = logging.getLogger(__name__)


class SwiGLUPattern:
    @staticmethod
    def create(dtype):
        def get_inputs():
            minimal_shape = (1, 1)
            gate = torch.empty(*minimal_shape, dtype=dtype, device="meta")
            up = torch.empty(*minimal_shape, dtype=dtype, device="meta")
            return [gate, up]

        def pattern(gate, up):
            """
            Pattern function for SwiGLU activation fusion (exclude matmul)
            Matches only the activation computation segment:
            gate → fp32 conversion → sigmoid → mul → fp16 conversion → mul with up
            """
            gate_fp32 = prims.convert_element_type(gate, torch.float32)
            sigmoid_gate = torch.ops.aten.sigmoid.default(gate_fp32)
            silu_gate_fp32 = torch.ops.aten.mul.Tensor(gate_fp32, sigmoid_gate)
            silu_gate = prims.convert_element_type(silu_gate_fp32, dtype)
            return torch.ops.aten.mul.Tensor(silu_gate, up)

        def replacement(gate, up):
            return torch.ops.tensor_cast.swiglu(gate, up)

        return pattern, replacement, get_inputs()


def register_all_patterns():
    from . import register_pattern

    if config.compilation.fusion_patterns.enable_swiglu:
        for dtype in _SWIGLU_DTYPE_LIST:
            pattern, replacement, example_inputs = SwiGLUPattern.create(dtype)
            register_pattern(
                f"basic_swiglu_pattern_{dtype}_cast",
                pattern,
                replacement,
                example_inputs,
            )
