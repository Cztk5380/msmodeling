import torch

from ... import config

_RMS_NORM_DTYPE_LIST = [torch.float16, torch.bfloat16]


class RMSNormPattern:
    """
    Pattern for RMS normalization.
    This pattern computes the RMS normalization of the input tensor.
    """

    @staticmethod
    def create(dtype, eps: float = 1e-6):
        def get_inputs():
            hidden_states = torch.empty(2, 4, dtype=dtype, device="meta")
            weight = torch.empty(4, dtype=dtype, device="meta")
            return [hidden_states, weight]

        def pattern(hidden_states, weight):
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + eps)
            out = weight * hidden_states.to(dtype)
            return out

        def replacement(hidden_states, weight):
            out = torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)
            return out

        return (pattern, replacement, get_inputs())


class AddRMSNormPattern:
    @staticmethod
    def create(eps: float = 1e-6):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            return [hidden_states, residual, weight]

        def pattern(hidden_states, residual, weight):
            out = torch.ops.tensor_cast.rms_norm(hidden_states + residual, weight, eps)
            return out

        def replacement(hidden_states, residual, weight):
            out = torch.ops.tensor_cast.add_rms_norm(
                hidden_states, residual, weight, eps
            )
            return out

        return (pattern, replacement, get_inputs())


class AddRMSNorm2Pattern:
    @staticmethod
    def create(eps: float = 1e-6):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            return [hidden_states, residual, weight]

        def pattern(hidden_states, residual, weight):
            residual = hidden_states + residual
            out = torch.ops.tensor_cast.rms_norm(residual, weight, eps)
            return out, residual

        def replacement(hidden_states, residual, weight):
            out, residual = torch.ops.tensor_cast.add_rms_norm2(
                hidden_states, residual, weight, eps
            )
            return out, residual

        return (pattern, replacement, get_inputs())


class RMSNormQuantPattern:
    @staticmethod
    def create(eps: float = 1e-6):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            scale = torch.empty(1, device="meta")
            offset = torch.empty(1, device="meta")
            return [hidden_states, weight, scale, offset]

        def pattern(hidden_states, weight, scale, offset):
            out = torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)
            out = torch.ops.tensor_cast.quantize(out, scale, offset)
            return out

        def replacement(hidden_states, weight, scale, offset):
            out = torch.ops.tensor_cast.rms_norm_quant(
                hidden_states, weight, scale, offset, eps
            )
            return out

        return (pattern, replacement, get_inputs())


class AddRMSNormQuantPattern:
    @staticmethod
    def create(eps: float = 1e-6):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            scale = torch.empty(1, device="meta")
            offset = torch.empty(1, device="meta")
            return [hidden_states, residual, weight, scale, offset]

        def pattern(hidden_states, residual, weight, scale, offset):
            out = torch.ops.tensor_cast.rms_norm_quant(
                hidden_states + residual, weight, scale, offset, eps
            )
            return out

        def replacement(hidden_states, residual, weight, scale, offset):
            out = torch.ops.tensor_cast.add_rms_norm_quant(
                hidden_states, residual, weight, scale, offset, eps
            )
            return out

        return (pattern, replacement, get_inputs())


class AddRMSNormQuant2Pattern:
    @staticmethod
    def create(eps: float = 1e-6):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            scale = torch.empty(4, device="meta")
            offset = torch.empty(4, device="meta")
            return [hidden_states, residual, weight, scale, offset]

        def pattern(hidden_states, residual, weight, scale, offset):
            residual = hidden_states + residual
            out = torch.ops.tensor_cast.rms_norm_quant(
                residual, weight, scale, offset, eps
            )
            return out, residual

        def replacement(hidden_states, residual, weight, scale, offset):
            out, residual = torch.ops.tensor_cast.add_rms_norm_quant2(
                hidden_states, residual, weight, scale, offset, eps
            )
            return out, residual

        return (pattern, replacement, get_inputs())


def register_all_patterns():
    from . import register_pattern

    if config.compilation.fusion_patterns.enable_rms_norm:
        for dtype in _RMS_NORM_DTYPE_LIST:
            pattern, replacement, example_inputs = RMSNormPattern.create(dtype)
            # Register the pattern with the PatternManager
            register_pattern(
                f"rms_norm_pattern_{dtype}",
                pattern,
                replacement,
                example_inputs,
                level=0,
            )

    if config.compilation.fusion_patterns.enable_rms_norm_quant:
        register_pattern(
            "rms_norm_quant_pattern",
            *RMSNormQuantPattern.create(),
        )

    if config.compilation.fusion_patterns.enable_add_rms_norm:
        register_pattern(
            "add_rms_norm_pattern",
            *AddRMSNormPattern.create(),
        )
        register_pattern(
            "add_rms_norm2_pattern",
            *AddRMSNorm2Pattern.create(),
        )
        if config.compilation.fusion_patterns.enable_rms_norm_quant:
            register_pattern(
                "add_rms_norm_quant_pattern",
                *AddRMSNormQuantPattern.create(),
            )

            register_pattern(
                "add_rms_norm_quant2_pattern",
                *AddRMSNormQuant2Pattern.create(),
            )
