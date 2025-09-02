import torch

from ... import config


class RMSNormPattern:
    """
    Pattern for RMS normalization.
    This pattern computes the RMS normalization of the input tensor.
    """

    @staticmethod
    def create(dtype, eps: float = 1e-6):
        RMSNormPattern.eps = eps

        def pattern(hidden_states, weight):
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + RMSNormPattern.eps)
            out = weight * hidden_states.to(dtype)
            return out

        def replacement(hidden_states, weight):
            out = torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)
            return out

        return (pattern, replacement)


class AddRMSNormPattern:
    @staticmethod
    def create(eps: float = 1e-6):
        def pattern(hidden_states, residual, weight):
            out = torch.ops.tensor_cast.rms_norm(hidden_states + residual, weight, eps)
            return out

        def replacement(hidden_states, residual, weight):
            out = torch.ops.tensor_cast.add_rms_norm(
                hidden_states, residual, weight, eps
            )
            return out

        return (pattern, replacement)


class AddRMSNorm2Pattern:
    @staticmethod
    def create(eps: float = 1e-6):
        def pattern(hidden_states, residual, weight):
            residual = hidden_states + residual
            out = torch.ops.tensor_cast.rms_norm(residual, weight, eps)
            return out, residual

        def replacement(hidden_states, residual, weight):
            out, residual = torch.ops.tensor_cast.add_rms_norm2(
                hidden_states, residual, weight, eps
            )
            return out, residual

        return (pattern, replacement)


def quant_no_offset_wrapper(pattern, replacement):
    def pattern_wrapper(hidden_states, weight, scale):
        return pattern(hidden_states, weight, scale, None)

    def replacement_wrapper(hidden_states, weight, scale):
        return replacement(hidden_states, weight, scale, None)

    return pattern_wrapper, replacement_wrapper


def add_quant_no_offset_wrapper(pattern, replacement):
    def pattern_wrapper(hidden_states, residual, weight, scale):
        return pattern(hidden_states, residual, weight, scale, None)

    def replacement_wrapper(hidden_states, residual, weight, scale):
        return replacement(hidden_states, residual, weight, scale, None)

    return pattern_wrapper, replacement_wrapper


class RMSNormQuantPattern:
    @staticmethod
    def create(eps: float = 1e-6, has_offset=True):
        def pattern(hidden_states, weight, scale, offset):
            out = torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)
            out = torch.ops.tensor_cast.quantize(
                out, scale, offset, out_dtype=torch.int8
            )
            return out

        def replacement(hidden_states, weight, scale, offset):
            out = torch.ops.tensor_cast.rms_norm_quant(
                hidden_states, weight, scale, offset, eps
            )
            return out

        if has_offset:
            return (pattern, replacement)
        else:
            return quant_no_offset_wrapper(pattern, replacement)


class AddRMSNormQuantPattern:
    @staticmethod
    def create(eps: float = 1e-6, has_offset=True):
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

        if has_offset:
            return (pattern, replacement)
        else:
            return add_quant_no_offset_wrapper(pattern, replacement)


class AddRMSNormQuant2Pattern:
    @staticmethod
    def create(eps: float = 1e-6, has_offset=True):
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

        if has_offset:
            return (pattern, replacement)
        else:
            return add_quant_no_offset_wrapper(pattern, replacement)


def register_all_patterns():
    from . import register_pattern

    if config.compilation.fusion_patterns.enable_rms_norm:
        for dtype in [torch.float16, torch.bfloat16]:
            pattern, replacement = RMSNormPattern.create(dtype)
            # Register the pattern with the PatternManager
            register_pattern(f"rms_norm_pattern_{dtype}", pattern, replacement, level=0)

    if config.compilation.fusion_patterns.enable_rms_norm_quant:
        for has_offset in [True, False]:
            register_pattern(
                f"rms_norm_quant_pattern_{has_offset}",
                *RMSNormQuantPattern.create(has_offset=has_offset),
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
            for has_offset in [True, False]:
                register_pattern(
                    f"add_rms_norm_quant_pattern_{has_offset}",
                    *AddRMSNormQuantPattern.create(has_offset=has_offset),
                )
            for has_offset in [True, False]:
                register_pattern(
                    f"add_rms_norm_quant2_pattern_{has_offset}",
                    *AddRMSNormQuant2Pattern.create(has_offset=has_offset),
                )
