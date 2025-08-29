import torch

from ..pattern_manager import PatternManager
from .utils import NORMAL_PATTERN_DTYPES


class RMSNormPattern:
    """
    Pattern for RMS normalization.
    This pattern computes the RMS normalization of the input tensor.
    """

    eps = 1e-6

    @staticmethod
    def create_pattern_replace(dtype, eps: float = 1e-6):
        RMSNormPattern.eps = eps

        def pattern(hidden_states, weight):
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + RMSNormPattern.eps)
            out = weight * hidden_states.to(dtype)
            return out

        def replace(hidden_states, weight):
            out = torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)
            return out

        return (pattern, replace)


for dtype in NORMAL_PATTERN_DTYPES:
    pattern, replace = RMSNormPattern.create_pattern_replace(dtype)
    # Register the pattern with the PatternManager
    PatternManager.register_pattern(
        f"rms_norm_pattern_{dtype}", (pattern, replace), level=0
    )
