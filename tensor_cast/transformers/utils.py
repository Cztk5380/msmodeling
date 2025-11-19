import contextlib
from typing import Dict, Optional

import torch

from ..layers.mla import MultiheadLatentAttentionBase

from ..model_config import AttentionQuantConfig, MoEConfig, MoEFieldNames

# TODO: Allow users to extend these default configurations from config.py


_model_id_to_json_tbl: dict[str, str] = {
    "moonshotai/Kimi-K2-Base": "kimi_k2.json",
    "deepseek-ai/DeepSeek-V3.1": "deepseek_v3.1.json",
}


def model_id_to_json(model_id: str) -> Optional[str]:
    return _model_id_to_json_tbl.get(model_id)


# model_id -> MoEConfig
_model_id_to_moe_config: Dict[str, MoEConfig] = {
    "zai-org/GLM-4.5": MoEConfig(
        module_name="Glm4MoeMoE",
    ),
    "Qwen/Qwen3-235B-A22B": MoEConfig(
        module_name="Qwen3MoeSparseMoeBlock",
        gate_returns_raw_logits=True,
    ),
    "Qwen/Qwen3-30B-A3B": MoEConfig(
        module_name="Qwen3MoeSparseMoeBlock",
        gate_returns_raw_logits=True,
    ),
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": MoEConfig(
        module_name="Qwen3MoeSparseMoeBlock",
        gate_returns_raw_logits=True,
    ),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": MoEConfig(
        module_name="Qwen3NextSparseMoeBlock",
        gate_returns_raw_logits=True,
        field_names=MoEFieldNames(
            shared_experts="shared_expert", shared_experts_gate="shared_expert_gate"
        ),
    ),
    "moonshotai/Kimi-K2-Base": MoEConfig(
        module_name="DeepseekV3MoE",
    ),
    "deepseek-ai/DeepSeek-V3.1": MoEConfig(
        module_name="DeepseekV3MoE",
    ),
    "baidu/ERNIE-4.5-300B-A47B-PT": MoEConfig(
        # This is not a strict mapping to ERNIE MoE which has bias correction
        # and minimal routing weights normalization factor introducing additional
        # computation (div and mul) on the intermediate tensors. But we simply map
        # this to the standard MoE implementation since the additional computation
        # is minor and ignorable compared to other primary ones.
        module_name="Ernie4_5_MoeSparseMoeBlock",
        gate_returns_raw_logits=True,
    ),
}


def model_id_to_moe_config(model_id: str) -> Optional[MoEConfig]:
    return _model_id_to_moe_config.get(model_id)


_model_id_to_mla_module_name: Dict[str, str] = {
    "deepseek-ai/DeepSeek-V3.1": "DeepseekV3Attention",
    "moonshotai/Kimi-K2-Base": "DeepseekV3Attention",
}


def model_id_to_mla_module_name(model_id: str):
    return _model_id_to_mla_module_name.get(model_id)


_model_id_to_mtp_block_module_name: Dict[str, str] = {
    "deepseek-ai/DeepSeek-V3.1": "DeepseekV3DecoderLayer",
    "moonshotai/Kimi-K2-Base": "DeepseekV3DecoderLayer",
}


def model_id_to_mtp_block_module_name(model_id: str) -> str:
    return _model_id_to_mtp_block_module_name.get(model_id)


def strip_module_name(name: str) -> str:
    """Strip `_inner` module name from the given module path name"""
    stripped = name.removeprefix("_inner.")
    stripped_before = name
    while stripped != stripped_before:
        stripped_before = stripped
        stripped = stripped_before.removeprefix("_inner.")
    stripped = stripped.replace("._inner.", ".")
    stripped_before = stripped
    stripped = stripped_before.removesuffix("._inner")
    while stripped != stripped_before:
        stripped_before = stripped
        stripped = stripped_before.removesuffix("._inner")
    return stripped


def get_attention_quant_config(model, layer_idx) -> Optional[AttentionQuantConfig]:
    if model.model_config.mla_config is not None:
        for _, module in model._inner.named_modules():
            if (
                isinstance(module, MultiheadLatentAttentionBase)
                and hasattr(module, "layer_idx")
                and module.layer_idx == layer_idx
                and (attn_quant_config := module.quant_config) is not None
            ):
                return attn_quant_config
    if hasattr(model, "attention_by_layers") and layer_idx in model.attention_by_layers:
        return model.attention_by_layers[layer_idx].quant_config
    return None


# Copied from `accelerate`
@contextlib.contextmanager
def init_on_device_without_buffers(device: torch.device):
    """
    A context manager under which models are initialized with all
    parameters on the specified device. However buffers are not
    initialized on specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
    """

    old_register_parameter = torch.nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    tensor_constructors_to_patch = [
        # Not a full list of tensor factory functions
        # TODO: align the list with torch._lazy.tensor_factory_functions
        "empty",
        "zeros",
        "ones",
        "arange",
        "randn",
        "rand",
        "randint",
    ]
    old_tensor_constructors = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        for torch_function_name in tensor_constructors_to_patch:
            old_tensor_constructors[torch_function_name] = getattr(
                torch, torch_function_name
            )
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)),
            )
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        for torch_function_name, old_torch_function in old_tensor_constructors.items():
            setattr(torch, torch_function_name, old_torch_function)
