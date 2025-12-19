import contextlib
import logging
import os
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from ..layers.mla import MultiheadLatentAttentionBase
from ..model_config import AttentionQuantConfig, ModelConfig, MoEConfig, MoEFieldNames

logger = logging.getLogger(__name__)

# TODO: Allow users to extend these default configurations from config.py

# TODO: Using model_type as the key to load the Moe and Attention Config，

# TODO: The model initialization logic needs to be optimized:
#  first load the base config, then load other configs,
#  parse the user's custom inputs,
#  override the corresponding fields in the initialized config, and finally load the model.

_model_id_to_json_tbl: dict[str, str] = {
    "moonshotai/Kimi-K2-Base": "kimi_k2.json",
    "deepseek-ai/DeepSeek-V3.1": "deepseek_v3.1.json",
}


def model_id_to_json(model_id: str) -> Optional[str]:
    return _model_id_to_json_tbl.get(model_id)


# model_id -> MoEConfig
_model_id_to_moe_config: Dict[str, MoEConfig] = {
    "deepseek_v3": MoEConfig(
        module_name="DeepseekV3MoE",
    ),
    "zai-org/GLM-4.5": MoEConfig(
        module_name="Glm4MoeMoE",
    ),
    "glm4_moe": MoEConfig(
        module_name="Glm4MoeMoE",
    ),
    "minimax_m2": MoEConfig(
        module_name="MiniMaxM2SparseMoeBlock",
        gate_returns_raw_logits=True,
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
    "mimo_v2_flash": MoEConfig(
        module_name="MiMoV2MoE",
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


def model_id_to_moe_config(model_id: str, model_type: str = "") -> Optional[MoEConfig]:
    return _model_id_to_moe_config.get(model_id) or _model_id_to_moe_config.get(
        model_type
    )


_model_id_to_mla_module_name: Dict[str, str] = {
    "deepseek-ai/DeepSeek-V3.1": "DeepseekV3Attention",
    "moonshotai/Kimi-K2-Base": "DeepseekV3Attention",
    "deepseek_v3": "DeepseekV3Attention",
}


def model_id_to_mla_module_name(model_id: str, model_type: str = ""):
    return _model_id_to_mla_module_name.get(
        model_id
    ) or _model_id_to_mla_module_name.get(model_type)


_model_id_to_mtp_block_module_name: Dict[str, str] = {
    "deepseek-ai/DeepSeek-V3.1": "DeepseekV3DecoderLayer",
    "moonshotai/Kimi-K2-Base": "DeepseekV3DecoderLayer",
    "deepseek_v3": "DeepseekV3DecoderLayer",
    "glm4_moe": "Glm4MoeDecoderLayer",
    "mimo_v2_flash": "MiMoV2DecoderLayer",
}


def model_id_to_mtp_block_module_name(model_id: str, model_type: str = "") -> str:
    return _model_id_to_mtp_block_module_name.get(
        model_id
    ) or _model_id_to_mtp_block_module_name.get(model_type)


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


class AutoModelConfigLoader:
    def __init__(self):
        self.is_transformers_natively_supported: bool = False

    @staticmethod
    def is_model_type_different(config: PretrainedConfig) -> Tuple[bool, str]:
        """
        Check whether the model type has changed.
        for example: kimi_k2's real model_type is deepseek_v3

        Args:
            config: hf_config.

        Returns:
            tuple: (is_different, type)
                - (False, original_type) if the types are the same
                - (True, current_type) if the types are different
        """
        # Some model config instances do not have a model_type, for example, mimo_v2_flash
        maybe_real_type = config.to_dict()["model_type"]
        if maybe_real_type and config.model_type != maybe_real_type:
            return True, maybe_real_type
        return False, config.model_type

    @staticmethod
    def check_model_path(path):
        """
        Check whether a config.json file and Python files starting with 'configuration' exist in the specified path.

        Args:
            path (str): The directory path to check.

        Returns:
            dict: A dictionary containing the check results:
                - has_config_json (bool): Whether config.json exists.
                - has_configuration_py (bool): Whether any Python file starting with 'configuration' exists.
                - configuration_py_files (list[str]): List of Python files starting with 'configuration'.
        """

        result = {
            "has_config_json": False,
            "has_configuration_py": False,
            "configuration_py_files": [],
        }

        if not os.path.exists(path) or not os.path.isdir(path):
            return result

        for file in os.listdir(path):
            if file == "config.json":
                result["has_config_json"] = True
            elif file.startswith("configuration") and file.endswith(".py"):
                result["has_configuration_py"] = True
                result["configuration_py_files"].append(file)

        return result

    def load_config(self, model_id: str) -> Optional[PretrainedConfig]:
        """
        load config
        """
        check_model_path_res = self.check_model_path(model_id)
        if (
            check_model_path_res["has_config_json"]
            and not check_model_path_res["has_configuration_py"]
        ):
            model_id = os.path.join(
                model_id, "config.json"
            )  # When there's only one configuration file, you should pass the path to the configuration file itself.

        # First, try loading with the native Transformers code; if it's not supported, fall back to using remote code.
        try:
            hf_config = AutoConfig.from_pretrained(model_id)
            self.is_transformers_natively_supported = True
        except Exception:
            hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

            # TODO: Maybe add a config for user to set model_type
            is_diff, real_type = self.is_model_type_different(hf_config)
            if is_diff:
                # Using the real config class to load again
                # for example: use native deepseek_v3 to load kimi-k2`s config.json
                logger.warning(
                    "Using a model of type %s to instantiate again.", real_type
                )
                hf_config = AutoConfig.for_model(real_type).from_dict(
                    hf_config.to_dict()
                )
                self.is_transformers_natively_supported = True

        logger.info(
            "is_transformers_natively_supported = %s",
            self.is_transformers_natively_supported,
        )
        return hf_config

    def load_model(
        self, hf_config: PretrainedConfig, dtype: torch.dtype, **kwargs
    ) -> Optional[PreTrainedModel]:
        trust_remote_code = not self.is_transformers_natively_supported
        if "trust_remote_code" in kwargs:
            trust_remote_code = kwargs.pop("trust_remote_code")

        return self.try_to_load_model(
            hf_config, dtype=dtype, trust_remote_code=trust_remote_code
        )

    def auto_load_model_and_config(
        self, model_id: str, model_config: ModelConfig
    ) -> Tuple[PretrainedConfig, PreTrainedModel]:
        """
        Load the model and config using model_id and model_config.
        """
        hf_config = self.load_config(model_id)
        if model_config.num_hidden_layers_override:
            hf_config.num_hidden_layers = model_config.num_hidden_layers_override
        hf_model = self.load_model(hf_config, model_config.dtype)
        return hf_config, hf_model

    @staticmethod
    def try_to_load_model(*args, **kwarg):
        try:
            hf_model = AutoModel.from_config(*args, **kwarg)
        except Exception:
            hf_model = AutoModelForCausalLM.from_config(*args, **kwarg)
        return hf_model
