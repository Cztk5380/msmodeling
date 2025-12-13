import contextlib
import os
from typing import Dict, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig

from ..layers.mla import MultiheadLatentAttentionBase
from ..model_config import AttentionQuantConfig, MoEConfig, MoEFieldNames, ModelConfig

# TODO: Allow users to extend these default configurations from config.py
# TODO: 全部改成model_type的映射，读取逻辑要优化，先读基础config，然后读取其他的，然后加载model

_model_id_to_json_tbl: dict[str, str] = {
    "moonshotai/Kimi-K2-Base": "kimi_k2.json",
    "deepseek-ai/DeepSeek-V3.1": "deepseek_v3.1.json",
}


def model_id_to_json(model_id: str) -> Optional[str]:
    return _model_id_to_json_tbl.get(model_id)


# todo 后续改成全部是model_type索引
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
    return _model_id_to_moe_config.get(model_id) or _model_id_to_moe_config.get(model_type)


_model_id_to_mla_module_name: Dict[str, str] = {
    "deepseek-ai/DeepSeek-V3.1": "DeepseekV3Attention",
    "moonshotai/Kimi-K2-Base": "DeepseekV3Attention",
    "deepseek_v3": "DeepseekV3Attention"
}


def model_id_to_mla_module_name(model_id: str, model_type: str = ""):
    return _model_id_to_mla_module_name.get(model_id) or _model_id_to_mla_module_name.get(model_type)


_model_id_to_mtp_block_module_name: Dict[str, str] = {
    "deepseek-ai/DeepSeek-V3.1": "DeepseekV3DecoderLayer",
    "moonshotai/Kimi-K2-Base": "DeepseekV3DecoderLayer",
    "deepseek_v3": "DeepseekV3DecoderLayer",
    "glm4_moe": "Glm4MoeDecoderLayer"
}


def model_id_to_mtp_block_module_name(model_id: str, model_type: str = "") -> str:
    return _model_id_to_mtp_block_module_name.get(model_id) or _model_id_to_mtp_block_module_name.get(model_type)


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
        self.is_transformers_natively_supported = False

    @classmethod
    def check_model_path(cls, path):
        """
        检查指定路径下是否存在config.json文件和以configuration开头的py文件

        Args:
            path (str): 要检查的目录路径

        Returns:
            dict: 包含检查结果的字典
                - has_config_json: 是否存在config.json
                - has_configuration_py: 是否存在以configuration开头的py文件
                - configuration_py_files: 以configuration开头的py文件列表
        """

        result = {
            'has_config_json': False,
            'has_configuration_py': False,
            'configuration_py_files': []
        }

        # 检查路径是否存在
        if not os.path.exists(path) or not os.path.isdir(path):
            return result

        # 遍历目录中的文件
        for file in os.listdir(path):
            # 检查是否为config.json
            if file == 'config.json':
                result['has_config_json'] = True
            # 检查是否为以configuration开头的py文件
            elif file.startswith('configuration') and file.endswith('.py'):
                result['has_configuration_py'] = True
                result['configuration_py_files'].append(file)

        return result

    def load_config(self, model_id: str) -> Optional[PretrainedConfig]:
        check_model_path_res = AutoModelConfigLoader.check_model_path(model_id)
        if check_model_path_res["has_config_json"] and not check_model_path_res["has_configuration_py"]:
            model_id = os.path.join(model_id, "config.json")  # 只有一个配置文件的时候要传配置文件本身的路径

        # 先用transformers原生的代码加载，如果不支持，则用remote_code
        try:
            hf_config = AutoConfig.from_pretrained(model_id)
            self.is_transformers_natively_supported = True
        except Exception as e:
            hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        print(f"{self.is_transformers_natively_supported=}")
        return hf_config

    def load_model(self, hf_config: PretrainedConfig, dtype: torch.dtype, trust_remote_code: Optional[bool] = None):

        if trust_remote_code is None:
            trust_remote_code = not self.is_transformers_natively_supported

        # AutoModel,AutoModelForCausalLM 通常情况下，ModelForCausalLM = AutoModelWithLmhead
        auto_map = getattr(hf_config, "auto_map", {})
        # TODO 按照 AutoModelForCausalLM作为默认的处理
        if not auto_map or "AutoModel" in auto_map:
            hf_model = AutoModel.from_config(hf_config, dtype=dtype, trust_remote_code=trust_remote_code)
        elif "AutoModelForCausalLM" in hf_config.auto_map:
            hf_model = AutoModelForCausalLM.from_config(hf_config, dtype=dtype, trust_remote_code=trust_remote_code)
        else:
            raise RuntimeError("Can not load model by one of [AutoModel,AutoModelForCausalLM].")
        return hf_model

    def auto_load_model_and_config(self, model_id: str, model_config: ModelConfig):
        """
        通过model_id和model_config加载model和config

        在本地只有一个config，使用transformers内的代码，才需要把trust_remote_code设置为False
        TODO： 是否要内置一些配置
        """
        hf_config = self.load_config(model_id)
        if model_config.num_hidden_layers_override is not None:
            hf_config.num_hidden_layers = model_config.num_hidden_layers_override
        hf_model = self.load_model(hf_config, model_config.dtype)
        return hf_config, hf_model
