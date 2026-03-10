import contextlib
import logging
import os
from operator import attrgetter
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.utils.quantization_config import (
    CompressedTensorsConfig,
    FineGrainedFP8Config,
    QuantizationConfigMixin,
)

from ..layers import COLWISE_LINEAR, ROWWISE_LINEAR
from ..layers.attention_adapters import BailingMoeV2AttentionAdapter
from ..layers.mla import MultiheadLatentAttentionBase
from ..layers.moe_layer import TensorQwen3VLMoeTextMLP
from ..model_config import (
    AttentionQuantConfig,
    ModelConfig,
    MoEConfig,
    MoEFieldNames,
    RemoteSource,
)

logger = logging.getLogger(__name__)

# TODO: Allow users to extend these default configurations from config.py


_model_type_to_moe_config: Dict[str, MoEConfig] = {
    "deepseek_v3": MoEConfig(
        module_name="DeepseekV3MoE",
        num_experts_key="n_routed_experts",
    ),
    "glm4_moe": MoEConfig(
        module_name="Glm4MoeMoE",
        num_experts_key="n_routed_experts",
    ),
    "minimax_m2": MoEConfig(
        module_name="MiniMaxM2SparseMoeBlock",
        gate_returns_raw_logits=True,
        num_experts_key="num_local_experts",
    ),
    "qwen3_moe": MoEConfig(
        module_name="Qwen3MoeSparseMoeBlock",
        gate_returns_raw_logits=True,
    ),
    "qwen3_vl_moe": MoEConfig(
        module_name="Qwen3VLMoeTextSparseMoeBlock",
        gate_returns_raw_logits=True,
        num_experts_key=["text_config", "num_experts"],
    ),
    "glm4v_moe": MoEConfig(
        module_name="Glm4vMoeTextMoE",
        gate_returns_raw_logits=False,
        num_experts_key=["text_config", "n_routed_experts"],
    ),
    "qwen3_next": MoEConfig(
        module_name="Qwen3NextSparseMoeBlock",
        gate_returns_raw_logits=True,
        field_names=MoEFieldNames(
            shared_experts="shared_expert", shared_experts_gate="shared_expert_gate"
        ),
    ),
    "mimo_v2_flash": MoEConfig(
        module_name="MiMoV2MoE",
        num_experts_key="n_routed_experts",
    ),
    "ernie4_5_moe": MoEConfig(
        # This is not a strict mapping to ERNIE MoE which has bias correction
        # and minimal routing weights normalization factor introducing additional
        # computation (div and mul) on the intermediate tensors. But we simply map
        # this to the standard MoE implementation since the additional computation
        # is minor and ignorable compared to other primary ones.
        module_name="Ernie4_5_MoeSparseMoeBlock",
        gate_returns_raw_logits=True,
        num_experts_key="moe_num_experts",
    ),
    "bailing_moe": MoEConfig(
        module_name="BailingMoeV2SparseMoeBlock",
        gate_returns_raw_logits=False,
    ),
}


def get_moe_config(model_type: str = "") -> Optional[MoEConfig]:
    return _model_type_to_moe_config.get(model_type)


_model_type_to_mla_module_name: Dict[str, str] = {
    "deepseek_v3": "DeepseekV3Attention",
}


def get_mla_module_name(model_type: str = "") -> str:
    return _model_type_to_mla_module_name.get(model_type)


_model_type_to_mtp_block_module_name: Dict[str, str] = {
    "deepseek_v3": "DeepseekV3DecoderLayer",
    "glm4_moe": "Glm4MoeDecoderLayer",
    "mimo_v2_flash": "MiMoV2DecoderLayer",
}


def get_mtp_block_module_name(model_type: str = "") -> str:
    return _model_type_to_mtp_block_module_name.get(model_type)


_model_type_to_custom_attention_module_mapping: Dict[str, tuple] = {
    "bailing_moe": ("BailingMoe.*Attention", BailingMoeV2AttentionAdapter),
}


def model_type_to_custom_attention_module_mapping(model_type: str) -> tuple:
    return _model_type_to_custom_attention_module_mapping.get(model_type, (None, None))


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


_model_type_to_custom_expert_module_mapping: Dict[str, type[torch.nn.Module]] = {
    "qwen3_vl_moe": TensorQwen3VLMoeTextMLP,
}


def model_type_to_custom_expert_module_mapping(
    model_type: str,
) -> Optional[type[torch.nn.Module]]:
    return _model_type_to_custom_expert_module_mapping.get(model_type)


"""
This dictionary defines the access paths for model components and their
structural mapping during weight conversion or parallelization.

Key Descriptions:
visual:
    - Meaning: Retrieves the Vision Encoder instance.
    - Purpose: Points to the root module responsible for image feature extraction.

language_model:
    - Meaning: Retrieves the Language Model (LLM) instance.
    - Purpose: Points to the core LLM responsible for text processing and multi-modal fusion.

visual.layers:
    - Meaning: Points to the list of layers (Transformer Layers) within the vision module.
    - Distinction: This is an [Object Accessor]. It tells the program how to retrieve the
      actual Layer objects from the model instance.
    - Mapping: Internally usually corresponds to `visual.blocks` (e.g., Qwen2-VL or GLM).

path.visual.layers:
    - Meaning: The [String Path Representation] of vision layers inside the model.
    - Distinction: This is a [Path Mapping]. It returns a string "visual.blocks" rather than an object.
    - Purpose: Used for distributed strategies or logging to identify weight namespaces in state_dict.

path.language_model.layers:
    - Meaning: The [String Path Representation] of language model layers.
    - Purpose: Same as above, mapping to "language_model.layers".

visual_merger_linear:
    - Meaning: Configuration for linear layers in the vision feature fusion layer (Merger/Projector).
    - Purpose: Targets linear mapping layers that merge or transform multiple visual tokens.
      Returning an empty dict typically indicates using the default parallel strategy.

visual_mlp_linear:
    - Meaning: Configuration for linear layers within the MLP blocks of the vision module.
    - Purpose: Points to the Feed-Forward Network (FFN) inside each Vision Transformer layer.
"""
common_visual_config = {
    "visual": attrgetter("visual"),
    "language_model": attrgetter("language_model"),
    "visual.layers": attrgetter("visual.blocks"),
    "path.visual.layers": lambda _: "visual.blocks",
    "path.language_model.layers": lambda _: "language_model.layers",
    "visual_merger_linear": lambda _: {},
    "visual_mlp_linear": lambda _: {},
}


def resolve_model_config(custom_config=None):
    if custom_config is None:
        return common_visual_config
    visual_config = common_visual_config.copy()
    visual_config.update(custom_config)
    return visual_config


_VISUAL_FAMILY = {
    "default": resolve_model_config(),
    "qwen3_vl": resolve_model_config(
        {
            "visual_merger_linear": lambda _: {
                "visual.merger.linear_fc1": COLWISE_LINEAR,
                "visual.merger.linear_fc2": ROWWISE_LINEAR,
                "visual.deepstack_merger_list.*.linear_fc1": COLWISE_LINEAR,
                "visual.deepstack_merger_list.*.linear_fc2": ROWWISE_LINEAR,
            },
            "visual_mlp_linear": lambda _: {
                "visual.blocks.*.mlp.linear_fc1": COLWISE_LINEAR,
                "visual.blocks.*.mlp.linear_fc2": ROWWISE_LINEAR,
            },
        }
    ),
    "glm4v": resolve_model_config(
        {
            "visual_merger_linear": lambda _: {
                "visual.merger.gate_proj": COLWISE_LINEAR,
                "visual.merger.up_proj": COLWISE_LINEAR,
                "visual.merger.down_proj": ROWWISE_LINEAR,
            },
            "visual_mlp_linear": lambda _: {
                "visual.blocks.*.mlp.gate_proj": COLWISE_LINEAR,
                "visual.blocks.*.mlp.up_proj": COLWISE_LINEAR,
                "visual.blocks.*.mlp.down_proj": ROWWISE_LINEAR,
            },
        }
    ),
    "internvl": resolve_model_config(
        {
            "visual.layers": attrgetter("vision_tower.encoder.layer"),
            "path.visual.layers": lambda _: "vision_tower.encoder.layer",
            "visual_mlp_linear": lambda _: {
                "vision_tower.encoder.layer.*.mlp.fc1": COLWISE_LINEAR,
                "vision_tower.encoder.layer.*.mlp.fc2": ROWWISE_LINEAR,
            },
        }
    ),
}

_MODEL_TYPE_TO_FAMILY = {
    "qwen3_vl": "qwen3_vl",
    "qwen3_vl_moe": "qwen3_vl",
    "glm4v_moe": "glm4v",
    "internvl": "internvl",
}


def patch_method_for_glm4_vl():
    """
    Patch the GLM4V-MoE model to fix simulation issues in meta mode.

    Problem background:
    1. VisionEmbeddings.forward converts lengths in list form to a meta tensor,
        while subsequent computations require actual values (implicitly calling item), which causes errors;
    2. get_placeholder_mask uses boolean-mask-based tensor indexing operations,
        which fail or cause dimension mismatch in meta mode.

    Solution:
    * Convert list-based lengths to a tensor before entering forward, avoiding the creation of a meta tensor.
    * Force image_features=None to skip image-related checks in get_placeholder_mask.
    """

    from transformers.models.glm4v_moe import Glm4vMoeModel

    original_get_placeholder_mask = Glm4vMoeModel.get_placeholder_mask

    def patched_get_placeholder_mask(self, *args, **kwargs):
        # Forcibly skip image_features
        kwargs["image_features"] = None
        return original_get_placeholder_mask(self, *args, **kwargs)

    Glm4vMoeModel.get_placeholder_mask = patched_get_placeholder_mask

    from transformers.models.glm4v_moe.modeling_glm4v_moe import (
        Glm4vMoeVisionEmbeddings,
    )

    original_forward = Glm4vMoeVisionEmbeddings.forward

    def patched_forward(self, *args, **kwargs):
        if len(args) > 1 and isinstance(args[1], list):
            lengths_tensor = torch.tensor(args[1], dtype=torch.long)
            args = (args[0], lengths_tensor) + args[2:]
        return original_forward(self, *args, **kwargs)

    Glm4vMoeVisionEmbeddings.forward = patched_forward


def patch_method_for_vl(model_type):
    patchers = {
        "qwen3_vl": patch_method_for_qwen3_vl,
        "qwen3_vl_moe": patch_method_for_qwen3_vl,
        "glm4v_moe": patch_method_for_glm4_vl,
    }
    patcher = patchers.get(model_type)
    if patcher is not None:
        patcher()


def patch_method_for_qwen3_vl():
    """
    Patch the Qwen3-VL model to fix simulation issues in meta mode.
      Problem background:
      1. The Qwen3-VL model uses boolean-mask-based tensor indexing operations
        (e.g., inputs_embeds[special_image_mask], hidden_states[visual_pos_masks, :]).
      2. These operations cannot run correctly in meta mode because:
         * They internally call nonzero(), whose output shape depends on actual values and
           cannot be inferred in meta mode.
         * Even with meta_nonzero_assume_all_nonzero enabled, dimension mismatch errors still occur.

      Solution:
      * Skip tensor count validation in get_placeholder_mask.
      * Skip the deep stack fusion logic in _deepstack_process.
    """

    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLTextModel,
    )
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeModel,
        Qwen3VLMoeTextModel,
    )

    # Class to be patched
    TARGET_CLASSES = [Qwen3VLModel, Qwen3VLMoeModel]
    # Save the original method of each class.
    ORIGINAL_METHODS = {cls: cls.get_placeholder_mask for cls in TARGET_CLASSES}

    def patched_get_placeholder_mask(self, *args, **kwargs):
        # Forcibly skip image_features
        kwargs["image_features"] = None
        # Invoke the original method of the corresponding class.
        return ORIGINAL_METHODS[type(self)](self, *args, **kwargs)

    for cls in TARGET_CLASSES:
        cls.get_placeholder_mask = patched_get_placeholder_mask

    DEEPSTACK_PROCESS_TARGET_CLASSES = [Qwen3VLTextModel, Qwen3VLMoeTextModel]

    def _patched_deepstack_process(
        self, hidden_states, visual_pos_masks, visual_embeds
    ):
        return hidden_states

    for cls in DEEPSTACK_PROCESS_TARGET_CLASSES:
        cls._deepstack_process = _patched_deepstack_process


# Copied from `accelerate`
@contextlib.contextmanager
def init_on_device_without_buffers(device: torch.device):
    """
    A context manager under which models are initialized with all
    parameters on the specified device. However, buffers are not
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
    modules_to_not_convert_map = {
        # The list of modules to not quantize, useful for quantizing models that explicitly require to have
        #   some modules left in their original precision.
        "fp8": "modules_to_not_convert",
        "fp_quant": "modules_to_not_convert",
        # layer names or types to not quantize, supports regex prefixed by 're:'
        "compressed-tensors": "ignore",
    }

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

    def load_config(
        self, model_id: str, remote_source: str = RemoteSource.huggingface
    ) -> Optional[PretrainedConfig]:
        """
        load config
        """
        if remote_source == RemoteSource.modelscope:
            from modelscope import AutoConfig
        else:
            from transformers import AutoConfig
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
        self,
        hf_config: PretrainedConfig,
        dtype: torch.dtype,
        remote_source: str = RemoteSource.huggingface,
        **kwargs,
    ) -> Optional[PreTrainedModel]:
        trust_remote_code = not self.is_transformers_natively_supported
        if "trust_remote_code" in kwargs:
            trust_remote_code = kwargs.pop("trust_remote_code")

        return self.try_to_load_model(
            hf_config,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            remote_source=remote_source,
        )

    @staticmethod
    def load_quant_config(hf_config: PretrainedConfig) -> QuantizationConfigMixin:
        quant_config = AutoQuantizationConfig.from_dict(hf_config.quantization_config)
        return quant_config

    @staticmethod
    def get_modules_to_not_convert(quant_config) -> List[Optional[str]]:
        modules_to_not_convert = []
        if isinstance(quant_config, FineGrainedFP8Config):
            modules_to_not_convert = quant_config.modules_to_not_convert
        elif isinstance(quant_config, CompressedTensorsConfig):
            modules_to_not_convert = quant_config.quantization_config.ignore
        return modules_to_not_convert

    def auto_load_model_and_config(
        self, model_id: str, model_config: ModelConfig
    ) -> Tuple[PretrainedConfig, PreTrainedModel]:
        """
        Load the model and config using model_id and model_config.
        """
        hf_config = self.load_config(model_id, remote_source=model_config.remote_source)
        if model_config.num_hidden_layers_override:
            hf_config.num_hidden_layers = model_config.num_hidden_layers_override
        hf_model = self.load_model(
            hf_config, model_config.dtype, remote_source=model_config.remote_source
        )
        return hf_config, hf_model

    @staticmethod
    def try_to_load_model(
        *args, remote_source: str = RemoteSource.huggingface, **kwarg
    ):
        if remote_source == RemoteSource.modelscope:
            from modelscope import AutoModel
        else:
            from transformers import AutoModel
        try:
            hf_model = AutoModel.from_config(*args, **kwarg)
        except Exception:
            hf_model = AutoModelForCausalLM.from_config(*args, **kwarg)
        return hf_model
