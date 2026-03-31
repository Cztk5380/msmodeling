import dataclasses
import fnmatch
import importlib
import logging
import operator
import os
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING, Union

import torch

from ..layers.mla import MultiheadLatentAttentionTensorCast
from ..model_config import MlaConfig, MlaFieldNames, MoEConfig, MoEFieldNames, MtpConfig

if TYPE_CHECKING:
    from ..layers.utils import ModelWrapperBase
    from .model import TransformerModel


logger = logging.getLogger(__name__)


_CUSTOM_MODEL_REGISTRY: Dict[str, Callable] = {}
_USER_CUSTOM_MODEL_LOADED = False


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
COMMON_VISUAL_CONFIG = {
    "visual_module_path": "visual",
    "language_module_path": "language_model",
    "visual_layers_module_path": "visual.blocks",
    "visual_layers_path_str": "visual.blocks",
    "language_layers_path_str": "language_model.layers",
    "visual_merger_linear_mapping": {},
    "visual_mlp_linear_mapping": {},
}


def resolve_visual_config(
    custom_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve visual configuration by merging custom config with common defaults.
    Used to generate arguments for ModelProfile's visual configuration fields.
    """
    config = COMMON_VISUAL_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    return config


class MoeExpertMLP(torch.nn.Module):
    def __init__(self, original_experts_module: torch.nn.Module, expert_idx: int):
        super().__init__()
        self.expert_idx = expert_idx
        self.hidden_size = original_experts_module.hidden_dim
        self.intermediate_size = original_experts_module.intermediate_dim
        self.act_fn = original_experts_module.act_fn

        intermediate_dim = original_experts_module.intermediate_dim
        hidden_dim = original_experts_module.hidden_dim

        self.gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, bias=False)

        with torch.no_grad():
            gate_up_weight = original_experts_module.gate_up_proj.data[expert_idx]
            gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)
            self.gate_proj.weight.copy_(gate_weight)
            self.up_proj.weight.copy_(up_weight)
            self.down_proj.weight.copy_(
                original_experts_module.down_proj.data[expert_idx]
            )

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(up * self.act_fn(gate))
        return hidden_states


@dataclasses.dataclass
class ModelProfile:
    """Model Profile containing static metadata and factory methods to build runtime configurations.

    Supported configurations:
    - MoE (Mixture-of-Experts)
    - MTP (Multi-Task Processing)
    - MLA (Multihead Latent Attention)
    - Custom expert module
    - Model family mapping
    - Visual language model patching

    Model families group related model types for unified processing.
    """

    # Unique identifier for the model architecture, usually corresponding to `model_type` in HuggingFace config.
    # Example: "llama", "qwen2", "glm"
    model_type: str

    # --- MoE (Mixture-of-Experts) configuration ---
    # Fully-qualified class name of the MoE expert module.
    # Used to dynamically locate and instantiate the MoE block during tensor conversion or model loading.
    # Example: "transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock"
    moe_module_name: Optional[str] = None

    # Indicates whether the MoE gate network returns raw logits instead of softmax probabilities.
    # If True, subsequent logic will handle the softmax (e.g., for computing load balancing loss).
    moe_gate_returns_raw_logits: bool = False

    # The configuration key(s) used to retrieve the number of experts from the model config.
    # Supports a string or a list of strings (tried in order).
    # Example: ["num_local_experts", "num_experts"]
    moe_num_experts_key: Union[str, List[str]] = "num_experts"

    # Dictionary to override default MoE field mappings defined in MoEFieldNames.
    # Used when a specific model's field naming deviates from the standard.
    # Example: {"gate_proj": "router", "up_proj": "w1"}
    moe_field_names_override: Optional[Dict[str, Any]] = None

    # --- MTP (Multi-Task Processing) configuration ---
    # Fully-qualified class name of the MTP (Multi-Task Processing) block module.
    # Points to the module path handling multi-task or multi-token prediction (e.g., in DeepSeek V3).
    mtp_block_module_name: Optional[str] = None

    # --- MLA (Multihead Latent Attention) configuration ---
    # Fully-qualified class name of the MLA (Multihead Latent Attention) module.
    # Points to the MLA attention class path for tensor conversion or dynamic loading.
    # Example: "transformers.models.deepseek_v2.modeling_deepseek.DeepseekV2Attention"
    mla_module_name: Optional[str] = None

    # Dictionary to override default MLA field mappings defined in MlaFieldNames.
    # Example overriding default names: {"q_proj": "q_a_proj", "kv_a_proj_with_mqa": "kv_a_layernorm"}
    mla_field_names_override: Optional[Dict[str, Any]] = None

    # Python class type for the MLA module.
    # Defaults to the built-in MultiheadLatentAttentionTensorCast.
    # Can be specified if a custom MLA implementation is needed.
    mla_module_class_type: Optional[Type["torch.nn.Module"]] = (
        MultiheadLatentAttentionTensorCast
    )

    # --- Custom expert module configuration ---
    # Python type used for dynamically creating a custom expert module.
    # Provided when the standard MoE expert structure does not meet the requirements.
    custom_expert_module_type: Optional[Type["torch.nn.Module"]] = MoeExpertMLP

    # --- General configuration ---
    # Model family identifier used to group related model types for unified processing.
    # For example, the "llama" family might include "llama", "baichuan", "yi", etc.
    model_family: Optional[str] = None

    # Method for dynamic model patching.
    # A callable executed during model loading or conversion to modify the model structure
    # (e.g., replacing specific attention operators).
    patch_method: Optional[Callable] = None

    # --- Visual language model patching ---
    # Access path for the Vision Encoder instance within the model.
    # Points to the root module responsible for image feature extraction.
    # Example: "model.vision_tower" or "visual"
    visual_module_path: Optional[str] = None

    # Access path for the Language Model (LLM) instance within the model.
    # Points to the core LLM responsible for text processing and multi-modal fusion.
    # Example: "model.text_model" or "language_model"
    language_module_path: Optional[str] = None

    # [Module Import Path] Python module where vision layer classes are defined.
    # Used to dynamically import layer types for model parsing and modification.
    # Example: "transformers.models.clip.modeling_clip"
    visual_layers_module_path: Optional[str] = None

    # [Model Instance Path] Dot-separated attribute chain to access vision layers in the model object.
    # Used to locate actual layer instances in the model structure.
    # Example: "vision_model.encoder.layers"
    visual_layers_path_str: Optional[str] = None

    # [String Path Representation] of the language model layers.
    # Similar to visual_layers_path_str, identifying the LLM layers' weight namespace. Example: "language_model.layers"
    language_layers_path_str: Optional[str] = None

    # Mapping for linear layers in the vision feature merger/projector.
    # Defines how visual features are fused or projected. Empty = default strategy.
    # Example: {"proj": "visual_projection"}
    visual_merger_linear_mapping: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )

    # Mapping for linear layers inside vision MLP blocks (FFN).
    # Used to locate fc1/fc2 in each transformer layer.
    # Example: {"fc1": "fc1", "fc2": "fc2"}
    visual_mlp_linear_mapping: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )

    def _build_field_names(
        self, base_class: Type, override_dict: Optional[Dict[str, Any]]
    ) -> Any:
        if not override_dict:
            return base_class()

        existing_fields = {
            f.name: getattr(base_class(), f.name)
            for f in dataclasses.fields(base_class())
        }
        existing_fields.update(override_dict)
        return base_class(**existing_fields)

    def build_moe_config(
        self,
        enable_redundant: bool = False,
        enable_external_shared: bool = False,
        host_external_shared: bool = False,
        fused_moe_cls: Optional[Type] = None,
    ) -> Optional[MoEConfig]:
        if not self.moe_module_name:
            return None

        field_names = self._build_field_names(
            MoEFieldNames, self.moe_field_names_override
        )

        return MoEConfig(
            module_name=self.moe_module_name,
            fused_moe_cls=fused_moe_cls,
            field_names=field_names,
            gate_returns_raw_logits=self.moe_gate_returns_raw_logits,
            enable_redundant_experts=enable_redundant,
            enable_external_shared_experts=enable_external_shared,
            host_external_shared_experts=host_external_shared,
            num_experts_key=self.moe_num_experts_key,
        )

    def build_mtp_config(self, num_mtp_layers: int) -> Optional[MtpConfig]:
        if not self.mtp_block_module_name or num_mtp_layers <= 0:
            return None

        return MtpConfig(
            num_mtp_layers=num_mtp_layers,
            mtp_block_module_name=self.mtp_block_module_name,
        )

    def build_mla_config(self) -> Optional[MlaConfig]:
        if not self.mla_module_name:
            return None

        field_names = self._build_field_names(
            MlaFieldNames, self.mla_field_names_override
        )

        return MlaConfig(
            module_name=self.mla_module_name,
            field_names=field_names,
        )


def get_model_family(model_type: str) -> Optional[str]:
    profile = get_model_profile(model_type)
    if profile is None:
        return None
    return profile.model_family


def get_mla_module(model_type: str) -> Type["torch.nn.Module"]:
    profile = get_model_profile(model_type)
    if profile is None:
        return MultiheadLatentAttentionTensorCast
    return profile.mla_module_class_type


_MODEL_PROFILE_REGISTRY: Dict[str, ModelProfile] = {}


def register_model_profile(profile: ModelProfile):
    """
    Registers a ModelProfile instance.
    Should be used as a decorator or called directly after defining the profile.
    """
    if profile.model_type in _MODEL_PROFILE_REGISTRY:
        raise ValueError(
            f"ModelProfile for '{profile.model_type}' is already registered."
        )

    _MODEL_PROFILE_REGISTRY[profile.model_type] = profile
    return profile


def get_model_profile(model_type: str) -> Optional[ModelProfile]:
    """
    Retrieves the ModelProfile for a given model type.
    Returns None if the model type is not registered.
    """
    return _MODEL_PROFILE_REGISTRY.get(model_type)


def get_moe_config(model_type: str = "") -> Optional[MoEConfig]:
    if not model_type:
        return None

    profile = get_model_profile(model_type)
    if profile is None:
        return None

    return profile.build_moe_config(
        enable_redundant=False,
        enable_external_shared=False,
        host_external_shared=False,
        fused_moe_cls=None,
    )


def get_vl_model_module(model: "ModelWrapperBase", profile_attr: str, default_key: str):
    profile = get_model_profile(model.hf_config.model_type)
    path = getattr(profile, profile_attr, None)
    if not path and profile and profile.model_family == "default":
        path = COMMON_VISUAL_CONFIG[default_key]
    return operator.attrgetter(path)(model.unwrap()) if path else None


def get_visual(model: "ModelWrapperBase"):
    return get_vl_model_module(model, "visual_module_path", "visual_module_path")


def get_vl_language_model(model: "ModelWrapperBase"):
    return get_vl_model_module(model, "language_module_path", "language_module_path")


def get_visual_layers(model: "ModelWrapperBase"):
    return get_vl_model_module(
        model, "visual_layers_module_path", "visual_layers_module_path"
    )


def get_vl_model_profile_attr(
    model_type: str, profile_attr: str, default_key: str, fallback_value=None
):
    profile = get_model_profile(model_type)
    if profile and getattr(profile, profile_attr, None):
        return getattr(profile, profile_attr)

    if profile and profile.model_family == "default":
        return COMMON_VISUAL_CONFIG[default_key]
    return fallback_value


def get_visual_merger_linear(model_type: str):
    return get_vl_model_profile_attr(
        model_type, "visual_merger_linear_mapping", "visual_merger_linear_mapping", {}
    )


def get_visual_mlp_linear(model_type: str):
    return get_vl_model_profile_attr(
        model_type, "visual_mlp_linear_mapping", "visual_visual_mlp_linear_mapping", {}
    )


def get_visual_layers_path(model_type: str) -> Optional[str]:
    return get_vl_model_profile_attr(
        model_type, "visual_layers_path_str", "visual_layers_path_str", None
    )


def get_language_layers(model_type: str) -> str:
    return get_vl_model_profile_attr(
        model_type, "language_layers_path_str", "language_layers_path_str", "layers"
    )


def get_mla_module_name(model_type: str = "") -> str:
    if not model_type:
        return None
    profile = get_model_profile(model_type)
    return profile.mla_module_name if profile else None


def get_mtp_block_module_name(model_type: str = "") -> str:
    if not model_type:
        return None
    profile = get_model_profile(model_type)
    return profile.mtp_block_module_name if profile else None


def find_matching_key(registry: Dict[str, Any], key: str) -> Optional[str]:
    if not key:
        return None
    for pattern in registry.keys():
        if fnmatch.fnmatchcase(key, pattern) or fnmatch.fnmatch(key, pattern):
            return pattern
    return None


def register_custom_model(model_type: str):
    def decorator(
        fn: Callable[["TransformerModel"], "TransformerModel"],
    ) -> Callable[["TransformerModel"], "TransformerModel"]:
        _CUSTOM_MODEL_REGISTRY[model_type] = fn
        return fn

    return decorator


def get_custom_model(model_type: str) -> Optional[Callable]:
    if not _USER_CUSTOM_MODEL_LOADED:
        import_custom_model_modules()

    match_key = find_matching_key(_CUSTOM_MODEL_REGISTRY, model_type)
    return _CUSTOM_MODEL_REGISTRY.get(match_key) if match_key else None


def import_custom_model_modules():
    global _USER_CUSTOM_MODEL_LOADED
    if _USER_CUSTOM_MODEL_LOADED:
        return

    _PACKAGE_ROOT = os.path.dirname(importlib.util.find_spec("tensor_cast").origin)
    custom_model_path = os.path.join(_PACKAGE_ROOT, "custom_model")
    if not os.path.exists(custom_model_path):
        return
    from tensor_cast import custom_model  # noqa: F401

    _USER_CUSTOM_MODEL_LOADED = True
