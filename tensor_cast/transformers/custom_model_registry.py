import dataclasses
import fnmatch
import importlib
import logging
import os

from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING, Union

from ..layers.mla import MultiheadLatentAttentionTensorCast
from ..model_config import MlaConfig, MlaFieldNames, MoEConfig, MoEFieldNames, MtpConfig

if TYPE_CHECKING:
    import torch

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

    model_type: str

    # MoE (Mixture-of-Experts) configuration
    moe_module_name: Optional[str] = None
    """Full-qualified class name of MoE module"""
    moe_gate_returns_raw_logits: bool = False
    """Whether MoE gate returns raw logits"""
    moe_num_experts_key: Union[str, List[str]] = "num_experts"
    """Configuration key(s) to get number of experts"""
    moe_field_names_override: Optional[Dict[str, Any]] = None
    """Field name overrides for MoE"""

    # MTP (Multi-Task Processing) configuration
    mtp_block_module_name: Optional[str] = None
    """Full-qualified class name of MTP block module"""

    # MLA (Multihead Latent Attention) configuration
    mla_module_name: Optional[str] = None
    """Full-qualified class name of MLA module"""
    mla_field_names_override: Optional[Dict[str, Any]] = None
    """Field name overrides for MLA"""

    mla_module_class_type: Optional[Type["torch.nn.Module"]] = (
        MultiheadLatentAttentionTensorCast
    )
    # Custom expert module configuration
    custom_expert_module_type: Optional[Type["torch.nn.Module"]] = None
    """Python type for dynamic custom expert module creation"""

    model_family: Optional[str] = None
    """Model family identifier for grouping related model types"""

    vl_patch_method: Optional[Callable] = None
    """Method for visual language model patching"""

    visual_module_path: Optional[str] = None
    language_module_path: Optional[str] = None
    visual_layers_module_path: Optional[str] = None
    visual_layers_path_str: Optional[str] = None
    language_layers_path_str: Optional[str] = None
    visual_merger_linear_mapping: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
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

    def build_custom_expert_module(
        self, original_module: "torch.nn.Module"
    ) -> Optional["torch.nn.Module"]:
        if self.custom_expert_module_type is None:
            return None

        return self.custom_expert_module_type(original_module)


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
