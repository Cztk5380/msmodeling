import contextlib
import dataclasses
import fnmatch
import importlib
import json
import logging
import os
import re
import typing
from typing import Dict, Optional, Union

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, no_init_weights

from ..layers.attention import flash_attention_forward
from ..layers.internal import CopyLayerWrapper, RegionMarkerWrapper
from ..layers.moe_layer import MoELayer, ParallelMoELayer
from ..layers.mtp import MtpWrapper

from ..layers.parallel_linear import COLWISE_LINEAR, PARALLEL_MODULE_CLS, ROWWISE_LINEAR
from ..layers.rotary_embedding import CachingRotaryEmb
from ..layers.utils import ModelWrapperBase
from ..model_config import ModelConfig, MoEConfig
from ..parallel_group import ParallelGroupManager

from ..performance_model.utils import bytes_of_tensor
from .utils import model_id_to_moe_config, strip_module_name

if typing.TYPE_CHECKING:
    from ..layers.sampler import SamplingMetadata

logger = logging.getLogger(__name__)

ALL_ATTENTION_FUNCTIONS["tensor_cast"] = flash_attention_forward


class TensorDict:
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors


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


class CausalLmWrapper(ModelWrapperBase):
    def __init__(self, hf_config, model: torch.nn.Module):
        super().__init__(model)
        self.hf_config = hf_config
        self.lm_head = torch.nn.Linear(
            self.hf_config.hidden_size, self.hf_config.vocab_size, bias=False
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_intermediate_hidden_states: bool = False,  # output hidden_states before lm_head
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> Union[torch.Tensor, TensorDict]:
        hidden_states = self._inner(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=False,
            **kwargs,
        )[0]
        intermediate_hidden_states = hidden_states
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        if sampling_metadata and sampling_metadata.selected_token_indices is not None:
            hidden_states = hidden_states.index_select(
                1, sampling_metadata.selected_token_indices
            )
        hidden_states = self.lm_head(hidden_states)
        if output_intermediate_hidden_states:
            return hidden_states, intermediate_hidden_states
        else:
            return hidden_states


class ModelWrapper(ModelWrapperBase):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> Union[torch.Tensor, TensorDict]:
        hidden_states = self._inner(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=False,
            **kwargs,
        )[0]
        return hidden_states


class TransformerModel(ModelWrapperBase):
    def __init__(
        self,
        model_id: str,
        model_config: ModelConfig,
    ):
        """
        Construct a transformer model wrapper that auto-loads a transformer model and converts
        it into a model according to the given model configuration.

        Args:
            model_id: transformer model id
            model_config: specify how we should load and convert the transformer model
        """
        super().__init__(None)
        self.model_id = model_id
        self.model_config = model_config
        hf_config_json = self.model_config.hf_config_json
        disable_auto_map = self.model_config.disable_auto_map
        with init_on_device_without_buffers("meta"), no_init_weights():
            if hf_config_json is not None:
                if not os.path.isabs(hf_config_json):
                    hf_config_json = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "conf",
                        hf_config_json,
                    )
                self.hf_config = self.load_hf_config_from_json(hf_config_json)
                if disable_auto_map is not None and not disable_auto_map:
                    raise ValueError(
                        "When hf_config_json is specified, `disable_auto_map` should not be set to False."
                    )
                disable_auto_map = True
            else:
                self.hf_config = AutoConfig.from_pretrained(model_id)
            if model_config.num_hidden_layers_override is not None:
                self.hf_config.num_hidden_layers = (
                    model_config.num_hidden_layers_override
                )
            self.text_config = self.hf_config.get_text_config()
            if (
                self.model_config.attention_cls
                and self.model_config.attention_cls.attn_implmentation
            ):
                self.text_config._attn_implementation = (
                    self.model_config.attention_cls.attn_implmentation
                )
            if disable_auto_map and hasattr(self.hf_config, "auto_map"):
                delattr(self.hf_config, "auto_map")
            self._inner = AutoModel.from_config(
                self.hf_config,
                dtype=self.model_config.dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            # the order of these functions matters!
            self.wrap_model()
            self.maybe_enable_mtp()
            self.maybe_repeat_layers()
            self.patch_model()
            self.shard_model()
            self.quantize_model()
        self.load_weights()

    def load_hf_config_from_json(self, hf_config_json: str) -> PretrainedConfig:
        with open(hf_config_json) as f:
            data = json.load(f)
        if "auto_map" not in data or "AutoConfig" not in data["auto_map"]:
            raise ValueError(f"Missing auto_map/AutoConfig in {hf_config_json}")
        autoconfig_name: str = data["auto_map"]["AutoConfig"]
        match = re.match(
            r"configuration_([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", autoconfig_name
        )
        if not match:
            raise ValueError(
                f"Unable to find model name or class name according to pattern"
                f"'configuration_<model_name>.<class_name>' from the value of {autoconfig_name}"
            )
        model_name = match.group(1)
        class_name = match.group(2)
        module = importlib.import_module(f"transformers.models.{model_name}")
        config_cls = getattr(module, class_name)
        return config_cls.from_json_file(hf_config_json)

    def wrap_model(self):
        """
        Normalize the forward interface so that we don't have to adapt to transformers specifics outside:
        1. We already return torch.Tensor or a tuple of tensors when intermediates are needed
        2. We don't need to pass transformers specific args like `use_cache` or `return_dict` etc. outside.
        This makes other wrappers' life simpler.
        """
        self.enable_lmhead = self.model_config.enable_lmhead is True
        if self.model_config.mtp_config and not self.enable_lmhead:
            assert self.model_config.enable_lmhead is None, "MTP on but lmhead is off"
            self.enable_lmhead = True
        if self.enable_lmhead:
            self._inner = CausalLmWrapper(hf_config=self.hf_config, model=self._inner)
        else:
            self._inner = ModelWrapper(self._inner)

    def maybe_enable_mtp(self):
        if not self.model_config.mtp_config:
            return

        self._inner = MtpWrapper(
            self.model_config.mtp_config, self.hf_config, self._inner
        )

    def maybe_repeat_layers(self):
        if not self.model_config.enable_repetition:
            return

        def get_submodule_structure_key(module: torch.nn.Module) -> str:
            submodule_types = []
            for name, sub_module in module.named_modules():
                submodule_types.append(name)
                submodule_types.append(
                    ".".join([type(sub_module).__module__, type(sub_module).__name__])
                )
            return ",".join(submodule_types)

        def repeat_layers(layers):
            # We analyze the structure of sub-modules of each layer to detect repetition patterns.
            # For the first layer of the repetition, we wrap it with RegionMarkerWrapper and then
            # wrap the rest layers of the same pattern with CopyLayerWrapper. This tells the runtime
            # that we can copy the execution of the first layer to the rest layers.
            seen_keys = {}
            for i, layer in enumerate(layers):
                key = get_submodule_structure_key(layer)
                if key not in seen_keys:
                    seen_keys[key] = id(layer)
                    layers[i] = RegionMarkerWrapper(
                        region_id=seen_keys[key],
                        layer=layer,
                    )
                else:
                    layers[i] = CopyLayerWrapper(
                        region_id=seen_keys[key],
                        layer=layer,
                    )

        unwrapped = self.unwrap()
        if hasattr(unwrapped, "layers"):
            repeat_layers(unwrapped.layers)

        if isinstance(self._inner, MtpWrapper):
            repeat_layers(self._inner.mtp.layers)

    def patch_model(self):
        """
        Patch the transformer model for
        1. torch.compile compatible
        2. Frontend PyTorch module-level fusion
        """
        # cache rotary embedding to avoid computing it each time per forward
        unwrapped = self.unwrap()
        if self.model_config.cache_rotary_embedding and hasattr(
            unwrapped, "rotary_emb"
        ):
            unwrapped.rotary_emb = CachingRotaryEmb(
                unwrapped.rotary_emb,
                act_dtype=self.model_config.dtype,
                max_position_embeddings=self.text_config.max_position_embeddings,
            )

        # replace attention with custom implementation if defined
        if self.model_config.attention_cls is not None:
            self.attention_by_layers = {}
            for i in range(self.text_config.num_hidden_layers):
                self.attention_by_layers[i] = self.model_config.attention_cls()
                if i in self.model_config.quant_config.attention_configs:
                    self.attention_by_layers[
                        i
                    ].quant_config = self.model_config.quant_config.attention_configs[i]

        self.patch_mla()

        # replace the vanilla mixture-of-expert (MOE) module with the fused one
        # so that it can be "meta" and torch.compile traced and easily optimized
        # by the backend.
        #
        # NOTE: Why we have to replace the vanilla moe module with the fused one:
        # 1. MOE is data-dependent and the vanilla MOE module usually uses the
        #    data-dependent ops like torch.nonzero or torch.where to route the
        #    experts. This makes it impossible to trace with the "meta" device and
        #    torch.compile based on which we conduct the analysis and graph optimizations.
        # 2. The vanilla MOE usually uses a naive python-based for-loop to distribute
        #    the tokens to the experts, which is slow.
        # 3. The vanilla MOE is not written in a way that can be easily scaled up/out
        #    with expert-parallelism (EP).
        self.patch_moe(self.get_moe_config())

    def _all_required_fields_exist(self, module: torch.nn.Module, field_names):
        def is_optional(annotation):
            if typing.get_origin(annotation) is Union:
                return type(None) in typing.get_args(annotation)
            return False

        for field in dataclasses.fields(field_names):
            if not is_optional(
                type(field_names).__annotations__[field.name]
            ) and not hasattr(module, getattr(field_names, field.name)):
                logger.warning(
                    "Field %s not found in module %s",
                    getattr(field_names, field.name),
                    module,
                )
                return False
        return True

    def patch_mla(self):
        if self.model_config.mla_config:
            mla_config = self.model_config.mla_config
            named_modules = list(self._inner.named_modules())
            for name, module in named_modules:
                if type(module).__name__ == mla_config.module_name:
                    # check if all the required fields exist
                    if not self._all_required_fields_exist(
                        module, mla_config.field_names
                    ):
                        continue
                    mla = mla_config.mla_cls(mla_config, module)
                    self._replace_module(name, mla)

    def get_moe_config(self):
        moe_config = self.model_config.moe_config
        if not moe_config:
            moe_config = model_id_to_moe_config(self.model_id)
        return moe_config

    def patch_moe(self, moe_config: Optional[MoEConfig]):
        if not moe_config:
            return

        named_modules = list(self._inner.named_modules())
        for name, module in named_modules:
            if type(module).__name__ == moe_config.module_name:
                if not self._all_required_fields_exist(module, moe_config.field_names):
                    continue
                moe_layer = MoELayer(
                    moe_config,
                    module,
                )
                self._replace_module(name, moe_layer)

    def get_shard_plan(self):
        tp_group = self.parallel_group_manager.tp_group
        mlp_tp_group = self.parallel_group_manager.mlp_tp_group
        lmhead_tp_group = self.parallel_group_manager.lmhead_tp_group

        def get_tp_plan():
            # TODO:
            # 1. the name of modules should be configured;
            # 2. we can define a class to represent the data with clearer semantics
            tp_plan = {}

            groups = {
                "tp_group": tp_group,
                "global_tp_group": tp_group,
            }
            tp_plan.update(
                {
                    "layers.*.self_attn.q_proj": (COLWISE_LINEAR, groups),
                    "layers.*.self_attn.k_proj": (COLWISE_LINEAR, groups),
                    "layers.*.self_attn.v_proj": (COLWISE_LINEAR, groups),
                    "layers.*.self_attn.o_proj": (ROWWISE_LINEAR, groups),
                }
            )

            groups = {
                "tp_group": mlp_tp_group,
                "global_tp_group": tp_group,
            }
            tp_plan.update(
                {
                    "layers.*.mlp.gate_proj": (COLWISE_LINEAR, groups),
                    "layers.*.mlp.up_proj": (COLWISE_LINEAR, groups),
                    "layers.*.mlp.down_proj": (ROWWISE_LINEAR, groups),
                }
            )

            groups = {
                "tp_group": lmhead_tp_group,
                "global_tp_group": tp_group,
            }
            params = {"gather_output": True}
            tp_plan.update({"lm_head": (COLWISE_LINEAR, {**groups, **params})})
            return tp_plan

        return {"tp_plan": get_tp_plan()}

    def shard_model_by_tp(self):
        """
        Replaces all nn.Linear modules with ParallelLinear modules based on the
        parallel configuration stored in self.model_config.
        """
        shard_plan = self.get_shard_plan()
        tp_plan = shard_plan["tp_plan"]

        linear_modules = {}
        linear_module_stripped_to_names = {}
        for name, module in self._inner.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_modules[name] = module
                linear_module_stripped_to_names[strip_module_name(name)] = name
        for pattern, tp_config in tp_plan.items():
            matches = fnmatch.filter(linear_module_stripped_to_names.keys(), pattern)
            for stripped_name in matches:
                name = linear_module_stripped_to_names[stripped_name]
                module = linear_modules[name]
                parallel_module = PARALLEL_MODULE_CLS[tp_config[0]](
                    module, **tp_config[1]
                )
                self._replace_module(name, parallel_module)

    def shard_model_by_ep(self):
        if not self.model_config.parallel_config.has_ep():
            return

        dp_group = self.parallel_group_manager.dp_group
        tp_group = self.parallel_group_manager.tp_group
        ep_group = self.parallel_group_manager.ep_group
        for name, module in self._inner.named_modules():
            if isinstance(module, MoELayer):
                self._replace_module(
                    name, ParallelMoELayer(module, dp_group, tp_group, ep_group)
                )

    def shard_model(self):
        self.parallel_group_manager = ParallelGroupManager(
            self.model_config.parallel_config
        )

        self.shard_model_by_tp()
        self.shard_model_by_ep()

    def quant_linear(self):
        """
        Replaces all nn.Linear modules with QuantLinear modules based on the
        quantization configuration stored in self.model_config.
        """
        if self.model_config.quant_linear_cls is None:
            return

        # get all the wildcard names from the configuration
        wildcard_linear_configs = {}
        for name in self.model_config.quant_config.linear_configs:
            if "*" in name or "?" in name:
                wildcard_linear_configs[name] = (
                    self.model_config.quant_config.linear_configs[name]
                )

        def get_quant_config(name):
            quant_config = None
            if name in self.model_config.quant_config.linear_configs:
                quant_config = self.model_config.quant_config.linear_configs[name]
            else:
                for wildcard_name in wildcard_linear_configs:
                    if fnmatch.fnmatch(name, wildcard_name):
                        # we only count in the first match
                        quant_config = wildcard_linear_configs[wildcard_name]
                        break
            return quant_config

        for name, module in self._inner.named_modules():
            # We need to find the parent module to replace the child
            if isinstance(module, torch.nn.Linear):
                # remove "_inner" from the module path since we may wrap original
                # module with "_inner".
                # TODO(jgong5): avoid name clashing?
                quant_config = get_quant_config(strip_module_name(name))
                if quant_config:
                    # Create and set the new quantized module
                    quantized_module = self.model_config.quant_linear_cls(
                        module, quant_config
                    )
                    self._replace_module(name, quantized_module)

    def quantize_model(self):
        self.quant_linear()

    def load_weights(self):
        """TODO: load real weights"""

    def _replace_module(self, name: str, new_module: torch.nn.Module):
        # Split module path to get parent and child name
        path = name.split(".")
        parent_name = ".".join(path[:-1])
        child_name = path[-1]
        # Find the parent module
        parent_module = self._inner
        if parent_name:
            parent_module = self._inner.get_submodule(parent_name)
        setattr(parent_module, child_name, new_module)

    @property
    def num_hidden_layers(self):
        num_hidden_layers = self.text_config.num_hidden_layers
        if self.model_config.mtp_config:
            num_hidden_layers += self.model_config.mtp_config.num_mtp_layers
        return num_hidden_layers

    @property
    def hidden_size(self):
        return self.text_config.hidden_size

    @property
    def intermediate_size(self):
        return self.text_config.intermediate_size

    @property
    def vocab_size(self):
        return self.text_config.vocab_size

    @property
    def head_dim(self):
        return getattr(
            self.text_config,
            "head_dim",
            self.hidden_size // self.text_config.num_attention_heads,
        )

    @property
    def weight_size(self):
        def get_weight_size_nested(mod):
            total_size = 0
            for _, param in mod.named_parameters():
                total_size += bytes_of_tensor(param)
            for _, buffer in mod.named_buffers():
                total_size += bytes_of_tensor(buffer)
            return total_size

        return get_weight_size_nested(self)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> Union[torch.Tensor, TensorDict]:
        return self._inner(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            attention_by_layers=getattr(self, "attention_by_layers", None),
            **kwargs,
        )
