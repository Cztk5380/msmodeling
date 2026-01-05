import contextlib
import copy
import dataclasses
import fnmatch
import logging
import math
import typing
from typing import Dict, Optional, Union

import torch
from transformers import PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, no_init_weights

from ..layers import (
    COLWISE_LINEAR,
    PARALLEL_EMBEDDING,
    PARALLEL_MODULE_CLS,
    ROWWISE_LINEAR,
)
from ..layers.attention import flash_attention_forward
from ..layers.internal import CopyLayerWrapper, RegionMarkerWrapper
from ..layers.mla import MultiheadLatentAttentionBase
from ..layers.moe_layer import MoELayer, ParallelMoELayer
from ..layers.mtp import MtpWrapper
from ..layers.quant_linear import QuantLinearBase
from ..layers.rotary_embedding import CachingRotaryEmb
from ..layers.utils import ModelWrapperBase
from ..model_config import ModelConfig, MoEConfig
from ..parallel_group import ParallelGroupManager
from ..performance_model.utils import bytes_of_tensor
from ..quantize_utils import quantize_linear_modules
from .utils import (
    AutoModelConfigLoader,
    init_on_device_without_buffers,
    strip_module_name,
    _MODEL_TYPE_TO_FAMILY, _VISUAL_FAMILY,
    patch_method_for_qwen3_vl,
)

if typing.TYPE_CHECKING:
    from ..layers.sampler import SamplingMetadata

logger = logging.getLogger(__name__)

ALL_ATTENTION_FUNCTIONS["tensor_cast"] = flash_attention_forward


class TensorDict:
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors


class CausalLmWrapper(ModelWrapperBase):
    def __init__(self, hf_config, model: torch.nn.Module):
        super().__init__(model)
        self.hf_config = hf_config
        self.lm_head = torch.nn.Linear(
            self.hf_config.hidden_size,
            self.hf_config.vocab_size,
            bias=False,
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



class VLModelWrapper(ModelWrapperBase):
    """
    Vision-Language model wrapper, for Qwen3 VL multimodal models
    """

    def __init__(self, hf_config, model: torch.nn.Module):
        super().__init__(model)
        self.hf_config = hf_config
        hidden_size = hf_config.text_config.hidden_size
        vocab_size = hf_config.text_config.vocab_size
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
            self,
            input_ids: Optional[torch.Tensor],
            position_ids: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs: object,
    ) -> Union[torch.Tensor, TensorDict]:
        outputs = self._inner(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        if sampling_metadata and sampling_metadata.selected_token_indices is not None:
            hidden_states = hidden_states.index_select(
                1, sampling_metadata.selected_token_indices
            )
        logits = self.lm_head(hidden_states)

        return logits


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
        self, model_id: str, model_config: ModelConfig, hf_model: PreTrainedModel = None
    ):
        """
        Construct a transformer model wrapper that auto-loads a transformer model and converts
        it into a model according to the given model configuration.

        Args:
            model_id: transformer model id, (`str` or `os.PathLike`)
            model_config: specify how we should load and convert the transformer model
            hf_model: native model
            #TODO: native model + running config(model_config) = running model,do not need model_id
        """
        super().__init__(None)
        self.model_id = model_id
        self.model_config = model_config
        with init_on_device_without_buffers("meta"), no_init_weights():
            auto_loader = AutoModelConfigLoader()
            if self.model_config.hf_config is not None:
                self.hf_config = self.model_config.hf_config
                if self.model_config.num_hidden_layers_override:
                    self.hf_config.num_hidden_layers = (
                        model_config.num_hidden_layers_override
                    )
                self._inner = auto_loader.load_model(
                    self.hf_config,
                    self.model_config.dtype,
                    trust_remote_code=self.model_config.trust_remote_code,
                )
            else:
                self.hf_config, self._inner = auto_loader.auto_load_model_and_config(
                    self.model_id, self.model_config
                )
            logger.info("origin model and config are loaded successfully")

            self.text_config = self.hf_config.get_text_config()
            self.is_vl_model = hasattr(self.hf_config, "vision_config")
            if (
                self.model_config.attention_cls
                and self.model_config.attention_cls.attn_implmentation
            ):
                self.text_config._attn_implementation = (
                    self.model_config.attention_cls.attn_implmentation
                )
                if self.is_vl_model:
                    self.hf_config.vision_config._attn_implementation = (
                    self.model_config.attention_cls.attn_implmentation)

            self.parallel_group_manager = ParallelGroupManager(
                self.model_config.parallel_config
            )
            # the order of these functions matters!
            with self.set_default_dtype():
                self.wrap_model()
                self.maybe_enable_mtp()
                self.maybe_repeat_layers()
                self.patch_model()
                self.quantize_model()
                self.shard_model()
        self.load_weights()

    @contextlib.contextmanager
    def set_default_dtype(self):
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.model_config.dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(orig_dtype)

    def wrap_model(self):
        """
        Normalize the forward interface so that we don't have to adapt to transformers specifics outside:
        1. We already return torch.Tensor or a tuple of tensors when intermediates are needed
        2. We don't need to pass transformers specific args like `use_cache` or `return_dict` etc. outside.
        This makes other wrappers' life simpler.
        """
        if not self._inner.get_output_embeddings():
            if self.is_vl_model:
                self._inner = VLModelWrapper(
                    hf_config=self.hf_config,
                    model=self._inner,
                )
            else:
                self._inner = CausalLmWrapper(
                    hf_config=self.hf_config,
                    model=self._inner,
                )
        else:
            self._inner = ModelWrapper(self._inner)

    def maybe_enable_mtp(self):
        if not self.model_config.mtp_config:
            return

        mtp_config = copy.deepcopy(self.model_config.mtp_config)
        unwrapped = self.unwrap()
        if mtp_config.mtp_block_module_name is None and hasattr(unwrapped, "layers"):
            # auto mode: use the last decoder layer's class
            decoder_cls_name = type(unwrapped.layers[-1]).__name__
            mtp_config.mtp_block_module_name = decoder_cls_name
            # expand the layer_types used by Qwen model
            if hasattr(self.hf_config, "layer_types") and isinstance(
                self.hf_config.layer_types, list
            ):
                self.hf_config.layer_types.extend(
                    [self.hf_config.layer_types] * mtp_config.num_mtp_layers
                )
            logger.info("Automatic MTP mode using decoder class: %s", decoder_cls_name)

        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.model_config.dtype)
        self._inner = MtpWrapper(mtp_config, self.hf_config, self._inner)
        torch.set_default_dtype(orig_dtype)

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
        visual_layers = self.get_visual_layers()
        if visual_layers is not None:
            repeat_layers(visual_layers)
            language_model = self.get_vl_language_model()
            if hasattr(language_model, "layers"):
                repeat_layers(language_model.layers)

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
        vl_language_model = self.get_vl_language_model()
        if vl_language_model is not None:
            unwrapped = vl_language_model
            patch_method_for_qwen3_vl()
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
            for i in range(self.num_hidden_layers):
                self.attention_by_layers[i] = self.model_config.attention_cls()
            visual_model = self.get_visual()
            if visual_model is not None:
                pattern = 'blocks.*.attn'
                # Assign a depth_layer_idx to each attention layer in the vision model
                # and append them sequentially to attention_by_layers.
                # This allows:
                # 1) vision attention and text attention to use the same attention_by_layers registry
                # 2) each vision attention layer to have a corresponding index
                # 3) during the subsequent flash_attention_forward invocation,
                #    the corresponding attention instance can be retrieved via depth_layer_idx
                depth_layer_idx = len(self.attention_by_layers)
                for name, module in visual_model.named_modules():
                    if fnmatch.fnmatchcase(strip_module_name(name), pattern):
                        module._tensor_cast_context = {
                            "attention_by_layers": self.attention_by_layers,
                            "depth_layer_idx": depth_layer_idx
                        }
                        self.attention_by_layers[depth_layer_idx] = self.model_config.attention_cls()
                        depth_layer_idx += 1

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
        mla_config = self.model_config.mla_config
        if mla_config is None:
            return
        named_modules = list(self._inner.named_modules())
        for name, module in named_modules:
            if type(module).__name__ == mla_config.module_name:
                # check if all the required fields exist
                if not self._all_required_fields_exist(module, mla_config.field_names):
                    continue
                mla = mla_config.mla_cls(
                    mla_config, module, self.parallel_group_manager.tp_group
                )
                self._replace_module(name, mla)

    def get_moe_config(self):
        return self.model_config.moe_config

    def patch_moe(self, moe_config: Optional[MoEConfig]):
        if not moe_config:
            return

        self.top_k = None
        self.num_routing_experts = None
        named_modules = list(self._inner.named_modules())
        for name, module in named_modules:
            if type(module).__name__ == moe_config.module_name:
                if not self._all_required_fields_exist(module, moe_config.field_names):
                    continue
                moe_layer = MoELayer(
                    moe_config,
                    module,
                )
                if self.top_k is None:
                    self.top_k = moe_layer.top_k
                    self.num_routing_experts = len(moe_layer.fused_moe.experts)
                else:
                    assert self.top_k == moe_layer.top_k
                    assert self.num_routing_experts == len(moe_layer.fused_moe.experts)

                self._replace_module(name, moe_layer)

    def get_shard_plan(self):
        tp_group = self.parallel_group_manager.tp_group
        o_proj_tp_group = self.parallel_group_manager.o_proj_tp_group
        mlp_tp_group = self.parallel_group_manager.mlp_tp_group
        lmhead_tp_group = self.parallel_group_manager.lmhead_tp_group
        all_rank_group = self.parallel_group_manager.all_rank_group

        def get_tp_plan():
            # TODO:
            # 1. the name of modules should be configured;
            # 2. we can define a class to represent the data with clearer semantics
            tp_plan = {}

            if self.model_config.parallel_config.embedding_parallel:
                params = {"tp_group": tp_group}
                tp_plan.update({"embed_tokens": (PARALLEL_EMBEDDING, params)})

            params = {
                "tp_group": tp_group,
                "global_tp_group": tp_group,
            }
            config_info = self.hf_config if not self.is_vl_model else self.text_config
            language_layers = self.get_language_layers()
            if self.model_config.mla_config:
                params.update({"head_num": config_info.num_attention_heads})
                tp_plan.update(
                    {
                        "layers.*.self_attn.q_proj": (COLWISE_LINEAR, params),
                        "layers.*.self_attn.q_b_proj": (COLWISE_LINEAR, params),
                        "layers.*.self_attn.kv_b_proj": (COLWISE_LINEAR, params),
                    }
                )
            else:
                params.update({"head_num": config_info.num_attention_heads})
                tp_plan.update({f"{language_layers}.*.self_attn.q_proj": (COLWISE_LINEAR, params)})
                params = params.copy()
                params.update(
                    {
                        "head_num": config_info.num_key_value_heads,
                        "is_replicable": True,
                    }
                )
                tp_plan.update(
                    {
                        f"{language_layers}.*.self_attn.k_proj": (COLWISE_LINEAR, params),
                        f"{language_layers}.*.self_attn.v_proj": (COLWISE_LINEAR, params),
                    }
                )

            params = {
                "tp_group": o_proj_tp_group,
                "global_tp_group": tp_group,
                "head_num": config_info.num_attention_heads,
            }
            tp_plan.update({f"{language_layers}.*.self_attn.o_proj": (ROWWISE_LINEAR, params)})

            params = {
                "tp_group": mlp_tp_group,
                "global_tp_group": tp_group,
            }
            tp_plan.update(
                {
                    # TODO: first complete tensor parallelism for the language_model;
                    #  vision parallelism needs to be handled later
                    f"{language_layers}.*.mlp.gate_proj": (COLWISE_LINEAR, params),
                    f"{language_layers}.*.mlp.up_proj": (COLWISE_LINEAR, params),
                    f"{language_layers}.*.mlp.down_proj": (ROWWISE_LINEAR, params),
                }
            )

            if not self.model_config.parallel_config.has_ep():
                params = {
                    "tp_group": all_rank_group,
                    "global_tp_group": all_rank_group,
                }
                tp_plan.update(
                    {
                        "layers.*.experts.*.gate_proj": (COLWISE_LINEAR, params),
                        "layers.*.experts.*.up_proj": (COLWISE_LINEAR, params),
                        "layers.*.experts.*.down_proj": (ROWWISE_LINEAR, params),
                    }
                )

            params = {
                "tp_group": lmhead_tp_group,
                "global_tp_group": tp_group,
                "gather_output": True,
            }
            tp_plan.update({"lm_head": (COLWISE_LINEAR, params)})
            return tp_plan

        return {"tp_plan": get_tp_plan()}

    def shard_model_by_tp(self):
        """
        Replaces all nn.Linear and nn.Embedding modules with Parallel modules based on the
        parallel configuration stored in self.model_config.
        """
        shard_plan = self.get_shard_plan()
        tp_plan = shard_plan["tp_plan"]

        modules = {}
        module_stripped_to_names = {}
        for name, module in self._inner.named_modules():
            if isinstance(
                module, (torch.nn.Embedding, torch.nn.Linear, QuantLinearBase)
            ):
                modules[name] = module
                module_stripped_to_names[strip_module_name(name)] = name
        for pattern, tp_config in tp_plan.items():
            matches = fnmatch.filter(module_stripped_to_names.keys(), pattern)
            for stripped_name in matches:
                name = module_stripped_to_names[stripped_name]
                module = modules[name]
                parallel_module = PARALLEL_MODULE_CLS[tp_config[0]](
                    module, **tp_config[1]
                )
                self._replace_module(name, parallel_module)

    def shard_model_by_ep(self):
        moe_config = self.get_moe_config()
        if not moe_config or not self.top_k or not self.num_routing_experts:
            return

        ep_group = self.parallel_group_manager.ep_group
        self.num_external_shared_experts = 0
        self.num_redundant_experts = 0
        if not self.model_config.parallel_config.has_ep():
            assert (
                not moe_config.enable_redundant_experts
                and not moe_config.enable_external_shared_experts
            )
        else:
            if moe_config.enable_external_shared_experts:
                assert ep_group.world_size >= 2
                if self.top_k + 1 > ep_group.world_size:
                    self.num_external_shared_experts = 1
                else:
                    self.num_external_shared_experts = math.ceil(
                        ep_group.world_size / (self.top_k + 1)
                    )

                num_routing_experts_device = (
                    ep_group.world_size - self.num_external_shared_experts
                )
                self.num_redundant_experts = (
                    num_routing_experts_device
                    - self.num_routing_experts % num_routing_experts_device
                )
                if (
                    not moe_config.enable_redundant_experts
                    and self.num_redundant_experts == num_routing_experts_device
                ):
                    self.num_redundant_experts = 0

                if not moe_config.host_external_shared_experts:
                    if self.model_config.parallel_config.rank == -1:
                        self.parallel_group_manager.set_rank(
                            self.num_external_shared_experts
                        )
                    else:
                        raise ValueError(
                            "If you want to check the performance of the device with external shared experts, "
                            f"set the rank to -1 or {self.num_external_shared_experts}."
                        )
            else:
                if moe_config.enable_redundant_experts:
                    self.num_redundant_experts = ep_group.world_size

        dp_group = self.parallel_group_manager.dp_group
        tp_group = self.parallel_group_manager.tp_group
        for name, module in self._inner.named_modules():
            if isinstance(module, MoELayer):
                self._replace_module(
                    name,
                    ParallelMoELayer(
                        module,
                        dp_group,
                        tp_group,
                        ep_group,
                        self.num_external_shared_experts,
                        self.num_redundant_experts,
                    ),
                )

    def shard_model(self):
        self.shard_model_by_ep()
        self.shard_model_by_tp()

    def quantize_linear(self):
        """
        Replaces all nn.Linear modules with QuantLinear modules based on the
        quantization configuration stored in self.model_config.
        """
        if not self.model_config.quant_linear_cls:
            return
        quantize_linear_modules(
            self._inner,
            self.model_config.quant_linear_cls,
            self.model_config.quant_config,
            default_config_name=None,
            strip_module_fn=lambda n: n.replace("_inner.", "") if "_inner." in n else n,
        )

    def quantize_attention(self):
        attention_configs = self.model_config.quant_config.attention_configs
        default_attention_config = attention_configs.get(-1)
        if self.model_config.mla_config:
            for _, module in self._inner.named_modules():
                if isinstance(module, MultiheadLatentAttentionBase):
                    if (
                        hasattr(module, "layer_idx")
                        and module.layer_idx in attention_configs
                    ):
                        module.quant_config = attention_configs[module.layer_idx]
                    else:
                        module.quant_config = default_attention_config
                    if module.quant_config is not None:
                        module.quantize_params()
        if hasattr(self, "attention_by_layers"):
            for i in range(self.num_hidden_layers):
                self.attention_by_layers[i].quant_config = attention_configs.get(
                    i, default_attention_config
                )

    def quantize_model(self):
        self.quantize_linear()
        self.quantize_attention()

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

    def _get_vl_model_spec(self):
        model_type = self.hf_config.model_type
        family = _MODEL_TYPE_TO_FAMILY.get(model_type)
        if family is None:
            return None
        return _VISUAL_FAMILY[family]

    def get_visual(self):
        spec = self._get_vl_model_spec()
        if spec is None:
            return None
        return spec["visual"](self.unwrap())

    def get_vl_language_model(self):
        spec = self._get_vl_model_spec()
        if spec is None:
            return None
        return spec["language_model"](self.unwrap())

    def get_visual_layers(self):
        spec = self._get_vl_model_spec()
        if spec is None:
            return None
        return spec["visual.layers"](self.unwrap())

    def get_language_layers(self) -> str:
        """
        Return the string prefix of transformer layers:
          - "language_model.layers"
          - "layers"
        """
        spec = self._get_vl_model_spec()
        if spec is None:
            return "layers"
        return spec["language_model.layers"](self.unwrap())

    @staticmethod
    def get_weight_size_nested(modules):
        total_size = 0
        for mod in modules:
            for _, param in mod.named_parameters():
                total_size += bytes_of_tensor(param)
            for _, buffer in mod.named_buffers():
                total_size += bytes_of_tensor(buffer)
        return total_size

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
        return self.get_weight_size_nested([self])

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
