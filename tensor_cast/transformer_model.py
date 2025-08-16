from typing import Dict, Optional, Union
import torch
import contextlib

from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .attention import flash_attention_forward
from .rotary_embedding import CachingRotaryEmb
from .model_config import ModelConfig


ALL_ATTENTION_FUNCTIONS["tensor_cast"] = flash_attention_forward


class TensorDict:
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors


class ModelBase(torch.nn.Module):
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> Union[torch.Tensor, TensorDict]:
        raise NotImplementedError("Subclasses must implement forward method")


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
                module._parameters[name].to(device), **kwargs)

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
            old_tensor_constructors[torch_function_name] = getattr(torch, torch_function_name)
            setattr(
                torch, torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        for torch_function_name, old_torch_function in (
                old_tensor_constructors.items()):
            setattr(torch, torch_function_name, old_torch_function)


class TransformerModel(ModelBase):

    def __init__(self, model_id: str, model_config: ModelConfig):
        super().__init__()
        self.model_id = model_id
        self.model_config = model_config
        with init_on_device_without_buffers("meta"):
            self.hf_config = AutoConfig.from_pretrained(model_id)
            self.text_config = self.hf_config.get_text_config()
            if self.model_config.attention_cls and self.model_config.attention_cls.attn_implmentation:
                self.text_config._attn_implementation = self.model_config.attention_cls.attn_implmentation
            self.model = AutoModel.from_config(
                self.hf_config,
                torch_dtype=self.model_config.dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )
        self.patch_model()
        self.shard_model()
        self.quantize_model()
        self.load_weights()

    def patch_model(self):
        """
        TODO:
        Patch the transformer model for
        1. torch.compile compatible
        2. Frontend PyTorch module-level fusion
        """
        # replace attention with custom implementation if defined
        if self.model_config.attention_cls is not None:
            self.attention_by_layers = {}
            for i in range(self.text_config.num_hidden_layers):
                self.attention_by_layers[i] = self.model_config.attention_cls()
                if i in self.model_config.quant_config.attention_configs:
                    self.attention_by_layers[i].quant_config = self.model_config.quant_config.attention_configs[i]
        
        # cache rotary embedding to avoid computing it each time per forward
        if hasattr(self.model, "rotary_emb"):
            self.model.rotary_emb = CachingRotaryEmb(
                self.model.rotary_emb,
                act_dtype=self.model_config.dtype,
                max_position_embeddings=self.text_config.max_position_embeddings
            )

    def shard_model(self):
        """TODO: Model parallel"""
        pass

    def quant_linear(self):
        """
        Replaces all nn.Linear modules with QuantLinear modules based on the
        quantization configuration stored in self.model_config.
        """
        if self.model_config.quant_linear_cls is None:
            return

        for name, module in self.model.named_modules():
            # We need to find the parent module to replace the child
            if isinstance(module, torch.nn.Linear):
                # Split module path to get parent and child name
                path = name.split('.')
                parent_name = '.'.join(path[:-1])
                child_name = path[-1]
                
                # Find the parent module
                parent_module = self.model
                if parent_name:
                    parent_module = self.model.get_submodule(parent_name)
                
                # Check if a quantization config exists for this specific layer
                if name in self.model_config.quant_config.linear_configs:
                    quant_config = self.model_config.quant_config.linear_configs[name]
                    # Create and set the new quantized module
                    quantized_module = self.model_config.quant_linear_cls(module, quant_config)
                    setattr(parent_module, child_name, quantized_module)

    def quantize_model(self):
        self.quant_linear()

    def load_weights(self):
        """TODO: load real weights"""
        pass

    @property
    def num_hidden_layers(self):
        return self.text_config.num_hidden_layers

    @property
    def hidden_size(self):
        return self.text_config.hidden_size

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> Union[torch.Tensor, TensorDict]:
        hidden_states = self.model(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            attention_by_layers=getattr(self, "attention_by_layers", None),
            return_dict=False,
            **kwargs)[0]
        return hidden_states
