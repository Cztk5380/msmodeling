from typing import Dict, Optional, Protocol, Union
import torch
from dataclasses import dataclass
import contextlib

from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .attention import flash_attention_forward, AttentionTensorCast

ALL_ATTENTION_FUNCTIONS["tensor_cast"] = flash_attention_forward


@dataclass
class QuantConfig:
    # TODO
    pass


@dataclass
class ParallelConfig:
    # TODO
    pass


@dataclass
class ModelConfig:
    parallel_config: ParallelConfig
    quant_config: QuantConfig
    dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = True


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
            self.text_config._attn_implementation = "tensor_cast"
            self.model = AutoModel.from_config(
                self.hf_config,
                torch_dtype=self.model_config.dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )
        self.build_attentions()
        self.patch_model()
        self.shard_model()
        self.quantize_model()
        self.load_weights()

    def build_attentions(self):
        self.attention_by_layers = {}
        for i in range(self.text_config.num_hidden_layers):
            # TODO: should not hard-code AttentionTensorCast class here
            self.attention_by_layers[i] = AttentionTensorCast()

    def patch_model(self):
        """
        TODO:
        Patch the transformer model for
        1. torch.compile compatible
        2. Frontend PyTorch module-level fusion
        """
        pass

    def shard_model(self):
        """TODO: Model parallel"""
        pass

    def quantize_model(self):
        """TODO: """
        pass

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
            attention_by_layers=self.attention_by_layers,
            return_dict=False,
            **kwargs)[0]
        return hidden_states
