import json
import logging
import os
from typing import Dict, Optional, Union

import torch
from transformers.modeling_utils import no_init_weights

from ..layers.attention import AttentionTensorCast
from ..layers.quant_linear import TensorCastQuantLinear

from ..layers.utils import ModelWrapperBase
from ..model_config import (
    DiffusersConfig,
    DiffusersTransformerConfig,
    DiffusersVaeConfig,
    ModelConfig,
)

from ..quantize_utils import quantize_linear_modules

from ..transformers.model import ModelWrapper

from ..transformers.utils import init_on_device_without_buffers

from .diffusers_utils import get_diffusers_transformer_module


logger = logging.getLogger(__name__)


def build_diffusers_transformer_model(
    model_id: str,
    parallel_config: None,
    quant_config: None,
    dtype: torch.dtype,
):
    model_config = load_config_from_file(
        model_path=model_id,
        parallel_config=parallel_config,
        quant_config=quant_config,
        quant_linear_cls=TensorCastQuantLinear,
        attention_cls=AttentionTensorCast,
        dtype=dtype,
    )
    model = DiffusersTransformerModel(model_id, model_config.transformer_config)
    return model, model_config


def load_config_from_file(
    model_path: str,
    parallel_config: None,
    quant_config: None,
    quant_linear_cls: None,
    attention_cls: None,
    dtype: torch.dtype,
):
    # TODO add seperate parallel_config and quant_config(atten_cls is needed?) for vae and text
    if not os.path.isdir(model_path):
        raise ValueError(f"Input args.model_id should be dir, but got {model_path}")

    config_path_dict = {}
    model_path = os.path.abspath(model_path)
    for root, _, files in os.walk(model_path):
        if "config.json" in files:
            folder_name = os.path.basename(root)
            config_path = os.path.join(root, "config.json")
            config_path = os.path.abspath(config_path)
            config_path_dict[folder_name] = config_path

    config_dict = {}
    for key, config_path in config_path_dict.items():
        with open(config_path) as f:
            config = json.load(f)
        config_dict[key] = config

    model_config = DiffusersConfig()
    transformer_config = config_dict.get("transformer")
    if transformer_config is None:
        raise ValueError("No transformer config.json found in input model path.")
    transformer_config_path = config_path_dict.get("transformer")
    model_config.transformer_config = DiffusersTransformerConfig(
        parallel_config=parallel_config,
        quant_config=quant_config,
        config_json=transformer_config_path,
        model_config=transformer_config,
        quant_linear_cls=quant_linear_cls,
        attention_cls=attention_cls,
        dtype=dtype,
    )

    vae_config = config_dict.get("vae")
    if vae_config is None:
        raise ValueError("No vae config.json found in input model path.")
    model_config.vae_config = DiffusersVaeConfig(
        parallel_config=parallel_config,
        quant_config=quant_config,
        model_config=vae_config,
        dtype=dtype,
    )
    return model_config


class DiffusersModel(ModelWrapperBase):
    def __init__(
        self,
        model_id: str,
        model_config: ModelConfig,
    ):
        super().__init__(None)
        self.model_id = model_id
        self.model_config = model_config

        # TODO Diffusers pipline include: TextModel VaeModel TransformerModel.
        # Only TransformerModel is supported by now.
        # TransformerModel refers to DiffusersTransformerModel.

    def wrap_model(self):
        self._inner = ModelWrapper(self._inner)

    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        res = self.transformer(*args, **kwargs)
        return res


class DiffusersTransformerModel(ModelWrapperBase):
    def __init__(
        self,
        model_id: str,
        model_config: DiffusersTransformerConfig,
    ):
        super().__init__(None)
        self.model_id = model_id
        self.model_config = model_config

        hf_config_json = self.model_config.config_json

        if hf_config_json is None:
            raise ValueError("hf_config_json should not be None.")
        hf_config_json = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "conf",
            hf_config_json,
        )
        model_class = get_diffusers_transformer_module(hf_config_json)

        with init_on_device_without_buffers("meta"), no_init_weights():
            self._inner = model_class.from_config(hf_config_json).to(model_config.dtype)
        self._inner.eval()
        self.wrap_model()
        self.quantize_model()

    def wrap_model(self):
        # diffusers attention backend registered in diffuser_attention.py
        self._inner.set_attention_backend("tensor_cast")

    def quantize_model(self):
        # TODO quantization on cuda: github NVIDIA/Model-Optimizer/tree/main/examples/diffusers
        # TODO whether linears outside blocks should be quant?
        self.quantize_linear()

    def quantize_linear(self):
        if not self.model_config.quant_linear_cls:
            return
        root = (
            self._inner.transformer_blocks
            if hasattr(self._inner, "transformer_blocks")
            else self._inner.blocks
            if hasattr(self._inner, "blocks")
            else None
        )
        quantize_linear_modules(
            root,
            self.model_config.quant_linear_cls,
            self.model_config.quant_config,
            default_config_name="default_dit",
            strip_module_fn=None,
            pattern_match_fn=None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_images: Optional[torch.Tensor] = None,
        return_dict=False,
        **kwargs: object,
    ):
        hidden_states = self._inner(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )[0]
        return hidden_states
