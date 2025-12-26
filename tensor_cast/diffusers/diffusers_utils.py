import importlib
import json
from contextlib import contextmanager
from functools import wraps

import torch


@contextmanager
def patch_torch_op():
    original_tensor_bool = torch.Tensor.__bool__
    original_tensor_item = torch.Tensor.item
    original_tensor_getitem = torch.Tensor.__getitem__

    def _patched_tensor_bool(self):
        if self.device.type == "meta":
            return True
        return original_tensor_bool(self)

    def _patched_tensor_item(self):
        if self.device.type == "meta":
            return True
        return original_tensor_item(self)

    @wraps(torch.Tensor.__getitem__)
    def _patched_tensor_getitem(self, idx):
        if self.device.type == "meta":
            if (
                isinstance(idx, torch.Tensor)
                and idx.dtype == torch.bool
                and idx.device.type == "meta"
            ):
                return torch.empty(
                    (self.shape[0],) + self.shape[1:], dtype=self.dtype, device="meta"
                )
            try:
                return original_tensor_getitem(self, idx)
            except RuntimeError:
                if isinstance(idx, int):
                    new_shape = self.shape[1:]
                elif isinstance(idx, slice):
                    start, stop, step = idx.indices(self.shape[0])
                    new_dim0 = max(0, (stop - start + step - 1) // step)
                    new_shape = (new_dim0,) + self.shape[1:]
                else:
                    new_shape = self.shape
                return torch.empty(new_shape, dtype=self.dtype, device="meta")
        return original_tensor_getitem(self, idx)

    try:
        torch.Tensor.__bool__ = _patched_tensor_bool
        torch.Tensor.item = _patched_tensor_item
        torch.Tensor.__getitem__ = _patched_tensor_getitem
        yield
    finally:
        torch.Tensor.__bool__ = original_tensor_bool
        torch.Tensor.item = original_tensor_item
        torch.Tensor.__getitem__ = original_tensor_getitem


def get_diffusers_transformer_module(json_path):
    with open(json_path) as f:
        config = json.load(f)

    try:
        module = importlib.import_module("diffusers.models")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Import models from diffusers failed.") from e

    class_name = config.get("_class_name")
    if class_name is None or not isinstance(class_name, str):
        raise ValueError(
            "Unable to find _class_name attribute "
            "or _class_name not a str from the diffusers transformer config.json."
        )
    if class_name not in dir(module):
        raise KeyError(f"The class {class_name} is not supported by diffusers.")
    model_class = getattr(module, class_name)
    return model_class


# 4 = temporal_compression_ratio, 8 = spatial_compression_ratio
_model_class_to_vae_stride = {
    "WanTransformer3DModel": (4, 8),
    "HunyuanVideoTransformer3DModel": (4, 8),
    "HunyuanVideo15Transformer3DModel": (4, 16),
    "Default": (4, 8),
}


def model_class_to_vae_stride(model_class: str) -> tuple:
    if model_class not in _model_class_to_vae_stride.keys():
        model_class = "Default"
    return _model_class_to_vae_stride.get(model_class)


# TODO only wan and hunyuanvideo is supported by now
def generate_hunyuanvideo_input(**kwargs):
    batch_size = kwargs.get("batch_size")
    assert isinstance(batch_size, int)

    seq_lens = kwargs.get("seq_lens")
    assert isinstance(seq_lens, int)

    dtype = kwargs.get("model_config").transformer_config.dtype

    attention_mask = torch.zeros(
        [batch_size, seq_lens],
        device=torch.device("meta"),
        dtype=dtype,
    )
    return {
        "encoder_attention_mask": attention_mask,
    }


def generate_hunyuanvideo15_input(**kwargs):
    res = {}
    batch_size = kwargs.get("batch_size")
    assert isinstance(batch_size, int)

    seq_lens = kwargs.get("seq_lens")
    assert isinstance(seq_lens, int)

    dtype = kwargs.get("model_config").transformer_config.dtype

    attention_mask = torch.zeros(
        [batch_size, seq_lens],
        device=torch.device("meta"),
        dtype=dtype,
    )
    res["encoder_attention_mask"] = attention_mask
    res["encoder_attention_mask_2"] = attention_mask
    text_embed_2_dim = kwargs.get("model_config").transformer_config.model_config.get(
        "text_embed_2_dim"
    )
    if text_embed_2_dim is not None:
        res["encoder_hidden_states_2"] = torch.zeros(
            [batch_size, seq_lens, text_embed_2_dim],
            device=torch.device("meta"),
            dtype=dtype,
        )
    image_embed_dim = kwargs.get("model_config").transformer_config.model_config.get(
        "image_embed_dim"
    )
    if image_embed_dim is not None:
        res["image_embeds"] = torch.zeros(
            [image_embed_dim, image_embed_dim, image_embed_dim],
            device=torch.device("meta"),
            dtype=dtype,
        )
    return res


_model_class_input = {
    "HunyuanVideoTransformer3DModel": generate_hunyuanvideo_input,
    "HunyuanVideo15Transformer3DModel": generate_hunyuanvideo15_input,
}


def model_class_to_input(model_class):
    def generate_empty_input(**kwargs):
        return {}

    input_func = _model_class_input.get(model_class)
    if input_func is None:
        return generate_empty_input
    return input_func
