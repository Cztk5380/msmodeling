import importlib
import json
import torch


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

    dtype = kwargs.get("dtype")

    attention_mask = torch.zeros(
        [batch_size, seq_lens],
        device=torch.device("meta"),
        dtype=dtype,
    )
    return {
        "encoder_attention_mask": attention_mask,
    }


_model_class_input = {
    "HunyuanVideoTransformer3DModel": generate_hunyuanvideo_input,
}


def model_class_to_input(model_class):
    return _model_class_input.get(model_class,lambda **kwargs: {})


def get_ulysses_split_dim(hidden_states: torch.Tensor, ulysses_size: int) -> int:
    if hidden_states is None:
        raise ValueError("hidden_states is None")
    if hidden_states.shape[-2] // 2 % ulysses_size == 0:
        split_dim = -2
    elif hidden_states.shape[-1] // 2 % ulysses_size == 0:
        split_dim = -1
    else:
        raise ValueError(f"Cannot split video sequence into {ulysses_size}")
    return split_dim
