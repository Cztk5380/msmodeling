# _*_coding:utf-8_*_
"""
model_builder
"""

import torch

from ..compilation import get_backend
from ..core.config_resolver import ConfigResolver
from ..core.user_config import UserInputConfig
from ..transformers.model import TransformerModel


def build_model(user_input: UserInputConfig = None) -> TransformerModel:
    """
    Build a transformer model based on the given args

    :param user_input: user_input
    :return: The loaded (and possibly compiled) Transformer model.
    """
    config_resolver = ConfigResolver(user_input=user_input)
    model_config = config_resolver.resolve()
    model = TransformerModel(user_input.model_id, model_config)
    use_full_graph = not user_input.allow_graph_break
    if user_input.do_compile:
        model = torch.compile(
            model, backend=get_backend(), dynamic=True, fullgraph=use_full_graph
        )
    return model
