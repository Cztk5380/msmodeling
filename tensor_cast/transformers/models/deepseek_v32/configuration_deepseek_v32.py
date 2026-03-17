from typing import Optional

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


class DeepseekV32Config(DeepseekV3Config):
    model_type = "deepseek_v32"

    def __init__(
        self,
        index_topk: Optional[int] = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_topk = index_topk


__all__ = ["DeepseekV32Config"]
