from .deepseek_v32 import DeepseekV32Config, DeepseekV32Model
from .model_register import register


register("deepseek_v32", DeepseekV32Config, DeepseekV32Model)
