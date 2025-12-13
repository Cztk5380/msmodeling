"""
test_auto_model_config


三类测试场景
1. config的两种情况，在本地、在远程
2. 代码的两种情况，在transformers中，和config在一起
3.

AutoModel,AutoModelForCausalLM 通常情况下，ModelForCausalLM=Model+LMHEAD，整体的加载原则是先在路径下找代码，最差在系统中找
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 要在导入任何transformers库之前

import unittest
from enum import Enum

from parameterized import parameterized
from transformers.modeling_utils import no_init_weights

from ..model_config import (
    ModelConfig,
    ParallelConfig,
    QuantConfig,
)
from ..transformers.utils import AutoModelConfigLoader, init_on_device_without_buffers


class ConfigMode(Enum):
    """
    配置文件位置
    """
    local = 0  # 配置文件在本地
    remote = 1  # 配置文件在远程


class CodeMode(Enum):
    """
    代码位置
    """
    local = 0  # 代码在本地
    inner = 1  # 代码在transformers库中


class AutoModelAndConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.model_config_dir = os.path.join(os.path.dirname(__file__), "data")

    @parameterized.expand(
        [
            ["deepseek_new", ConfigMode.local, CodeMode.inner],  # ds的新配置
            ["deepseekv3.1_remote", ConfigMode.local, CodeMode.local],  # ds的老配置+老代码
            ["deepseekv3.1_remote_json_only", ConfigMode.local, CodeMode.inner],  # 只有ds的老配置
            ["deepseek-ai/DeepSeek-V3.1", ConfigMode.remote, CodeMode.inner],  # 远程仓库有一份老代码，新的仓库没有
            ["zai-org/GLM-4.6", ConfigMode.remote, CodeMode.inner],
            ["minimax_m2", ConfigMode.local, CodeMode.local],
            ["MiniMaxAI/MiniMax-M2", ConfigMode.remote, CodeMode.local],
        ]
    )
    def test_auto_model_config(self, model_name_or_path, config_mode, code_mode):
        if config_mode == ConfigMode.local:
            model_name_or_path = os.path.join(self.model_config_dir, model_name_or_path)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
        )
        with init_on_device_without_buffers("meta"), no_init_weights():
            auto_loader = AutoModelConfigLoader()
            hf_config, hf_model = auto_loader.auto_load_model_and_config(model_name_or_path, model_config)
        self.assertTrue(hf_config)
        self.assertTrue(hf_model)
