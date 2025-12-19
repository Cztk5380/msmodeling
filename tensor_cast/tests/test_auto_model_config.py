"""
test_auto_model_config


Three types of test scenarios:

1. Two cases for the config: located in a local directory or in a remote repository.
2. Two cases for the code: either in the Transformers library or in the same directory as the config file.
3. A special scenario: the code exists both in a remote directory and in the Transformers library.

Whenever needed, you can execute the following code before importing transformers to configure the HuggingFace proxy.
```
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```
"""

import os
import unittest
from enum import Enum

from parameterized import parameterized
from transformers.modeling_utils import no_init_weights

from ..model_config import ModelConfig, ParallelConfig, QuantConfig
from ..transformers.utils import AutoModelConfigLoader, init_on_device_without_buffers


class ConfigMode(Enum):
    """
    配置文件位置
    """

    local = 0  # The configuration file is in a local directory.
    remote = 1  # The configuration file is in a remote directory


class CodeMode(Enum):
    """
    代码位置
    """

    local = 0  # The code is in a local directory.
    inner = 1  # The code is in the Transformers library.


class AutoModelAndConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.model_config_dir = os.path.join(os.path.dirname(__file__), "data")

    @parameterized.expand(
        [
            # new config of deepseek
            ["deepseek_new", ConfigMode.local, CodeMode.inner],
            # old deepseek configuration + old code
            ["deepseekv3.1_remote", ConfigMode.local, CodeMode.local],
            # only the old deepseek configuration
            ["deepseekv3.1_remote_json_only", ConfigMode.local, CodeMode.inner],
            ["deepseek-ai/DeepSeek-V3.1", ConfigMode.remote, CodeMode.inner],
            ["zai-org/GLM-4.6", ConfigMode.remote, CodeMode.inner],
            ["minimax_m2", ConfigMode.local, CodeMode.local],
            ["MiniMaxAI/MiniMax-M2", ConfigMode.remote, CodeMode.local],
            # model_type is k2,but real type is deepseek
            ["moonshotai/Kimi-K2-Base", ConfigMode.remote, CodeMode.local],
            # model config's model_type is "" and AutoModel in auto_map can not be found in the modeling.
            ["XiaomiMiMo/MiMo-V2-Flash", ConfigMode.remote, CodeMode.local],
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
            hf_config, hf_model = auto_loader.auto_load_model_and_config(
                model_name_or_path, model_config
            )
        self.assertIsNotNone(hf_config)
        self.assertIsNotNone(hf_model)

    @parameterized.expand(
    [
        ["deepseekv3.1_remote_json_only", ConfigMode.local,False],
        ["moonshotai/Kimi-K2-Base", ConfigMode.remote,False],
        ["minimax_m2", ConfigMode.local, True],
        ["moonshotai/Kimi-K2-Thinking", ConfigMode.remote,True],

     ]
    )
    def test_load_quantizer(self, model_name_or_path, config_mode,has_modules_to_not_convert):
        if config_mode == ConfigMode.local:
            model_name_or_path = os.path.join(self.model_config_dir, model_name_or_path)

        with init_on_device_without_buffers("meta"), no_init_weights():
            auto_loader = AutoModelConfigLoader()
            hf_config = auto_loader.load_config(model_name_or_path)
            quant_config, modules_to_not_convert =auto_loader.load_quant_config(hf_config)
            self.assertIsNotNone(quant_config)
            if has_modules_to_not_convert:
                self.assertTrue(modules_to_not_convert)
            else:
                self.assertFalse(modules_to_not_convert)