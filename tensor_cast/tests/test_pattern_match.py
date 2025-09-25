import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend
from ..device import TEST_DEVICE
from ..layers.quant_linear import TensorCastQuantLinear

from ..model_config import (
    LinearQuantConfig,
    LinearQuantType,
    ModelConfig,
    ParallelConfig,
    QuantConfig,
)
from ..performance_model.analytic import AnalyticPerformanceModel
from ..runtime import Runtime
from ..transformers.model import TransformerModel


def get_linear_quant_config():
    quant_type = LinearQuantType.W8A8
    config_args = {
        "weight_scale": torch.max(torch.abs(torch.randn(1))) / 127.0,
        "quant_type": quant_type,
        "activation_scale": torch.max(torch.abs(torch.randn(1))) / 127.0,
    }
    return LinearQuantConfig(**config_args)


def get_quant_config():
    quant_config = QuantConfig()
    quant_config.linear_configs["*"] = get_linear_quant_config()
    return quant_config


class PatternReplaceTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        num_tokens = 100
        self.compile_backend = get_backend()
        with torch.device("meta"):
            self.inputs = torch.empty([1, num_tokens], dtype=torch.long)
            self.position_ids = torch.empty([1, num_tokens], dtype=torch.long)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
        ]
    )
    def test_rms_norm_pattern(self, model_id):
        # FIXME: this test cannot pass with torch>=2.8, we should fix it
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(), QuantConfig(), num_hidden_layers_override=2
        )
        model = TransformerModel(model_id, model_config)
        model = torch.compile(
            model, backend=self.compile_backend, fullgraph=True, dynamic=True
        )
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm2.default", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
        ]
    )
    def test_rms_norm_quant_pattern(self, model_id):
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(),
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
        )
        model = TransformerModel(model_id, model_config)
        model = torch.compile(
            model, backend=self.compile_backend, fullgraph=True, dynamic=True
        )
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.rms_norm.default", result)
        self.assertIn("tensor_cast.add_rms_norm.default", result)
        self.assertIn("tensor_cast.rms_norm_quant.default", result)
        self.assertIn("tensor_cast.add_rms_norm_quant2.default", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
        ]
    )
    # TODO(hw-whx): add support for special rope type of GLM4.5.
    def test_rope_pattern(self, model_id):
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(),
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
        )
        model = TransformerModel(model_id, model_config)
        model = torch.compile(
            model, backend=self.compile_backend, fullgraph=True, dynamic=True
        )
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.apply_rope.default", result)
