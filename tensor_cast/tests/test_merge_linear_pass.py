import unittest

import torch

from ..compilation import get_backend
from ..device import TEST_DEVICE

from ..layers.attention import AttentionTensorCast

from ..layers.quant_linear import TensorCastQuantLinear
from ..model_config import LinearQuantType, ModelConfig, ParallelConfig, QuantConfig
from ..performance_model.analytic import AnalyticPerformanceModel

from ..performance_model.memory_tracker import MemoryTracker
from ..runtime import Runtime
from ..transformers.model import TransformerModel

from .test_common import count_events, get_quant_config


class MergeLinearPassTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    def test_qwen3_32b_fp(self):
        model_id = "Qwen/Qwen3-32B"
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=1,
        )
        model = TransformerModel(model_id, model_config)
        model = torch.compile(model, backend=get_backend(), fullgraph=True)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        self.assertEqual(count_events(runtime, torch.ops.aten.mm.default), 4)
        self.assertEqual(
            count_events(runtime, torch.ops.aten.split_with_sizes.default), 2
        )

    def test_qwen3_32b_static_int8(self):
        model_id = "Qwen/Qwen3-32B"
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(
                quant_type=LinearQuantType.W8A8,
                activation_scale=torch.empty(
                    [num_tokens], dtype=torch.float, device="meta"
                ),
            ),
            quant_linear_cls=TensorCastQuantLinear,
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=1,
        )
        model = TransformerModel(model_id, model_config)
        model = torch.compile(model, backend=get_backend(), fullgraph=True)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        self.assertEqual(
            count_events(runtime, torch.ops.tensor_cast.static_quant_linear.default), 4
        )
        self.assertEqual(
            count_events(runtime, torch.ops.aten.split_with_sizes.default), 2
        )

    def test_qwen3_32b_dynamic_int8(self):
        model_id = "Qwen/Qwen3-32B"
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(
                quant_type=LinearQuantType.W8A8,
            ),
            quant_linear_cls=TensorCastQuantLinear,
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=1,
        )
        model = TransformerModel(model_id, model_config)
        model = torch.compile(model, backend=get_backend(), fullgraph=True)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        self.assertEqual(
            count_events(runtime, torch.ops.tensor_cast.static_quant_linear.default), 4
        )
        self.assertEqual(
            count_events(runtime, torch.ops.aten.split_with_sizes.default), 2
        )
