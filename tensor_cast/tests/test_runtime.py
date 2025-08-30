import tempfile
import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend

from ..layers.attention import AttentionTensorCast
from ..machine import A2
from ..model_config import ModelConfig, ParallelConfig, QuantConfig
from ..performance_model.analytic import AnalyticPerformanceModel
from ..runtime import Runtime
from ..transformer_model import TransformerModel


class PerfAnalysisTestCase(unittest.TestCase):
    def test_simple_model_eager(self):
        def func(x):
            return x + x

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            x = torch.randn([100], device="meta")
            _ = func(x)
        self.assertEqual(len(runtime.event_list), 3)

    def test_simple_model_compile(self):
        @torch.compile(backend=get_backend())
        def func(x):
            return x + x

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            x = torch.randn([100], device="meta")
            _ = func(x)
        self.assertEqual(len(runtime.event_list), 3)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3-235B-A22B", False],
            # ["Qwen/Qwen3-235B-A22B", True],
            # ["deepseek-ai/DeepSeek-V3"],
            ["zai-org/GLM-4.5", False],
            # ["zai-org/GLM-4.5", True],
        ]
    )
    def test_model_prefill_eager(self, model_id, do_compile):
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(), QuantConfig(), attention_cls=AttentionTensorCast
        )
        model = TransformerModel(model_id, model_config)
        if do_compile:
            model = torch.compile(model, backend=get_backend(), fullgraph=True)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        self.assertIn("tensor_cast.", runtime.table_averages())

    def test_table_averages_default(self):
        def func(x):
            return x + 2 * x + x

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            x = torch.randn([100], device="meta")
            _ = func(x)
        result = runtime.table_averages()
        self.assertIn("analytic total", result)
        self.assertIn("analytic avg", result)
        self.assertIn("aten.randn", result)
        self.assertIn("aten.add", result)
        self.assertIn("aten.mul", result)
        self.assertIn("# of Calls", result)

    def test_table_averages_group_by_shape(self):
        def func(x, y):
            return x + 2 * x + x + y

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            x = torch.randn([10, 10], device="meta")
            y = torch.randn([10, 1], device="meta")
            _ = func(x, y)
        result = runtime.table_averages(group_by_input_shapes=True)
        self.assertIn("analytic total", result)
        self.assertIn("analytic avg", result)
        self.assertIn("Input Shapes", result)
        self.assertIn("aten.randn", result)
        self.assertIn("aten.add", result)
        self.assertIn("aten.mul", result)
        self.assertIn("# of Calls", result)

    def test_export_chrome_trace(self):
        def func(x):
            return x + 2 * x + x

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            x = torch.randn([100], device="meta")
            _ = func(x)
        with tempfile.TemporaryFile(mode="w+") as temp_file:
            runtime.export_chrome_trace(temp_file)
            temp_file.seek(0)
            content = temp_file.read()
            self.assertIn("aten.randn", content)
            self.assertIn("aten.add", content)
            self.assertIn("aten.mul", content)
