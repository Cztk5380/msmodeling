import unittest
import torch
import tempfile
from ..runtime import Runtime
from ..machine import A2
from ..performance_model.analytic import AnalyticPerformanceModel
from ..transformer_model import TransformerModel
from ..model_config import ModelConfig, ParallelConfig, QuantConfig
from ..attention import AttentionTensorCast

class PerfAnalysisTestCase(unittest.TestCase):
    def test_simple_model_eager(self):
        def func(x):
            return x + x

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            x = torch.randn([100], device="meta")
            y = func(x)
        self.assertEqual(len(runtime.event_list), 3)

    def test_simple_model_compile(self):
        @torch.compile(backend="eager")
        def func(x):
            return x + x

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            x = torch.randn([100], device="meta")
            y = func(x)
        self.assertEqual(len(runtime.event_list), 3)

    def test_qwen3_prefill_eager(self):
        num_tokens = 100
        model_id = "Qwen/Qwen3-32B"
        model_config = ModelConfig(ParallelConfig(), QuantConfig(), attention_cls=AttentionTensorCast)
        model = TransformerModel(model_id, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        self.assertTrue("tensor_cast." in runtime.table_averages())
        self.assertEqual(len(runtime.event_list), 6031)

    def test_qwen3_prefill_compile(self):
        num_tokens = 100
        model_id = "Qwen/Qwen3-32B"
        model_config = ModelConfig(ParallelConfig(), QuantConfig(), attention_cls=AttentionTensorCast)
        model = torch.compile(TransformerModel(model_id, model_config))
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        print(runtime.table_averages())

    def test_table_averages_default(self):
        def func(x):
            return x + 2 * x + x

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            x = torch.randn([100], device="meta")
            y = func(x)
        result = runtime.table_averages()
        self.assertTrue("analytic total" in result)
        self.assertTrue("analytic avg" in result)
        self.assertTrue("aten.randn" in result)
        self.assertTrue("aten.add" in result)
        self.assertTrue("aten.mul" in result)
        self.assertTrue("# of Calls" in result)

    def test_table_averages_group_by_shape(self):
        def func(x, y):
            return x + 2 * x + x + y

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            x = torch.randn([10, 10], device="meta")
            y = torch.randn([10, 1], device="meta")
            z = func(x, y)
        result = runtime.table_averages(group_by_input_shapes=True)
        self.assertTrue("analytic total" in result)
        self.assertTrue("analytic avg" in result)
        self.assertTrue("Input Shapes" in result)
        self.assertTrue("aten.randn" in result)
        self.assertTrue("aten.add" in result)
        self.assertTrue("aten.mul" in result)
        self.assertTrue("# of Calls" in result)

    def test_export_chrome_trace(self):
        def func(x):
            return x + 2 * x + x

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            x = torch.randn([100], device="meta")
            y = func(x)
        with tempfile.TemporaryFile(mode="w+") as temp_file:
            runtime.export_chrome_trace(temp_file)
            temp_file.seek(0)
            content = temp_file.read()
            self.assertTrue("aten.randn" in content)
            self.assertTrue("aten.add" in content)
            self.assertTrue("aten.mul" in content)
