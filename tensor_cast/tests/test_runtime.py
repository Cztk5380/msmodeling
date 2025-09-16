import tempfile
import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend
from ..device import A2

from ..layers.attention import AttentionTensorCast
from ..layers.mla import MultiheadLatentAttentionTensorCast
from ..model_config import MlaConfig, ModelConfig, ParallelConfig, QuantConfig
from ..performance_model.analytic import AnalyticPerformanceModel
from ..performance_model.memory_tracker import MemoryTracker
from ..runtime import Runtime
from ..transformers.model import TransformerModel
from ..transformers.utils import model_id_to_json

from .test_common import create_mla_metadata_and_kv_cache, has_submodule_with_cls_name


class PerfAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    def test_simple_model_eager(self):
        def func(x):
            return x + x

        device_profile = A2
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            x = torch.randn([100], device="meta")
            _ = func(x)
        self.assertEqual(len(runtime.event_list), 3)

    def test_simple_model_compile(self):
        @torch.compile(backend=get_backend())
        def func(x):
            return x + x

        device_profile = A2
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            x = torch.randn([100], device="meta")
            _ = func(x)
        self.assertEqual(len(runtime.event_list), 3)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3-235B-A22B", False],
            ["Qwen/Qwen3-235B-A22B", True],
            ["zai-org/GLM-4.5", False],
            ["zai-org/GLM-4.5", True],
        ]
    )
    def test_model(self, model_id, do_compile):
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=2,
        )
        model = TransformerModel(model_id, model_config)
        if do_compile:
            model = torch.compile(model, backend=get_backend(), fullgraph=True)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = A2
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        self.assertIn("tensor_cast.", runtime.table_averages())

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            ["moonshotai/Kimi-K2-Base", True],
        ]
    )
    def test_deepseek(self, model_id, do_compile):
        hf_config_json = model_id_to_json(model_id)
        self.assertIsNotNone(hf_config_json)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            hf_config_json=hf_config_json,
            num_hidden_layers_override=5,  # large enough to include MoE layers
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        model = TransformerModel(model_id, model_config)
        if do_compile:
            model = torch.compile(model, backend=get_backend(), fullgraph=True)
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(
            model, model_config
        )
        # make sure all original attention modules have been replaced
        self.assertTrue(
            has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast")
        )
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        device_profile = A2
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.dispatch_tokens", result)
        self.assertIn("tensor_cast.concat_and_cache_mla", result)
        self.assertIn("tensor_cast.multihead_latent_attention", result)

    def test_table_averages_default(self):
        def func(x):
            return x + 2 * x + x

        device_profile = A2
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
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

        device_profile = A2
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
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

        device_profile = A2
        perf_model = AnalyticPerformanceModel(device_profile)
        with (
            Runtime(
                perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)
            ) as runtime,
            torch.no_grad(),
        ):
            x = torch.randn([100], device="meta")
            _ = func(x)
        with tempfile.TemporaryFile(mode="w+") as temp_file:
            runtime.export_chrome_trace(temp_file)
            temp_file.seek(0)
            content = temp_file.read()
            self.assertIn("aten.randn", content)
            self.assertIn("aten.add", content)
            self.assertIn("aten.mul", content)
