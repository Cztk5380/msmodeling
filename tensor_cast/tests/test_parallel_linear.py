import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend
from ..device import TEST_DEVICE
from ..layers.mla import MultiheadLatentAttentionTensorCast

from ..layers.quant_linear import TensorCastQuantLinear
from ..model_config import (
    LinearQuantType,
    MlaConfig,
    ModelConfig,
    ParallelConfig,
    QuantConfig,
)
from ..performance_model.analytic import AnalyticPerformanceModel
from ..runtime import Runtime
from ..transformers.model import TransformerModel
from ..transformers.utils import model_id_to_json
from .test_common import create_mla_metadata_and_kv_cache

from .test_quant_linear import get_quant_config


def get_parallel_config(parallel_configuration: tuple):
    parallel_config = ParallelConfig(
        world_size=parallel_configuration[0],
        tensor_parallel_size=parallel_configuration[1],
        o_proj_tensor_parallel_size=parallel_configuration[2],
        mlp_tensor_parallel_size=parallel_configuration[3],
        lmhead_tensor_parallel_size=parallel_configuration[4],
    )
    if len(parallel_configuration) > 5:
        parallel_config.embedding_parallel = parallel_configuration[5]
    return parallel_config


def has_dp_transform(parallel_config: ParallelConfig):
    if parallel_config.data_parallel_size != parallel_config.mlp_data_parallel_size:
        return True
    if parallel_config.data_parallel_size != parallel_config.o_proj_data_parallel_size:
        return True
    if parallel_config.data_parallel_size != parallel_config.lmhead_data_parallel_size:
        return True
    return False


class ParallelLinearTestCase(unittest.TestCase):
    def setUp(self):
        num_tokens = 100
        self.input_batch_size = 2
        self.compile_backend = get_backend()
        with torch.device("meta"):
            self.inputs = torch.empty(
                [self.input_batch_size, num_tokens], dtype=torch.long
            )
            self.position_ids = torch.empty(
                [self.input_batch_size, num_tokens], dtype=torch.long
            )

    def _check_comm_analytic(self, trace_events, comm_op_name):
        count = 0
        for event in trace_events:
            if event["name"] == comm_op_name:
                self.assertIn("message_size_bytes", event["args"])
                count += 1
        self.assertGreater(count, 0)

    def _validate_comm_result(
        self, result: dict, runtime: Runtime, parallel_config: ParallelConfig
    ):
        comm_op_name = "tensor_cast.all_reduce.default"
        if (
            parallel_config.has_attn_tp()
            or parallel_config.has_o_proj_tp()
            or parallel_config.has_mlp_tp()
        ):
            self.assertIn(comm_op_name, result)
            self._check_comm_analytic(runtime.get_trace_events(), comm_op_name)
        else:
            self.assertNotIn(comm_op_name, result)

        comm_op_name = "tensor_cast.all_gather.default"
        if parallel_config.has_lmhead_tp() or has_dp_transform(parallel_config):
            self.assertIn(comm_op_name, result)
            self._check_comm_analytic(runtime.get_trace_events(), comm_op_name)
        else:
            self.assertNotIn(comm_op_name, result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", (16, 1, 1, 1, 1)],
            ["Qwen/Qwen3-32B", (16, 8, 8, 8, 8)],
            ["Qwen/Qwen3-32B", (16, 4, 2, 8, 16)],
            ["Qwen/Qwen3-32B", (16, 4, 2, 8, 16, True)],
            ["zai-org/GLM-4.5", (16, 4, 2, 8, 16)],
        ]
    )
    def test_model_with_tp_and_dp(self, model_id, parallel_configuration):
        parallel_config = get_parallel_config(parallel_configuration)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            enable_lmhead=True,
            enable_repetition=True,
        )
        model = TransformerModel(model_id, model_config)

        num_tokens = 100
        output_batch_size = self.input_batch_size
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(
                outputs.shape, (output_batch_size, num_tokens, model.vocab_size)
            )
        result = runtime.table_averages()
        self._validate_comm_result(result, runtime, parallel_config)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, 4, 2, 8, 16)],
            ["deepseek-ai/DeepSeek-V3.1", (16, 4, 2, 8, 16, True)],
            ["moonshotai/Kimi-K2-Base", (16, 4, 2, 8, 16)],
        ]
    )
    def test_deepseek_with_tp_and_dp(self, model_id, parallel_configuration):
        parallel_config = get_parallel_config(parallel_configuration)

        hf_config_json = model_id_to_json(model_id)
        self.assertIsNotNone(hf_config_json)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            hf_config_json=hf_config_json,
            enable_lmhead=True,
            enable_repetition=True,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        model = TransformerModel(model_id, model_config)

        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(
            model, model_config
        )
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))

        result = runtime.table_averages()
        self._validate_comm_result(result, runtime, parallel_config)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", (16, 1, 1, 1, 1)],
            ["Qwen/Qwen3-32B", (16, 8, 8, 8, 8)],
            ["Qwen/Qwen3-32B", (16, 4, 2, 8, 16)],
            ["Qwen/Qwen3-32B", (16, 4, 2, 8, 16, True)],
            ["zai-org/GLM-4.5", (16, 4, 2, 8, 16)],
        ]
    )
    def test_model_quant_with_tp_and_dp(self, model_id, parallel_configuration):
        parallel_config = get_parallel_config(parallel_configuration)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            enable_lmhead=True,
            enable_repetition=True,
        )
        model = TransformerModel(model_id, model_config)

        model_config.quant_config = get_quant_config(
            model.unwrap(), quant_type=LinearQuantType.W4A8
        )
        model_config.quant_linear_cls = TensorCastQuantLinear
        qmodel = TransformerModel(model_id, model_config)

        num_tokens = 100
        output_batch_size = self.input_batch_size
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = qmodel.forward(self.inputs, self.position_ids)
            self.assertEqual(
                outputs.shape, (output_batch_size, num_tokens, qmodel.vocab_size)
            )
        result = runtime.table_averages()

        self._validate_comm_result(result, runtime, parallel_config)
        self.assertTrue("tensor_cast.dynamic_quantize_symmetric.default" in result)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, 4, 2, 8, 16)],
            ["deepseek-ai/DeepSeek-V3.1", (16, 4, 2, 8, 16, True)],
            ["moonshotai/Kimi-K2-Base", (16, 4, 2, 8, 16)],
        ]
    )
    def test_deepseek_quant_with_tp_and_dp(self, model_id, parallel_configuration):
        parallel_config = get_parallel_config(parallel_configuration)

        hf_config_json = model_id_to_json(model_id)
        self.assertIsNotNone(hf_config_json)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            hf_config_json=hf_config_json,
            enable_lmhead=True,
            enable_repetition=True,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        model = TransformerModel(model_id, model_config)

        model_config.quant_config = get_quant_config(
            model.unwrap(), quant_type=LinearQuantType.W4A8
        )
        model_config.quant_linear_cls = TensorCastQuantLinear
        qmodel = TransformerModel(model_id, model_config)

        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(
            qmodel, model_config
        )
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = qmodel.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.vocab_size))

        result = runtime.table_averages()
        self._validate_comm_result(result, runtime, parallel_config)
        self.assertTrue("tensor_cast.dynamic_quantize_symmetric.default" in result)
