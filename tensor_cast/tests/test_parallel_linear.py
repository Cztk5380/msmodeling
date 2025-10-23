import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend
from ..device import TEST_DEVICE

from ..layers.quant_linear import TensorCastQuantLinear
from ..model_config import LinearQuantType, ModelConfig, ParallelConfig, QuantConfig
from ..performance_model.analytic import AnalyticPerformanceModel
from ..runtime import Runtime
from ..transformers.model import TransformerModel

from .test_quant_linear import get_quant_config


def get_parallel_config(parallel_configuration: tuple):
    parallel_config = ParallelConfig(
        world_size=parallel_configuration[0],
        tensor_parallel_size=parallel_configuration[1],
        data_parallel_size=parallel_configuration[2],
        mlp_tensor_parallel_size=parallel_configuration[3],
        mlp_data_parallel_size=parallel_configuration[4],
        lmhead_tensor_parallel_size=parallel_configuration[5],
        lmhead_data_parallel_size=parallel_configuration[6],
    )
    if len(parallel_configuration) > 7:
        parallel_config.embedding_parallel = parallel_configuration[7]
    return parallel_config


def has_dp_transform(parallel_config: ParallelConfig):
    if parallel_config.data_parallel_size != parallel_config.mlp_data_parallel_size:
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

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", (16, 1, 16, 1, 16, 1, 16)],
            ["Qwen/Qwen3-32B", (16, 8, 2, 8, 2, 8, 2)],
            ["Qwen/Qwen3-32B", (16, 4, 4, 8, 2, 16, 1)],
            ["Qwen/Qwen3-32B", (16, 4, 4, 8, 2, 16, 1, True)],
            ["zai-org/GLM-4.5", (16, 4, 4, 8, 2, 16, 1)],
        ]
    )
    def test_model_with_tp_and_dp(self, model_id, parallel_configuration):
        parallel_config = get_parallel_config(parallel_configuration)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            enable_lmhead=True,
            num_hidden_layers_override=6,
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
        comm_op_name = "tensor_cast.all_reduce.default"
        if parallel_config.has_attn_tp() or parallel_config.has_mlp_tp():
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
            ["Qwen/Qwen3-32B", (16, 1, 16, 1, 16, 1, 16)],
            ["Qwen/Qwen3-32B", (16, 8, 2, 8, 2, 8, 2)],
            ["Qwen/Qwen3-32B", (16, 4, 4, 8, 2, 16, 1)],
            ["Qwen/Qwen3-32B", (16, 4, 4, 8, 2, 16, 1, True)],
            ["zai-org/GLM-4.5", (16, 4, 4, 8, 2, 16, 1)],
        ]
    )
    def test_model_quant_with_tp_and_dp(self, model_id, parallel_configuration):
        parallel_config = get_parallel_config(parallel_configuration)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            enable_lmhead=True,
            num_hidden_layers_override=6,
        )
        model = TransformerModel(model_id, model_config)

        model_config_with_quant = ModelConfig(
            parallel_config,
            get_quant_config(model.unwrap(), quant_type=LinearQuantType.W4A8),
            quant_linear_cls=TensorCastQuantLinear,
            enable_lmhead=True,
            num_hidden_layers_override=2,
        )
        qmodel = TransformerModel(model_id, model_config_with_quant)

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

        comm_op_name = "tensor_cast.all_reduce.default"
        if parallel_config.has_attn_tp() or parallel_config.has_mlp_tp():
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

        self.assertTrue("tensor_cast.dynamic_quantize_symmetric.default" in result)
