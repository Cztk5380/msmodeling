import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend
from ..device import TEST_DEVICE

from ..model_config import ModelConfig, ParallelConfig, QuantConfig
from ..performance_model.analytic import AnalyticPerformanceModel
from ..runtime import Runtime
from ..transformers.model import TransformerModel
from ..transformers.utils import model_id_to_json


def get_parallel_config(parallel_configuration: tuple):
    parallel_config = ParallelConfig(
        world_size=parallel_configuration[0],
        tensor_parallel_size=parallel_configuration[1],
        data_parallel_size=parallel_configuration[2],
        mlp_tensor_parallel_size=parallel_configuration[3],
        mlp_data_parallel_size=parallel_configuration[4],
        lmhead_tensor_parallel_size=parallel_configuration[5],
        lmhead_data_parallel_size=parallel_configuration[6],
        expert_parallel=parallel_configuration[7],
    )
    return parallel_config


class ParallelMoETestCase(unittest.TestCase):
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
            ["Qwen/Qwen3-235B-A22B", (16, 1, 16, 1, 16, 1, 16, False)],
            ["Qwen/Qwen3-235B-A22B", (16, 2, 8, 4, 4, 1, 16, False)],
            ["Qwen/Qwen3-235B-A22B", (16, 1, 16, 1, 16, 1, 16, True)],
            ["Qwen/Qwen3-235B-A22B", (16, 2, 8, 4, 4, 1, 16, True)],
            ["deepseek-ai/DeepSeek-V3.1", (16, 1, 16, 1, 16, 1, 16, True)],
            ["moonshotai/Kimi-K2-Base", (16, 1, 16, 1, 16, 1, 16, True)],
        ]
    )
    def test_model_with_ep(self, model_id, parallel_configuration):
        parallel_config = get_parallel_config(parallel_configuration)
        hf_config_json = model_id_to_json(model_id)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
            hf_config_json=hf_config_json,
            enable_lmhead=True,
            num_hidden_layers_override=6,
        )
        model = TransformerModel(model_id, model_config)

        num_tokens = 100
        output_batch_size = (
            self.input_batch_size
            * parallel_config.data_parallel_size
            // parallel_config.lmhead_data_parallel_size
        )
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(
                outputs.shape, (output_batch_size, num_tokens, model.vocab_size)
            )
        result = runtime.table_averages()
        self.assertIn("tensor_cast.permute_tokens.default", result)
        self.assertIn("tensor_cast.unpermute_tokens.default", result)
        if parallel_config.has_ep():
            self.assertIn("tensor_cast.all_to_all.default", result)
            self._check_comm_analytic(
                runtime.get_trace_events(), "tensor_cast.all_to_all.default"
            )
