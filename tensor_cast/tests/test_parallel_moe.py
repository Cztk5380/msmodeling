import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend
from ..core.utils import build_model, UserInputConfig
from ..device import TEST_DEVICE
from ..model_config import ModelConfig, ParallelConfig, QuantConfig
from ..performance_model.analytic import AnalyticPerformanceModel
from ..runtime import Runtime
from ..transformers.model import TransformerModel
from .test_common import create_mla_metadata_and_kv_cache


def get_parallel_config(parallel_configuration: tuple):
    parallel_config = ParallelConfig(
        world_size=parallel_configuration[0],
        tensor_parallel_size=parallel_configuration[1],
        mlp_tensor_parallel_size=parallel_configuration[2],
        lmhead_tensor_parallel_size=parallel_configuration[3],
        expert_parallel=parallel_configuration[4],
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
            ["Qwen/Qwen3-235B-A22B", (16, 1, 1, 1, False)],
            ["Qwen/Qwen3-235B-A22B", (16, 2, 4, 1, False)],
            ["Qwen/Qwen3-235B-A22B", (16, 1, 1, 1, True)],
            ["Qwen/Qwen3-235B-A22B", (16, 2, 4, 1, True)],
        ]
    )
    def test_model_with_ep(self, model_id, parallel_configuration):
        parallel_config = get_parallel_config(parallel_configuration)
        model_config = ModelConfig(
            parallel_config,
            QuantConfig(),
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
        self.assertIn("tensor_cast.permute_tokens.default", result)
        self.assertIn("tensor_cast.unpermute_tokens.default", result)
        if parallel_config.has_ep():
            self.assertIn("tensor_cast.all_to_all.default", result)
            self._check_comm_analytic(
                runtime.get_trace_events(), "tensor_cast.all_to_all.default"
            )

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, 2, 4, 1, True), (False, False)],
            # ["deepseek-ai/DeepSeek-V3.1", (16, 2, 4, 1, True), (True, False)],
            # ["deepseek-ai/DeepSeek-V3.1", (16, 2, 4, 1, True), (False, True)],
            # ["moonshotai/Kimi-K2-Base", (16, 2, 4, 1, True), (True, True)],
        ]
    )
    def test_deepseek_with_ep(
        self, model_id, parallel_configuration, moe_configuration
    ):
        user_config = UserInputConfig(
            model_id=model_id,
            world_size=parallel_configuration[0],
            tp_size=parallel_configuration[1],
            mlp_tp_size=parallel_configuration[2],
            lmhead_tp_size=parallel_configuration[3],
            ep=parallel_configuration[4],
            enable_redundant_experts=moe_configuration[0],
            enable_external_shared_experts=moe_configuration[1],
        )

        model = build_model(user_config)

        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(
            model, model.model_config
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
        self.assertIn("tensor_cast.permute_tokens.default", result)
        self.assertIn("tensor_cast.unpermute_tokens.default", result)
        if model.model_config.parallel_config.has_ep():
            self.assertIn("tensor_cast.all_to_all.default", result)
            self._check_comm_analytic(
                runtime.get_trace_events(), "tensor_cast.all_to_all.default"
            )

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (64, 2, 4, 1, True), (True, True), 8, 24],
            ["deepseek-ai/DeepSeek-V3.1", (64, 2, 4, 1, True), (False, False), 0, 0],
            ["deepseek-ai/DeepSeek-V3.1", (64, 2, 4, 1, True), (True, False), 0, 64],
            ["deepseek-ai/DeepSeek-V3.1", (64, 2, 4, 1, True), (False, True), 8, 24],
        ]
    )
    def test_deepseek_with_redundant_experts_and_external_shared_expert(
        self,
        model_id,
        parallel_configuration,
        moe_configuration,
        num_external_shared_experts,
        num_redundant_experts,
    ):
        user_config = UserInputConfig(
            model_id=model_id,
            world_size=parallel_configuration[0],
            tp_size=parallel_configuration[1],
            mlp_tp_size=parallel_configuration[2],
            lmhead_tp_size=parallel_configuration[3],
            ep=parallel_configuration[4],
            enable_redundant_experts=moe_configuration[0],
            enable_external_shared_experts=moe_configuration[1],
        )

        model = build_model(user_config)
        self.assertEqual(
            model.num_external_shared_experts,
            num_external_shared_experts,
        )
        self.assertEqual(
            model.num_redundant_experts,
            num_redundant_experts,
        )
