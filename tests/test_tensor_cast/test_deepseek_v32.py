import unittest

import torch

from tensor_cast.core.input_generator import generate_inputs, RequestInfo
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime


class TestDeepseekV32Model(unittest.TestCase):
    def test_model_init(self):
        model_id = "deepseek-ai/DeepSeek-V3.2"
        num_queries = 3500
        user_input = UserInputConfig(
            model_id=model_id,
            num_queries=1,
            query_len=num_queries,
            context_length=num_queries,
            device="TEST_DEVICE",
            num_mtp_tokens=2,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        model = build_model(user_input)
        inputs = generate_inputs(
            model,
            [
                RequestInfo(
                    query_len=num_queries,
                    seq_len=num_queries,
                    concurrency=1,
                    is_decode=True,
                )
            ],
        )
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            model.forward(**inputs)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.multihead_latent_attention_quant.default", result)
        total_time_s = runtime.total_execution_time_s()[perf_model.name]
        self.assertGreater(total_time_s, 0)
