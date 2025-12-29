# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from typing import List

from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.utils import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
    RequestInfo,
    UserInputConfig,
)


class TestModelRunner(unittest.TestCase):
    def test_init_valid_device(self):
        runner = ModelRunner(
            UserInputConfig(
                device="TEST_DEVICE",
                model_id="Qwen/Qwen3-32B",
                world_size=1,
                tp_size=1,
            )
        )
        self.assertIsNotNone(runner.model)
        self.assertEqual(runner.model.model_config.parallel_config.world_size, 1)
        self.assertEqual(
            runner.model.model_config.parallel_config.tensor_parallel_size, 1
        )
        self.assertIsNotNone(runner.model.model_config.quant_config)

    def test_init_invalid_device(self):
        with self.assertRaises(ValueError):
            ModelRunner(
                UserInputConfig(
                    device="invalid-device",
                    model_id="test-model",
                    world_size=1,
                    tp_size=1,
                )
            )

    def test_run_inference_basic(self):
        mock_requests: List[RequestInfo] = [
            RequestInfo(query_len=10, seq_len=10, is_decode=False),
            RequestInfo(query_len=1, seq_len=10, is_decode=True),
        ]

        runner = ModelRunner(
            UserInputConfig(
                device="TEST_DEVICE",
                model_id="Qwen/Qwen3-32B",
            )
        )

        metrics = runner.run_inference(mock_requests)
        self.assertIsNotNone(metrics)

    def test_run_inference_with_ep(self):
        model_runner = ModelRunner(
            UserInputConfig(
                device="TEST_DEVICE",
                model_id="deepseek-ai/DeepSeek-V3.1",
                quantize_linear_action=QuantizeLinearAction.FP8,
                quantize_attention_action=QuantizeAttentionAction.INT8,
                world_size=8,
                tp_size=8,
                dp_size=1,
                ep=True,
            )
        )
        requests = [RequestInfo(1, 65, True)]
        metrics = model_runner.run_inference(requests)
        self.assertIsNotNone(metrics)


if __name__ == "__main__":
    unittest.main()
