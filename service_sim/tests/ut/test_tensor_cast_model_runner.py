# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from typing import List
from unittest.mock import Mock, patch, MagicMock

from tensor_cast.interface.model_runner import (
    ModelRunner,
    ModelRunnerMetrics,
    RequestInfo,
)


class TestModelRunner(unittest.TestCase):
    def test_init_valid_device(self):
        runner = ModelRunner(
            device="TEST_DEVICE", 
            model_id="Qwen/Qwen3-32B",
            world_size=1,
            tp_size=1,
        )
        self.assertIsNotNone(runner.model)
        self.assertEqual(runner.parallel_config.world_size, 1)
        self.assertEqual(runner.parallel_config.tensor_parallel_size, 1)
        self.assertIsNotNone(runner.quant_config)

    def test_init_invalid_device(self):
        with self.assertRaises(ValueError):
            ModelRunner(
                device="invalid-device",
                model_id="test-model",
                world_size=1,
                tp_size=1,
            )


    def test_run_inference_basic(self):
        mock_requests: List[RequestInfo] = [
            RequestInfo(query_len=10, seq_len=10, is_decode=False),
            RequestInfo(query_len=1, seq_len=10, is_decode=True)]

        runner = ModelRunner(
            device="TEST_DEVICE",
            model_id="Qwen/Qwen3-32B",
        )

        metrics = runner.run_inference(mock_requests)
        self.assertIsNotNone(metrics)


    def test_run_inference_with_ep(self):
        model_runner = ModelRunner(
            device="ATLAS_800_A2_280T_64G",
            model_id="deepseek-ai/DeepSeek-V3.1",
            quantize_linear_action="FP8",
            quantize_attention_action="INT8",
            world_size=8,
            tp_size=8,
            dp_size=1,
            ep=True,
        )
        requests = [RequestInfo(1, 65, True)]
        metrics = model_runner.run_inference(requests)
        self.assertIsNotNone(metrics)


if __name__ == "__main__":
    unittest.main()