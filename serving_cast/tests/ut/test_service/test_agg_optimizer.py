# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

from serving_cast.service.agg_throughput_optimizer import AggThroughputOptimizer
from serving_cast.service.utils import OptimizerData

from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile
from .test_common import SimpleArgs


class TestAggThroughputOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.strategy = AggThroughputOptimizer()
        self.args = SimpleArgs()
        self.args.model_id = "Qwen/Qwen3-32B"

        self.device_profile = DeviceProfile.all_device_profiles[self.args.device]

        self.user_input = UserInputConfig.from_args(self.args)
        self.model_runner = ModelRunner(self.user_input)
        # Initialize strategy
        self.strategy.initialize(self.model_runner)

    def test_name_attribute(self):
        """Test that name attribute is set correctly"""
        self.assertEqual(self.strategy.name, "aggregation")

    def test_get_or_compute_prefill_latency_cached(self):
        """Test _get_or_compute_prefill_latency with cached value"""
        # Set up cache with a pre-computed value
        self.strategy._prefill_cache[4] = (50.0, 2.0, "")

        optimizer_data = OptimizerData()
        latency, memory_left, _ = self.strategy._get_or_compute_latency(
            4, optimizer_data, is_decode=False
        )

        # Should return cached value
        self.assertEqual(latency, 50.0)
        self.assertEqual(memory_left, 2.0)

    def test_get_or_compute_prefill_latency_new(self):
        """Test _get_or_compute_prefill_latency with new value"""

        optimizer_data = OptimizerData(
            input_length=10,
            output_length=10,
        )
        latency, memory_left, breakdown = self.strategy._get_or_compute_latency(
            4, optimizer_data, is_decode=False
        )

        # Should cache the result
        self.assertEqual(
            self.strategy._prefill_cache[4],
            (latency, memory_left, breakdown),
        )

    def test_get_or_compute_decode_latency_cached(self):
        """Test _get_or_compute_decode_latency with cached value"""
        # Set up cache with a pre-computed value
        self.strategy._decode_cache[4] = (10.0, 2.0, "")

        optimizer_data = OptimizerData()
        latency, memory_left, _ = self.strategy._get_or_compute_latency(
            4, optimizer_data, is_decode=True
        )

        self.assertEqual(latency, 10.0)
        self.assertEqual(memory_left, 2.0)


if __name__ == "__main__":
    unittest.main()
