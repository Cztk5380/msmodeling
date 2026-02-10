# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock

from serving_cast.service.agg_throughput_optimizer import AggThroughputOptimizer
from serving_cast.service.disagg_throughput_optimizer import DisaggThroughputOptimizer

from serving_cast.service.optimizer_factory import OptimizerFactory


class TestStrategyFactory(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_model = Mock()

    def test_create_aggregation_optimizer(self):
        """Test creating aggregation strategy"""
        strategy = OptimizerFactory.create_strategy(self.mock_model)

        self.assertIsInstance(strategy, AggThroughputOptimizer)
        self.assertEqual(strategy.name, "aggregation")

    def test_create_disaggregation_optimizer(self):
        """Test creating disaggregation strategy"""
        strategy = OptimizerFactory.create_strategy(self.mock_model, True)

        self.assertIsInstance(strategy, DisaggThroughputOptimizer)
        self.assertEqual(strategy.name, "disaggregation")

    def test_initialize_method_called(self):
        """Test that initialize method is called when model is provided"""
        with unittest.mock.patch.object(
            AggThroughputOptimizer, "initialize"
        ) as mock_init:
            _ = OptimizerFactory.create_strategy(self.mock_model)

            mock_init.assert_called_once_with(self.mock_model)


if __name__ == "__main__":
    unittest.main()
