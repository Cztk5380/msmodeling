# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer
from serving_cast.service.optimizer_summary import OptimizerSummary
from serving_cast.service.utils import AGG_COLUMNS


class ConcreteThroughputOptimizer(BaseThroughputOptimizer):
    """Concrete implementation of BaseThroughputOptimizer for testing purposes"""

    def initialize(self, model):
        self.model = model

    def get_inference_info(self, optimizer_data):
        # Return a mock Summary object
        summary = Mock(spec=OptimizerSummary)
        summary.check_early_stop_flag.return_value = False
        summary.get_summary_df.return_value = pd.DataFrame(columns=AGG_COLUMNS)
        return summary


class TestBaseBackend(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.backend = ConcreteThroughputOptimizer()
        self.mock_args = Mock()
        self.mock_args.batch_range = None
        self.mock_model = Mock()
        self.backend.initialize(self.mock_model)

        self.mock_data_config = Mock()
        self.mock_data_config.batch_size = 1
        self.mock_data_config.input_length = 128
        self.mock_data_config.output_length = 64
        self.mock_data_config.ttft_limits = 1000
        self.mock_data_config.tpot_limits = 200

    def test_name_attribute(self):
        """Test that name attribute is set correctly"""
        self.assertEqual(self.backend.name, "base")

    @patch.object(ConcreteThroughputOptimizer, "get_inference_info")
    def test_optimizer_basic(self, mock_get_inference_info):
        """Test optimizer with basic scenario"""

        # Mock the run_inference method to return different stop flags
        def side_effect(data_config):
            summary = Mock(spec=OptimizerSummary)
            # Simulate the behavior: lower batch sizes don't stop, higher ones do
            if data_config.batch_size < 10:
                summary.check_early_stop_flag.return_value = False
            else:
                summary.check_early_stop_flag.return_value = True

            summary.get_summary_df.return_value = (
                pd.DataFrame(columns=AGG_COLUMNS, data=[[None] * len(AGG_COLUMNS)])
                if not summary.check_early_stop_flag.return_value
                else pd.DataFrame(columns=AGG_COLUMNS)
            )

            # Mock the get_summary_df to return proper data
            if not summary.check_early_stop_flag.return_value:
                mock_df = pd.DataFrame(
                    columns=AGG_COLUMNS,
                    data=[
                        [
                            "TEST_DEVICE",
                            1,
                            f"model_{data_config.batch_size}",
                            "DISABLED",
                            "DISABLED",
                            128,
                            64,
                            data_config.batch_size * 2,
                            100.0,
                            50.0,
                            1000.0,
                            500.0,
                            "tp1pp1dp1",
                            data_config.batch_size,
                            "prefill_breakdonws",
                            "decode_breakdowns",
                        ]
                    ],
                )
                summary.get_summary_df.return_value = mock_df
            else:
                summary.get_summary_df.return_value = pd.DataFrame(columns=AGG_COLUMNS)

            return summary

        mock_get_inference_info.side_effect = side_effect

        result = self.backend.run(self.mock_data_config, [5, 20])

        # Verify that run_inference was called
        self.assertGreater(mock_get_inference_info.call_count, 0)
        self.assertIsNotNone(result)

    @patch.object(ConcreteThroughputOptimizer, "get_inference_info")
    def test_optimizer_early_stop(self, mock_get_inference_info):
        """Test optimizer with early stop condition"""
        # Mock to always return stop flag
        mock_summary = Mock(spec=OptimizerSummary)
        mock_summary.check_early_stop_flag.return_value = True
        mock_get_inference_info.return_value = mock_summary

        result = self.backend.run(self.mock_data_config, None)

        # Should return None if early stop occurs
        self.assertIsNone(result)

    @patch.object(ConcreteThroughputOptimizer, "get_inference_info")
    def test_optimizer_no_results(self, mock_get_inference_info):
        """Test optimizer when no valid results found"""

        # Mock to return stop flag for all calls
        def side_effect(data_config):
            summary = Mock(spec=OptimizerSummary)
            summary.check_early_stop_flag.return_value = True
            summary.get_summary_df.return_value = pd.DataFrame(columns=AGG_COLUMNS)
            return summary

        mock_get_inference_info.side_effect = side_effect

        _ = self.backend.run(self.mock_data_config, None)

        mock_get_inference_info.assert_called()

    def test_abstract_methods_exist(self):
        """Test that abstract methods exist"""
        self.assertTrue(hasattr(BaseThroughputOptimizer, "initialize"))
        self.assertTrue(hasattr(BaseThroughputOptimizer, "get_inference_info"))


if __name__ == "__main__":
    unittest.main()
