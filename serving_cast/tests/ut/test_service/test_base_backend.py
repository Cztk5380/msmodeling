# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from serving_cast.service.base_backend import BaseBackend
from serving_cast.service.report_and_save import Summary
from serving_cast.service.utils import AGG_COLUMNS


class ConcreteBackend(BaseBackend):
    """Concrete implementation of BaseBackend for testing purposes"""

    def initialize(self, args, model):
        self.args = args
        self.model = model

    def run_inference(self, data_config):
        # Return a mock Summary object
        summary = Mock(spec=Summary)
        summary.check_stop_flag.return_value = False
        summary.get_summary_df.return_value = pd.DataFrame(columns=AGG_COLUMNS)
        return summary


class TestBaseBackend(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.backend = ConcreteBackend()
        self.mock_args = Mock()
        self.mock_model = Mock()
        self.backend.initialize(self.mock_args, self.mock_model)

        self.mock_data_config = Mock()
        self.mock_data_config.batch_size = 1
        self.mock_data_config.input_length = 128
        self.mock_data_config.output_length = 64
        self.mock_data_config.ttft_limits = 1000
        self.mock_data_config.tpot_limits = 200

    def test_name_attribute(self):
        """Test that name attribute is set correctly"""
        self.assertEqual(self.backend.name, "base")

    @patch.object(ConcreteBackend, "run_inference")
    def test_find_best_result_under_constraints_basic(self, mock_run_inference):
        """Test find_best_result_under_constraints with basic scenario"""

        # Mock the run_inference method to return different stop flags
        def side_effect(data_config):
            summary = Mock(spec=Summary)
            # Simulate the behavior: lower batch sizes don't stop, higher ones do
            if data_config.batch_size < 10:
                summary.check_stop_flag.return_value = False
            else:
                summary.check_stop_flag.return_value = True

            summary.get_summary_df.return_value = (
                pd.DataFrame(columns=AGG_COLUMNS, data=[[None] * len(AGG_COLUMNS)])
                if not summary.check_stop_flag.return_value
                else pd.DataFrame(columns=AGG_COLUMNS)
            )

            # Mock the get_summary_df to return proper data
            if not summary.check_stop_flag.return_value:
                mock_df = pd.DataFrame(
                    columns=AGG_COLUMNS,
                    data=[
                        [
                            f"model_{data_config.batch_size}",
                            128,
                            64,
                            data_config.batch_size * 2,
                            100.0,
                            50.0,
                            1,
                            "test",
                            "gpu",
                            1000.0,
                            500.0,
                            "tp1pp1dp1",
                            data_config.batch_size,
                        ]
                    ],
                )
                summary.get_summary_df.return_value = mock_df
            else:
                summary.get_summary_df.return_value = pd.DataFrame(columns=AGG_COLUMNS)

            return summary

        mock_run_inference.side_effect = side_effect

        result = self.backend.find_best_result_under_constraints(self.mock_data_config)

        # Verify that run_inference was called
        self.assertGreater(mock_run_inference.call_count, 0)
        self.assertIsNotNone(result)

    @patch.object(ConcreteBackend, "run_inference")
    def test_find_best_result_under_constraints_early_stop(self, mock_run_inference):
        """Test find_best_result_under_constraints with early stop condition"""
        # Mock to always return stop flag
        mock_summary = Mock(spec=Summary)
        mock_summary.check_stop_flag.return_value = True
        mock_run_inference.return_value = mock_summary

        result = self.backend.find_best_result_under_constraints(self.mock_data_config)

        # Should return None if early stop occurs
        self.assertIsNone(result)
        mock_run_inference.assert_called_once()

    @patch.object(ConcreteBackend, "run_inference")
    def test_find_best_result_under_constraints_no_results(self, mock_run_inference):
        """Test find_best_result_under_constraints when no valid results found"""

        # Mock to return stop flag for all calls
        def side_effect(data_config):
            summary = Mock(spec=Summary)
            summary.check_stop_flag.return_value = True
            summary.get_summary_df.return_value = pd.DataFrame(columns=AGG_COLUMNS)
            return summary

        mock_run_inference.side_effect = side_effect

        _ = self.backend.find_best_result_under_constraints(self.mock_data_config)

        mock_run_inference.assert_called()

    def test_abstract_methods_exist(self):
        """Test that abstract methods exist"""
        self.assertTrue(hasattr(BaseBackend, "initialize"))
        self.assertTrue(hasattr(BaseBackend, "run_inference"))


if __name__ == "__main__":
    unittest.main()
