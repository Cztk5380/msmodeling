# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from serving_cast.service.optimizer_summary import _get_agg_table_buf, OptimizerSummary
from serving_cast.service.utils import OptimizerData


class TestSummary(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.data_config = OptimizerData()
        self.data_config.ttft_limits = 1000.0
        self.data_config.tpot_limits = 50.0
        self.summary = OptimizerSummary(self.data_config)

    def test_initialization(self):
        """Test Summary initialization"""
        self.assertIsNone(self.summary._early_stop_flag)
        self.assertIsNone(self.summary._summary_df)
        self.assertEqual(self.summary.data_config, self.data_config)

    def test_set_and_get_summary_df(self):
        """Test setting and getting summary DataFrame"""
        test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        self.summary.set_summary_df(test_df)

        retrieved_df = self.summary.get_summary_df()
        pd.testing.assert_frame_equal(retrieved_df, test_df)

    def test_set_stop_flag_memory_negative(self):
        """Test set_stop_flag when memory_left is negative"""
        self.summary.set_early_stop_flag(memory_left=-1, tpot=10.0, ttft=100.0)
        self.assertTrue(self.summary.check_early_stop_flag())

    def test_set_stop_flag_tpot_exceeds_limit(self):
        """Test set_stop_flag when tpot exceeds limit"""
        self.summary.set_early_stop_flag(
            memory_left=10, tpot=60.0, ttft=100.0
        )  # 60 > 50 (limit)
        self.assertTrue(self.summary.check_early_stop_flag())

    def test_set_stop_flag_ttft_exceeds_limit(self):
        """Test set_stop_flag when ttft exceeds limit"""
        self.summary.set_early_stop_flag(
            memory_left=10, tpot=10.0, ttft=1500.0
        )  # 1500 > 1000 (limit)
        self.assertTrue(self.summary.check_early_stop_flag())

    def test_set_stop_flag_all_within_limits(self):
        """Test set_stop_flag when all values are within limits"""
        self.summary.set_early_stop_flag(
            memory_left=10, tpot=10.0, ttft=100.0
        )  # All within limits
        self.assertFalse(self.summary.check_early_stop_flag())

    def test_check_early_stop_flag_initial_state(self):
        """Test check_early_stop_flag initial state (should be None which evaluates to False)"""
        # Initially _stop_flag is None, which should evaluate to False
        flag = self.summary.check_early_stop_flag()
        self.assertIsNone(flag)

    @patch("serving_cast.service.optimizer_summary.logger")
    def test_report_final_result_no_dataframe(self, mock_logger):
        """Test report_final_result when no DataFrame is set"""
        mock_args = MagicMock()
        self.summary.report_final_result(mock_args)
        mock_logger.warning.assert_called_once_with(
            "Summary DataFrame is None. Please set it first."
        )

    @patch("serving_cast.service.optimizer_summary.logger")
    def test_report_final_result_empty_dataframe(self, mock_logger):
        """Test report_final_result when DataFrame is empty"""
        self.summary.set_summary_df(pd.DataFrame())
        mock_args = MagicMock()
        self.summary.report_final_result(mock_args)
        mock_logger.warning.assert_called_once_with(
            "Summary DataFrame is None. Please set it first."
        )

    @patch("serving_cast.service.optimizer_summary.logger")
    @patch("serving_cast.service.optimizer_summary._get_agg_table_buf")
    def test_report_final_result_successful(self, mock_get_agg_table_buf, mock_logger):
        """Test report_final_result with valid DataFrame"""
        # Create a test DataFrame with values within limits
        test_df = pd.DataFrame(
            {
                "token/s": [100.0, 80.0, 90.0],
                "ttft": [100.0, 200.0, 150.0],
                "tpot": [20.0, 30.0, 25.0],
                "concurrency": [1, 2, 1],
                "num_devices": [1, 1, 1],
                "parallel": [1, 1, 1],
                "batch_size": [1, 2, 1],
            }
        )
        self.summary.set_summary_df(test_df)

        mock_args = MagicMock()
        mock_args.model_id = "test_model"
        mock_args.num_devices = 1
        mock_args.device = "test_device"
        mock_args.dump_original_results = False

        mock_get_agg_table_buf.return_value = "mocked table buffer"

        self.summary.report_final_result(mock_args)

        # Verify logger was called (indicating successful execution)
        mock_logger.info.assert_called_once()


class TestGetAggTableBuf(unittest.TestCase):
    def test_get_agg_table_buf_with_different_parallel_values(self):
        """Test _get_agg_table_buf with different parallel values"""
        df = pd.DataFrame(
            {
                "token/s": [100.0, 80.0, 90.0, 110.0],
                "ttft": [100.0, 200.0, 150.0, 90.0],
                "tpot": [20.0, 30.0, 25.0, 18.0],
                "concurrency": [1, 2, 1, 1],
                "num_devices": [1, 1, 1, 1],
                "parallel": [1, 2, 1, 2],  # Different parallel values
                "batch_size": [1, 2, 1, 1],
            }
        )

        result = _get_agg_table_buf(df)

        # Should group by parallel and take first of each group, then sort by token/s
        self.assertIn("Top 4 Aggregation Configurations:", result)
        self.assertIn("Throughput", result)

    def test_get_agg_table_buf_single_row(self):
        """Test _get_agg_table_buf with single row DataFrame"""
        df = pd.DataFrame(
            {
                "token/s": [100.0],
                "ttft": [100.0],
                "tpot": [20.0],
                "concurrency": [1],
                "num_devices": [1],
                "parallel": [1],
                "batch_size": [1],
            }
        )

        result = _get_agg_table_buf(df)
        self.assertIn("Top 1 Aggregation Configurations:", result)
        self.assertIn("1", result)  # Top rank
        self.assertIn("100.00", result)  # Throughput value


if __name__ == "__main__":
    unittest.main()
