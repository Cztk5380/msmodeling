# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

from cli.inference.throughput_optimizer import ParallelRunner

from serving_cast.service.optimizer_summary import OptimizerSummary

from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile


class SimpleArgs:
    def __init__(self):
        self.model_id = "Qwen/Qwen3-8B"
        self.device = "TEST_DEVICE"
        self.compile = True
        self.compile_allow_graph_break = False
        self.num_mtp_tokens = 0
        self.mtp_acceptance_rate = [0.9, 0.8]
        self.quantize_linear_action = "DISABLED"
        self.mxfp4_group_size = 128
        self.quantize_attention_action = "DISABLED"
        self.backend = "mindie"
        self.max_prefill_tokens = 2048
        self.input_length = 512
        self.output_length = 128
        self.ttft_limits = 1000
        self.tpot_limits = 100
        self.disagg = False
        self.tp_sizes = None
        self.num_devices = 1
        self.batch_range = None


class TestTaskRunner(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.args = SimpleArgs()
        self.args.serving_cost = 0
        self.args.jobs = 4
        self.device_profile = DeviceProfile.all_device_profiles[self.args.device]

    def test_initialization(self):
        """Test TaskRunner initialization"""
        task_runner = ParallelRunner(self.args)
        user_input = UserInputConfig.from_args(self.args)

        self.assertEqual(task_runner.user_input, user_input)

    def test_get_user_config_multiple_tp(self):
        """Test _get_user_config with multiple TP values"""
        self.args.tp_sizes = [2, 4]
        self.args.num_devices = 4

        task_runner = ParallelRunner(self.args)

        configs = list(task_runner._get_user_config())
        # Should only include TPs that divide evenly into num_devices
        self.assertEqual(len(configs), 2)  # TP=2, 4 all divide 4
        tps = [config.tp_size for config in configs]
        self.assertIn(2, tps)
        self.assertIn(4, tps)

    def test_get_user_config_default_tps(self):
        """Test _get_user_config with default TP values"""
        self.args.tp_sizes = None
        self.args.num_devices = 8

        task_runner = ParallelRunner(self.args)

        configs = list(task_runner._get_user_config())
        # Default TPs should be powers of 2 up to num_devices
        expected_tps = [1, 2, 4, 8]  # 2^0 to 2^3
        actual_tps = [config.tp_size for config in configs]
        for expected_tp in expected_tps:
            self.assertIn(expected_tp, actual_tps)

    def test_run_with_tpot_limit(self):
        """Test run method with TPOT limit"""
        self.args.tpot_limits = 50
        self.args.batch_range = [2, 2]
        task_runner = ParallelRunner(self.args)
        result = task_runner.run_agg()

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], OptimizerSummary)

        summary_df = result[0].get_summary_df()
        row = summary_df.iloc[0]
        self.assertEqual(row["concurrency"], 2)


if __name__ == "__main__":
    unittest.main()
