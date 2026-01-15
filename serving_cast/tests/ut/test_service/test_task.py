# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

from serving_cast.service.task import TaskRunner


class TestTaskRunner(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_args = Mock()
        self.mock_args.model_id = "Qwen/Qwen3-8B"
        self.mock_args.device = "TEST_DEVICE"
        self.mock_args.compile = True
        self.mock_args.compile_allow_graph_break = False
        self.mock_args.num_mtp_tokens = 0
        self.mock_args.quantize_linear_action = "DISABLED"
        self.mock_args.mxfp4_group_size = 128
        self.mock_args.quantize_attention_action = "DISABLED"
        self.mock_args.backend = "mindie"
        self.mock_args.num_devices = 1
        self.mock_args.max_prefill_tokens = 2048
        self.mock_args.input_length = 512
        self.mock_args.output_length = 128
        self.mock_args.ttft_limits = 1000
        self.mock_args.tpot_limits = [100]
        self.mock_args.disaggregation = False

        self.task_runner = TaskRunner(self.mock_args)

    @patch("serving_cast.service.task.DeviceProfile")
    def test_initialization(self, mock_device_profile):
        """Test TaskRunner initialization"""
        mock_device_profile.all_device_profiles = {"TEST_DEVICE": Mock()}

        task_runner = TaskRunner(self.mock_args)

        self.assertEqual(task_runner.model_id, "Qwen/Qwen3-8B")
        self.assertEqual(task_runner.device, "TEST_DEVICE")
        self.assertEqual(task_runner.num_devices, 1)

    def test_get_user_config_multiple_tp(self):
        """Test _get_user_config with multiple TP values"""
        self.mock_args.tp = [1, 2, 4]
        self.mock_args.num_devices = 4
        with patch(
            "serving_cast.service.task.UserInputConfig"
        ) as mock_user_input_config:
            mock_user_input_config.from_args.return_value = Mock()
            task_runner = TaskRunner(self.mock_args)

        configs = list(task_runner._get_user_config())
        # Should only include TPs that divide evenly into num_devices
        self.assertEqual(len(configs), 3)  # TP=1, 2, 4 all divide 4
        tps = [config.tp_size for config in configs]
        self.assertIn(1, tps)
        self.assertIn(2, tps)
        self.assertIn(4, tps)

    def test_get_user_config_default_tps(self):
        """Test _get_user_config with default TP values"""
        self.mock_args.tp = None
        self.mock_args.num_devices = 8
        with patch(
            "serving_cast.service.task.UserInputConfig"
        ) as mock_user_input_config:
            mock_user_input_config.from_args.return_value = Mock()
            task_runner = TaskRunner(self.mock_args)

        configs = list(task_runner._get_user_config())
        # Default TPs should be powers of 2 up to num_devices
        expected_tps = [1, 2, 4, 8]  # 2^0 to 2^3
        actual_tps = [config.tp_size for config in configs]
        for expected_tp in expected_tps:
            self.assertIn(expected_tp, actual_tps)

    @patch("serving_cast.service.task.build_model")
    @patch("serving_cast.service.task.logger")
    def test_get_model_failure(self, mock_logger, mock_build_model):
        """Test _get_model handles failure gracefully"""
        mock_build_model.side_effect = Exception("Build failed")

        model = self.task_runner._get_model(Mock())
        self.assertIsNone(model)
        mock_logger.error.assert_called_once()

    def test_run_when_device_not_supported(self):
        """Test run method when device doesn't support requested number of devices"""
        mock_device_profile = Mock()
        mock_device_profile.comm_grid.grid = Mock()
        mock_device_profile.comm_grid.grid.nelement.return_value = 1
        self.task_runner.num_devices = 8
        self.task_runner.device_profile = mock_device_profile

        result = self.task_runner.run()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
