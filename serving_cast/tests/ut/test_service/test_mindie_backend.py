# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

from serving_cast.service.mindie_backend import MindIEAggBackend
from serving_cast.service.utils import DataConfig


class TestMindIEAggBackend(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.backend = MindIEAggBackend()
        self.mock_args = Mock()
        self.mock_args.model_id = "test_model"
        self.mock_args.device = "TEST_DEVICE"
        self.mock_args.num_devices = 1
        self.mock_args.mtp_acceptance_rate = [0.8, 0.7, 0.6]
        self.mock_args.num_mtp_tokens = 3

        # Mock model and its components
        self.mock_model = Mock()
        self.mock_model_config = Mock()
        self.mock_parallel_config = Mock()
        self.mock_mtp_config = Mock()

        self.mock_parallel_config.data_parallel_size = 1
        self.mock_parallel_config.tensor_parallel_size = 1
        self.mock_parallel_config.pipeline_parallel_size = 1
        self.mock_mtp_config.num_mtp_layers = 3

        self.mock_model.model_config.parallel_config = self.mock_parallel_config
        self.mock_model.model_config.mtp_config = self.mock_mtp_config
        self.mock_model.model_config = self.mock_model_config

        # Initialize backend
        self.backend.initialize(self.mock_args, self.mock_model)

    def test_name_attribute(self):
        """Test that name attribute is set correctly"""
        self.assertEqual(self.backend.name, "mindie_aggregation")

    @patch("serving_cast.service.mindie_backend.generate_inputs")
    @patch("serving_cast.service.mindie_backend.run_static")
    def test_get_prefill_forward(self, mock_run_static, mock_generate_inputs):
        """Test _get_prefill_forward method"""
        mock_device_profile = Mock()
        data_config = DataConfig(input_length=512, device_profile=mock_device_profile)

        mock_inputs = {"kv_cache_by_layers": {}}
        mock_generate_inputs.return_value = mock_inputs
        mock_run_static.return_value = {
            "execution_time_s": 50.0,
            "device_memory_available_gb": 2.0,
        }

        _ = self.backend._get_prefill_forward(4, data_config)

        # Verify the call to generate_inputs and run_static
        mock_generate_inputs.assert_called()
        mock_run_static.assert_called_with(
            self.mock_model, mock_inputs, mock_device_profile
        )

    @patch.object(MindIEAggBackend, "_get_prefill_forward")
    def test_get_or_compute_prefill_latency_cached(self, mock_get_prefill_forward):
        """Test _get_or_compute_prefill_latency with cached value"""
        # Set up cache with a pre-computed value
        self.backend._prefill_cache[4] = (50.0, 2.0)

        data_config = DataConfig(device_profile=Mock())
        latency, memory_left = self.backend._get_or_compute_prefill_latency(
            4, data_config
        )

        # Should return cached value without calling _get_prefill_forward
        self.assertEqual(latency, 50.0)
        self.assertEqual(memory_left, 2.0)
        mock_get_prefill_forward.assert_not_called()

    @patch.object(MindIEAggBackend, "_get_prefill_forward")
    def test_get_or_compute_prefill_latency_new(self, mock_get_prefill_forward):
        """Test _get_or_compute_prefill_latency with new value"""
        # Mock the forward method to return values
        mock_get_prefill_forward.return_value = {
            "execution_time_s": 50.0,
            "device_memory_available_gb": 2.0,
        }

        data_config = DataConfig(device_profile=Mock())
        latency, memory_left = self.backend._get_or_compute_prefill_latency(
            4, data_config
        )

        # Should call _get_prefill_forward and cache the result
        mock_get_prefill_forward.assert_called_once()
        self.assertEqual(latency, 50.0)
        self.assertEqual(memory_left, 2.0)
        self.assertEqual(self.backend._prefill_cache[4], (50.0, 2.0))

    @patch.object(MindIEAggBackend, "_get_decode_forward")
    def test_get_or_compute_decode_latency_cached(self, mock_get_decode_forward):
        """Test _get_or_compute_decode_latency with cached value"""
        # Set up cache with a pre-computed value
        self.backend._decode_cache[4] = (10.0, 2.0)

        data_config = DataConfig(device_profile=Mock())
        latency, memory_left = self.backend._get_or_compute_decode_latency(
            4, data_config
        )

        # Should return cached value without calling _get_decode_forward
        self.assertEqual(latency, 10.0)
        self.assertEqual(memory_left, 2.0)
        mock_get_decode_forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
