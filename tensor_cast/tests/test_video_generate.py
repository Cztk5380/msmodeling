import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import torch
from parameterized import parameterized

from ..scripts.video_generate import process_input, run_inference


class TestVideoGeneration(unittest.TestCase):
    """Unit tests for video_generate.py script."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "TEST_DEVICE"
        # Create temporary directory structure for mock model
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "mock_model")
        os.makedirs(self.model_dir, exist_ok=True)

        # Create transformer subdirectory and config
        transformer_dir = os.path.join(self.model_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)

        # Create a mock transformer config
        transformer_config = {
            "_class_name": "HunyuanVideoTransformer3DModel",
            "_diffusers_version": "0.32.0.dev0",
            "attention_head_dim": 128,
            "guidance_embeds": "true",
            "in_channels": 16,
            "mlp_ratio": 4.0,
            "num_attention_heads": 24,
            "num_layers": 20,
            "num_refiner_layers": 2,
            "num_single_layers": 40,
            "out_channels": 16,
            "patch_size": 2,
            "patch_size_t": 1,
            "pooled_projection_dim": 768,
            "qk_norm": "rms_norm",
            "rope_axes_dim": [16, 56, 56],
            "rope_theta": 256.0,
            "text_embed_dim": 4096,
        }

        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(transformer_config, f)

        # Create vae subdirectory and config
        vae_dir = os.path.join(self.model_dir, "vae")
        os.makedirs(vae_dir, exist_ok=True)

        # Create a mock vae config
        vae_config = {
            "_class_name": "AutoencoderKLCogVideoX",
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": [
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
            ],
            "up_block_types": [
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
            ],
            "block_out_channels": [128, 256, 512],
            "layers_per_block": 4,
            "act_fn": "silu",
            "sample_size": [16, 128, 128],
            "mid_block_type": "CogVideoXMidBlock3D",
            "norm_num_groups": 32,
            "temporal_compression_ratio": 4,
            "z_dim": 16,
        }

        with open(os.path.join(vae_dir, "config.json"), "w") as f:
            json.dump(vae_config, f)

        self.model_id = self.model_dir
        self.batch_size = 2
        self.seq_len = 10
        self.height = 400
        self.width = 832
        self.frame_num = 81
        self.sample_step = 1
        torch.compiler.reset()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _validate_inference_result(self, test_name: str = ""):
        """
        Validate the result from run_inference doesn't raise exceptions.
        Since run_inference returns None, we check for successful execution.

        Args:
            test_name: Name of the test for better error messages
        """
        # If we reach this point, the function executed successfully
        self.assertTrue(True, f"{test_name}: Inference ran without exceptions")

    def test_basic_video_inference(self):
        """Test basic video inference without Ulysses parallelism."""
        try:
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=self.sample_step,
                profiler=False,
                dtype="float16",
                world_size=1,
                ulysses_size=1,
            )
            self._validate_inference_result("test_basic_video_inference")
        except Exception as e:
            self.fail(f"test_basic_video_inference failed with exception: {str(e)}")

    def test_process_input_with_ulysses_size_1(self):
        """Test process_input function when ulysses_size is 1."""
        # Mock model_config
        mock_parallel_config = MagicMock()
        mock_parallel_config.ulysses_size = 1

        mock_transformer_config = MagicMock()
        mock_transformer_config.parallel_config = mock_parallel_config

        mock_model_config = MagicMock()
        mock_model_config.transformer_config = mock_transformer_config

        input_kwargs = {
            "hidden_states": torch.randn(2, 10, 16, 10, 25)  # Example tensor
        }

        result_kwargs, split_dim = process_input(input_kwargs, mock_model_config)

        # When ulysses_size is 1, input should remain unchanged
        self.assertEqual(result_kwargs, input_kwargs)
        self.assertIsNone(split_dim)

    @parameterized.expand(
        [
            ["float16"],
            ["float32"],
            ["bfloat16"],
        ]
    )
    def test_video_inference_with_different_dtypes_param(self, dtype):
        """Parameterized test for different data types."""
        try:
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=self.sample_step,
                profiler=False,
                dtype=dtype,
                world_size=1,
                ulysses_size=1,
            )
            self._validate_inference_result(f"test_video_inference_with_{dtype}_param")
        except Exception as e:
            self.fail(
                f"test_video_inference_with_{dtype}_param failed with exception: {str(e)}"
            )

    @parameterized.expand(
        [
            [1, 1],
            [2, 2],
            [4, 4],
            [8, 2],
        ]
    )
    def test_video_inference_with_different_parallel_sizes(
        self, world_size, ulysses_size
    ):
        """Parameterized test for different parallel configurations."""
        try:
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                height=self.height,
                width=self.width,
                frame_num=self.frame_num,
                sample_step=self.sample_step,
                profiler=False,
                dtype="float16",
                world_size=world_size,
                ulysses_size=ulysses_size,
            )
            self._validate_inference_result(
                f"test_video_inference_with_world_{world_size}_ulysses_{ulysses_size}"
            )
        except Exception as e:
            self.fail(
                f"test_video_inference_with_world_{world_size}_ulysses_{ulysses_size} failed with exception: {str(e)}"
            )


if __name__ == "__main__":
    unittest.main()
