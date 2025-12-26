import unittest
from unittest.mock import MagicMock, patch, call
from parameterized import parameterized

import torch


class TestVideoGenerateCfgLogicOnly(unittest.TestCase):
    """Unit test class for testing use_cfg/cfg_parallel logic branches only (full Mock)"""

    @parameterized.expand([
        # Test combinations: (use_cfg, cfg_parallel, world_size, expected extra forward, expected cfg all_gather, test description)
        (False, False, 1, False, False, "CFG disabled + parallel disabled → no extra operations"),
        (True, False, 1, True, False, "CFG enabled + parallel disabled → execute extra forward"),
        (True, True, 2, False, True, "CFG enabled + parallel enabled → execute cfg all_gather"),
        (False, True, 2, False, False, "CFG disabled + parallel enabled → no extra operations"),
    ])
    @patch("tensor_cast.scripts.video_generate.process_input")
    @patch("tensor_cast.scripts.video_generate.generate_diffusers_inputs")
    @patch("tensor_cast.scripts.video_generate.set_sp_group")
    @patch("tensor_cast.scripts.video_generate.build_diffusers_transformer_model")
    @patch("tensor_cast.scripts.video_generate.Runtime")
    @patch("tensor_cast.scripts.video_generate.ParallelGroup")
    @patch("tensor_cast.scripts.video_generate.AnalyticPerformanceModel")
    @patch("tensor_cast.scripts.video_generate.DeviceProfile")
    def test_cfg_logic_branches(
        self,
        use_cfg,
        cfg_parallel,
        world_size,
        expect_extra_forward,
        expect_cfg_all_gather,
        test_desc,
        mock_device_profile,
        mock_analytic_perf,
        mock_parallel_group,
        mock_runtime,
        mock_build_model,
        mock_set_sp_group,
        mock_gen_inputs,
        mock_process_input
    ):
        """Core test: verify that logic branches corresponding to use_cfg/cfg_parallel are executed"""
        # ------------------------------ 1. Mock all external dependencies ------------------------------
        # Mock DeviceProfile
        mock_device_profile.all_device_profiles = {"TEST_DEVICE": MagicMock()}
        
        # Mock model construction (return fake model + fake config)
        mock_model = MagicMock()
        mock_model.forward.return_value = torch.randn(1, 16, 10, 10)  # Fake forward output
        mock_model.sp_group = MagicMock()  # Fake sp_group
        mock_model_config = MagicMock()
        mock_build_model.return_value = (mock_model, mock_model_config)
        
        # Mock Runtime (context manager)
        mock_runtime_instance = MagicMock()
        mock_runtime.return_value.__enter__.return_value = mock_runtime_instance
        
        # Mock parallel group instance
        mock_cfg_parallel_group = MagicMock()
        mock_parallel_group.return_value = mock_cfg_parallel_group
        
        # Mock input generation/processing (return fake data)
        mock_gen_inputs.return_value = {"hidden_states": MagicMock()}
        mock_process_input.return_value = ({"hidden_states": MagicMock()}, None)

        # ------------------------------ 2. Import and execute run_inference ------------------------------
        from tensor_cast.scripts.video_generate import run_inference
        
        try:
            run_inference(
                device="TEST_DEVICE",
                model_id="fake_model_id",  # Completely fake model ID, no actual meaning
                batch_size=2,
                seq_len=10,
                height=400,
                width=832,
                frame_num=81,
                sample_step=1,
                profiler=False,
                dtype="float16",
                world_size=world_size,
                ulysses_size=1,
                use_cfg=use_cfg,
                cfg_parallel=cfg_parallel,
            )
        except Exception as e:
            self.fail(f"Test failed [{test_desc}]: {str(e)}")

        # ------------------------------ 3. Verify core logic branches ------------------------------
        # Verification 1: Number of forward calls (core logic)
        expected_forward_calls = 1 * (2 if expect_extra_forward else 1)
        self.assertEqual(
            mock_model.forward.call_count, expected_forward_calls,
            f"[{test_desc}] Incorrect forward call count: expected {expected_forward_calls} times, actual {mock_model.forward.call_count} times"
        )

        # Verification 2: cfg_parallel all_gather call (core logic)
        if expect_cfg_all_gather:
            mock_cfg_parallel_group.all_gather.assert_called_once()
        else:
            mock_cfg_parallel_group.all_gather.assert_not_called()

        # Verification 3: ParallelGroup initialization logic (auxiliary verification)
        if use_cfg and cfg_parallel and world_size == 2:
            mock_parallel_group.assert_called()  # Initialize parallel group
        elif world_size != 2 and cfg_parallel:
            mock_parallel_group.assert_not_called()  # Do not initialize when world_size≠2

        # Verification 4: Test passed marker
        self.assertTrue(True, f"Test passed [{test_desc}]: core logic branches executed correctly")


if __name__ == "__main__":
    unittest.main(verbosity=2)