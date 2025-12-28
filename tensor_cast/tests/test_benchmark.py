import copy
import unittest

from ..core.utils import (
    build_model,
    create_quant_config,
    QuantizeAttentionAction,
    QuantizeLinearAction,
    UserInputConfig,
)
from ..device import DeviceProfile
from ..scripts.benchmark import find_best_throughput, get_benchmark_query_and_seq_length
from ..transformers.utils import get_moe_config
from .test_common import update_parallel_parameter


class TestBenchmark(unittest.TestCase):
    """Unit tests for benchmark.py script."""

    def setUp(self):
        """Set up test fixtures."""
        self.device_profile = DeviceProfile.all_device_profiles["TEST_DEVICE"]
        self.model_id = "Qwen/Qwen3-32B"
        self.input_length = 100
        self.output_length = 50
        self.default_config_dict = dict(
            model_id=self.model_id,
            quantize_attention_action=QuantizeLinearAction.DISABLED,
            num_mtp_tokens=0,
            do_compile=False,
        )
        self.user_input = UserInputConfig(**self.default_config_dict)
        update_parallel_parameter(self.user_input, world_size=1, tp_size=1, ep=False)

    def test_get_benchmark_query_and_seq_length_decode(self):
        """Test query and sequence length calculation for decode mode."""
        query_len, seq_len = get_benchmark_query_and_seq_length(
            input_length=100,
            output_length=50,
            is_decode=True,
            num_mtp_tokens=0,
            context_length=0,
        )
        # For decode: query_len = num_mtp_tokens + 1 = 0 + 1 = 1
        # seq_len = input_length + output_length // 2 + context_length + query_len
        #          = 100 + 25 + 0 + 1 = 126
        self.assertEqual(query_len, 1)
        self.assertEqual(seq_len, 126)

    def test_get_benchmark_query_and_seq_length_prefill(self):
        """Test query and sequence length calculation for prefill mode."""
        query_len, seq_len = get_benchmark_query_and_seq_length(
            input_length=100,
            output_length=50,
            is_decode=False,
            num_mtp_tokens=0,
            context_length=200,
        )
        # For prefill: query_len = input_length = 100
        # seq_len = context_length + query_len = 200 + 100 = 300
        self.assertEqual(query_len, 100)
        self.assertEqual(seq_len, 300)

    def test_get_benchmark_query_and_seq_length_with_mtp(self):
        """Test query and sequence length calculation with MTP tokens."""
        query_len, seq_len = get_benchmark_query_and_seq_length(
            input_length=100,
            output_length=50,
            is_decode=True,
            num_mtp_tokens=3,
            context_length=0,
        )
        # For decode with MTP: query_len = num_mtp_tokens + 1 = 3 + 1 = 4
        self.assertEqual(query_len, 4)

    def test_find_best_throughput_decode_basic(self):
        """Test finding the best throughput for decode mode."""
        model = build_model(self.user_input)

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,  # 100ms TPOT limit
            is_decode=True,
            reserved_memory_size_gb=1,
        )

        # Should return valid results
        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)
        self.assertIsInstance(breakdown, dict)

    def test_find_best_throughput_prefill_basic(self):
        """Test finding the best throughput for prefill mode."""
        model = build_model(self.user_input)

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=1.0,  # 1s TTFT limit
            is_decode=False,
            reserved_memory_size_gb=1,
        )

        # Should return valid results
        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)
        self.assertIsInstance(breakdown, dict)

    def test_find_best_throughput_with_w8a8_quant(self):
        """Test finding the best throughput with W8A8 quantization."""
        user_config = copy.deepcopy(self.default_config_dict)
        user_config.update(
            {"quantize_linear_action": QuantizeLinearAction.W8A8_DYNAMIC}
        )
        model = build_model(UserInputConfig(**user_config))

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)

    def test_find_best_throughput_with_w4a8_quant(self):
        """Test finding the best throughput with W4A8 quantization."""
        user_config = copy.deepcopy(self.default_config_dict)
        user_config.update(
            {"quantize_linear_action": QuantizeLinearAction.W4A8_DYNAMIC}
        )
        model = build_model(UserInputConfig(**user_config))
        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)

    def test_find_best_throughput_with_fp8_quant(self):
        """Test finding the best throughput with FP8 quantization."""
        user_config = copy.deepcopy(self.default_config_dict)
        user_config.update({"quantize_linear_action": QuantizeLinearAction.FP8})
        model = build_model(UserInputConfig(**user_config))
        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)
        self.assertIsInstance(breakdown, dict)

    def test_find_best_throughput_fp8_prefill(self):
        """Test finding the best throughput with FP8 quantization in prefill mode."""
        user_config = copy.deepcopy(self.default_config_dict)
        user_config.update({"quantize_linear_action": QuantizeLinearAction.FP8})
        model = build_model(UserInputConfig(**user_config))

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=1.0,  # 1s TTFT limit
            is_decode=False,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)
        self.assertIsInstance(breakdown, dict)

    def test_find_best_throughput_with_mxfp4_quant(self):
        """Test finding the best throughput with MXFP4 quantization."""

        user_config = copy.deepcopy(self.default_config_dict)
        user_config.update(
            {
                "quantize_linear_action": QuantizeLinearAction.MXFP4,
                "mxfp4_group_size": 32,
            }
        )
        model = build_model(UserInputConfig(**user_config))

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)
        self.assertIsInstance(breakdown, dict)

    def test_find_best_throughput_mxfp4_prefill(self):
        """Test finding the best throughput with MXFP4 quantization in prefill mode."""
        user_config = copy.deepcopy(self.default_config_dict)
        user_config.update(
            {
                "quantize_linear_action": QuantizeLinearAction.MXFP4,
                "mxfp4_group_size": 32,
            }
        )
        model = build_model(UserInputConfig(**user_config))

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=1.0,  # 1s TTFT limit
            is_decode=False,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)
        self.assertIsInstance(breakdown, dict)

    def test_find_best_throughput_with_kvcache_int8(self):
        """Test finding the best throughput with INT8 KV cache quantization."""
        user_config = copy.deepcopy(self.default_config_dict)
        user_config.update({"quantize_attention_action": QuantizeAttentionAction.INT8})
        model = build_model(UserInputConfig(**user_config))

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)
        self.assertIsInstance(breakdown, dict)

    def test_find_best_throughput_kvcache_int8_with_linear_quant(self):
        """Test finding the best throughput with INT8 KV cache and linear quantization."""

        user_config = copy.deepcopy(self.default_config_dict)
        user_config.update(
            {
                "quantize_attention_action": QuantizeAttentionAction.INT8,
                "quantize_linear_action": QuantizeLinearAction.W8A8_DYNAMIC,
            }
        )
        model = build_model(UserInputConfig(**user_config))
        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)
        self.assertIsInstance(breakdown, dict)

    def test_find_best_throughput_kvcache_int8_prefill(self):
        """Test finding the best throughput with INT8 KV cache in prefill mode."""
        user_config = copy.deepcopy(self.default_config_dict)
        user_config.update(
            {
                "quantize_attention_action": QuantizeAttentionAction.INT8,
                "quantize_linear_action": QuantizeLinearAction.W8A8_DYNAMIC,
            }
        )
        model = build_model(UserInputConfig(**user_config))

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=1.0,  # 1s TTFT limit
            is_decode=False,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)
        self.assertIsInstance(breakdown, dict)

    def test_find_best_throughput_with_tp(self):
        """Test finding the best throughput with tensor parallelism."""
        user_config = copy.deepcopy(self.user_input)
        update_parallel_parameter(user_config, world_size=2, tp_size=2, ep=False)
        model = build_model(user_config)

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        # Concurrency should be at least equal to DP size
        if model.model_config.parallel_config.data_parallel_size is not None:
            self.assertGreaterEqual(
                concurrency, model.model_config.parallel_config.data_parallel_size
            )

    def test_find_best_throughput_with_mtp(self):
        """Test finding the best throughput with MTP tokens."""
        # Use DeepSeek-V3.1 which supports MTP
        user_config = copy.deepcopy(self.default_config_dict)
        user_config.update(
            {"model_id": "deepseek-ai/DeepSeek-V3.1", "num_mtp_tokens": 2}
        )
        model = build_model(UserInputConfig(**user_config))

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            mtp_acceptance_rate=[0.9, 0.6],
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)

    def test_find_best_throughput_strict_slo(self):
        """Test with very strict SLO limit (should fail or return low concurrency)."""
        model = build_model(self.user_input)

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.001,  # Very strict 1ms limit
            is_decode=True,
            reserved_memory_size_gb=1,
        )

        # May return error or very low concurrency
        if error_msg:
            self.assertIn("exceeds limit", error_msg)

    def test_find_best_throughput_large_reserved_memory(self):
        """Test with large reserved memory (should hit OOM or return low concurrency)."""
        model = build_model(self.user_input)

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            reserved_memory_size_gb=60,  # Very large reserved memory
        )

        # May return OOM error or very low concurrency
        if error_msg:
            self.assertTrue("OOM" in error_msg or "exceeds limit" in error_msg)

    def test_find_best_throughput_custom_serving_overhead(self):
        """Test with custom serving overhead."""
        model = build_model(self.user_input)

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            serving_overhead_s=0.005,  # 5ms serving overhead
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        self.assertGreater(concurrency, 0)

    def test_find_best_throughput_long_sequence(self):
        """Test with long input/output sequences."""
        model = build_model(self.user_input)

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=1000,
            output_length=500,
            slo_limit=1.0,
            is_decode=False,
            reserved_memory_size_gb=1,
        )

        self.assertGreater(latency, 0)
        # May have lower concurrency due to memory constraints
        self.assertGreater(concurrency, 0)

    def test_find_best_throughput_breakdown_contents(self):
        """Test that breakdown contains expected keys."""
        model = build_model(self.user_input)

        latency, concurrency, breakdown, error_msg = find_best_throughput(
            model=model,
            device_profile=self.device_profile,
            input_length=self.input_length,
            output_length=self.output_length,
            slo_limit=0.1,
            is_decode=True,
            reserved_memory_size_gb=1,
        )

        # Breakdown should contain performance metrics
        self.assertIsInstance(breakdown, dict)
        # The sum of breakdown values should represent total execution time
        if breakdown:
            self.assertGreater(sum(breakdown.values()), 0)

    def test_model_id_to_moe_config(self):
        """Test model ID to MoE config mapping."""
        # Test with a standard model (should return None)
        moe_config = get_moe_config("gpt2")
        self.assertIsNone(moe_config)

        # Add more specific tests if MoE models are available
        # moe_config = model_id_to_moe_config("deepseek-ai/DeepSeek-V3")
        # self.assertIsNotNone(moe_config)

    def test_parallel_config_creation(self):
        """Test parallel configuration creation."""
        user_config = copy.deepcopy(self.user_input)

        # Test basic TP
        update_parallel_parameter(user_config, world_size=2, tp_size=2, ep=False)
        self.assertEqual(user_config.get_parallel_config().tensor_parallel_size, 2)
        self.assertEqual(user_config.get_parallel_config().data_parallel_size, 1)

        # Test TP + DP
        update_parallel_parameter(user_config, world_size=4, tp_size=2, ep=False)
        self.assertEqual(user_config.get_parallel_config().tensor_parallel_size, 2)
        self.assertEqual(user_config.get_parallel_config().data_parallel_size, 2)

        # Test EP
        update_parallel_parameter(user_config, world_size=4, tp_size=2, ep=True)
        self.assertEqual(user_config.get_parallel_config().expert_parallel, True)

    def test_quant_config_creation(self):
        """Test quantization configuration creation."""
        # Test W8A8 dynamic
        config = create_quant_config(QuantizeLinearAction.W8A8_DYNAMIC)
        self.assertIsNotNone(config)

        # Test W4A8 dynamic
        config = create_quant_config(QuantizeLinearAction.W4A8_DYNAMIC)
        self.assertIsNotNone(config)

        # Test disabled
        config = create_quant_config(QuantizeLinearAction.DISABLED)
        self.assertIsNotNone(config)

    def test_benchmark_with_different_devices(self):
        """Test benchmark with different device types."""
        for device_name in ["TEST_DEVICE"]:  # Add more devices if available
            device_profile = DeviceProfile.all_device_profiles[device_name]
            user_config = copy.deepcopy(self.user_input)
            user_config.device = device_name
            model = build_model(user_config)

            latency, concurrency, breakdown, error_msg = find_best_throughput(
                model=model,
                device_profile=device_profile,
                input_length=50,
                output_length=25,
                slo_limit=0.1,
                is_decode=True,
                reserved_memory_size_gb=1,
            )

            self.assertGreater(latency, 0)
            self.assertGreater(concurrency, 0)

    def test_benchmark_various_input_lengths(self):
        """Test benchmark with various input lengths."""
        model = build_model(self.user_input)

        for input_len in [10, 50, 100, 200]:
            latency, concurrency, breakdown, error_msg = find_best_throughput(
                model=model,
                device_profile=self.device_profile,
                input_length=input_len,
                output_length=50,
                slo_limit=0.5,
                is_decode=False,
                reserved_memory_size_gb=1,
            )

            self.assertGreater(latency, 0, f"Failed for input_len={input_len}")
            self.assertGreater(concurrency, 0, f"Failed for input_len={input_len}")


if __name__ == "__main__":
    unittest.main()
