import unittest

from parameterized import parameterized

from ..scripts.text_generate import run_inference

from ..scripts.utils import QuantizeAttentionAction, QuantizeLinearAction


class TestTextGenerate(unittest.TestCase):
    """Unit tests for text_generate.py script."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "TEST_DEVICE"
        self.model_id = "Qwen/Qwen3-32B"
        self.num_queries = 2
        self.query_len = 10
        self.context_length = 0

    def _validate_inference_result(self, result: dict, test_name: str = ""):
        """
        Validate the result from run_inference.

        Args:
            result: Dictionary containing inference metrics
            test_name: Name of the test for better error messages
        """
        # Check that result is a dictionary
        self.assertIsInstance(result, dict, f"{test_name}: Result should be a dict")

        # Check required keys exist
        required_keys = [
            "total_device_memory_gb",
            "model_weight_size_gb",
            "peak_memory_usage_gb",
            "kv_cache_size_gb",
            "model_activation_size_gb",
            "device_memory_available_gb",
            "execution_time_s",
            "table_result",
            "breakdowns",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"{test_name}: Missing key '{key}' in result")

        # Validate memory metrics are non-negative
        self.assertGreaterEqual(
            result["total_device_memory_gb"],
            0,
            f"{test_name}: Total device memory should be non-negative",
        )
        self.assertGreaterEqual(
            result["model_weight_size_gb"],
            0,
            f"{test_name}: Model weight size should be non-negative",
        )
        self.assertGreaterEqual(
            result["peak_memory_usage_gb"],
            0,
            f"{test_name}: Peak memory usage should be non-negative",
        )
        self.assertGreaterEqual(
            result["kv_cache_size_gb"],
            0,
            f"{test_name}: KV cache size should be non-negative",
        )
        self.assertGreaterEqual(
            result["model_activation_size_gb"],
            0,
            f"{test_name}: Model activation size should be non-negative",
        )

        # Validate memory consistency: peak = weight + kv_cache + activation
        expected_peak = (
            result["model_weight_size_gb"]
            + result["kv_cache_size_gb"]
            + result["model_activation_size_gb"]
        )
        self.assertAlmostEqual(
            result["peak_memory_usage_gb"],
            expected_peak,
            places=2,
            msg=f"{test_name}: Peak memory should equal weight + kv_cache + activation",
        )

        # Validate execution time is positive
        self.assertGreater(
            result["execution_time_s"],
            0,
            f"{test_name}: Execution time should be positive",
        )

        # Validate table result is a string
        self.assertIsInstance(
            result["table_result"], str, f"{test_name}: Table result should be a string"
        )
        self.assertGreater(
            len(result["table_result"]),
            0,
            f"{test_name}: Table result should not be empty",
        )

        # Validate breakdowns is a dictionary
        self.assertIsInstance(
            result["breakdowns"], dict, f"{test_name}: Breakdowns should be a dict"
        )

    def test_basic_prefill(self):
        """Test basic prefill operation without quantization."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=self.num_queries,
            query_len=self.query_len,
            context_length=self.context_length,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        self._validate_inference_result(result, "test_basic_prefill")

    def test_prefill_with_context(self):
        """Test prefill with context length (similar to README example)."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=100,
            context_length=200,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        self._validate_inference_result(result, "test_prefill_with_context")

    def test_prefill_with_w8a8_dynamic_quant(self):
        """Test prefill with W8A8 dynamic quantization (README example)."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=50,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
        )
        self._validate_inference_result(result, "test_prefill_with_w8a8_dynamic_quant")

    def test_decode_with_w8a8_static_quant(self):
        """Test decode with W8A8 static quantization (README example)."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=10,
            query_len=1,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            is_decode=True,
        )
        self._validate_inference_result(result, "test_decode_with_w8a8_static_quant")

    def test_decode_mode(self):
        """Test decode mode with single token input."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=5,
            query_len=1,
            context_length=50,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            is_decode=True,
        )
        self._validate_inference_result(result, "test_decode_mode")

    def test_with_compilation(self):
        """Test with torch.compile enabled."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        self._validate_inference_result(result, "test_with_compilation")

    def test_with_compilation_and_graph_break(self):
        """Test with torch.compile and allow graph break."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=True,
            allow_graph_break=True,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        self._validate_inference_result(result, "test_with_compilation_and_graph_break")

    def test_w4a8_dynamic_quantization(self):
        """Test with W4A8 dynamic quantization."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=20,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_DYNAMIC,
        )
        self._validate_inference_result(result, "test_w4a8_dynamic_quantization")

    def test_w4a8_static_quantization(self):
        """Test with W4A8 static quantization."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=20,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W4A8_STATIC,
        )
        self._validate_inference_result(result, "test_w4a8_static_quantization")

    def test_fp8_quantization(self):
        """Test with FP8 quantization."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=20,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.FP8,
        )
        self._validate_inference_result(result, "test_fp8_quantization")

    def test_fp8_with_context(self):
        """Test FP8 quantization with context length."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=50,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.FP8,
        )
        self._validate_inference_result(result, "test_fp8_with_context")
        # Should have KV cache due to context
        self.assertGreater(result["kv_cache_size_gb"], 0)

    def test_fp8_decode_mode(self):
        """Test FP8 quantization in decode mode."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=5,
            query_len=1,
            context_length=50,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.FP8,
            is_decode=True,
        )
        self._validate_inference_result(result, "test_fp8_decode_mode")

    def test_mxfp4_quantization(self):
        """Test with MXFP4 quantization."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=20,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.MXFP4,
        )
        self._validate_inference_result(result, "test_mxfp4_quantization")

    def test_mxfp4_with_context(self):
        """Test MXFP4 quantization with context length."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=50,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.MXFP4,
        )
        self._validate_inference_result(result, "test_mxfp4_with_context")
        # Should have KV cache due to context
        self.assertGreater(result["kv_cache_size_gb"], 0)

    def test_mxfp4_decode_mode(self):
        """Test MXFP4 quantization in decode mode."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=5,
            query_len=1,
            context_length=50,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.MXFP4,
            is_decode=True,
        )
        self._validate_inference_result(result, "test_mxfp4_decode_mode")

    def test_kvcache_int8_quantization(self):
        """Test with INT8 KV cache quantization."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=20,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        self._validate_inference_result(result, "test_kvcache_int8_quantization")
        # Should have KV cache due to context
        self.assertGreater(result["kv_cache_size_gb"], 0)
        self.assertIn("tensor_cast.attention_quant", result["table_result"])

    def test_kvcache_int8_with_linear_quant(self):
        """Test INT8 KV cache quantization combined with linear quantization."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=50,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        self._validate_inference_result(result, "test_kvcache_int8_with_linear_quant")
        # Should have KV cache due to context
        self.assertGreater(result["kv_cache_size_gb"], 0)
        self.assertIn("tensor_cast.attention_quant", result["table_result"])

    def test_kvcache_int8_decode_mode(self):
        """Test INT8 KV cache quantization in decode mode."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=5,
            query_len=1,
            context_length=50,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            quantize_attention_action=QuantizeAttentionAction.INT8,
            is_decode=True,
        )
        self._validate_inference_result(result, "test_kvcache_int8_decode_mode")
        self.assertIn("tensor_cast.attention_quant", result["table_result"])

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
        ]
    )
    def test_mla_int8_with_linear_quant(self, model_id):
        """Test INT8 KV cache quantization combined with linear quantization."""
        result = run_inference(
            device=self.device,
            model_id=model_id,
            num_queries=2,
            query_len=50,
            context_length=100,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            quantize_attention_action=QuantizeAttentionAction.INT8,
        )
        self._validate_inference_result(result, "test_kvcache_int8_with_linear_quant")
        # Should have KV cache due to context
        self.assertGreater(result["kv_cache_size_gb"], 0)
        self.assertIn(
            "tensor_cast.multihead_latent_attention_quant", result["table_result"]
        )

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
        ]
    )
    def test_mla_int8_decode_mode(self, model_id):
        """Test INT8 KV cache quantization in decode mode."""
        result = run_inference(
            device=self.device,
            model_id=model_id,
            num_queries=5,
            query_len=1,
            context_length=50,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            quantize_attention_action=QuantizeAttentionAction.INT8,
            is_decode=True,
        )
        self._validate_inference_result(result, "test_kvcache_int8_decode_mode")
        self.assertIn(
            "tensor_cast.multihead_latent_attention_quant", result["table_result"]
        )

    def test_with_quantized_lmhead(self):
        """Test with LM head quantization enabled."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
            quantize_lmhead=True,
        )
        self._validate_inference_result(result, "test_with_quantized_lmhead")

    def test_tensor_parallel(self):
        """Test with tensor parallelism."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=2,
            tp_size=2,
        )
        self._validate_inference_result(result, "test_tensor_parallel")

    def test_data_parallel(self):
        """Test with data parallelism."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=4,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=2,
            tp_size=1,
            dp_size=2,
        )
        self._validate_inference_result(result, "test_data_parallel")

    def test_mixed_parallelism(self):
        """Test with mixed TP and DP."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=4,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=4,
            tp_size=2,
            dp_size=2,
        )
        self._validate_inference_result(result, "test_mixed_parallelism")

    def test_with_mtp_tokens(self):
        """Test with MTP (Multi-Token Prediction) tokens."""
        # Use DeepSeek-V3.1 which supports MTP
        result = run_inference(
            device=self.device,
            model_id="deepseek-ai/DeepSeek-V3.1",
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            num_mtp_tokens=2,
        )
        self._validate_inference_result(result, "test_with_mtp_tokens")

    def test_disable_repetition(self):
        """Test with repetition disabled."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            disable_repetition=True,
        )
        self._validate_inference_result(result, "test_disable_repetition")

    def test_with_reserved_memory(self):
        """Test with reserved memory configuration."""
        reserved_gb = 5
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            reserved_memory_gb=reserved_gb,
        )
        self._validate_inference_result(result, "test_with_reserved_memory")
        # Verify reserved memory reduces available memory
        expected_available = (
            result["total_device_memory_gb"]
            - result["peak_memory_usage_gb"]
            - reserved_gb
        )
        self.assertAlmostEqual(
            result["device_memory_available_gb"], expected_available, places=2
        )

    def test_num_hidden_layers_override(self):
        """Test with overridden number of hidden layers."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            num_hidden_layers_override=2,
        )
        self._validate_inference_result(result, "test_num_hidden_layers_override")

    def test_mlp_specific_parallelism(self):
        """Test with MLP-specific tensor/data parallelism."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=4,
            tp_size=2,
            mlp_tp_size=2,
            mlp_dp_size=2,
        )
        self._validate_inference_result(result, "test_mlp_specific_parallelism")

    def test_lmhead_specific_parallelism(self):
        """Test with LM head-specific tensor/data parallelism."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=4,
            tp_size=2,
            lmhead_tp_size=2,
            lmhead_dp_size=2,
        )
        self._validate_inference_result(result, "test_lmhead_specific_parallelism")

    def test_expert_parallel(self):
        """Test with expert parallelism enabled."""
        # Use Qwen3-235B-A22B MoE model for EP testing
        result = run_inference(
            device=self.device,
            model_id="Qwen/Qwen3-235B-A22B",
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=2,
            tp_size=1,
            ep=True,
        )
        self._validate_inference_result(result, "test_expert_parallel")

    def test_invalid_device(self):
        """Test with invalid device name."""
        with self.assertRaises(ValueError):
            run_inference(
                device="INVALID_DEVICE",
                model_id=self.model_id,
                num_queries=self.num_queries,
                query_len=self.query_len,
                context_length=self.context_length,
                do_compile=False,
                allow_graph_break=False,
                quantize_linear_action=QuantizeLinearAction.DISABLED,
            )

    def test_large_batch_size(self):
        """Test with large batch size."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=32,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        self._validate_inference_result(result, "test_large_batch_size")

    def test_long_context(self):
        """Test with long context length."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=500,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        self._validate_inference_result(result, "test_long_context")

    def test_qwen3_32b_4_a3die_decode_result(self):
        """Make sure the result of qwen3-32b model on 4 A3 dies is as expected in some range"""
        result = run_inference(
            device="ATLAS_800_A3_560T_128G_DIE",
            model_id="Qwen/Qwen3-32B",
            num_queries=60,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            world_size=4,
            tp_size=4,
        )
        self._validate_inference_result(result, "qwen3_32b_4_a3die_decode")
        self.assertLess(result["execution_time_s"], 0.0328)

    def test_padding(self):
        """Test with padding tokens."""
        result = run_inference(
            device=self.device,
            model_id="Qwen/Qwen3-235B-A22B",
            num_queries=1,
            query_len=1,
            context_length=500,
            world_size=16,
            ep=True,
            tp_size=2,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )
        self._validate_inference_result(result, "test_padding")

    def test_fullmesh_subgroup_bandwidth_result(self):
        """Full Mesh with subgroup bandwidth is smaller than CLOS"""
        result_a3 = run_inference(
            device="ATLAS_800_A3_752T_128G_DIE",
            model_id="Qwen/Qwen3-32B",
            num_queries=60,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            world_size=4,
            tp_size=4,
        )
        self._validate_inference_result(result_a3)
        result_a2 = run_inference(
            device="ATLAS_800_A2_376T_64G",
            model_id="Qwen/Qwen3-32B",
            num_queries=60,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            world_size=4,
            tp_size=4,
        )
        self._validate_inference_result(result_a2)
        self.assertLess(result_a3["execution_time_s"], result_a2["execution_time_s"])

    def test_fullmesh_fullgroup_bandwidth_result(self):
        """Full Mesh with full group bandwidth is smaller than CLOS"""
        result_a3 = run_inference(
            device="ATLAS_800_A3_752T_128G_DIE",
            model_id="Qwen/Qwen3-32B",
            num_queries=60,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            world_size=8,
            tp_size=8,
        )
        self._validate_inference_result(result_a3)
        result_a2 = run_inference(
            device="ATLAS_800_A2_376T_64G",
            model_id="Qwen/Qwen3-32B",
            num_queries=60,
            query_len=1,
            context_length=4250,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            world_size=8,
            tp_size=8,
        )
        self._validate_inference_result(result_a2)
        self.assertEqual(result_a3["execution_time_s"], result_a2["execution_time_s"])

    @parameterized.expand(
        [
            [QuantizeLinearAction.W8A8_DYNAMIC],
            [QuantizeLinearAction.W8A8_STATIC],
            [QuantizeLinearAction.DISABLED],
        ]
    )
    def test_qwen2_5_with_compile(self, quant_linear_action):
        result = run_inference(
            device=self.device,
            model_id="Qwen/Qwen2.5-7B",
            num_queries=2,
            query_len=1,
            context_length=500,
            do_compile=True,
            allow_graph_break=False,
            quantize_linear_action=quant_linear_action,
        )
        self._validate_inference_result(result, "test_qwen2_5_with_compile")

    def test_o_proj_specific_parallelism(self):
        """Test with o_proj-specific tensor/data parallelism."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=4,
            tp_size=2,
            o_proj_tp_size=4,
        )
        self._validate_inference_result(result, "test_o_proj_specific_parallelism")

    def test_word_embedding_parallel(self):
        """Test with word embedding parallel."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=4,
            tp_size=2,
            word_embedding_tp=True,
        )
        self._validate_inference_result(result, "test_word_embedding_parallel")

    def test_qwen3_32b_tp16(self):
        """Make sure tp_size can be greater than num_key_value_heads."""
        result = run_inference(
            device=self.device,
            model_id=self.model_id,
            num_queries=2,
            query_len=10,
            context_length=0,
            do_compile=False,
            allow_graph_break=False,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            world_size=16,
            tp_size=16,
        )
        self._validate_inference_result(result, "qwen3_32b_tp16")


if __name__ == "__main__":
    unittest.main()
