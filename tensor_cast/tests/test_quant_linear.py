import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend
from ..device import TEST_DEVICE
from ..layers.mla import MultiheadLatentAttentionTensorCast

from ..layers.quant_linear import QuantLinearBase, TensorCastQuantLinear
from ..layers.sampler import SamplingMetadata
from ..model_config import (
    LinearQuantConfig,
    LinearQuantType,
    MlaConfig,
    ModelConfig,
    MtpConfig,
    ParallelConfig,
    QuantConfig,
    QuantGranularity,
    QuantScheme,
)
from ..patch_torch import patch_torch
from ..performance_model.analytic import AnalyticPerformanceModel
from ..runtime import Runtime
from ..transformers.model import TransformerModel
from ..transformers.utils import (
    model_id_to_json,
    model_id_to_mtp_block_module_name,
    strip_module_name,
)

from .test_common import create_mla_metadata_and_kv_cache, has_submodule_with_cls_name

# Define common parameters for tests
IN_FEATURES = 32
OUT_FEATURES = 64
BATCH_SIZE = 4
MODEL_DTYPE = torch.bfloat16
DEVICE = "cpu"


def get_linear_quant_config(quant_type, weight, **kwargs):
    """Helper to create a default symmetric per-tensor weight quant config."""
    # Per-tensor symmetric quantization for weight
    w_scale = torch.max(torch.abs(weight)) / 127.0
    config_args = {
        "weight_scale": w_scale,
        "quant_type": quant_type,
    }
    config_args.update(kwargs)
    return LinearQuantConfig(**config_args)


def get_quant_config(model=None, quant_type=LinearQuantType.W4A8, **kwargs):
    quant_config = QuantConfig()
    if model is None:
        quant_config.linear_configs["*"] = get_linear_quant_config(
            quant_type,
            torch.randn(1),
            **kwargs,
        )
        return quant_config
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            quant_config.linear_configs[strip_module_name(name)] = (
                get_linear_quant_config(
                    quant_type,
                    torch.randn(1),
                    **kwargs,
                )
            )
    return quant_config


class TestQuantLinear(unittest.TestCase):
    def setUp(self):
        """Set up common resources for tests."""
        torch.compiler.reset()
        torch.manual_seed(0)
        self.linear_layer_with_bias = torch.nn.Linear(
            IN_FEATURES, OUT_FEATURES, bias=True
        ).to(DEVICE, dtype=MODEL_DTYPE)

        torch.manual_seed(0)
        self.linear_layer_no_bias = torch.nn.Linear(
            IN_FEATURES, OUT_FEATURES, bias=False
        ).to(DEVICE, dtype=MODEL_DTYPE)

        torch.manual_seed(1)
        self.input_tensor = torch.randn(BATCH_SIZE, IN_FEATURES).to(
            DEVICE, dtype=MODEL_DTYPE
        )

    def test_pack_unpack_int4_roundtrip(self):
        """Tests if packing and then unpacking an int4 tensor restores the original."""
        original_tensor = torch.randint(
            -8, 8, (OUT_FEATURES, IN_FEATURES), dtype=torch.int8, device=DEVICE
        )
        dummy_layer = QuantLinearBase(
            self.linear_layer_no_bias,
            get_linear_quant_config(LinearQuantType.W4A8, torch.randn(1)),
        )

        # Test packing along dimension 1
        packed_dim1 = dummy_layer.pack_int4(original_tensor, dim=1)
        unpacked_dim1 = dummy_layer.unpack_int4(packed_dim1, dim=1)
        self.assertTrue(torch.equal(original_tensor, unpacked_dim1))

        # Test packing along dimension 0
        dummy_layer.quant_config.weight_int4_pack_dim = 0
        packed_dim0 = dummy_layer.pack_int4(original_tensor, dim=0)
        unpacked_dim0 = dummy_layer.unpack_int4(packed_dim0, dim=0)
        self.assertTrue(torch.equal(original_tensor, unpacked_dim0))

    def test_pack_int4_raises_error_for_invalid_dim(self):
        """Tests that pack_int4 raises a ValueError for an unsupported dimension."""
        tensor = torch.randint(
            -8, 8, (OUT_FEATURES, IN_FEATURES), dtype=torch.int8, device=DEVICE
        )
        dummy_layer = QuantLinearBase(
            self.linear_layer_no_bias,
            get_linear_quant_config(LinearQuantType.W4A8, torch.randn(1)),
        )
        with self.assertRaises(ValueError):
            dummy_layer.pack_int4(tensor, dim=2)

    def test_dequantize_weight(self):
        """Tests that dequantized weight is close to the original float weight."""
        original_weight = self.linear_layer_with_bias.weight.data

        # W8 Symmetric per-tensor quantization
        config = get_linear_quant_config(LinearQuantType.W8A16, original_weight)
        quant_layer = QuantLinearBase(self.linear_layer_with_bias, config)

        dequantized_weight = quant_layer.dequantize_weight()

        # NOTE: we only check shapes
        self.assertEqual(dequantized_weight.shape, original_weight.shape)

    def test_calculate_dynamic_qparams(self):
        """Tests the calculation of dynamic quantization parameters."""
        dummy_layer = QuantLinearBase(
            self.linear_layer_no_bias,
            get_linear_quant_config(LinearQuantType.W8A8, torch.randn(1)),
        )

        # Case 1: Per-tensor Symmetric
        cfg = dummy_layer.quant_config
        cfg.dynamic_quant_granularity = QuantGranularity.PER_TENSOR
        cfg.dynamic_quant_scheme = QuantScheme.SYMMETRIC
        scale, offset = dummy_layer._calculate_dynamic_qparams(self.input_tensor)

        expected_scale = torch.max(torch.abs(self.input_tensor)) / 127.0
        self.assertEqual(len(scale.shape), 0)
        self.assertEqual(len(offset.shape), 0)
        self.assertEqual(offset.item(), 0)
        self.assertTrue(torch.allclose(scale, expected_scale))

        # Case 2: Per-sample Asymmetric
        cfg.dynamic_quant_granularity = QuantGranularity.PER_SAMPLE
        cfg.dynamic_quant_scheme = QuantScheme.ASYMMETRIC
        scale, offset = dummy_layer._calculate_dynamic_qparams(self.input_tensor)

        min_vals = torch.amin(self.input_tensor, dim=1, keepdim=True)
        max_vals = torch.amax(self.input_tensor, dim=1, keepdim=True)
        expected_scale = (max_vals - min_vals) / 255.0

        self.assertEqual(scale.shape, (BATCH_SIZE,))
        self.assertEqual(offset.shape, (BATCH_SIZE,))
        self.assertTrue(torch.allclose(scale.flatten(), expected_scale.flatten()))

    def test_forward_pass_equivalence(self):
        """
        Tests the forward pass against a standard nn.Linear layer for various configs.
        """
        test_configs = [
            {
                "quant_type": LinearQuantType.W8A16,
                "use_bias": True,
                "w_scheme": "symmetric",
                "a_scheme": None,
            },
            {
                "quant_type": LinearQuantType.W8A16,
                "use_bias": False,
                "w_scheme": "asymmetric",
                "a_scheme": None,
            },
            {
                "quant_type": LinearQuantType.W8A8,
                "use_bias": True,
                "w_scheme": "symmetric",
                "a_scheme": "symmetric",
            },
            {
                "quant_type": LinearQuantType.W8A8,
                "use_bias": True,
                "w_scheme": "asymmetric",
                "a_scheme": "asymmetric",
            },
            {
                "quant_type": LinearQuantType.W4A8,
                "use_bias": True,
                "w_scheme": "symmetric",
                "a_scheme": "symmetric",
            },
            {
                "quant_type": LinearQuantType.W4A8,
                "use_bias": False,
                "w_scheme": "symmetric",
                "a_scheme": "symmetric",
            },
        ]

        for params in test_configs:
            with self.subTest(**params):
                torch.manual_seed(42)
                linear_layer = torch.nn.Linear(
                    IN_FEATURES, OUT_FEATURES, bias=params["use_bias"]
                ).to(DEVICE, dtype=MODEL_DTYPE)
                weight = linear_layer.weight.data

                config_kwargs = {}
                if params["w_scheme"] == "symmetric":
                    w_scale = torch.max(torch.abs(weight)) / 127.0
                    config_kwargs["weight_scale"] = w_scale
                else:  # asymmetric
                    w_min, w_max = torch.min(weight), torch.max(weight)
                    w_scale = (w_max - w_min) / 255.0
                    w_offset = -128.0 - w_min / w_scale
                    config_kwargs["weight_scale"] = w_scale
                    config_kwargs["weight_offset"] = w_offset

                if params["a_scheme"]:
                    config_kwargs["dynamic_quant_scheme"] = (
                        QuantScheme.SYMMETRIC
                        if params["a_scheme"] == "symmetric"
                        else QuantScheme.ASYMMETRIC
                    )

                config = get_linear_quant_config(
                    params["quant_type"], weight, **config_kwargs
                )
                quant_linear_layer = QuantLinearBase(linear_layer, config)

                expected_output = linear_layer(self.input_tensor)
                actual_output = quant_linear_layer(self.input_tensor)

                # NOTE: we only check shapes
                self.assertEqual(actual_output.shape, expected_output.shape)
                self.assertEqual(actual_output.dtype, MODEL_DTYPE)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model_quant_wildcard(self, model_id):
        import time

        model_config_with_quant = ModelConfig(
            ParallelConfig(),
            get_quant_config(),
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
        )
        start = time.time()
        qmodel = TransformerModel(model_id, model_config_with_quant)
        print(f"model init time: {time.time() - start}")
        num_linear_modules = sum(
            1
            for _, module in qmodel.named_modules()
            if isinstance(module, torch.nn.Linear)
        )
        self.assertEqual(num_linear_modules, 0)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = qmodel.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.hidden_size))

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model_quant_base(self, model_id):
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            num_hidden_layers_override=2,
        )
        model = TransformerModel(model_id, model_config)
        num_linear_modules = sum(
            1
            for _, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        )

        model_config_with_quant = ModelConfig(
            ParallelConfig(),
            get_quant_config(model.unwrap()),
            quant_linear_cls=QuantLinearBase,
            num_hidden_layers_override=2,
        )
        qmodel = TransformerModel(model_id, model_config_with_quant)
        num_qlinear_modules = sum(
            1
            for _, module in qmodel.named_modules()
            if isinstance(module, QuantLinearBase)
        )
        self.assertEqual(num_qlinear_modules, num_linear_modules)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = qmodel.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.hidden_size))

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model_quant_tensorcast_dynamic_w4a8(self, model_id):
        model_config = ModelConfig(ParallelConfig(), QuantConfig())
        model = TransformerModel(model_id, model_config)

        model_config_with_quant = ModelConfig(
            ParallelConfig(),
            get_quant_config(model.unwrap(), quant_type=LinearQuantType.W4A8),
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
        )
        qmodel = TransformerModel(model_id, model_config_with_quant)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = qmodel.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.hidden_size))
        result = runtime.table_averages()
        self.assertTrue("tensor_cast.dynamic_quant_linear_int4.default" in result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model_quant_tensorcast_static_w8a8(self, model_id):
        model_config = ModelConfig(ParallelConfig(), QuantConfig())
        model = TransformerModel(model_id, model_config)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        model_config_with_quant = ModelConfig(
            ParallelConfig(),
            get_quant_config(
                model.unwrap(),
                quant_type=LinearQuantType.W8A8,
                activation_scale=torch.empty(
                    [num_tokens], dtype=torch.float, device="meta"
                ),
            ),
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
        )
        qmodel = TransformerModel(model_id, model_config_with_quant)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = qmodel.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.hidden_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.quantize.default", result)
        self.assertIn("tensor_cast.static_quant_linear.default", result)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            # ["moonshotai/Kimi-K2-Base", True],  # long test time
        ]
    )
    def test_deepseek_mtp_quant_tensorcast_static_w8a8(self, model_id, do_compile):
        hf_config_json = model_id_to_json(model_id)
        self.assertIsNotNone(hf_config_json)
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(
                quant_type=LinearQuantType.W8A8,
            ),
            quant_linear_cls=TensorCastQuantLinear,
            hf_config_json=hf_config_json,
            enable_repetition=True,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        num_mtp_layers = 1
        mtp_block_module_name = model_id_to_mtp_block_module_name(model_id)
        self.assertIsNotNone(mtp_block_module_name)
        mtp_config = MtpConfig(
            num_mtp_layers=num_mtp_layers,
            mtp_block_module_name=mtp_block_module_name,
        )
        model_config.mtp_config = mtp_config
        model = TransformerModel(model_id, model_config)
        if do_compile:
            model = torch.compile(
                model, backend=get_backend(), dynamic=True, fullgraph=True
            )
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(
            model, model_config
        )
        # make sure all original attention modules have been replaced
        self.assertTrue(
            has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast")
        )

        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(
                    query_start_loc=attn_meta.query_start_loc
                ),
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.dynamic_quant_linear.default", result)

    def test_quant_lmhead(self):
        model_id = "Qwen/Qwen3-32B"
        linear_quant_config = get_linear_quant_config(
            LinearQuantType.W8A8,
            torch.randn(1),
        )
        quant_config = QuantConfig()
        quant_config.linear_configs["lm_head"] = linear_quant_config
        model_config = ModelConfig(
            ParallelConfig(),
            quant_config,
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
            enable_lmhead=True,
        )
        model = TransformerModel(model_id, model_config)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            _ = model.forward(inputs, position_ids)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.dynamic_quant_linear.default", result)

    def test_quant_lmhead_mtp(self):
        model_id = "deepseek-ai/DeepSeek-V3.1"
        hf_config_json = model_id_to_json(model_id)
        linear_quant_config = get_linear_quant_config(
            LinearQuantType.W8A8,
            torch.randn(1),
        )
        quant_config = QuantConfig()
        quant_config.linear_configs["*.lm_head"] = linear_quant_config
        model_config = ModelConfig(
            ParallelConfig(),
            quant_config,
            hf_config_json=hf_config_json,
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=1,
            enable_lmhead=True,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        num_mtp_layers = 1
        mtp_block_module_name = model_id_to_mtp_block_module_name(model_id)
        self.assertIsNotNone(mtp_block_module_name)
        mtp_config = MtpConfig(
            num_mtp_layers=num_mtp_layers,
            mtp_block_module_name=mtp_block_module_name,
        )
        model_config.mtp_config = mtp_config
        model = TransformerModel(model_id, model_config)
        # make sure all original attention modules have been replaced
        self.assertTrue(
            has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast")
        )
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(
            model, model_config
        )
        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(
                    query_start_loc=attn_meta.query_start_loc
                ),
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.dynamic_quant_linear.default", result)
