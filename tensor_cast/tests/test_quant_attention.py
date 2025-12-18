import unittest

import torch
from parameterized import parameterized

from ..device import TEST_DEVICE

from ..layers.attention import AttentionTensorCast

from ..layers.mla import MultiheadLatentAttentionTensorCast

from ..layers.quant_linear import TensorCastQuantLinear

from ..layers.sampler import SamplingMetadata
from ..model_config import (
    AttentionQuantConfig,
    AttentionQuantType,
    LinearQuantType,
    MlaConfig,
    ModelConfig,
    MtpConfig,
    MultiheadLatentAttentionQuantConfig,
    ParallelConfig,
    QuantConfig,
)
from ..performance_model.analytic import AnalyticPerformanceModel
from ..runtime import Runtime
from ..transformers.model import TransformerModel

from ..transformers.utils import model_id_to_json, model_id_to_mtp_block_module_name

from .test_common import (
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    has_submodule_with_cls_name,
)


def get_quant_config(start_layer_id=-1, end_layer_id=-1):
    quant_config = QuantConfig()
    config = AttentionQuantConfig(
        quant_type=AttentionQuantType.INT8,
        query_scale=torch.tensor(1.0),
        kv_scale=torch.tensor(1.0),
        attention_prob_scale=torch.tensor(1.0),
    )
    if start_layer_id == -1 or end_layer_id == -1:
        quant_config.attention_configs[-1] = config
    for i in range(start_layer_id, end_layer_id):
        quant_config.attention_configs[i] = config
    return quant_config


def get_mla_quant_config(start_layer_id=-1, end_layer_id=-1):
    from .test_common import get_quant_config as get_quant_config_common

    quant_config = get_quant_config_common(quant_type=LinearQuantType.W8A8)
    config = MultiheadLatentAttentionQuantConfig(
        quant_type=AttentionQuantType.INT8,
        query_scale=torch.tensor(1.0),
        kv_scale=torch.tensor(1.0),
        attention_prob_scale=torch.tensor(1.0),
        kv_projected_scale=torch.tensor(1.0),
        qk_scale=torch.tensor(1.0),
        v_scale=torch.tensor(1.0),
        out_scale=torch.tensor(1.0),
    )
    if start_layer_id == -1 or end_layer_id == -1:
        quant_config.attention_configs[-1] = config
    for i in range(start_layer_id, end_layer_id):
        quant_config.attention_configs[i] = config
    return quant_config


class TestQuantAttention(unittest.TestCase):
    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_standard_attention_int8(self, model_id):
        kv_quant_start_idx = 0
        kv_quant_end_idx = 1
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(kv_quant_start_idx, kv_quant_end_idx),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=2,
        )
        model = TransformerModel(model_id, model_config)
        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(
            model, model_config
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
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.vocab_size))
        result = runtime.table_averages()
        self.assertIn("quantize.default", result)
        self.assertIn("reshape_and_cache.default", result)
        self.assertIn("attention_quant.default", result)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
            # ["moonshotai/Kimi-K2-Base"],
        ]
    )
    def test_mla_int8(self, model_id):
        hf_config_json = model_id_to_json(model_id)
        self.assertIsNotNone(hf_config_json)
        model_config = ModelConfig(
            ParallelConfig(),
            get_mla_quant_config(),
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
                sampling_metadata=SamplingMetadata(),
            )
            self.assertEqual(outputs.shape, (1, num_mtp_layers + 1))
        result = runtime.table_averages()
        self.assertIn("quantize.default", result)
        self.assertIn("concat_and_cache_mla.default", result)
        self.assertIn("multihead_latent_attention_quant.default", result)
