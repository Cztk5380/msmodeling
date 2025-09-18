import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend

from ..layers.attention import AttentionTensorCast
from ..layers.mla import MultiheadLatentAttentionTensorCast
from ..model_config import MlaConfig, ModelConfig, ParallelConfig, QuantConfig
from ..patch_torch import patch_torch
from ..transformers.model import TransformerModel
from ..transformers.utils import model_id_to_json

from .test_common import (
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    has_submodule_with_cls_name,
)


# TODO: we comment all the compilation cases for large MoE models due to slow compilation time
#       need to find out solution to speed things up...


class ModelLoadTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3-235B-A22B", False],
            ["Qwen/Qwen3-235B-A22B", True],
            ["zai-org/GLM-4.5", False],
            ["zai-org/GLM-4.5", True],
        ]
    )
    def test_vanilla_transformer_model(self, model_id, do_compile):
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(), QuantConfig(), num_hidden_layers_override=2
        )
        model = TransformerModel(model_id, model_config)
        if do_compile:
            model = torch.compile(
                model, backend=get_backend(), dynamic=True, fullgraph=True
            )
        inputs = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (2, num_tokens, model.hidden_size))

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            ["moonshotai/Kimi-K2-Base", True],
        ]
    )
    def test_deepseek_without_kvcache(self, model_id, do_compile):
        num_tokens = 100
        hf_config_json = model_id_to_json(model_id)
        self.assertIsNotNone(hf_config_json)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            hf_config_json=hf_config_json,
            num_hidden_layers_override=2,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        model = TransformerModel(model_id, model_config)
        if do_compile:
            model = torch.compile(
                model, backend=get_backend(), dynamic=True, fullgraph=True
            )
        # make sure all original attention modules have been replaced
        self.assertTrue(
            has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast")
        )
        inputs = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (2, num_tokens, model.hidden_size))

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            ["moonshotai/Kimi-K2-Base", True],
        ]
    )
    def test_deepseek_with_kvcache(self, model_id, do_compile):
        hf_config_json = model_id_to_json(model_id)
        self.assertIsNotNone(hf_config_json)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            hf_config_json=hf_config_json,
            num_hidden_layers_override=2,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
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
        with torch.no_grad(), patch_torch():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3-235B-A22B", False],
            ["Qwen/Qwen3-235B-A22B", True],
            ["zai-org/GLM-4.5", False],
            ["zai-org/GLM-4.5", True],
        ]
    )
    def test_prefill_without_kvcache(self, model_id, do_compile):
        num_tokens = 100
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=2,
        )
        model = TransformerModel(model_id, model_config)
        if do_compile:
            model = torch.compile(
                model, backend=get_backend(), dynamic=True, fullgraph=True
            )
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3-235B-A22B", False],
            ["Qwen/Qwen3-235B-A22B", True],
            ["zai-org/GLM-4.5", False],
            ["zai-org/GLM-4.5", True],
        ]
    )
    def test_prefill_with_kvcache(self, model_id, do_compile):
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=2,
        )
        model = TransformerModel(model_id, model_config)
        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(
            model, model_config
        )
        if do_compile:
            model = torch.compile(
                model, backend=get_backend(), dynamic=True, fullgraph=True
            )
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))

    @parameterized.expand(
        [
            ["Qwen/Qwen3-Next-80B-A3B-Instruct", False],
            ["Qwen/Qwen3-Next-80B-A3B-Instruct", True],
        ]
    )
    # temporarily disable since it relies on Transformers mainline
    def _test_qwen3_next_with_kvcache(self, model_id, do_compile):
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            num_hidden_layers_override=2,
        )
        model = TransformerModel(model_id, model_config)
        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(
            model, model_config
        )
        if do_compile:
            model = torch.compile(
                model,
                backend=get_backend(),
                dynamic=True,
                fullgraph=False,  # data dependency code in QwenNext
            )
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                cache_position=torch.arange(
                    0, num_tokens, dtype=torch.long, device="cpu"
                ),
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
