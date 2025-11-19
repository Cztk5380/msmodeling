import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend
from ..layers.attention import AttentionTensorCast
from ..layers.mla import MultiheadLatentAttentionTensorCast
from ..layers.sampler import SamplingMetadata
from ..model_config import (
    MlaConfig,
    ModelConfig,
    MtpConfig,
    ParallelConfig,
    QuantConfig,
)
from ..patch_torch import patch_torch
from ..transformers.model import TransformerModel
from ..transformers.utils import model_id_to_json, model_id_to_mtp_block_module_name

from .test_common import (
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    has_submodule_with_cls_name,
)


class MtpTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            # ["moonshotai/Kimi-K2-Base", True],  # long test time
        ]
    )
    def test_deepseek_prefill_without_kvcache(self, model_id, do_compile):
        num_tokens = 100
        hf_config_json = model_id_to_json(model_id)
        self.assertIsNotNone(hf_config_json)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            hf_config_json=hf_config_json,
            enable_repetition=True,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        num_mtp_layers = 3
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
        # make sure all original attention modules have been replaced
        self.assertTrue(
            has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast")
        )
        inputs = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([2, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(
                inputs, position_ids, sampling_metadata=SamplingMetadata()
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            # ["moonshotai/Kimi-K2-Base", True],  # long test time
        ]
    )
    def test_deepseek_prefill_with_kvcache(self, model_id, do_compile):
        hf_config_json = model_id_to_json(model_id)
        self.assertIsNotNone(hf_config_json)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            hf_config_json=hf_config_json,
            enable_repetition=True,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        num_mtp_layers = 3
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
        with torch.no_grad(), patch_torch():
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

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
            ["deepseek-ai/DeepSeek-V3.1", True],
            ["moonshotai/Kimi-K2-Base", False],
            # ["moonshotai/Kimi-K2-Base", True],  # long test time
        ]
    )
    def test_deepseek_decode_with_kvcache(self, model_id, do_compile):
        hf_config_json = model_id_to_json(model_id)
        self.assertIsNotNone(hf_config_json)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            hf_config_json=hf_config_json,
            enable_repetition=True,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        num_mtp_layers = 3
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
            model,
            model_config,
            query_len_1=num_mtp_layers + 1,
            query_len_2=num_mtp_layers + 1,
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
                sampling_metadata=SamplingMetadata(
                    query_start_loc=attn_meta.query_start_loc,
                    selected_token_indices=None,
                ),
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", False],
            ["Qwen/Qwen3-32B", True],
            ["Qwen/Qwen3-235B-A22B", False],
            ["zai-org/GLM-4.5", False],
        ]
    )
    def test_automatic_mtp_mode(self, model_id, do_compile):
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            enable_repetition=True,
        )
        num_mtp_layers = 3
        mtp_config = MtpConfig(
            num_mtp_layers=num_mtp_layers,
        )
        model_config.mtp_config = mtp_config
        model = TransformerModel(model_id, model_config)
        if do_compile:
            model = torch.compile(
                model, backend=get_backend(), dynamic=True, fullgraph=True
            )
        attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(
            model, model_config
        )
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(
                    query_start_loc=attn_meta.query_start_loc,
                    selected_token_indices=None,
                ),
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))
