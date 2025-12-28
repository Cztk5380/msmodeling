import unittest

import torch
from parameterized import parameterized

from ..core.utils import build_model, UserInputConfig
from ..layers.sampler import SamplingMetadata
from ..patch_torch import patch_torch
from ..transformers.utils import get_mtp_block_module_name
from .test_common import create_mla_metadata_and_kv_cache, has_submodule_with_cls_name


class MtpTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", (16, True), False],
            ["deepseek-ai/DeepSeek-V3.1", (16, True), True],
            ["deepseek-ai/DeepSeek-V3.1", (16, False), True],
            ["moonshotai/Kimi-K2-Base", (16, True), False],
        ]
    )
    def test_deepseek_prefill_without_kvcache(
        self, model_id, parallel_configuration, do_compile
    ):
        num_tokens = 100

        num_mtp_layers = 3
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
            do_compile=do_compile,
            world_size=parallel_configuration[0],
            ep=parallel_configuration[1],
        )
        model = build_model(user_config)
        mtp_block_module_name = get_mtp_block_module_name(
            model.model_config.hf_config.model_type
        )
        self.assertIsNotNone(mtp_block_module_name)

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
            ["deepseek-ai/DeepSeek-V3.1", (16, True), False],
            ["deepseek-ai/DeepSeek-V3.1", (16, True), True],
            ["deepseek-ai/DeepSeek-V3.1", (16, False), True],
            ["moonshotai/Kimi-K2-Base", (16, True), False],
        ]
    )
    def test_deepseek_prefill_with_kvcache(
        self, model_id, parallel_configuration, do_compile
    ):
        num_mtp_layers = 3
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
            do_compile=do_compile,
            world_size=parallel_configuration[0],
            ep=parallel_configuration[1],
        )
        model = build_model(user_config)
        mtp_block_module_name = get_mtp_block_module_name(
            model.model_config.hf_config.model_type
        )
        self.assertIsNotNone(mtp_block_module_name)
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(
            model, model.model_config
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
            ["deepseek-ai/DeepSeek-V3.1", (16, True), False],
            ["deepseek-ai/DeepSeek-V3.1", (16, True), True],
            ["deepseek-ai/DeepSeek-V3.1", (16, False), True],
            ["moonshotai/Kimi-K2-Base", (16, True), False],
            # ["moonshotai/Kimi-K2-Base", True],  # long test time
        ]
    )
    def test_deepseek_decode_with_kvcache(
        self, model_id, parallel_configuration, do_compile
    ):
        num_mtp_layers = 3
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
            do_compile=do_compile,
            world_size=parallel_configuration[0],
            ep=parallel_configuration[1],
        )
        model = build_model(user_config)
        mtp_block_module_name = get_mtp_block_module_name(
            model.model_config.hf_config.model_type
        )
        self.assertIsNotNone(mtp_block_module_name)
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(
            model,
            model.model_config,
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
