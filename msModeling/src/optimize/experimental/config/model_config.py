# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from msserviceprofiler.modelevalstate.common import get_npu_total_memory
from msserviceprofiler.modelevalstate.config.config import get_settings
from msserviceprofiler.msguard.security import open_s


class ModelConfig:
    def __init__(self, config_path: Path):
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path!r}")
        try:
            with open_s(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"The JSON format of the configuration file '{config_path!r}' is invalid") from e
        except Exception as e:
            raise IOError(f"An error occurred while reading the configuration file '{config_path!r}'") from e
        logger.debug(f"Successfully loaded configuration file: {config_path!r} ")
        self.hidden_size = self._get_required_param(config_data, 'hidden_size')
        self.intermediate_size = self._get_required_param(config_data, 'intermediate_size')
        self.num_attention_heads = self._get_required_param(config_data, 'num_attention_heads')
        self.num_hidden_layers = self._get_required_param(config_data, 'num_hidden_layers')
        self.num_key_value_heads = self._get_required_param(config_data, 'num_key_value_heads')
        self.vocab_size = self._get_required_param(config_data, 'vocab_size')
        self.max_position_embeddings = self._get_required_param(config_data, 'max_position_embeddings')
        self.kvcache_dtype_byte = self._get_kvcache_dtype_byte(config_data)
        logger.debug(f"kvcache_dtype_byte: {self.kvcache_dtype_byte}")

        # 暂时默认为2
        self.cache_num = 2
        logger.debug(f"cache_num (K/V cache): {self.cache_num}")
        self.memory_mb = self.calculate_model_weights_size()

    def __repr__(self):
        return (
            f"ModelConfig(\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  intermediate_size={self.intermediate_size},\n"
            f"  num_attention_heads={self.num_attention_heads},\n"
            f"  num_hidden_layers={self.num_hidden_layers},\n"
            f"  num_key_value_heads={self.num_key_value_heads},\n"
            f"  kvcache_dtype_byte={self.kvcache_dtype_byte},\n"
            f"  cache_num={self.cache_num}\n"
            f"  vocab_size={self.vocab_size}\n"
            f"  max_position_embeddings={self.max_position_embeddings}\n"
            f")"
        )

    @staticmethod
    def _get_required_param(config_data: dict, key: str):
        # 获取必需的配置参数，如果缺失报错。
        value = config_data.get(key)
        if value is None:
            raise ValueError(f"配置文件中缺少必需的参数: '{key}'。")
        logger.debug(f"{key}: {value}")
        return value

    @staticmethod
    def _get_kvcache_dtype_byte(config_data: dict) -> int:
        # 根据 torch.dtype 信息推断 kvcache 的字节数。

        dtype_str = config_data.get('torch_dtype') or config_data.get('dtype')
        if dtype_str:
            dtype_str = str(dtype_str).lower()
            if '16' in dtype_str:  # e.g., "fp16", "bfloat16", "float16"
                return 2
            elif 'int8' in dtype_str:
                return 1
            elif '32' in dtype_str:  # e.g., "fp32", "float32"
                return 4
            else:
                logger.warning(f"Unrecognized dtype: '{dtype_str}'. kvcache_dtype_byte defaults to 2 (bf16/fp16).")
                return 2
        else:
            logger.warning(
                "The 'torch_dtype' or 'dtype' was not found in the configuration file."
                "The default value for kvcache_dtype_byte is 2 (bf16/fp16)."
                )
            return 2

    def get_one_token_cache(self):
        # 计算一个token的KV Cache占用的显存大小(Bytes)
        if self.num_attention_heads == 0:
            return 0

        head_size = self.hidden_size / self.num_attention_heads

        # 计算每个 token 在所有层中 K 和 V 的总元素数量
        total_elements_per_token = head_size * self.num_key_value_heads * self.num_hidden_layers * self.cache_num

        kvcache_size_bytes = total_elements_per_token * self.kvcache_dtype_byte

        return kvcache_size_bytes

    def get_peak_activations_size(self, max_prefill_token: int, sequence_length) -> float:
        # 估算峰值激活值显存占用(Bytes)
        # 公式: (B*S*H + 3*B*S*H + B*N*S*S + B*S*I) * dtype_byte
        # max_prefill_token: batch_size (int): 批大小 * sequence_length (int): 平均序列长度
        input_output_elements = max_prefill_token * self.hidden_size

        qkv_proj_elements = 3 * max_prefill_token * self.hidden_size

        attn_scores_elements = self.num_attention_heads * max_prefill_token * sequence_length

        mlp_intermediate_elements = max_prefill_token * self.intermediate_size

        total_elements_at_peak = (
                input_output_elements +
                qkv_proj_elements +
                attn_scores_elements +
                mlp_intermediate_elements
        )

        peak_activations_bytes = total_elements_at_peak * self.kvcache_dtype_byte

        return peak_activations_bytes

    # 计算模型权重总大小
    def calculate_model_weights_size(self) -> float:

        # 1. Token Embeddings 参数数量 = vocab_size * hidden_size
        embedding_params = self.vocab_size * self.hidden_size

        # 2. Position Embeddings 参数数量 = max_position_embeddings * hidden_size
        position_embedding_params = self.max_position_embeddings * self.hidden_size

        # 3. 每层 Transformer 块 参数估算
        # 简化近似：12 * hidden_size^2
        params_per_layer = 12 * self.hidden_size * self.hidden_size

        total_transformer_params = self.num_hidden_layers * params_per_layer

        # 4. 最终的语言模型头 (LM Head)
        # 参数数量 = hidden_size * vocab_size
        # 注意：如果 LM Head 和词嵌入层权重共享 (weight tying)，则这部分不额外计算。
        # 这里假设不共享，或者计算的是总参数量。
        lm_head_params = self.vocab_size * self.hidden_size

        # 5. 最终的 LayerNorm (如果存在，通常在 LM Head 之前)
        final_layernorm_params = 2 * self.hidden_size  # gamma, beta

        # 总参数数量
        total_params = (
                embedding_params +
                position_embedding_params +
                total_transformer_params +
                lm_head_params +
                final_layernorm_params
        )

        memory_bytes = total_params * self.kvcache_dtype_byte

        # Bytes to MB
        memory_mb = memory_bytes / (1024 * 1024)

        logger.debug(f"估算模型权重参数数量: {total_params / 1_000_000_000:.2f} Billion")
        logger.debug(f"每个参数字节数: {self.kvcache_dtype_byte}")
        logger.debug(f"估算模型权重显存占用: {memory_mb:.2f} MB")

        return memory_mb


class MindieModelConfig:
    def __init__(self, config_path, avg_input_length: int = 254, max_output_length: int = 256,
                 max_input_length: int = 8192, npu_total_mem: Optional[int] = None,
                 memory_usage_rate: Optional[int] = None):
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path!r}")
        try:
            with open_s(config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"The JSON format of the configuration file '{config_path!r}' is invalid") from e
        except Exception as e:
            raise IOError(f"An error occurred while reading the configuration file '{config_path!r}'") from e
        self.mem_coefficient = get_settings().mem_coefficient
        self.npu_device_ids = self.config_data["BackendConfig"]["npuDeviceIds"]
        self.cache_block_size = self.config_data["BackendConfig"]["ScheduleConfig"]["cacheBlockSize"]
        self.max_prefill_tokens = self.config_data["BackendConfig"]["ScheduleConfig"]["maxPrefillTokens"]
        self.avg_input_length = avg_input_length
        self.max_input_length = max_input_length
        self.max_output_length = self.config_data["BackendConfig"]["ScheduleConfig"]["maxIterTimes"]
        _model_path = Path(self.config_data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["modelWeightPath"])
        self.model_config = ModelConfig(_model_path.joinpath("config.json"))
        self.byte_to_gb = 1024 * 1024 * 1024
        self.tp_size = max(1, self.config_data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0].get("tp", 1))
        if npu_total_mem is None and memory_usage_rate is None:
            self.npu_total_mem, self.memory_usage_rate = get_npu_total_memory(self.npu_device_ids[0][0])
        else:
            self.npu_total_mem, self.memory_usage_rate = npu_total_mem, memory_usage_rate
        self.mem_for_kv_cache_gb = self.get_npu_mem_size()
        self.mem_for_kv_cache = self.mem_for_kv_cache_gb * self.byte_to_gb
        logger.debug(f"mem_for_kv_cache_gb, {self.mem_for_kv_cache_gb}")

    def get_npu_mem_size(self):
        # 获取npu可以用来分配kv cache的size
        mem_for_kv_cache_gb = int(
            self.config_data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["npuMemSize"])
        if mem_for_kv_cache_gb != -1:
            return mem_for_kv_cache_gb
        npu_total_mem = self.npu_total_mem / 1024
        npu_available_memory = npu_total_mem * (100 - self.memory_usage_rate) / 100
        # 单卡模型占用显存 tp_size最小为1不会为0
        model_weights_gb_per_npu = self.model_config.calculate_model_weights_size() / 1024 / self.tp_size
        activation_gb = self.model_config.get_peak_activations_size(max_prefill_token=self.max_prefill_tokens,
                                    sequence_length=self.max_input_length) / self.byte_to_gb / self.tp_size
        logger.debug(f"activation_gb: {activation_gb}")
        mem_for_kv_cache_gb = (
                npu_available_memory * self.mem_coefficient - model_weights_gb_per_npu - activation_gb)
        return mem_for_kv_cache_gb

    def get_max_batch_size_bound(self):
        one_token_cache = self.model_config.get_one_token_cache()
        one_kv_block_size = self.cache_block_size * one_token_cache
        if one_kv_block_size == 0:
            raise ValueError("one_kv_block_size is 0, which would cause division by zero error")
        total_block_num = np.floor(self.mem_for_kv_cache / one_kv_block_size)
        logger.debug(f"total_block_num: {total_block_num}")
        if self.cache_block_size == 0 or self.avg_input_length == 0 or self.max_output_length == 0:
            raise ValueError("cache_block_size is 0, which would cause division by zero error")
        min_one_sequence_block_num = self.avg_input_length / self.cache_block_size
        max_one_sequence_block_num = min_one_sequence_block_num + self.max_output_length / self.cache_block_size
        min_one_sequence_block_num = np.ceil(min_one_sequence_block_num)
        max_one_sequence_block_num = np.ceil(max_one_sequence_block_num)
        if min_one_sequence_block_num == 0 or max_one_sequence_block_num == 0:
            raise ValueError("one_sequence_block_num is 0, which would cause division by zero error")
        max_batch_size_lb = np.floor(total_block_num / max_one_sequence_block_num)
        max_batch_size_ub = np.ceil(total_block_num / min_one_sequence_block_num)
        return max_batch_size_lb, max_batch_size_ub