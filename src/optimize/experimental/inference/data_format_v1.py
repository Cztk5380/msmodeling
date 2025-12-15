# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple

BATCH_SIZE = "batch_size"
MAX_SEQ_LEN = "max_seq_len"

BATCH_FIELD = (
    "batch_stage", "batch_size", "total_need_blocks", "total_prefill_token", "max_seq_len")
BatchField = namedtuple("BatchField", BATCH_FIELD)

REQUEST_FIELD = ("input_length", "need_blocks", "output_length")
RequestField = namedtuple("RequestField", REQUEST_FIELD)
MODEL_OP_FIELD = (
    "op_name", "call_count", "input_count", "input_dtype", "input_shape", "output_count", "output_dtype",
    "output_shape", "host_setup_time", "host_execute_time", "kernel_execute_time", "aic_cube_fops", "aiv_vector_fops")
ModelOpField = namedtuple("ModelOpField", MODEL_OP_FIELD)

MODEL_STRUCT_FIELD = (
    "total_param_num", "total_param_size", "embed_tokens_param_size_rate", "self_attn_param_size_rate",
    "mlp_param_size_rate", "input_layernorm_param_size_rate", "post_attention_layernorm_param_size_rate",
    "norm_param_size_rate",
    "lm_head_param_size_rate")
ModelStruct = namedtuple("ModelStruct", MODEL_STRUCT_FIELD, defaults=[0 for i in range(len(MODEL_STRUCT_FIELD))])
MODEL_CONFIG_FIELD = (
    "architectures", "hidden_act", "initializer_range", "intermediate_size", "max_position_embeddings", "model_type",
    "num_attention_heads", "num_hidden_layers", "tie_word_embeddings", "torch_dtype", "use_cache", "vocab_size",
    "quantize", "quantization_config")
ModelConfig = namedtuple("ModelConfig", MODEL_CONFIG_FIELD)

MINDIE_FIELD = (
    "cache_block_size", "mindie__max_seq_len", "world_size", "cpu_mem_size", "npu_mem_size", "max_prefill_tokens",
    "max_prefill_batch_size", "max_batch_size")
MindieConfig = namedtuple("MindieConfig", MINDIE_FIELD)
ENV_FIELD = (
    "atb_llm_razor_attention_enable", "atb_llm_razor_attention_rope", "bind_cpu",
    "mies_use_mb_swapper", "mies_pecompute_threshold",
    "mies_tokenizer_sliding_window_size", "atb_llm_lcoc_enable", "lccl_deterministic",
    "hccl_deterministic", "atb_matmul_shuffle_k_enable")
EnvField = namedtuple("EnvField", ENV_FIELD)

HARDWARE_FIELD = ("cpu_count", "cpu_mem", "soc_name", "npu_mem")
HardWare = namedtuple("HardWare", HARDWARE_FIELD, defaults=[0, 0, "", 0])
_config_field = ["model_path", "static_file_dir", "req_and_decode_file", "cache_data"]
ConfigPath = namedtuple("ConfigPath", _config_field, defaults=[None for _ in _config_field])
