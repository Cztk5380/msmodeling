# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

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
