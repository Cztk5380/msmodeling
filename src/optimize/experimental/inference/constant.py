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
from statistics import mean
from enum import Enum

IS_SLEEP_FLAG = "MODEL_EVAL_STATE_IS_SLEEP_FLAG"

ALL_OP = (
    "ActivationOperation",
    "AsStridedOperation",
    "CumsumOperation",
    "GatherOperation",
    "MatmulOperation",
    "MultinomialOperation",
    "SplitOperation",
    "ConcatOperation",
    "SliceOperation",
    "SoftmaxOperation",
    "TransposeOperation",
    "ElewiseOperation",
    "KVCacheOperation",
    "ReshapeAndCacheOperation",
    "LinearActivationOperation",
    "LinearActivationQuantOperation",
    "LayerNormOperation",
    "RmsNormOperation",
    "FillOperation",
    "AllGatherOperation",
    "AllReduceOperation",
    "BroadcastOperation",
    "LinearOperation",
    "LinearQuantOperation",
    "LinearParallelOperation",
    "LinearSparseOperation",
    "RopeOperation",
    "SelfAttentionOperation",
    "PagedAttentionOperation",
    "TransdataOperation",
    "WhereOperation",
    "RepeatOperation",
    "SetValueOperation",
    "ReduceOperation",
    "TopkToppSamplingOperation",
)



ALL_ARCHITECTURE = (
    "LlamaForCausalLM",
    "InternLM2ForCausalLM",
    "Qwen2ForCausalLM",
    "LlavaNextVideoForConditionalGeneration",
    "glm4v",
    "internvl_chat",
    "internvl",
    "minicpm_llama3_v2",
    "bunny",
)

ALL_ARCHITECTURE_MAPPING = {architecture.lower(): f"architecture__{architecture.lower()}" 
                            for architecture in ALL_ARCHITECTURE}
ALL_BATCH_STAGE = ("prefill", "decode")
ALL_HIDDEN_ACT = ("silu", "gelu_pytorch_tanh", "gelu", "gelu_fast", "fastgelu")
ALL_MODEL_TYPE = (
    "bloom",
    "codeshell",
    "deepseekv2",
    "gpt_neox",
    "internlm",
    "llama",
    "minicpm_llama3_v2",
    "phi3",
    "qwen2_moe",
    "starcoder2",
    "zhinao",
    "aquila",
    "bunny",
    "cogvlm2",
    "deepseekvl",
    "gpt2",
    "internlm2",
    "llava",
    "minicpmv2",
    "qwen",
    "qwen2_vl",
    "telechat",
    "ziya",
    "baichuan",
    "chatglm",
    "dbrx",
    "gemma",
    "gte_qwen",
    "internlmxcomposer2",
    "llava_next",
    "mistral",
    "qwen2",
    "skywork",
    "vlmo",
    "clip",
    "deepseek",
    "glm4v",
    "idefics2",
    "internvl",
    "minicpm",
    "mixtral",
    "qwen2_audio",
    "starcoder",
    "yivl",
    "kclgt",
    "internvl_chat",
    "llava_next_video",
    "minicpmv",
    "MiniCPM-Llama3-V-2_5",
    "bunny-qwen2",
)

DTYPE_CATEGORY = ("float16", "float32", "int32", "int64")
SIZE_CATEGORY = ("min", "max", "mean")

SIZE_CATEGORY_HANDLER = {"min": lambda x: min(x), "max": lambda x: max(x), "mean": lambda x: mean(x)}

UNDEFINED = "undefined"
ALL_QUANTIZE = ("w8a8", "w8a8s", "w8a8sc", "w8a16", "w8a8_dynamic", UNDEFINED)
ALL_KV_QUANT_TYPE = ("c8", UNDEFINED)
ALL_GROUP_SIZE = ("0", "64", "128", UNDEFINED)
ALL_REDUCE_QUANT_TYPE = ("per_channel", UNDEFINED)


class OpParamType(Enum):
    TENSOR = "tensor"


one_input_output_op = {"input": [OpParamType.TENSOR], "output": [OpParamType.TENSOR]}
two_input_output_op = {"input": [OpParamType.TENSOR, OpParamType.TENSOR], "output": [OpParamType.TENSOR]}
# 根据文档预定义每个op需要几个参数
ALL_OP_PARAM_TYPE = {
    "ActivationOperation": one_input_output_op,
    "AllGatherOperation": one_input_output_op,
    "AllReduceOperation": {
        "input": [OpParamType.TENSOR, OpParamType.TENSOR, OpParamType.TENSOR],
        "output": [OpParamType.TENSOR],
    },
    "AsStridedOperation": one_input_output_op,
    "BroadcastOperation": one_input_output_op,
    "ConcatOperation": two_input_output_op,
    "CumsumOperation": one_input_output_op,
    "ElewiseOperation": {
        "input": [OpParamType.TENSOR, OpParamType.TENSOR, OpParamType.TENSOR, OpParamType.TENSOR],
        "output": [OpParamType.TENSOR],
    },
    "FillOperation": two_input_output_op,
    "GatherOperation": two_input_output_op,
    "KVCacheOperation": {
        "input": [OpParamType.TENSOR, OpParamType.TENSOR, OpParamType.TENSOR, OpParamType.TENSOR, OpParamType.TENSOR],
        "output": [OpParamType.TENSOR],
    },
    "LayerNormOperation": {"input": [OpParamType.TENSOR] * 8, "output": [OpParamType.TENSOR]},
    "LinearOperation": {"input": [OpParamType.TENSOR] * 4, "output": [OpParamType.TENSOR]},
    "LinearParallelOperation": {"input": [OpParamType.TENSOR] * 5, "output": [OpParamType.TENSOR] * 2},
    "LinearSparseOperation": {"input": [OpParamType.TENSOR] * 4, "output": [OpParamType.TENSOR]},
    "MultinomialOperation": one_input_output_op,
    "PagedAttentionOperation": {"input": [OpParamType.TENSOR] * 16, "output": [OpParamType.TENSOR]},
    "ReduceOperation": one_input_output_op,
    "RepeatOperation": one_input_output_op,
    "ReshapeAndCacheOperation": {"input": [OpParamType.TENSOR] * 8, "output": [OpParamType.TENSOR] * 2},
    "RmsNormOperation": {"input": [OpParamType.TENSOR] * 7, "output": [OpParamType.TENSOR]},
    "RopeOperation": {"input": [OpParamType.TENSOR] * 5, "output": [OpParamType.TENSOR] * 2},
    "SelfAttentionOperation": {"input": [OpParamType.TENSOR] * 11, "output": [OpParamType.TENSOR]},
    "SetValueOperation": two_input_output_op,
    "SliceOperation": one_input_output_op,
    "SoftmaxOperation": one_input_output_op,
    "SplitOperation": {"input": [OpParamType.TENSOR], "output": [OpParamType.TENSOR] * 3},
    "TopkToppSamplingOperation": {"input": [OpParamType.TENSOR] * 4, "output": [OpParamType.TENSOR] * 2},
    "TransdataOperation": one_input_output_op,
    "TransposeOperation": one_input_output_op,
    "WhereOperation": {"input": [OpParamType.TENSOR] * 3, "output": [OpParamType.TENSOR]},
    "MatmulOperation": two_input_output_op,
    "LinearActivationOperation": {"input": [OpParamType.TENSOR] * 3, "output": [OpParamType.TENSOR]},
    "LinearActivationQuantOperation": {"input": [OpParamType.TENSOR] * 4, "output": [OpParamType.TENSOR]},
    "LinearQuantOperation": {"input": [OpParamType.TENSOR] * 4, "output": [OpParamType.TENSOR]},
}

OP_EXECUTE_DELTA_FIELD = (
    "host_setup_time",
    "host_execute_time",
    "kernel_execute_time",
    "aic_cube_fops",
    "aiv_vector_fops",
)
for op in ALL_OP:
    if op not in ALL_OP_PARAM_TYPE:
        raise ValueError(f"Not Found {op} in {ALL_OP_PARAM_TYPE}")


class OpAlgorithm:
    EXPECTED = "expected"
    SCALE = "scale"


class BatchStage:
    PREFILL = "prefill"
    DECODE = "decode"
