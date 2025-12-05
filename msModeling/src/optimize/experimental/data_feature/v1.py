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

import math
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from warnings import warn
from loguru import logger
import pandas as pd
from msserviceprofiler.msguard import Rule
from msserviceprofiler.modelevalstate.inference.common import get_bins_and_label
from msserviceprofiler.modelevalstate.inference.file_reader import StaticFile
from msserviceprofiler.modelevalstate.common import read_csv_s


HARDWARE_FIELD = ("cpu_count", "cpu_mem", "soc_name", "npu_mem")
HardWare = namedtuple("HardWare", HARDWARE_FIELD, defaults=[0, 0, "", 0])

ENV_FIELD = (
    "atb_llm_razor_attention_enable", "atb_llm_razor_attention_rope", "bind_cpu", "mies_use_mb_swapper",
    "mies_pecompute_threshold",
    "mies_tokenizer_sliding_window_size", "atb_llm_lcoc_enable", "lccl_deterministic",
    "hccl_deterministic", "atb_matmul_shuffle_k_enable")
EnvField = namedtuple("EnvField", ENV_FIELD)

MINDIE_FIELD = (
    "cache_block_size", "mindie__max_seq_len", "world_size", "cpu_mem_size", "npu_mem_size", "max_prefill_tokens",
    "max_prefill_batch_size", "max_batch_size")
MindieConfig = namedtuple("MindieConfig", MINDIE_FIELD)

MODEL_CONFIG_FIELD = (
    "architectures", "hidden_act", "initializer_range", "intermediate_size", "max_position_embeddings", "model_type",
    "num_attention_heads", "num_hidden_layers", "tie_word_embeddings", "torch_dtype", "use_cache", "vocab_size",
    "quantize", "quantization_config")

ModelConfig = namedtuple("ModelConfig", MODEL_CONFIG_FIELD)


BATCH_FIELD = ("batch_stage", "batch_size", "total_need_blocks", "total_prefill_token", "max_seq_len", \
               "model_execute_time")
BATCH_FILE_FIELD = ("ibis_batchid", *BATCH_FIELD, "req_info")
BatchField = namedtuple("BatchField", BATCH_FIELD)
BatchFileField = namedtuple("BatchFileField", BATCH_FILE_FIELD)

REQUEST_FIELD = ("input_length", "need_blocks", "output_length")
REQUEST_FILE_FIELD = ("ibis_reqid", "execute_id", *REQUEST_FIELD)
RequestField = namedtuple("RequestField", REQUEST_FIELD)
RequestFileField = namedtuple("RequestFileField", REQUEST_FILE_FIELD)

MODEL_OP_FIELD = (
    "op_name", "call_count", "input_count", "input_dtype", "input_shape", "output_count", "output_dtype",
    "output_shape", "host_setup_time", "host_execute_time", "kernel_execute_time", "aic_cube_fops", "aiv_vector_fops")
ModelOpField = namedtuple("ModelOpField", MODEL_OP_FIELD)
BATCH_SIZE = "batch_size"
MAX_SEQ_LEN = "max_seq_len"

MODEL_STRUCT_FIELD = (
    "total_param_num", "total_param_size", "embed_tokens_param_size_rate", "self_attn_param_size_rate",
    "mlp_param_size_rate", "input_layernorm_param_size_rate", "post_attention_layernorm_param_size_rate",
    "norm_param_size_rate",
    "lm_head_param_size_rate")
ModelStruct = namedtuple("ModelStruct", MODEL_STRUCT_FIELD, defaults=[0 for i in range(len(MODEL_STRUCT_FIELD))])

QUESTION_FIELD = ("question", "answer")
QuestionField = namedtuple("QuestionField", QUESTION_FIELD)


class HistInfo:
    input_length = get_bins_and_label("input_length", interval=80)
    need_blocks = get_bins_and_label("need_blocks", interval=1)
    need_slots = get_bins_and_label("need_slots", interval=128)
    output_length = get_bins_and_label("output_length", interval=10, )


class FileReader:
    def __init__(self, file_paths: List[Path], num_lines: int = math.inf, start_lines: int = 0,
                 start_file_index: int = 0,
                 columns: Optional[List[str]] = None):
        self.file_paths = file_paths
        self.num_lines = num_lines
        self.current_file_index = start_file_index
        self.current_line_index = start_lines
        for _file in file_paths:
            if not Rule.input_file_read.is_satisfied_by(_file):
                logger.error("please check the file for train exist and permissions")
                return
        self.columns = columns

    def read_rows_number(self, lines: List[pd.DataFrame]) -> int:
        if not lines:
            return self.num_lines
        rows_number = 0
        for _df in lines:
            rows_number += _df.shape[0]
        return self.num_lines - rows_number

    def read_lines(self) -> pd.DataFrame:
        lines = []
        while len(lines) < self.num_lines:
            try:
                if self.current_file_index >= len(self.file_paths):
                    # 读取完所有文件结束
                    break
                file_path = self.file_paths[self.current_file_index]
                if math.isclose(self.num_lines, math.inf):
                    df = read_csv_s(file_path, skiprows=self.current_line_index)
                    lines.append(df)
                    # 继续读取下一个文件
                    self.current_file_index += 1
                    self.current_line_index = 0
                else:
                    _expect_nrows = self.read_rows_number(lines)
                    df = read_csv_s(file_path, nrows=_expect_nrows, skiprows=self.current_line_index)
                    if self.columns:
                        df.columns = self.columns
                    else:
                        self.columns = df.columns.tolist()
                    lines.append(df)
                    lines_rows = sum([k.shape[0] for k in lines])
                    if lines_rows == self.num_lines:
                        # 读取到所需行结束
                        self.current_line_index += _expect_nrows
                        break
                    else:
                        # 未读取到所需行数据
                        self.current_line_index = 0
                        self.current_file_index += 1
            except Exception as e:
                warn(f"读取文件 {self.file_paths[self.current_file_index]} 时发生错误: {e}. 请核对。暂时跳过读取该文件的数据。")
                self.current_file_index += 1
                self.current_line_index = 0
        if not lines and self.current_file_index >= len(self.file_paths):
            raise StopIteration
        elif not lines:
            raise ValueError(f"lines is empty. lines: {lines}")
        return pd.concat(lines)

