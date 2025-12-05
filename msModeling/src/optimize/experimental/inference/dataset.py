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
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from msserviceprofiler.modelevalstate.inference.constant import DTYPE_CATEGORY, ALL_HIDDEN_ACT, ALL_MODEL_TYPE, \
    ALL_QUANTIZE, \
    ALL_KV_QUANT_TYPE, ALL_GROUP_SIZE, ALL_REDUCE_QUANT_TYPE, ALL_BATCH_STAGE
from msserviceprofiler.modelevalstate.inference.data_format_v1 import BatchField, RequestField, \
    ModelOpField, ModelStruct, ModelConfig, MindieConfig, \
    EnvField, HardWare, BATCH_FIELD, REQUEST_FIELD, MODEL_OP_FIELD, MODEL_STRUCT_FIELD, MODEL_CONFIG_FIELD, \
    MINDIE_FIELD, ENV_FIELD, HARDWARE_FIELD
from msserviceprofiler.modelevalstate.inference.utils import PreprocessTool, TOTAL_OUTPUT_LENGTH, TOTAL_SEQ_LENGTH, \
    TOTAL_PREFILL_TOKEN


@dataclass
class InputData:
    batch_field: BatchField
    request_field: Tuple[RequestField, ...]
    model_op_field: Optional[Tuple[ModelOpField, ...]] = None
    model_struct_field: Optional[ModelStruct] = None
    model_config_field: Optional[ModelConfig] = None
    mindie_field: Optional[MindieConfig] = None
    env_field: Optional[EnvField] = None
    hardware_field: Optional[HardWare] = None


@dataclass
class CategoryInfo:
    name: str = "batch_stage"
    all_value: Tuple[str] = ("prefill", "decode")


preset_category_data = [
    CategoryInfo("batch_stage", ALL_BATCH_STAGE),
    CategoryInfo("hidden_act", ALL_HIDDEN_ACT),
    CategoryInfo("model_type", ALL_MODEL_TYPE),
    CategoryInfo("torch_dtype", DTYPE_CATEGORY),
    CategoryInfo("quantize", ALL_QUANTIZE),
    CategoryInfo("kv_quant_type", ALL_KV_QUANT_TYPE),
    CategoryInfo("group_size", ALL_GROUP_SIZE),
    CategoryInfo("reduce_quant_type", ALL_REDUCE_QUANT_TYPE)
]


class CustomOneHotEncoder:
    def __init__(self, one_hots: Optional[List[CategoryInfo]] = None):
        if one_hots:
            self.one_hots: List[CategoryInfo] = one_hots
        else:
            self.one_hots = []
        self.one_hot_encoders: List[OneHotEncoder] = []
        self.first = True

    def fit(self):
        self.one_hot_encoders = []
        for _one_hot in self.one_hots:
            _cur_one_hot = OneHotEncoder(handle_unknown='infrequent_if_exist')
            _cur_one_hot.fit(np.array([[k] for k in _one_hot.all_value]))
            self.one_hot_encoders.append(_cur_one_hot)

    def update_encoders(self, columns):
        if self.first:
            new_one_hot_encoders = []
            new_one_hosts = []
            for i, v in enumerate(self.one_hot_encoders):
                _one_hot_info = self.one_hots[i]
                if _one_hot_info.name in columns:
                    new_one_hot_encoders.append(v)
                    new_one_hosts.append(_one_hot_info)
            self.one_hot_encoders = new_one_hot_encoders
            self.one_hots = new_one_hosts
            self.first = False

    def transformer(self, x: DataFrame):
        self.update_encoders(x.columns)
        for i, _one_hot_encoder in enumerate(self.one_hot_encoders):
            _one_hot_info = self.one_hots[i]
            encode_value = _one_hot_encoder.transform(x[_one_hot_info.name].values.reshape(-1, 1)).toarray()
            encode_columns = []
            for categories in _one_hot_encoder.categories_:
                for category in categories:
                    encode_columns.append(f"{_one_hot_info.name}__{category}")
            _encode_df = pd.DataFrame(encode_value, columns=encode_columns)
            x = pd.concat([_encode_df, x], axis=1)
            x = x.drop(_one_hot_info.name, axis=1)

        return x

    def transformer_optimize(self, data: List, data_column: List):
        self.update_encoders(data_column)
        new_data = []
        new_data_column = []
        for i, _one_hot_encoder in enumerate(self.one_hot_encoders):
            _one_hot_info = self.one_hots[i]
            _col_index = data_column.index(_one_hot_info.name)
            encode_value = _one_hot_encoder.transform(np.array([[data[_col_index], ]]))
            _new_column = [
                f"{_one_hot_info.name}__{i}"
                for k in _one_hot_encoder.categories_
                for i in k
            ]
            new_data.extend(*encode_value.toarray().tolist())
            new_data_column.extend(_new_column)
        return new_data, new_data_column


class CustomLabelEncoder:
    def __init__(self, category_info: Optional[List[CategoryInfo]] = None):
        if category_info:
            self.category_info: List[CategoryInfo] = category_info
        else:
            self.category_info = []
        self.category_encoders: List[LabelEncoder] = []
        self.first = True
        self.encode_cache = {}

    def fit(self):
        self.category_encoders = []
        for _cate_info in self.category_info:
            _cur_encoder = LabelEncoder()
            _cur_encoder.fit(np.array([[k] for k in _cate_info.all_value]))
            self.category_encoders.append(_cur_encoder)

    def update_encoders(self, columns):
        if self.first:
            new_category_encoders = []
            new_category_info = []
            for i, v in enumerate(self.category_encoders):
                _cate_info = self.category_info[i]
                if _cate_info.name in columns:
                    new_category_encoders.append(v)
                    new_category_info.append(_cate_info)
            self.category_encoders = new_category_encoders
            self.category_info = new_category_info
            self.first = False

    def transformer(self, x: DataFrame):
        self.update_encoders(x.columns)
        for i, _cate_encoder in enumerate(self.category_encoders):
            _cate_info = self.category_info[i]
            encode_value = _cate_encoder.transform(x[_cate_info.name].values)
            x[_cate_info.name] = encode_value
        return x

    def transformer_optimize(self, data: List, data_column: List):
        self.update_encoders(data_column)
        for i, _cate_encoder in enumerate(self.category_encoders):
            _cate_info = self.category_info[i]
            _col_index = data_column.index(_cate_info.name)
            if 0 <= _col_index < len(data):
                _cache = (_cate_info.name, data[_col_index])
            else:
                raise IndexError(f"Column index {_col_index} is out of valid range [0, {len(data) - 1}] for data.")
            if _cache in self.encode_cache:
                data[_col_index] = self.encode_cache.get(_cache)
            else:
                encode_value = _cate_encoder.transform([data[_col_index], ])
                data[_col_index] = encode_value[0]
                self.encode_cache[_cache] = data[_col_index]
        return data, data_column


class DataProcessor:
    def __init__(self, custom_encoder: Optional[Union[CustomOneHotEncoder, CustomLabelEncoder]] = None):
        self.custom_encoder = custom_encoder

    def preprocessor(self, input_data: InputData) -> np.ndarray:
        batch_v, batch_col = PreprocessTool.generate_data(input_data.batch_field, BATCH_FIELD)
        request_v, request_col = PreprocessTool.generate_data_with_request_info(input_data.request_field, REQUEST_FIELD)
        for i, v in enumerate(request_col):
            if v == TOTAL_OUTPUT_LENGTH:
                batch_v = [*batch_v, request_v[i]]
                batch_col = [*batch_col, TOTAL_OUTPUT_LENGTH]
                request_v = list(request_v)
                request_v.pop(i)
                request_col = list(request_col)
                request_col.pop(i)
                break
        total_seq_length = 0
        for i, v in enumerate(batch_col):
            if v in (TOTAL_PREFILL_TOKEN, TOTAL_OUTPUT_LENGTH):
                total_seq_length += batch_v[i]

        batch_v.append(total_seq_length)
        batch_col.append(TOTAL_SEQ_LENGTH)

        load_value = [*batch_v, *request_v]
        load_col = [*batch_col, *request_col]

        if input_data.model_op_field:
            v, col = PreprocessTool.generate_data_with_op_info(input_data.model_op_field, MODEL_OP_FIELD)
            load_value.extend(v)
            load_col.extend(col)
        if input_data.model_struct_field:
            v, col = PreprocessTool.generate_data_with_struct_info(input_data.model_struct_field,
                                                                   MODEL_STRUCT_FIELD)
            load_value.extend(v)
            load_col.extend(col)
        if input_data.model_config_field:
            v, col = PreprocessTool.generate_data_with_model_config(input_data.model_config_field,
                                                                    MODEL_CONFIG_FIELD)
            load_value.extend(v)
            load_col.extend(col)
        if input_data.mindie_field:
            v, col = PreprocessTool.generate_data(input_data.mindie_field, MINDIE_FIELD)
            load_value.extend(v)
            load_col.extend(col)
        if input_data.env_field:
            v, col = PreprocessTool.generate_data(input_data.env_field, ENV_FIELD)
            load_value.extend(v)
            load_col.extend(col)
        if input_data.hardware_field:
            v, col = PreprocessTool.generate_data(input_data.hardware_field, HARDWARE_FIELD)
            load_value.extend(v)
            load_col.extend(col)
        load_value, _ = self.custom_encoder.transformer_optimize(load_value, load_col)
        return load_value
