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
from typing import Optional
import ast
import pandas as pd

import swifter
from loguru import logger
from pandas import DataFrame

from msserviceprofiler.modelevalstate.data_feature.dataset import MyDataSet
from msserviceprofiler.modelevalstate.inference.constant import OpAlgorithm
from msserviceprofiler.modelevalstate.inference.data_format_v1 import (
    MODEL_OP_FIELD,
    MODEL_STRUCT_FIELD,
    MODEL_CONFIG_FIELD,
    MINDIE_FIELD,
    ENV_FIELD,
    HARDWARE_FIELD,
)
from msserviceprofiler.modelevalstate.inference.dataset import TOTAL_OUTPUT_LENGTH, \
    TOTAL_SEQ_LENGTH, TOTAL_PREFILL_TOKEN

logger.debug(f'swifter version {getattr(swifter, "__version__")}')


class MyDataSetWithSwifter(MyDataSet):

    def preprocess_dispatch(self, lines_data: Optional[DataFrame] = None):
        logger.debug(f"start construct_data with swifter, shape {lines_data.shape}")
        try:
            return self.proprocess_with_swifter(lines_data)
        except Exception as e:
            logger.error(f"Failed in construct data with swifter. error: {e}")
            return super(MyDataSetWithSwifter, self).preprocess_dispatch(lines_data)

    def proprocess_with_swifter(self, lines_data: Optional[DataFrame] = None):
        logger.debug("dataset preprocess.")
        # 数据预处理
        if len(lines_data.columns) < 2:
            logger.error(f"DataFrame for train with swifter 列数不足，实际列数为 {len(lines_data.columns)}")
            return None, None
        

        # 将各个特征数据转换为列数据
        batch_df = pd.concat(
            lines_data.iloc[:, 0].swifter.apply(self.convert_batch_info, args=(lines_data.columns[0],)).values
        )
        request_df = pd.concat(
            lines_data.iloc[:, 1].swifter.apply(self.convert_request_info, args=(lines_data.columns[1],)).values
        )
        batch_df[TOTAL_OUTPUT_LENGTH] = request_df[TOTAL_OUTPUT_LENGTH]
        batch_df[TOTAL_SEQ_LENGTH] = batch_df[TOTAL_OUTPUT_LENGTH] + batch_df[TOTAL_PREFILL_TOKEN]
        request_df = request_df.drop(TOTAL_OUTPUT_LENGTH, axis=1)
        self.sub_columns = [batch_df.columns.tolist(), request_df.columns.tolist()]
        _load_data = [batch_df, request_df]
        convert_funcs = {
            MODEL_OP_FIELD: (
                self.convert_op_info if self.op_algorithm == OpAlgorithm.EXPECTED else self.convert_op_info_with_ratio
            ),
            MODEL_STRUCT_FIELD: self.convert_struct_info,
            MODEL_CONFIG_FIELD: self.convert_config_info,
            MINDIE_FIELD: self.convert_mindie_info,
            ENV_FIELD: self.convert_env_info,
            HARDWARE_FIELD: self.convert_hardware_info,
        }
        if len(lines_data.columns) > 2:
            columns_list = lines_data.columns[2:].tolist()
            field_cache = {col: ast.literal_eval(col) for col in columns_list}
            for col in columns_list:
                field_type = field_cache[col]
                if field_type in convert_funcs:
                    func = convert_funcs[field_type]
                    df = pd.concat(lines_data[col].swifter.apply(func, args=(col,)).values)
                    self.sub_columns.append(df.columns.tolist())
                    _load_data.append(df)

        # 提取 features 和labels
        self.load_data = pd.concat(_load_data, axis=1)
        self.labels = self.load_data[[self.predict_field]]
        self.features = self.load_data.drop(self.predict_field, axis=1)
        # 使用sklearn 进行 one-hot
        self.features = self.custom_encoder.transformer(self.features)
        return self.features, self.labels
