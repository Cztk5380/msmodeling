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

"""
训练预测每个状态速度的线性模型
"""
import ast
import re
from collections import namedtuple
from pathlib import Path
from typing import Optional, Union
import os

import matplotlib
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from msserviceprofiler.modelevalstate.inference.constant import OpAlgorithm
from msserviceprofiler.modelevalstate.inference.data_format_v1 import (
    MODEL_OP_FIELD,
    MODEL_STRUCT_FIELD,
    MODEL_CONFIG_FIELD,
    MINDIE_FIELD,
    ENV_FIELD,
    HARDWARE_FIELD,
)
from msserviceprofiler.modelevalstate.inference.dataset import CustomOneHotEncoder, CustomLabelEncoder, \
    preset_category_data
from msserviceprofiler.modelevalstate.inference.utils import PreprocessTool, TOTAL_OUTPUT_LENGTH, \
    TOTAL_SEQ_LENGTH, TOTAL_PREFILL_TOKEN
from msserviceprofiler.msguard.security.io import open_s

matplotlib.use("Agg")


class MyDataSet:
    def __init__(
        self,
        custom_encoder: Optional[Union[CustomOneHotEncoder, CustomLabelEncoder]] = None,
        predict_field="model_execute_time",
        test_size=0.1,
        shuffle=True,
        op_algorithm: OpAlgorithm = OpAlgorithm.EXPECTED,
    ):
        self.predict_field = predict_field
        self.test_size = test_size
        self.shuffle = shuffle
        if custom_encoder:
            self.custom_encoder = custom_encoder
        else:
            self.custom_encoder = CustomOneHotEncoder()
        self.features = None
        self.labels = None
        self.load_data = None
        self.op_algorithm = op_algorithm
        self.sub_columns = []
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None

    @staticmethod
    def convert_batch_info(row: str, index: str) -> DataFrame:
        origin_row = ast.literal_eval(row)
        origin_index = ast.literal_eval(index)
        v, col = PreprocessTool.generate_data(origin_row, origin_index)
        return pd.DataFrame((v,), columns=col)

    @staticmethod
    def convert_request_info(row: str, index: str) -> DataFrame:
        origin_index = ast.literal_eval(index)
        origin_row = ast.literal_eval(row)
        request_info = namedtuple("request_info", origin_index)
        _row_request_info = tuple([request_info(*[int(float(i)) for i in _row]) for _row in origin_row])
        v, col = PreprocessTool.generate_data_with_request_info(_row_request_info, origin_index)
        return pd.DataFrame((v,), columns=col)

    @staticmethod
    def convert_request_info_by_df(row: str, index: str) -> DataFrame:
        origin_index = ast.literal_eval(index)
        origin_row = ast.literal_eval(row)
        v, col = PreprocessTool.generate_data_with_request_info_by_df(origin_row, origin_index)
        return pd.DataFrame((v,), columns=col)

    @staticmethod
    def convert_op_info(row: str, index: str) -> DataFrame:
        origin_index = ast.literal_eval(index)
        origin_row = ast.literal_eval(row)
        v, col = PreprocessTool.generate_data_with_op_info(origin_row, origin_index)
        return pd.DataFrame((v,), columns=col)

    @staticmethod
    def convert_op_info_with_ratio(row: str, index: str) -> DataFrame:
        origin_index = ast.literal_eval(index)
        origin_row = ast.literal_eval(row)
        v, col = PreprocessTool.generate_data_with_op_info_use_ratio(origin_row, origin_index)
        return pd.DataFrame((v,), columns=col)

    @staticmethod
    def convert_struct_info(row: str, index: str) -> DataFrame:
        origin_index = ast.literal_eval(index)
        origin_row = ast.literal_eval(row)
        v, col = PreprocessTool.generate_data_with_struct_info(origin_row, origin_index)
        return pd.DataFrame((v,), columns=col)

    @staticmethod
    def convert_config_info(row: str, index: str) -> DataFrame:
        origin_index = ast.literal_eval(index)
        origin_row = ast.literal_eval(row)
        v, col = PreprocessTool.generate_data_with_model_config(origin_row, origin_index)
        return pd.DataFrame((v,), columns=col)

    @staticmethod
    def convert_mindie_info(row: str, index: str) -> DataFrame:
        origin_index = ast.literal_eval(index)
        origin_row = ast.literal_eval(row)
        v, col = PreprocessTool.generate_data(origin_row, origin_index)
        return pd.DataFrame((v,), columns=col)

    @staticmethod
    def convert_env_info(row: str, index: str) -> DataFrame:
        origin_index = ast.literal_eval(index)
        origin_row = ast.literal_eval(row)
        v, col = PreprocessTool.generate_data(origin_row, origin_index)
        return pd.DataFrame((v,), columns=col)

    @staticmethod
    def convert_hardware_info(row: str, index: str) -> DataFrame:
        origin_index = ast.literal_eval(index)
        origin_row = ast.literal_eval(row)
        v, col = PreprocessTool.generate_data(origin_row, origin_index)
        return pd.DataFrame((v,), columns=col)

    @staticmethod
    def get_all_request_info(row: str, index: str) -> DataFrame:
        # 获取所有request原始数据特征，用来分析原始数据
        origin_index = ast.literal_eval(index)
        origin_row = ast.literal_eval(row)
        _row_request_info = []
        for _row in origin_row:
            _row_request_info.append([int(float(i)) for i in _row])
        return pd.DataFrame(_row_request_info, columns=origin_index)
    
    @staticmethod
    def plot_custom_pairplot(
        df: DataFrame, middle_save_path: Optional[Path] = None, file_name: str = "pairplot.png"
    ):
        col_num = df.shape[1]
        fig, axs = plt.subplots(col_num, col_num, figsize=(4 * col_num, 4 * col_num))
        for i in range(col_num):
            for j in range(col_num):
                if i == j:
                    if df.columns[i].lower() in ["max_seq_len", "input_length", "total_prefill_token"]:
                        sns.histplot(df.iloc[:, i], ax=axs[i, j], bins=100)
                    else:
                        sns.histplot(df.iloc[:, i], ax=axs[i, j])
                elif j > i:
                    if df.columns[i].lower() == "model_execute_time" and j == i + 1:
                        sns.histplot(df.iloc[:, i], ax=axs[i, j], bins=10000)
                    elif df.columns[i].lower() == "model_execute_time" and j == i + 2:
                        sns.scatterplot(x=df.reset_index().index, y=df["model_execute_time"], ax=axs[i, j])
                    elif df.columns[j].lower() == "model_execute_time" and i == j - 1:
                        sns.scatterplot(x=df["batch_size"], y=df["model_execute_time"], ax=axs[i, j])
                    elif df.columns[i].lower() == "batch_size" and j == i + 1:
                        sns.scatterplot(x=df.reset_index().index, y=df["batch_size"], ax=axs[i, j])
                    continue
                else:
                    sns.regplot(x=df.iloc[:, i], y=df.iloc[:, j], ax=axs[i, j])
        plt.tight_layout()
        if middle_save_path:
            plt.savefig(middle_save_path.joinpath(file_name))
        else:
            plt.show()
        plt.close()
        
    def analysis_batch_feature(self, middle_save_path: Optional[Path] = None):
        cur_batch_df = self.load_data.iloc[:, 0:len(self.sub_columns[0])]
        custom_label_encoder = CustomLabelEncoder([preset_category_data[0]])
        custom_label_encoder.fit()
        cur_batch_df = custom_label_encoder.transformer(cur_batch_df)
        try:
            sns.scatterplot(x=cur_batch_df.reset_index().index, y=cur_batch_df["batch_size"])
            plt.xlabel = "index"
            plt.savefig(middle_save_path.joinpath("index_batch_size.png"))
            plt.close()
            self.plot_custom_pairplot(cur_batch_df, middle_save_path, "batch_pairplot.png")
        except Exception as e:
            logger.error(f"analysis_batch_feature {e}")

    def construct_data(
        self, lines_data: Optional[DataFrame] = None, plt_data: bool = False, middle_save_path: Optional[Path] = None
    ):
        logger.debug(f"start construct_data, shape {lines_data.shape}")
        features, labels = self.preprocess_dispatch(lines_data)
        if self.features.shape[0] != self.labels.shape[0]:
            logger.error(
                f"Failed construct_data, because the shapes of features and labels do not match. "
                f"features shape {self.features.shape}, labels shape {self.labels.shape}"
            )
        if features.shape[0] == 1:
            self.train_x = self.test_x = features
            self.train_y = self.test_y = labels
        else:
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            features, labels, test_size=self.test_size, shuffle=self.shuffle
        )
        logger.debug("finished preprocess.")
        # 检查处理的数据是否有重复的column name
        if len(self.features.columns) != len(self.features.columns.unique()):
            raise ValueError("Duplicate columns exist in the data.")
        if plt_data:
            self.plt_data(lines_data, middle_save_path)

    def preprocess(self, lines_data: Optional[DataFrame] = None):
        logger.debug("dataset preprocess.")
        # 数据预处理
        if len(lines_data.columns) < 2:
            logger.error(f"DataFrame for train with swifter 列数不足，实际列数为 {len(lines_data.columns)}")
            return None, None
        # 将各个特征数据转换为列数据
        batch_df = pd.concat(lines_data.iloc[:, 0].apply(self.convert_batch_info, args=(lines_data.columns[0],)).values)
        request_df = pd.concat(
            lines_data.iloc[:, 1].apply(self.convert_request_info, args=(lines_data.columns[1],)).values
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
                    df = pd.concat(lines_data[col].apply(func, args=(col,)).values)
                    self.sub_columns.append(df.columns.tolist())
                    _load_data.append(df)

        # 提取 features 和labels
        self.load_data = pd.concat(_load_data, axis=1)
        self.labels = self.load_data[[self.predict_field]]
        self.features = self.load_data.drop(self.predict_field, axis=1)
        # 使用sklearn 进行 one-hot
        self.features = self.custom_encoder.transformer(self.features)
        return self.features, self.labels

    def preprocess_with_list_comprehension(self, lines_data: Optional[DataFrame] = None):
        logger.debug("dataset preprocess with list comprehension")
        # 数据预处理
        columns_list = lines_data.columns.tolist()
        field_cache = {col: ast.literal_eval(col) for col in columns_list}
        batch_df = pd.concat([self.convert_batch_info(item, columns_list[0]) for item in lines_data.iloc[:, 0]], axis=0)
        request_df = pd.concat(
            [self.convert_request_info_by_df(item, columns_list[1]) for item in lines_data.iloc[:, 1]], axis=0
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
        logger.debug("finished request batch")
        if len(columns_list) < 3:
            logger.warning(f"columns_list length is less than 3, skip processing")
            return self.features, self.labels
        
        for col in columns_list[2:]:
            field_type = field_cache[col]
            if field_type in convert_funcs:
                func = convert_funcs[field_type]
                df = pd.concat([func(item, col) for item in lines_data[col]], axis=0)
                self.sub_columns.append(df.columns.tolist())
                _load_data.append(df)

        # 提取 features 和labels
        self.load_data = pd.concat(_load_data, axis=1)
        self.labels = self.load_data[[self.predict_field]]
        self.features = self.load_data.drop(self.predict_field, axis=1)
        # 使用sklearn 进行 one-hot
        self.features = self.custom_encoder.transformer(self.features)
        return self.features, self.labels

    def preprocess_dispatch(self, lines_data: Optional[DataFrame] = None):
        try:
            features, labels = self.preprocess_with_list_comprehension(lines_data)
        except Exception as e:
            logger.error(f"Failed preprocess with list comprehension. error: {e}")
            features, labels = self.preprocess(lines_data)
        return features, labels

    def plt_data(self, line_data: DataFrame, middle_save_path: Optional[Path] = None):
        self.analysis_batch_feature(middle_save_path)
        self.analysis_origin_request_hist(line_data, middle_save_path)

    def analysis_origin_request_hist(self, df: DataFrame, middle_save_path: Optional[Path] = None):
        logger.debug("analysis_origin_request_hist")
        if len(df.columns) < 2:
            logger.warning("Dataframe has less than 2 columns.")
            return 

        request_series = df.iloc[:, 1].apply(self.get_all_request_info, args=(df.columns[1],))
        request_df = pd.concat(request_series.values, ignore_index=True)
        try:
            self.plot_custom_pairplot(request_df, middle_save_path, "request_pairplot.png")
        except Exception as e:
            logger.error(f"error occur when plot request pairplot: {e}")

