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
系统状态评估
1. 需要安装xgboost, pandas, numpy, sklearn
2. 获取训练好的模型及编码器。

功能：
1. 预测v1版本的数据。参考： example


"""
import os
import atexit
import queue
import signal
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Thread
from typing import List, Optional, Tuple, Sequence

import pandas as pd
import xgboost
from loguru import logger
from pandas import DataFrame
from xgboost import DMatrix

from msserviceprofiler.modelevalstate.config.config import get_settings
from msserviceprofiler.modelevalstate.inference.data_format_v1 import BatchField, RequestField, ConfigPath
from msserviceprofiler.modelevalstate.inference.dataset import InputData, DataProcessor, \
    CustomLabelEncoder, preset_category_data
from msserviceprofiler.modelevalstate.inference.file_reader import FileHanlder, StaticFile
from msserviceprofiler.modelevalstate.common import read_csv_s
from msserviceprofiler.modelevalstate.inference.utils import save_dataframe_to_csv


sub_thread = None
predict_queue = queue.Queue()


class CachePredict:
    def __init__(self, data_path: Path, data: Optional[DataFrame] = None, label_name: str = "label"):
        self.label_name = label_name
        self.data_path = data_path
        self.data = None
        self.label = None
        if data is not None:
            self.label = data[label_name]
            self.data = data.drop(label_name, axis=1)
        elif data_path and data_path.exists():
            read_datas = [read_csv_s(_child) for _child in data_path.iterdir()]
            if read_datas:
                data = pd.concat(read_datas)
                self.label = data[label_name]
                self.data = data.drop(label_name, axis=1)
        self.new_data = None
        self.new_label = None
        self.output = data_path.joinpath(f"train_{os.getpid()}_{datetime.today().strftime('%Y%m%d%H%M%S')}.csv")
        if not self.output.parent.exists():
            self.output.parent.mkdir(parents=True, mode=0o750)

    def update(self, data: List, label: float):
        _compare_data = [round(k) for k in data]
        if self.data is not None:
            history_exists = (round(self.data) == _compare_data).all(axis=1).any()
            if history_exists:
                return

        if self.new_data is None and self.new_label is None:
            if self.data is None:
                self.new_data = pd.DataFrame([data])
                self.new_label = pd.Series([label], name=self.label_name)
            else:
                self.new_data = pd.DataFrame([data], columns=self.data.columns)
                self.new_label = pd.Series([label], name=self.label.name)
            return
        current_exists = (round(self.new_data) == _compare_data).all(axis=1).any()
        if current_exists:
            return
        self.new_data.loc[len(self.new_data)] = {k: v for k, v in zip(self.new_data.columns, data)}
        self.new_label.loc[len(self.new_label)] = label

    def save(self):
        if self.new_data is None and self.new_label is None:
            return
        data = self.new_data.copy()
        data[self.new_label.name] = self.new_label
        parent_dir, filename = os.path.split(self.output)
        save_dataframe_to_csv(data, parent_dir, filename)


def update_cache(cache_predict: Optional[CachePredict], persistent_threshold: int = 100):
    while True:
        flag = False
        items = []
        while not predict_queue.empty():
            items.append(predict_queue.get())
        for res in items:
            if res is None:
                flag = True
                break
            cache_predict.update(*res)
        if cache_predict.new_data is not None and len(cache_predict.new_data) > persistent_threshold:
            cache_predict.save()
        if flag:
            break
        time.sleep(1)
    cache_predict.save()


def signal_process():
    predict_queue.put(None)
    if sub_thread:
        sub_thread.join(timeout=3)
        if sub_thread.is_alive():
            raise TimeoutError("子线程未在指定时间完成")
    

atexit.register(signal_process)


class XGBStateEvaluate:
    """
    1. 预处理数据
    2. 进行预测
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(XGBStateEvaluate, cls).__new__(cls)
        return cls._instance

    def __init__(self, xgb_model_path: Path, dataprocessor: DataProcessor, cache_data: Optional[Path] = None):
        if not XGBStateEvaluate._initialized:
            self.xgb_model_path = xgb_model_path
            self.prefill_type = "prefill"
            self.decode_type = "decode"
            cache = None
            if cache_data:
                global sub_thread
                try:
                    cache, cache_predict = self.load_cache_predict(cache_data)
                    sub_thread = Thread(target=update_cache, args=(cache_predict,))
                    sub_thread.start()
                except Exception as ee:
                    logger.exception(f"cache failure {ee}")
            self.xgb_model = self.load_model(self.xgb_model_path, cache)
            self.data_processor = dataprocessor
            XGBStateEvaluate._initialized = True


    @staticmethod
    def load_model(model_path, cache: Optional[Sequence[DMatrix]] = None):
        # 获取当前在那个gpu上
        try:
            import torch
            import torch_npu
            params = {"device": f"gpu:{torch.npu.current_device()}"}
        except ImportError:
            params = {}
        _model = xgboost.Booster(params=params, cache=cache)
        _model.load_model(model_path)
        return _model

    @staticmethod
    def load_cache_predict(cache_data: Optional[Path] = None):
        cache = None
        read_datas = []
        if cache_data and cache_data.exists():
            for _child in cache_data.iterdir():
                p = Path(_child)
                if p.stat().st_size == 0 and p.is_file():
                    p.unlink()
                    continue
                try:
                    _df = read_csv_s(_child)
                    read_datas.append(_df)
                except (FileNotFoundError, pd.errors.EmptyDataError, RuntimeError) as e:
                    logger.error("Failed in read cache data. cache data {}, child {}, error: {}",
                                 cache_data, _child, e)
        if read_datas:
            data = pd.concat(read_datas)
            data = data.dropna()
            _label = "label"
            if _label not in data.columns:
                _label = data.columns[-1]
            label = data[_label]
            cache = (xgboost.DMatrix(data.drop(_label, axis=1), label),)
            cache_predict = CachePredict(cache_data, data, label_name=_label)
        else:
            cache_predict = CachePredict(get_settings().latency_model.cache_data)
        return cache, cache_predict

    def predict(self, input_data: InputData) -> Tuple[float, float]:
        """
        根据输入预测Up，Ud.
        :param config_info: 运行系统的配置信息
        :param input_data: 运行系统的状态
        :return: Up, Ud
        """
        # 1. 处理数据
        stage = input_data.batch_field.batch_stage.lower()
        line_info = self.data_processor.preprocessor(input_data)
        # 2. 进行预测
        res = self.xgb_model.predict(xgboost.DMatrix([line_info, ], feature_names=self.xgb_model.feature_names))[
            0].item()
        if sub_thread:
            predict_queue.put_nowait((line_info, res))
        _up = _ud = -1
        if stage == self.prefill_type:
            _up = res
        elif stage == self.decode_type:
            _ud = res
        else:
            raise ValueError(f"Data error. expected Data Type {self.prefill_type, self.decode_type}. got is {stage}")
        return _up, _ud


# 接口提供三个参数1个batch字段，1个request字段，1个config 字段。

@lru_cache(maxsize=32)
def predict_v1(batch_info: BatchField, request_info: Tuple[RequestField, ...], config_path: ConfigPath):
    # 读取其他字段数据
    static_file = StaticFile(base_path=config_path.static_file_dir)
    fh = FileHanlder(static_file)
    fh.load_static_data()
    # 组合为input data
    input_data = InputData(
        batch_field=batch_info,
        request_field=request_info,
        model_op_field=fh.get_op_field(
            batch_info.batch_stage, batch_info.batch_size, batch_info.max_seq_len, fh.prefill_op_data, fh.decode_op_data
        ),
        model_struct_field=fh.model_struct_info,
        model_config_field=fh.model_config_info,
        mindie_field=fh.mindie_info,
        env_field=fh.env_info,
        hardware_field=fh.hardware
    )
    # 进行预测
    custom_encoder = CustomLabelEncoder(preset_category_data)
    custom_encoder.fit()
    # 加载模型
    data_processor = DataProcessor(custom_encoder)
    xgb_state_eval = XGBStateEvaluate(xgb_model_path=config_path.model_path, dataprocessor=data_processor)
    # 预测
    res = xgb_state_eval.predict(input_data)
    return res


def predict_v1_with_cache(
        batch_info: BatchField,
        request_info: Tuple[RequestField, ...],
        config_path: ConfigPath,
        fh: FileHanlder,
        data_processor: DataProcessor,
        **kwargs
):
    # 组合为input data
    input_data = InputData(
        batch_field=batch_info,
        request_field=request_info,
        model_op_field=fh.get_op_field(
            batch_info.batch_stage, batch_info.batch_size, batch_info.max_seq_len, fh.prefill_op_data, fh.decode_op_data
        ),
        model_struct_field=fh.model_struct_info,
        model_config_field=fh.model_config_info,
        mindie_field=fh.mindie_info,
        env_field=fh.env_info,
        hardware_field=fh.hardware,
    )
    xgb_state_eval = XGBStateEvaluate(xgb_model_path=config_path.model_path, dataprocessor=data_processor, **kwargs)
    # 预测
    res = xgb_state_eval.predict(input_data)
    return res
