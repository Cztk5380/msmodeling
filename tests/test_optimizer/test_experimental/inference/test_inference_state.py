# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, List, Callable, Dict
import unittest
from unittest.mock import patch, MagicMock, Mock
from pandas import DataFrame

import numpy as np
import pandas as pd
from src.optimize.experimental.data_feature.v1 import FileReader, BATCH_FIELD
from src.optimize.experimental.inference.state_eval_v1 import XGBStateEvaluate, predict_v1, CachePredict, \
    predict_v1_with_cache
from src.optimize.experimental.inference.dataset import CustomOneHotEncoder, CustomLabelEncoder, InputData,\
    preset_category_data, DataProcessor
from src.optimize.experimental.inference.data_format_v1 import ConfigPath, ModelOpField, ModelStruct, \
    ModelConfig, MindieConfig, EnvField, HardWare, RequestField, BatchField
from src.optimize.experimental.inference.data_format_v1 import REQUEST_FIELD, MODEL_OP_FIELD, \
    MODEL_STRUCT_FIELD, MODEL_CONFIG_FIELD, MINDIE_FIELD, ENV_FIELD, HARDWARE_FIELD
from src.optimize.experimental.train.pretrain import NodeInfo
from src.optimize.experimental.inference.file_reader import FileHanlder, StaticFile


def test_update_new_data_none(tmpdir):
    cache_predict = CachePredict(Path(tmpdir))
    cache_predict.update([1, 2, 3], 1.0)
    assert cache_predict.new_data.equals(pd.DataFrame([1, 2, 3]).T)
    assert cache_predict.new_label.equals(pd.Series([1.0], name=cache_predict.label_name))


def test_update_data_none(tmpdir):
    cache_predict = CachePredict(Path(tmpdir))
    cache_predict.new_data = pd.DataFrame([1, 2, 3]).T
    cache_predict.new_label = pd.Series([1.0], name=cache_predict.label_name)
    cache_predict.update([4, 5, 6], 2.0)
    assert cache_predict.new_data.equals(pd.DataFrame([[1, 2, 3], [4, 5, 6]]))
    assert cache_predict.new_label.equals(pd.Series([1.0, 2.0], name=cache_predict.label_name))
    cache_predict.save()
    assert cache_predict.output.exists()


def test_update_data_exists(tmpdir):
    cache_predict = CachePredict(Path(tmpdir))
    cache_predict.data = pd.DataFrame([1, 2, 3]).T
    cache_predict.label = pd.Series([1.0], name=cache_predict.label_name)
    cache_predict.update([1, 2, 3], 1.0)
    assert cache_predict.data.equals(pd.DataFrame([1, 2, 3]).T)
    assert cache_predict.label.equals(pd.Series([1.0], name=cache_predict.label_name))


@patch('src.optimize.experimental.inference.state_eval_v1.DataProcessor')
@patch('src.optimize.experimental.inference.state_eval_v1.XGBStateEvaluate')
def test_predict_v1(mock_data_processor, mock_xgb_state_evaluate, tmpdir, static_file):
    mock_data_processor.return_value = MagicMock()
    mock_xgb_state_evaluate.return_value = MagicMock()

    # Create the necessary objects
    batch_info = BatchField("decode", 20, 20.0, 580.0, 29.0)
    request_info = (
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
    )
    config_path = ConfigPath(
        Path(fr"{tmpdir}\xgb_model.ubj"),
        static_file.base_path,
        Path(fr"{tmpdir}\req_and_decode_file.json"),
        Path(fr"{tmpdir}\cache_data"),

    )

    # Call the method under test
    predict_v1(batch_info, request_info, config_path)

    # Assert that the necessary methods were called
    mock_data_processor.assert_called()
    mock_xgb_state_evaluate.assert_called()


class MockBooster:
    def __init__(self, *args, **kwargs):
        self.feature_names = None

    @staticmethod
    def predict(*args, **kwargs):
        return np.array([66666])
    
    def load_model(self, model_path):
        pass


@patch("xgboost.Booster", MockBooster)
def predict_with_model(lines_data: DataFrame,
                       xgb_model_path: Optional[Path] = None,
                       ohe_path: Optional[Path] = None,
                       train_field="model_execute_time",
                       dataset_type: DataProcessor = DataProcessor):
    # 转换格式为接口需要格式
    origin_data: List[NodeInfo] = []
    predict_data: List[NodeInfo] = []
    custom_encoder = CustomOneHotEncoder(preset_category_data)
    custom_encoder.fit()
    custom_encoder = CustomLabelEncoder(preset_category_data)
    custom_encoder.fit()
    data_processor = dataset_type(custom_encoder)
    xgb_state_eval = XGBStateEvaluate(
        xgb_model_path=Path(xgb_model_path),
        dataprocessor=data_processor)
    for _, row in lines_data.iterrows():
        batch_field, request_field, model_op_field, model_struct_field, model_config_field, mindie_field, env_field, \
            hardware_field = None, None, None, None, None, None, None, None
        for i, _cur_columns in enumerate(lines_data.columns):
            _cur_columns = eval(_cur_columns)
            if _cur_columns == BATCH_FIELD:
                # 获取真实结果
                batch_data = eval(row[i])
                batch_field = BatchField(*batch_data[:-1])
                _cur_node = NodeInfo(batch_field.batch_stage, batch_field.batch_size)
                setattr(_cur_node, train_field, float(batch_data[-1]))
                origin_data.append(_cur_node)
            elif _cur_columns == REQUEST_FIELD:
                request_field = tuple([RequestField(*[int(float(j)) for j in _req]) for _req in eval(row[i])])
            elif _cur_columns == MODEL_OP_FIELD:
                model_op_field = tuple([ModelOpField(*_op) for _op in eval(row[i])])
            elif _cur_columns == MODEL_STRUCT_FIELD:
                model_struct_field = ModelStruct(*eval(row[i]))
            elif _cur_columns == MODEL_CONFIG_FIELD:
                model_config_field = ModelConfig(*eval(row[i]))
            elif _cur_columns == MINDIE_FIELD:
                mindie_field = MindieConfig(*eval(row[i]))
            elif _cur_columns == ENV_FIELD:
                env_field = EnvField(*eval(row[i]))
            elif _cur_columns == HARDWARE_FIELD:
                hardware_field = HardWare(*eval(row[i]))

        input_data = InputData(
            batch_field=batch_field,
            request_field=request_field,
            model_op_field=model_op_field,
            model_struct_field=model_struct_field,
            model_config_field=model_config_field,
            mindie_field=mindie_field,
            env_field=env_field,
            hardware_field=hardware_field
        )
        # 使用模型进行预测
        _up, _ud = xgb_state_eval.predict(input_data)
        _cur_node = deepcopy(_cur_node)
        if _up != -1:
            setattr(_cur_node, train_field, _up)
        else:
            setattr(_cur_node, train_field, _ud)
        predict_data.append(_cur_node)


def run_case(process_num: int, save_result_path: Path, fl: FileReader, call_func: Callable, kwargs: Dict):
    count = 1
    with Pool(process_num) as p:
        while True:
            try:
                # 读取数据
                lines = fl.read_lines()
                # 增量拟合
                save_path = save_result_path.joinpath(str(count))
                save_path.mkdir(exist_ok=True, parents=True)
                if process_num == 1:
                    call_func(lines, save_path, **kwargs)
                else:
                    p.apply_async(call_func, args=(lines, save_path), kwds=kwargs)
                count += 1
            except StopIteration:
                break
        p.close()
        p.join()


@patch('src.optimize.experimental.inference.state_eval_v1.XGBStateEvaluate')
@patch('src.optimize.experimental.inference.state_eval_v1.InputData')
def test_predict_v1_with_cache(mock_input_data, mock_xgb_state_eval, tmpdir, static_file):
    mock_input_data.return_value = MagicMock()
    mock_xgb_state_eval.return_value = MagicMock()
    # Arrange
    batch_info = BatchField("decode", 20, 20.0, 580.0, 29.0)
    request_info = (
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
    )
    config_path = ConfigPath(
        Path(fr"{tmpdir}\xgb_model.ubj"),
        static_file.base_path,
        Path(fr"{tmpdir}\req_and_decode_file.json"),
        Path(fr"{tmpdir}\cache_data"),
    )
    static_file = StaticFile(base_path=static_file.base_path)
    fh = FileHanlder(static_file)
    fh.load_static_data()
    custom_encoder = CustomLabelEncoder(preset_category_data)
    custom_encoder.fit()
    data_processor = DataProcessor(custom_encoder)

    # Act
    result = predict_v1_with_cache(batch_info, request_info, config_path, fh, data_processor)

    # Assert
    mock_input_data.assert_called_once()
    mock_xgb_state_eval.assert_called_once_with(xgb_model_path=config_path.model_path, dataprocessor=data_processor)


class TestCachePredict(unittest.TestCase):
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_dataloader_with_data(self, mock_open, mock_exists):
        # 测试当data不为None时的情况
        data = pd.DataFrame({
            'label': [1, 2, 3],
            'feature1': [4, 5, 6],
            'feature2': [7, 8, 9]
        })
        loader = CachePredict(data_path=Path(""), data=data, label_name='label')
        self.assertEqual(loader.label.tolist(), [1, 2, 3])
        self.assertEqual(loader.data.columns.tolist(), ['feature1', 'feature2'])

    @patch('src.optimize.experimental.config.config.settings')
    @patch('src.optimize.experimental.inference.state_eval_v1.CachePredict')
    def test_no_cache_data(self, mock_cache_predict, mock_settings):
        mock_settings.latency_model.cache_data = 'default_cache_data'
        cache, _ = XGBStateEvaluate.load_cache_predict()
        self.assertIsNone(cache)

    @patch('src.optimize.experimental.config.config.settings')
    @patch('src.optimize.experimental.inference.state_eval_v1.CachePredict')
    def test_empty_cache_data(self, mock_cache_predict, mock_settings):
        mock_settings.latency_model.cache_data = 'default_cache_data'
        cache_data = Path('empty_cache_data')
        cache_data.mkdir(exist_ok=True)
        cache, _ = XGBStateEvaluate.load_cache_predict(cache_data)
        self.assertIsNone(cache)

    @patch('src.optimize.experimental.config.config.settings')
    @patch('src.optimize.experimental.inference.state_eval_v1.CachePredict')
    @patch('src.optimize.experimental.inference.state_eval_v1.read_csv_s')
    def test_non_empty_cache_data(self, mock_read_csv_s, mock_cache_predict, mock_settings):
        mock_settings.latency_model.cache_data = 'default_cache_data'
        cache_data = Path('non_empty_cache_data')
        cache_data.mkdir(exist_ok=True)
        (cache_data / 'file.csv').touch()
        mock_read_csv_s.return_value = pd.DataFrame({'label': [1], 'feature': [2]})
        cache, _ = XGBStateEvaluate.load_cache_predict(cache_data)
        self.assertIsNone(cache)