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
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from src.optimize.experimental.inference.utils import (
    PreprocessTool, HistInfo, OP_EXPECTED_FIELD_MAPPING,
    OperatorProcessingConfig, _preprocess_dataframe,
    RowData, OpData
)


class TestPreprocessTool(unittest.TestCase):
    def setUp(self):
        # 测试用请求数据
        self.sample_request_row = (
            ("req1", 10, 2, 5),
            ("req2", 15, 3, 8)
        )
        self.request_columns = ("request_id", "input_length", "block_count", "output_length")
        
        # 测试用算子数据
        self.sample_op_row = (
            ("MatMul", "2", "float32;float32", "100,200;50,60", "200,100", "0.5", "0.1"),
            ("Add", "1", "float32", "30,40", "30,40", "0.2", "0.05")
        )
        self.op_columns = ("op_name", "call_count", "input_dtype", "input_shape", "output_shape", 
                          "execute_delta", "execute_delta_ratio")

    def test_generate_data_basic(self):
        """测试基础数据生成"""
        row = ("1", "2.5", "text")
        columns = ("id", "value", "desc")
        
        result, new_columns = PreprocessTool.generate_data(row, columns)
        
        self.assertEqual(result, (1.0, 2.5, "text"))
        self.assertEqual(new_columns, columns)

    def test_generate_data_with_request_info_by_df(self):
        """测试DataFrame模式的请求信息生成"""
        # 构造
        row = ((1015, 19, 0),)
        columns = ("input_length", "need_blocks", "output_length")
        
        result, new_columns = PreprocessTool.generate_data_with_request_info_by_df(row, columns)
        
        self.assertEqual(result[0], 0.0)  # 总输出长度
        self.assertIn("total_output_length", new_columns)


    @patch('src.optimize.experimental.inference.utils.PreprocessTool.get_op_in_origin_row_index')
    @patch('src.optimize.experimental.inference.utils.PreprocessTool.get_all_op_input_ratio')
    @patch('src.optimize.experimental.inference.utils.PreprocessTool.get_all_op_execute_delta_ratio')
    @patch('src.optimize.experimental.inference.utils.PreprocessTool.get_label_hist_value')
    def test_generate_data_with_op_info_use_ratio_empty_input(
        self,
        mock_get_label_hist_value,
        mock_get_all_op_execute_delta_ratio,
        mock_get_all_op_input_ratio,
        mock_get_op_in_origin_row_index
    ):
        """测试算子信息生成"""
        # 模拟输入
        origin_row = ()
        origin_index = ()

        # 模拟返回值
        mock_get_op_in_origin_row_index.return_value = {}
        mock_get_all_op_input_ratio.return_value = {}
        mock_get_all_op_execute_delta_ratio.return_value = {}
        mock_get_label_hist_value.return_value = {}

        # 调用方法
        radio_result = PreprocessTool.generate_data_with_op_info_use_ratio(origin_row, origin_index)
        result = PreprocessTool.generate_data_with_op_info(origin_row, origin_index)
        # 验证结果
        self.assertEqual(sum(radio_result[0]), 0)
        self.assertEqual(sum(result[0]), 0)

    def test_generate_data_with_struct_info(self):
        """测试结构信息转换（比率放大）"""
        row = (0.15, 100)
        columns = ("hit_rate", "count")
        
        result, new_columns = PreprocessTool.generate_data_with_struct_info(row, columns)
        
        self.assertEqual(result[0], 15)  # 0.15 -> 15%
        self.assertEqual(result[1], 100.0)

    def test_generate_data_with_model_config(self):
        """测试模型配置信息生成"""
        row = (
            {"kv_quant_type": "int8", "group_size": 64},  # quantization_config
            ["Transformer"],  # architectures
            True  # quantize
        )
        columns = ("quantization_config", "architectures", "quantize")
        
        result, new_columns = PreprocessTool.generate_data_with_model_config(row, columns)
        
        # 验证量化配置
        self.assertEqual(result[new_columns.index("kv_quant_type")], "int8")
    
    def test_get_all_op_input_ratio_empty_input(self):
        # 测试空输入的情况
        input_data = ()
        input_index = ("op_name", "call_count", "input_shape")
        result = PreprocessTool.get_all_op_input_ratio(input_data, input_index)
        self.assertEqual(result, {})

    def test_get_all_op_input_ratio_single_input(self):
        # 测试单个输入的情况
        input_data = (("conv2d", "1", "1,2,3"),)
        input_index = ("op_name", "call_count", "input_shape")
        result = PreprocessTool.get_all_op_input_ratio(input_data, input_index)
        self.assertEqual(result, {"conv2d": {0: [100.0]}})

    def test_get_all_op_input_ratio_multiple_inputs(self):
        # 测试多个输入的情况
        input_data = (("conv2d", "2", "1,2,3"), ("maxpool", "1", "2,3"))
        input_index = ("op_name", "call_count", "input_shape")
        result = PreprocessTool.get_all_op_input_ratio(input_data, input_index)
        self.assertEqual(result, {"conv2d": {0: [50.0, 50.0]}, 'maxpool': {0: [50.0]}})

    def test_generate_data_with_request_info_by_df_negative_value(self):
        """测试负值检查机制"""
        row = ((-1, -2, -3), (-4, -5, -6))  # 添加output_length列的负值
        column = ("output_length", "hist1", "hist2")
        with self.assertRaises(ValueError):
            PreprocessTool.generate_data_with_request_info_by_df(row, column)


class TestGenerateDataWithRequestInfo(unittest.TestCase):
    def setUp(self):
        self.output_length_field = "output_length"
        self.total_output_length = "total_output_length"

    def test_generate_data_with_request_info_normal_case(self):
        # 准备测试数据
        row = (
            MagicMock(**{self.output_length_field: 5}),
            MagicMock(**{self.output_length_field: 15}),
            MagicMock(**{self.output_length_field: 25})
        )
        column = (self.output_length_field, "input_length", "need_blocks")

        # 模拟get_field_bins_count的返回值
        with patch('src.optimize.experimental.inference.common.get_field_bins_count') as \
            mock_get_field_bins_count:
            mock_get_field_bins_count.side_effect = [
                [0, 2, 1],  # 对于input_length的返回值
                [1, 0, 1],  # 对于need_blocks的返回值
                [0, 4, 1]   # 对于output_length的返回值
            ]

            # 调用方法
            expected_new_index = (
                self.total_output_length,
                "input_length_0-80", "input_length_80-160", "input_length_160-240",
                "need_blocks_0-1", "need_blocks_1-2", "need_blocks_2-3"
            )
            result = PreprocessTool.generate_data_with_request_info(row, column)

            # 验证结果
            self.assertEqual(sum(result[0]), 54)

    def test_generate_data_with_request_info_empty_row(self):
        # 测试空row的情况
        row = ()
        column = (self.output_length_field, "input_length", "need_blocks")

        # 预期行为：返回空的元组
        result = PreprocessTool.generate_data_with_request_info(row, column)
        self.assertEqual(sum(result[0]), 0)

    def test_generate_data_with_request_info_empty_column(self):
        # 测试空column的情况
        row = (
            MagicMock(**{self.output_length_field: 5}),
            MagicMock(**{self.output_length_field: 15}),
            MagicMock(**{self.output_length_field: 25})
        )
        column = ()

        # 预期行为：返回空的元组
        result = PreprocessTool.generate_data_with_request_info(row, column)
        self.assertEqual(result[0], ())
        self.assertEqual(result[1], ())

    def test_get_all_op_execute_delta_ratio_empty_input(self):
        # 测试空输入
        input_data = ()
        input_index = ("op_name", "call_count", "execute_delta")
        result = PreprocessTool.get_all_op_execute_delta_ratio(input_data, input_index)
        self.assertEqual(result, {})

    def test_get_all_op_execute_delta_ratio_single_input(self):
        # 测试单个输入
        input_data = (("op1", 1, 1.0),)
        input_index = ("op_name", "call_count", "execute_delta")
        result = PreprocessTool.get_all_op_execute_delta_ratio(input_data, input_index)
        self.assertEqual(result, {"op1": [1.0]})

    def test_get_all_op_execute_delta_ratio_multiple_inputs(self):
        # 测试多个输入
        input_data = (("op1", 2, 2.0), ("op2", 1, 1.0))
        input_index = ("op_name", "call_count", "execute_delta")
        result = PreprocessTool.get_all_op_execute_delta_ratio(input_data, input_index)
        self.assertEqual(result, {'op1': [0.6666666666666666, 0.6666666666666666], 'op2': [0.3333333333333333]})

    def test_get_all_op_execute_delta_ratio_same_op_name(self):
        # 测试相同操作名
        input_data = (("op1", 2, 2.0), ("op1", 1, 1.0))
        input_index = ("op_name", "call_count", "execute_delta")
        result = PreprocessTool.get_all_op_execute_delta_ratio(input_data, input_index)
        self.assertEqual(result, {'op1': [0.6666666666666666, 0.6666666666666666, 0.3333333333333333]})

    def test_get_all_op_execute_delta_ratio_different_field(self):
        # 测试不同的字段
        input_data = (("op1", 2, 2.0), ("op2", 1, 1.0))
        input_index = ("op_name", "call_count", "execute_delta")
        result = PreprocessTool.get_all_op_execute_delta_ratio(input_data, input_index, field="execute_delta")
        self.assertEqual(result, {'op1': [0.6666666666666666, 0.6666666666666666], 'op2': [0.3333333333333333]})


class TestProcessOperatorInfo(unittest.TestCase):
    def setUp(self):
        self.config = OperatorProcessingConfig(
            origin_row=[],
            origin_index=[],
            op_index_on_origin_rows=[],
            dtype_category=[],
            op_input_param_expected={},
            op_output_expected={},
            op_execute_delta_field=[],
            op_delta_expected={},
            op=""
        )

    def test_op_name(self):
        new_row = []
        result = PreprocessTool.process_operator_info("op_name", "value", self.config, new_row)
        self.assertEqual(result, [1])

    def test_call_count(self):
        self.config.origin_row = [["10"], ["20"]]
        self.config.origin_index = ["call_count"]
        self.config.op_index_on_origin_rows = [0, 1]
        new_row = []
        result = PreprocessTool.process_operator_info("call_count", "value", self.config, new_row)
        self.assertEqual(result, [30])

    def test_input_dtype(self):
        self.config.origin_row = [["input_dtype", "int;float"], ["input_dtype", "float"]]
        self.config.origin_index = ["input_dtype"]
        self.config.op_index_on_origin_rows = [0, 1]
        self.config.dtype_category = ["int", "float"]
        new_row = []
        result = PreprocessTool.process_operator_info("input_dtype__int", "value", self.config, new_row)
        self.assertEqual(result, [0])

    def test_input_size(self):
        self.config.op_input_param_expected = {0: "10"}
        new_row = []
        result = PreprocessTool.process_operator_info("input_size__0", "value", self.config, new_row)
        self.assertEqual(result, ["10"])

    def test_output_dtype(self):
        self.config.origin_row = [["output_dtype", "int;float"], ["output_dtype", "float"]]
        self.config.origin_index = ["output_dtype"]
        self.config.op_index_on_origin_rows = [0, 1]
        self.config.dtype_category = ["int", "float"]
        new_row = []
        result = PreprocessTool.process_operator_info("output_dtype__int", "value", self.config, new_row)
        self.assertEqual(result, [0])

    def test_output_size(self):
        self.config.op_output_expected = {0: "10"}
        new_row = []
        result = PreprocessTool.process_operator_info("output_size__0", "value", self.config, new_row)
        self.assertEqual(result, ["10"])

    def test_op_execute_delta_field(self):
        self.config.op_execute_delta_field = ["field"]
        self.config.op_delta_expected = {"field": {"op": "10"}}
        self.config.op = "op"
        new_row = []
        result = PreprocessTool.process_operator_info("field", "value", self.config, new_row)
        self.assertEqual(result, ["10"])


class TestPreprocessDataFrame(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': ["1", "2", "3"],
            'B': ["4", "5", "6"],
        })

    def test_all_columns_valid(self):
        # 测试所有检查的列都有效的情况
        result = _preprocess_dataframe(self.df, check_columns=['A', 'B'])
        self.assertTrue(result)

    def test_some_columns_invalid(self):
        # 测试一些检查的列包含无效值的情况
        self.df.loc[0, 'A'] = "+A3"  # 添加一个无效值
        result = _preprocess_dataframe(self.df, check_columns=['A', 'B'])
        self.assertFalse(result)

    @patch('src.optimize.experimental.inference.utils.logger')
    def test_warning_called(self, mock_warning):
        # 测试当列包含无效值时，logger.warning 是否被调用
        self.df.loc[0, 'A'] = "=A2+A3"  # 添加一个无效值
        _preprocess_dataframe(self.df, check_columns=['A', 'B'])
        mock_warning.warning.assert_called_once_with("Column A contains malicious values")


class TestProcessRowData(unittest.TestCase):
    def setUp(self):
        # 创建模拟数据
        self.row_data = RowData(
            origin_row=[["input_dtype", "int32;float32"], ["output_dtype", "int32;float32"]],
            origin_index=["input_dtype", "output_dtype"],
            op_index_on_origin_rows=[0, 1],
            dtype_category=["int32", "float32"]
        )

        self.op_data = OpData(
            op="test_op",
            op_input_param_hist_ratio={"test_op": {"0__input_size__1": 0}},
            op_output_hist_ratio={"test_op": {"0__output_size__1": 0}},
            op_delta_hist_ratio={"test_op": {"0__execute_delta__1": 0}},
        )

    def test_op_name(self):
        new_row = []
        self.row_data.dtype_category = ["int32", "float32"]
        self.row_data.op_index_on_origin_rows = [0, 1]
        self.row_data.origin_row = [["input_dtype", "int32;float32"], ["output_dtype", "int32;float32"]]
        self.row_data.origin_index = ["input_dtype", "output_dtype"]
        PreprocessTool.process_row_data("op_name", self.row_data, self.op_data, new_row)
        self.assertEqual(new_row, [1])

    def test_input_dtype(self):
        new_row = []
        PreprocessTool.process_row_data("input_dtype__int32", self.row_data, self.op_data, new_row)
        self.assertEqual(new_row, [0])

    def test_input_size(self):
        new_row = []
        PreprocessTool.process_row_data("input_size__0__input_size__1", self.row_data, self.op_data, new_row)
        self.assertEqual(new_row, [0])

    def test_output_dtype(self):
        new_row = []
        PreprocessTool.process_row_data("output_dtype__int32", self.row_data, self.op_data, new_row)
        self.assertEqual(new_row, [2])

    def test_output_size(self):
        new_row = []
        PreprocessTool.process_row_data("output_size__0__output_size__1", self.row_data, self.op_data, new_row)
        self.assertEqual(new_row, [0])

    def test_execute_delta(self):
        new_row = []
        PreprocessTool.process_row_data("execute_delta__0__execute_delta__1", self.row_data, self.op_data, new_row)
        self.assertEqual(new_row, [0])