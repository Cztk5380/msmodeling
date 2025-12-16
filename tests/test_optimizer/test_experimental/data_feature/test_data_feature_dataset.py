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
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path
from pandas import DataFrame
import pytest
import pandas as pd
import numpy as np
from src.optimize.experimental.inference.constant import OpAlgorithm
from src.optimize.experimental.inference.data_format_v1 import (
    MODEL_OP_FIELD,
    MODEL_STRUCT_FIELD,
    MODEL_CONFIG_FIELD,
    MINDIE_FIELD,
    ENV_FIELD,
    HARDWARE_FIELD,
)
from src.optimize.experimental.inference.dataset import CustomOneHotEncoder, CustomLabelEncoder, \
    preset_category_data
from src.optimize.experimental.inference.utils import PreprocessTool, TOTAL_OUTPUT_LENGTH, \
    TOTAL_SEQ_LENGTH, TOTAL_PREFILL_TOKEN
from src.optimize.experimental.data_feature.dataset import MyDataSet, CustomOneHotEncoder 


class TestMyDataSet(unittest.TestCase):
    def setUp(self):
        # 基础测试数据
        self.sample_data = pd.DataFrame({
            "('batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len',\
                  'model_execute_time')": ("""('prefill', 1.0, 1.0, 41, 41334, "['0']")""", ),
            "('input_length', 'need_blocks', 'output_length')": ("(1015, 19, 0)",)
        })
        self.test_dir = Path("/tmp/test_output_1")
        self.test_dir.mkdir(exist_ok=True)

    def test_init_defaults(self):
        """测试默认初始化"""
        ds = MyDataSet()
        self.assertEqual(ds.predict_field, "model_execute_time")
        self.assertIsInstance(ds.custom_encoder, CustomOneHotEncoder)
        self.assertIsNone(ds.features)

    def test_convert_request_info(self):
        # Test convert_request_info
        row = ("((1015, 19, 0),)")
        index = ("('input_length', 'need_blocks', 'output_length')")
        df = MyDataSet.convert_request_info(row, index)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns.tolist()), 154)

    @patch('experimental.inference.utils.PreprocessTool.generate_data_with_request_info_by_df')
    def test_convert_request_info_by_df(self, mock_generate_data):
        # 测试convert_request_info_by_df
        mock_generate_data.return_value = (['value1', 'value2'], ['column1', 'column2'])
        result = MyDataSet.convert_request_info_by_df('["row1", "row2"]', '["index1", "index2"]')
        self.assertEqual(result.columns.tolist(), ['column1', 'column2'])
        self.assertEqual(result.values.tolist(), [['value1', 'value2']])

    def test_init_custom_encoder(self):
        """测试自定义编码器初始化"""
        mock_encoder = MagicMock()
        ds = MyDataSet(custom_encoder=mock_encoder)
        self.assertEqual(ds.custom_encoder, mock_encoder)

    @patch('experimental.data_feature.dataset.PreprocessTool.generate_data')
    def test_convert_batch_info(self, mock_generate):
        """测试批次信息转换"""
        mock_generate.return_value = ([1, 2], ["col1", "col2"])
        result = MyDataSet.convert_batch_info("[[1, 2]]", "['idx1', 'idx2']")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["col1", "col2"])

    @patch('experimental.data_feature.dataset.plt.savefig')
    @patch('experimental.data_feature.dataset.sns.scatterplot')
    def test_analysis_batch_feature(self, mock_plot, mock_save):
        """测试test_analysis_batch_feature"""
        ds = MyDataSet()
        ds.load_data = pd.DataFrame({
            "batch_size": (1, 2, 3),
            "other_col": ("a", "b", "c")
        })
        ds.sub_columns = [["batch_size"]]
        ds.analysis_batch_feature(self.test_dir)
        mock_save.assert_called()

    @patch('experimental.data_feature.dataset.logger.error')
    def test_construct_data_shape_mismatch(self, mock_logger):
        """测试特征和标签维度不匹配的情况"""
        ds = MyDataSet()
        with patch.object(ds, 'preprocess_dispatch', 
                        return_value=(pd.DataFrame(), pd.DataFrame([1, 2]))):
            with self.assertRaises(AttributeError):
                ds.construct_data(self.sample_data)

    def tearDown(self):
        # 清理测试文件
        for f in self.test_dir.glob("*"):
            f.unlink()
        self.test_dir.rmdir()


class TestGetAllRequestInfo(unittest.TestCase):
    def test_get_all_request_info_valid_input(self):
        # 测试有效的输入
        row = "[['1.0', '2.0', '3.0'], ['4.0', '5.0', '6.0']]"
        index = "['A', 'B', 'C']"
        expected_output = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
        self.assertTrue(expected_output.equals(MyDataSet.get_all_request_info(row, index)))

    def test_get_all_request_info_empty_input(self):
        # 测试空输入
        row = "[]"
        index = "[]"
        expected_output = pd.DataFrame([], columns=[])
        self.assertTrue(expected_output.equals(MyDataSet.get_all_request_info(row, index)))

    def test_get_all_request_info_single_row(self):
        # 测试单行输入
        row = "[['1.0', '2.0', '3.0']]"
        index = "['A', 'B', 'C']"
        expected_output = pd.DataFrame([[1, 2, 3]], columns=['A', 'B', 'C'])
        self.assertTrue(expected_output.equals(MyDataSet.get_all_request_info(row, index)))

    def test_get_all_request_info_single_column(self):
        # 测试单列输入
        row = "[['1.0'], ['2.0'], ['3.0']]"
        index = "['A']"
        expected_output = pd.DataFrame([[1], [2], [3]], columns=['A'])
        self.assertTrue(expected_output.equals(MyDataSet.get_all_request_info(row, index)))

    def test_get_all_request_info_non_integer_values(self):
        # 测试非整数值输入
        row = "[['1.5', '2.5', '3.5'], ['4.5', '5.5', '6.5']]"
        index = "['A', 'B', 'C']"
        expected_output = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
        self.assertTrue(expected_output.equals(MyDataSet.get_all_request_info(row, index)))


class TestPlotCustomPairplot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """创建测试用的DataFrame"""
        np.random.seed(42)
        data = {
            'max_seq_len': np.random.randint(1, 100, 50),
            'input_length': np.random.randint(1, 200, 50),
            'total_prefill_token': np.random.randint(1, 300, 50),
            'model_execute_time': np.random.normal(10, 2, 50),
            'batch_size': np.random.randint(1, 32, 50),
            'other_column': np.random.rand(50)
        }
        cls.sample_df = DataFrame(data)
        cls.temp_dir = Path("test_output_1")
        cls.temp_dir.mkdir(exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """清理测试文件"""
        for file in cls.temp_dir.glob("*.png"):
            file.unlink()
        cls.temp_dir.rmdir()
    
    def test_basic_functionality(self):
        """测试不保存文件时是否调用plt.show()"""
        with patch('matplotlib.pyplot.show') as mock_show:
            # 假设plot_custom_pairplot是在MyPlotClass类中
            MyDataSet.plot_custom_pairplot(self.sample_df)
            mock_show.assert_called_once()
    
    def test_saving_to_file(self):
        """测试图表保存功能"""
        file_name = "test_pairplot.png"
        save_path = self.temp_dir / file_name
        
        # 确保文件不存在
        if save_path.exists():
            save_path.unlink()
            
        MyDataSet.plot_custom_pairplot(
            df=self.sample_df,
            middle_save_path=self.temp_dir,
            file_name=file_name
        )
        
        # 验证文件已创建
        self.assertTrue(save_path.exists())
    
    def test_different_column_combinations(self):
        """测试对max_seq_len等特殊列的处理"""
        with patch('seaborn.histplot') as mock_hist:
            MyDataSet.plot_custom_pairplot(self.sample_df)
            
            # 检查是否对特殊列调用了histplot
            hist_calls = [str(call[0][0]) for call in mock_hist.call_args_list]
            self.assertTrue(any('max_seq_len' in call.lower() for call in hist_calls))
            self.assertTrue(any('input_length' in call.lower() for call in hist_calls))
            self.assertTrue(any('total_prefill_token' in call.lower() for call in hist_calls))
    
    def test_model_execute_time(self):
        """测试对model_execute_time的特殊处理"""
        with patch('seaborn.scatterplot') as mock_scatter:
            MyDataSet.plot_custom_pairplot(self.sample_df)
            
            # 检查是否对model_execute_time调用了scatterplot
            scatter_calls = [str(call[1]) for call in mock_scatter.call_args_list]
            self.assertTrue(any('model_execute_time' in call for call in scatter_calls))
    
    def test_empty_dataframe(self):
        """测试传入空DataFrame时的行为"""
        empty_df = DataFrame()
        with self.assertRaises(ValueError):
            MyDataSet.plot_custom_pairplot(empty_df)
    
    def test_figure_closing(self):
        """测试图表是否被正确关闭"""
        with patch('matplotlib.pyplot.close') as mock_close:
            MyDataSet.plot_custom_pairplot(self.sample_df)
            mock_close.assert_called_once()


class TestConstructData(unittest.TestCase):
    def setUp(self):
        self.obj = MyDataSet()  # replace YourClass with the actual class name
        self.obj.preprocess_dispatch = MagicMock()
        self.obj.plt_data = MagicMock()
        self.obj.features = DataFrame()
        self.obj.labels = DataFrame()
        self.obj.test_size = 0.2
        self.obj.shuffle = True

    def test_construct_data_none_lines_data(self):
        with self.assertRaises(AttributeError):
            self.obj.construct_data()


    def test_construct_data_shape_mismatch(self):
        self.obj.features = DataFrame({'A': [1, 2, 3]})
        self.obj.labels = DataFrame({'B': [4, 5, 6, 7]})
        with self.assertRaises(ValueError):
            self.obj.construct_data(DataFrame())


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.preprocessor = MyDataSet()
        self.preprocessor.custom_encoder = MagicMock()
        self.preprocessor.custom_encoder.transformer = MagicMock(return_value='mocked_transformed_features')

    def test_preprocess_lines_data_none(self):
        with self.assertRaises(Exception):
            self.preprocessor.preprocess(None)

    def test_preprocess_lines_data_columns_less_than_3(self):
        data = {
        "('batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len',\
              'model_execute_time')": ["('prefill', 1, 1, 41, 41334, '0')"],
        "('input_length', 'need_blocks', 'output_length')": ["(1015, 19, 0)"]
    }
        with self.assertRaises(Exception):
            self.preprocessor.preprocess(data)


@pytest.fixture
def feature_csv(tmpdir):
    _feature_file = Path(tmpdir).joinpath("feature.csv")
    with open(_feature_file, "w") as f:
        f.write("""
"('batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len',"""\
                """ 'model_execute_time')","('input_length', 'need_blocks', 'output_length')"
"('prefill', 1, '16', '2048', '2048', '307133.19778442377')","(('2048', '16', '0'),)"
"('prefill', 3, '48', '6144', '2048', '764642.9538726807')","(('2048', '16', '0'), ('2048', '16', '0'),"""\
     """ ('2048', '16', '0'))"
""")
    yield _feature_file
    _feature_file.unlink()


@pytest.fixture
def all_feature_csv(tmpdir):
    _feature_file = Path(tmpdir).joinpath("feature.csv")
    with open(_feature_file, "w") as f:
        _txt = '''"('batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len','''\
             ''' 'model_execute_time')"'''
        _txt_2 = '''"('input_length', 'need_blocks', 'output_length')"'''
        _txt_3 = '''"('op_name', 'call_count', 'input_count', 'input_dtype', 'input_shape', 'output_count','''\
             ''' 'output_dtype', 'output_shape', 'host_setup_time', 'host_execute_time', 'kernel_execute_time','''\
                 ''' 'aic_cube_fops', 'aiv_vector_fops')"'''
        _txt_4 = '''"('total_param_num', 'total_param_size', 'embed_tokens_param_size_rate','''\
             ''' 'self_attn_param_size_rate', 'mlp_param_size_rate', 'input_layernorm_param_size_rate','''\
                 ''' 'post_attention_layernorm_param_size_rate', 'norm_param_size_rate', 'lm_head_param_size_rate')"'''
        _txt_5 = '''"('architectures', 'hidden_act', 'initializer_range', 'intermediate_size','''\
             ''' 'max_position_embeddings', 'model_type', 'num_attention_heads', 'num_hidden_layers', 'tie_word_'''\
                '''embeddings', 'torch_dtype', 'use_cache', 'vocab_size', 'quantize', 'quantization_config')"'''
        _txt_6 = '''"('cache_block_size', 'mindie__max_seq_len', 'world_size', 'cpu_mem_size', 'npu_mem_size','''\
             ''' 'max_prefill_tokens', 'max_prefill_batch_size', 'max_batch_size')"'''
        _txt_7 = '''"('atb_llm_razor_attention_enable', 'atb_llm_razor_attention_rope','''\
             ''' 'bind_cpu', 'mies_use_mb_swapper', 'mies_pecompute_threshold', 'mies_tokenizer_sliding'''\
                '''_window_size', 'atb_llm_lcoc_enable', 'lccl_deterministic', 'hccl_deterministic','''\
                     ''' 'atb_matmul_shuffle_k_enable')"'''
        _txt_8 = '''"('cpu_count', 'cpu_mem', 'soc_name', 'npu_mem')"'''
        f.write(f"{_txt},{_txt_2},{_txt_3},{_txt_4},{_txt_5},{_txt_6},{_txt_7},{_txt_8}")
        f.write("\n")
        _txt = '''"('prefill', 1, '11.0', '1396.0', '1396.0', '639593.0')"'''
        _txt_2 = '''"(('1396.0', '11', '0'),)"'''
        _txt_3 = '''"(('LinearOperation', '32', '2', 'float16;float16', '1396,4096;6144,4096', '1', 'float16','''\
             ''' '1396,6144', '158.13848484848492', '343.5108080808081', '296.228', '75587665920.0', '0.0'), )"'''
        _txt_4 = '''"('206', '8030265344', '0.06542008881344383', '0.16713984189860415', '0.7019873359741374','''\
             ''' '1.632225018541056e-05', '1.632225018541056e-05', '5.1007031829408e-07', '0.06542067874312553')"'''
        _txt_5 = '''"(('LlamaForCausalLM',), 'silu', 0.02, 14336, 8192, 'llama', 32, 32, False, 'float16','''\
             ''' True, 128256, None, {'group_size': 0, 'kv_quant_type': None, 'reduce_quant_type': None})"'''
        _txt_6 = '''"(128, 2560, 1, 5, -1, 8192, 50, 200)"'''
        _txt_7 = '''"(0, 0, 1, 0, 0.5, 0, 0, 0, 0, 1)"'''
        _txt_8 = '''"(256, 2026542, 'xxx', 7864)"'''
        f.write(f"{_txt},{_txt_2},{_txt_3},{_txt_4},{_txt_5},{_txt_6},{_txt_7},{_txt_8}")
    yield _feature_file
    _feature_file.unlink()


def test_dataset_function_with_all_feature(all_feature_csv, tmpdir):
    df = pd.read_csv(all_feature_csv)
    custom_label_encoder = CustomLabelEncoder(preset_category_data)
    custom_label_encoder.fit()
    my_data_set = MyDataSet(predict_field="model_execute_time", custom_encoder=custom_label_encoder)
    my_data_set.construct_data(df)
    assert my_data_set.features.shape == (1, 902)
    assert my_data_set.labels.shape == (1, 1)
