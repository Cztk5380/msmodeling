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
import os
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.optimize.experimental.optimizer import analyze_profiler
from src.optimize.experimental.optimizer.analyze_profiler import analyze


class TestFindFirstSimulateCSV(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = Path("test_analyze_profiler")
        self.test_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_valid_directory_with_matching_files(self):
        """Test with directory containing matching simulate*.csv files"""
        # Create test files
        open(os.path.join(self.test_dir, "simulate1.csv"), 'a').close()
        open(os.path.join(self.test_dir, "simulate2.csv"), 'a').close()
        open(os.path.join(self.test_dir, "other.csv"), 'a').close()
        
        result = analyze_profiler.find_first_simulate_csv(self.test_dir)
        self.assertEqual(result, os.path.join(self.test_dir, "simulate1.csv"))
    
    def test_valid_directory_no_matching_files(self):
        """Test with directory containing no simulate*.csv files"""
        # Create non-matching files
        open(os.path.join(self.test_dir, "test1.csv"), 'a').close()
        open(os.path.join(self.test_dir, "data.csv"), 'a').close()
        
        with self.assertRaises(FileNotFoundError) as context:
            analyze_profiler.find_first_simulate_csv(self.test_dir)
        self.assertEqual(str(context.exception), 
                         "No CSV files starting with 'simulate' found in the directory.")
    
    def test_nonexistent_directory(self):
        """Test with non-existent directory"""
        nonexistent_path = os.path.join(self.test_dir, "nonexistent")
        with self.assertRaises(NotADirectoryError) as context:
            analyze_profiler.find_first_simulate_csv(nonexistent_path)
        self.assertEqual(str(context.exception), 
                         "The provided path is not a valid directory.")
    
    def test_file_instead_of_directory(self):
        """Test when path points to a file instead of directory"""
        file_path = os.path.join(self.test_dir, "testfile")
        open(file_path, 'a').close()
        
        with self.assertRaises(NotADirectoryError) as context:
            analyze_profiler.find_first_simulate_csv(file_path)
        self.assertEqual(str(context.exception), 
                         "The provided path is not a valid directory.")
    
    def test_empty_directory(self):
        """Test with empty directory"""
        with self.assertRaises(FileNotFoundError) as context:
            analyze_profiler.find_first_simulate_csv(self.test_dir)
        self.assertEqual(str(context.exception), 
                         "No CSV files starting with 'simulate' found in the directory.")
    
    def test_file_sorting(self):
        """Test that files are properly sorted"""
        # Create files in non-sorted order
        open(os.path.join(self.test_dir, "simulate10.csv"), 'a').close()
        open(os.path.join(self.test_dir, "simulate2.csv"), 'a').close()
        open(os.path.join(self.test_dir, "simulate1.csv"), 'a').close()
        
        result = analyze_profiler.find_first_simulate_csv(self.test_dir)
        self.assertEqual(result, os.path.join(self.test_dir, "simulate1.csv"))


class TestAnalyzeFunction(unittest.TestCase):
    def setUp(self):
        """准备测试数据"""
        # 基础测试数据
        self.request_data = pd.DataFrame({
            'http_rid': [0, 1, 2],
            'reply_token_size': [10, np.nan, 20],
            'first_token_latency': [100, 200, 300],
            'execution_time(microsecond)': [1000, 2000, 3000],
            'start_time_httpReq(microsecond)': [500, 1500, 2500]
        })
        
        self.batch_data = pd.DataFrame({
            'name': ['modelExec', 'modelExec', 'modelExec'],
            'batch_type': ['prefill', 'decode', 'prefill'],
            'reqinfo': ['0,100,2,200', '1,300', '2,400'],
            'start_time(microsecond)': [1000, 2000, 3000]
        })
        
        self.simulate_data = pd.DataFrame({
            'simulate_time': [1000000, 2000000, 3000000]
        })

    @patch('src.optimize.experimental.optimizer.analyze_profiler.read_csv_s')
    @patch('src.optimize.experimental.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_normal_flow(self, mock_find_csv, mock_read_csv):
        """测试正常流程"""
        # 设置mock返回值
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            self.request_data,
            self.batch_data,
            self.simulate_data
        ]
        
        # 执行分析
        throughput, avg_prefill, avg_decode, success_rate = analyze(
            input_path_1='/fake/path1',
            input_path_2='/fake/path2'
        )
        
        # 验证结果
        self.assertIsInstance(throughput, float)
        self.assertIsInstance(avg_prefill, float)
        self.assertIsInstance(avg_decode, float)
        self.assertIsInstance(success_rate, float)
        self.assertGreater(throughput, 0)
        self.assertGreater(avg_prefill, 0)
        self.assertGreater(avg_decode, 0)
        self.assertLessEqual(success_rate, 1.0)
        self.assertGreaterEqual(success_rate, 0.0)

    @patch('src.optimize.experimental.optimizer.analyze_profiler.read_csv_s')
    @patch('src.optimize.experimental.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_all_successful_requests(self, mock_find_csv, mock_read_csv):
        """测试全部成功请求的场景"""
        # 创建全部成功的请求数据
        all_success_request = self.request_data.copy()
        all_success_request['reply_token_size'] = 10  # 设置所有请求都有回复
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            all_success_request,
            self.batch_data,
            self.simulate_data
        ]
        
        _, _, _, success_rate = analyze(
            input_path_1='/fake/path1',
            input_path_2='/fake/path2'
        )
        
        self.assertEqual(success_rate, 1.0)

    @patch('src.optimize.experimental.optimizer.analyze_profiler.read_csv_s')
    @patch('src.optimize.experimental.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_mismatched_rows(self, mock_find_csv, mock_read_csv):
        """测试行数不匹配的情况"""
        # 创建行数不匹配的模拟数据
        mismatched_simulate = pd.DataFrame({
            'simulate_time': [1000000, 2000000]  # 只有2行
        })
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            self.request_data,
            self.batch_data,
            mismatched_simulate
        ]
        
        with self.assertRaises(ValueError):
            analyze(input_path_1='/fake/path1', input_path_2='/fake/path2')

    @patch('src.optimize.experimental.optimizer.analyze_profiler.read_csv_s')
    @patch('src.optimize.experimental.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_invalid_data_format(self, mock_find_csv, mock_read_csv):
        """测试无效的数据格式"""
        # 创建缺少必要列的数据
        invalid_request = pd.DataFrame({
            'http_rid': [0, 1, 2]  # 缺少其他必要列
        })
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            invalid_request,
            self.batch_data,
            self.simulate_data
        ]
        
        with self.assertRaises(KeyError):
            analyze(input_path_1='/fake/path1', input_path_2='/fake/path2')

    @patch('src.optimize.experimental.optimizer.analyze_profiler.read_csv_s')
    @patch('src.optimize.experimental.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_edge_case_single_request(self, mock_find_csv, mock_read_csv):
        """测试单个请求的边界情况"""
        # 创建只有一个请求的数据
        single_request = pd.DataFrame({
            'http_rid': [0],
            'reply_token_size': [10],
            'first_token_latency': [100],
            'execution_time(microsecond)': [1000],
            'start_time_httpReq(microsecond)': [500]
        })
        
        single_batch = pd.DataFrame({
            'name': ['modelExec'],
            'batch_type': ['prefill'],
            'reqinfo': ['0,100'],
            'start_time(microsecond)': [1000]
        })
        
        single_simulate = pd.DataFrame({
            'simulate_time': [1000000]
        })
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            single_request,
            single_batch,
            single_simulate
        ]
        
        throughput, avg_prefill, avg_decode, success_rate = analyze(
            input_path_1='/fake/path1',
            input_path_2='/fake/path2'
        )
        
        self.assertEqual(success_rate, 1.0)
        self.assertGreater(throughput, 0)

    @patch('src.optimize.experimental.optimizer.analyze_profiler.read_csv_s')
    @patch('src.optimize.experimental.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_no_successful_requests(self, mock_find_csv, mock_read_csv):
        """测试没有成功请求的场景"""
        # 创建全部失败的请求数据
        all_failed_request = self.request_data.copy()
        all_failed_request['reply_token_size'] = np.nan  # 设置所有请求都没有回复
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            all_failed_request,
            self.batch_data,
            self.simulate_data
        ]
        
        throughput, avg_prefill, avg_decode, success_rate = analyze(
            input_path_1='/fake/path1',
            input_path_2='/fake/path2'
        )
        
        self.assertEqual(success_rate, 0.0)
        self.assertEqual(throughput, 0.0)


if __name__ == '__main__':
    unittest.main()
