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
from pathlib import Path
from unittest.mock import MagicMock, patch
import os

import json
import shutil
import sqlite3
import pandas as pd
import numpy as np

from src.optimize.experimental.train.source_to_train import (
    DatabaseConnector,
    source_to_model,
    req_decodetimes,
    read_batch_exec_data
)


class TestSourceToTrainMindie(unittest.TestCase):
    """测试数据预处理和训练流程的功能"""

    def setUp(self):
        # 创建临时测试环境
        self.test_dir = Path("test_source_to_train")
        self.test_dir.mkdir(exist_ok=True)

        # 创建样本数据库
        self.db_path = self.test_dir / "profiler.db"
        self.create_sample_database()

        # 创建样本CSV文件
        self.create_sample_csv()

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)

    def create_sample_database(self):
        """创建样本SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建batch表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch (
            name TEXT,
            res_list TEXT,
            start_time REAL,
            end_time REAL,
            batch_size REAL,
            batch_type TEXT,
            during_time REAL
        )
        """)

        # 插入样本数据
        batch_data = [
            ("BatchSchedule", "[{'rid': 101, 'iter': 0}]", 1749451414153,
             1749451414154, 1, "Prefill", 0.22175),
            ("BatchSchedule", "[{'rid': 101, 'iter': 0}]", 1749451414154,
             1749451414155, 1, "Decode", 0.223)
        ]
        cursor.executemany(
            "INSERT INTO batch (name,res_list, start_time,end_time,batch_size,batch_type,"
            "during_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
            batch_data)

        # 创建batch_exec表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_exec (
            batch_id INTEGER PRIMARY KEY,
            event TEXT,
            pid INTEGER,
            start REAL,
            end REAL
        )
        """)

        # 插入样本数据
        exec_data = [
            (1, 'forward', 1001, 1000.0, 1500.0),
            (2, 'forward', 1001, 2000.0, 2500.0)
        ]
        cursor.executemany("INSERT INTO batch_exec (batch_id, event, pid, start, end) VALUES (?, ?, ?, ?, ?)",
                           exec_data)

        # 创建batch_req表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_req (
            batch_id INTEGER,
            req_id TEXT,
            rid TEXT,
            iter INTEGER,
            block INTEGER
        )
        """)

        # 插入样本数据
        req_data = [
            (1, "101", "101", "0", 256),
            (2, "101", "101", "1", 192)
        ]
        cursor.executemany("INSERT INTO batch_req (batch_id, req_id, rid, iter, block) VALUES (?, ?, ?, ?, ?)",
                           req_data)

        conn.commit()
        conn.close()

    def create_sample_csv(self):
        """创建样本CSV文件"""
        # 创建request.csv
        request_data = pd.DataFrame({
            "http_rid": ["101"],
            "start_time": ["1749451414153"],
            "recv_token_size": ["256"],
            "reply_token_size": ["128"],
            "execution_time": ["1"],
            "queue_wait_time": ["0.11"],
            "first_token_latency": ["0.5"]
        })
        request_data.to_csv(self.test_dir / "request.csv", index=False)

    def test_database_connector(self):
        """测试数据库连接器"""
        # 测试正常连接
        db_conn = DatabaseConnector(str(self.db_path))
        cursor = db_conn.connect()
        self.assertIsNotNone(cursor)

        # 测试读取batch_exec数据
        exec_rows = read_batch_exec_data(cursor)
        self.assertEqual(len(exec_rows), 2)

        # 测试关闭连接
        db_conn.close()

    def test_source_to_model_mindie(self):
        """测试Mindie数据预处理流程"""
        # 执行数据处理
        source_to_model(self.test_dir, model_type='mindie')

        # 验证输出目录创建
        output_csv = self.test_dir / "output_csv"
        self.assertTrue(output_csv.exists())

        # 验证输出文件存在
        for pid_dir in output_csv.iterdir():
            if pid_dir.is_dir():
                self.assertTrue((pid_dir / "feature.csv").exists())

    def test_req_decodetimes(self):
        """测试解码时间处理"""
        # 执行处理
        json_file = self.test_dir / "output" / "req_id_and_decode_num.json"
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        # 创建空 JSON 文件
        with open(json_file, "w", encoding="utf-8") as f:
            f.write("")
        req_decodetimes(self.test_dir, self.test_dir / "output")

        # 验证JSON文件生成
        self.assertTrue(json_file.exists())

        # 验证JSON内容
        with open(json_file, "r") as f:
            data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data["0"], 128)


class TestSourceToTrainVllm(unittest.TestCase):
    """测试数据预处理和训练流程的功能"""

    def setUp(self):
        # 创建临时测试环境
        self.test_dir = Path("test_source_to_train")
        self.test_dir.mkdir(exist_ok=True)

        # 创建样本数据库
        self.db_path = self.test_dir / "profiler.db"
        self.create_sample_database()

        # 创建样本CSV文件
        self.create_sample_csv()

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)

    def create_sample_database(self):
        """创建样本SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建batch表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch (
            name TEXT,
            res_list TEXT,
            start_time REAL,
            end_time REAL,
            batch_size REAL,
            batch_type TEXT,
            during_time REAL
        )
        """)

        # 插入样本数据
        batch_data = [
            ("batchFrameworkProcessing", "[{'rid': 101, 'iter_size': 0}]", 1749451414153,
             1749451414154, 1, "Prefill", 0.22175),
            ("batchFrameworkProcessing", "[{'rid': 101, 'iter_size': 0}]", 1749451414154,
             1749451414155, 1, "Decode", 0.223)
        ]
        cursor.executemany(
            "INSERT INTO batch (name,res_list, start_time,end_time,batch_size,batch_type,"
            "during_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
            batch_data)

        # 创建batch_exec表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_exec (
            batch_id INTEGER PRIMARY KEY,
            event TEXT,
            pid INTEGER,
            start REAL,
            end REAL
        )
        """)

        # 插入样本数据
        exec_data = [
            (1, 'forward', 1001, 1000.0, 1500.0),
            (2, 'forward', 1001, 2000.0, 2500.0)
        ]
        cursor.executemany("INSERT INTO batch_exec (batch_id, event, pid, start, end) VALUES (?, ?, ?, ?, ?)",
                           exec_data)

        # 创建batch_req表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_req (
            batch_id INTEGER,
            req_id TEXT,
            rid TEXT,
            iter_size INTEGER
        )
        """)

        # 插入样本数据
        req_data = [
            (1, "101", "101", "0"),
            (2, "101", "101", "1")
        ]
        cursor.executemany("INSERT INTO batch_req (batch_id, req_id, rid, iter_size) VALUES (?, ?, ?, ?)",
                           req_data)

        conn.commit()
        conn.close()

    def create_sample_csv(self):
        """创建样本CSV文件"""
        # 创建request.csv
        request_data = pd.DataFrame({
            "http_rid": ["101"],
            "start_time": ["1749451414153"],
            "recv_token_size": ["256"],
            "reply_token_size": ["128"],
            "execution_time": ["1"],
            "queue_wait_time": ["0.11"],
            "first_token_latency": ["0.5"]
        })
        request_data.to_csv(self.test_dir / "request.csv", index=False)

        # 创建kvcache.csv (仅用于vllm模式)
        kvcache_data = pd.DataFrame({
            "domain": ["KVCache", "KVCache"],
            "rid": ["101", "101"],
            "timestamp": ["1749451415160", "1749451415161"],
            "name": ["Allocate", "blocks"],
            "device_kvcache_left": ["128", "256"]
        })
        kvcache_data.to_csv(self.test_dir / "kvcache.csv", index=False)

    def test_source_to_model_vllm(self):
        """测试vLLM数据预处理流程"""
        # 执行数据处理
        source_to_model(self.test_dir, model_type="vllm")

        # 验证输出目录创建
        output_csv = self.test_dir / "output_csv"
        self.assertTrue(output_csv.exists())

        # 验证输出文件存在
        for pid_dir in output_csv.iterdir():
            if pid_dir.is_dir():
                self.assertTrue((pid_dir / "feature.csv").exists())


if __name__ == "__main__":
    unittest.main()
