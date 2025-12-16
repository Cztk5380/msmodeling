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
import json
import shutil
import re
from pathlib import Path
import unittest
import shutil
from unittest.mock import patch, MagicMock, mock_open
import csv
import pandas as pd
import pytest
from src.optimize.experimental.config.config import PerformanceIndex, get_settings, AisBenchConfig, \
    OptimizerConfigField
from src.optimize.experimental.optimizer.plugins.benchmark import parse_result, AisBench, VllmBenchMark
from msguard import GlobalConfig


class TestParseResult(unittest.TestCase):
    def test_string_with_ms(self):
        # 测试输入为字符串，且单位为ms的情况
        self.assertAlmostEqual(parse_result("123 ms"), 0.123)

    def test_string_with_us(self):
        # 测试输入为字符串，且单位为us的情况
        self.assertAlmostEqual(parse_result("456 us"), 0.000456)

    def test_string_with_other_unit(self):
        # 测试输入为字符串，但单位不是ms或us的情况
        self.assertAlmostEqual(parse_result("789 s"), 789.0)

    def test_string_without_unit(self):
        # 测试输入为字符串，但没有单位的情况
        self.assertAlmostEqual(parse_result("1010"), 1010.0)


@pytest.fixture
def results_per_request_file(tmpdir):
    file_path = Path(tmpdir).joinpath("results_per_request_202507181613.json")
    data = {
        "1": {
            "input_len": 1735,
            "output_len": 1,
            "prefill_bsz": 4,
            "decode_bsz": [],
            "req_latency": 13058.372889645398,
            "latency": [
                13058.23168065399
            ],
            "queue_latency": [
                12598012
            ],
            "input_data": "", "output": ""
        },
        "2": {
            "prefill_bsz": 4,
            "decode_bsz": [],
            "req_latency": 15173.639830201864,
            "latency": [
                15173.517209477723
            ],
            "queue_latency": [
                14708480
            ],
            "input_data": "", "output": ""
        },
        "3": {
            "input_len": 1777,
            "output_len": 3,
            "prefill_bsz": 4,
            "decode_bsz": [
                157,
                157
            ],
            "req_latency": 15456.984990276396,
            "latency": [
                15178.787489421666,
                208.4683496505022,
                69.54238004982471
            ],
            "queue_latency": [
                14711475,
                127888,
                3709
            ],
            "input_data": "", "output": "\t\tif ("
        },
        "4": {
            "input_len": 1770,
            "output_len": 3,
            "prefill_bsz": 4,
            "decode_bsz": [
                157,
                157
            ],
            "req_latency": 15481.421849690378,
            "latency": [
                14745.695400051773,
                670.0493693351746,
                64.66158013790846
            ],
            "queue_latency": [
                14280800,
                584221,
                3686
            ],
            "input_data": "", "output": "Passage "
        },
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path
