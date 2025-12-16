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
import unittest
import tempfile
from pathlib import Path
from src.optimize.experimental.data_feature.v1 import FileReader


def generate_data(fp):
    header = """world_size,request_rate,concurrency,prefill_batchsize,decode_batchsize,select_batch,"""\
        """prefillTimeMsPerReq,decodeTimeMsPerReq,line,batch_num,"('codellama_34b',)","('llama3_70b',)","('"""\
            """llama3_8b',)","('qwen1.5_14b',)","('qwen1.5_72b',)","('decode',)","('prefill',)"
            """
    fp.write(header)
    fp.write("\n")
    for _ in range(2000):
        fp.write("2,15,1000,100,200,True,600,50,3090,209,0.0,0.0,1.0,0.0,0.0,1.0,0.0")
        fp.write("\n")


def test_file_reader():
    _file_1 = tempfile.NamedTemporaryFile(mode="w", delete=False)
    _file_2 = tempfile.NamedTemporaryFile(mode="w", delete=False)
    file_paths = [Path(_file_1.name),
                  Path(_file_2.name)]
    generate_data(_file_1)
    generate_data(_file_2)
    _file_1.close()
    _file_2.close()
    fr = FileReader(file_paths)
    res = fr.read_lines()
    assert res.shape[0] > 1977
    num_lines = 1000
    fr = FileReader(file_paths, num_lines=num_lines)
    res = fr.read_lines()
    assert res.shape[0] == num_lines
    res = fr.read_lines()
    assert res.shape[0] == num_lines
    res = fr.read_lines()
    assert res.shape[0] == num_lines
    res = fr.read_lines()
    assert res.shape[0] == num_lines
    Path(_file_1.name).unlink()
    Path(_file_2.name).unlink()
