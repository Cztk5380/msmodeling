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

import subprocess
import shutil

from src.optimize.experimental.common import get_module_version, get_npu_total_memory


def test_module_version():
    version = get_module_version("xgboost")
    assert version == "2.0.0"


def test_get_npu_total_memory_success(monkeypatch):
    def _npu_info_usages(*args):
        key_word = "H" + "B" + "M"
        if args and args[0][2] == "-m":
            return "0 0 0".encode()
        return f"""
    NPU ID                         : 0
    Chip Count                     : 1

    DDR Capacity(MB)               : 0
    DDR Usage Rate(%)              : 0
    DDR Hugepages Total(page)      : 0
    DDR Hugepages Usage Rate(%)    : 0
    {key_word} Capacity(MB)               : 65536
    {key_word} Usage Rate(%)              : 3
    Aicore Usage Rate(%)           : 0
    Aivector Usage Rate(%)         : 0
    Aicpu Usage Rate(%)            : 0
    Ctrlcpu Usage Rate(%)          : 2
    DDR Bandwidth Usage Rate(%)    : 0
    {key_word} Bandwidth Usage Rate(%)    : 0
    Chip ID                        : 0
    """.encode()

    monkeypatch.setattr(subprocess, "check_output", _npu_info_usages)
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/npu-smi")

    # Call the function and check the result
    total_memory, usage_rate = get_npu_total_memory()
    assert total_memory == 65536
    assert usage_rate == 3