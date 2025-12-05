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

from collections import OrderedDict

import numpy as np

from msserviceprofiler.modelevalstate.inference.constant import ALL_OP, DTYPE_CATEGORY, \
    OP_EXECUTE_DELTA_FIELD, ALL_OP_PARAM_TYPE


def get_bins_and_label(field, interval=20, number=51, start=0, end=float("inf")):
    _hist_label = []
    _hist_bins = []
    for i in range(0, number):
        _v = start + i * interval
        _hist_bins.append(_v)
        if len(_hist_bins) > 1:
            _hist_label.append(f"{field}__{_v}")
    if end > interval * (number - 1):
        _hist_bins.append(end)
        _hist_label.append(f"{field}___{end}")
    return {"label": _hist_label, "bins": _hist_bins}


def get_field_bins_count(target, field, bins):
    _value = []
    for _request_info in target:
        _value.append(float(getattr(_request_info, field, 0)))
    if not _value and len(bins) >= 1:
        return [0 for _ in range(len(bins) - 1)]
    hist, bins = np.histogram(_value, bins)
    return hist


class HistInfo:
    input_length = get_bins_and_label("input_length", interval=80)
    need_blocks = get_bins_and_label("need_blocks", interval=1)
    output_length = get_bins_and_label(
        "output_length",
        interval=10,
    )


OP_EXPECTED_FIELD_MAPPING = {}
for _op in ALL_OP:
    OP_EXPECTED_FIELD_MAPPING[_op] = OrderedDict(
        {
            f"{_op}__op_name": 0,
            f"{_op}__call_count": 0,
            **{f"{_op}__input_dtype__{k}": 0 for k in DTYPE_CATEGORY},
            **{f"{_op}__input_size__{k}": 0 for k in range(len(ALL_OP_PARAM_TYPE[_op]["input"]))},
            **{f"{_op}__output_dtype__{k}": 0 for k in DTYPE_CATEGORY},
            **{f"{_op}__output_size__{k}": 0 for k in range(len(ALL_OP_PARAM_TYPE[_op]["output"]))},
            **{f"{_op}__{k}": 0 for k in OP_EXECUTE_DELTA_FIELD},
        }
    )

model_op_size = get_bins_and_label("ratio", interval=20, number=6, end=100)

OP_SCALE_HIST_FIELD_MAPPING = {}
for _op in ALL_OP:
    input_size_label = {}
    for k in range(len(ALL_OP_PARAM_TYPE[_op]["input"])):
        for _size_hist in model_op_size["label"]:
            input_size_label[f"{_op}__input_size__{k}__{_size_hist}"] = 0
    output_size_label = {}
    for k in range(len(ALL_OP_PARAM_TYPE[_op]["output"])):
        for _size_hist in model_op_size["label"]:
            output_size_label[f"{_op}__output_size__{k}__{_size_hist}"] = 0
    time_size_label = {}
    for _field in OP_EXECUTE_DELTA_FIELD:
        for _size_hist in model_op_size["label"]:
            time_size_label[f"{_op}__{_field}__{_size_hist}"] = 0

    OP_SCALE_HIST_FIELD_MAPPING[_op] = OrderedDict(
        {
            f"{_op}__op_name": 0,
            **{f"{_op}__input_dtype__{k}": 0 for k in DTYPE_CATEGORY},
            **input_size_label,
            **{f"{_op}__output_dtype__{k}": 0 for k in DTYPE_CATEGORY},
            **output_size_label,
            **time_size_label,
        }
    )
