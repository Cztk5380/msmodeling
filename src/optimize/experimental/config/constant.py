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
import os

MODEL_EVAL_STATE_SIMULATE = "MODEL_EVAL_STATE_SIMULATE"
MODEL_EVAL_STATE_ALL = "MODEL_EVAL_STATE_ALL"
SIMULATE = "simulate"

simulate_env = os.getenv(MODEL_EVAL_STATE_SIMULATE) or os.getenv(MODEL_EVAL_STATE_SIMULATE.lower())
simulate_flag = simulate_env and (simulate_env.lower() == "true" or simulate_env.lower() != "false")

REAL_EVALUATION = "real_evaluation"

REQUESTRATES = ("REQUESTRATE",)
CONCURRENCYS = ("CONCURRENCY", "MAXCONCURRENCY")
METRIC_TTFT = 'ttft'
METRIC_TPOT = 'tpot'