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
import unittest
from unittest.mock import patch, MagicMock
from src.optimize.experimental.sitecustomize import dispatch, MODEL_EVAL_STATE_SIMULATE, MODEL_EVAL_STATE_ALL


class TestDispatch(unittest.TestCase):
    
    @patch('experimental.patch.enable_patch')
    def test_dispatch_simulate_true(self, mock_enable_patch):
        # Set environment variable for simulation
        os.environ[MODEL_EVAL_STATE_SIMULATE] = 'True'
        dispatch(MODEL_EVAL_STATE_SIMULATE)
        mock_enable_patch.assert_called_with(MODEL_EVAL_STATE_SIMULATE)

    @patch('src.optimize.experimental.sitecustomize.logger')
    def test_dispatch_simulate_false(self, mock_logger):
        # Set environment variable for simulation to false
        os.environ[MODEL_EVAL_STATE_SIMULATE] = 'False'
        dispatch(MODEL_EVAL_STATE_SIMULATE)
        mock_logger.debug.assert_called_with(f"{MODEL_EVAL_STATE_SIMULATE}: False")

    @patch('src.optimize.experimental.sitecustomize.logger')
    def test_dispatch_all_false(self, mock_logger):
        # Set environment variable for optimization to false
        os.environ[MODEL_EVAL_STATE_ALL] = 'false'
        dispatch(MODEL_EVAL_STATE_ALL)
        mock_logger.debug.assert_called_with(f"{MODEL_EVAL_STATE_ALL}: false")

    @patch('src.optimize.experimental.sitecustomize.logger')
    def test_dispatch_simulate_not_set(self, mock_logger):
        # Do not set environment variable for simulation
        if MODEL_EVAL_STATE_SIMULATE in os.environ:
            del os.environ[MODEL_EVAL_STATE_SIMULATE]
        dispatch(MODEL_EVAL_STATE_SIMULATE)
        mock_logger.debug.assert_called_with(f"{MODEL_EVAL_STATE_SIMULATE}: None")

    @patch('src.optimize.experimental.sitecustomize.logger')
    def test_dispatch_all_not_set(self, mock_logger):
        # Do not set environment variable for optimization
        if MODEL_EVAL_STATE_ALL in os.environ:
            del os.environ[MODEL_EVAL_STATE_ALL]
        dispatch(MODEL_EVAL_STATE_ALL)
        mock_logger.debug.assert_called_with(f"{MODEL_EVAL_STATE_ALL}: None")