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
from unittest.mock import patch
import pytest

from src.optimize.experimental.config.custom_command import VllmCommand


class MockVllmCommandConfig:
    pass


class TestVllmCommand:
    @patch('shutil.which')
    def test_init_success(self, mock_which):
        """Test successful initialization when vllm is found in PATH"""
        # Setup
        mock_which.return_value = "/usr/bin/vllm"
        config = MockVllmCommandConfig()
        
        # Execute
        command = VllmCommand(config)
        
        # Verify
        assert command.process == "/usr/bin/vllm"
        assert command.command_config == config

    @patch('shutil.which')
    def test_init_failure_vllm_not_found(self, mock_which):
        """Test initialization fails when vllm is not found in PATH"""
        # Setup
        mock_which.return_value = None
        config = MockVllmCommandConfig()
        
        # Execute & Verify
        with pytest.raises(ValueError) as excinfo:
            VllmCommand(config)
        
        assert "Error: The 'vllm' executable was not found in the system PATH." in str(excinfo.value)