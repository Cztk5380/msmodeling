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
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import shutil
import yaml
import pytest
import requests

import experimental
from experimental.config.config import get_settings, OptimizerConfigField, KubectlConfig, \
        MindieConfig
from experimental.optimizer.simulator import enable_simulate_old
from experimental.optimizer.plugins.simulate import Simulator, DisaggregationSimulator
from msguard import GlobalConfig


class TestSimulate(unittest.TestCase):

    def test_set_config_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        Simulator.set_config(origin_config, "a.b.c", 4)
        assert origin_config["a"]["b"]["c"] == 4

    def test_set_config_list(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.0.c", 4)
        assert origin_config["a"]["b"][0]["c"] == 4

    def test_set_config_new_key(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.0.d", 4)
        assert origin_config["a"]["b"][0]["d"] == 4

    def test_set_config_add_dict_list_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        Simulator.set_config(origin_config, "a.d.0.c", 4)
        assert origin_config["a"]["d"][0]["c"] == 4

    def test_set_config_add_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.1.c", 4)
        assert origin_config["a"]["b"][1]["c"] == 4

    def test_set_config_add_dict_list_dict_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.d.0.c.e", 4)
        assert origin_config["a"]["d"][0]["c"]["e"] == 4
    
    def test_is_int(self):
        # 测试is_int静态方法
        self.assertTrue(Simulator.is_int(1))
        self.assertTrue(Simulator.is_int("1"))
        self.assertFalse(Simulator.is_int("a"))


def test_enable_simulate_with_simulator(tmpdir, monkeypatch):
    config_path = Path(tmpdir).joinpath("config.json")
    with open(config_path, 'w') as f:
        f.write("""{
    "Version": "1.0.0",
    "ServerConfig": {
        "tlsCaFile": [
            "ca.pem"
        ],
        "tlsCert": "security/certs/server.pem"
    },
    "BackendConfig": {
        "backendName": "mindieservice_llm_engine",
        "ModelDeployConfig": {
            "maxSeqLen": 2560,
            "maxInputTokenLen": 2048,
            "truncation": false,
            "ModelConfig": [
                {
                    "modelInstanceType": "Standard"
                }
            ]
        },
        "ScheduleConfig": {
            "templateType": "Standard"
        }
    }
}""")
    get_settings().mindie.config_path = config_path
    get_settings().mindie.config_bak_path = Path(tmpdir).joinpath("config_bak.json")
    simulator = Simulator(get_settings().mindie)
    monkeypatch.setattr(experimental.optimizer.simulator, "simulate_flag", True)
    with enable_simulate_old(simulator) as flag:
        with open(config_path, 'r') as f:
            data = json.load(f)
            assert data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0][
                       "plugin_params"] == '{"plugin_type": "simulate"}'
    with open(config_path, 'r') as f:
        data = json.load(f)
        assert "plugin_params" not in data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]


def test_enable_simulate_with_simulator_plugin_params_exists(tmpdir, monkeypatch):
    config_path = Path(tmpdir).joinpath("config.json")
    data = {
        "BackendConfig": {
            "backendName": "mindieservice_llm_engine",
            "ModelDeployConfig": {
                "maxSeqLen": 2560,
                "ModelConfig": [
                    {
                        "modelInstanceType": "Standard",
                        "plugin_params": "{\"plugin_type\":\"tp\"}"
                    }
                ]

            },
            "ScheduleConfig": {
                "templateType": "Standard"
            }
        }
    }
    with open(config_path, 'w') as f:
        json.dump(data, f)
    get_settings().mindie.config_path = config_path
    get_settings().mindie.config_bak_path = Path(tmpdir).joinpath("config_bak.json")
    simulator = Simulator(get_settings().mindie)
    monkeypatch.setattr(experimental.optimizer.simulator, "simulate_flag", True)
    with enable_simulate_old(simulator) as flag:
        with open(config_path, 'r') as f:
            data = json.load(f)
            assert data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0][
                       "plugin_params"] == '{"plugin_type": "tp,simulate"}'
    with open(config_path, 'r') as f:
        data = json.load(f)
        assert data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["plugin_params"] == '{"plugin_type":"tp"}'


class TestDisaggregationSimulator(unittest.TestCase):
    def setUp(self):
        # 创建临时测试环境
        self.test_dir = Path("conf")
        self.yaml_dir = Path("deployment")
        self.test_dir.mkdir(exist_ok=True)
        self.yaml_dir.mkdir(exist_ok=True)
        self.config_single_path = self.test_dir / "config.json"
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(b"MindIE-MS coordinator is ready!!!")
        self.temp_file.close()
        data = {
            "BackendConfig": {
                "backendName": "mindieservice_llm_engine",
                "ModelDeployConfig": {
                    "maxSeqLen": 2560,
                    "ModelConfig": [
                        {
                            "modelInstanceType": "Standard",
                            "plugin_params": "{\"plugin_type\":\"tp\"}"
                        }
                    ]

                },
                "ScheduleConfig": {
                    "templateType": "Standard"
                }
            }
        }
        with open(self.config_single_path, 'w') as f:
            json.dump(data, f)
        pd_data = {
            "default_p_rate": 1,
            "default_d_rate": 3
        }
        self.kubectl_single_path = self.test_dir / "deploy.sh"
        self.config_single_pd_path = self.test_dir / "ms_controller.json"
        self.yaml_path = self.yaml_dir / "mindie_service_single_container.yaml"
        service_config = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "mindie-service",
                "labels": {
                    "app": "mindie-server"
                }
            },
            "spec": {
                "selector": {
                    "app": "mindie-server"
                },
                "ports": [
                    {
                        "name": "http",
                        "port": 1025,
                        "targetPort": 1025,
                        "nodePort": 31015,
                        "protocol": "TCP"
                    }
                ],
                "type": "NodePort",
                "sessionAffinity": "None"
            }
        }
        with open(self.yaml_path, 'w') as file:
            yaml.dump(service_config, file, default_flow_style=False)
        with open(self.config_single_pd_path, 'w') as fout:
            json.dump(pd_data, fout)
        self.config_single_bak_path = self.test_dir / "config_bak.json"
        self.config_single_pd_bak_path = self.test_dir / "ms_bak_controller.json"


    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.yaml_dir)
        os.unlink(self.temp_file.name)

    def test_set_config_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        DisaggregationSimulator.set_config(origin_config, "a.b.c", 4)
        assert origin_config["a"]["b"]["c"] == 4

    def test_set_config_list(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.b.0.c", 4)
        assert origin_config["a"]["b"][0]["c"] == 4

    def test_set_config_new_key(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.b.0.d", 4)
        assert origin_config["a"]["b"][0]["d"] == 4

    def test_set_config_add_dict_list_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        DisaggregationSimulator.set_config(origin_config, "a.d.0.c", 4)
        assert origin_config["a"]["d"][0]["c"] == 4

    def test_set_config_add_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.b.1.c", 4)
        assert origin_config["a"]["b"][1]["c"] == 4

    def test_set_config_add_dict_list_dict_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.d.0.c.e", 4)
        assert origin_config["a"]["d"][0]["c"]["e"] == 4

    @patch('experimental.optimizer.plugins.simulate.logger')
    def test_is_int(self, mock_logger):
        # 测试is_int方法
        self.assertTrue(DisaggregationSimulator.is_int('123'))
        self.assertFalse(DisaggregationSimulator.is_int('abc'))

    @patch('experimental.optimizer.plugins.simulate.subprocess')
    @patch('experimental.optimizer.plugins.simulate.logger')
    def test_prepare_before_start_server(self, mock_logger, mock_subprocess):
        GlobalConfig.custom_return = True
        # 测试prepare_before_start_server方法
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)
        simulator.prepare_before_start_server()
        # 验证子进程是否正确运行
        mock_subprocess.run.assert_called()
        # 验证日志记录是否正确
        mock_logger.debug.assert_called()
        GlobalConfig.reset()

    @patch('experimental.optimizer.plugins.simulate.logger')
    def test_stop(self, mock_logger):
        # 测试stop方法
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)
        simulator.stop()
        # 验证日志记录是否正确
        mock_logger.debug.assert_called()

    @patch('requests.post') 
    def test_curl_success(self, mock_post):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.config_single_path = self.config_single_path
        mindie_config.kubectl_single_path = self.kubectl_single_path
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        test_class = DisaggregationSimulator(mindie_config) 

        # Act
        result = test_class.test_curl()

        # Assert
        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch('requests.post') 
    def test_curl_failure(self, mock_post):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.kubectl_single_path = self.kubectl_single_path
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response
        test_class = DisaggregationSimulator(mindie_config)  

        # Act
        result = test_class.test_curl()

        # Assert
        self.assertFalse(result)
        mock_post.assert_called_once()

    @patch('requests.post')  
    def test_curl_exception(self, mock_post):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.kubectl_single_path = self.kubectl_single_path
        mock_post.side_effect = requests.exceptions.RequestException
        test_class = DisaggregationSimulator(mindie_config)  

        # Act
        result = test_class.test_curl()

        # Assert
        self.assertFalse(result)
        mock_post.assert_called_once()

    def test_update_config(self):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.config_single_path = self.config_single_path
        mindie_config.config_single_pd_path = self.config_single_pd_path
        simulator = DisaggregationSimulator(mindie_config)
        
        # 创建测试参数
        params = [
            OptimizerConfigField(config_position="BackendConfig.ModelDeployConfig.maxSeqLen", value=4096),
            OptimizerConfigField(config_position="default_p_rate", value=2)
        ]
        
        # Act
        simulator.update_config(params)
        
        # Assert
        with open(self.config_single_path, 'r') as f:
            config_data = json.load(f)
            self.assertEqual(config_data["BackendConfig"]["ModelDeployConfig"]["maxSeqLen"], 4096)
        
        with open(self.config_single_pd_path, 'r') as f:
            pd_config_data = json.load(f)
            self.assertEqual(pd_config_data["default_p_rate"], 2)

    @patch('experimental.optimizer.plugins.simulate.DisaggregationSimulator.test_curl')
    @patch('msguard.security.io.open_s')
    def test_health(self, mock_open, mock_test_curl):
        # Arrange
        GlobalConfig.custom_return = True
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)
        simulator.run_log = self.temp_file.name
        simulator.mindie_log_offset = 0
        
        # 模拟文件读取返回包含成功信息的内容
        mock_file = Mock()
        mock_file.read.return_value = "MindIE-MS coordinator is ready!!!"
        mock_file.tell.return_value = 100
        mock_open.return_value.__enter__.return_value = mock_file
        
        # 模拟test_curl返回True
        mock_test_curl.return_value = True
        
        # Act
        result = simulator.health()
        
        # Assert
        self.assertTrue(result)
        mock_test_curl.assert_called_once()
        GlobalConfig.reset()
    
    @patch('experimental.optimizer.plugins.simulate.DisaggregationSimulator.update_config')
    @patch('experimental.optimizer.plugins.simulate.DisaggregationSimulator.start_server')
    @patch('experimental.optimizer.plugins.simulate.logger')
    def test_run(self, mock_logger, mock_start_server, mock_update_config):
        # Arrange
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)
        
        # 创建测试参数
        params = [OptimizerConfigField(config_position="BackendConfig.ModelDeployConfig.maxSeqLen", value=4096)]
        
        # Act
        simulator.run(params)
        
        # Assert
        mock_logger.info.assert_called_once()
        mock_update_config.assert_called_once_with(params)
        mock_start_server.assert_called_once_with(params)