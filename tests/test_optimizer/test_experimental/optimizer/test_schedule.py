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
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from experimental.config.config import (
    CommunicationConfig, map_param_with_value, 
    get_settings, default_support_field, Stage,
    OptimizerConfigField, PerformanceIndex
)
from experimental.optimizer.communication import CommunicationForFile, CustomCommand
from experimental.optimizer.scheduler import ScheduleWithMultiMachine
from experimental.config.base_config import FOLDER_LIMIT_SIZE
from experimental.optimizer.scheduler import Scheduler


class TestScheduleWithMultiMachine:
    @classmethod
    def test_schedule_with_multi_machine_init(cls):
        # 创建模拟对象
        mock_communication_config = MagicMock(spec=CommunicationConfig)
        mock_communication = MagicMock(spec=CommunicationForFile)
        mock_custom_command = MagicMock(spec=CustomCommand)

        # 模拟CommunicationForFile和CustomCommand的构造函数
        with patch('experimental.optimizer.scheduler.CommunicationForFile',
                   return_value=mock_communication) as mock_communication_class, \
                patch('experimental.optimizer.scheduler.CustomCommand',
                      return_value=mock_custom_command) as mock_custom_command_class:
            mock_communication_config.cmd_file = None
            mock_communication_config.res_file = None
            # 创建ScheduleWithMultiMachine实例
            scheduler = ScheduleWithMultiMachine(mock_communication_config, MagicMock(), MagicMock(), MagicMock())

            # 验证CommunicationForFile和CustomCommand是否被正确初始化
            mock_communication_class.assert_called_once_with(mock_communication_config.cmd_file,
                                                             mock_communication_config.res_file)
            mock_custom_command_class.assert_called_once()

            # 验证communication属性是否被正确设置
            assert scheduler.communication_config == mock_communication_config
            assert scheduler.communication == mock_communication

            # 验证cmd属性是否被正确设置
            assert scheduler.cmd == mock_custom_command

            # 验证send_command和clear_cmd方法是否被正确调用
            mock_communication.send_command.assert_called_once_with(mock_custom_command.init)
            mock_communication.clear_cmd.assert_called_once_with(mock_custom_command.init)
    
    @classmethod
    def test_run_simulate(cls, schedule_with_multi_machine):
        # 模拟方法
        schedule_with_multi_machine.cmd = CustomCommand()
        schedule_with_multi_machine.benchmark.prepare.return_value = None
        schedule_with_multi_machine.communication.send_command.return_value = None
        schedule_with_multi_machine.communication.clear_cmd.return_value = None
        schedule_with_multi_machine.simulator.run.return_value = None
        schedule_with_multi_machine.wait_simulate = MagicMock(return_value=None)
        _params = np.random.random(len(default_support_field))
        # 调用run_simulate方法
        schedule_with_multi_machine.simulate_run_info = map_param_with_value(_params, default_support_field)
        schedule_with_multi_machine.run_simulate(_params, default_support_field)

        # 验证方法调用
        schedule_with_multi_machine.benchmark.prepare.assert_called_once()
        assert schedule_with_multi_machine.communication.send_command.call_count == 2
        assert schedule_with_multi_machine.communication.clear_cmd.call_count == 2
        schedule_with_multi_machine.wait_simulate.assert_called_once()

    @pytest.fixture
    def schedule_with_multi_machine(self):
        mock_communication_config = MagicMock(spec=CommunicationConfig)
        mock_communication = MagicMock(spec=CommunicationForFile)
        mock_custom_command = MagicMock(spec=CustomCommand)
        mock_communication_config.cmd_file = None
        mock_communication_config.res_file = None
        with patch('experimental.optimizer.scheduler.CommunicationForFile',
                   return_value=mock_communication, autospec=True) as mock_communication_class, \
                patch('experimental.optimizer.scheduler.CustomCommand',
                      return_value=mock_custom_command, autospec=True) as mock_custom_command_class:
            schedule = ScheduleWithMultiMachine(get_settings().communication, MagicMock(), MagicMock(), MagicMock())
            schedule.cmd = MagicMock()
            schedule.communication = MagicMock()
            yield schedule

    @patch('experimental.optimizer.scheduler.get_train_sub_path')
    def test_set_back_up_path_with_bak_path(self, mock_get_train_sub_path, schedule_with_multi_machine, tmpdir):
        # Arrange
        bak_path = Path(tmpdir)
        schedule_with_multi_machine.bak_path = bak_path
        mock_get_train_sub_path.return_value = tmpdir
        # Act
        schedule_with_multi_machine.set_back_up_path()

        # Assert
        mock_get_train_sub_path.assert_called_once_with(bak_path)
        schedule_with_multi_machine.simulator.bak_path == bak_path
        schedule_with_multi_machine.benchmark.bak_path == bak_path
        schedule_with_multi_machine.communication.send_command.assert_called_once_with(
            f'{schedule_with_multi_machine.cmd.backup} params:{bak_path}')
        schedule_with_multi_machine.communication.clear_cmd.assert_called_once_with(
            f'{schedule_with_multi_machine.cmd.backup} params:{bak_path}')

    @patch('experimental.optimizer.scheduler.time.sleep', return_value=None)
    def test_monitoring_status_success(self, mock_sleep, schedule_with_multi_machine):
        type(schedule_with_multi_machine.cmd).process_poll = PropertyMock(return_value="mocked process poll")
        schedule_with_multi_machine.communication.send_command = MagicMock()
        schedule_with_multi_machine.simulator.process.poll = MagicMock(return_value=None)
        schedule_with_multi_machine.communication.clear_cmd = MagicMock(return_value=None)
        schedule_with_multi_machine.benchmark.check_success = MagicMock(return_value=True)
        schedule_with_multi_machine.stop_target_server = MagicMock()

        schedule_with_multi_machine.monitoring_status()

        schedule_with_multi_machine.communication.send_command.assert_called_with("mocked process poll")
        schedule_with_multi_machine.simulator.process.poll.assert_called()
        schedule_with_multi_machine.communication.clear_cmd.assert_called_with("mocked process poll")
        schedule_with_multi_machine.benchmark.check_success.assert_called()
        schedule_with_multi_machine.stop_target_server.assert_not_called()

    @patch('experimental.optimizer.scheduler.time.sleep', return_value=None)
    def test_monitoring_status_failure(self, mock_sleep, schedule_with_multi_machine):
        type(schedule_with_multi_machine.cmd).process_poll = PropertyMock(return_value='mocked process poll')
        schedule_with_multi_machine.communication.send_command = MagicMock()
        schedule_with_multi_machine.simulator.process.poll = MagicMock(return_value=1)
        schedule_with_multi_machine.communication.clear_cmd = MagicMock(return_value=None)
        schedule_with_multi_machine.benchmark.check_success = MagicMock(return_value=False)
        schedule_with_multi_machine.stop_target_server = MagicMock()

        with pytest.raises(subprocess.SubprocessError):
            schedule_with_multi_machine.monitoring_status()

        schedule_with_multi_machine.communication.send_command.assert_called_with('mocked process poll')
        schedule_with_multi_machine.simulator.process.poll.assert_called()
        schedule_with_multi_machine.communication.clear_cmd.assert_called_with('mocked process poll')
        schedule_with_multi_machine.benchmark.check_success.assert_not_called()
        schedule_with_multi_machine.stop_target_server.assert_called()

    @patch('experimental.optimizer.scheduler.Scheduler.stop_target_server')
    def test_stop_target_server_with_del_log(self, mock_super_stop, schedule_with_multi_machine):
        # 测试当del_log为True时的行为
        schedule_with_multi_machine.cmd = CustomCommand()
        schedule_with_multi_machine.stop_target_server(del_log=True)

        # 验证Scheduler.stop_target_server被调用
        mock_super_stop.assert_called_once_with(True)

        # 验证communication.send_command和communication.clear_cmd被正确调用
        assert "params:True" in schedule_with_multi_machine.cmd.history[-1]
        schedule_with_multi_machine.communication.send_command.assert_called_once_with(
            f"{schedule_with_multi_machine.cmd.history[-1]}")
        schedule_with_multi_machine.communication.clear_cmd.assert_called_once_with(
            f"{schedule_with_multi_machine.cmd.history[-1]}")
        

class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.bak_path = MagicMock()
        self.scheduler = Scheduler(self.simulator, self.benchmark, self.data_storage, self.bak_path)

    @patch('experimental.optimizer.utils.get_folder_size')
    def test_set_back_up_path_folder_size_exceeds_limit(self, mock_get_folder_size):
        mock_get_folder_size.return_value = FOLDER_LIMIT_SIZE + 1
        self.scheduler.set_back_up_path()

    @patch('experimental.optimizer.utils.get_folder_size')
    @patch('experimental.common.get_train_sub_path')
    def test_set_back_up_path_folder_size_within_limit(self, mock_get_train_sub_path, mock_get_folder_size):
        mock_get_folder_size.return_value = FOLDER_LIMIT_SIZE - 1
        mock_get_train_sub_path.return_value = 'sub_path'
        self.scheduler.set_back_up_path()

    @patch('time.sleep', return_value=None)
    def test_wait_simulate_success(self, mock_sleep):
        self.simulator.health.return_value.stage = Stage.running
        self.scheduler.wait_simulate()

    @patch('time.sleep', return_value=None)
    def test_wait_simulate_timeout(self, mock_sleep):
        self.simulator.health.return_value.stage = Stage.error
        self.scheduler.wait_simulate()

    @patch('time.sleep', return_value=None)
    def test_run_target_server_benchmark_error(self, _):
        self.benchmark.run.side_effect = Exception('Benchmark error')
        with self.assertRaises(Exception):
            self.scheduler.run_target_server(np.array([1, 2, 3]), ('field1', 'field2'))

    @patch('time.sleep', return_value=None)
    def test_run_target_server_monitoring_error(self, _):
        self.scheduler.monitoring_status = MagicMock(side_effect=Exception('Monitoring error'))
        with self.assertRaises(Exception):
            self.scheduler.run_target_server(np.array([1, 2, 3]), ('field1', 'field2'))


class TestSchedulerRunMethods(unittest.TestCase):
    
    def setUp(self):
        # 创建测试对象
        self.simulator = MagicMock()
        self.benchmark = MagicMock()
        self.data_storage = MagicMock()
        self.scheduler = Scheduler(
            simulator=self.simulator,
            benchmark=self.benchmark,
            data_storage=self.data_storage
        )
        
        # 准备测试数据
        self.params = np.array([1.0, 2.0, 3.0])
        self.field1 = OptimizerConfigField(name="param1", value=1.0, min=0.0, max=5.0)
        self.field2 = OptimizerConfigField(name="param2", value=2.0, min=0.0, max=5.0)
        self.field3 = OptimizerConfigField(name="param3", value=3.0, min=0.0, max=5.0)
        self.params_field = (self.field1, self.field2, self.field3)
        
        # 模拟性能指标
        self.performance_index = PerformanceIndex()
        self.performance_index.throughput = 100.0
        self.benchmark.get_performance_index.return_value = self.performance_index
    
    @patch('time.time')
    def test_run_with_fixed_request_rate(self, mock_time):
        """测试run_with_request_rate方法在固定request rate时的行为"""
        # 设置模拟返回值
        mock_time.return_value = 1000.0
        
        # 添加固定REQUESTRATE字段 (min == max)
        req_rate_field = OptimizerConfigField(name="REQUESTRATE", value=50.0, min=50.0, max=50.0)
        params_field_with_fixed_req_rate = self.params_field + (req_rate_field,)
        
        # 调用被测方法
        self.scheduler.run_with_request_rate(self.params, params_field_with_fixed_req_rate)
        
        # 验证request rate没有被修改
        self.assertEqual(req_rate_field.value, 50.0)
        self.assertEqual(req_rate_field.min, 50.0)
        self.assertEqual(req_rate_field.max, 50.0)
    
    @patch('time.time')
    @patch('experimental.optimizer.scheduler.logger')
    def test_run_logging(self, mock_logger, mock_time):
        """测试run方法的日志记录"""
        # 设置模拟返回值
        mock_time.return_value = 1000.0
        
        # 调用被测方法
        self.scheduler.run(self.params, self.params_field)