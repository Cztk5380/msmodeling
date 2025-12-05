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
import time
from pathlib import Path
from typing import Tuple, Optional
from math import isclose
import numpy as np
from loguru import logger

from msserviceprofiler.modelevalstate.common import get_train_sub_path, is_mindie, is_vllm
from msserviceprofiler.modelevalstate.config.base_config import REAL_EVALUATION
from msserviceprofiler.modelevalstate.config.config import get_settings, PerformanceIndex, OptimizerConfigField, \
    map_param_with_value, CommunicationConfig, Stage
from msserviceprofiler.modelevalstate.config.base_config import FOLDER_LIMIT_SIZE, REQUESTRATES
from msserviceprofiler.modelevalstate.optimizer.communication import CommunicationForFile, CustomCommand
from msserviceprofiler.modelevalstate.optimizer.plugins.simulate import VllmSimulator, Simulator, \
    DisaggregationSimulator
from msserviceprofiler.modelevalstate.optimizer.store import DataStorage
from msserviceprofiler.modelevalstate.optimizer.utils import get_folder_size


class Scheduler:
    def __init__(self, simulator, benchmark, data_storage: DataStorage,
                 bak_path: Optional[Path] = None, retry_number: int = 3, wait_start_time=1800):
        self.simulator = simulator
        self.benchmark = benchmark
        self.data_storage = data_storage
        self.bak_path = bak_path
        self.retry_number = retry_number
        self.wait_time = wait_start_time
        self.current_back_path = None
        self.simulate_run_info = None
        self.performance_index = None
        self.error_info = None
        self.run_start_timestamp = None
        self.first_duration = None
        self.del_log = None

    def set_back_up_path(self):
        if self.bak_path:
            if get_folder_size(self.bak_path) > FOLDER_LIMIT_SIZE:
                self.simulator.bak_path = None
                self.benchmark.bak_path = None
            else:
                self.current_back_path = get_train_sub_path(self.bak_path)
                self.simulator.bak_path = self.current_back_path
                self.benchmark.bak_path = self.current_back_path

    def wait_simulate(self):
        logger.debug("wait run simulator")
        for _ in range(self.wait_time):
            time.sleep(1)
            if hasattr(self.simulator, "check_success") and self.simulator.check_success():
                logger.info(f"Successfully started the {self.simulator.process} process.")
                return
            if hasattr(self.simulator, "health") and self.simulator.health().stage == Stage.running:
                logger.info(f"Successfully started the {self.simulator.process} process.")
                return
        raise TimeoutError(self.wait_time)

    def run_simulate(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        if hasattr(self.benchmark, "prepare"):
            self.benchmark.prepare()
        self.simulator.run(tuple(self.simulate_run_info))
        self.wait_simulate()

    def backup(self):
        self.simulator.backup()
        self.benchmark.backup()

    def monitoring_status(self):
        logger.debug("monitor status")
        for _ in range(get_settings().particles_time_out):
            if hasattr(self.simulator, "check_success"):
                if is_mindie() or is_vllm():
                    if self.simulator.process.poll() is not None:
                        raise subprocess.SubprocessError(f"Failed in run simulator. "
                                                        f"return code: {self.simulator.process.returncode}.")
                if self.benchmark.check_success():
                    return
            if hasattr(self.simulator, "health"):
                if not isinstance(self.simulator, (DisaggregationSimulator, Simulator)):
                    res = self.simulator.health()
                    if res.stage != Stage.running:
                        raise subprocess.SubprocessError(f"Failed in run simulator. error: {res.stage} "
                                                        f"info: {res.info}.")
                res = self.benchmark.health()
                if res.stage != Stage.running:
                    return
            if self.run_start_timestamp and self.first_duration:
                _duration = time.time() - self.run_start_timestamp
                if _duration > 2 * self.first_duration:
                    raise TimeoutError("The current runtime is more than twice the duration of the first run.")
            time.sleep(1)

        raise TimeoutError(f"{get_settings().particles_time_out}")

    def run_target_server(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        """
        1. 启动mindie仿真
        2. 启动benchmark 测试
        3. 检查mindie状态，检查benchmark状态
        """
        for _ in range(self.retry_number):
            try:
                self.run_simulate(params, params_field)
            except Exception as e:
                logger.error(f"Failed in simulator Running. error: {e}, \n"
                             f"simulator log {self.simulator.run_log}. \n"
                             f"log last info \n{self.simulator.get_last_log()}")
                logger.exception("what?!")
                self.stop_target_server(False)
                continue
            time.sleep(1)
            try:
                self.benchmark.run(tuple(self.simulate_run_info))
            except Exception as e:
                logger.error(f"Failed in Benchmark Running. error: {e},\n"
                             f"benchmark log {self.benchmark.run_log},\n"
                             f"log last info \n{self.benchmark.get_last_log()}")
                self.stop_target_server(False)
                continue
            time.sleep(1)
            try:
                self.monitoring_status()
            except Exception as e:
                self.stop_target_server(False)
                logger.error(f"Failed in monitoring status. error: {e}, \n"
                             f"simulator log {self.simulator.run_log}, \n"
                             f"log last info {self.simulator.get_last_log()}.\n"
                             f"benchmark log {self.benchmark.run_log}, \n"
                             f"log last info \n{self.benchmark.get_last_log()}.")
                continue
            return
        raise ValueError(
            f"Failed in run_target_server")

    def stop_target_server(self, del_log: bool = False):
        self.simulator.stop(del_log)
        self.benchmark.stop(del_log)

    def save_result(self, **kwargs):
        duration = None
        if self.run_start_timestamp:
            duration = time.time() - self.run_start_timestamp
            if not self.first_duration:
                self.first_duration = duration
        real_evaluation = True
        if REAL_EVALUATION in kwargs:
            real_evaluation = kwargs.pop(REAL_EVALUATION)
        self.data_storage.save(self.performance_index, tuple(self.simulate_run_info),
                               error=self.error_info, backup=self.current_back_path, duration=duration,
                               real_evaluation=real_evaluation, **kwargs)
        if self.bak_path:
            self.backup()
        self.stop_target_server()

    def update_data_field(self, params_field: Tuple[OptimizerConfigField]):
        if hasattr(self.simulator, "data_field"):
            self.simulator.data_field = params_field
        if hasattr(self.simulator, "update_command"):
            self.simulator.update_command()
        if hasattr(self.benchmark, "data_field"):
            self.benchmark.data_field = params_field
        if hasattr(self.benchmark, "update_command"):
            self.benchmark.update_command()

    def run(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]) -> PerformanceIndex:
        """
        1. 启动mindie仿真
        2. 启动benchmark 测试
        3. 获取benchmark测试结果
        4. 关闭mindie仿真
        5. 返回benchmark测试结果
        params: 是一维数组，其值对应mindie 的相关配置。
        """
        self.run_start_timestamp = time.time()
        logger.debug("Start run in scheduler.")
        self.set_back_up_path()
        self.simulate_run_info = map_param_with_value(params, params_field)
        logger.info("run param info {}", {v.name: v.value for v in self.simulate_run_info})
        self.error_info = None
        self.del_log = True
        self.performance_index = PerformanceIndex()
        try:
            self.update_data_field(self.simulate_run_info)
            self.run_target_server(params, params_field)
            time.sleep(1)
            self.performance_index = self.benchmark.get_performance_index()
        except Exception as e:
            logger.error(f"Failed running. bak path: {self.simulator.bak_path}. error {e}"
                         f"simulator log {self.simulator.run_log}, benchmark log {self.benchmark.run_log}")
            self.error_info = e
            self.del_log = False
        finally:
            self.stop_target_server(self.del_log)
        return self.performance_index

    def run_with_request_rate(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]) -> PerformanceIndex:
        """
        运行服务，先运行最大并发，获取request rate，然后再根据并发和request rate运行，最后一组作为评估结果
        params: 是一维数组，其值对应mindie 的相关配置。
        """
        self.run_start_timestamp = time.time()
        self.set_back_up_path()
        self.simulate_run_info = map_param_with_value(params, params_field)
        logger.info("run param info {}", {v.name: v.value for v in self.simulate_run_info})
        self.error_info = None
        self.del_log = True
        self.performance_index = PerformanceIndex()
        try:
            self.update_data_field(self.simulate_run_info)
            self.run_target_server(params, params_field)
            time.sleep(1)
            self.performance_index = self.benchmark.get_performance_index()
            self.benchmark.stop()
            for _field in self.simulate_run_info:
                if _field.name in REQUESTRATES:
                    if not isclose(_field.min, _field.max):
                        _field.value = _field.find_available_value(self.performance_index.throughput * 1.05)
            logger.info("second run param info {}", {v.name: v.value for v in self.simulate_run_info})
            if hasattr(self.benchmark, "data_field"):
                self.benchmark.data_field = params_field
            self.benchmark.update_command()
            try:
                if hasattr(self.benchmark, "prepare"):
                    self.benchmark.prepare()
                self.benchmark.run(tuple(self.simulate_run_info))
            except Exception as e:
                logger.error(f"Failed in Benchmark Running. error: {e}, benchmark log {self.benchmark.run_log}")
                raise e
            try:
                self.monitoring_status()
            except Exception as e:
                logger.error(f"Failed in monitoring status. error: {e}, simulator log {self.simulator.run_log}, "
                             f"benchmark log {self.benchmark.run_log}")
                raise e
            time.sleep(1)
            self.performance_index = self.benchmark.get_performance_index()
        except Exception as e:
            logger.error(f"Failed running. bak path: {self.simulator.bak_path}. error {e}"
                         f"simulator log {self.simulator.run_log}, benchmark log {self.benchmark.run_log}")
            self.error_info = e
            self.del_log = False
        finally:
            self.stop_target_server(self.del_log)
        return self.performance_index


class ScheduleWithMultiMachine(Scheduler):
    def __init__(self, communication_config: CommunicationConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.communication_config = communication_config
        self.communication = CommunicationForFile(self.communication_config.cmd_file,
                                                  self.communication_config.res_file,
                                                  )
        self.cmd = CustomCommand()
        _cmd = self.cmd.init
        self.communication.send_command(_cmd)
        self.communication.clear_command(_cmd)

    def set_back_up_path(self):
        if self.bak_path:
            if get_folder_size(self.bak_path) > FOLDER_LIMIT_SIZE:
                self.simulator.bak_path = None
                self.benchmark.bak_path = None
            else:
                _cur_bak_path = get_train_sub_path(self.bak_path)
                self.simulator.bak_path = _cur_bak_path
                self.benchmark.bak_path = _cur_bak_path
                _cmd = f"{self.cmd.backup} params:{_cur_bak_path}"
                self.communication.send_command(_cmd)
                self.communication.clear_command(_cmd)

    def monitoring_status(self):
        logger.debug("Start monitoring")
        while True:
            _cmd = self.cmd.process_poll
            self.communication.send_command(_cmd)
            all_poll = [self.simulator.process.poll(), self.communication.clear_command(_cmd)]
            if any([_i is not None for _i in all_poll]):
                self.stop_target_server(del_log=False)
                raise subprocess.SubprocessError(
                    f"Failed in run simulator. all status: {all_poll}.")
            if self.benchmark.check_success():
                return
            time.sleep(1)

    def run_simulate(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        self.benchmark.prepare()
        _cmd = f"{self.cmd.start} params:{params.tolist()}"
        self.cmd.history = _cmd
        self.communication.send_command(_cmd)
        self.communication.clear_command(_cmd)
        self.simulator.run(tuple(self.simulate_run_info))
        self.wait_simulate()
        # wait 其他服务器上的服务成功。
        _cmd = self.cmd.check_success
        self.cmd.history = _cmd
        self.communication.send_command(_cmd)
        self.communication.clear_command(_cmd)

    def stop_target_server(self, del_log: bool = True):
        super(ScheduleWithMultiMachine, self).stop_target_server(del_log)
        # wait 其他服务器上的服务成功。
        _cmd = f"{self.cmd.stop} params:{del_log}"
        self.communication.send_command(_cmd)
        self.communication.clear_command(_cmd)
        self.cmd.history = _cmd
