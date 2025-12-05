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
import subprocess
import tempfile
import time
from math import isnan, isinf
from pathlib import Path
from typing import Any, Tuple, Optional, List

import psutil
from loguru import logger

from msserviceprofiler.modelevalstate.config.base_config import CUSTOM_OUTPUT, MODEL_EVAL_STATE_CONFIG_PATH, \
    modelevalstate_config_path
from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField, get_settings, ProcessState, Stage
from msserviceprofiler.modelevalstate.optimizer.utils import close_file_fp, remove_file, kill_children, \
    backup, kill_process
from msserviceprofiler.msguard.security import open_s


class CustomProcess:
    def __init__(self, bak_path: Optional[Path] = None, command: Optional[List[str]] = None,
                 work_path: Optional[Path] = None, print_log: bool = False,
                 process_name: str = ""):
        self.command = command
        self.bak_path = bak_path
        self.work_path = work_path if work_path else os.getcwd()
        self.run_log = None
        self.run_log_offset = None
        self.run_log_fp = None
        self.process = None
        self.print_log = print_log
        self.process_name = process_name
        self.env = os.environ.copy()
        self.command = None

    @staticmethod
    def kill_residual_process(process_name):
        """
        检查环境，查看是否有残留任务  清理残留任务
        """
        logger.debug("check env")
        _residual_process = []
        _all_process_name = process_name.split(",")
        for proc in psutil.process_iter(["pid", "name"]):
            if not hasattr(proc, "info"):
                continue
            _proc_flag = []
            for p in _all_process_name:
                if p not in proc.info["name"]:
                    _proc_flag.append(True)
                else:
                    _proc_flag.append(False)
            if all(_proc_flag):
                continue
            _residual_process.append(proc)
        if _residual_process:
            logger.debug("kill residual_process")
            for _p_name in _all_process_name:
                try:
                    kill_process(_p_name)
                except Exception as e:
                    logger.error(f"Failed to kill process. {e}")
        time.sleep(1)


    def backup(self):
        # 备份操作，默认备份日志
        backup(self.run_log, self.bak_path, self.__class__.__name__)

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None):
        """
        运行命令前的准备工作
        Args:
            run_params: 调优参数列表，元组，每一个元素的value和config position进行定义
        """
        self.run_log_fp, self.run_log = tempfile.mkstemp(prefix="modelevalstate_")
        self.run_log_offset = 0
        if not run_params:
            return
        for k in run_params:
            if k.config_position == "env":
                # env 类型的数据，设置环境变量和更新命令中包含的变量,设置时全部为大写
                self.env[k.name.upper().strip()] = str(k.value)
                _var_name = f"${k.name.upper().strip()}"
                if _var_name not in self.command:
                    continue
                _i = self.command.index(_var_name)
                value_flag = k.value is None or isnan(k.value) or isinf(k.value)
                if value_flag or not str(k).strip():
                    self.command.pop(_i)
                    if _i > 0:
                        self.command.pop(_i - 1)
                else:
                    self.command[_i] = str(k.value)
        if CUSTOM_OUTPUT not in self.env:
            # 设置输出目录
            self.env[CUSTOM_OUTPUT] = str(get_settings().output)
        # 设置读取的json文件
        if MODEL_EVAL_STATE_CONFIG_PATH not in self.env:
            self.env[MODEL_EVAL_STATE_CONFIG_PATH] = str(modelevalstate_config_path)
                

    def run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None, **kwargs):
        # 启动测试
        if self.process_name:
            try:
                self.kill_residual_process(self.process_name)
            except Exception as e:
                logger.error(f"Failed to kill residual process. {e}")
        self.before_run(run_params)
        for i, v in enumerate(self.command):
            if not v.strip():
                continue
            if '-' or '--' not in v:
                continue
            if v in self.command[:i]:
                logger.warning("{} field appears multiple times in the command. please confirm.", v)
        for k, v in self.env.items():
            if isinstance(k, str) and isinstance(v, str):
                continue
            else:
                logger.error(f"Possible Problem with Environment Variable Type. "
                             f"env: {k}={v}, k type: {type(k)}, v type: {type(v)}")
        try:
            self.process = subprocess.Popen(self.command, env=self.env, stdout=self.run_log_fp,
                                            stderr=subprocess.STDOUT, cwd=self.work_path)
        except OSError as e:
            logger.error(f"Failed to run {self.command}. error {e}")
            raise e
        logger.info(f"Start running the command: {' '.join(self.command)}, log file: {self.run_log}")

    def get_log(self):
        output = None
        if not self.run_log:
            return output
        run_log_path = Path(self.run_log)
        if run_log_path.exists():
            try:
                with open_s(run_log_path, "r", encoding="utf-8") as f:
                    f.seek(self.run_log_offset)
                    output = f.read()
                    self.run_log_offset = f.tell()
            except (UnicodeError, OSError) as e:
                logger.error(f"Failed read {self.command} log. error {e}")
        return output

    def health(self):
        """
        检查任务是否运行成功
        Returns: 返回bool值，检查程序是否成功启动
        """
        if self.print_log:
            output = self.get_log()
            logger.debug(output)
        if self.process.poll() is None:
            return ProcessState(stage=Stage.running)
        elif self.process.poll() == 0:
            return ProcessState(stage=Stage.stop)
        else:
            return ProcessState(stage=Stage.error, info=f"Failed in run {self.command!r}. \
                                        return code: {self.process.returncode}. log: {self.run_log}")

    def stop(self, del_log: bool = True):
        self.run_log_offset = 0
        close_file_fp(self.run_log_fp)
        if del_log and self.run_log:
            remove_file(Path(self.run_log))
        if not self.process:
            return
        _process_state = self.process.poll()
        if _process_state is not None:
            logger.info(f"The program has exited. exit_code: {_process_state}")
            return
        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
            self.process.kill()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.send_signal(9)
            if self.process.poll() is not None:
                logger.debug(f"The {self.process.pid} process has been shut down.")
            else:
                logger.error(f"The {self.process.pid} process shutdown failed.")
            kill_children(children)
        except Exception as e:
            logger.error(f"Failed to stop simulator process. {e}")

    def get_last_log(self, number: int = 5):
        output = None
        if not self.run_log:
            return output
        run_log_path = Path(self.run_log)
        if run_log_path.exists():
            file_lines = []
            try:
                with open_s(run_log_path, "r", encoding="utf-8") as f:
                    file_lines = f.readlines()
            except (UnicodeError, OSError) as e:
                logger.error(f"Failed read {self.command} log. error {e}")
            number = min(number, len(file_lines))
            output = '\n'.join(file_lines[-number:])
        return output


class BaseDataField:
    def __init__(self, config: Optional[Any] = None):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.ais_bench
 
    @property
    def data_field(self) -> Tuple[OptimizerConfigField, ...]:
        """
        获取data field 属性
        """
        if hasattr(self.config, "target_field") and self.config.target_field:
            return tuple(self.config.target_field)
        return ()
 
    @data_field.setter
    def data_field(self, value: Tuple[OptimizerConfigField] = ()) -> None:
        """
        提供新的数据，更新替换data field属性。
        """
        _default_name = []
        if hasattr(self.config, "target_field") and self.config.target_field:
            _default_name = [_f.name for _f in self.config.target_field]
        for _field in value:
            if _field.name not in _default_name:
                continue
            _index = _default_name.index(_field.name)
            self.config.target_field[_index] = _field
