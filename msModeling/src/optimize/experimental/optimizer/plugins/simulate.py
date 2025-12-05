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
import json
import os
import stat
import subprocess
import tempfile
import time
from copy import deepcopy
from typing import Any, Optional, Tuple, List
import shutil
from dataclasses import dataclass
import yaml
from loguru import logger
from msserviceprofiler.modelevalstate.config.config import get_settings, OptimizerConfigField, VllmConfig, \
    MindieConfig, KubectlConfig, Stage, ProcessState
from msserviceprofiler.modelevalstate.config.custom_command import VllmCommand, MindieCommand
from msserviceprofiler.modelevalstate.optimizer.interfaces.simulator import SimulatorInterface
from msserviceprofiler.modelevalstate.optimizer.utils import remove_file, close_file_fp, backup
from msserviceprofiler.msguard.security import open_s


@dataclass
class ConfigContextdict:
    origin_config: dict
    cur_key: str
    next_key: str
    next_level: str
    value: Any
    current_depth: int


@dataclass
class ConfigContextlist:
    origin_config: list
    cur_key: str
    next_key: str
    next_level: str
    value: Any
    current_depth: int


class Simulator(SimulatorInterface):
    def __init__(self, *args, config: Optional[MindieConfig] = None, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.mindie
        super().__init__(*args, process_name=self.config.process_name, **kwargs)
        logger.debug(f"config path {self.config.config_path}", )
        if not self.config.config_path.exists():
            raise FileNotFoundError(self.config.config_path)
        with open_s(self.config.config_path, "r") as f:
            data = json.load(f)
        self.default_config = data
        logger.debug(f"config bak path {self.config.config_bak_path}", )
        if self.config.config_bak_path.exists():
            self.config.config_bak_path.unlink()
        with open_s(self.config.config_bak_path, 'w') as fout:
            json.dump(self.default_config, fout, indent=4)
        self.command = MindieCommand(self.config.command).command

    @property
    def base_url(self) -> str:
        """
        获取服务的base url 属性
        Returns:

        """
        pass

    @staticmethod
    def is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False

    @staticmethod
    def set_config_for_dict(context: ConfigContextdict):
        if context.cur_key in context.origin_config:
            Simulator.set_config(context.origin_config[context.cur_key], context.next_level, context.value, 
                                 context.current_depth)
        elif Simulator.is_int(context.cur_key):
            raise KeyError(f"data: {context.origin_config}, key: {context.cur_key}")
        elif Simulator.is_int(context.next_key):
            context.origin_config[context.cur_key] = []
            Simulator.set_config(context.origin_config[context.cur_key], context.next_level, context.value, 
                                 context.current_depth)
        else:
            context.origin_config[context.cur_key] = {}
            Simulator.set_config(context.origin_config[context.cur_key], context.next_level, context.value, 
                                 context.current_depth)

    @staticmethod
    def set_config_for_list(context: ConfigContextlist):
        if len(context.origin_config) > int(context.cur_key):
            Simulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, context.value, 
                                 context.current_depth)
        elif len(context.origin_config) == int(context.cur_key) and Simulator.is_int(context.next_key):
            context.origin_config.append([])
            Simulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, context.value, 
                                 context.current_depth)
        elif len(context.origin_config) == int(context.cur_key) and not Simulator.is_int(context.next_key):
            context.origin_config.append({})
            Simulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, context.value, 
                                 context.current_depth)
        else:
            raise IndexError(f"data: {context.origin_config}, index: {context.cur_key}")

    @staticmethod
    def set_config(origin_config, key: str, value: Any, current_depth=0):
        if current_depth > 10:
            raise RecursionError("Exceeded maximum recursion depth")
        next_level = None
        try:
            if "." in key:
                _f_index = key.index(".")
                _cur_key, next_level = key[:_f_index], key[_f_index + 1:]
            else:
                _cur_key = key
            if next_level is None:
                if isinstance(origin_config, dict):
                    origin_config[_cur_key] = value
                elif isinstance(origin_config, list):
                    if len(origin_config) > int(_cur_key):
                        origin_config[int(_cur_key)] = value
                    else:
                        origin_config.append(value)
                return
            if "." in next_level:
                _next_index = next_level.index(".")
                _next_key = next_level[:_next_index]
            elif next_level:
                _next_key = next_level
            else:
                _next_key = None
        except Exception as e:
            logger.error(f"Unexpected error occurred at {key}")
            raise e
        if isinstance(origin_config, dict):
            context = ConfigContextdict(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1
            )
            Simulator.set_config_for_dict(context)
        elif isinstance(origin_config, list):
            context = ConfigContextlist(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1
            )
            Simulator.set_config_for_list(context)
        else:
            raise ValueError(f"Not Support type {type(origin_config)}")

    def update_command(self):
        self.command = MindieCommand(self.config.command).command

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None):
        # 根据params 修改配置文件
        # 启动mindie仿真
        self.update_config(run_params)
        super().before_run(run_params)
        subprocess.run(["pkill", "-9", "mindie"], env=self.env, stdout=self.run_log_fp,
                       stderr=subprocess.STDOUT, text=True, cwd=self.work_path)
        subprocess.run(["npu-smi", "info"], env=self.env, stdout=self.run_log_fp,
                       stderr=subprocess.STDOUT, text=True, cwd=self.work_path)
    
    def backup(self):
        super().backup()
        backup(self.config.config_path, self.bak_path, self.__class__.__name__)

    def health(self):
        """
        获取当前服务的状态.
        当前实现为基础的vllm url
        Returns: None

        """
        process_res = super().health()
        if process_res.stage != Stage.running:
            # 当前mindie health接口不可用http://127.0.0.1:8825/v2/health/live
            proxy_status = super(SimulatorInterface, self).health()
            self.run_log_offset = 0
            output = self.get_log()
            if output and "Daemon start success!" in output and proxy_status.stage == Stage.running:
                return proxy_status
        return process_res

    def update_config(self, params: Optional[Tuple[OptimizerConfigField]] = None):
        # 将params值更新到新的config中
        if not params:
            return
        new_config = deepcopy(self.default_config)
        for p in params:
            if not p.config_position.startswith("BackendConfig"):
                continue
            Simulator.set_config(new_config, p.config_position, p.value)

        # 将新的config写入到config文件中
        logger.debug(f"new config {new_config}")
        if self.config.config_path.exists():
            self.config.config_path.unlink()
        with open_s(self.config.config_path, "w") as fout:
            json.dump(new_config, fout, indent=4, ensure_ascii=False)

    def stop(self, del_log: bool = True):
        # 恢复默认的mindie 配置
        remove_file(self.config.config_path)
        with open_s(self.config.config_path, "w") as fout:
            json.dump(self.default_config, fout, indent=4, ensure_ascii=False)
        super().stop(del_log)


class VllmSimulator(SimulatorInterface):
    def __init__(self, config: Optional[VllmConfig] = None, *args, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.vllm
        super().__init__(*args, process_name=self.config.process_name, **kwargs)

        self.command = VllmCommand(self.config.command).command

    @property
    def base_url(self) -> str:
        """
        获取服务的base url 属性
        Returns:

        """
        return f"http://127.0.0.1:{self.config.command.port}/health"

    def stop(self, del_log: bool = True):
        """
        运行时，其他的准备工作。
        Returns:

        """
        pkill_path = shutil.which("pkill")
        try:
            subprocess.run([pkill_path, "-15", "vllm"], stderr=subprocess.STDOUT, text=True)
        except subprocess.SubprocessError:
            logger.warning(f"Failed to stop vllm process with pkill.")
        super().stop(del_log)

    def update_command(self):
        self.command = VllmCommand(self.config.command).command


class DisaggregationSimulator(SimulatorInterface):
    from msserviceprofiler.modelevalstate.config.custom_command import KubectlCommand

    def __init__(self, *args, config: Optional[KubectlConfig] = None, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.kubectl
            self.config.target_field = settings.mindie.target_field
        super().__init__(*args, process_name=self.config.process_name, **kwargs)
        if not self.config.config_single_path.exists():
            raise FileNotFoundError(self.config.config_single_path)
        with open_s(self.config.config_single_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed in read config.json. file: {self.config.config_single_path}")
        with open_s(self.config.config_single_pd_path, "r") as f:
            try:
                pd_data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed in read ms_controller.json. file: {self.config.config_single_pd_path}")
        self.default_pd_config = pd_data
        self.default_config = data
        logger.debug(f"config bak path {self.config.config_single_bak_path!r}", )
        if self.config.config_single_bak_path.exists():
            self.config.config_single_bak_path.unlink()
        with open_s(self.config.config_single_bak_path, "w") as fout:
            json.dump(self.default_config, fout, indent=4)
        self.run_log = None
        self.mindie_log_offset = 0
        self.mindie_log_fp = None
        self.process = None
        self.command = self.KubectlCommand(self.config.command).command
        self.log_command = self.KubectlCommand(self.config.command).log_command
        self.monitor_command = self.KubectlCommand(self.config.command).monitor_command

    @property
    def base_url(self) -> str:
        """
        获取服务的base url 属性
        Returns:

        """
        pass

    @staticmethod
    def set_config_for_dict(context: ConfigContextdict):
        if context.cur_key in context.origin_config:
            DisaggregationSimulator.set_config(context.origin_config[context.cur_key], context.next_level, 
                                               context.value, context.current_depth)
        elif DisaggregationSimulator.is_int(context.cur_key):
            raise KeyError(f"data: {context.origin_config}, key: {context.cur_key}")
        elif DisaggregationSimulator.is_int(context.next_key):
            context.origin_config[context.cur_key] = []
            DisaggregationSimulator.set_config(context.origin_config[context.cur_key], context.next_level, 
                                               context.value, context.current_depth)
        else:
            context.origin_config[context.cur_key] = {}
            DisaggregationSimulator.set_config(context.origin_config[context.cur_key], context.next_level, 
                                               context.value, context.current_depth)

    @staticmethod
    def set_config_for_list(context: ConfigContextlist):
        if len(context.origin_config) > int(context.cur_key):
            DisaggregationSimulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, 
                                               context.value, context.current_depth)
        elif len(context.origin_config) == int(context.cur_key) and DisaggregationSimulator.is_int(context.next_key):
            context.origin_config.append([])
            DisaggregationSimulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, 
                                               context.value, context.current_depth)
        elif len(context.origin_config) == int(context.cur_key) and not \
            DisaggregationSimulator.is_int(context.next_key):
            context.origin_config.append({})
            DisaggregationSimulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, 
                                               context.value, context.current_depth)
        else:
            raise IndexError(f"data: {context.origin_config}, index: {context.cur_key}")

    @staticmethod
    def set_config(origin_config, key: str, value: Any, current_depth=0):
        if current_depth > 10:
            raise RecursionError("Exceeded maximum recursion depth")
        next_level = None
        try:
            if "." in key:
                _f_index = key.index(".")
                _cur_key, next_level = key[:_f_index], key[_f_index + 1:]
            else:
                _cur_key = key
            if next_level is None:
                if isinstance(origin_config, dict):
                    origin_config[_cur_key] = value
                elif isinstance(origin_config, list):
                    if len(origin_config) > int(_cur_key):
                        origin_config[int(_cur_key)] = value
                    else:
                        origin_config.append(value)
                return
            if "." in next_level:
                _next_index = next_level.index(".")
                _next_key = next_level[:_next_index]
            elif next_level:
                _next_key = next_level
            else:
                _next_key = None
        except Exception as e:
            logger.error(f"Unexpected error occurred at {key}")
            raise e
        if isinstance(origin_config, dict):
            context = ConfigContextdict(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1
            )
            DisaggregationSimulator.set_config_for_dict(context)
        elif isinstance(origin_config, list):
            context = ConfigContextlist(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1
            )
            DisaggregationSimulator.set_config_for_list(context)
        else:
            raise ValueError(f"Not Support type {type(origin_config)}")
    
    @staticmethod
    def is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False
        
    def prepare_before_start_server(self):
        bash_path = shutil.which("bash")
        if bash_path is not None:
            subprocess.run([bash_path, self.config.delete_path, "mindie", "."], 
                           cwd=self.config.kubectl_default_path)
            while True:
                signal = True
                proc = subprocess.run(self.log_command, stdout=subprocess.PIPE, text=True, 
                                      cwd=self.config.kubectl_default_path)
                lines = proc.stdout.splitlines()
                for line in lines:
                    if line.startswith('mindie'):
                        signal = False
                if signal is True:
                    break
                time.sleep(1)
        else:
            logger.error("bash not found in path")

    def backup(self):
        super().backup()
        backup(self.config.config_path, self.bak_path, self.__class__.__name__)

    def update_config(self, params: Tuple[OptimizerConfigField]):
        # 将params值更新到新的config中
        new_config = deepcopy(self.default_config)
        pd_config = deepcopy(self.default_pd_config)
        for p in params:
            if p.config_position.startswith("default"):
                DisaggregationSimulator.set_config(pd_config, p.config_position, p.value)
            if not p.config_position.startswith("BackendConfig"):
                continue
            DisaggregationSimulator.set_config(new_config, p.config_position, p.value)

        # 将新的config写入到config文件中
        logger.debug(f"new config {new_config}")
        if self.config.config_single_path.exists():
            self.config.config_single_path.unlink()
        with open_s(self.config.config_single_path, "w") as fout:
            json.dump(new_config, fout, indent=4)
        if self.config.config_single_pd_path.exists():
            self.config.config_single_pd_path.unlink()
        with open_s(self.config.config_single_pd_path, "w") as fout:
            json.dump(pd_config, fout, indent=4)
        
    def update_command(self):
        self.command = self.KubectlCommand(self.config.command).command
        self.log_command = self.KubectlCommand(self.config.command).log_command
        self.monitor_command = self.KubectlCommand(self.config.command).monitor_command
        
    def test_curl(self):
        import requests
        logger.info(f"kubectl_single_path: {self.config.kubectl_single_path}")
        curl_port = None
        yaml_dir = self.config.kubectl_single_path.parent
        yaml_path = os.path.join(yaml_dir, "deployment/mindie_service_single_container.yaml")
        logger.info(f"yaml_path: {yaml_path}")
        with open_s(yaml_path, 'r') as file:
            all_documents = yaml.safe_load_all(file)
            for doc in all_documents:
                # 检查文档中是否存在 spec.ports 部分
                if 'spec' in doc and 'ports' in doc['spec']:
                    ports = doc['spec']['ports']
                    for port in ports:
                        if 'nodePort' in port:
                            curl_port = port['nodePort']
        if curl_port:
            url = f"http://127.0.0.1:{curl_port}"
        else:
            raise ("cannot find port from mindie_service_single_container.yaml, please check")

        # 定义请求体的 JSON 数据
        payload = {
            "inputs": "Please introduce yourself.",
            "parameters": {
                "max_new_tokens": 20,
                "temperature": 0.3,
                "top_p": 0.3,
                "top_k": 5,
                "do_sample": True,
                "repetition_penalty": 1.05,
                "seed": 128
            }
        }

        # 定义请求头（如果需要）
        headers = {
            "Content-Type": "application/json"
        }

        try:
            # 发送 POST 请求
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            # 检查响应状态码
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.RequestException as e:
            return False

    def health(self):
        process_res = ProcessState()
        process_res.stage = Stage.error
        with open_s(self.run_log, "r") as f:
            try:
                f.seek(self.mindie_log_offset)
                output = f.read()
                self.mindie_log_offset = f.tell()
            except Exception as e:
                logger.warning(f"Failed in read mindie log. error: {e}")
        if output:
            logger.debug(f"simulate out: \n{output}")
            if "MindIE-MS coordinator is ready!!!" in output:
                while True:
                    if self.test_curl() is True:
                        process_res.stage = Stage.running
                        return process_res
                    time.sleep(1)
        return process_res

    def start_server(self, run_params: Tuple[OptimizerConfigField]):
        super().before_run(run_params)
        self.prepare_before_start_server()
        self.mindie_log_fp, self.run_log = tempfile.mkstemp(prefix="modelevalstate_mindie")
        self.mindie_log_offset = 0
        if self.config.work_path:
            cwd = self.config.work_path
        else:
            cwd = os.getcwd()
        logger.info(f"start running the command: {self.command}")
        self.process = subprocess.run(self.command, env=self.env, text=True, 
                                      cwd=self.config.kubectl_default_path)
        logger.info(f"self.log_command: {self.log_command}")
        while True:
            proc = subprocess.run(self.log_command, stdout=subprocess.PIPE, text=True, 
                                cwd=self.config.kubectl_default_path)
            
            lines = proc.stdout.splitlines()
            for line in lines:
                if line.startswith('mindie'):
                    mindie_name = line.split()
                    break
            if mindie_name[3] == 'Running':
                break
            time.sleep(1)
        kubectl_monitor_command = self.KubectlCommand(self.config.command).monitor_command
        kubectl_monitor_command.append(mindie_name[1])
        logger.debug(f"mindie_name: {mindie_name[1]}")
        self.log_process = subprocess.Popen(kubectl_monitor_command, stdout=self.mindie_log_fp, 
                                            stderr=subprocess.STDOUT, env=self.env, text=True, 
                                            cwd=self.config.kubectl_default_path)
        logger.info(f"Start running the command: {' '.join(kubectl_monitor_command)}, log file: {self.run_log}")

    def run(self, run_params: Tuple[OptimizerConfigField]):
        logger.info(f'Start running in simulator. params are: {run_params}')
        # 根据params 修改配置文件
        self.update_config(run_params)
        # 启动mindie仿真
        self.start_server(run_params)

    def stop(self, del_log=True):
        logger.debug("Stop simulator process")
        close_file_fp(self.mindie_log_fp)
        if del_log:
            remove_file(self.run_log)
        self.mindie_log_offset = 0
        try:
            bash_path = shutil.which("bash")
            if bash_path is not None:
                subprocess.run([bash_path, self.config.delete_path, "mindie", "./"])
            else:
                logger.error("bash not found in path")
        except Exception as e:
            logger.error(f"Failed to stop simulator process. {e}")
