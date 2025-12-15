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
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from experimental.optimizer.register import simulates
from experimental.config.config import get_settings, map_param_with_value, CommunicationConfig, Stage, \
    OptimizerConfigField
from experimental.optimizer.communication import CommunicationForFile, CustomCommand


class Scheduler:
    def __init__(self, engine: str, communication_config: CommunicationConfig):
        self.engine = engine
        self.simulator_class = simulates[engine]
        self.simulator = None
        self.communication_config = communication_config
        self.communication = CommunicationForFile(self.communication_config.res_file,
                                                  self.communication_config.cmd_file, )
        self.cmd = CustomCommand()
 
    def backup(self, params):
        back_path = Path(params)
        if not back_path.exists():
            back_path.mkdir(parents=True, mode=0o750)
        _result = f"{self.cmd.history[-1]}:done"
        self.communication.send_command(_result)
        self.communication.clear_res()
 
    def start(self, params):
        d = json.loads(params)
        self.simulator = self.simulator_class()
        _params = np.array(d["params"])
        _params_field = tuple([OptimizerConfigField(**_item) for _item in d["params_field"]])
        # 注意因为当前设计是simulator 参数在调优参数的前一部分，所以下面的才会对应上。
        _simulate_run_info = map_param_with_value(_params, _params_field)
        logger.debug(f"simulate run info {_simulate_run_info}")
        self.simulator.run(tuple(_simulate_run_info))
        _result = f"{self.cmd.history[-1]}:done"
        self.communication.send_command(_result)
        self.communication.clear_res()
 
    def check_success(self):
        if not self.simulator:
            return
        flag = False
        for _ in range(30):
            res = self.simulator.health()
            if res.stage == Stage.running:
                flag = True
                break
 
            time.sleep(1)
        _result = f"{self.cmd.history[-1]}:{flag}"
        self.communication.send_command(_result)
        self.communication.clear_res()
 
    def stop(self, params):
        del_log = ast.literal_eval(params)
        if self.simulator:
            self.simulator.stop(del_log)
            _result = f"{self.cmd.history[-1]}:done"
            self.communication.send_command(_result)
            self.communication.clear_res()
 
    def get_cmd_param(self):
        cmd = self.communication.recv_command()
        if not cmd or cmd.strip().lower() == self.cmd.cmd_eof:
            return None, None
        # 已经接收过的命令，不再处理。
        if cmd in self.cmd.history:
            return None, None
        _cmd_list = cmd.split()
        if len(_cmd_list) < 2:
            logger.error("Format does not match.")
            return None, None
        _cmd = _cmd_list[0]
        if not _cmd:
            return None, None
        _param = cmd[cmd.find("params:") + 7:] if "params:" in cmd else None
        self.cmd.history = cmd
        return _cmd, _param
 
    def process_poll(self):
        flag = None
        if self.simulator:
            res = self.simulator.health()
            if res.stage == Stage.running:
                flag = None
            elif res.stage == Stage.error:
                flag = f"error. info {res.info}"
            else:
                flag = f"stage: {res.stage}, info: {res.info}"
        _result = f"{self.cmd.history[-1]}:{flag}"
        self.communication.send_command(_result)
        self.communication.clear_res()
 
    def init(self):
        _cmd, _param = self.get_cmd_param()
        logger.debug("params {}", _param)
        if not _cmd:
            return False
        logger.info("cmd {}", _cmd)
        if _cmd.strip().lower() == "init":
            _result = f"{self.cmd.history[-1]}:done"
            self.communication.send_command(_result)
            self.communication.clear_res()
            return True
        return False
 
    def run(self):
        _cmd, _param = self.get_cmd_param()
 
        if not _cmd:
            return ''
        if not hasattr(self, _cmd):
            logger.error("Unknown command found, {}.", _cmd)
            return ''
        logger.debug("cmd {}, params {}", _cmd, _param)
        if _param:
            res = getattr(self, _cmd)(_param)
        else:
            res = getattr(self, _cmd)()
        return res
 
 
def main(engine: str):
    # 创建服务器
    schduler = Scheduler(engine, get_settings().communication)
    _init_flag = True
    for _ in range(60):
        if schduler.init():
            _init_flag = False
            break
        time.sleep(1)
    if _init_flag:
        raise ValueError("Verification of communication failed within 60 seconds ")
    while True:
        # 运行服务器的主循环
        try:
            # 接收命令执行命令
            schduler.run()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, exiting.")
            break