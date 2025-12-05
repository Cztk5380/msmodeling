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
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Tuple, Optional, Union
import shutil
import tempfile
import time
from loguru import logger
import yaml
from msserviceprofiler.modelevalstate.config.config import MindieConfig, VllmConfig, OptimizerConfigField, KubectlConfig
from msserviceprofiler.modelevalstate.config.base_config import simulate_flag, SIMULATE
from msserviceprofiler.modelevalstate.config.custom_command import MindieCommand, VllmCommand
from msserviceprofiler.modelevalstate.optimizer.custom_process import CustomProcess
from msserviceprofiler.modelevalstate.optimizer.utils import backup, remove_file, close_file_fp
from msserviceprofiler.msguard.security import open_s
from msserviceprofiler.modelevalstate.optimizer.plugins.simulate import Simulator


@contextmanager
def enable_simulate_old(simulate):
    if simulate_flag and isinstance(simulate, Simulator):
        origin_data = simulate.default_config
        data = deepcopy(origin_data)
        simulate.default_config = data
        model_config = data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]
        if "plugin_params" in model_config:
            _plugin_params = json.loads(model_config["plugin_params"])
            if SIMULATE not in _plugin_params["plugin_type"]:
                _plugin_params["plugin_type"] += "," + SIMULATE
                model_config["plugin_params"] = json.dumps(_plugin_params)
        else:
            model_config["plugin_params"] = json.dumps({"plugin_type": SIMULATE})
        with open_s(simulate.config.config_path, 'w') as f:
            json.dump(data, f, indent=4)
        yield simulate_flag
        if simulate.config.config_path.exists():
            simulate.config.config_path.unlink()
        with open_s(simulate.config.config_path, 'w') as f:
            json.dump(origin_data, f, indent=4)
    else:
        yield simulate_flag
    return