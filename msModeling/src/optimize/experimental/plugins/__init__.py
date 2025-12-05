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
from typing import Callable, Dict

from loguru import logger

# make sure one process only loads plugins once
plugins_loaded = False


def load_plugins_by_group(group: str) -> Dict[str, Callable]:
    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    discovered_plugins = entry_points(group=group)
    if len(discovered_plugins) == 0:
        logger.debug("No plugins for group {} found.", group)
        return {}
    logger.info("Available plugins for group {}:", group)
    for plugin in discovered_plugins:
        logger.info("name={}, value={}", plugin.name, plugin.value)
    plugins = {}
    for plugin in discovered_plugins:
        try:
            func = plugin.load()
            plugins[plugin.name] = func
            logger.info("plugin {} loaded.", plugin.name)
        except Exception:
            logger.exception("Failed to load plugin {}", plugin.name)
    return plugins


def load_general_plugins():
    global plugins_loaded
    if plugins_loaded:
        return None
    plugins_loaded = True

    plugins = load_plugins_by_group(group='modelevalstate.plugins')
    # general plugins, we only need to execute the loaded functions
    for func in plugins.values():
        func()
    return plugins