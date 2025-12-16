# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
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