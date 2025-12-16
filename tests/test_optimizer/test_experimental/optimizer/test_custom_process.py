# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from experimental.config.config import (
    CUSTOM_OUTPUT,
    MODEL_EVAL_STATE_CONFIG_PATH,
    OptimizerConfigField
)
from experimental.optimizer.custom_process import CustomProcess, tempfile, os


def test_before_run_no_run_params(monkeypatch):
    # 模拟 tempfile.mkstemp
    monkeypatch.setattr(tempfile, "mkstemp", lambda prefix="": (1234, 'tempfile'))
    # 模拟 os.environ
    monkeypatch.setattr(os, "environ", {})
    process = CustomProcess()
    process.before_run()

    # 验证属性设置
    assert process.run_log_fp == 1234
    assert process.run_log == 'tempfile'
    assert process.run_log_offset == 0


def test_before_run_with_run_params():
    process = CustomProcess()
    process.command = ["benchmark", "$CONCURRENCY", "$REQUESTRATE"]
    run_params = (
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1000, dtype="int", value=10),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0.1, max=0.7, value=0.3, dtype="float"),
    )
    process.before_run(run_params)
    assert process.command == ["benchmark", "10", "0.3"]


def test_before_run_env_var_already_set(monkeypatch):
    # 模拟 os.environ
    monkeypatch.setattr(os, "environ", {CUSTOM_OUTPUT: "/result",
                                        MODEL_EVAL_STATE_CONFIG_PATH: "config.toml"})

    process = CustomProcess()
    process.before_run()

    # 验证 tempfile.mkstemp 被调用
    assert os.environ[CUSTOM_OUTPUT] == "/result"
    assert os.environ[MODEL_EVAL_STATE_CONFIG_PATH] == "config.toml"


def test_check_success_process_still_running(tmpdir):
    # 模拟子进程仍在运行
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = None
    custom_process.print_log = True
    result = custom_process.check_success()

    assert result is False


def test_check_success_process_succeeded(tmpdir):
    # 模拟子进程成功完成
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = 0
    custom_process.print_log = True
    result = custom_process.check_success()

    assert result is True


def test_check_success_process_failed(tmpdir):
    # 模拟子进程失败
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = 1
    custom_process.print_log = True
    with pytest.raises(subprocess.SubprocessError) as e:
        custom_process.check_success()


@patch("psutil.process_iter")
@patch("experimental.optimizer.custom_process.kill_process")
def test_check_env_no_residual_process(mock_kill_process, mock_process_iter):
    # 模拟没有残留进程的情况
    mock_process_iter.return_value = [
        MagicMock(info={"pid": 1, "name": "not_process"}),
        MagicMock(info={"pid": 2, "name": "also_not_target"}),
        MagicMock()
    ]
 
    CustomProcess.kill_residual_process("target_process")
 
    # 确保kill_process没有被调用
    mock_process_iter.assert_called_once()
    mock_kill_process.assert_not_called()
 
 
@patch("psutil.process_iter")
@patch("experimental.optimizer.custom_process.kill_process")
def test_check_env_with_residual_process(mock_kill_process, mock_process_iter):
    # 模拟有残留进程的情况
    mock_process_iter.return_value = [
        MagicMock(info={"pid": 1, "name": "not_target_process"}),
        MagicMock(info={"pid": 2, "name": "target_process"}),
        MagicMock(info={"pid": 3, "name": "another_target_process"})
    ]
 
    CustomProcess.kill_residual_process("target_process,another_target_process")
 
    # 确保kill_process被调用
    mock_kill_process.assert_any_call("target_process")
    mock_kill_process.assert_any_call("another_target_process")
 
 
@patch("psutil.process_iter")
@patch("experimental.optimizer.custom_process.kill_process")
def test_check_env_kill_process_exception(mock_kill_process, mock_process_iter):
    # 模拟在尝试杀死进程时发生异常的情况
    mock_process_iter.return_value = [
        MagicMock(info={"pid": 1, "name": "target_process"})
    ]
    mock_kill_process.side_effect = Exception("Failed to kill process")
 
    CustomProcess.kill_residual_process("target_process")
 
    # 确保kill_process被调用，并且异常被捕获
    mock_kill_process.assert_called_once_with("target_process")
 
 
# 测试用例1：测试process_name存在且check_env成功的情况
@patch('experimental.optimizer.custom_process.CustomProcess.kill_residual_process')
@patch('experimental.optimizer.custom_process.CustomProcess.before_run')
@patch('experimental.optimizer.custom_process.subprocess.Popen')
def test_run_process_name_exists_and_check_env_success(mock_popen, mock_before_run, mock_check_env):
    process = CustomProcess()
    process.process_name = 'test_process'
    process.command = ['test_command']
    process.work_path = '/test/work/path'
    process.run_log_fp = MagicMock()
    process.run_log = '/test/run/log'
    process.run()
    mock_check_env.assert_called_once_with('test_process')
    mock_before_run.assert_called_once()
 
 
# 测试用例2：测试process_name存在但check_env失败的情况
@patch('experimental.optimizer.custom_process.CustomProcess.kill_residual_process')
@patch('experimental.optimizer.custom_process.CustomProcess.before_run')
@patch('experimental.optimizer.custom_process.subprocess.Popen')
def test_run_process_name_exists_and_check_env_fail(mock_popen, mock_before_run, mock_check_env):
    process = CustomProcess()
    process.process_name = 'test_process'
    process.command = ['test_command']
    process.work_path = '/test/work/path'
    process.run_log_fp = MagicMock()
    process.run_log = '/test/run/log'
    mock_check_env.side_effect = Exception('kill_residual_process failed')
    process.run()
    mock_check_env.assert_called_once_with('test_process')
    mock_before_run.assert_called_once()
    mock_popen.assert_called_once()
 
 
# 测试用例3：测试process_name不存在的情况
@patch('experimental.optimizer.custom_process.CustomProcess.before_run')
@patch('experimental.optimizer.custom_process.subprocess.Popen')
def test_run_process_name_not_exists(mock_popen, mock_before_run):
    process = CustomProcess()
    process.process_name = None
    process.command = ['test_command']
    process.work_path = '/test/work/path'
    process.run_log_fp = MagicMock()
    process.run_log = '/test/run/log'
    process.run()
    mock_before_run.assert_called_once()
 
 
# 测试用例4：测试subprocess.Popen抛出OSError的情况
@patch('experimental.optimizer.custom_process.CustomProcess.before_run')
@patch('experimental.optimizer.custom_process.subprocess.Popen')
def test_run_subprocess_popen_os_error(mock_popen, mock_before_run):
    process = CustomProcess()
    process.process_name = None
    process.command = ['test_command']
    process.work_path = '/test/work/path'
    process.run_log_fp = MagicMock()
    process.run_log = '/test/run/log'
    mock_popen.side_effect = OSError('subprocess.Popen failed')
    with pytest.raises(OSError) as e:
        process.run()
    assert str(e.value) == 'subprocess.Popen failed'
    mock_before_run.assert_called_once()
 
 
# 测试用例1：测试run_log为None的情况
def test_get_log_run_log_none():
    process = CustomProcess()
    process.run_log = None
    assert process.get_log() is None
 
 
# 测试用例2：测试run_log文件不存在的情况
@patch('pathlib.Path.exists', return_value=False)
def test_get_log_run_log_not_exists(mock_exists):
    process = CustomProcess()
    process.run_log = 'nonexistent.log'
    assert process.get_log() is None
 
