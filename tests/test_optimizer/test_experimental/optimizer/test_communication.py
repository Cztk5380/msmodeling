# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

from pathlib import Path
from unittest.mock import MagicMock
import unittest

import pytest

from src.optimize.experimental.optimizer.communication import CommunicationForFile, CustomCommand


def test_init(tmpdir):
    # 创建临时文件路径
    work_dir = Path(tmpdir)
    cmd_file = work_dir.joinpath("parent", "cmd.txt")
    res_file = work_dir.joinpath("parent", "res.txt")

    # 测试当父目录不存在时，是否能正确创建
    assert not cmd_file.parent.exists()
    assert not res_file.parent.exists()
    comm = CommunicationForFile(cmd_file, res_file)
    assert cmd_file.parent.exists()
    assert res_file.parent.exists()
    assert comm.cmd_file_lock.exists()
    assert comm.res_file_lock.exists()

    # 测试当锁文件存在时，是否不会重新创建
    old_cmd_lock_mtime = comm.cmd_file_lock.stat().st_mtime
    old_res_lock_mtime = comm.res_file_lock.stat().st_mtime
    comm = CommunicationForFile(cmd_file, res_file)
    assert comm.cmd_file_lock.stat().st_mtime == old_cmd_lock_mtime
    assert comm.res_file_lock.stat().st_mtime == old_res_lock_mtime

    # 测试timeout参数是否正确设置
    comm = CommunicationForFile(cmd_file, res_file, timeout=30)
    assert comm.timeout == 30


class TestCommunicationForFile:
    @classmethod
    def test_send_command_file_exists(cls, comm):
        comm.send_command("new command")
        with open(comm.cmd_file, 'r') as f:
            assert f.read() == "new command"
        comm.send_command("two command")
        with open(comm.cmd_file, 'r') as f:
            assert f.read() == "two command"

    @classmethod
    def test_recv_command_file_exists(cls, comm):
        assert not comm.res_file.exists()
        assert comm.recv_command() == ''
        with open(comm.res_file, 'w', encoding="utf-8") as f:
            f.write("test data")
        assert comm.recv_command() == "test data"
        with open(comm.res_file, 'w', encoding="utf-8") as f:
            f.write("two data")
        assert comm.recv_command() == "two data"

    @classmethod
    def test_clear_cmd_done(cls, comm):
        _cmd = "init 11111"
        comm.recv_command = MagicMock(return_value="init 11111:done")
        comm.send_command = MagicMock()
        comm.clear_res = MagicMock()
        assert comm.clear_cmd(_cmd) == "done"
        comm.send_command.assert_called_once_with(CustomCommand.cmd_eof)
        comm.clear_res.assert_called_once()

    @classmethod
    def test_clear_cmd_true(cls, comm):
        _cmd = "init 11111"
        comm.recv_command = MagicMock(return_value="init 11111:true")
        comm.send_command = MagicMock()
        comm.clear_res = MagicMock()
        assert comm.clear_cmd(_cmd) is True
        comm.send_command.assert_called_once_with(CustomCommand.cmd_eof)
        comm.clear_res.assert_called_once()

    @classmethod
    def test_clear_cmd_false(cls, comm):
        _cmd = "init 11111"
        comm.recv_command = MagicMock(return_value="init 11111:false")
        comm.send_command = MagicMock()
        comm.clear_res = MagicMock()
        assert comm.clear_cmd(_cmd) is False
        comm.send_command.assert_called_once_with(CustomCommand.cmd_eof)
        comm.clear_res.assert_called_once()

    @classmethod
    def test_clear_cmd_none(cls, comm):
        _cmd = "init 11111"
        comm.recv_command = MagicMock(return_value="init 11111:none")
        comm.send_command = MagicMock()
        comm.clear_res = MagicMock()
        assert comm.clear_cmd(_cmd) is None
        comm.send_command.assert_called_once_with(CustomCommand.cmd_eof)
        comm.clear_res.assert_called_once()

    @classmethod
    def test_clear_cmd_error(cls, comm):
        _cmd = "init 11111"
        comm.recv_command = MagicMock(return_value="init 11111:error")
        comm.send_command = MagicMock()
        comm.clear_res = MagicMock()
        with pytest.raises(ValueError):
            comm.clear_cmd(_cmd)

    @classmethod
    def test_clear_cmd_other(cls, comm):
        comm.recv_command = MagicMock(return_value="init 111111:other")
        comm.send_command = MagicMock()
        comm.clear_res = MagicMock()
        assert comm.clear_cmd("init 111111") == "other"
        comm.send_command.assert_called_once_with(CustomCommand.cmd_eof)
        comm.clear_res.assert_called_once()

    @classmethod
    def test_clear_res(cls, comm):
        comm.recv_command = MagicMock()
        comm.send_command = MagicMock()

        # 测试当recv_command返回cmd_eof时，clear_res会发送cmd_eof并退出循环
        comm.recv_command.return_value = CustomCommand.cmd_eof
        comm.clear_res()
        comm.send_command.assert_called_once_with(CustomCommand.cmd_eof)

        # 测试当recv_command返回非cmd_eof时，clear_res会继续循环
        comm.recv_command.reset_mock()
        comm.send_command.reset_mock()
        comm.recv_command.side_effect = ['not_eof', CustomCommand.cmd_eof]
        comm.clear_res()
        assert comm.send_command.call_count == 1
        comm.send_command.assert_called_once_with(CustomCommand.cmd_eof)

    @pytest.fixture(autouse=True)
    def comm(self, tmpdir):
        work_dir = Path(tmpdir)
        cmd_file = work_dir.joinpath("cmd.txt")
        res_file = work_dir.joinpath("res.txt")
        comm = CommunicationForFile(cmd_file, res_file)
        return comm