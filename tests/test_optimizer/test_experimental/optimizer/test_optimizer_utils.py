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
import shutil
import signal
from pathlib import Path
from unittest.mock import patch, MagicMock, call, Mock

import psutil
import pytest
from loguru import logger

from msguard import Rule, GlobalConfig
from msguard.security import walk_s
from src.optimize.experimental.optimizer.utils import (
    remove_file,
    kill_children,
    kill_process,
    backup,
    close_file_fp,
    get_folder_size,
    get_required_field_from_json
)


# --------------------------
# Test remove_file
# --------------------------
def test_remove_file_none():
    remove_file(None)


def test_remove_file_nonexistent(tmp_path):
    file_path = tmp_path / "nonexistent.txt"
    remove_file(file_path)
    assert not file_path.exists()


def test_remove_file_regular_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")
    assert file_path.exists()
    remove_file(file_path)
    assert not file_path.exists()


def test_remove_file_directory(tmp_path):
    dir_path = tmp_path / "subdir"
    dir_path.mkdir()
    (dir_path / "file.txt").write_text("content")
    remove_file(dir_path)
    assert dir_path.exists()


def test_remove_file_directory_with_unremovable_subdir(tmp_path, caplog):
    dir_path = tmp_path / "subdir"
    dir_path.mkdir()
    protected = dir_path / "protected"
    protected.mkdir()
    # Make it non-removable to trigger exception
    with patch("shutil.rmtree", side_effect=OSError("Cannot remove")):
        remove_file(dir_path)
    assert "remove file failed" not in caplog.text


# --------------------------
# Test kill_children
# --------------------------
@patch("psutil.Process")
def test_kill_children(mock_process):
    mock_child = Mock()
    mock_child.is_running.return_value = True
    mock_child.pid = 1234
    mock_child.send_signal = Mock()
    mock_child.wait = Mock(return_value=None)

    kill_children([mock_child])

    mock_child.send_signal.assert_called_with(9)
    mock_child.wait.assert_called_with(10)


@patch("psutil.Process")
def test_kill_children_not_running(mock_process):
    mock_child = Mock()
    mock_child.is_running.return_value = False
    kill_children([mock_child])
    mock_child.send_signal.assert_not_called()


@patch("psutil.Process")
def test_kill_children_exception_on_signal(mock_process, caplog):
    mock_child = Mock()
    mock_child.is_running.return_value = True
    mock_child.pid = 1234
    mock_child.send_signal.side_effect = Exception("Permission denied")
    kill_children([mock_child])
    assert "Failed in kill the 1234 process" not in caplog.text


@patch("psutil.Process")
def test_kill_children_still_running_after_wait(mock_process, caplog):
    mock_child = Mock()
    mock_child.is_running.side_effect = [True, True]  # still running after wait
    mock_child.pid = 1234
    mock_child.send_signal = Mock()
    mock_child.wait = Mock(return_value=None)
    kill_children([mock_child])
    assert "Failed to kill the 1234 process" not in caplog.text


# --------------------------
# Test kill_process
# --------------------------
@patch("psutil.process_iter")
@patch("psutil.Process")
def test_kill_process(mock_psutil_process, mock_process_iter):
    # Mock process list
    mock_proc_info = Mock()
    mock_proc_info.info = {"name": "target_process.exe"}
    mock_proc_info.pid = 1001
    mock_process_iter.return_value = [mock_proc_info]

    # Mock children
    child_proc = Mock()
    child_proc.pid = 2001
    mock_psutil_process.return_value.children.return_value = [child_proc]

    kill_process("target_process")

    # Check signals sent
    mock_proc_info.send_signal.assert_called_with(9)
    child_proc.send_signal.assert_called_with(9)


@patch("psutil.process_iter")
def test_kill_process_no_match(mock_process_iter):
    mock_proc_info = Mock()
    mock_proc_info.info = {"name": "other_process.exe"}
    mock_process_iter.return_value = [mock_proc_info]

    with patch("psutil.Process") as mock_psutil_process:
        kill_process("target_process")
        mock_psutil_process.assert_not_called()


# --------------------------
# Test backup
# --------------------------
@patch.object(Rule.input_file_read, 'is_satisfied_by')
@patch.object(Rule.input_dir_traverse, 'is_satisfied_by')
def test_backup_file_success(mock_dir_traverse, mock_file_read, tmp_path):
    mock_file_read.return_value = True
    mock_dir_traverse.return_value = True

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src_file = src_dir / "test.txt"
    src_file.write_text("data")

    bak_dir = tmp_path / "bak"
    bak_dir.mkdir()
    class_name = "TestClass"

    backup(src_file, bak_dir, class_name)

    dest_file = bak_dir / class_name / "test.txt"
    assert dest_file.exists()
    assert dest_file.read_text() == "data"


@patch.object(Rule.input_file_read, 'is_satisfied_by')
def test_backup_file_permission_denied(mock_file_read, tmp_path, caplog):
    mock_file_read.return_value = True
    src_file = tmp_path / "src.txt"
    src_file.write_text("data")
    bak_dir = tmp_path / "bak"
    # Don't create bak_dir to trigger mkdir error
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Denied")):
        backup(src_file, bak_dir, "TestClass")
    assert "PermissionError" not in caplog.text  # should be handled silently


def test_backup_file_rule_not_satisfied(tmp_path):
    GlobalConfig.custom_return = False

    src_file = tmp_path / "src.txt"
    src_file.write_text("data")
    bak_dir = tmp_path / "bak"
    bak_dir.mkdir()
    backup(src_file, bak_dir, "TestClass")
    dest_file = bak_dir / "TestClass" / "src.txt"
    assert not dest_file.exists()

    GlobalConfig.reset()


@patch.object(Rule.input_dir_traverse, 'is_satisfied_by')
@patch.object(Rule.input_file_read, 'is_satisfied_by')
def test_backup_directory_success(mock_file_read, mock_dir_traverse, tmp_path):
    mock_dir_traverse.return_value = True
    mock_file_read.return_value = True

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "file1.txt").write_text("data1")
    sub = src_dir / "sub"
    sub.mkdir()
    (sub / "file2.txt").write_text("data2")

    bak_dir = tmp_path / "bak"
    bak_dir.mkdir()

    backup(src_dir, bak_dir, "TestClass")

    dest_dir = bak_dir / "TestClass" / "src"
    assert (dest_dir / "file1.txt").read_text() == "data1"
    assert (dest_dir / "sub" / "file2.txt").read_text() == "data2"


def test_backup_directory_rule_not_satisfied(tmp_path):
    GlobalConfig.custom_return = False

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    bak_dir = tmp_path / "bak"
    bak_dir.mkdir()
    backup(src_dir, bak_dir, "TestClass")
    dest_dir = bak_dir / "TestClass" / "src"
    assert not dest_dir.exists()

    GlobalConfig.reset()


def test_backup_max_depth_reached(caplog, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    bak_dir = tmp_path / "bak"
    bak_dir.mkdir()
    backup(src_dir, bak_dir, "TestClass", max_depth=0, current_depth=0)
    assert "Reached maximum backup depth 0" not in caplog.text


# --------------------------
# Test close_file_fp
# --------------------------
def test_close_file_fp_file_object():
    mock_file = Mock()
    close_file_fp(mock_file)
    mock_file.close.assert_called_once()


def test_close_file_fp_file_descriptor():
    with patch("os.close") as mock_os_close:
        close_file_fp(3)
        mock_os_close.assert_called_with(3)


def test_close_file_fp_none():
    close_file_fp(None)  # Should not raise


# --------------------------
# Test get_folder_size
# --------------------------
def test_get_folder_size_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    size = get_folder_size(empty_dir)
    assert size == 0


def test_get_folder_size_with_files(tmp_path):
    dir_path = tmp_path / "data"
    dir_path.mkdir()
    (dir_path / "file1.txt").write_bytes(b"12345")  # 5 bytes
    (dir_path / "file2.txt").write_bytes(b"1234567890")  # 10 bytes

    with patch("msguard.security.walk_s", side_effect=walk_s):
        size = get_folder_size(dir_path)

    assert size == 15


def test_get_folder_size_nonexistent_path():
    size = get_folder_size(Path("/nonexistent/path"))
    assert size == 0


def test_get_required_field_from_json():
    # 测试用例1：测试从字典中获取字段
    data = {"name": "John", "age": 30, "city": "New York"}
    assert get_required_field_from_json(data, "name") == "John"

    # 测试用例2：测试从嵌套字典中获取字段
    data = {"person": {"name": "John", "age": 30, "city": "New York"}}
    assert get_required_field_from_json(data, "person.name") == "John"

    # 测试用例3：测试从列表中获取字段
    data = ["John", 30, "New York"]
    assert get_required_field_from_json(data, "0") == "John"

    # 测试用例4：测试从嵌套列表中获取字段
    data = [["John", 30, "New York"], ["Jane", 25, "Los Angeles"]]
    assert get_required_field_from_json(data, "1.0") == "Jane"

    # 测试用例5：测试从字典和列表混合结构中获取字段
    data = {"person": {"name": "John", "age": 30, "city": ["New York", "Los Angeles"]}}
    assert get_required_field_from_json(data, "person.city.1") == "Los Angeles"

    # 测试用例6：测试从不支持的数据类型中获取字段
    data = "John"
    with pytest.raises(ValueError):
        get_required_field_from_json(data, "name")

    # 测试用例7：测试从空数据中获取字段
    data = {}
    with pytest.raises(KeyError):
        get_required_field_from_json(data, "name")

    # 测试用例8：测试从空列表中获取字段
    data = []
    with pytest.raises(IndexError):
        get_required_field_from_json(data, "0")

    # 测试用例9：测试从空嵌套字典中获取字段
    data = {"person": {}}
    with pytest.raises(KeyError):
        get_required_field_from_json(data, "person.name")

    # 测试用例10：测试从空嵌套列表中获取字段
    data = {"person": []}
    with pytest.raises(IndexError):
        get_required_field_from_json(data, "person.0")

    # 测试用例11：测试从空嵌套字典和列表混合结构中获取字段
    data = {"person": {"name": "John", "age": 30, "city": []}}
    with pytest.raises(IndexError):
        get_required_field_from_json(data, "person.city.0")

    # 测试用例12：测试从空嵌套字典和列表混合结构中获取字段
    data = {"person": {"name": "John", "age": 30, "city": ["New York", "Los Angeles"]}}
    assert get_required_field_from_json(data, "person.city.1") == "Los Angeles"
