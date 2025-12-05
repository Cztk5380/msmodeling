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
import argparse
import ast
import csv
import json
import os
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import pandas as pd
from loguru import logger
from msserviceprofiler.msguard.security import open_s
from msserviceprofiler.msguard import Rule
from msserviceprofiler.modelevalstate.common import read_csv_s
from msserviceprofiler.modelevalstate.optimizer.utils import is_root


def fetch_rids_from_db(db_path):
    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 执行查询
        cursor.execute("SELECT * FROM batch WHERE name IN ('BatchSchedule', 'batchFrameworkProcessing');")
        batch_rows = cursor.fetchall()

        rids = []
        for row in batch_rows:
            if isinstance(row, str):
                data = ast.literal_eval(row)
                # 提取rid值
                rids.extend([item['rid'] for item in data])
            else:
                rids.append(row[1])
        return rids
    except sqlite3.Error as e:
        logger.error(f"数据库错误: {e}")
        return []
    except Exception as e:
        logger.error(f"其他错误: {e}")
        return []
    finally:
        # 确保关闭游标和连接
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


class DatabaseConnector:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """连接到数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            return self.cursor
        except sqlite3.Error as e:
            raise ConnectionError(f"无法连接到数据库: {e}") from e

    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def read_batch_exec_data(cursor) -> pd.DataFrame:
    """读取 batch_exec 表中的数据，并返回包含列名的 DataFrame"""
    try:
        cursor.execute("SELECT * FROM batch_exec WHERE event = 'forward';")
        data = cursor.fetchall()
        # 获取列名
        columns = [col[0] for col in cursor.description]
        # 创建 DataFrame
        df = pd.DataFrame(data, columns=columns)
        return df
    except sqlite3.Error as e:
        raise ValueError(f"读取 batch_exec 表时出错: {e}") from e


def group_exec_data_by_pid(exec_rows: List[Tuple]) -> Dict[int, List[Tuple]]:
    """按 pid 分组 batch_exec 数据"""
    data_by_pid = {}
    grouped = exec_rows.groupby('pid')
    for pid, group in grouped:
        data_by_pid[pid] = group
    return data_by_pid


def read_batch_data(cursor) -> List[Tuple]:
    """读取 batch 表中的数据"""
    try:
        cursor.execute("SELECT * FROM batch WHERE name IN ('BatchSchedule', 'batchFrameworkProcessing');")
        data = cursor.fetchall()
        # 获取列名
        columns = [col[0] for col in cursor.description]
        # 创建 DataFrame
        df = pd.DataFrame(data, columns=columns)
        return df
    except sqlite3.Error as e:
        raise ValueError(f"读取 batch 表时出错: {e}") from e


def read_batch_req_data(cursor) -> List[Tuple]:
    """读取 batch_req 表中的数据"""
    try:
        cursor.execute("SELECT * FROM batch_req;")
        data = cursor.fetchall()
        # 获取列名
        columns = [col[0] for col in cursor.description]
        # 创建 DataFrame
        df = pd.DataFrame(data, columns=columns)
        return df
    except sqlite3.Error as e:
        raise ValueError(f"读取 batch_req 表时出错: {e}") from e


def calculate_block_sums(req_df: pd.DataFrame) -> Dict[int, float]:
    """按 batch_id 分组并计算 block 的总和"""
    return req_df.groupby('batch_id')['block'].sum().to_dict()


def create_output_folder(input_path: str) -> str:
    """创建输出文件夹"""
    output_folder = os.path.join(input_path, 'output_csv')
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def process_batch_data(exec_data: List[Tuple], batch_rows: List[Tuple], current_batch_index: int) -> List[Tuple]:
    """处理批量数据"""
    num_exec_rows = len(exec_data)
    num_available_batch_rows = len(batch_rows) - current_batch_index

    if num_exec_rows > num_available_batch_rows:
        return batch_rows[current_batch_index:current_batch_index + num_available_batch_rows]
    else:
        return batch_rows[current_batch_index:current_batch_index + num_exec_rows]


def write_csv_header(csvfile) -> None:
    writer = csv.writer(csvfile)
    writer.writerow([
        ('batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len', 'model_execute_time'),
        ('input_length', 'need_blocks', 'output_length')
    ])


@dataclass
class ExecutionDataVllm:
    merged_df: List[Tuple]
    req_df: pd.DataFrame
    rids_ori: List[Any]
    kvcache_df: pd.DataFrame


@dataclass
class ExecutionDataMindie:
    merged_df: List[Tuple]
    req_df: pd.DataFrame
    rids_ori: List[Any]
    index_dict: Dict[Tuple, Any]
    batch_id_block_sum: Dict[int, float]


def process_execution_data_vllm(csv_data: ExecutionDataVllm) -> List[Tuple]:
    check_attrs = ["merged_df", "req_df", "rids_ori", "kvcache_df"]
    for attr in check_attrs:
        if getattr(csv_data, attr, None) is None:
            raise ValueError(f"{attr} cannot be None")
    processed_data = []
    for i, merged_row in csv_data.merged_df.iterrows():
        total_prefill_token = 0
        max_seq_len = 0
        total_req_info = []
        data = ast.literal_eval(csv_data.rids_ori[i])
        rids = []
        iters = []
        req_df = csv_data.req_df
        kvcache_df = csv_data.kvcache_df
        for item in data:
            try:
                rid = item['rid']
                rids.append(rid)
                iter_val = item['iter_size']
                iters.append(iter_val)
            except KeyError as e:
                logger.error(f"缺少键 '{e}'，跳过该条目")
        block_sum = 0
        for j, _ in enumerate(rids):
            recv_token = req_df[req_df['http_rid'] == rids[j]]['recv_token_size'].values[0]
            total_prefill_token += recv_token
            if recv_token > max_seq_len:
                max_seq_len = recv_token
            filtered_df = kvcache_df[(kvcache_df['rid'] == rids[j]) & (kvcache_df['name'].isin(['blocks', 'Allocate']))]
            target_row = filtered_df.iloc[iters[j] - 1]
            req_block = target_row['device_kvcache_left']
            block_sum += req_block
            req_info = (int(recv_token), req_block, iters[j])
            total_req_info.append(req_info)
        process_req_info = tuple(total_req_info)
        start = merged_row['start']  # forward开始时间
        end = merged_row['end']  # forward结束时间
        model_exec = end - start
        current_batch_id = i
        merged_row['model_exec'] = model_exec
        merged_row['block_sum'] = block_sum
        merged_row['total_prefill_token'] = total_prefill_token
        merged_row['max_seq_len'] = max_seq_len
        tuple_elements = process_row_data(merged_row)
        process_row = tuple([tuple_elements]) + tuple([process_req_info])
        processed_data.append(process_row)
    return processed_data


def process_row_data(combined_row):
    if combined_row['batch_type'] == 'Prefill':
        combined_row['batch_type'] = 'prefill'
    else:
        combined_row['batch_type'] = 'decode'
    tuple_elements = (
        combined_row['batch_type'],  # batch_type
        combined_row['batch_size'],  # batch_size
        combined_row['block_sum'],  # total_need_blocks
        int(combined_row['total_prefill_token']),  # total_prefill_token
        int(combined_row['max_seq_len']),  # max_seq_len
        combined_row['model_exec']  # forwar时长
    )
    return tuple_elements
    

def process_execution_data_mindie(csv_data: ExecutionDataMindie) -> List[Tuple]:
    check_attrs = ["merged_df", "req_df", "rids_ori", "index_dict", "batch_id_block_sum"]
    for attr in check_attrs:
        if getattr(csv_data, attr, None) is None:
            raise ValueError(f"{attr} cannot be None")
    processed_data = []
    for i, merged_row in csv_data.merged_df.iterrows():
        total_prefill_token = 0
        max_seq_len = 0
        total_req_info = []
        data = ast.literal_eval(csv_data.rids_ori[i])
        rids = []
        iters = []
        req_df = csv_data.req_df
        for item in data:
            try:
                rid = item['rid']
                rids.append(rid)
                iter_val = item['iter']
                iters.append(iter_val)
            except KeyError as e:
                logger.error(f"缺少键 '{e}'，跳过该条目")
        for j, _ in enumerate(rids):
            recv_token = req_df[req_df['http_rid'] == rids[j]]['recv_token_size'].values[0]
            total_prefill_token += recv_token
            if recv_token > max_seq_len:
                max_seq_len = recv_token
            req_block = csv_data.index_dict.get((rids[j], iters[j]), 0)
            req_info = (int(recv_token), req_block, iters[j])
            total_req_info.append(req_info)
        process_req_info = tuple(total_req_info)
        start = merged_row['start']  # forward开始时间
        end = merged_row['end']  # forward结束时间
        model_exec = end - start
        current_batch_id = i
        block_sum = csv_data.batch_id_block_sum.get(current_batch_id + 1, 0)
        merged_row['model_exec'] = model_exec
        merged_row['block_sum'] = block_sum
        merged_row['total_prefill_token'] = total_prefill_token
        merged_row['max_seq_len'] = max_seq_len
        tuple_elements = process_row_data(merged_row)
        process_row = tuple([tuple_elements]) + tuple([process_req_info])
        processed_data.append(process_row)
    return processed_data


def write_csv_row(csvfile, row: Tuple) -> None:
    writer = csv.writer(csvfile)
    writer.writerow(row)


@dataclass
class ProcessedData:
    input_path: str
    data_by_pid: Dict[int, List[Tuple]]
    batch_rows: List[Tuple]
    req_df: pd.DataFrame
    rids_ori: List[Any]


@dataclass
class ProcessedDataVllm(ProcessedData):
    kvcache_df: pd.DataFrame


@dataclass
class ProcessedDataMindie(ProcessedData):
    batch_id_block_sum: Dict[int, float]
    index_dict: Dict[Tuple, Any]


def save_processed_data_to_csv_vllm(
        processed_data: ProcessedDataVllm
) -> None:
    output_folder = create_output_folder(processed_data.input_path)
    batch_rows = processed_data.batch_rows
    for pid, exec_data in processed_data.data_by_pid.items():
        current_batch_index = 0
        batch_data = process_batch_data(exec_data, batch_rows, current_batch_index)

        parent_path = os.path.join(output_folder, f'pid_{pid}')
        os.makedirs(parent_path, exist_ok=True)
        file_path = os.path.join(parent_path, 'feature.csv')
        exec_data_reset = exec_data.reset_index(drop=True)
        batch_data_reset = batch_data.reset_index(drop=True)

        # 使用 merge 方法按行合并，并指定后缀处理重叠列名
        merged_df = exec_data_reset.merge(
            batch_data_reset,
            left_index=True,
            right_index=True,
            suffixes=('_exec', '_batch')
        )
        with open_s(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            write_csv_header(csvfile)
            an_data = ExecutionDataVllm(merged_df, processed_data.req_df, processed_data.rids_ori, 
                                    processed_data.kvcache_df)
            feature_data = process_execution_data_vllm(an_data)
            for row in feature_data:
                write_csv_row(csvfile, row)


def save_processed_data_to_csv_mindie(
        processed_data: ProcessedDataMindie
) -> None:
    output_folder = create_output_folder(processed_data.input_path)
    batch_rows = processed_data.batch_rows
    for pid, exec_data in processed_data.data_by_pid.items():
        current_batch_index = 0
        batch_data = process_batch_data(exec_data, batch_rows, current_batch_index)
        parrent_path = os.path.join(output_folder, f'pid_{pid}')
        os.makedirs(parrent_path, exist_ok=True)
        file_path = os.path.join(parrent_path, 'feature.csv')
        exec_data_reset = exec_data.reset_index(drop=True)
        batch_data_reset = batch_data.reset_index(drop=True)

        # 使用 merge 方法按行合并，并指定后缀处理重叠列名
        merged_df = exec_data_reset.merge(
            batch_data_reset,
            left_index=True,
            right_index=True,
            suffixes=('_exec', '_batch')
        )
        with open_s(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            write_csv_header(csvfile)
            an_data = ExecutionDataMindie(merged_df, processed_data.req_df, processed_data.rids_ori, 
                                    processed_data.index_dict, processed_data.batch_id_block_sum)
            feature_data = process_execution_data_mindie(an_data)
            for row in feature_data:
                write_csv_row(csvfile, row)


def source_to_model(input_path: str, model_type: str):
    ori_db_path = os.path.join(input_path, 'profiler.db')
    if not Rule.input_file_read.is_satisfied_by(ori_db_path):
        logger.error("please check the db from profiling")
        return
    db_connector = DatabaseConnector(ori_db_path)
    cursor = db_connector.connect()
    try:
        exec_rows = read_batch_exec_data(cursor)
        batch_rows = read_batch_data(cursor)
        data_by_pid = group_exec_data_by_pid(exec_rows)
        csv_file = os.path.join(input_path, 'request.csv')
        req_df = read_csv_s(csv_file, header=0)
        rids_ori = fetch_rids_from_db(ori_db_path)
        if model_type == 'vllm':
            kvcache_file = os.path.join(input_path, 'kvcache.csv')
            kvcache_df = read_csv_s(kvcache_file, header=0)
            csv_data = ProcessedDataVllm(input_path,
                data_by_pid,
                batch_rows,
                req_df,
                rids_ori,
                kvcache_df)
            save_processed_data_to_csv_vllm(csv_data)
        else:
            req_rows = read_batch_req_data(cursor)
            index_dict = {}
            for _, row in req_rows.iterrows():
                key = (row['req_id'], row['iter'])
                value = row['block']
                index_dict[key] = value
            batch_id_block_sum = calculate_block_sums(req_rows)
            csv_data = ProcessedDataMindie(input_path,
                data_by_pid,
                batch_rows,
                req_df,
                rids_ori,
                batch_id_block_sum,
                index_dict)
            save_processed_data_to_csv_mindie(csv_data)
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        raise e
    finally:
        db_connector.close()


def req_decodetimes(input_path, output_path):
    csv_file = os.path.join(input_path, f'request.csv')
    json_file = os.path.join(output_path, f'req_id_and_decode_num.json')
    # 初始化一个空字典来存储数据
    data = {}

    # 打开并读取CSV文件
    with open_s(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        req_id = 0
        for row in reader:
            # 检查reply_token_size是否为空
            if row['reply_token_size'].strip() == '':
                continue
            http_reqid = row['http_rid']
            try:
                reply_token_size = int(float(row['reply_token_size']))
                data[req_id] = reply_token_size
                req_id += 1
            except ValueError:
                # 跳过无法转换为整数的行
                continue

    # 将字典写入JSON文件
    with open_s(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def arg_parse(subparsers):
    parser = subparsers.add_parser(
        "train", formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="train for auto optimize"
    )

    parser.add_argument("-i", "--input", default=None, type=Path, required=True)
    parser.add_argument("-o", "--output", default=Path("output"), type=Path)
    parser.add_argument(
        "-t", 
        "--type", 
        type=str, 
        choices=["vllm", "mindie"], 
        default="mindie",
        help="Specify the type, either 'vllm' or 'mindie' (default: mindie)"
    )
    parser.set_defaults(func=main)


def main(args):
    from msserviceprofiler.modelevalstate.train.pretrain import pretrain
    if is_root():
        logger.warning(
            "Security Warning: Do not run this tool as root. "
            "Running with elevated privileges may compromise system security. "
            "Use a regular user account."
        )
    input_path = args.input
    output_path = args.output
    model_type = args.type
    # 读取输入文件
    try:
        source_to_model(input_path, model_type)
    except IOError as e:
        logger.error(f"无法读取输入文件: {e}")
        raise e
    # 确保输出目录存在
    input_csv_path = os.path.join(input_path, f'output_csv')
    try:
        pretrain(input_csv_path, output_path)
        req_decodetimes(input_path, output_path)
    except Exception as e:
        logger.error(f"pretrain failed, please check. {e}")
