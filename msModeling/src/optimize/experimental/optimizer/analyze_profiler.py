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

import glob
import os
import re
from loguru import logger

import pandas as pd
from msserviceprofiler.modelevalstate.common import read_csv_s


def find_first_simulate_csv(input_path_2):
    # 检查输入路径是否为有效目录
    if not os.path.exists(input_path_2) or not os.path.isdir(input_path_2):
        raise NotADirectoryError("The provided path is not a valid directory.")

    # 构建查找模式
    pattern = os.path.join(input_path_2, "simulate*.csv")

    # 查找所有匹配的文件
    files = glob.glob(pattern)

    # 检查是否有匹配的文件
    if not files:
        raise FileNotFoundError("No CSV files starting with 'simulate' found in the directory.")

    # 按文件名排序
    files.sort()

    # 返回第一个文件的路径
    return files[0]


def calculate_total_simulate_time(df_prefill, df3, df1):
    total_simulate_time = 0
    total_decode_simulate_time = 0
    
    for _, row in df_prefill.iterrows():
        digits = re.findall(r'\d+', row['reqinfo'])
        reqinfo_list = [int(num) for num in digits]
        non_zero_values = [x for i, x in enumerate(reqinfo_list) if i % 2 == 0]
        
        for val in non_zero_values:
            if pd.isna(df3.iloc[val]['reply_token_size']):
                continue
            during_time = df3.iloc[val]['first_token_latency']
            decode_time = (df3.iloc[val]['execution_time(microsecond)'] - df3.iloc[val]['first_token_latency'])
            arrive_time = row['start_time(microsecond)'] - during_time
            complete_time = row['start_time(microsecond)'] + decode_time
            
            filtered_df1 = df1[df1['start_time(microsecond)'] > arrive_time]
            filtered_df1 = filtered_df1[filtered_df1['start_time(microsecond)'] <= row['start_time(microsecond)']]
            
            filtered_df2 = df1[df1['start_time(microsecond)'] < complete_time]
            filtered_df2 = filtered_df2[filtered_df2['start_time(microsecond)'] >= row['start_time(microsecond)']]
            
            total_simulate_time += filtered_df1['simulate_time'].sum()
            total_decode_simulate_time += filtered_df2['simulate_time'].sum()
    
    return total_simulate_time, total_decode_simulate_time


def analyze(input_path_1, input_path_2):
    profiling_path = os.path.join(input_path_1, 'request.csv')
    df3 = read_csv_s(profiling_path, header=0)

    total_req = df3.shape[0]
    filtered_df = df3[df3['reply_token_size'].notna()]
    success_req = filtered_df.shape[0]

    batch_path = os.path.join(input_path_1, 'batch.csv')
    df1 = read_csv_s(batch_path, header=0)
    df1 = df1[df1['name'] == 'modelExec']
    simulate_path = find_first_simulate_csv(input_path_2)
    df2 = read_csv_s(simulate_path, header=None)
    column_name = ['simulate_time']
    df2.columns = column_name
    # 确认两文件的行数相同
    if len(df1) != len(df2):
        raise ValueError("两个CSV文件的行数必须相同")
    # 将第二个CSV文件中的`during_time`列添加到第一个CSV文件中，并修改列名
    df1['simulate_time'] = df2['simulate_time'] / 10 ** 6
    df3.sort_values(by='http_rid', ascending=True, inplace=True)
    df_prefill = df1[df1['batch_type'] == 'prefill']
    total_simulate_time, total_decode_simulate_time = calculate_total_simulate_time(df_prefill, df3, df1)

    df3['completed_time'] = df3['start_time_httpReq(microsecond)'] + df3['execution_time(microsecond)']
    total_latency = df3['first_token_latency'].sum() + total_simulate_time
    total_token = df3['reply_token_size'].sum()
    try:
        avg_prefill_latency = total_latency / success_req / 10 ** 6
    except ZeroDivisionError as ex:
        raise ValueError(f"success_req cannot be zero. {ex}") from ex
    total_time = df3['completed_time'].max() - df3['start_time_httpReq(microsecond)'].min() + df1['simulate_time'].sum()
    try:
        throughput = total_token / total_time * 10 ** 6
    except ZeroDivisionError as ex:
        raise ValueError(f"total_time cannot be zero. {ex}") from ex
    total_decode_time = df3['execution_time(microsecond)'].sum() + total_decode_simulate_time - df3[
        'first_token_latency'].sum()
    try:
        average_decode_latency = total_decode_time / (total_token - success_req) / 10 ** 6
    except ZeroDivisionError as ex:
        raise ValueError(f"total_token - success_req cannot be zero. {ex}") from ex
    try:
        success_precent = success_req / total_req
    except ZeroDivisionError as ex:
        raise ValueError(f"total_req cannot be zero. {ex}") from ex
    return throughput, avg_prefill_latency, average_decode_latency, success_precent
