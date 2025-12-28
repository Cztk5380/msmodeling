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

from collections import defaultdict

import pandas as pd

from tensor_cast.core.utils import generate_inputs, RequestInfo
from tensor_cast.service.base_backend import BaseBackend
from tensor_cast.service.report_and_save import Summary
from tensor_cast.service.utils import AGG_COLUMNS, logger, run_static


class MindIEAggBackend(BaseBackend):
    name = "mindie_aggregation"

    def initialize(self, args, model):
        self.args = args
        self.model = model
        self.num_mtp_tokens = (
            self.model.model_config.mtp_config.num_mtp_layers
            if self.model.model_config.mtp_config is not None
            else 0
        )
        self.dp = self.model.model_config.parallel_config.data_parallel_size
        self.tp = self.model.model_config.parallel_config.tensor_parallel_size
        self.pp = self.model.model_config.parallel_config.pipeline_parallel_size
        self._prefill_cache = defaultdict(lambda: None)
        self._decode_cache = defaultdict(lambda: None)

    def run_inference(self, data_config) -> Summary:
        max_prefill_tokens = data_config.max_prefill_tokens
        batch_size = data_config.batch_size
        input_length = data_config.input_length
        output_length = data_config.output_length
        concurrency = batch_size * self.dp * self.pp

        # calculate TTFT, we get average ttft = sum_for_ttft / concurrency, for sum_for_ttft,
        # we assume the prefill batch size is the max prefill tokens divided by input length.
        # so prefill_batch_size = max_prefill_tokens // input_length. And request was processed in
        # prefill_batch_size steps one by one.
        # For example, if we have 12 requests, and max_prefill_tokens is 8192, input_length is 2048,
        # then prefill_batch_size is 4. And 8 requests was processed in 3 steps.
        # so sum_for_ttft = (prefill_latency * prefill_batch_size(4 in this case) )) *
        #                       (1 + (calc_nums_for_ttft(3 in this case) )) * (calc_nums_for_ttft) / 2
        # ttft = sum_for_ttft / concurrency (12 in this case)
        # LEFT: the number of tokens that cannot be calculated in one prefill batch
        prefill_batch_size = max_prefill_tokens // input_length
        calc_nums_for_ttft = concurrency // prefill_batch_size
        left_calc_num = concurrency % prefill_batch_size

        prefill_latency, prefill_memory_left_gb = self._get_or_compute_prefill_latency(
            prefill_batch_size, data_config, True
        )
        left_latency = 0
        if left_calc_num != 0:
            left_latency, _ = self._get_or_compute_prefill_latency(
                left_calc_num, data_config, True
            )

        left_batch_time = (
            calc_nums_for_ttft * prefill_latency + left_latency
        ) * left_calc_num
        sum_for_ttft = (prefill_batch_size * prefill_latency) * (
            1 + calc_nums_for_ttft
        ) * calc_nums_for_ttft / 2 + left_batch_time
        ttft = sum_for_ttft / concurrency

        # calculate TPOT
        decode_latency, decode_memory_left_gb = self._get_or_compute_decode_latency(
            batch_size, data_config, True
        )
        # LEFT: we don't consider the bubble time
        tpot = (ttft + decode_latency * output_length) / output_length
        # calculate output throughput: we assume e2e latency is ttft + tpot * output_length
        output_throughput = (
            1000 * (output_length * concurrency) / (ttft + tpot * output_length)
        )

        memory_left = min(prefill_memory_left_gb, decode_memory_left_gb)
        token_s_device = output_throughput / self.dp / self.pp / self.tp
        parallel = f"tp{self.tp}pp{self.pp}dp{self.dp}"

        logger.debug(
            "Prefill Latency: %.4f ms, "
            "Decode Latency: %.4f ms, "
            "TTFT: %.4f ms, TPOT: %.4f ms, "
            "Output Throughput: %.2f token/s, "
            "Concurrency: %d, "
            "parallel: %s, "
            "Memory Left: %.2f GB",
            prefill_latency,
            decode_latency,
            ttft,
            tpot,
            output_throughput,
            concurrency,
            parallel,
            memory_left,
        )

        summary = Summary(data_config)
        result_df = pd.DataFrame(
            columns=AGG_COLUMNS,
            data=[
                [
                    self.args.model_id,
                    input_length,
                    output_length,
                    concurrency,
                    ttft,
                    tpot,
                    self.args.num_devices,
                    self.name,
                    self.args.device,
                    output_throughput,
                    token_s_device,
                    parallel,
                    batch_size,
                ]
            ],
        ).round(3)
        summary.set_summary_df(result_df)
        summary.set_stop_flag(memory_left, tpot, ttft)

        return summary

    def _get_prefill_forward(self, batch_size, data_config, is_concurrency=False):
        query_len = data_config.input_length
        device_profile = data_config.device_profile
        seq_len = query_len
        if is_concurrency:
            batch_size = batch_size * self.dp * self.pp
        requests = [
            RequestInfo(
                query_len=query_len,
                seq_len=seq_len,
                concurrency=batch_size,
                is_decode=False,
            )
        ]
        input_kwargs = generate_inputs(self.model, requests)

        return run_static(self.model, input_kwargs, device_profile)

    def _get_decode_forward(self, batch_size, data_config, is_concurrency=False):
        query_len = self.num_mtp_tokens + 1
        seq_len = data_config.output_length // 2 + data_config.input_length + query_len
        device_profile = data_config.device_profile
        if is_concurrency:
            batch_size = batch_size * self.dp * self.pp
        requests = [
            RequestInfo(
                query_len=query_len,
                seq_len=seq_len,
                concurrency=batch_size,
                is_decode=False,
            )
        ]
        inputs_kwargs = generate_inputs(self.model, requests)
        return run_static(self.model, inputs_kwargs, device_profile)

    def _get_or_compute_prefill_latency(
        self, batch_size, data_config, is_concurrency=False
    ):
        # Check if result already exists in cache
        batch_flag = self._prefill_cache.get(batch_size)

        if batch_flag:
            (prefill_latency, memory_left_gb) = self._prefill_cache[batch_size]
        else:
            # Compute prefill result
            batch_result = self._get_prefill_forward(
                batch_size, data_config, is_concurrency
            )
            prefill_latency = batch_result.get("execution_time_s")
            memory_left_gb = batch_result.get("device_memory_available_gb")

            # Cache result if memory is sufficient
            if memory_left_gb > 0:
                self._prefill_cache[batch_size] = (prefill_latency, memory_left_gb)

        return prefill_latency, memory_left_gb

    def _get_or_compute_decode_latency(
        self, batch_size, data_config, is_concurrency=False
    ):
        # Check if result already exists in cache
        batch_flag = self._decode_cache.get(batch_size)

        if batch_flag:
            (decode_latency, memory_left_gb) = self._decode_cache[batch_size]
        else:
            # Compute decode result
            batch_result = self._get_decode_forward(
                batch_size, data_config, is_concurrency
            )
            decode_latency = batch_result.get("execution_time_s")
            average_tokens = (
                sum(self.args.mtp_acceptance_rate[: self.args.num_mtp_tokens]) + 1
            )
            # average_tokens is always greater than 0
            decode_latency /= average_tokens
            memory_left_gb = batch_result.get("device_memory_available_gb")

            # Cache result if memory is sufficient
            if memory_left_gb > 0:
                self._decode_cache[batch_size] = (decode_latency, memory_left_gb)

        return decode_latency, memory_left_gb
