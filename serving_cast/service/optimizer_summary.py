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

import logging

import pandas as pd
from prettytable import PrettyTable


logger = logging.getLogger()


TTFT_COLUMN = "TTFT (ms)"
TPOT_COLUMN = "TPOT (ms)"
SHOW_COLUMNS = [
    "Top",
    "\033[1mThroughput\033[0m (token/s)",
    TTFT_COLUMN,
    TPOT_COLUMN,
    "concurrency",
    "num_devices",
    "parallel",
    "batch_size",
]


class OptimizerSummary:
    def __init__(self, data_config):
        self._early_stop_flag = None
        self._summary_df = None
        self.data_config = data_config

    def set_summary_df(self, summary_df):
        self._summary_df = summary_df

    def get_summary_df(self):
        return self._summary_df

    def set_early_stop_flag(self, memory_left, tpot, ttft):
        def check(value, limit):
            return value is not None and limit is not None and value > limit

        self._early_stop_flag = (
            (memory_left < 0)
            or check(tpot, self.data_config.tpot_limits)
            or check(ttft, self.data_config.ttft_limits)
        )

    def check_early_stop_flag(self):
        return self._early_stop_flag

    def report_final_result(self, args):
        if self._summary_df is None or self._summary_df.empty:
            logger.warning("Summary DataFrame is None. Please set it first.")
            return
        # process None value
        tpot_limit = self.data_config.tpot_limits or float("inf")
        ttft_limit = self.data_config.ttft_limits or float("inf")

        mask = (
            pd.to_numeric(self._summary_df["tpot"], errors="coerce").fillna(
                float("inf")
            )
            <= tpot_limit
        ) & (
            pd.to_numeric(self._summary_df["ttft"], errors="coerce").fillna(
                float("inf")
            )
            <= ttft_limit
        )

        # get the first row for each parallel
        first_rows = (
            self._summary_df[mask]
            .sort_values(by="token/s", ascending=False)
            .groupby("parallel")
            .first()
            .reset_index()
            .sort_values(by="token/s", ascending=False)
            .reset_index(drop=True)
        )

        if args.dump_original_results:
            print("\n" + first_rows.to_string(index=False) + "\n")
        else:
            final_out = self._get_final_out(args, first_rows)
            print("\n" + "\n".join(final_out))

    def _get_final_out(self, args, sorted_summary_df):
        best_result = sorted_summary_df.loc[0]

        final_out = []
        final_out.append("*" * 80)

        final_out.append("  " + "-" * 76)
        final_out.append("  Input Configuration: ")
        final_out.append(f"    Model: {args.model_id}")
        final_out.append(f"    Quantize Linear action: {args.quantize_linear_action}")
        final_out.append(
            f"    Quantize Attention action: {args.quantize_attention_action}"
        )
        final_out.append(f"    Devices: {args.num_devices} {args.device}")
        final_out.append(f"    TTFT Limits: {self.data_config.ttft_limits} ms")
        final_out.append(f"    TPOT Limits: {self.data_config.tpot_limits} ms")
        final_out.append("  " + "-" * 76)

        final_out.append("  Overall Best Configuration: ")
        final_out.append(f"    Best Throughput: {best_result['token/s']:.2f} tokens/s")
        if best_result["ttft"] is not None:
            final_out.append(f"    TTFT: {best_result['ttft']:.2f} ms")
        if best_result["tpot"] is not None:
            final_out.append(f"    TPOT: {best_result['tpot']:.2f} ms")
        final_out.append("  " + "-" * 76)

        table_buf = (
            _get_disagg_table_buf(sorted_summary_df)
            if args.disagg
            else _get_agg_table_buf(sorted_summary_df)
        )
        final_out.append(table_buf)
        final_out.append("*" * 80)

        return final_out


def _get_agg_table_buf(df: pd.DataFrame):
    show_len = len(df)
    table_buf = []
    table_buf.append(f"Top {show_len} Aggregation Configurations: ")
    table = PrettyTable()
    table.field_names = SHOW_COLUMNS
    for i in range(show_len):
        row = df.loc[i]
        table.add_row(
            [
                i + 1,
                f"\033[1m{row['token/s']:.2f}\033[0m",
                f"{row['ttft']:.2f}",
                f"{row['tpot']:.2f}",
                row["concurrency"],
                row["num_devices"],
                row["parallel"],
                row["batch_size"],
            ]
        )
    table_buf.append(table.get_string())
    return "\n".join(table_buf)


def _get_disagg_table_buf(df: pd.DataFrame):
    is_decode = df.get("ttft").loc[0] is None
    local_column = SHOW_COLUMNS.copy()
    show_len = len(df)
    table_buf = []
    table = PrettyTable()
    if is_decode:
        table_buf.append(f"Top {show_len} Disaggregation (Decode) Configurations: ")
        local_column.remove(TTFT_COLUMN)
    else:
        table_buf.append(f"Top {show_len} Disaggregation (Prefill) Configurations: ")
        local_column.remove(TPOT_COLUMN)

    table.field_names = local_column
    for i in range(show_len):
        row = df.loc[i]
        table.add_row(
            [
                i + 1,
                f"\033[1m{row['token/s']:.2f}\033[0m",
                f"{row['tpot']:.2f}" if is_decode else f"{row['ttft']:.2f}",
                row["concurrency"],
                row["num_devices"],
                row["parallel"],
                row["batch_size"],
            ]
        )
    table_buf.append(table.get_string())
    return "\n".join(table_buf)
