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

from prettytable import PrettyTable

from tensor_cast.service.utils import logger


class Summary:
    def __init__(self, data_config):
        self._stop_flag = None
        self._summary_df = None
        self.data_config = data_config

    def set_summary_df(self, summary_df):
        self._summary_df = summary_df

    def get_summary_df(self):
        return self._summary_df

    def set_stop_flag(self, memory_left, tpot, ttft):
        self._stop_flag = (
            (memory_left < 0)
            or (tpot > self.data_config.tpot_limits)
            or (ttft > self.data_config.ttft_limits)
        )

    def check_stop_flag(self):
        return self._stop_flag

    def report_final_result(self, args):
        if self._summary_df is None or self._summary_df.empty:
            logger.warning("Summary DataFrame is None. Please set it first.")
            return
        sorted_summary_df = (
            self._summary_df[
                (self._summary_df["tpot"] <= self.data_config.tpot_limits)
                & (self._summary_df["ttft"] <= self.data_config.ttft_limits)
            ]
            .sort_values(by="token/s", ascending=False)
            .reset_index(drop=True)
            .copy()
        )
        best_result = sorted_summary_df.loc[0]

        final_out = []
        final_out.append("*" * 80)

        final_out.append("  " + "-" * 76)
        final_out.append("  Input Configuration: ")
        final_out.append(f"    Model: {args.model_id}")
        final_out.append(f"    Devices: {args.num_devices} {args.device}")
        final_out.append(f"    TTFT Limits: {self.data_config.ttft_limits}")
        final_out.append(f"    TPOT Limits: {self.data_config.tpot_limits}")
        final_out.append("  " + "-" * 76)

        final_out.append("  Overall Best Configuration: ")
        final_out.append(f"    Best Throughput: {best_result['token/s']:.2f}")
        final_out.append(f"    TTFT: {best_result['ttft']:.2f}")
        final_out.append(f"    TPOT: {best_result['tpot']:.2f}")
        final_out.append("  " + "-" * 76)

        table_buf = _get_agg_table_buf(sorted_summary_df)
        final_out.append(table_buf)
        final_out.append("*" * 80)
        logger.info("%s", "\n" + "\n".join(final_out))


def _get_agg_table_buf(df):
    first_rows = df.groupby("parallel").first().reset_index()
    first_rows = (
        first_rows.sort_values(by="token/s", ascending=False)
        .reset_index(drop=True)
        .copy()
    )
    show_len = len(first_rows)
    table_buf = []
    table_buf.append(f"Top {show_len} Aggregation Configurations: ")
    table = PrettyTable()
    table.field_names = [
        "Top",
        "\033[1mThroughput\033[0m",
        "TTFT",
        "TPOT",
        "concurrency",
        "total_devices",
        "parallel",
        "batch_size",
    ]
    for i in range(show_len):
        row = first_rows.loc[i]
        table.add_row(
            [
                i + 1,
                f"\033[1m{row['token/s']:.2f}\033[0m",
                f"{row['ttft']:.2f}",
                f"{row['tpot']:.2f}",
                row["concurrency"],
                row["total_devices"],
                row["parallel"],
                row["batch_size"],
            ]
        )
    table_buf.append(table.get_string())
    return "\n".join(table_buf)
