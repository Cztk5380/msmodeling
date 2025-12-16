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

from abc import ABC, abstractmethod

import pandas as pd

from tensor_cast.service.report_and_save import Summary

from tensor_cast.service.utils import AGG_COLUMNS, MAX_ITER_NUMS


class BaseBackend(ABC):
    name = "base"

    @abstractmethod
    def initialize(self, args, model):
        pass

    @abstractmethod
    def run_inference(self, data_config) -> Summary:
        pass

    def find_best_result_under_constraints(self, data_config) -> Summary:
        left, right = 1, 512
        result = []
        result_df = pd.DataFrame(columns=AGG_COLUMNS)

        # early_stop
        data_config.batch_size = left
        summary = self.run_inference(data_config)
        if summary.check_stop_flag():
            return None
        for _ in range(MAX_ITER_NUMS):
            data_config.batch_size = right
            summary = self.run_inference(data_config)
            if summary.check_stop_flag():
                break
            else:
                left, right = right, right * 2

        while left <= right:
            mid = (left + right) // 2
            data_config.batch_size = mid
            summary = self.run_inference(data_config)
            if summary.check_stop_flag():
                right = mid - 1
            else:
                left = mid + 1
                result.append(summary.get_summary_df())

        if result:
            result_df = pd.concat(result, axis=0, ignore_index=True)

        sorted_df = result_df.sort_values(by=["token/s"], ascending=[True]).round(3)

        ret_summary = Summary(data_config)
        ret_summary.set_summary_df(sorted_df)

        return ret_summary
