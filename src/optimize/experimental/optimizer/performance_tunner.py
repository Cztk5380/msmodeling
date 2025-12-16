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
from math import exp, inf


class PerformanceTuner:
    def __init__(self,
                 ttft_penalty: float = 3.0,
                 tpot_penalty: float = 3.0,
                 success_rate_penalty: float = 5.0,
                 ttft_slo: float = 0.5,
                 tpot_slo: float = 0.05,
                 success_rate_slo: float = 1,
                 generate_speed_target: float = 5300):
        # 权重 总和为1
        self.w_gen = 0.4
        self.w_ft = 0.2
        self.w_pot = 0.3
        self.w_succ = 0.1

        # 2. SLO 和 目标值
        self.gen_speed_target = generate_speed_target  # 全局最大的生成速度 (token/s)
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.success_rate_slo = success_rate_slo

        # 3. 惩罚系数 (k值越大，惩罚越重)
        self.ttft_penalty = ttft_penalty
        self.tpot_penalty = tpot_penalty
        self.success_rate_penalty = success_rate_penalty

    def minimum_algorithm(self, performance_index) -> float:
        """
        最小化适应度值（成本）
        """
        total_cost = 0.0

        if performance_index.generate_speed is not None and performance_index.generate_speed > 0:
            cost_gen = self.gen_speed_target / performance_index.generate_speed
            total_cost += self.w_gen * cost_gen
        else:
            return inf

        if performance_index.time_to_first_token is not None and self.ttft_slo > 0:
            try:
                cost_ft = exp(self.ttft_penalty * (performance_index.time_to_first_token / self.ttft_slo - 1))
                total_cost += self.w_ft * cost_ft
            except (OverflowError, ZeroDivisionError):
                return inf

        if performance_index.time_per_output_token is not None and self.tpot_slo > 0:
            try:
                cost_pot = exp(self.tpot_penalty * (performance_index.time_per_output_token / self.tpot_slo - 1))
                total_cost += self.w_pot * cost_pot
            except (OverflowError, ZeroDivisionError):
                return inf

        if performance_index.success_rate is not None and performance_index.success_rate > 0:
            try:
                cost_succ = exp(
                    self.success_rate_penalty * (self.success_rate_slo / performance_index.success_rate - 1))
                total_cost += self.w_succ * cost_succ
            except (OverflowError, ZeroDivisionError):
                return inf
        else:
            return inf

        return total_cost
