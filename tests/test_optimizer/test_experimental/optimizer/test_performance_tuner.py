# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
from math import inf

from src.optimize.experimental.config.config import PerformanceIndex
from src.optimize.experimental.optimizer.performance_tunner import PerformanceTuner


def test_minimum_algorithm():
    tuner = PerformanceTuner()

    # 测试generate_speed为None的情况
    index = PerformanceIndex(generate_speed=None)
    assert tuner.minimum_algorithm(index) == inf

    # 测试generate_speed为0的情况
    index = PerformanceIndex(generate_speed=0)
    assert tuner.minimum_algorithm(index) == inf

    # 测试time_to_first_token为None的情况
    index = PerformanceIndex(generate_speed=1, time_to_first_token=None)
    assert tuner.minimum_algorithm(index) == inf

    # 测试time_to_first_token导致OverflowError的情况
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1e10)
    assert tuner.minimum_algorithm(index) == inf

    # 测试time_per_output_token为None的情况
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=None)
    assert tuner.minimum_algorithm(index) == inf

    # 测试time_per_output_token导致OverflowError的情况
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=1e10)
    assert tuner.minimum_algorithm(index) == inf

    # 测试success_rate为None的情况
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=1, success_rate=None)
    assert tuner.minimum_algorithm(index) == inf

    # 测试success_rate为0的情况
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=1, success_rate=0)
    assert tuner.minimum_algorithm(index) == inf

    # 测试success_rate导致OverflowError的情况
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=1, success_rate=1e-10)
    assert tuner.minimum_algorithm(index) == inf

    # 测试所有参数都正常的情况
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=1, success_rate=1)
    assert tuner.minimum_algorithm(index) != inf
    # 测试参数组 较好的组合，算出的值更小
    index = PerformanceIndex(generate_speed=1000, time_to_first_token=0.49, time_per_output_token=0.049, success_rate=1)
    index2 = PerformanceIndex(generate_speed=1000, time_to_first_token=0.29, time_per_output_token=0.014,
                              success_rate=1)
    assert tuner.minimum_algorithm(index) > tuner.minimum_algorithm(index2)
    index = PerformanceIndex(generate_speed=1000, time_to_first_token=0.89, time_per_output_token=0.049, success_rate=1)
    index2 = PerformanceIndex(generate_speed=1000, time_to_first_token=0.59, time_per_output_token=0.014,
                              success_rate=1)
    assert tuner.minimum_algorithm(index) > tuner.minimum_algorithm(index2)
    index = PerformanceIndex(generate_speed=1000, time_to_first_token=0.89, time_per_output_token=0.099, success_rate=1)
    index2 = PerformanceIndex(generate_speed=1000, time_to_first_token=0.59, time_per_output_token=0.054,
                              success_rate=1)
    assert tuner.minimum_algorithm(index) > tuner.minimum_algorithm(index2)
    index = PerformanceIndex(generate_speed=1000, time_to_first_token=0.49, time_per_output_token=0.049, success_rate=1)
    index2 = PerformanceIndex(generate_speed=2000, time_to_first_token=0.49, time_per_output_token=0.049,
                              success_rate=1)
    assert tuner.minimum_algorithm(index) > tuner.minimum_algorithm(index2)
