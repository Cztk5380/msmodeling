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
import unittest
from copy import deepcopy
import numpy as np
from src.optimize.experimental.config.config import map_param_with_value, OptimizerConfigField


class TestMapParamWithValueRealFields(unittest.TestCase):
    def setUp(self):

        self.default_support_field = [
            OptimizerConfigField(name="max_batch_size", 
                               config_position="BackendConfig.ScheduleConfig.maxBatchSize", 
                               min=25, max=300, dtype="int"),
            OptimizerConfigField(name="max_prefill_batch_size",
                               config_position="BackendConfig.ScheduleConfig.maxPrefillBatchSize", 
                               min=1, max=25, dtype="int"),
            OptimizerConfigField(name="prefill_time_ms_per_req",
                               config_position="BackendConfig.ScheduleConfig.prefillTimeMsPerReq", 
                               max=1000, dtype="int"),
            OptimizerConfigField(name="decode_time_ms_per_req",
                               config_position="BackendConfig.ScheduleConfig.decodeTimeMsPerReq", 
                               max=1000, dtype="int"),
            OptimizerConfigField(name="support_select_batch",
                               config_position="BackendConfig.ScheduleConfig.supportSelectBatch", 
                               max=1, dtype="bool"),
            OptimizerConfigField(name="max_prefill_token",
                               config_position="BackendConfig.ScheduleConfig.maxPrefillTokens", 
                               min=4096, max=409600, dtype="int"),
            OptimizerConfigField(name="max_queue_deloy_microseconds",
                               config_position="BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds", 
                               min=500, max=1000000, dtype="int"),
            OptimizerConfigField(name="prefill_policy_type",
                               config_position="BackendConfig.ScheduleConfig.prefillPolicyType", 
                               min=0, max=1, dtype="enum", dtype_param=[0, 1, 3]),
            OptimizerConfigField(name="decode_policy_type",
                               config_position="BackendConfig.ScheduleConfig.decodePolicyType", 
                               min=0, max=1, dtype="enum", dtype_param=[0, 1, 3]),
            OptimizerConfigField(name="max_preempt_count",
                               config_position="BackendConfig.ScheduleConfig.maxPreemptCount", 
                               min=0, max=1, dtype="ratio", dtype_param="max_batch_size")
        ]
        self.pd_field = [
            OptimizerConfigField(name="default_p_rate", 
                               config_position="default_p_rate", 
                               min=1, max=3, dtype="int", value=1),
            OptimizerConfigField(name="default_d_rate", 
                               config_position="default_d_rate", 
                               min=1, max=3, dtype="share", dtype_param="default_p_rate"),
        ]

    def test_int_type_with_min_max(self):
        # 测试 int 类型（带 min/max 约束）
        params = np.array([26.7, 12.3, 999.9, 500.0, 0.6, 40960.0, 750000.0])
        result = map_param_with_value(params, self.default_support_field[:7])
        
        # 验证字段值是否符合预期
        self.assertEqual(result[0].value, 26) 
        self.assertEqual(result[1].value, 12)  
        self.assertEqual(result[2].value, 999)  
        self.assertEqual(result[3].value, 500)   
        self.assertTrue(result[4].value)  
        self.assertEqual(result[5].value, 40960) 
        self.assertEqual(result[6].value, 750000) 

    def test_enum_type_mapping(self):
        # 测试 enum 类型的分段映射
        params = np.array([0.0, 0.3, 0.6, 1.0])
        enum_fields = [
            self.default_support_field[7],  # prefill_policy_type (enum [0,1,3])
            self.default_support_field[8]   # decode_policy_type (enum [0,1,3])
        ]
        result = map_param_with_value(params, enum_fields)
        
        # 验证 enum 分段逻辑
        self.assertEqual(result[0].value, 0) 
        self.assertEqual(result[1].value, 0)


    def test_ratio_type_dependency(self):
        # 测试 ratio 类型（依赖 max_batch_size）
        params = np.array([0.5])
        ratio_field = self.default_support_field[9]  # max_preempt_count (ratio)
        
        # 手动设置依赖字段的值
        max_batch_size_field = OptimizerConfigField(
            name="max_batch_size", config_position="BackendConfig.ScheduleConfig.maxBatchSize", dtype="int", value=100,   
        )
        
        result = map_param_with_value(params, [max_batch_size_field])
        self.assertEqual(result[0].value, 0)  
    
    def test_share_type_mapping(self):
        params = np.array([1, 2])
        share_ratio = map_param_with_value(params, self.pd_field)
        self.assertEqual(share_ratio[1].value, 3)

    def test_edge_cases(self):
        # 测试边界条件
        params = np.array([24.9, 0.0, 0.0, 0.0, 0.4, 4095.9, 499.9, -1.0, 2.0, 1.1])
        result = map_param_with_value(params, self.default_support_field)
        
        # 验证边界处理
        self.assertEqual(result[0].value, 24)  
        self.assertEqual(result[1].value, 1)   
        self.assertFalse(result[4].value) 
        self.assertEqual(result[5].value, 4095) 
        self.assertEqual(result[6].value, 499)  
        self.assertEqual(result[7].value, 0)   
