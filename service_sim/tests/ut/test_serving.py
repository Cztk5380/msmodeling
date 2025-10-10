# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

import stime
from service_sim.config import Config
from service_sim.device import DummyDeviceConfig, MachineConfig

from service_sim.instance import Instance
from service_sim.load_gen import FixedLengthLoadGen
from service_sim.model_runner import ModelConfig
from service_sim.request import Request
from service_sim.serving import PdAggregationServing, PdDisaggregationServing
from service_sim.utils import main_processing


class ServingTestCase(unittest.TestCase):
    def setUp(self):
        stime.init_simulation()
        self.mock_cfg = Mock()
        self.mock_cfg.common_config.serving_config.max_concurrency = 100
        self.mock_cfg.enable_profiling = False


        self.patch_get = patch.object(Config, "get_instance")
        mock_get = self.patch_get.start()
        mock_get.return_value = self.mock_cfg

    def tearDown(self):
        self.patch_get.stop()

    def test_pd_disaggregation_dummy_model(self):
        dummy_duration = 0.3
        num_prefill_instances = 8
        num_decode_instances = 8
        prefill_instances = []
        decode_instances = []
        model_config_kwargs = {
            "head_dim": 64,
            "num_heads": 64,
            "precision_bytes": 2,
            "num_layers": 8,
        }
        for _ in range(num_prefill_instances):
            prefill = Instance(
                MachineConfig(DummyDeviceConfig(), 4),
                ModelConfig(
                    num_dp_partitions=2, duration=dummy_duration, **model_config_kwargs
                ),
            )
            prefill_instances.append(prefill)
        for _ in range(num_decode_instances):
            decode = Instance(
                MachineConfig(DummyDeviceConfig(), 8),
                ModelConfig(
                    num_dp_partitions=2, duration=dummy_duration, **model_config_kwargs
                ),
            )
            decode_instances.append(decode)

        num_requests = 10
        num_input_tokens = 2048
        num_output_tokens = 50
        serving = PdDisaggregationServing(prefill_instances, decode_instances)
        load_runner = FixedLengthLoadGen(
            model_name=None,
            num_requests=num_requests,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            request_rate=1.0,
        )

        main_task = stime.CallableTask(main_processing, serving, load_runner)

        stime.start_simulation()
        requests = load_runner.get_finished_requests()
        # all requests have been served
        self.assertEqual(len(requests), num_requests)

        for request in requests.values():
            self.assertAlmostEqual(request.time_to_first_token(), dummy_duration)
            self.assertEqual(request.num_decoded_tokens, num_output_tokens)
            self.assertGreater(request.time_per_output_token(), dummy_duration - 0.1)
            self.assertLess(request.time_per_output_token(), dummy_duration + 0.1)

    def test_pd_aggregation_dummy_model(self):
        dummy_duration = 0.3
        num_prefill_decode_instances = 8
        prefill_decode_instances = []
        model_config_kwargs = {
            "head_dim": 128,
            "num_heads": 64,
            "precision_bytes": 2,
            "num_layers": 64,
        }
        for _ in range(num_prefill_decode_instances):
            instance = Instance(
                MachineConfig(DummyDeviceConfig(), 4),
                ModelConfig(
                    num_dp_partitions=2, duration=dummy_duration, **model_config_kwargs
                ),
            )
            prefill_decode_instances.append(instance)

        num_requests = 10
        num_input_tokens = 2048
        num_output_tokens = 50
        serving = PdAggregationServing(prefill_decode_instances)
        load_runner = FixedLengthLoadGen(
            model_name=None,
            num_requests=num_requests,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            request_rate=1.0,
        )

        def count_completed(request: Request):
            nonlocal requests
            requests[request.id] = request

        main_task = stime.CallableTask(main_processing, serving, load_runner)

        stime.start_simulation()
        requests = load_runner.get_finished_requests()
        # all requests have been served
        self.assertEqual(len(requests), num_requests)
        for request in requests.values():
            self.assertEqual(request.num_decoded_tokens, num_output_tokens)
            self.assertAlmostEqual(request.time_to_first_token(), dummy_duration)
            self.assertAlmostEqual(request.time_per_output_token(), dummy_duration)

    def test_pd_aggregation_dummy_model_single_scheduler(self):
        dummy_duration = 0.3
        num_prefill_decode_instances = 1
        prefill_decode_instances = []
        model_config_kwargs = {
            "head_dim": 64,
            "num_heads": 64,
            "precision_bytes": 2,
            "num_layers": 8,
        }  # smaller model
        for _ in range(num_prefill_decode_instances):
            instance = Instance(
                MachineConfig(DummyDeviceConfig(), 4),
                ModelConfig(
                    num_dp_partitions=1, duration=dummy_duration, **model_config_kwargs
                ),
            )
            prefill_decode_instances.append(instance)

        num_requests = 100
        num_input_tokens = 2048
        num_output_tokens = 50
        serving = PdAggregationServing(prefill_decode_instances)
        load_runner = FixedLengthLoadGen(
            model_name=None,
            num_requests=num_requests,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            request_rate=1.0,
        )

        def count_completed(request: Request):
            nonlocal requests
            requests[request.id] = request

        main_task = stime.CallableTask(main_processing, serving, load_runner)

        stime.start_simulation()
        requests = load_runner.get_finished_requests()
        # all requests have been served
        self.assertEqual(len(requests), num_requests)
        for request in requests.values():
            self.assertEqual(request.num_decoded_tokens, num_output_tokens)
            self.assertAlmostEqual(request.time_per_output_token(), dummy_duration)

    def test_pd_aggregation_dummy_model_single_scheduler_trigger_preempt(self):
        dummy_duration = 0.3
        num_prefill_decode_instances = 1
        prefill_decode_instances = []
        model_config_kwargs = {
            "head_dim": 64,
            "num_heads": 64,
            "precision_bytes": 2,
            "num_layers": 8,
        }  # smaller model
        for _ in range(num_prefill_decode_instances):
            instance = Instance(
                MachineConfig(DummyDeviceConfig(), 4),
                ModelConfig(
                    num_dp_partitions=1, duration=dummy_duration, **model_config_kwargs
                ),
            )
            prefill_decode_instances.append(instance)

        num_requests = 100
        num_input_tokens = 2048
        num_output_tokens = 50
        serving = PdAggregationServing(prefill_decode_instances)
        load_runner = FixedLengthLoadGen(
            model_name=None,
            num_requests=num_requests,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            request_rate=5.0,  # increase sending rate to trigger preempt
        )
        requests = {}

        def count_completed(request: Request):
            nonlocal requests
            requests[request.id] = request

        main_task = stime.CallableTask(main_processing, serving, load_runner)

        stime.start_simulation()
        requests = load_runner.get_finished_requests()
        # all requests have been served
        self.assertEqual(len(requests), num_requests)
        for request in requests.values():
            self.assertEqual(request.num_decoded_tokens, num_output_tokens)


if __name__ == "__main__":
    unittest.main()
