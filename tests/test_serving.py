import unittest
from device import DummyDeviceConfig, MachineConfig

from instance import DecodeInstance, PrefillInstance
from load_gen import FixedLengthLoadGen
from model import ModelConfig
from request import Request
from serving import Serving


class ServingTestCase(unittest.TestCase):
    def test_dummpy_model(self):
        dummy_duration = 0.3
        num_prefill_instances = 8
        num_decode_instances = 8
        prefill_instances = []
        decode_instances = []
        for _ in range(num_prefill_instances):
            prefill = PrefillInstance(MachineConfig(DummyDeviceConfig(), 4), ModelConfig(num_dp_partitions=2, duration=dummy_duration))
            prefill_instances.append(prefill)
        for _ in range(num_decode_instances):
            decode = DecodeInstance(MachineConfig(DummyDeviceConfig(), 8), ModelConfig(num_dp_partitions=2, duration=dummy_duration))
            decode_instances.append(decode)

        num_requests = 100
        num_input_tokens = 2048
        num_output_tokens = 50
        serving = Serving(prefill_instances, decode_instances)
        load_runner = FixedLengthLoadGen(
            model_name=None,
            num_requests=num_requests,
            num_output_tokens=num_output_tokens,
            request_rate=1.0,
        )
        requests = {}

        def count_completed(request: Request):
            nonlocal requests
            requests[request.id] = request

        while load_runner.has_request():
            request = load_runner.next_requst()
            request.decode_done_signal.connect(count_completed)
            serving.serve(request)
        serving.join()
        # all requests have been served
        self.assertEqual(len(requests), num_requests)
        for request in requests.values():
            self.assertEqual(request.num_decoded_tokens, num_output_tokens)
            self.assertAlmostEqual(request.time_to_first_token(), dummy_duration)
            self.assertAlmostEqual(request.time_per_output_token(), dummy_duration)
        for instance in prefill_instances:
            # no requests remaining in the prefill instance
            self.assertEqual(len(instance.requests), 0)
            # make sure load balancing works somehow
            self.assertGreater(instance.max_concurrent_requests, 0)
            for engine in instance.engines:
                # no requests remaining in the decode instance
                self.assertEqual(len(instance.requests), 0)
        for instance in decode_instances:
            # no requests remaining in the decode instances
            self.assertEqual(len(instance.requests), 0)
            # make sure load balancing works somehow
            self.assertGreater(instance.max_concurrent_requests, 0)
            for engine in instance.engines:
                # no requests remaining in the engine
                self.assertEqual(len(engine.requests), 0)