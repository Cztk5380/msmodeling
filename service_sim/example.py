# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from service_sim.device import DummyDeviceConfig, MachineConfig
from service_sim.instance import Instance
from service_sim.load_gen import FixedLengthLoadGen
from service_sim.model_runner import ModelConfig
from service_sim.request import Request
from service_sim.serving import PdDisaggregationServing, PdAggregationServing
import stime
import salabim as sim
from service_sim.utils import main_processing

logger = stime.get_logger(__name__)


def main():
    stime.init_simulation()

    dummy_duration = 0.3
    num_prefill_decode_instances = 1
    prefill_decode_instances = []
    model_config_kwargs = {"head_dim": 64, "num_heads": 64, "precision_bytes": 2, "num_layers": 8} # smaller model
    for _ in range(num_prefill_decode_instances):
        instance = Instance(MachineConfig(DummyDeviceConfig(), 4), 
                            ModelConfig(num_dp_partitions=1, duration=dummy_duration, **model_config_kwargs))
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

    main_task = stime.CallableTask(main_processing, serving, load_runner)

    stime.start_simulation()

if __name__ == "__main__":
    main()