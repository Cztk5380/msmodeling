# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import stime
from serving_cast.device import DummyDeviceConfig, MachineConfig
from serving_cast.instance import Instance
from serving_cast.load_gen import FixedLengthLoadGen
from serving_cast.model_runner import ModelConfig
from serving_cast.serving import PdAggregationServing, PdDisaggregationServing
from serving_cast.utils import main_processing, summarize

logger = stime.get_logger(__name__)


def main_pd_aggregation():
    stime.init_simulation()

    dummy_duration = 0.3
    unit_time = 0.0001
    num_prefill_decode_instances = 2
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
                num_dp_partitions=1, unit_time=unit_time, **model_config_kwargs
            ),
        )
        prefill_decode_instances.append(instance)

    num_requests = 100
    num_input_tokens = 1500
    num_output_tokens = 3500
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
    summarize(load_runner.get_finished_requests().values())


def main_pd_disaggregation():
    stime.init_simulation()

    dummy_duration = 0.3
    unit_time = 0.0001
    num_prefill_instances = 1
    num_decode_instances = 1
    prefill_instances = []
    decode_instances = []
    model_config_kwargs = {
        "head_dim": 64,
        "num_heads": 64,
        "precision_bytes": 2,
        "num_layers": 8,
    }  # smaller model
    for _ in range(num_prefill_instances):
        prefill = Instance(
            MachineConfig(DummyDeviceConfig(), 4),
            ModelConfig(
                num_dp_partitions=1, unit_time=unit_time, **model_config_kwargs
            ),
        )
        prefill_instances.append(prefill)
    for _ in range(num_decode_instances):
        decode = Instance(
            MachineConfig(DummyDeviceConfig(), 4),
            ModelConfig(
                num_dp_partitions=1, unit_time=unit_time, **model_config_kwargs
            ),
        )
        decode_instances.append(decode)

    num_requests = 100
    num_input_tokens = 1500
    num_output_tokens = 3500
    serving = PdDisaggregationServing(prefill_instances, decode_instances)
    load_runner = FixedLengthLoadGen(
        model_name=None,
        num_requests=num_requests,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        request_rate=1000.0,
    )

    main_task = stime.CallableTask(main_processing, serving, load_runner)

    stime.start_simulation()
    summarize(load_runner.get_finished_requests().values())


if __name__ == "__main__":
    main_pd_aggregation()
