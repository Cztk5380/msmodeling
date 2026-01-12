# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

from serving_cast.communication import CommunicationManager
from serving_cast.config import CommunicationConfig
from stime import (
    CallableTask,
    elapse,
    get_logger,
    init_simulation,
    now,
    start_simulation,
    stop_simulation,
)

logger = get_logger(__name__)


class TestCommunicationManager(unittest.TestCase):
    def setUp(self) -> None:
        # Create a new manager with 10 blocks before each test case
        self.host2device_bandwidth = 100
        self.host2device_rate = 0.5
        init_simulation()
        commun_args = CommunicationConfig(
            host2device_bandwidth=self.host2device_bandwidth,
            host2device_rate=self.host2device_rate,
        )
        self.mgr = CommunicationManager(commun_args)

    def test_host2device_async(self):
        send_bytes = 100
        target_bytes_commun_time = 2
        interval_time = 10

        def func():
            for i in range(5):
                elapse(interval_time)

                def check_callback(index):
                    _ = now()
                    target_now_time = (
                        index + 1
                    ) * interval_time + target_bytes_commun_time
                    self.assertEqual(now(), target_now_time)

                self.mgr.host2device_async(send_bytes, check_callback, i)
            elapse(2)
            stop_simulation()

        _ = CallableTask(func)
        start_simulation()

    def test_host2device_sync(self):
        send_bytes = 100
        target_bytes_commun_time = 2
        interval_time = 10

        def func():
            for i in range(5):
                elapse(interval_time)
                self.mgr.host2device_sync(send_bytes)
                _ = now()
                target_now_time = (i + 1) * (interval_time + target_bytes_commun_time)
                self.assertEqual(now(), target_now_time)
            stop_simulation()

        _ = CallableTask(func)
        start_simulation()

    def test_host2device_async_workload_stack(self):
        send_bytes = 1000
        target_bytes_commun_time = 20
        interval_time = 10

        def func():
            for i in range(5):
                elapse(interval_time)

                def check_callback(index):
                    _ = now()
                    target_now_time = (
                        index + 1
                    ) * target_bytes_commun_time + interval_time
                    self.assertEqual(now(), target_now_time)

                self.mgr.host2device_async(send_bytes, check_callback, i)
            elapse(2)
            stop_simulation()

        _ = CallableTask(func)
        start_simulation()

    def test_host2device_async_fifo(self):
        unit_send_bytes = 50
        unit_target_bytes_commun_time = 1
        interval_time = 0

        def func():
            for i in range(5):
                elapse(interval_time)

                def check_callback(index):
                    _ = now()
                    target_now_time = unit_target_bytes_commun_time * (
                        (index + 2) * (index + 1) // 2
                    )
                    self.assertEqual(now(), target_now_time)

                self.mgr.host2device_async(unit_send_bytes * (i + 1), check_callback, i)
            elapse(2)
            stop_simulation()

        _ = CallableTask(func)
        start_simulation()


if __name__ == "__main__":
    unittest.main()
