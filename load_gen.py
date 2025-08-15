# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict

import stime
from request import Request, RequestState


class LoadGen(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def next_request(self) -> Request:
        """
        Each request is a stime object (i.e. has a timestamp attached to it) meaning its
        expected arriving time.
        When the caller invokes this method and get a request, the timestamp of the caller
        thread would be aligned to the timestamp of the returned request if the current
        timestamp of the thread is no later than the arriving time of the request.
        """
        return None

    @abstractmethod
    def has_request(self):
        """
        Check if the load runner has any request to generate. This includes all the requests
        that have not arrived yet but would come in the future.
        """
        return False


class FixedLengthLoadGen(LoadGen):
    """
    A load runner that always produces fixed-length input and output sequences
    """

    def __init__(
        self,
        model_name: str,
        num_requests: int,
        num_input_tokens: int,
        num_output_tokens: int,
        request_rate: float,
    ):
        super().__init__(model_name)
        self.request_rate = request_rate
        self.requests: Dict[int, Request] = {}
        for _ in range(num_requests):
            request = Request(
                num_input_tokens=num_input_tokens, num_output_tokens=num_output_tokens
            )
            request.decode_done_signal.connect(self._on_complete)
            self.requests[request.id] = request

    def next_request(self) -> Request:
        if not self.requests:
            raise ValueError("self.requests is None")
        first_key = next(iter(self.requests))
        request = self.requests.pop(first_key)
        stime.elapse(1 / self.request_rate)
        request.state = RequestState.LEAVES_CLIENT
        return request

    def has_request(self) -> Request:
        return self.requests

    def _on_complete(self, request: Request):
        # TOBEDONE: record metrics of this request
        if request.state != RequestState.DECODE_DONE:
            raise ValueError("request.state != RequestState.DECODE_DONE")
