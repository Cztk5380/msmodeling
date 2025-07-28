# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import Dict, List
import threading
import logging

import config
from engine import Engine
from instance import DecodeInstanceLoadBalancer, PrefillInstaceLoadBalacer
from request import Request, RequestState
import stime

logger = stime.getLogger(__name__)


class Serving:
    """
    The entrypoint of inference request serving.

    Requests could come from either the client side (an initial request) or some server instance such as 
    Prefill instance which has completed prefill and wants to hand over the request to the Decode instance.
    Serving is responsible for picking the right server instaces to dispatch to according
    to a pre-defined policy.

    The overall request serving flow looks liker below:
    Requests are firstly dispatched to a server instance (P or D), then the instance dispatches the requests to
    an Engine which corresponds to a Data-Parallel partition. Then the Engine batches on the incoming Requests.
    
    """

    def __init__(self, prefill_instances: List[Engine], decode_instances: List[Engine]):
        # TOBEDONEL use InstaceGroup to group these prefill and decode instances, pass InstanceGroup to Serving
        self.requests: Dict[int, Request] = {}
        self.requests_condition = stime.Condition()
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances

        # TOBEDONE: check type of prefill_instance, except List of Instance but got List of Engine
        self.prefill_balancer = PrefillInstaceLoadBalacer(prefill_instances)
        # TOBEDONE: same to prefill
        self.decode_balancer = DecodeInstanceLoadBalancer(decode_instances)

    def serve(self, request: Request):
        """Handle the request from the client side"""
        if request.state != RequestState.LEAVES_CLIENT:
            raise ValueError("request.state != RequestState.LEAVES_CLIENT")
        request.state = RequestState.ARRIVES_SERVER
        logger.debug("Start serving %s", request)
        with self.requests_condition:
            if request.id in self.requests:
                raise ValueError("request.id in self.requests")
            # TOBEDONE: stop serving new requests if concurrency
            #       is already reached.
            self.requests[request.id] = request

        request.decode_done_signal.connect(self._complete_serve)
        request.prefill_done_signal.connect(self._continue_serve)

        # Assume P/D disaggregation now and hard-coer the dispatch policy
        # to dispatch to prefill instance first.
        # TOBEDONE: add more dispatch policy, such as dispatch to D first and
        #       aggregated P/D
        if config.PD_DEPLOYMENT_POLICY != config.PdDeploymentPolicy.DISAGGREGRATE:
            raise ValueError("config.pd_deployment_policy != config.PdDeploymentPolicy.DISAGGREGRATE")
        prefill_instance = self.prefill_balancer.select(request)
        prefill_instance.handle(request)
    
    def join(self):
        """Wait for all the requests to complete, i.e., self.requests is empty"""
        with self.requests_condition:
            self.requests_condition.wait_for(lambda: len(self.requests) == 0)

    def _continue_serve(self, request: Request):
        """Continue serving"""
        logger.debug("Continue serving %s", request)
        with self.requests_condition:
            if request.id not in self.requests:
                raise ValueError("request.id not in self.requests")
        if request.state == RequestState.DECODE_DONE:
            # EOS after prefill
            self._complete_serve(request)

        if request.state != RequestState.PREFILL_DONE:
            raise ValueError("In continue serving: request.state shoulf be PREFILL_DONE, but get %s", request.state)
        decode_instance = self.decode_balancer.select(request)
        decode_instance.handle(request)

    def _complete_serve(self, request: Request):
        """Completed serving"""
        logger.debug("Completed serving %s", request)
        with self.requests_condition:
            if request.id not in self.requests:
                raise ValueError("request.id not in self.requests")
        if request.state != RequestState.DECODE_DONE:
            raise ValueError("request.state != RequestState.DECODE_DONE")

        # We should return the result to the client, but we do not simulate it here
        with self.requests_condition:
            self.requests.pop(request.id)
            self.requests_condition.notify_all()
