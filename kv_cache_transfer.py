# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from stime import Condition
import stime

logger = stime.get_logger(__name__)


class KVTransfer:
    def __init__(self):
        self.req_id2msg = {}
        self.condition = Condition()

    def check(self, request_id):
        with self.condition:
            return request_id in self.req_id2msg

    def get_msg(self, request_id, key):
        with self.condition:
            msg = self.req_id2msg.get(request_id, None)
            if msg is None:
                return None
            return msg.get(key, None)

    def remove(self, request_id):
        with self.condition:
            self.req_id2msg.pop(request_id, None)

    def send(self, request_id, **msg):
        with self.condition:
            if request_id in self.req_id2msg:
                raise ValueError('request id already exists')
            self.req_id2msg[request_id] = msg


kv_transfer = KVTransfer()