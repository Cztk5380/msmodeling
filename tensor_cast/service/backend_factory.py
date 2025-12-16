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

from tensor_cast.service.mindie_backend import MindIEAggBackend


class StrategyFactory:
    _frameworks_cls = {
        MindIEAggBackend.name: MindIEAggBackend,
    }

    @staticmethod
    def create_backend(backend_name: str, args, model):
        backend_name = backend_name.lower()
        backend_name += "_disaggreation" if args.disaggregation else "_aggregation"
        if backend_name not in StrategyFactory._frameworks_cls.keys():
            raise ValueError(f"Unsupported backend: {backend_name}")

        framework = StrategyFactory._frameworks_cls[backend_name]()
        if model:
            framework.initialize(args, model)
        return framework
