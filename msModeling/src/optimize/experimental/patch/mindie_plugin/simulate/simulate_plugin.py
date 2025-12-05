# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
from typing import List

from msserviceprofiler.modelevalstate.inference.simulate import Simulate, ServiceField
from ..plugin import Plugin
from ....utils.log.logging import logger


def simulate_forward(func, simulate_plugin):
    def wrapper(model_inputs, *args, **kwargs):
        if ServiceField.batch_field:
            try:
                Simulate.predict_and_save()
                result = Simulate.generate_logits(model_inputs.block_tables.shape[0],
                                                  simulate_plugin.model_wrapper.config.vocab_size,
                                                  simulate_plugin.model_wrapper.device)
            except Exception as e:
                logger.error(f"Failed in generate features, error {e}")
                raise e
            return result
        else:
            return func(model_inputs, *args, **kwargs)

    return wrapper


def simulate_sample(func, simulate_plugin):
    def wrapper(*args, **kwargs):
        sampling_output = func(*args, **kwargs)
        if ServiceField.batch_field:
            try:
                Simulate.update_token(simulate_plugin, simulate_plugin.input_metadata, simulate_plugin.cache_ids,
                                      sampling_output)
            except Exception as e:
                raise e
        return sampling_output

    return wrapper


class SimulatePlugin(Plugin):

    def __init__(self, generator_backend, cache_manager, input_manager, output_filter, plugin_data_param, **kwargs):
        self.generator_backend = generator_backend
        self.model_wrapper = self.generator_backend.model_wrapper
        self.cache_manager = cache_manager
        self.input_manager = input_manager
        self.output_filter = output_filter
        self.input_metadata = None
        self.cache_ids = None
        self.plugin_data_param = plugin_data_param
        if isinstance(input_manager.cache_config.eos_token_id, int):
            self.eos_token_id = input_manager.cache_config.eos_token_id
        elif isinstance(input_manager.cache_config.eos_token_id, List):
            self.eos_token_id = input_manager.cache_config.eos_token_id[0]
        else:
            self.eos_token_id = input_manager.cache_config.eos_token_id[0][0]
        self.generator_backend.forward = simulate_forward(self.generator_backend.forward, self)
        self.generator_backend.sample = simulate_sample(self.generator_backend.sample, self)
        try:
            Simulate.init(self)
        except Exception as e:
            logger.error(f"Failed in simulate init. error {e}")
            raise e

    def model_inputs_update(self, model_inputs, input_metadata, cache_ids, input_len_mask, **kwargs):
        try:
            Simulate.generate_features(self, input_metadata, cache_ids)
            self.input_metadata = input_metadata
            self.cache_ids = cache_ids
        except Exception as e:
            logger.error(f"Failed in generate features, error {e}")
            raise e
        return model_inputs, input_len_mask

    def sample_preprocess(self, logits, result, sampling_metadata, input_metadata):
        return logits

    def plugin_cache_clear(self, cache_ids, finish_reason):
        self.input_manager.sampling_cache.clear()

    def plugin_cache_update(self, cache_ids, sampling_output, la_cache_input, is_prefill=False):
        pass

    def plugin_verify(self, sampling_output, cache_ids, result):
        pass
