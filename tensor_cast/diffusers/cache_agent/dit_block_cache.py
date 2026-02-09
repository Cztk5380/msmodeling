import logging
from typing import List, Tuple

from .cache import CacheBase

logger = logging.getLogger(__name__)


class DiTBlockCache(CacheBase):
    """DiT block-cache strategy for simulation."""

    def __init__(self, config):
        super().__init__(config)
        self._cache = [None] * 2
        self._time_cache = {}
        self._output_count = 1

    def apply_imp(self, func: callable, *args, **kwargs):
        if "hidden_states" not in kwargs:
            raise ValueError("[DiTBlockCache]: Cannot find 'hidden_states' in kwargs.")

        hidden_states = kwargs.pop("hidden_states")
        if hidden_states is None:
            raise ValueError("[DiTBlockCache]: Input 'hidden_states' is None.")

        if not self._use_cache():
            if "encoder_hidden_states" not in kwargs:
                encoder_hidden_states = None
                res = func(hidden_states, *args, **kwargs)
            else:
                encoder_hidden_states = kwargs.pop("encoder_hidden_states")
                res = func(hidden_states, encoder_hidden_states, *args, **kwargs)
            self._update_cache(res, hidden_states, encoder_hidden_states)
            return res

        delta = self._get_cache()

        if self._output_count == 2:
            if "encoder_hidden_states" not in kwargs:
                raise ValueError(
                    "[DiTBlockCache] 'encoder_hidden_states' is required when cache output count is 2."
                )
            encoder_hidden_states = kwargs.pop("encoder_hidden_states")
            if self._cur_block == self._config.block_start:
                return hidden_states + delta[0], encoder_hidden_states + delta[1]
            return hidden_states, encoder_hidden_states

        if self._cur_block == self._config.block_start:
            return hidden_states + delta[0]
        return hidden_states

    def _use_cache(self) -> bool:
        if self._cur_step < self._config.step_start:
            return False
        if self._cur_step > self._config.step_end:
            return False

        diftime = self._cur_step - self._config.step_start
        if diftime not in self._time_cache:
            self._time_cache[diftime] = diftime % self._config.step_interval == 0

        if self._time_cache[diftime]:
            return False
        if (
            self._cur_block < self._config.block_start
            or self._cur_block >= self._config.block_end
        ):
            return False
        return True

    def _get_cache(self):
        logger.debug(
            "[DiTBlockCache] step: %d block: %d reuse cache.",
            self._cur_step,
            self._cur_block,
        )
        if self._cur_block == self._config.block_start:
            return self._cache
        return [0, 0]

    def _update_cache(self, res, ori_hidden_states, ori_encoder_hidden_states):
        diftime = self._cur_step - self._config.step_start
        if not (
            self._cur_step >= self._config.step_start
            and self._time_cache.get(diftime, False)
        ):
            return
        if self._cur_step >= self._config.step_end:
            return

        self._output_count = len(res) if isinstance(res, (List, Tuple)) else 1
        if self._output_count > 2 or self._output_count < 1:
            raise RuntimeError(
                f"[DiTBlockCache] The output count must be 1 or 2, but got {self._output_count}."
            )

        if self._cur_block == self._config.block_start:
            logger.debug(
                "[DiTBlockCache] step: %d block: %d update cache begin.",
                self._cur_step,
                self._cur_block,
            )
            # Temporarily store the "range input"; later we'll convert it into delta.
            self._cache = [ori_hidden_states, ori_encoder_hidden_states]
            return

        if self._cur_block != (self._config.block_end - 1):
            return

        logger.debug(
            "[DiTBlockCache] step: %d block: %d update cache end.",
            self._cur_step,
            self._cur_block,
        )

        if self._output_count == 2:
            hidden_states, encoder_hidden_states = res
            if hidden_states is None or encoder_hidden_states is None:
                raise RuntimeError("[DiTBlockCache] Cache function output is None.")
            if ori_encoder_hidden_states is None:
                raise ValueError(
                    "[DiTBlockCache] 'encoder_hidden_states' is required when output count is 2."
                )
            self._cache[0] = hidden_states - self._cache[0]
            self._cache[1] = encoder_hidden_states - self._cache[1]
            return

        if res is None:
            raise RuntimeError("[DiTBlockCache] Cache function output is None.")
        self._cache[0] = res - self._cache[0]

    def _release(self):
        self._cache = [None] * 2
        self._time_cache = {}
