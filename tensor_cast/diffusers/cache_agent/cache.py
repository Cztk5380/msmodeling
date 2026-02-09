from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Model-agnostic cache schedule/config used by cache strategies."""

    method: str
    blocks_count: int
    steps_count: int
    step_start: int = 0
    step_interval: int = 1
    step_end: int = 10000
    block_start: int = 0
    block_end: int = 10000


class CacheBase(ABC):
    """Base cache class with per-block invocation counters."""

    def __init__(self, config: CacheConfig):
        self._config = config
        self._cur_step = 0
        self._cur_block = 0

    def apply(self, func: callable, *args, **kwargs):
        res = self.apply_imp(func, *args, **kwargs)
        self._counter()
        return res

    @abstractmethod
    def apply_imp(self, func: callable, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _release(self):
        raise NotImplementedError

    def _counter(self):
        self._cur_block += 1
        if self._cur_block == self._config.blocks_count:
            self._cur_step += 1
            self._cur_block = 0
            if self._cur_step == self._config.steps_count:
                self._cur_step = 0
                self._release()
