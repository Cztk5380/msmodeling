import logging

from .dit_block_cache import DiTBlockCache

logger = logging.getLogger(__name__)


_CACHE_METHODS = {
    "dit_block_cache": DiTBlockCache,
}


class CacheAgent:
    """Validate cache config and dispatch to a concrete cache strategy."""

    def __init__(self, config):
        self._config = config
        self._check_config()
        self._cache_method = _CACHE_METHODS[self._config.method](self._config)

    def apply(self, function: callable, *args, **kwargs):
        if not callable(function):
            raise ValueError("Input function must be callable.")

        if (
            self._config.step_start >= self._config.steps_count
            or self._config.step_end == self._config.step_start
        ):
            return function(*args, **kwargs)

        if (
            self._config.block_start >= self._config.blocks_count
            or self._config.block_end == self._config.block_start
        ):
            return function(*args, **kwargs)

        if self._config.step_interval == 1:
            return function(*args, **kwargs)

        return self._cache_method.apply(function, *args, **kwargs)

    def _check_config(self):
        if self._config.method not in _CACHE_METHODS:
            raise ValueError(
                f"Method {self._config.method!r} is not supported. "
                f"Supported methods: {list(_CACHE_METHODS.keys())}."
            )

        if self._config.blocks_count <= 0:
            raise ValueError(
                f"'blocks_count' must be > 0, but got {self._config.blocks_count}."
            )
        if self._config.steps_count <= 0:
            raise ValueError(
                f"'steps_count' must be > 0, but got {self._config.steps_count}."
            )
        if self._config.step_start < 0:
            raise ValueError(
                f"'step_start' must be >= 0, but got {self._config.step_start}."
            )
        if self._config.step_interval <= 0:
            raise ValueError(
                f"'step_interval' must be > 0, but got {self._config.step_interval}."
            )
        if self._config.block_start < 0:
            raise ValueError(
                f"'block_start' must be >= 0, but got {self._config.block_start}."
            )
        if self._config.step_end < self._config.step_start:
            raise ValueError(
                "'step_end' must be >= 'step_start', but got "
                f"{self._config.step_end} and {self._config.step_start}."
            )
        if self._config.block_end < self._config.block_start:
            raise ValueError(
                "'block_end' must be >= 'block_start', but got "
                f"{self._config.block_end} and {self._config.block_start}."
            )

        if self._config.block_end > self._config.blocks_count:
            raise ValueError(
                "'block_end' must be <= 'blocks_count', but got "
                f"{self._config.block_end} and {self._config.blocks_count}."
            )

        if (
            self._config.step_start >= self._config.steps_count
            or self._config.step_end == self._config.step_start
        ):
            logger.debug(
                "'step_start' >= 'steps_count' or 'step_end' == 'step_start', "
                "do not apply cache."
            )
        if (
            self._config.block_start >= self._config.blocks_count
            or self._config.block_end == self._config.block_start
        ):
            logger.debug(
                "'block_start' >= 'blocks_count' or 'block_end' == 'block_start', "
                "do not apply cache."
            )
        if self._config.step_interval == 1:
            logger.debug("'step_interval' is 1, do not apply cache.")
