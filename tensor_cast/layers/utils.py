from typing import Optional

import torch


class ModelWrapperBase(torch.nn.Module):
    def __init__(self, wrapped: Optional[torch.nn.Module]):
        super().__init__()
        self._inner = wrapped

    def unwrap(self) -> torch.nn.Module:
        wrapped = self
        while isinstance(wrapped, ModelWrapperBase):
            wrapped = wrapped._inner
        return wrapped

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            if hasattr(self._inner, item):
                return getattr(self._inner, item)
            raise
