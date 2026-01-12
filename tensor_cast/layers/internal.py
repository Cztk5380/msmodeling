import torch

from .. import ops  # noqa: F401
from .utils import ModelWrapperBase


class RegionMarkerWrapper(ModelWrapperBase):
    def __init__(
        self,
        region_id: int,
        layer: torch.nn.Module,
    ):
        """
        Wrap a layer with region markers.
        Args:
            region_id: The id of the region to mark.
            layer: Original layer instance to wrap
        """
        super().__init__(layer)
        self.region_id = region_id

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        hidden_states = torch.ops.tensor_cast._internal_mark_region_begin(
            hidden_states,
            self.region_id,
        )
        hidden_states = self._inner.forward(*args, **kwargs)
        hidden_states = torch.ops.tensor_cast._internal_mark_region_end(
            hidden_states,
            self.region_id,
        )
        return hidden_states


class CopyLayerWrapper(ModelWrapperBase):
    def __init__(
        self,
        region_id: int,
        layer: torch.nn.Module,
    ):
        """
        Wrap a layer with a copy operation that copies a previously marked region.
        Args:
            region_id: The id of the range to repeat.
            layer: Original layer instance to repeat from
        """
        super().__init__(layer)
        self.region_id = region_id

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        # The following copy operation would be equivalent to:
        # hidden_states = self._inner.forward(*args, **kwargs)
        hidden_states = torch.ops.tensor_cast._internal_copy_region(
            hidden_states,
            self.region_id,
        )
        return hidden_states
