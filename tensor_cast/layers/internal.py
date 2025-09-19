import torch

from .. import ops  # noqa: F401
from ..model_config import RepetitiveRange
from .utils import ModelWrapperBase


class RepetitiveLayerWrapper(ModelWrapperBase):
    def __init__(
        self,
        repetitive_range: RepetitiveRange,
        range_id: int,
        layer: torch.nn.Module,
        layer_idx,
    ):
        """
        Args:
            repetitive_range: Describe a range of the layers for repeating
            range_id: The id of the range to repeat.
            layer: Original layer instance to repeat from
            layer_idx: The index of the layer within the repetitive range
        """
        super().__init__(layer)
        self.repetitive_range = repetitive_range
        self.range_id = range_id
        self.layer_idx = layer_idx

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        if self.repetitive_range.start == self.layer_idx:
            hidden_states = torch.ops.tensor_cast._internal_repeat_marker_begin(
                hidden_states,
                self.range_id,
                self.repetitive_range.repeats,
            )
        hidden_states = self._inner.forward(*args, **kwargs)
        if self.repetitive_range.stop == self.layer_idx + 1:
            hidden_states = torch.ops.tensor_cast._internal_repeat_marker_end(
                hidden_states,
                self.range_id,
            )
        return hidden_states
