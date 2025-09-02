import logging
from typing import Any, Callable, Dict, Tuple

import torch
from torch.fx.subgraph_rewriter import replace_pattern

from ..pass_base import TensorCastGraphModulePass

logger = logging.getLogger(__name__)


class PatternMatchPass(TensorCastGraphModulePass):
    def __init__(self):
        self.pattern_replacements: Dict[
            str, Tuple[Callable[..., Any], Callable[..., Any]]
        ] = {}

    def __call__(self, graph: torch.fx.GraphModule) -> None:
        for name, (pattern, replacement) in self.pattern_replacements.items():
            logger.debug("Applying pattern '%s'...", name)
            replace_pattern(graph, pattern, replacement)
        return graph

    def uuid(self) -> Any:
        # TODO: hash all registered patterns
        return super().uuid()

    def register_pattern(
        self, name: str, pattern: Callable[..., Any], replacement: Callable[..., Any]
    ):
        if name in self.pattern_replacements:
            raise ValueError(f"Pattern '{name}' is already registered.")

        self.pattern_replacements[name] = (pattern, replacement)

    @staticmethod
    def has_pattern(self, pattern_name: str) -> bool:
        return pattern_name in self.pattern_replacements
