from typing import Dict, List

import torch.fx as fx
from torch.fx.subgraph_rewriter import replace_pattern

from ..utils import init_logger

logger = init_logger(__name__)


class PatternManager:
    # A simple pattern manager to register and apply patterns
    # Patterns are applied in levels, lower level patterns are applied first
    # Currently supports up to 3 levels of patterns
    _max_levels = 3
    _leveled_patterns: List[Dict[str, tuple]] = [{} for _ in range(_max_levels)]

    @staticmethod
    def register_pattern(name: str, pattern: tuple, level: int = 0):
        assert level < PatternManager._max_levels, (
            f"Level {level} exceeds max number of levels {PatternManager._max_levels}"
        )
        if name in PatternManager._leveled_patterns[level]:
            logger.warning(f"Pattern '{name}' is already registered, skip.")
        PatternManager._leveled_patterns[level][name] = pattern

    @staticmethod
    def apply_patterns(graph: fx.GraphModule) -> fx.GraphModule:
        """
        Apply registered patterns to the given graph.

        Args:
            graph (Any): The graph to which patterns will be applied.

        Returns:
            Any: The modified graph after applying patterns.
        """
        for level in range(PatternManager._max_levels):
            for name, pattern in PatternManager._leveled_patterns[level].items():
                logger.debug(f"Applying pattern '{name}' of level {level}...")
                source, target = pattern
                source_fx = fx.symbolic_trace(source)
                logger.debug(f"Source pattern FX Graph: {source_fx}")
                replace_pattern(graph, source, target)
        return graph

    @staticmethod
    def has_pattern(pattern_name: str) -> bool:
        for level in range(PatternManager._max_levels):
            if pattern_name in PatternManager._leveled_patterns[level]:
                return True
        return False
