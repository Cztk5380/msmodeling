import logging
from typing import Any, Callable, Dict, Tuple, List

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter

from ..pass_base import TensorCastGraphModulePass

logger = logging.getLogger(__name__)


class PatternMatchPass(TensorCastGraphModulePass):
    def __init__(self):
        self.pattern_replacements: Dict[
            str, Tuple[Callable[..., Any], Callable[..., Any]]
        ] = {}
        self.pattern_pass: PatternMatcherPass = PatternMatcherPass(
                pass_name="pattern_match_pass"
            )

    def __call__(self, graph: torch.fx.GraphModule) -> None:
        matched_cnt = 0
        while True:
            cnt = self.pattern_pass.apply(graph)
            if cnt == 0:
                break
            matched_cnt += cnt
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"PatternMatchPass replace {matched_cnt} patterns.")
            pattern_idx = 0
            logger.debug("Patterns registered for replacement:")
            for _, pattern_entry in self.pattern_pass.patterns.items():
                for p in pattern_entry:
                    p_str = PatternPrettyPrinter.run(p.pattern)
                    logger.debug(f"Pattern {pattern_idx}: {p_str}")
                    pattern_idx += 1
        return graph

    def uuid(self) -> Any:
        # TODO: hash all registered patterns
        return super().uuid()

    def register_pattern(
        self, name: str, pattern: Callable[..., Any], replacement: Callable[..., Any],
        example_inputs: List[torch.Tensor]
    ):
        if name in self.pattern_replacements:
            raise ValueError(f"Pattern '{name}' is already registered.")

        self.pattern_replacements[name] = (pattern, replacement)
        logger.debug(f"Registering pattern: {name}")
        try:
            pm.register_replacement(pattern, replacement, example_inputs, pm.fwd_only, self.pattern_pass)
            logger.debug(f"Successfully register pattern: {name}")
        except RuntimeError as e:
            if "Duplicate pattern" in str(e):
                logger.warning(f"Pattern '{name}' is already registered. Skipping duplicate registration.")
            else:
                raise e

    @staticmethod
    def has_pattern(self, pattern_name: str) -> bool:
        return pattern_name in self.pattern_replacements
