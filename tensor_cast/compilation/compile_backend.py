import functools
import logging
from typing import Callable

import torch
import torch.fx as fx

from .. import config

from . import patterns
from .passes.lift_quant_pass import LiftCombineQuantPass

logger = logging.getLogger(__name__)


class CompilerBackend:
    """
    The compilation backend for 'torch.compile'.
    It is used to process the FX graph and perform custom operation fusing etc.
    """

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        """
        Process the FX graph and perform custom operation fusing.

        Args:
            graph (fx.Graph): The FX graph to be processed.
            example_inputs (optional): Example inputs for the graph.

        Returns:
            fx.Graph: The processed FX graph with custom operation fusing applied.
        """
        logger.debug("Graph before compiling:")
        logger.debug(graph.print_readable(print_output=False))
        self.apply_quantization_passes(graph)
        self.apply_pattern_match_passes(graph)
        logger.debug("Graph after compiling:")
        logger.debug(graph.print_readable(print_output=False))
        return graph

    def apply_quantization_passes(self, graph: fx.GraphModule):
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="quantization_passes",
            log_url=config.compilation.debug.graph_log_url,
        )
        if config.compilation.passes.enable_life_combine_quant:
            GraphTransformObserver(graph, "life_combine_quant_pass").apply_gm_pass(
                LiftCombineQuantPass()
            )

    def apply_pattern_match_passes(self, graph: fx.GraphModule):
        patterns.lazy_init()
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="pattern_match_passes",
            log_url=config.compilation.debug.graph_log_url,
        )
        for i, pattern_match_pass in enumerate(patterns.all_passes):
            GraphTransformObserver(graph, f"pattern_match_pass_{i}").apply_gm_pass(
                pattern_match_pass
            )
