from typing import Callable

import torch.fx as fx

from ..utils import init_logger

from .pattern_manager import PatternManager

logger = init_logger(__name__)


class CompilerBackend:
    """
    The compilation backend for 'torch.compile'.
    It is used to process the FX graph and perform custom operation fusing.
    """

    def __init__(self):
        self.graph = None

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        """
        Process the FX graph and perform custom operation fusing.

        Args:
            graph (fx.Graph): The FX graph to be processed.
            example_inputs (optional): Example inputs for the graph.

        Returns:
            fx.Graph: The processed FX graph with custom operation fusing applied.
        """
        logger.debug("Original FX Graph: ")
        logger.debug(graph.graph)
        # Apply registered patterns to the graph
        PatternManager.apply_patterns(graph)
        logger.debug("Processed FX Graph: ")
        logger.debug(graph.graph)
        self.graph = graph
        return graph

    def get_graph(self):
        return self.graph
