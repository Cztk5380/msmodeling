import functools
import logging
from typing import Callable, Any, Optional

import torch
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized

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
        graph = self.compile(graph, example_inputs)
        return graph

    def compile(
        self,
        gm: fx.GraphModule,
        example_inputs,
        **kwargs,
    ) -> tuple[Callable, Optional[Any]]:
        def compile_inner(fx_graph, inputs):
            logger.debug("Graph before compiling:")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(fx_graph.print_readable(print_output=False))
            self.apply_quantization_passes(fx_graph)
            self.apply_pattern_match_passes(fx_graph)
            self.apply_decompose_auto_functionalized_pass(fx_graph)
            logger.debug("Graph after compiling:")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(fx_graph.print_readable(print_output=False))
            return fx_graph

        # Use the default decomposition table to decompose operators.
        decompositions = (select_decomp_table())
        # Use AOT Autograd to handle the forward compilation.
        return aot_autograd(fw_compiler=compile_inner, decompositions=decompositions)(gm, example_inputs)

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
        
    def apply_decompose_auto_functionalized_pass(self, graph: fx.GraphModule):
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="decompose_auto_functionalized_pass",
            log_url=config.compilation.debug.graph_log_url,
        )
        GraphTransformObserver(graph, "decompose_auto_functionalized").apply_graph_pass(
            decompose_auto_functionalized
        )  
