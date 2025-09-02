import logging
from typing import Any, Dict, Set, Tuple

import torch
import torch.fx as fx

from ..pass_base import TensorCastGraphModulePass

logger = logging.getLogger(__name__)


class LiftCombineQuantPass(TensorCastGraphModulePass):
    _SWAPPABLE_FUNCTION_NAMES: Set[str] = {
        "aten::view",
        "aten::reshape",
    }

    _SWAPPABLE_METHOD_NAMES: Set[str] = {
        "view",
        "reshape",
    }

    """
    An FX graph pass that lifts `tensor_cast.quantize` operations as early
    as possible and combines identical quantization operations into one.
    This makes later fusions possible such as rms_norm+quant fusion.

    The pass works as follows:
    1. Iterates through all nodes in the graph to find `quantize` calls.
    2. For each `quantize` call, it traces its input backward through a chain
       of "swappable" ops (like reshape, view, transpose).
    3. It determines the true, pre-view-change input tensor.
    4. It uses a cache to see if this tensor has already been quantized with the
       same scale/offset.
       - If yes, it reuses the existing quantized tensor.
       - If no, it inserts a new `quantize` op right after the true input tensor
         and adds it to the cache.
    5. It rebuilds the chain of swappable ops on top of the (new or cached)
       quantized tensor.
    6. It replaces the original `quantize` node's uses with the final node of
       the rebuilt chain.
    7. Finally, it removes all the old, now-unused nodes.
    """

    def __call__(self, graph_module: fx.GraphModule) -> None:
        # TODO(jgong5): we should apply AOT dispatch to normalize the graph
        # before going through the remaining passes like this so that
        # the graph would only contain call_function not call_method
        # and the target would always be the op override like
        # torch.ops.tensor_cast.quantize.default
        def is_swappable(node: fx.Node) -> bool:
            """Checks if a node represents a swappable operation."""
            if node.op == "call_function" and hasattr(node.target, "_schema"):
                return (
                    node.target._schema.name.split(".")[0]
                    in self._SWAPPABLE_FUNCTION_NAMES
                )
            if node.op == "call_method":
                return node.target in self._SWAPPABLE_METHOD_NAMES
            return False

        _QUANTIZE_OP = torch.ops.tensor_cast.quantize

        graph = graph_module.graph

        # Cache to store (input_node, scale, offset, dtype) -> new_quantize_node
        # This enables combining identical quantization ops.
        quantize_cache: Dict[Tuple[fx.Node, Any, Any, torch.dtype], fx.Node] = {}
        # Iterate over a copy of nodes, as we'll be modifying the graph
        for node in graph.find_nodes(op="call_function", target=_QUANTIZE_OP):
            original_quant_node = node

            # --- Phase 1: Lifting ---
            # Traverse up the graph through swappable ops to find the earliest
            # possible point to place the quantization op.
            current_input = original_quant_node.args[0]
            swappable_ops_chain = []

            while is_swappable(current_input):
                # As a safety check, we only lift through an op if it has a single user.
                # Lifting an op with multiple users would affect other paths in the graph.
                if len(current_input.users) > 1:
                    break

                swappable_ops_chain.append(current_input)
                current_input = current_input.args[0]

            # --- Phase 2: Combination / Creation ---
            # Get the arguments of the original quantization operation.
            scale = original_quant_node.args[1]
            offset = original_quant_node.args[2]
            out_dtype = original_quant_node.kwargs.get("out_dtype", torch.int8)

            # Create a unique key for the quantization operation.
            cache_key = (current_input, scale, offset, out_dtype)

            if cache_key in quantize_cache:
                # An identical quantization op has already been created. Reuse it.
                lifted_quant_node = quantize_cache[cache_key]
            else:
                # This is the first time we see this quantization. Create a new node.
                with graph.inserting_after(current_input):
                    lifted_quant_node = graph.call_function(
                        _QUANTIZE_OP,
                        args=(current_input, scale, offset),
                        kwargs={"out_dtype": out_dtype},
                    )
                # Add the new node to the cache for future reuse.
                quantize_cache[cache_key] = lifted_quant_node

            # --- Phase 3: Rebuilding ---
            # Re-apply the chain of swappable ops after the new quantization node.
            final_node = lifted_quant_node
            for swappable_node in reversed(swappable_ops_chain):
                with graph.inserting_after(final_node):
                    # Create a new swappable op node with the same arguments,
                    # but with its input connected to the previous node in our new chain.
                    call = getattr(graph, swappable_node.op)
                    final_node = call(
                        swappable_node.target,
                        args=(final_node, *swappable_node.args[1:]),
                        kwargs=swappable_node.kwargs,
                    )

            # --- Phase 4: Replacement ---
            # Replace all uses of the original quantize node with our new chain's output.
            original_quant_node.replace_all_uses_with(final_node)
            # TODO(jgong5): after we enable AOT dispatcher, we should be able to do DCE
            #               and don't need explicit removal of nodes
            graph.erase_node(original_quant_node)
            for swappable_node in swappable_ops_chain:
                graph.erase_node(swappable_node)

        # TODO(jgong5): turn on DCE after enabling AOT dispatcher
        # graph.eliminate_dead_code()
        graph_module.recompile()
