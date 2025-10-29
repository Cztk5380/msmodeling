import collections
import logging
import operator
from typing import List, Optional, Tuple

import torch
from torch.fx import Node

from ..pass_base import TensorCastGraphModulePass

from ..utils import stable_topo_sort

logger = logging.getLogger(__name__)

# --- Type Alias for Merge Specification ---
# (grouping_key_args, grouping_key_kwargs, w_idx, bias_idx, split_dim)
OpMergeSpec = Tuple[Tuple[int, ...], int, Optional[int], int]


class MergeLinearPass(TensorCastGraphModulePass):
    """
    FX Pass to find and merge linear-like ops (quant_linear, mm).

    It inserts 'aten.cat' nodes to merge the 'w' and 'bias' inputs.

    It relies on 'node.meta['val']' containing shape information for
    the 'w' tensors to correctly create the output split.

    **Requirements:**
    1. Merges ops that share the same non-weight/non-bias arguments.
    2. For ops with an optional bias, a group is only merged if ALL
       ops in the group have a tensor bias OR all have a None bias.
    """

    def _get_merge_spec(self, node: Node) -> Optional[OpMergeSpec]:
        """
        Returns the merge specification for a given node, if supported.
        """
        # static_quant_linear(x, w, w_scale, w_offset, x_scale, x_offset, bias, out_dtype)
        if node.target == torch.ops.tensor_cast.static_quant_linear.default:
            # Group by: x, w_scale, w_offset, x_scale, x_offset, out_dtype
            grouping_key_args = (0, 2, 3, 4, 5, 7)
            # Concat: w (idx 1) and bias (idx 6)
            w_idx, bias_idx = 1, 6
            split_dim = 1
            return grouping_key_args, w_idx, bias_idx, split_dim

        # aten.mm.default(x, w)
        if node.target == torch.ops.aten.mm.default:
            # Group by: x
            grouping_key_args = (0,)
            # Concat: w (idx 1)
            w_idx, bias_idx = 1, None
            split_dim = 1
            return grouping_key_args, w_idx, bias_idx, split_dim

        # aten.addmm.default(bias, x, w)
        if node.target == torch.ops.aten.addmm.default:
            # Group by: x
            grouping_key_args = (1,)
            # Concat: w (idx 2)
            w_idx, bias_idx = 2, 0
            split_dim = 1
            return grouping_key_args, w_idx, bias_idx, split_dim

        return None

    def __call__(self, graph: torch.fx.GraphModule) -> None:
        logger.debug("Running MergeLinearPass.........")

        # {grouping_key: [list_of_nodes]}
        groups = collections.defaultdict(list)

        # 1. Group all target nodes
        for node in graph.graph.nodes:
            if node.op != "call_function":
                continue

            spec = self._get_merge_spec(node)
            if spec is None:
                continue

            grouping_key_args_idx, *_ = spec
            key_args = tuple(node.args[i] for i in grouping_key_args_idx)
            key = (node.target, key_args)
            groups[key].append(node)

        # 2. Process each group
        for group in groups.values():
            if len(group) <= 1:
                continue

            ref_node = group[0]
            spec = self._get_merge_spec(ref_node)
            assert spec is not None
            _, w_arg_idx, bias_arg_idx, split_dim = spec

            w_nodes: List[Node] = []
            bias_nodes: List[Node] = []
            split_sizes: List[int] = []
            can_merge = True
            has_bias = bias_arg_idx is not None

            # 3. Enforce bias consistency and collect inputs
            all_bias_none = True
            if has_bias:
                is_first_bias_none = ref_node.args[bias_arg_idx] is None
                all_bias_none = is_first_bias_none

                for node in group:
                    current_bias_arg = node.args[bias_arg_idx]
                    current_bias_is_none = current_bias_arg is None

                    if current_bias_is_none != is_first_bias_none:
                        can_merge = False
                        break

                    if not current_bias_is_none:
                        assert isinstance(current_bias_arg, Node)
                        bias_nodes.append(current_bias_arg)

                if not can_merge:
                    continue  # Skip group, mixed bias

            # 4. Collect 'w' nodes and check for shape metadata
            for node in group:
                w_node = node.args[w_arg_idx]
                assert isinstance(w_node, Node), "Expected w_node to be a Node"

                # Need shape meta to create the split
                if "val" not in w_node.meta or not hasattr(w_node.meta["val"], "shape"):
                    can_merge = False
                    break

                w_shape = w_node.meta["val"].shape
                split_sizes.append(w_shape[split_dim])
                w_nodes.append(w_node)

            if not can_merge:
                continue

            # 5. Insert concatenation and merged op nodes
            insertion_point = group[-1]

            # Insert 'w' concatenation
            with graph.graph.inserting_before(insertion_point):
                w_cat_node = graph.graph.create_node(
                    "call_function",
                    torch.ops.aten.cat.default,
                    args=(w_nodes, split_dim),
                )

            # Insert 'bias' concatenation
            bias_cat_node = None
            if has_bias and not all_bias_none:
                with graph.graph.inserting_before(insertion_point):
                    bias_cat_node = graph.graph.create_node(
                        "call_function",
                        torch.ops.aten.cat.default,
                        args=(bias_nodes, -1),
                    )

            # Create the new merged op
            new_args = list(ref_node.args)
            new_kwargs = dict(ref_node.kwargs)

            new_args[w_arg_idx] = w_cat_node
            if has_bias:
                new_args[bias_arg_idx] = bias_cat_node

            with graph.graph.inserting_before(insertion_point):
                merged_node = graph.graph.create_node(
                    "call_function",
                    ref_node.target,
                    args=tuple(new_args),
                    kwargs=new_kwargs,
                )

            # 6. Split the output of the merged op and replace uses
            with graph.graph.inserting_after(merged_node):
                split_node = graph.graph.create_node(
                    "call_function",
                    torch.ops.aten.split_with_sizes.default,
                    args=(merged_node, split_sizes, split_dim),
                )

            for i, old_node in enumerate(group):
                with graph.graph.inserting_after(old_node):
                    getitem_node = graph.graph.create_node(
                        "call_function", operator.getitem, args=(split_node, i)
                    )
                    old_node.replace_all_uses_with(getitem_node)

        # 7. Clean up the graph
        # Make sure the graph is topologically ordered first before DCE
        stable_topo_sort(graph)
        graph.graph.eliminate_dead_code()
        graph.graph.lint()
        graph.recompile()
