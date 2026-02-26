import dataclasses
from typing import Any, Dict, List, Optional, Set

import torch

from ..device import DeviceProfile
from ..performance_model import OpInvokeInfo
from ..performance_model.op_invoke_info import Region
from .utils import bytes_of_tensor


@dataclasses.dataclass
class OpMemoryProfile:
    op_invoke_info: Optional[OpInvokeInfo]
    usage_before_call_bytes: float
    usage_after_call_bytes: float


@dataclasses.dataclass
class _TensorInfo:
    size_bytes: int
    def_op_idx: int = -1
    use_op_indices: List[int] = dataclasses.field(default_factory=list)
    use_op_indices_by_alias: List[int] = dataclasses.field(default_factory=list)
    """Indirect op use via its alias"""
    last_use_op_idx: int = -1


class MemoryTracker:
    """
    Tracks the memory allocation during the execution of a PyTorch program
    and gives its memory profiles.

    The tracker works in two stages:
    Stage 1 - Recording stage
        Record the input and output tensors of each op execution via multiple calls to `record_op_invocation`.
        We also record the memory consumption during each op execution where op might allocate temporary memory
        as the workspace.
    Stage 2 - Analysis stage
        After the execution of the model, we analyze the def-use and liveness of the input/output tensors,
        identifying what are the model inputs (no def) and outputs (no use). Together with the temporary memory
        usage, we can plan the memory allocations as follows:
        1. All model inputs are allocated before all op execution.
        2. Model inputs and outputs are never freed.
        3. Intermediate tensors (def>=1 and use>=1) are allocated from a memory pool so that buffers are reused.
        4. Intermediate tensors are freed immediately after last use.
        5. Workspace buffers are allocated and freed in the same memory pool as intermediate tensors.
        6. Tensors have aliases. We don't allocate aliased outputs and only free those with all aliases unused later.

    After these two stages, the caller can get a memory profile of the program by calling `get_profile` which gives
    the memory consumption before and after each op invocation.
    """

    def __init__(self, device_profile: DeviceProfile):
        self.device_profile = device_profile
        # A list to store invocation details for each operation.
        self.op_invoke_infos_with_repeat_id: List[tuple[OpInvokeInfo, int]] = []
        # A dictionary to store metadata for each tensor encountered.
        # Key: (tensor id, repeat_id)
        self.tensor_infos: Dict[tuple[int, int], _TensorInfo] = {}
        # Key: (tensor id, repeat_id), Value: id of tensor owning the buffer
        self.alias_info: Dict[tuple[int, int], int] = {}
        # Sets to store the IDs of model input and output tensors.
        self.model_input_tensors: Set[int] = set()
        self.model_output_tensors: Set[int] = set()
        # Internal list to store the calculated memory profiles after analysis.
        self.memory_profiles: List[OpMemoryProfile] = []
        # A flag to ensure analysis is done before retrieving the profile.
        self.is_analyzed: bool = False

    def _extract_tensors(self, data: Any) -> List[torch.Tensor]:
        """A helper function to recursively find all torch.Tensor objects
        within a nested data structure (e.g., list, tuple, dict)."""
        tensors = []
        if isinstance(data, torch.Tensor):
            return [data]
        if isinstance(data, (list, tuple, set)):
            for item in data:
                tensors.extend(self._extract_tensors(item))
        elif isinstance(data, dict):
            for value in data.values():
                tensors.extend(self._extract_tensors(value))
        return tensors

    def _get_real_tensor_id(self, tensor, repeat_id) -> tuple[int, int]:
        return Region.get_tensor_id(tensor, repeat_id)

    def _handle_aliasing(self, op_invoke_info: OpInvokeInfo, repeat_id: int):
        def get_input_id(index):
            input_schema = op_invoke_info.func._schema.arguments[index]
            if index < len(op_invoke_info.args):
                input_tensor = op_invoke_info.args[index]
            else:
                assert input_schema.name in op_invoke_info.kwargs, (
                    f"{input_schema.name} not found in {op_invoke_info.kwargs}"
                )
                input_tensor = op_invoke_info.kwargs[input_schema.name]
            assert isinstance(input_tensor, torch.Tensor), input_tensor
            return self._get_real_tensor_id(input_tensor, repeat_id)

        def set_alias(input_id, output_id):
            if input_id == output_id:
                return
            aliased_tensor_id = self.alias_info.get(input_id)
            if aliased_tensor_id is not None:
                self.alias_info[output_id] = aliased_tensor_id
            else:
                self.alias_info[output_id] = input_id

        outputs = (
            op_invoke_info.out
            if isinstance(op_invoke_info.out, tuple)
            else [op_invoke_info.out]
        )
        for i, output_schema in enumerate(op_invoke_info.func._schema.returns):
            if output_schema.alias_info is None:
                continue
            output = outputs[i]
            output_alias_set = output_schema.alias_info.before_set
            if isinstance(output_schema.real_type, torch.TensorType):
                output_id = self._get_real_tensor_id(output, repeat_id)
                for j, input_schema in enumerate(op_invoke_info.func._schema.arguments):
                    if input_schema.alias_info is None:
                        continue
                    if output_alias_set & input_schema.alias_info.before_set:
                        # found the arg the output aliases
                        input_id = get_input_id(j)
                        set_alias(input_id, output_id)
                        aliased_tensor_id = self.alias_info.get(input_id)
                        if aliased_tensor_id is not None:
                            self.alias_info[output_id] = aliased_tensor_id
                        else:
                            self.alias_info[output_id] = input_id
                        break
            elif isinstance(output_schema.real_type, torch.ListType):
                # for ops like torch.chunk, the input alias_info.after_set is a special `{"*"}` meaning
                # all the tensors in the output alias the input tensor
                assert isinstance(outputs[i], list)
                for j, input_schema in enumerate(op_invoke_info.func._schema.arguments):
                    if input_schema.alias_info is None:
                        continue
                    if input_schema.alias_info.after_set == {"*"}:
                        input_id = get_input_id(j)
                        for output_tensor in outputs[i]:
                            assert isinstance(output_tensor, torch.Tensor), (
                                output_tensor
                            )
                            output_id = self._get_real_tensor_id(
                                output_tensor, repeat_id
                            )
                            set_alias(input_id, output_id)
                        break

    def record_op_invocation(self, op_info_or_region):
        """
        Record the memory usage of an op invocation. Client code calls this method
        multiple times for ops executed by the PyTorch program.
        """
        if isinstance(op_info_or_region, Region):
            region = op_info_or_region
            for op_invoke_info in region.op_invoke_infos:
                self.record_single_op_invocation(op_invoke_info, region.reference_id)
        elif isinstance(op_info_or_region, OpInvokeInfo):
            op_invoke_info = op_info_or_region
            self.record_single_op_invocation(op_invoke_info, 0)
        else:
            raise ValueError(
                f"record_op_invocation failed: Unsupported type: {type(op_info_or_region)}"
            )

    def record_single_op_invocation(
        self, op_invoke_info: OpInvokeInfo, repeat_id: int = 0
    ):
        """
        Record the memory usage of an op invocation. Client code calls this method
        multiple times for ops executed by the PyTorch program.
        """
        op_idx = len(self.op_invoke_infos_with_repeat_id)
        self.op_invoke_infos_with_repeat_id.append((op_invoke_info, repeat_id))

        # Identify all input tensors and record their usage.
        input_tensors = self._extract_tensors(
            op_invoke_info.args
        ) + self._extract_tensors(op_invoke_info.kwargs)
        for tensor in input_tensors:
            tensor_id = self._get_real_tensor_id(tensor, repeat_id)
            if tensor_id not in self.tensor_infos:
                # If a tensor is used before it's defined, it's a model input.
                # We initialize its info here.
                self.tensor_infos[tensor_id] = _TensorInfo(
                    size_bytes=bytes_of_tensor(tensor)
                )
            self.tensor_infos[tensor_id].use_op_indices.append(op_idx)
            aliased_tensor_id = self.alias_info.get(tensor_id)
            if aliased_tensor_id is not None:
                self.tensor_infos[aliased_tensor_id].use_op_indices_by_alias.append(
                    op_idx
                )

        # Identify all output tensors and record their definition site.
        output_tensors = self._extract_tensors(op_invoke_info.out)
        for tensor in output_tensors:
            tensor_id = self._get_real_tensor_id(tensor, repeat_id)
            if tensor_id not in self.tensor_infos:
                self.tensor_infos[tensor_id] = _TensorInfo(
                    size_bytes=bytes_of_tensor(tensor)
                )
                # This op is the one that defines (creates) the tensor.
                self.tensor_infos[tensor_id].def_op_idx = op_idx

        self._handle_aliasing(op_invoke_info, repeat_id)

    def analyze(self):
        """
        Analyze the memory usage of the executed PyTorch program by simulating
        the allocation and deallocation of tensors based on their lifecycles.
        """
        if not self.op_invoke_infos_with_repeat_id:
            self.is_analyzed = True
            return

        # Step 1: Finalize tensor lifecycle info (inputs, outputs, last use).
        for tensor_id, info in self.tensor_infos.items():
            if not info.use_op_indices:
                if tensor_id in self.alias_info:
                    # we treat the aliased tensor as output, not the aliasing ones
                    self.model_output_tensors.add(self.alias_info[tensor_id])
                else:
                    self.model_output_tensors.add(tensor_id)
            else:
                info.last_use_op_idx = max(
                    info.use_op_indices + info.use_op_indices_by_alias
                )

            if info.def_op_idx == -1:
                self.model_input_tensors.add(tensor_id)

        # Step 2: Simulate memory usage over the sequence of operations.
        # Start with memory consumed by model inputs, which are pre-allocated.
        current_memory_usage = sum(
            self.tensor_infos[t_id].size_bytes for t_id in self.model_input_tensors
        )

        for op_idx, (op_info, repeat_id) in enumerate(
            self.op_invoke_infos_with_repeat_id
        ):
            usage_before_call = current_memory_usage

            # Calculate memory allocated for new output tensors. We don't allocate for aliases.
            output_tensor_ids = {
                self._get_real_tensor_id(t, repeat_id)
                for t in self._extract_tensors(op_info.out)
            }

            mem_allocated = sum(
                self.tensor_infos[t_id].size_bytes
                for t_id in output_tensor_ids
                if t_id not in self.alias_info
            )

            # The memory usage after the call includes the newly allocated tensors.
            # TODO(jgong5): This model does not account for temporary workspace memory, which would
            #               require a more sophisticated model to estimate the workspace usage.
            usage_after_call = usage_before_call + mem_allocated

            self.memory_profiles.append(
                OpMemoryProfile(op_info, usage_before_call, usage_after_call)
            )

            # Update memory state for the beginning of the next operation.
            current_memory_usage = usage_after_call

            # Free tensors whose lifecycle ends after this operation.
            input_tensor_ids = {
                self._get_real_tensor_id(t, repeat_id)
                for t in self._extract_tensors(op_info.args)
                + self._extract_tensors(op_info.kwargs)
            }
            mem_freed = 0
            for t_id in input_tensor_ids:
                info = self.tensor_infos[t_id]
                # A tensor is freed if this op is its last use and it's not a model input/output.
                t_id = self.alias_info[t_id] if t_id in self.alias_info else t_id
                if (
                    info.last_use_op_idx == op_idx
                    and t_id not in self.model_input_tensors
                    and t_id not in self.model_output_tensors
                ):
                    mem_freed += info.size_bytes

            # Note: we do not count in the memory fragmentation here.
            current_memory_usage -= mem_freed

        # memory usage after model execution, all intermediate tensors should be freed
        self.memory_profiles.append(
            OpMemoryProfile(None, current_memory_usage, current_memory_usage)
        )

        self.is_analyzed = True

    def get_profile(
        self, initial_mem_usage_bytes: float = 0.0
    ) -> List[OpMemoryProfile]:
        """
        Return the memory profile of the recorded PyTorch program.

        Args:
            initial_mem_usage_bytes: memory usage before executing the PyTorch program.
        """
        if not self.is_analyzed:
            raise RuntimeError("`analyze()` must be called before `get_profile()`.")

        # Adjust the calculated profile with the initial memory offset.
        if initial_mem_usage_bytes == 0:
            return self.memory_profiles
        else:
            return [
                OpMemoryProfile(
                    op_invoke_info=profile.op_invoke_info,
                    usage_before_call_bytes=profile.usage_before_call_bytes
                    + initial_mem_usage_bytes,
                    usage_after_call_bytes=profile.usage_after_call_bytes
                    + initial_mem_usage_bytes,
                )
                for profile in self.memory_profiles
            ]

    def peak_mem_usage(self, initial_mem_usage_bytes: float = 0.0):
        return max(
            [mem_profile.usage_before_call_bytes for mem_profile in self.get_profile()]
        )
