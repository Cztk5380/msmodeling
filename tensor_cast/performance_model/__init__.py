import dataclasses
import hashlib
import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch

from .. import ops  # noqa: F401
from ..device import DeviceProfile
from .utils import bytes_of_elements, bytes_of_tensor, is_view_op, run_once

logger = logging.getLogger(__name__)


class OpInvokeInfo:
    _op_properties_functors = {}

    @dataclasses.dataclass
    class ComputeOps:
        mma_ops: int = 0
        """Number of Matrix-Multiply-Accumulate ops"""
        gp_ops: int = 0
        """Number of General-Purpose ops"""

    @dataclasses.dataclass
    class PerformanceProperties:
        compute_ops: Dict[torch.dtype, "OpInvokeInfo.ComputeOps"] = dataclasses.field(
            default_factory=dict
        )
        memory_read_bytes: int = 0
        """Read-only bytes"""
        memory_write_bytes: int = 0
        """Write-only bytes"""
        memory_readwrite_bytes: int = 0
        """Read-write bytes"""

        def combine(
            self, other: "OpInvokeInfo.PerformanceProperties", compute_only=False
        ):
            for dtype, compute_ops in other.compute_ops.items():
                if dtype not in self.compute_ops:
                    self.compute_ops[dtype] = OpInvokeInfo.ComputeOps()
                self.compute_ops[dtype].mma_ops += compute_ops.mma_ops
                self.compute_ops[dtype].gp_ops += compute_ops.gp_ops
            if not compute_only:
                self.memory_read_bytes += other.memory_read_bytes
                self.memory_write_bytes += other.memory_write_bytes
                self.memory_readwrite_bytes += other.memory_readwrite_bytes

    def __init__(self, func, args, kwargs, out, cache_key=None):
        self.func = func
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.out = out
        self.cache_key = cache_key or self.compute_cache_key()

    @classmethod
    def get_op_properties_functor(cls, op):
        def default_functor(self: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
            """Default functor only counts in the memory accesses"""
            if is_view_op(self.func):
                return OpInvokeInfo.PerformanceProperties()
            run_once(
                self.func,
                logger.warning,
                f"No op properties function defined for {self.func}, "
                f"assuming it is memory-bandwidth bound.",
            )
            return self.get_memory_access_properties()

        if op not in OpInvokeInfo._op_properties_functors:
            return default_functor
        return OpInvokeInfo._op_properties_functors[op]

    @classmethod
    def register_op_properties(cls, op):
        def decorator(functor):
            assert op not in OpInvokeInfo._op_properties_functors, (
                f"Op properties functor for {op} already registered"
            )
            OpInvokeInfo._op_properties_functors[op] = functor
            return functor

        return decorator

    def get_memory_access_properties(
        self,
        exclude_input_ids: Optional[set] = None,
        exclude_output_ids: Optional[set] = None,
    ) -> "OpInvokeInfo.PerformanceProperties":
        """Get memory read/write properties"""
        exclude_input_ids = set() if exclude_input_ids is None else exclude_input_ids
        exclude_output_ids = set() if exclude_output_ids is None else exclude_output_ids
        memory_read_bytes = 0
        memory_write_bytes = 0
        memory_readwrite_bytes = 0
        args_schema = self.func._schema.arguments
        for i, arg in enumerate(itertools.chain(self.args, self.kwargs.values())):
            if i not in exclude_input_ids:
                inputs = arg if isinstance(arg, (list, tuple)) else [arg]
                if inputs and isinstance(inputs[0], torch.Tensor):
                    for tensor in inputs:
                        access_bytes = bytes_of_tensor(tensor)
                        if args_schema[i].is_out:
                            memory_write_bytes += access_bytes
                        elif args_schema[i].is_write:
                            memory_readwrite_bytes += access_bytes
                        else:
                            memory_read_bytes += access_bytes
        out = self.out if isinstance(self.out, (list, tuple)) else [self.out]
        for i, arg in enumerate(out):
            if isinstance(arg, torch.Tensor) and i not in exclude_output_ids:
                access_bytes = bytes_of_tensor(arg)
                memory_write_bytes += access_bytes
        return OpInvokeInfo.PerformanceProperties(
            memory_read_bytes=memory_read_bytes,
            memory_write_bytes=memory_write_bytes,
            memory_readwrite_bytes=memory_readwrite_bytes,
        )

    def get_perf_properties(self) -> "OpInvokeInfo.PerformanceProperties":
        functor = self.get_op_properties_functor(self.func)
        return functor(self)

    def compute_cache_key(self) -> str:
        """
        Compute an efficient cache key based on operation signature and tensor properties.
        This key represents the computational characteristics of the operation.

        Returns:
            A string hash that can be used as a cache key
        """
        key_components = []

        key_components.append(str(self.func))

        def add_tensor_info(t, components):
            if isinstance(t, torch.Tensor):
                components.extend(
                    [
                        str(t.shape),
                        str(t.dtype),
                        str(t.device),
                        str(t.stride()) if not t.is_contiguous() else "contiguous",
                        str(t.requires_grad),
                    ]
                )
            else:
                components.append(str(t))

        for arg in self.args:
            add_tensor_info(arg, key_components)

        if self.kwargs:
            for k, v in sorted(self.kwargs.items()):
                key_components.append(k)
                add_tensor_info(v, key_components)

        # Create hash
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def __repr__(self):
        return f"OpInvokeInfo({self.func}, {self.args}, {self.kwargs}, {self.out})"


class PerformanceModel(ABC):
    """
    Performance model used to estimate the execution time of op invocations
    on a given device.
    """

    @dataclasses.dataclass
    class Result:
        execution_time_s: float
        statistics: Dict[str, Any] = dataclasses.field(default_factory=dict)
        """Misc runtime statistics produced by implementation"""

        def combine(self, other: "PerformanceModel.Result", method: str = "max"):
            if method == "max":
                self.execution_time_s = max(
                    self.execution_time_s, other.execution_time_s
                )
            elif method == "sum":
                self.execution_time_s += other.execution_time_s
            else:
                raise ValueError(
                    f"Unsupported method {method} for combining performance result"
                )
            self.statistics.update(other.statistics)

    class OpClassifier(Protocol):
        @property
        def name(self): ...

        def classify(
            self, event_list: List[Tuple[OpInvokeInfo, "PerformanceModel.Result"]]
        ) -> Dict[str, float]:
            """
            Classify an event list into a breakdown.

            [NOTE: Breakdown from Op Classifier] The semantics of the values are defined by the performance
            models but they should account for a breakdown of sum(values) so that the caller can then compute the
            percentage of each category according to the values.

            :param event_list: Event list of classify
            :return: category name -> value
            """
            ...

    def __init__(self, name, device_profile: DeviceProfile):
        self.name = name
        self.device_profile = device_profile

    @abstractmethod
    def process_op(self, op_invoke_info: OpInvokeInfo) -> "PerformanceModel.Result":
        """
        Estimate the execution time of an op invocation on the given device.
        Returns:
            op execution time in seconds and misc runtime statistics
        """

    def get_classifiers(self) -> List[OpClassifier]:
        return []


class CachingPerformanceModel(PerformanceModel):
    """
    A performance model that caches the results of another performance model.
    """

    def __init__(self, base_model: PerformanceModel):
        super().__init__(base_model.name, base_model.device_profile)
        self._base_model = base_model
        self._cache: Dict[str, PerformanceModel.Result] = {}

    def process_op(self, op_invoke_info: OpInvokeInfo) -> "PerformanceModel.Result":
        if op_invoke_info.cache_key in self._cache:
            return self._cache[op_invoke_info.cache_key]
        result = self._base_model.process_op(op_invoke_info)
        self._cache[op_invoke_info.cache_key] = result
        return result

    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self._base_model.get_classifiers()


@OpInvokeInfo.register_op_properties(torch.ops.aten.bmm.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 2
    mat1 = op_invoke_info.args[0]
    mat2 = op_invoke_info.args[1]
    assert isinstance(mat1, torch.Tensor)
    assert isinstance(mat2, torch.Tensor)
    assert mat1.ndim == 3
    assert mat2.ndim == 3
    b = mat1.size(0)
    m = mat1.size(1)
    k = mat1.size(2)
    n = mat2.size(2)
    assert mat2.size(0) == b
    assert mat2.size(1) == k

    mma_ops = b * m * n * k * 2
    if mma_ops == 0:
        return OpInvokeInfo.PerformanceProperties()

    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[mat1.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[mat1.dtype].mma_ops = mma_ops
    return properties


def _mm_properties_helper(
    op_invoke_info: OpInvokeInfo, mat1, mat2, bias
) -> OpInvokeInfo.PerformanceProperties:
    # Get the logical dimensions of the operation.
    # mat1 is (M, K).
    m = mat1.size(0)
    k = mat1.size(1)
    n = mat2.size(1)

    # Matrix Multiplication: mat1 @ mat2
    # Cost is M * N * K fused multiply-adds (FMAs), which are 2 FLOPs each.
    matmul_ops = m * n * k * 2

    # Bias Addition: ... + bias
    # M * N additions.
    bias_ops = 0
    if bias is not None:
        bias_ops = m * n

    if matmul_ops == 0:
        return OpInvokeInfo.PerformanceProperties()

    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[mat1.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[mat1.dtype].mma_ops = matmul_ops
    if bias is not None:
        compute_ops = properties.compute_ops.setdefault(
            bias.dtype, OpInvokeInfo.ComputeOps()
        )
        compute_ops.gp_ops = bias_ops
        properties.compute_ops[bias.dtype] = compute_ops

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.mm.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 2
    return _mm_properties_helper(
        op_invoke_info, op_invoke_info.args[0], op_invoke_info.args[1], None
    )


def _static_quant_linear_properties_helper(
    op_invoke_info: OpInvokeInfo, x, w, w_offset, bias, is_int4: bool
) -> OpInvokeInfo.PerformanceProperties:
    # Get the logical dimensions of the operation.
    # x is (M, K).
    m = x.size(0)
    k = x.size(1)

    if is_int4:
        if w.size(0) == k:
            n = w.size(1) * 2
        else:
            assert w.size(0) == k // 2
            n = w.size(1)
    else:
        n = w.size(1)

    # Dequantization of weights: dequant(w) if w is int4
    # Here, we suppose HW supports int8 @ int8 but not int8 @ int4 directly.
    # The operation is semantically `(w - w_offset) * w_scale`.
    dequant_ops = 0
    if is_int4:
        if w_offset is not None:
            # K * N subtractions (offset) + K * N multiplications (scale)
            dequant_ops = k * n * 2
        else:
            # K * N multiplications (scale only)
            dequant_ops = k * n

    # Matrix Multiplication: dequant(x) @ dequant(w)
    # Cost is M * N * K fused multiply-adds (FMAs), which are 2 FLOPs each.
    matmul_ops = m * n * k * 2

    # Bias Addition: ... + bias
    # M * N additions.
    bias_ops = 0
    if bias is not None:
        bias_ops = m * n

    if matmul_ops == 0:
        return OpInvokeInfo.PerformanceProperties()

    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[x.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[x.dtype].mma_ops = matmul_ops
    if is_int4:
        # TODO(jgong5): use fp32 flops for int4->int8, should use something more accurate
        compute_ops = properties.compute_ops.setdefault(
            torch.float32, OpInvokeInfo.ComputeOps()
        )
        compute_ops.gp_ops = dequant_ops
        properties.compute_ops[torch.float32] = OpInvokeInfo.ComputeOps()
    if bias is not None:
        compute_ops = properties.compute_ops.setdefault(
            bias.dtype, OpInvokeInfo.ComputeOps()
        )
        compute_ops.gp_ops = bias_ops
        properties.compute_ops[bias.dtype] = compute_ops

    return properties


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.static_quant_linear_int4.default
)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3] if len(op_invoke_info.args) > 3 else None
    bias = op_invoke_info.args[6] if len(op_invoke_info.args) > 6 else None
    return _static_quant_linear_properties_helper(
        op_invoke_info, x, w, w_offset, bias, is_int4=True
    )


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.static_quant_linear.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3] if len(op_invoke_info.args) > 3 else None
    bias = op_invoke_info.args[6] if len(op_invoke_info.args) > 6 else None
    return _static_quant_linear_properties_helper(
        op_invoke_info, x, w, w_offset, bias, is_int4=False
    )


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.fp8_linear.default)
@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.mxfp4_linear.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    bias = op_invoke_info.args[4] if len(op_invoke_info.args) > 4 else None
    return _static_quant_linear_properties_helper(
        op_invoke_info, x, w, None, bias, is_int4=False
    )


@OpInvokeInfo.register_op_properties(torch.ops.aten.embedding.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 2
    weight = op_invoke_info.args[0]
    indices = op_invoke_info.args[1]
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={0})
    properties.memory_read_bytes += (
        bytes_of_tensor(indices, weight.dtype) * weight.shape[-1]
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.index_select.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    input = op_invoke_info.args[0]
    dim = op_invoke_info.args[1]
    index = op_invoke_info.args[2]
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={0})
    properties.memory_read_bytes += (
        bytes_of_tensor(input) * index.numel() / input.shape[dim]
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.reshape_and_cache.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 4
    key = op_invoke_info.args[0]
    value = op_invoke_info.args[1]
    kv_cache = op_invoke_info.args[2]

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={2})
    properties.memory_write_bytes += bytes_of_tensor(
        key, kv_cache.dtype
    ) + bytes_of_tensor(value, kv_cache.dtype)
    return properties


def _attention_properties_helper(
    op_invoke_info: OpInvokeInfo,
    query,
    key,
    seq_lens,
    query_lens,
    softmax_dtype,
) -> OpInvokeInfo.PerformanceProperties:
    hidden_size = query.size(-1)
    head_size = key.size(-1)
    is_vl_attention = False
    if key.ndim == 2 and op_invoke_info.kwargs.get('hf_config'):
        is_vl_attention = True
        vision_config = op_invoke_info.kwargs.get('hf_config').vision_config
        head_size = vision_config.hidden_size // vision_config.num_heads
    # Infer the number of query heads
    # hidden_size = num_q_heads * head_size
    assert hidden_size % head_size == 0, "hidden_size must be divisible by head_size"
    num_q_heads = hidden_size // head_size

    num_tokens_per_seq = query_lens

    # The core computation involves multiplying query tokens for a sequence with all
    # key tokens of that same sequence. We sum this product over all sequences.
    # This gives a measure of the total QK^T and Score*V interactions.
    # E.g., for seq `i`: num_tokens_per_seq[i] * seq_lens[i]
    # We sum this value across the entire batch.
    context_len_product_sum = torch.sum(
        num_tokens_per_seq.to(seq_lens.dtype) * seq_lens
    ).item()

    # 1. First Batched Matrix Multiplication (BMM): Q @ K^T
    # For each query head, this is a sum of (num_tokens_per_seq * seq_len) dot products,
    # where each dot product has `head_size` multiply-adds.
    # Total FMA ops = sum(num_tokens_i * seq_len_i) * num_q_heads * head_size
    # Total FLOPs = FMA_ops * 2
    bmm1_ops = context_len_product_sum * num_q_heads * head_size * 2

    # 2. Softmax
    # This operates on the score matrix. The number of elements is sum(num_tokens_i * seq_len_i) * num_q_heads.
    # Each softmax element (exp, sum, div) is often approximated as ~4 FLOPs.
    softmax_ops = context_len_product_sum * num_q_heads * 4

    # 3. Second Batched Matrix Multiplication (BMM): Scores @ V
    # This has the same computational cost as the first BMM.
    # Total FMA ops = sum(num_tokens_i * seq_len_i) * num_q_heads * head_size
    # Total FLOPs = FMA_ops * 2
    bmm2_ops = context_len_product_sum * num_q_heads * head_size * 2

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={1, 2})
    # KV cache access: query i accesses 2 * seq_len_i slots with kv_head_num * head_size * element_size each.
    assert key.ndim >= 2
    if is_vl_attention:
        properties.memory_read_bytes += torch.sum(
            seq_lens * 2 * bytes_of_elements(key.size(-1), key.dtype)
        ).item()
    else:
        properties.memory_read_bytes += torch.sum(
            seq_lens * 2 * bytes_of_elements(key.size(-1) * key.size(-2), key.dtype)
        ).item()
    compute_ops = properties.compute_ops.setdefault(
        query.dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.mma_ops = bmm1_ops + bmm2_ops
    compute_ops = properties.compute_ops.setdefault(
        softmax_dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.gp_ops = softmax_ops

    return properties


def _default_query_lens_and_seq_lens(query) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = query.size(-2)
    batch_size = query.size(0) if query.ndim == 3 else 1
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    query_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    return query_lens, seq_lens


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.attention.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 8
    query = op_invoke_info.args[0]
    key = op_invoke_info.args[1]
    seq_lens = op_invoke_info.args[6]
    query_lens = op_invoke_info.args[7]
    if query_lens is None or seq_lens is None:
        query_lens, seq_lens = _default_query_lens_and_seq_lens(query)
    return _attention_properties_helper(
        op_invoke_info, query, key, seq_lens, query_lens, query.dtype
    )


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.attention_quant.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 15
    query = op_invoke_info.args[0]
    key = op_invoke_info.args[1]
    seq_lens = op_invoke_info.args[6]
    query_lens = op_invoke_info.args[7]
    out_dtype = op_invoke_info.args[14]
    if query_lens is None or seq_lens is None:
        query_lens, seq_lens = _default_query_lens_and_seq_lens(query)
    if out_dtype is None or out_dtype == query.dtype:
        # use half as default softmax dtype
        softmax_dtype = torch.half
    else:
        softmax_dtype = out_dtype
    properties = _attention_properties_helper(
        op_invoke_info, query, key, seq_lens, query_lens, softmax_dtype
    )

    # According to
    #   `out = dequant(quant(softmax(dequant(Q @ K^T)), attention_prob_scale/offset) @ V)`
    # Calculate additional quantization and dequantization ops

    # 0. Calculate dimensions for quantization ops
    hidden_size = query.size(-1)
    head_size = key.size(-1)
    num_q_heads = hidden_size // head_size
    num_tokens_per_seq = query_lens
    context_len_product_sum = torch.sum(
        num_tokens_per_seq.to(seq_lens.dtype) * seq_lens
    ).item()

    # 1. Dequantization of Q @ K^T (score matrix):
    #    scale multiplication + optional offset subtraction
    # Number of elements: context_len_product_sum * num_q_heads
    # Assuming 2 ops per element (scale + offset) for worst case
    dequant_qkt_ops = context_len_product_sum * num_q_heads * 2

    # 2. Quantization of softmax output (attention probabilities):
    #    scale multiplication + optional offset addition
    # Same number of elements as above
    quant_softmax_ops = context_len_product_sum * num_q_heads * 2

    # 3. Dequantization of final output:
    #    scale multiplication + optional offset subtraction
    # Number of elements: total_tokens * num_q_heads * head_size
    if out_dtype is None or out_dtype == query.dtype:
        dequant_output_ops = 0
    else:
        total_tokens = torch.sum(num_tokens_per_seq).item()
        dequant_output_ops = total_tokens * num_q_heads * head_size * 2

    # Add quantization/dequantization ops to gp_ops
    total_quant_dequant_ops = dequant_qkt_ops + quant_softmax_ops + dequant_output_ops
    compute_ops = properties.compute_ops.setdefault(
        softmax_dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.gp_ops += total_quant_dequant_ops

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.concat_and_cache_mla.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 4
    kv_c_normed = op_invoke_info.args[0]
    k_rot = op_invoke_info.args[1]
    kv_cache = op_invoke_info.args[2]

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={2})
    properties.memory_write_bytes += bytes_of_tensor(
        kv_c_normed, dtype=kv_cache.dtype
    ) + bytes_of_tensor(k_rot, dtype=kv_cache.dtype)
    return properties


_PREDICTIVE_DECODING_THRESHOLD = 5


def _multihead_latent_attention_properties_helper(
    op_invoke_info: OpInvokeInfo,
    softmax_dtype: torch.dtype,
) -> OpInvokeInfo.PerformanceProperties:
    # 1. Argument and Dimension Extraction
    assert len(op_invoke_info.args) >= 12
    (
        q,
        kv_c_normed,
        k_rot,
        kv_cache,
        _,
        query_start_loc,
        seq_lens,
        query_lens,
        W_UK_T,
        W_UV,
        kv_b_proj,
        v_head_dim,
        *_,
    ) = op_invoke_info.args

    # Extract dimensions from input tensors
    num_heads = q.size(1)
    q_head_dim = q.size(2)
    kv_lora_rank = kv_c_normed.size(1)
    qk_rope_head_dim = k_rot.size(1)
    qk_nope_head_dim = q_head_dim - qk_rope_head_dim

    # 2. Separate Prefill and Decode Sequences
    # A sequence is in "decode" if it's processing only one query token.
    # Otherwise, it's in "prefill".
    num_tokens_per_seq = query_lens
    is_decode = num_tokens_per_seq < _PREDICTIVE_DECODING_THRESHOLD
    is_prefill = ~is_decode

    total_fma_ops = 0
    total_gp_ops = 0
    exclude_input_ids = {3, 7, 8, 9}  # kv_cache, W_UK_T, W_UV, kv_b_proj

    # 3. Calculate FLOPs for the Prefill Phase
    num_prefill_tokens = torch.sum(num_tokens_per_seq[is_prefill]).item()
    if num_prefill_tokens > 0:
        assert kv_b_proj is not None
        exclude_input_ids = exclude_input_ids - {9}
        prefill_seq_lens = seq_lens[is_prefill]
        prefill_num_tokens_per_seq = num_tokens_per_seq[is_prefill]

        # Op 1: Project compressed KV: `kv_c_normed @ kv_b_proj`
        # Shapes: (num_prefill_tokens, kv_lora_rank) @ (kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
        kv_proj_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
        prefill_op1_ops = num_prefill_tokens * kv_proj_out_dim * kv_lora_rank * 2

        # For attention ops, we need the sum of (query_len * key_len) over the batch
        prefill_context_sum = torch.sum(
            prefill_num_tokens_per_seq.to(prefill_seq_lens.dtype) * prefill_seq_lens
        ).item()

        # Op 2: Score calculation: `q @ K`
        prefill_op2_ops = prefill_context_sum * num_heads * q_head_dim * 2

        # Op 3: Softmax
        prefill_op3_ops = prefill_context_sum * num_heads * 4

        # Op 4: Score aggregation: `Scores @ V`
        prefill_op4_ops = prefill_context_sum * num_heads * v_head_dim * 2

        total_fma_ops += prefill_op1_ops + prefill_op2_ops + prefill_op4_ops
        total_gp_ops += prefill_op3_ops

    # 4. Calculate FLOPs for the Decode Phase
    num_decode_tokens = torch.sum(num_tokens_per_seq[is_decode]).item()
    if num_decode_tokens > 0:
        assert W_UK_T is not None and W_UV is not None
        exclude_input_ids = exclude_input_ids - {7, 8}
        decode_seq_lens = seq_lens[is_decode]
        decode_num_tokens_per_seq = num_tokens_per_seq[is_decode]

        # The total number of key/value tokens to attend to across all decode sequences
        decode_context_sum = torch.sum(
            decode_num_tokens_per_seq.to(decode_seq_lens.dtype) * decode_seq_lens
        ).item()

        # The decode formula is: softmax(q_nope @ W_UK_T @ k_cache) @ v_cache @ W_UV
        # Op 1: `q_nope @ W_UK_T`
        # Shapes: (num_decode_tokens, num_heads, qk_nope_head_dim) @ (num_heads, qk_nope_head_dim, kv_lora_rank)
        decode_op1_ops = (
            num_decode_tokens * num_heads * qk_nope_head_dim * kv_lora_rank * 2
        )

        # Op 2: `(result_op1, q_rope) @ kv_cache`
        decode_op2_ops = (
            decode_context_sum * num_heads * (kv_lora_rank + qk_rope_head_dim) * 2
        )

        # Op 3: Softmax
        decode_op3_ops = decode_context_sum * num_heads * 4

        # Op 4: `Scores @ v_cache`
        decode_op4_ops = decode_context_sum * num_heads * kv_lora_rank * 2

        # Op 5: `(result_op4) @ W_UV`
        # Shapes: (num_decode_tokens, num_heads, kv_lora_rank) @ (num_heads, kv_lora_rank, v_head_dim)
        decode_op5_ops = num_decode_tokens * num_heads * kv_lora_rank * v_head_dim * 2

        total_fma_ops += (
            decode_op1_ops + decode_op2_ops + decode_op4_ops + decode_op5_ops
        )
        total_gp_ops += decode_op3_ops

    properties = op_invoke_info.get_memory_access_properties(
        exclude_input_ids=exclude_input_ids
    )  # exclude kv_cache

    # Estimate memory read from the KV Cache.
    # Each token in a sequence reads all previous key/value states for that sequence.
    # The size of a cached entry is (kv_lora_rank + qk_rope_head_dim).
    cache_entry_size = bytes_of_elements(kv_cache.size(-1), kv_cache.dtype)

    # `context_len_product_sum` from the previous example can be reused here.
    context_len_product_sum = torch.sum(
        num_tokens_per_seq.to(seq_lens.dtype) * seq_lens
    ).item()
    properties.memory_read_bytes += context_len_product_sum * cache_entry_size

    compute_ops = properties.compute_ops.setdefault(q.dtype, OpInvokeInfo.ComputeOps())
    compute_ops.mma_ops = total_fma_ops
    compute_ops = properties.compute_ops.setdefault(
        softmax_dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.gp_ops = total_gp_ops

    return properties


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.multihead_latent_attention.default
)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    q = op_invoke_info.args[0]
    return _multihead_latent_attention_properties_helper(op_invoke_info, q.dtype)


def _calculate_mla_quant_ops(
    op_invoke_info: OpInvokeInfo,
    num_heads: int,
    q_head_dim: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    out_dtype: torch.dtype,
    q_dtype: torch.dtype,
) -> int:
    """
    Calculate quantization/dequantization ops for MLA quantization.
    Check `torch.ops.tensor_cast.multihead_latent_attention_quant` docstring for details.
    """
    # Separate prefill and decode sequences
    num_tokens_per_seq = query_lens
    is_decode = num_tokens_per_seq < _PREDICTIVE_DECODING_THRESHOLD
    is_prefill = ~is_decode

    total_quant_dequant_ops = 0

    # Calculate quant/dequant ops for prefill phase
    num_prefill_tokens = torch.sum(num_tokens_per_seq[is_prefill]).item()
    if num_prefill_tokens > 0:
        prefill_seq_lens = seq_lens[is_prefill]
        prefill_num_tokens_per_seq = num_tokens_per_seq[is_prefill]
        prefill_context_sum = torch.sum(
            prefill_num_tokens_per_seq.to(prefill_seq_lens.dtype) * prefill_seq_lens
        ).item()

        # 1. Quantization of kv_c_normed @ kv_b_proj output
        # Number of elements: num_prefill_tokens * num_heads * (qk_nope_head_dim + v_head_dim)
        # Each quantization: scale multiplication + optional offset addition (2 ops worst case)
        kv_proj_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
        quant_kv_proj_ops = num_prefill_tokens * kv_proj_out_dim * 2

        # 2. Quantization of attention probabilities (softmax output)
        # Number of elements: prefill_context_sum * num_heads
        quant_attention_prob_ops = prefill_context_sum * num_heads * 2

        total_quant_dequant_ops += quant_kv_proj_ops + quant_attention_prob_ops

    # Calculate quant/dequant ops for decode phase
    num_decode_tokens = torch.sum(num_tokens_per_seq[is_decode]).item()
    if num_decode_tokens > 0:
        decode_seq_lens = seq_lens[is_decode]
        decode_num_tokens_per_seq = num_tokens_per_seq[is_decode]
        decode_context_sum = torch.sum(
            decode_num_tokens_per_seq.to(decode_seq_lens.dtype) * decode_seq_lens
        ).item()

        # 1. Quantization of q @ W_UK_T output
        # Number of elements: num_decode_tokens * num_heads * kv_lora_rank
        quant_qk_ops = num_decode_tokens * num_heads * kv_lora_rank * 2

        # 2. Quantization of attention probabilities (softmax output)
        # Number of elements: decode_context_sum * num_heads
        quant_attention_prob_ops = decode_context_sum * num_heads * 2

        # 3. Quantization of (Scores @ v_cache) output before @ W_UV
        # Number of elements: num_decode_tokens * num_heads * kv_lora_rank
        quant_v_ops = num_decode_tokens * num_heads * kv_lora_rank * 2

        total_quant_dequant_ops += quant_qk_ops + quant_attention_prob_ops + quant_v_ops

    # Optional final output quantization (both prefill and decode)
    # This is only applied if out_dtype is same as q_dtype
    if out_dtype is None or out_dtype == q_dtype:
        total_tokens = torch.sum(num_tokens_per_seq).item()
        # Number of elements: total_tokens * num_heads * v_head_dim
        quant_output_ops = total_tokens * num_heads * v_head_dim * 2
        total_quant_dequant_ops += quant_output_ops

    return total_quant_dequant_ops


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.multihead_latent_attention_quant.default
)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    q = op_invoke_info.args[0]
    kv_c_normed = op_invoke_info.args[1]
    k_rot = op_invoke_info.args[2]
    query_start_loc = op_invoke_info.args[5]
    seq_lens = op_invoke_info.args[6]
    query_lens = op_invoke_info.args[7]
    v_head_dim = op_invoke_info.args[11]
    out_dtype = op_invoke_info.args[-1]

    if out_dtype is None or out_dtype == q.dtype:
        # use half as default softmax dtype
        softmax_dtype = torch.half
    else:
        softmax_dtype = out_dtype

    # Get base properties from helper
    properties = _multihead_latent_attention_properties_helper(
        op_invoke_info, softmax_dtype
    )

    # Extract dimensions (reuse logic instead of duplicating)
    num_heads = q.size(1)
    q_head_dim = q.size(2)
    kv_lora_rank = kv_c_normed.size(1)
    qk_rope_head_dim = k_rot.size(1)
    qk_nope_head_dim = q_head_dim - qk_rope_head_dim

    # Calculate additional quant/dequant ops
    total_quant_dequant_ops = _calculate_mla_quant_ops(
        op_invoke_info,
        num_heads,
        q_head_dim,
        kv_lora_rank,
        qk_nope_head_dim,
        v_head_dim,
        query_start_loc,
        seq_lens,
        query_lens,
        out_dtype,
        q.dtype,
    )

    # Add all quantization/dequantization ops to gp_ops
    compute_ops = properties.compute_ops.setdefault(
        softmax_dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.gp_ops += total_quant_dequant_ops

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    bias = op_invoke_info.args[2]
    assert len(x) == len(w) == len(bias)
    properties = op_invoke_info.get_memory_access_properties()
    for xi, wi, biasi in zip(x, w, bias):
        properties_i = _mm_properties_helper(op_invoke_info, xi, wi, biasi)
        properties.combine(properties_i, compute_only=True)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_quant.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 8
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3]
    bias = op_invoke_info.args[6]
    assert len(x) == len(w) == len(w_offset) == len(bias)

    properties = op_invoke_info.get_memory_access_properties()
    for xi, wi, w_offseti, biasi in zip(x, w, w_offset, bias):
        properties_i = _static_quant_linear_properties_helper(
            op_invoke_info, xi, wi, w_offseti, biasi, is_int4=False
        )
        properties.combine(properties_i, compute_only=True)
    return properties


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.grouped_matmul_quant_int4.default
)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 8
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3]
    bias = op_invoke_info.args[6]
    assert len(x) == len(w) == len(w_offset) == len(bias)

    properties = op_invoke_info.get_memory_access_properties()
    for xi, wi, w_offseti, biasi in zip(x, w, w_offset, bias):
        properties_i = _static_quant_linear_properties_helper(
            op_invoke_info, xi, wi, w_offseti, biasi, is_int4=True
        )
        properties.combine(properties_i, compute_only=True)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_fp8.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 6
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    bias = op_invoke_info.args[4]
    assert len(x) == len(w) == len(bias)
    properties = op_invoke_info.get_memory_access_properties()
    for xi, wi, biasi in zip(x, w, bias):
        properties_i = _static_quant_linear_properties_helper(
            op_invoke_info, xi, wi, None, biasi, is_int4=False
        )
        properties.combine(properties_i, compute_only=True)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_mxfp4.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 6
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    bias = op_invoke_info.args[4]
    assert len(x) == len(w) == len(bias)
    properties = op_invoke_info.get_memory_access_properties()
    for xi, wi, biasi in zip(x, w, bias):
        properties_i = _static_quant_linear_properties_helper(
            op_invoke_info, xi, wi, None, biasi, is_int4=True
        )
        properties.combine(properties_i, compute_only=True)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.addmm.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 3 or len(op_invoke_info.args) == 5
    (input, mat1, mat2) = op_invoke_info.args[:3]

    # mat1:[M,K], mat2:[K,N]
    M, K = mat1.shape
    N = mat2.shape[-1]

    # mat_output = mat1 @ mat2 ; mat_output: [M,N]
    bmm1 = 2 * M * N * K

    if bmm1 == 0:
        return OpInvokeInfo.PerformanceProperties()

    properties = op_invoke_info.get_memory_access_properties()
    compute_ops = properties.compute_ops.setdefault(
        input.dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.mma_ops = bmm1
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.convolution.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    import math

    # op_invoke_info.args length: torch.nn.functional.conv2d is 7, nn.Conv2d is 9
    assert len(op_invoke_info.args) == 7 or len(op_invoke_info.args) == 9
    # Conv2D input:(B, C_in, H, W), weight:(C_out, C_in/groups, K_h, K_w)
    # Conv3D input:(B, C_in, D, H, W), weight:(C_out, C_in/groups, K_d, K_h, K_w)
    (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
    ) = op_invoke_info.args[:6]
    if len(op_invoke_info.args) == 9:
        groups = op_invoke_info.args[8]
    else:
        groups = op_invoke_info.args[6]

    input_shape = input.shape
    weight_shape = weight.shape
    B = input_shape[0]
    C_in = input_shape[1]
    C_out = weight_shape[0]
    if input.dim() == 3:
        # Conv1D
        _, _, L_in = input_shape
        _, _, K_l = weight_shape
        (s_l,) = stride
        (p_l,) = padding
        (d_l,) = dilation

        L_out = math.floor((L_in + 2 * p_l - d_l * (K_l - 1) - 1) / s_l + 1)

        flops_per_output = 2 * (C_in / groups) * K_l
        total_flops = B * C_out * L_out * flops_per_output
        if bias is not None:
            total_flops += B * C_out * L_out

    elif input.dim() == 4:
        # Conv2D
        _, _, H_in, W_in = input_shape
        _, _, K_h, K_w = weight_shape
        s_h, s_w = stride
        p_h, p_w = padding
        d_h, d_w = dilation

        H_out = math.floor((H_in + 2 * p_h - d_h * (K_h - 1) - 1) / s_h + 1)
        W_out = math.floor((W_in + 2 * p_w - d_w * (K_w - 1) - 1) / s_w + 1)

        flops_per_output = 2 * (C_in / groups) * K_h * K_w
        total_flops = B * C_out * H_out * W_out * flops_per_output

        if bias is not None:
            total_flops += B * C_out * H_out * W_out

    elif input.dim() == 5:
        # Conv3D
        _, _, D_in, H_in, W_in = input_shape
        _, _, K_d, K_h, K_w = weight_shape
        s_d, s_h, s_w = stride
        p_d, p_h, p_w = padding
        d_d, d_h, d_w = dilation

        D_out = math.floor((D_in + 2 * p_d - d_d * (K_d - 1) - 1) / s_d + 1)
        H_out = math.floor((H_in + 2 * p_h - d_h * (K_h - 1) - 1) / s_h + 1)
        W_out = math.floor((W_in + 2 * p_w - d_w * (K_w - 1) - 1) / s_w + 1)

        flops_per_output = 2 * (C_in / groups) * K_d * K_h * K_w
        total_flops = B * C_out * D_out * H_out * W_out * flops_per_output

        if bias is not None:
            total_flops += B * C_out * D_out * H_out * W_out

    else:
        raise ValueError(f"Unsupported convolution dimension: {input.dim()}")

    if total_flops == 0:
        return OpInvokeInfo.PerformanceProperties()

    properties = op_invoke_info.get_memory_access_properties()
    compute_ops = properties.compute_ops.setdefault(
        input.dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.mma_ops = total_flops
    return properties
