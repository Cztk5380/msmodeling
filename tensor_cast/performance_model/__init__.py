import dataclasses
import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from ..machine import MachineConfig
from .utils import is_view_op, run_once

logger = logging.getLogger(__name__)


class OpInvokeInfo:
    _op_properties_functors = {}

    @dataclasses.dataclass
    class ComputeOps:
        fused_multiply_add_ops: int = 0
        arithmetic_ops: int = 0

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
        network_send_bytes: int = 0
        network_receive_bytes: int = 0

    def __init__(self, func, args, kwargs, out):
        self.func = func
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.out = out

    @classmethod
    def get_op_properties_functor(cls, op):
        def default_functor(self: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
            """Default functor only counts in the memory accesses"""
            if is_view_op(self.func):
                return OpInvokeInfo.PerformanceProperties()
            run_once(
                self.func,
                logger.warning,
                f"No performance estimator defined for {self.func}, "
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
            if isinstance(arg, torch.Tensor) and i not in exclude_input_ids:
                access_bytes = arg.element_size() * arg.nelement()
                if args_schema[i].is_out:
                    memory_write_bytes += access_bytes
                elif args_schema[i].is_write:
                    memory_readwrite_bytes += access_bytes
                else:
                    memory_read_bytes += access_bytes
        out = self.out if isinstance(self.out, (list, tuple)) else [self.out]
        for i, arg in enumerate(out):
            if isinstance(arg, torch.Tensor) and i not in exclude_output_ids:
                access_bytes = arg.element_size() * arg.nelement()
                memory_write_bytes += access_bytes
        return OpInvokeInfo.PerformanceProperties(
            memory_read_bytes=memory_read_bytes,
            memory_write_bytes=memory_write_bytes,
            memory_readwrite_bytes=memory_readwrite_bytes,
        )

    def get_perf_properties(self) -> "OpInvokeInfo.PerformanceProperties":
        functor = self.get_op_properties_functor(self.func)
        return functor(self)

    def __repr__(self):
        return f"OpInvokeInfo({self.func}, {self.args}, {self.kwargs}, {self.out})"


class PerformanceModel(ABC):
    """
    Performance model used to estimate the execution time of op invocations
    on a given machine.
    """

    @dataclasses.dataclass
    class Result:
        execution_time_s: float
        statistics: Dict[str, Any] = dataclasses.field(default_factory=dict)
        """Misc runtime statistics produced by implementation"""

    def __init__(self, name, machine_config: MachineConfig):
        self.name = name
        self.machine_config = machine_config

    @abstractmethod
    def process_op(self, op_invoke_info: OpInvokeInfo) -> "PerformanceModel.Result":
        """
        Estimate the execution time of an op invocation on the given machine.
        Returns:
            op execution time in seconds and misc runtime statistics
        """


@OpInvokeInfo.register_op_properties(torch.ops.aten.bmm.default)
def _bmm_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
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
    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[mat1.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[mat1.dtype].fused_multiply_add_ops = b * m * n * k * 2
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.mm.default)
def _mm_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 2
    mat1 = op_invoke_info.args[0]
    mat2 = op_invoke_info.args[1]
    assert isinstance(mat1, torch.Tensor)
    assert isinstance(mat2, torch.Tensor)
    assert mat1.ndim == 2
    assert mat2.ndim == 2
    m = mat1.size(0)
    k = mat1.size(1)
    n = mat2.size(1)
    assert mat2.size(0) == k
    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[mat1.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[mat1.dtype].fused_multiply_add_ops = m * n * k * 2
    return properties


def _dynamic_quant_linear_properties_helper(
    op_invoke_info: OpInvokeInfo, is_int4: bool
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3] if len(op_invoke_info.args) > 3 else None
    bias = op_invoke_info.args[4] if len(op_invoke_info.args) > 4 else None

    # Get the logical dimensions of the operation.
    # x is (M, K), so its dimensions are not packed.
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

    # Dynamic quant of x: quant(x, x_scale, x_offset)
    dynamic_quant_ops = m * k * 2

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

    # Matrix Multiplication: x @ dequant(w)
    # This is an (M, K) @ (K, N) multiplication.
    # The cost is M * N * K fused multiply-adds (FMAs).
    # Each FMA is 2 FLOPs (1 multiplication, 1 addition).
    matmul_ops = m * n * k * 2

    # Bias Addition: ... + bias
    # This adds the bias vector (broadcasted) to the (M, N) result matrix.
    # This requires M * N additions.
    bias_ops = 0
    if bias is not None:
        bias_ops = m * n

    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[torch.int8] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[torch.int8].fused_multiply_add_ops = matmul_ops
    # TODO: we should use x.dtype instead of float32 here
    #       but this needs to revise the machine model, separating
    #       the FLOPS for FMA and arithmetic ops. Currently we assume
    #       arithmetic ops always map to fp32.
    properties.compute_ops[torch.float32] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[torch.float32].arithmetic_ops = (
        dynamic_quant_ops + dequant_ops + bias_ops
    )

    return properties


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.dynamic_quant_linear_int4.default
)
def _dynamic_quant_linear_int4_properties(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    return _dynamic_quant_linear_properties_helper(op_invoke_info, is_int4=True)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.dynamic_quant_linear.default)
def _dynamic_quant_linear_properties(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    return _dynamic_quant_linear_properties_helper(op_invoke_info, is_int4=False)


def _static_quant_linear_properties_helper(
    op_invoke_info: OpInvokeInfo, is_int4: bool
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3] if len(op_invoke_info.args) > 3 else None
    bias = op_invoke_info.args[6] if len(op_invoke_info.args) > 6 else None

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

    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[x.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[x.dtype].fused_multiply_add_ops = matmul_ops
    # TODO: we should use bias.dtype or out_dtype instead of float32 here
    #       but this needs to revise the machine model, separating
    #       the FLOPS for FMA and arithmetic ops. Currently we assume
    #       arithmetic ops always map to fp32.
    properties.compute_ops[torch.float32] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[torch.float32].arithmetic_ops = dequant_ops + bias_ops

    return properties


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.static_quant_linear_int4.default
)
def _static_quant_linear_int4_properties(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    return _static_quant_linear_properties_helper(op_invoke_info, is_int4=True)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.static_quant_linear.default)
def _static_quant_linear_properties(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    return _static_quant_linear_properties_helper(op_invoke_info, is_int4=False)


@OpInvokeInfo.register_op_properties(torch.ops.aten.embedding.default)
def _embedding_properties(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 2
    weight = op_invoke_info.args[0]
    indices = op_invoke_info.args[1]
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={0})
    properties.memory_read_bytes += (
        indices.nelement() * weight.shape[-1] * weight.element_size()
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.index_select.default)
def _index_select_properties(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    input = op_invoke_info.args[0]
    dim = op_invoke_info.args[1]
    index = op_invoke_info.args[2]
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={0})
    properties.memory_read_bytes += (
        index.nelement() * input.nelement() * input.element_size() / input.shape[dim]
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.reshape_and_cache.default)
def _reshape_and_cache_properties(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 4
    key = op_invoke_info.args[0]
    value = op_invoke_info.args[1]
    kv_cache = op_invoke_info.args[2]

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={2})
    properties.memory_write_bytes += (
        key.nelement() + value.nelement()
    ) * kv_cache.element_size()
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.attention.default)
def _attention_properties(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 7
    query = op_invoke_info.args[0]
    key = op_invoke_info.args[1]
    query_start_loc = op_invoke_info.args[5]
    seq_lens = op_invoke_info.args[6]

    if query_start_loc is None:
        assert seq_lens is None
        seq_len = query.size(-2)
        batch_size = query.size(0) if query.ndim == 3 else 1
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
        query_start_loc = torch.arange(
            0, batch_size * seq_len, seq_len, dtype=torch.long
        )

    hidden_size = query.size(-1)
    # key shape: (*, kv_head_num, head_size)
    kv_head_num = key.size(-2)
    head_size = key.size(-1)

    # Infer the number of query heads
    # hidden_size = num_q_heads * head_size
    assert hidden_size % head_size == 0, "hidden_size must be divisible by head_size"
    num_q_heads = hidden_size // head_size

    num_tokens_per_seq = query_start_loc[1:] - query_start_loc[:-1]

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
    properties.memory_read_bytes += torch.sum(
        seq_lens * 2 * kv_head_num * head_size * key.element_size()
    ).item()
    properties.compute_ops[query.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[query.dtype].fused_multiply_add_ops = bmm1_ops + bmm2_ops
    properties.compute_ops[query.dtype].arithmetic_ops = softmax_ops

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.concat_and_cache_mla.default)
def _concat_and_cache_mla_properties(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 4
    kv_c_normed = op_invoke_info.args[0]
    k_rot = op_invoke_info.args[1]
    kv_cache = op_invoke_info.args[2]

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={2})
    properties.memory_write_bytes += (
        kv_c_normed.nelement() + k_rot.nelement()
    ) * kv_cache.element_size()
    return properties


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.multihead_latent_attention.default
)
def _multihead_latent_attention_properties(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    # 1. Argument and Dimension Extraction
    assert len(op_invoke_info.args) == 11
    (
        q,
        kv_c_normed,
        k_rot,
        kv_cache,
        _,
        query_start_loc,
        seq_lens,
        W_UK_T,
        W_UV,
        kv_b_proj,
        v_head_dim,
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
    num_tokens_per_seq = query_start_loc[1:] - query_start_loc[:-1]
    PREDICTIVE_DECODING_THRESHOLD = 5
    is_decode = num_tokens_per_seq < PREDICTIVE_DECODING_THRESHOLD
    is_prefill = ~is_decode

    total_fma_ops = 0
    total_arith_ops = 0

    # 3. Calculate FLOPs for the Prefill Phase
    num_prefill_tokens = torch.sum(num_tokens_per_seq[is_prefill]).item()
    if num_prefill_tokens > 0:
        assert kv_b_proj is not None
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
        total_arith_ops += prefill_op3_ops

    # 4. Calculate FLOPs for the Decode Phase
    num_decode_tokens = torch.sum(is_decode).item()
    if num_decode_tokens > 0:
        assert W_UK_T is not None and W_UV is not None
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
        total_arith_ops += decode_op3_ops

    properties = op_invoke_info.get_memory_access_properties(
        exclude_input_ids={3}
    )  # exclude kv_cache

    # Estimate memory read from the KV Cache.
    # Each token in a sequence reads all previous key/value states for that sequence.
    # The size of a cached entry is (kv_lora_rank + qk_rope_head_dim).
    cache_entry_size = (kv_lora_rank + qk_rope_head_dim) * kv_cache.element_size()

    # `context_len_product_sum` from the previous example can be reused here.
    context_len_product_sum = torch.sum(
        num_tokens_per_seq.to(seq_lens.dtype) * seq_lens
    ).item()
    properties.memory_read_bytes += context_len_product_sum * cache_entry_size

    properties.compute_ops[q.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[q.dtype].fused_multiply_add_ops = total_fma_ops
    properties.compute_ops[q.dtype].arithmetic_ops = total_arith_ops

    return properties
