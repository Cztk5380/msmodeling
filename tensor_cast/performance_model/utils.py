from typing import Optional

import torch


_executed = set()


def run_once(key, fn, *args, **kargs):
    if (key, fn) not in _executed:
        _executed.add((key, fn))
        fn(*args, **kargs)


def is_view_op(op):
    return op.is_view or op == torch.ops.aten._unsafe_view.default


def bytes_of_tensor(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> float:
    """
    Calculates the size of a tensor in bytes.
    This is a centralized function to ensure consistent byte calculation,
    especially for dtypes that might be packed.
    """
    dtype = tensor.dtype if dtype is None else dtype
    return bytes_of_elements(tensor.numel(), dtype)


_bytes_of_dtype = {
    torch.int1: 1 / 8,
    torch.uint1: 1 / 8,
    torch.int2: 1 / 4,
    torch.uint2: 1 / 4,
    torch.int3: 3 / 8,
    torch.uint3: 3 / 8,
    torch.int4: 1 / 2,
    torch.uint4: 1 / 2,
    torch.int5: 5 / 8,
    torch.uint5: 5 / 8,
    torch.int6: 6 / 8,
    torch.uint6: 6 / 8,
    torch.int7: 7 / 8,
    torch.uint7: 7 / 8,
}


def bytes_of_elements(nelements: int, dtype: torch.dtype) -> float:
    """
    Calculates the size of a number of elements in bytes.
    This is a centralized function to ensure consistent byte calculation,
    especially for dtypes that might be packed.
    """
    return _bytes_of_dtype.get(dtype, dtype.itemsize) * nelements
