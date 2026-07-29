"""Editable Triton seed for the fused MoE W8A8 operator.

This is the only source file an optimization agent is expected to change.  It
is intentionally kept as ordinary Python/Triton source so the control plane can
inspect its AST on machines where ``torch`` and ``triton`` are not installed.

The seed is semantically specified for the MetaX C500 stack, but remains
unverified on that hardware until a C500 evaluation reports success.
"""

import torch
import triton
import triton.language as tl


_EXPERT_TILE_ROWS = 128


@triton.jit
def fused_moe_i8_tn_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    moe_weights_ptr,
    expert_ids_ptr,
    out_ptr,
    EM,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_sam,
    stride_sbe,
    stride_sbn,
    stride_mwm,
    stride_om,
    stride_on,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Compute one 128-row expert tile by one output-column tile."""

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # A program never crosses an expert boundary: pid_m is exactly the index
    # into expert_ids, whose entries each describe one 128-row routed tile.
    expert_id = tl.load(expert_ids_ptr + pid_m)
    # The full-suite B tensor spans up to 7.52e9 int8 elements. Cast before
    # multiplying so mcTriton cannot overflow the expert base in int32.
    b_expert_ptr = b_ptr + expert_id.to(tl.int64) * stride_be
    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros(
        (128, BLOCK_SIZE_N), dtype=tl.int32
    )
    for k_tile in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        current_k = k_tile * BLOCK_SIZE_K + offs_k
        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + current_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_expert_ptr
            + current_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )
        a_values = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < EM) & (current_k[None, :] < K),
            other=0,
        )
        b_values = tl.load(
            b_ptrs,
            mask=(current_k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        accumulator = tl.dot(
            a_values,
            b_values,
            acc=accumulator,
            out_dtype=tl.int32,
        )

    scale_a = tl.load(
        scale_a_ptr + offs_m * stride_sam,
        mask=offs_m < EM,
        other=0.0,
    ).to(tl.float32)
    scale_b = tl.load(
        scale_b_ptr + expert_id * stride_sbe + offs_n * stride_sbn,
        mask=offs_n < N,
        other=0.0,
    ).to(tl.float32)
    moe_weights = tl.load(
        moe_weights_ptr + offs_m * stride_mwm,
        mask=offs_m < EM,
        other=0.0,
    ).to(tl.float32)

    result = (
        accumulator.to(tl.float32)
        * scale_a[:, None]
        * scale_b[None, :]
        * moe_weights[:, None]
    )
    out_ptrs = (
        out_ptr
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )
    tl.store(
        out_ptrs,
        result.to(tl.bfloat16),
        mask=(offs_m[:, None] < EM) & (offs_n[None, :] < N),
    )


def _require_tensor(name, value, *, ndim, dtype):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name}.dtype must be {dtype}, got {value.dtype}")


def run_kernel(
    a,
    b_col_major,
    scale_a,
    scale_b,
    moe_weights,
    token_ids,
    expert_ids,
    topk,
    out,
) -> None:
    """Validate the public contract and write the fused result into ``out``.

    ``token_ids`` is validated and retained as a read-only compatibility input.
    The generated test data has already gathered ``a`` and ``scale_a`` into
    routed-row order, so this function deliberately does not gather with it.
    """

    if isinstance(topk, bool) or not isinstance(topk, int):
        raise TypeError("topk must be an int")
    if topk != 8:
        raise ValueError(f"topk must be 8, got {topk}")

    _require_tensor("a", a, ndim=2, dtype=torch.int8)
    _require_tensor("b_col_major", b_col_major, ndim=3, dtype=torch.int8)
    _require_tensor("scale_a", scale_a, ndim=1, dtype=torch.float32)
    _require_tensor("scale_b", scale_b, ndim=2, dtype=torch.float32)
    _require_tensor("moe_weights", moe_weights, ndim=1, dtype=torch.float32)
    _require_tensor("token_ids", token_ids, ndim=1, dtype=torch.int32)
    _require_tensor("expert_ids", expert_ids, ndim=1, dtype=torch.int32)
    _require_tensor("out", out, ndim=2, dtype=torch.bfloat16)

    em, k = a.shape
    num_experts, n, b_k = b_col_major.shape
    if em == 0 or n == 0 or k == 0 or num_experts == 0:
        raise ValueError("all logical dimensions must be non-zero")
    if em % _EXPERT_TILE_ROWS != 0:
        raise ValueError(f"EM must be divisible by {_EXPERT_TILE_ROWS}, got {em}")
    expected_shapes = {
        "b_col_major": (num_experts, n, k),
        "scale_a": (em,),
        "scale_b": (num_experts, n),
        "moe_weights": (em,),
        "token_ids": (em,),
        "expert_ids": (em // _EXPERT_TILE_ROWS,),
        "out": (em, n),
    }
    actual_shapes = {
        "b_col_major": tuple(b_col_major.shape),
        "scale_a": tuple(scale_a.shape),
        "scale_b": tuple(scale_b.shape),
        "moe_weights": tuple(moe_weights.shape),
        "token_ids": tuple(token_ids.shape),
        "expert_ids": tuple(expert_ids.shape),
        "out": tuple(out.shape),
    }
    # Report the explicit K mismatch separately; the generic expected shape is
    # otherwise derived from ``a`` and would obscure the source of the error.
    if b_k != k:
        raise ValueError(f"b_col_major K must match a K ({k}), got {b_k}")
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise ValueError(
                f"{name}.shape must be {expected}, got {actual_shapes[name]}"
            )

    tensors = (
        b_col_major,
        scale_a,
        scale_b,
        moe_weights,
        token_ids,
        expert_ids,
        out,
    )
    if any(tensor.device != a.device for tensor in tensors):
        raise ValueError("all tensors must be on the same device")

    block_size_n = 128
    block_size_k = 128
    grid = (em // _EXPERT_TILE_ROWS, triton.cdiv(n, block_size_n))
    fused_moe_i8_tn_kernel[grid](
        a,
        b_col_major,
        scale_a,
        scale_b,
        moe_weights,
        expert_ids,
        out,
        em,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b_col_major.stride(0),
        b_col_major.stride(1),
        b_col_major.stride(2),
        scale_a.stride(0),
        scale_b.stride(0),
        scale_b.stride(1),
        moe_weights.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        num_warps=16,
        num_stages=2,
    )
