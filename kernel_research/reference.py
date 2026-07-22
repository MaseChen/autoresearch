"""Pure NumPy correctness oracle for fused W8A8 MoE GEMM."""

from __future__ import annotations

import hashlib
from typing import Final, Mapping

import numpy as np

from .cases import CaseSpec, DataSet, FIXED_SEED, TILE_ROWS, TOPK, generate_case


RTOL: Final[float] = 2.0e-2
ATOL: Final[float] = 5.0e-3
REQUIRED_MATCH_RATIO: Final[float] = 0.99

_READ_ONLY_FIELDS: Final[tuple[str, ...]] = (
    "a",
    "b_col_major",
    "scale_a",
    "scale_b",
    "moe_weights",
    "token_ids",
    "expert_ids",
)


def bfloat16_round(values: object) -> np.ndarray:
    """Round float32 values to bfloat16 using round-to-nearest-even.

    The returned array uses float32 storage so this works on NumPy builds that
    do not provide a bfloat16 dtype.  Every finite returned value is exactly
    representable as bfloat16.  NaNs are kept as quiet NaNs rather than being
    truncated into infinity when their payload only occupies low bits.
    """

    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    exponent = bits & np.uint32(0x7F800000)
    mantissa = bits & np.uint32(0x007FFFFF)
    nan_mask = (exponent == np.uint32(0x7F800000)) & (mantissa != 0)

    retained_lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x00007FFF) + retained_lsb
    rounded &= np.uint32(0xFFFF0000)

    # Preserve sign and upper payload bits, force the quiet-NaN bit, and avoid
    # a tiny-payload NaN becoming infinity after truncation.
    quiet_nan = (bits & np.uint32(0xFFFF0000)) | np.uint32(0x00400000)
    rounded = np.where(nan_mask, quiet_nan, rounded).astype(np.uint32, copy=False)
    return rounded.view(np.float32)


# Descriptive alias used in docs and tests.
software_bfloat16 = bfloat16_round


def _require_dtype(name: str, value: np.ndarray, expected: np.dtype[object]) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != expected:
        raise TypeError(f"{name} must have dtype {expected}, got {value.dtype}")


def _validate_inputs(
    a: np.ndarray,
    b_col_major: np.ndarray,
    scale_a: np.ndarray,
    scale_b: np.ndarray,
    moe_weights: np.ndarray,
    token_ids: np.ndarray,
    expert_ids: np.ndarray,
    topk: int,
    out: np.ndarray | None,
) -> tuple[int, int, int, int]:
    _require_dtype("a", a, np.dtype(np.int8))
    _require_dtype("b_col_major", b_col_major, np.dtype(np.int8))
    _require_dtype("scale_a", scale_a, np.dtype(np.float32))
    _require_dtype("scale_b", scale_b, np.dtype(np.float32))
    _require_dtype("moe_weights", moe_weights, np.dtype(np.float32))
    _require_dtype("token_ids", token_ids, np.dtype(np.int32))
    _require_dtype("expert_ids", expert_ids, np.dtype(np.int32))

    if a.ndim != 2:
        raise ValueError("a must have shape (EM, K)")
    if b_col_major.ndim != 3:
        raise ValueError("b_col_major must have shape (num_experts, N, K)")
    em, k = a.shape
    num_experts, n, b_k = b_col_major.shape
    if em == 0 or n == 0 or k == 0 or num_experts == 0:
        raise ValueError("all operator dimensions must be positive")
    if k != b_k:
        raise ValueError(f"a K={k} does not match b_col_major K={b_k}")
    if em % TILE_ROWS:
        raise ValueError(f"EM must be a multiple of {TILE_ROWS}")
    if isinstance(topk, (bool, np.bool_)) or not isinstance(topk, (int, np.integer)):
        raise TypeError("topk must be an integer")
    if int(topk) != TOPK:
        raise ValueError(f"topk must be {TOPK}")
    if em % int(topk):
        raise ValueError("EM must be divisible by topk")

    expected_shapes = {
        "scale_a": (em,),
        "scale_b": (num_experts, n),
        "moe_weights": (em,),
        "token_ids": (em,),
        "expert_ids": (em // TILE_ROWS,),
    }
    actual_values = {
        "scale_a": scale_a,
        "scale_b": scale_b,
        "moe_weights": moe_weights,
        "token_ids": token_ids,
        "expert_ids": expert_ids,
    }
    for name, expected_shape in expected_shapes.items():
        if actual_values[name].shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {actual_values[name].shape}"
            )

    if np.any(expert_ids < 0) or np.any(expert_ids >= num_experts):
        raise ValueError("expert_ids contains an out-of-range expert")
    if np.any(token_ids < 0):
        raise ValueError("token_ids must be non-negative")

    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError("out must be a NumPy array")
        if out.shape != (em, n):
            raise ValueError(f"out must have shape {(em, n)}, got {out.shape}")
        if out.dtype != np.float32 and out.dtype.name != "bfloat16":
            raise TypeError("out must use float32 emulation storage or bfloat16")
        if not out.flags.writeable:
            raise ValueError("out must be writeable")

    return em, n, k, num_experts


def validate_dataset(dataset: DataSet) -> None:
    """Validate shapes, dtypes, routing ids, and the attached CaseSpec."""

    if not isinstance(dataset, DataSet):
        raise TypeError("dataset must be a DataSet")
    em, n, k, num_experts = _validate_inputs(
        dataset.a,
        dataset.b_col_major,
        dataset.scale_a,
        dataset.scale_b,
        dataset.moe_weights,
        dataset.token_ids,
        dataset.expert_ids,
        dataset.topk,
        dataset.out,
    )
    actual = (em, n, k, num_experts, int(dataset.topk))
    declared = (
        dataset.spec.em,
        dataset.spec.n,
        dataset.spec.k,
        dataset.spec.num_experts,
        dataset.spec.topk,
    )
    if actual != declared:
        raise ValueError(f"dataset arrays {actual} do not match CaseSpec {declared}")


def snapshot_inputs(dataset: DataSet) -> dict[str, str]:
    """Return stable fingerprints for every contractually read-only input."""

    validate_dataset(dataset)
    fingerprints: dict[str, str] = {}
    for name in _READ_ONLY_FIELDS:
        value = getattr(dataset, name)
        digest = hashlib.sha256()
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
        fingerprints[name] = digest.hexdigest()
    return fingerprints


def inputs_unchanged(dataset: DataSet, snapshot: Mapping[str, str]) -> bool:
    """Check that all non-output arrays still match a prior snapshot."""

    if set(snapshot) != set(_READ_ONLY_FIELDS):
        return False
    return snapshot_inputs(dataset) == dict(snapshot)


def fused_moe_i8_tn_reference(
    a: np.ndarray,
    b_col_major: np.ndarray,
    scale_a: np.ndarray,
    scale_b: np.ndarray,
    moe_weights: np.ndarray,
    token_ids: np.ndarray,
    expert_ids: np.ndarray,
    topk: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the operator by 128-row expert tiles.

    ``token_ids`` is validated but intentionally not used for gathering: ``a``
    and ``scale_a`` are already routed.  Dot products use int32 inputs and an
    int32 accumulator.  Scaling then happens in float32 before software bf16
    rounding.  Only ``out`` is ever written.
    """

    em, n, _, _ = _validate_inputs(
        a,
        b_col_major,
        scale_a,
        scale_b,
        moe_weights,
        token_ids,
        expert_ids,
        topk,
        out,
    )
    result = np.empty((em, n), dtype=np.float32) if out is None else out

    for tile_id, expert_value in enumerate(expert_ids):
        row_start = tile_id * TILE_ROWS
        rows = slice(row_start, row_start + TILE_ROWS)
        expert = int(expert_value)

        accumulator = np.matmul(
            a[rows].astype(np.int32),
            b_col_major[expert].astype(np.int32).T,
        )
        scaled = accumulator.astype(np.float32)
        scaled *= scale_a[rows, None]
        scaled *= scale_b[expert, None, :]
        scaled *= moe_weights[rows, None]
        result[rows] = bfloat16_round(scaled)

    return result


def fused_moe_reference(dataset: DataSet, *, out: np.ndarray | None = None) -> np.ndarray:
    """DataSet-oriented wrapper around :func:`fused_moe_i8_tn_reference`."""

    validate_dataset(dataset)
    return fused_moe_i8_tn_reference(
        dataset.a,
        dataset.b_col_major,
        dataset.scale_a,
        dataset.scale_b,
        dataset.moe_weights,
        dataset.token_ids,
        dataset.expert_ids,
        dataset.topk,
        out=out,
    )


def check_precision(
    actual: object,
    expected: object,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
    required_ratio: float = REQUIRED_MATCH_RATIO,
) -> tuple[bool, float]:
    """Apply the contest's element-wise isclose and matched-ratio criterion."""

    actual_array = np.asarray(actual, dtype=np.float32)
    expected_array = np.asarray(expected, dtype=np.float32)
    if actual_array.shape != expected_array.shape:
        raise ValueError(
            f"precision inputs must have equal shapes, got "
            f"{actual_array.shape} and {expected_array.shape}"
        )
    if actual_array.size == 0:
        raise ValueError("precision inputs must not be empty")
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if not 0.0 <= required_ratio <= 1.0:
        raise ValueError("required_ratio must be in [0, 1]")

    close = np.isclose(actual_array, expected_array, rtol=rtol, atol=atol)
    ratio = float(np.count_nonzero(close) / close.size)
    return ratio >= required_ratio, ratio


def reference_self_check() -> dict[str, object]:
    """Run a tiny deterministic oracle check for the mock control plane."""

    spec = CaseSpec("reference_self_check", 128, 3, 4, 2, "uniform")
    first = generate_case(spec, seed=FIXED_SEED)
    second = generate_case(spec, seed=FIXED_SEED)
    deterministic = all(
        np.array_equal(getattr(first, name), getattr(second, name))
        for name in _READ_ONLY_FIELDS
    )
    before = snapshot_inputs(first)
    output = fused_moe_reference(first)
    readonly = inputs_unchanged(first, before)

    expert = int(first.expert_ids[0])
    manual_accumulator = int(
        np.dot(first.a[0].astype(np.int32), first.b_col_major[expert, 0].astype(np.int32))
    )
    manual = bfloat16_round(
        np.float32(manual_accumulator)
        * first.scale_a[0]
        * first.scale_b[expert, 0]
        * first.moe_weights[0]
    )
    formula = bool(output[0, 0] == manual)
    precision_passed, ratio = check_precision(output, output.copy())
    ok = bool(deterministic and readonly and formula and precision_passed)
    return {
        "ok": ok,
        "deterministic": bool(deterministic),
        "inputs_unchanged": bool(readonly),
        "formula_checked": formula,
        "precision_passed": bool(precision_passed),
        "matched_ratio": ratio,
        "seed": FIXED_SEED,
    }


__all__ = [
    "ATOL",
    "REQUIRED_MATCH_RATIO",
    "RTOL",
    "bfloat16_round",
    "check_precision",
    "fused_moe_i8_tn_reference",
    "fused_moe_reference",
    "inputs_unchanged",
    "reference_self_check",
    "snapshot_inputs",
    "software_bfloat16",
    "validate_dataset",
]
