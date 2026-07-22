"""Deterministic NumPy fixtures for the fused MoE operator.

The full C500 cases are intentionally represented as metadata first.  Calling
``generate_case`` never changes their dimensions or silently substitutes a
smaller case; callers that cannot allocate a full case receive ``MemoryError``
from NumPy with an estimate of the requested footprint added for context.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Final, Literal, Mapping

import numpy as np


FIXED_SEED: Final[int] = 20260721
DEFAULT_SEED: Final[int] = FIXED_SEED
TILE_ROWS: Final[int] = 128
TOPK: Final[int] = 8
ZIPF_ALPHA: Final[float] = 1.2

ExpertDistribution = Literal["uniform", "zipf"]


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """Shape and routing distribution for one benchmark case."""

    name: str
    em: int
    n: int
    k: int
    num_experts: int
    expert_dist: ExpertDistribution = "uniform"
    topk: int = TOPK

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("case name must not be empty")
        for field_name in ("em", "n", "k", "num_experts"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.em % TILE_ROWS:
            raise ValueError(f"em must be a multiple of {TILE_ROWS}")
        if self.em % self.topk:
            raise ValueError("em must be divisible by topk")
        if self.topk != TOPK:
            raise ValueError(f"topk must be {TOPK}")
        if self.expert_dist not in {"uniform", "zipf"}:
            raise ValueError("expert_dist must be 'uniform' or 'zipf'")

    @property
    def tile_count(self) -> int:
        return self.em // TILE_ROWS

    @property
    def num_tokens(self) -> int:
        return self.em // self.topk

    # Shape aliases are useful when comparing a CaseSpec with the contest
    # notation without weakening the conventional lower-case Python API.
    @property
    def EM(self) -> int:  # noqa: N802 - intentional mathematical alias
        return self.em

    @property
    def N(self) -> int:  # noqa: N802 - intentional mathematical alias
        return self.n

    @property
    def K(self) -> int:  # noqa: N802 - intentional mathematical alias
        return self.k

    @property
    def E(self) -> int:  # noqa: N802 - intentional mathematical alias
        return self.num_experts


@dataclass(slots=True)
class DataSet:
    """One pre-routed NumPy input set and a fresh emulated-bf16 output.

    NumPy does not expose bfloat16 consistently across supported versions, so
    ``out`` is float32 storage.  The reference implementation writes values
    that have been rounded to the bfloat16 value set into this storage.
    """

    spec: CaseSpec
    a: np.ndarray
    b_col_major: np.ndarray
    scale_a: np.ndarray
    scale_b: np.ndarray
    moe_weights: np.ndarray
    token_ids: np.ndarray
    expert_ids: np.ndarray
    topk: int
    out: np.ndarray

    def kernel_args(self) -> tuple[object, ...]:
        """Return arguments in the public ``run_kernel`` order."""

        return (
            self.a,
            self.b_col_major,
            self.scale_a,
            self.scale_b,
            self.moe_weights,
            self.token_ids,
            self.expert_ids,
            self.topk,
            self.out,
        )


# The smoke suite is deliberately small enough for a deterministic CPU oracle.
SMOKE_CASES: Final[tuple[CaseSpec, ...]] = (
    CaseSpec("smoke_gate_up", em=256, n=128, k=224, num_experts=4),
    CaseSpec(
        "smoke_down", em=512, n=224, k=64, num_experts=8, expert_dist="zipf"
    ),
)

# Four reduced representatives preserve both projection shapes and routing
# regimes while remaining suitable for a short C500 bring-up run.
QUICK_CASES: Final[tuple[CaseSpec, ...]] = (
    CaseSpec("quick_decode_gate_up", 512, 256, 448, 16, "uniform"),
    CaseSpec("quick_prefill_gate_up", 1024, 256, 448, 16, "zipf"),
    CaseSpec("quick_decode_down", 512, 448, 128, 16, "uniform"),
    CaseSpec("quick_prefill_down", 1024, 448, 128, 16, "zipf"),
)

# Exact DeepSeek-V3 representative shapes from the problem statement.  These
# metadata objects are cheap; their arrays are intentionally not constructed
# during local mock evaluation.
FULL_CASES: Final[tuple[CaseSpec, ...]] = (
    CaseSpec("full_decode_gate_up", 4096, 4096, 7168, 256, "uniform"),
    CaseSpec("full_prefill_gate_up", 32768, 4096, 7168, 256, "zipf"),
    CaseSpec("full_decode_down", 4096, 7168, 2048, 256, "uniform"),
    CaseSpec("full_prefill_down", 32768, 7168, 2048, 256, "zipf"),
)

CASE_SUITES: Final[Mapping[str, tuple[CaseSpec, ...]]] = {
    "smoke": SMOKE_CASES,
    "quick": QUICK_CASES,
    "full": FULL_CASES,
}


def get_suite(name: str) -> tuple[CaseSpec, ...]:
    """Return an immutable suite by name."""

    try:
        return CASE_SUITES[name.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unknown suite {name!r}; expected smoke, quick, or full") from exc


def _stable_case_seed(spec: CaseSpec, seed: int) -> int:
    identity = (
        f"{seed}|{spec.name}|{spec.em}|{spec.n}|{spec.k}|"
        f"{spec.num_experts}|{spec.expert_dist}|{spec.topk}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")


def generate_expert_ids(
    tile_count: int,
    num_experts: int,
    distribution: str,
    *,
    seed: int = FIXED_SEED,
    alpha: float = ZIPF_ALPHA,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate one expert id per 128-row tile.

    Uniform routing samples experts with replacement.  Zipf routing samples
    rank ``r`` with probability proportional to ``(r + 1) ** -alpha``.  For
    suites with at least three tiles, the result deliberately contains both a
    repeated expert and an expert transition, exercising sparse/duplicate tile
    handling deterministically.  Sampling with replacement (and that forced
    duplicate) also guarantees an unhit expert whenever tile count is no
    greater than expert count.
    """

    if tile_count <= 0 or num_experts <= 0:
        raise ValueError("tile_count and num_experts must be positive")
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    normalized = distribution.lower()
    if normalized == "skewed":
        normalized = "zipf"
    if normalized not in {"uniform", "zipf"}:
        raise ValueError("distribution must be 'uniform', 'zipf', or 'skewed'")

    generator = rng if rng is not None else np.random.default_rng(seed)
    if normalized == "uniform":
        result = generator.integers(0, num_experts, size=tile_count, dtype=np.int32)
    else:
        ranks = np.arange(1, num_experts + 1, dtype=np.float64)
        probabilities = np.power(ranks, -alpha)
        probabilities /= probabilities.sum()
        result = generator.choice(
            num_experts, size=tile_count, replace=True, p=probabilities
        ).astype(np.int32, copy=False)

    if num_experts > 1 and tile_count == 2 and result[0] == result[1]:
        # The two-tile smoke case should cross an actual expert boundary.
        result[1] = (int(result[0]) + 1) % num_experts
    elif num_experts > 1 and tile_count >= 3:
        # Make duplicate and changing expert tiles explicit test invariants.
        result[1] = result[0]
        if np.all(result == result[0]):
            result[-1] = (int(result[0]) + 1) % num_experts

    return np.ascontiguousarray(result, dtype=np.int32)


def estimate_case_bytes(spec: CaseSpec) -> int:
    """Estimate bytes allocated by ``generate_case`` (including raw inputs)."""

    raw_a = spec.num_tokens * spec.k
    routed_a = spec.em * spec.k
    b = spec.num_experts * spec.n * spec.k
    float_inputs = (
        spec.num_tokens + spec.em + spec.num_experts * spec.n + spec.em
    ) * np.dtype(np.float32).itemsize
    integer_inputs = (spec.em + spec.tile_count) * np.dtype(np.int32).itemsize
    output = spec.em * spec.n * np.dtype(np.float32).itemsize
    return int(raw_a + routed_a + b + float_inputs + integer_inputs + output)


def generate_case(spec: CaseSpec, *, seed: int = FIXED_SEED) -> DataSet:
    """Build deterministic, pre-routed NumPy inputs for ``spec``.

    ``token_ids`` is a permutation of flattened token/top-k route ids.  ``a``
    and ``scale_a`` are gathered before being returned, exactly as required by
    the operator contract; a kernel must consume their routed rows directly.
    """

    if not isinstance(spec, CaseSpec):
        raise TypeError("spec must be a CaseSpec")
    generator = np.random.default_rng(_stable_case_seed(spec, seed))
    estimated_bytes = estimate_case_bytes(spec)

    try:
        raw_a = generator.integers(
            -8, 9, size=(spec.num_tokens, spec.k), dtype=np.int8
        )
        raw_scale_a = generator.uniform(
            1.0e-3, 5.0e-2, size=spec.num_tokens
        ).astype(np.float32)
        token_ids = generator.permutation(spec.em).astype(np.int32, copy=False)
        routed_tokens = token_ids // np.int32(spec.topk)
        a = np.ascontiguousarray(raw_a[routed_tokens], dtype=np.int8)
        scale_a = np.ascontiguousarray(raw_scale_a[routed_tokens], dtype=np.float32)

        b_col_major = generator.integers(
            -8,
            9,
            size=(spec.num_experts, spec.n, spec.k),
            dtype=np.int8,
        )
        scale_b = generator.uniform(
            1.0e-3, 5.0e-2, size=(spec.num_experts, spec.n)
        ).astype(np.float32)
        moe_weights = generator.uniform(0.0, 1.0, size=spec.em).astype(np.float32)
        expert_ids = generate_expert_ids(
            spec.tile_count,
            spec.num_experts,
            spec.expert_dist,
            alpha=ZIPF_ALPHA,
            rng=generator,
        )
        out = np.zeros((spec.em, spec.n), dtype=np.float32)
    except MemoryError as exc:
        gib = estimated_bytes / (1024**3)
        raise MemoryError(
            f"unable to allocate exact case {spec.name!r}; estimated footprint "
            f"is {gib:.2f} GiB (the case was not scaled)"
        ) from exc

    return DataSet(
        spec=spec,
        a=a,
        b_col_major=np.ascontiguousarray(b_col_major),
        scale_a=scale_a,
        scale_b=np.ascontiguousarray(scale_b),
        moe_weights=np.ascontiguousarray(moe_weights),
        token_ids=np.ascontiguousarray(token_ids),
        expert_ids=expert_ids,
        topk=spec.topk,
        out=out,
    )


__all__ = [
    "CASE_SUITES",
    "DEFAULT_SEED",
    "DataSet",
    "FIXED_SEED",
    "FULL_CASES",
    "QUICK_CASES",
    "SMOKE_CASES",
    "TILE_ROWS",
    "TOPK",
    "ZIPF_ALPHA",
    "CaseSpec",
    "estimate_case_bytes",
    "generate_case",
    "generate_expert_ids",
    "get_suite",
]
