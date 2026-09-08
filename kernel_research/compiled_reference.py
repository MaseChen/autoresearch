"""Whole-operator torch.compile reference used only as a scoring anchor.

Torch is deliberately injected by the evaluator worker.  Importing this module
on the trusted host therefore remains dependency-light and cannot initialize a
GPU runtime.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable

from .constants import EXPERT_TILE_ROWS


SCORING_COMPILER_CONFIG = {
    "backend": "inductor",
    "fullgraph": True,
    "dynamic": False,
    "mode": "default",
}


def make_scoring_reference(torch: Any) -> Callable[..., Any]:
    """Return a pure tensor implementation with a static tile loop."""

    def scoring_reference(
        a: Any,
        b_col_major: Any,
        scale_a: Any,
        scale_b: Any,
        moe_weights: Any,
        expert_ids: Any,
    ) -> Any:
        tile_count = a.shape[0] // EXPERT_TILE_ROWS
        blocks = []
        for tile in range(tile_count):
            row_start = tile * EXPERT_TILE_ROWS
            row_end = row_start + EXPERT_TILE_ROWS
            # Keep the expert selection inside the tensor graph.  Indexing a
            # Python container with a zero-dimensional device tensor asks
            # Dynamo for ``aten._local_scalar_dense`` and therefore violates
            # the fail-closed fullgraph contract.
            expert_index = expert_ids[tile : tile + 1].to(dtype=torch.int64)
            a_tile = a[row_start:row_end].contiguous()
            b_tile = torch.index_select(
                b_col_major, 0, expert_index
            )[0].transpose(0, 1).contiguous()
            scale_b_tile = torch.index_select(scale_b, 0, expert_index)[0]
            if hasattr(torch, "_int_mm"):
                accumulator = torch._int_mm(a_tile, b_tile)
            else:
                accumulator = torch.matmul(
                    a_tile.to(torch.int32), b_tile.to(torch.int32)
                )
            scaled = (
                accumulator.to(torch.float32)
                * scale_a[row_start:row_end, None]
                * scale_b_tile[None, :]
                * moe_weights[row_start:row_end, None]
            )
            blocks.append(scaled.to(torch.bfloat16))
        return torch.cat(blocks, dim=0)

    return scoring_reference


def compile_scoring_reference(torch: Any) -> Callable[..., Any]:
    """Compile the entire reference graph using the frozen configuration."""

    compile_function = getattr(torch, "compile", None)
    if not callable(compile_function):
        raise RuntimeError("torch.compile is unavailable")
    reference = make_scoring_reference(torch)
    return compile_function(reference, **SCORING_COMPILER_CONFIG)


def scoring_reference_source_sha256() -> str:
    """Return a tagged digest of the implementation source and constants."""

    material = (
        inspect.getsource(make_scoring_reference).encode("utf-8")
        + repr(sorted(SCORING_COMPILER_CONFIG.items())).encode("ascii")
        + str(EXPERT_TILE_ROWS).encode("ascii")
    )
    return "sha256:" + hashlib.sha256(material).hexdigest()


__all__ = [
    "SCORING_COMPILER_CONFIG",
    "compile_scoring_reference",
    "make_scoring_reference",
    "scoring_reference_source_sha256",
]
