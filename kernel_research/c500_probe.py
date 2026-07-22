"""Minimal mcTriton compile/launch probe, imported only inside a C500 worker."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_probe(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, values + 1.0)


def run_probe(device: str) -> None:
    input_tensor = torch.zeros((16,), dtype=torch.float32, device=device)
    output_tensor = torch.empty_like(input_tensor)
    _copy_probe[(1,)](input_tensor, output_tensor, BLOCK=16)
    for api_name in ("cuda", "maca"):
        api = getattr(torch, api_name, None)
        if api is not None and bool(api.is_available()):
            api.synchronize()
            break
    if not bool(torch.equal(output_tensor, torch.ones_like(output_tensor))):
        raise RuntimeError("mcTriton compile probe produced an incorrect result")
