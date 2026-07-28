"""Expected, user-facing failures raised by the trusted controller."""

from __future__ import annotations


class ControlledRuntimeError(RuntimeError):
    """An operational failure that the public CLI may serialize safely."""

