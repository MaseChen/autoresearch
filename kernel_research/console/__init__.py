"""Dependency-light contracts for the local Autoresearch Console.

The GPU host installs this package with ``--no-deps``.  Keep this module and
the remote agent import path free of FastAPI, Pydantic, NumPy, Torch, Triton,
and browser-only dependencies.
"""

from .protocol import (
    CompositeEventCursorV1,
    ConsoleAgentRequestV1,
    ConsoleAgentResponseV1,
    ConsoleSnapshotV1,
    DraftV1,
    ManualEvaluationRequestV1,
    OperationReceiptV1,
    PreparedOperationV1,
    ProblemV1,
    RuntimeIdentityV1,
    strict_json_loads,
)

__all__ = [
    "CompositeEventCursorV1",
    "ConsoleAgentRequestV1",
    "ConsoleAgentResponseV1",
    "ConsoleSnapshotV1",
    "DraftV1",
    "ManualEvaluationRequestV1",
    "OperationReceiptV1",
    "PreparedOperationV1",
    "ProblemV1",
    "RuntimeIdentityV1",
    "strict_json_loads",
]
