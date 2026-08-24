"""Dependency-light NDJSON agent used through a fixed SSH command."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from .protocol import (
    MAX_AGENT_LINE_BYTES,
    ConsoleAgentRequestV1,
    ConsoleAgentResponseV1,
    ProblemV1,
    encode_wire,
    strict_json_loads,
)
from .read_model import ConsolePaths, ConsoleReadModel

if TYPE_CHECKING:
    from .operations import ConsoleOperationService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernel-autoresearch-console-agent")
    parser.add_argument("--admin-manifest", type=Path, required=True)
    parser.add_argument("--git-binary", default="git", help=argparse.SUPPRESS)
    return parser


def _safe_problem(exc: BaseException) -> ProblemV1:
    if isinstance(exc, (ValueError, TypeError)):
        return ProblemV1(
            code="CONSOLE_REQUEST_REJECTED",
            title="Console request rejected",
            detail=str(exc)[:4096] or type(exc).__name__,
            retryable=False,
        )
    return ProblemV1(
        code="CONSOLE_INTERNAL_ERROR",
        title="Console agent failed closed",
        detail=type(exc).__name__,
        retryable=False,
    )


def handle_request(
    request: ConsoleAgentRequestV1,
    *,
    model: ConsoleReadModel,
    operations: "ConsoleOperationService | None" = None,
) -> ConsoleAgentResponseV1:
    if request.operation == "handshake":
        if request.payload:
            raise ValueError("handshake payload must be empty")
        identity = model.runtime_identity()
        health = model.health(deep=True)
        return ConsoleAgentResponseV1(
            request_id=request.request_id,
            status="SUCCESS",
            payload={"runtime_identity": identity.to_dict(), "health": health},
        )
    if request.operation in {"snapshot", "subscribe"}:
        unknown = sorted(set(request.payload) - {"limit", "artifact_id"})
        if unknown:
            raise ValueError(
                "snapshot contains unknown fields: " + ", ".join(unknown)
            )
        if "artifact_id" in request.payload:
            if set(request.payload) != {"artifact_id"}:
                raise ValueError("artifact snapshot selector may not be combined")
            artifact = model.scientific_artifact(request.payload["artifact_id"])
            return ConsoleAgentResponseV1(
                request_id=request.request_id,
                status="SUCCESS",
                payload={"scientific_artifact": artifact},
            )
        limit = request.payload.get("limit", 50)
        snapshot = model.snapshot(limit=limit)
        return ConsoleAgentResponseV1(
            request_id=request.request_id,
            status="SUCCESS",
            payload={"snapshot": snapshot.to_dict()},
        )
    if operations is None:
        raise ValueError("Console write operations are not configured")
    if request.operation == "prepare":
        prepared = operations.prepare(request.payload)
        return ConsoleAgentResponseV1(
            request_id=request.request_id,
            status="SUCCESS",
            payload={"prepared_operation": prepared.to_dict()},
        )
    if request.operation == "execute":
        receipt = operations.execute(request.payload)
        return ConsoleAgentResponseV1(
            request_id=request.request_id,
            status="SUCCESS",
            payload={"operation_receipt": receipt.to_dict()},
        )
    if request.operation == "reconcile":
        receipt = operations.reconcile(request.payload)
        return ConsoleAgentResponseV1(
            request_id=request.request_id,
            status="SUCCESS",
            payload={"operation_receipt": receipt.to_dict()},
        )
    raise ValueError(
        f"operation {request.operation} is reserved but not active in this release"
    )


def serve(
    *,
    model: ConsoleReadModel,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    operations: "ConsoleOperationService | None" = None,
) -> int:
    while True:
        raw = input_stream.readline(MAX_AGENT_LINE_BYTES + 2)
        if not raw:
            return 0
        request_id = "invalid-request"
        try:
            if len(raw) > MAX_AGENT_LINE_BYTES + 1 or not raw.endswith(b"\n"):
                raise ValueError("console agent request line exceeds its bound")
            value = strict_json_loads(raw[:-1], max_bytes=MAX_AGENT_LINE_BYTES)
            request = ConsoleAgentRequestV1.from_value(value)
            request_id = request.request_id
            if request.operation == "subscribe":
                if set(request.payload) != {"limit"}:
                    raise ValueError("subscribe payload must contain only limit")
                while True:
                    response = handle_request(
                        request, model=model, operations=operations
                    )
                    try:
                        output_stream.write(encode_wire(response.to_dict()))
                        output_stream.flush()
                    except BrokenPipeError:
                        return 0
                    time.sleep(2.0)
            response = handle_request(request, model=model, operations=operations)
        except BaseException as exc:
            response = ConsoleAgentResponseV1(
                request_id=request_id,
                status="ERROR",
                problem=_safe_problem(exc),
            )
        output_stream.write(encode_wire(response.to_dict()))
        output_stream.flush()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        from .operations import ConsoleOperationService

        paths = ConsolePaths.from_admin_manifest(args.admin_manifest)
        model = ConsoleReadModel(paths, git_binary=args.git_binary)
        operations = ConsoleOperationService(paths, model)
    except BaseException as exc:
        response = ConsoleAgentResponseV1(
            request_id="startup",
            status="ERROR",
            problem=_safe_problem(exc),
        )
        sys.stdout.buffer.write(encode_wire(response.to_dict()))
        sys.stdout.buffer.flush()
        return 2
    return serve(
        model=model,
        input_stream=sys.stdin.buffer,
        output_stream=sys.stdout.buffer,
        operations=operations,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["handle_request", "main", "serve"]
