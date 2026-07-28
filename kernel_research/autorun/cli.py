"""CLI for the trusted autonomous-research controller."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import signal
import sqlite3
import sys
from typing import Any, Iterator, Sequence

from .controller import ControllerSignal, ResearchController
from .errors import ControlledRuntimeError
from .models import ControllerConfig


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2))


def _print_error(exc: Exception, *, kind: str = "ERROR") -> None:
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "FAILED",
                "error_type": kind,
                "error": f"{type(exc).__name__}: {exc}",
            },
            sort_keys=True,
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


@contextmanager
def _controlled_signals() -> Iterator[None]:
    previous: dict[int, Any] = {}

    def handle(signum: int, _frame: Any) -> None:
        raise ControllerSignal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handle)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _controller(args: argparse.Namespace) -> ResearchController:
    return ResearchController(ControllerConfig.load(args.config))


def _doctor(args: argparse.Namespace) -> int:
    payload = _controller(args).doctor()
    _print(payload)
    return 0 if payload["status"] == "SUCCESS" else 2


def _start(args: argparse.Namespace) -> int:
    payload = _controller(args).start(proposal_only=args.proposal_only)
    _print(payload)
    return (
        0
        if payload["status"]
        in {"PROMOTED", "PROPOSAL_READY", "BUDGET_EXHAUSTED"}
        else 3
    )


def _resume(args: argparse.Namespace) -> int:
    payload = _controller(args).resume(args.run_id)
    _print(payload)
    return 0 if payload["status"] in {"PROMOTED", "BUDGET_EXHAUSTED"} else 3


def _status(args: argparse.Namespace) -> int:
    _print(_controller(args).status(args.run_id))
    return 0


def _stop(args: argparse.Namespace) -> int:
    _print(_controller(args).stop(args.run_id))
    return 0


def _checkpoint(args: argparse.Namespace) -> int:
    _print(_controller(args).checkpoint(args.run_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernel-autoresearch",
        description="Run a bounded OpenCode proposal and C500 evaluation loop.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, handler in (
        ("doctor", _doctor),
        ("start", _start),
        ("resume", _resume),
        ("status", _status),
        ("stop", _stop),
        ("checkpoint", _checkpoint),
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        if command in {"resume", "stop", "checkpoint"}:
            subparser.add_argument("--run-id", required=True)
        elif command == "status":
            subparser.add_argument("--run-id", default=None)
        elif command == "start":
            subparser.add_argument(
                "--proposal-only",
                action="store_true",
                help="make one live proposal and policy-check it without candidate evaluation",
            )
        subparser.set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with _controlled_signals():
            return int(args.handler(args))
    except ControllerSignal as exc:
        _print_error(exc, kind="INTERRUPTED")
        return 128 + exc.signum
    except KeyboardInterrupt:
        _print_error(KeyboardInterrupt("operator interrupt"), kind="INTERRUPTED")
        return 130
    except (
        OSError,
        ControlledRuntimeError,
        ValueError,
        sqlite3.DatabaseError,
    ) as exc:
        _print_error(exc)
        return 2
