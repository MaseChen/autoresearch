"""CLI for the trusted autonomous-research controller."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import sqlite3
import sys
from typing import Any, Iterator, Sequence

from .controller import ControllerSignal, ResearchController
from .errors import ControlledRuntimeError
from .models import ControllerConfig
from .summary import compact_status_payload
from ..migration import (
    CoordinatedMigrationError,
    coordinate_v2_to_v3_migration,
    production_cli_schema_guard,
)
from ..recovery import restore_checkpoint


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
    config = ControllerConfig.load(args.config)
    production_cli_schema_guard(
        history_db=config.state_dir / "history.sqlite3",
        controller_db=config.controller_dir / "controller.sqlite3",
    )
    return ResearchController(config)


def _doctor(args: argparse.Namespace) -> int:
    payload = _controller(args).doctor()
    _print(payload)
    return 0 if payload["status"] == "SUCCESS" else 2


def _start(args: argparse.Namespace) -> int:
    controller = _controller(args)
    payload = controller.start(proposal_only=args.proposal_only)
    output = payload
    if args.format == "compact":
        output = compact_status_payload(
            controller.status(str(payload["id"])),
            command="start",
        )
    _print(output)
    return (
        0
        if payload["status"]
        in {"PROMOTED", "PROPOSAL_READY", "BUDGET_EXHAUSTED"}
        else 3
    )


def _resume(args: argparse.Namespace) -> int:
    controller = _controller(args)
    payload = controller.resume(args.run_id)
    output = payload
    if args.format == "compact":
        output = compact_status_payload(
            controller.status(args.run_id),
            command="resume",
        )
    _print(output)
    return 0 if payload["status"] in {"PROMOTED", "BUDGET_EXHAUSTED"} else 3


def _status(args: argparse.Namespace) -> int:
    payload = _controller(args).status(args.run_id)
    if args.format == "compact":
        payload = compact_status_payload(payload)
    _print(payload)
    return 0


def _stop(args: argparse.Namespace) -> int:
    _print(_controller(args).stop(args.run_id))
    return 0


def _checkpoint(args: argparse.Namespace) -> int:
    _print(_controller(args).checkpoint(args.run_id))
    return 0


def _restore(args: argparse.Namespace) -> int:
    _print(
        restore_checkpoint(
            Path(args.checkpoint), Path(args.destination)
        )
    )
    return 0


def _migrate_v3(args: argparse.Namespace) -> int:
    config = ControllerConfig.load(args.config)
    _print(
        coordinate_v2_to_v3_migration(
            history_db=config.state_dir / "history.sqlite3",
            controller_db=config.controller_dir / "controller.sqlite3",
            state_dir=config.state_dir,
            config_path=Path(args.config),
            repository=config.repository_dir,
            expected_git_commit=config.expected_git_commit,
            checkpoint_root=Path(args.checkpoint_root),
            operation_id=args.operation_id,
        )
    )
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
                help=(
                    "make one live proposal and policy-check it without "
                    "candidate evaluation"
                ),
            )
        if command in {"start", "resume", "status"}:
            subparser.add_argument(
                "--format",
                choices=("json", "compact"),
                default="json",
            )
        subparser.set_defaults(handler=handler)
    restore = subparsers.add_parser(
        "restore-checkpoint",
        help="verify a checkpoint and restore it to a new inactive runtime root",
    )
    restore.add_argument("--checkpoint", required=True)
    restore.add_argument("--destination", required=True)
    restore.set_defaults(handler=_restore)
    migrate = subparsers.add_parser(
        "migrate-v3",
        help=(
            "checkpoint and migrate an offline terminal History/Controller "
            "V2 pair"
        ),
    )
    migrate.add_argument("--config", required=True)
    migrate.add_argument(
        "--checkpoint-root",
        "--output",
        dest="checkpoint_root",
        required=True,
    )
    migrate.add_argument("--operation-id", default=None)
    migrate.set_defaults(handler=_migrate_v3)
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
    except CoordinatedMigrationError as exc:
        print(
            json.dumps(
                exc.to_dict(),
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    except (
        OSError,
        ControlledRuntimeError,
        ValueError,
        sqlite3.DatabaseError,
    ) as exc:
        _print_error(exc)
        return 2
