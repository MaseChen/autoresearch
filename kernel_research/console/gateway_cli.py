"""CLI entry point for the single-user Mac-local Console Gateway."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import socket
import sys
import webbrowser

from .config import DEFAULT_CONFIG_PATH, GatewayConfig
from .transport import SSHAgentTransport
from .local_store import ConsoleLocalStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernel-autoresearch-console")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if type(args.port) is not int or not 0 <= args.port <= 65535:
        raise ValueError("port must be 0 or a valid TCP port")
    config = GatewayConfig.load(args.config)
    data_dir = config.ensure_data_dir()
    lock_path = data_dir / "gateway.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Console Gateway already owns this data_dir") from exc
        os.fchmod(descriptor, 0o600)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        from .gateway import GatewayState, create_app

        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "Mac Gateway dependencies are missing; install .[console]"
            ) from exc

        transport = SSHAgentTransport(config)
        state = GatewayState(
            transport=transport,
            static_dir=_static_dir(),
            poll_interval_ms=config.poll_interval_ms,
            local_store=ConsoleLocalStore(data_dir),
        )
        app = create_app(state)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", args.port))
        listener.listen(128)
        port = int(listener.getsockname()[1])
        url = f"http://127.0.0.1:{port}/#bootstrap={state.sessions.bootstrap_token}"
        print(f"CONSOLE_URL={url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url, new=2)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                access_log=False,
                server_header=False,
                date_header=False,
                log_level="warning",
            )
        )
        server.run(sockets=[listener])
        return 0
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["main"]
