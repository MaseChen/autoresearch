"""Fixed SSH transport for the dependency-light remote Console agent."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import select
import shlex
import subprocess
import time
from typing import Any, Iterator, Mapping, Protocol
import uuid

from .config import GatewayConfig
from .protocol import (
    MAX_AGENT_RESPONSE_BYTES,
    ConsoleAgentRequestV1,
    strict_json_loads,
)


class AgentTransport(Protocol):
    def call(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SSHAgentTransport:
    config: GatewayConfig
    timeout_sec: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.config, GatewayConfig):
            raise TypeError("config must be GatewayConfig")
        if not 1.0 <= self.timeout_sec <= 300.0:
            raise ValueError("transport timeout must be between 1 and 300 seconds")

    def remote_command(self) -> str:
        arguments = (
            self.config.remote_python,
            "-m",
            "kernel_research.console.agent",
            "--admin-manifest",
            self.config.remote_admin_manifest,
        )
        return shlex.join(arguments)

    def argv(self) -> list[str]:
        return [
            str(self.config.ssh_binary),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ClearAllForwardings=yes",
            "-T",
            self.config.ssh_target,
            self.remote_command(),
        ]

    def call(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = ConsoleAgentRequestV1(
            request_id=str(uuid.uuid4()), operation=operation, payload=payload
        )
        encoded = self._request_bytes(request)
        process = subprocess.run(
            self.argv(),
            input=encoded,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_sec,
            check=False,
        )
        if len(process.stdout) > MAX_AGENT_RESPONSE_BYTES:
            raise RuntimeError("remote Console response exceeds its size bound")
        if len(process.stderr) > 64 * 1024:
            raise RuntimeError("remote Console stderr exceeds its size bound")
        lines = process.stdout.splitlines()
        if process.returncode != 0 or len(lines) != 1:
            raise RuntimeError("remote Console agent failed closed")
        return self._response(request.request_id, lines[0])

    @staticmethod
    def _request_bytes(request: ConsoleAgentRequestV1) -> bytes:
        return json.dumps(
            request.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"

    @staticmethod
    def _response(request_id: str, raw: bytes) -> Mapping[str, Any]:
        response = strict_json_loads(raw, max_bytes=MAX_AGENT_RESPONSE_BYTES)
        if not isinstance(response, dict):
            raise RuntimeError("remote Console response is not an object")
        expected = {"schema_version", "request_id", "status", "payload", "problem"}
        if set(response) != expected or response.get("request_id") != request_id:
            raise RuntimeError("remote Console response identity mismatch")
        if response.get("status") != "SUCCESS" or not isinstance(
            response.get("payload"), dict
        ):
            problem = response.get("problem")
            code = problem.get("code") if isinstance(problem, dict) else "REMOTE_ERROR"
            raise RuntimeError(f"remote Console request failed: {code}")
        return response["payload"]

    def subscribe(self) -> Iterator[Mapping[str, Any]]:
        """Yield snapshots over one fixed SSH process until the consumer closes."""

        request = ConsoleAgentRequestV1(
            request_id=str(uuid.uuid4()),
            operation="subscribe",
            payload={"limit": 100},
        )
        process = subprocess.Popen(
            self.argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            shell=False,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(self._request_bytes(request))
            process.stdin.flush()
            process.stdin.close()
            descriptor = process.stdout.fileno()
            buffered = bytearray()
            deadline = time.monotonic() + self.timeout_sec
            while True:
                newline = buffered.find(b"\n")
                if newline >= 0:
                    line = bytes(buffered[:newline])
                    del buffered[: newline + 1]
                    if not line:
                        raise RuntimeError("remote Console subscription emitted an empty line")
                    deadline = time.monotonic() + self.timeout_sec
                    yield self._response(request.request_id, line)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("remote Console subscription timed out")
                ready, _, _ = select.select([descriptor], [], [], remaining)
                if not ready:
                    raise RuntimeError("remote Console subscription timed out")
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    raise RuntimeError("remote Console subscription ended")
                buffered.extend(chunk)
                if len(buffered) > MAX_AGENT_RESPONSE_BYTES + 1:
                    raise RuntimeError("remote Console response exceeds its size bound")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            process.stdout.close()


__all__ = ["AgentTransport", "SSHAgentTransport"]
