"""Docker command construction and bounded subprocess execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
from typing import Any, Sequence

from ..constants import (
    COMMAND_READ_CHUNK_BYTES,
    EVALUATOR_PID_LIMIT,
    OPENCODE_PROPOSER_STEPS,
    PROPOSER_PID_LIMIT,
)
from .model_catalog import (
    OPENCODE_OUTPUT_TOKEN_MAX_ENV,
    resolve_opencode_model,
)
from .models import ControllerConfig


EVALUATOR_REQUEST_IDENTITY_CONTAINER_PATH = "/request/identity.json"


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_limited: bool = False


class CommandRunner:
    """Execute fixed argv without a shell and strictly bound captured output."""

    @staticmethod
    def _kill_client(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    @staticmethod
    def _kill_exact_container(
        docker_binary: Path, container_name: str
    ) -> None:
        for command in (
            [str(docker_binary), "kill", container_name],
            [str(docker_binary), "rm", "-f", container_name],
        ):
            try:
                subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    shell=False,
                )
            except (OSError, subprocess.SubprocessError):
                # Cleanup is best effort here. The exact name remains recorded
                # in controller state for stop/resume cleanup.
                pass

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None,
        timeout_sec: float,
        max_output_bytes: int,
        container_name: str | None = None,
        docker_binary: Path | None = None,
    ) -> CommandResult:
        if timeout_sec <= 0 or max_output_bytes <= 0:
            raise ValueError("timeout and output limit must be positive")
        stdin_file = tempfile.TemporaryFile() if input_text is not None else None
        if stdin_file is not None:
            stdin_file.write(input_text.encode("utf-8"))
            stdin_file.seek(0)
        try:
            process = subprocess.Popen(
                list(argv),
                stdin=stdin_file if stdin_file is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            selector = selectors.DefaultSelector()
            streams = {
                process.stdout.fileno(): ("stdout", bytearray()),
                process.stderr.fileno(): ("stderr", bytearray()),
            }
            for descriptor in streams:
                os.set_blocking(descriptor, False)
                selector.register(descriptor, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout_sec
            timed_out = False
            output_limited = False
            captured_bytes = 0
            try:
                while selector.get_map():
                    remaining_time = deadline - time.monotonic()
                    if remaining_time <= 0:
                        timed_out = True
                        break
                    events = selector.select(
                        min(max(remaining_time, 0.0), 0.1)
                        if process.poll() is None
                        else 0
                    )
                    if not events and process.poll() is not None:
                        # A final nonblocking pass observes EOF on both pipes.
                        events = [
                            (key, selectors.EVENT_READ)
                            for key in selector.get_map().values()
                        ]
                    for key, _ in events:
                        descriptor = int(key.fd)
                        try:
                            chunk = os.read(
                                descriptor, COMMAND_READ_CHUNK_BYTES
                            )
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(descriptor)
                            continue
                        _, buffer = streams[descriptor]
                        available = max_output_bytes - captured_bytes
                        if available > 0:
                            retained = chunk[:available]
                            buffer.extend(retained)
                            captured_bytes += len(retained)
                        if len(chunk) > available:
                            output_limited = True
                            break
                    if output_limited:
                        break
            except KeyboardInterrupt:
                if container_name and docker_binary:
                    self._kill_exact_container(docker_binary, container_name)
                self._kill_client(process)
                raise
            finally:
                selector.close()
            if timed_out or output_limited:
                if container_name and docker_binary:
                    self._kill_exact_container(docker_binary, container_name)
                self._kill_client(process)
            else:
                try:
                    process.wait(
                        timeout=max(deadline - time.monotonic(), 0.0)
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if container_name and docker_binary:
                        self._kill_exact_container(
                            docker_binary, container_name
                        )
                    self._kill_client(process)
            returncode = int(process.returncode or 0)
            stdout = bytes(streams[process.stdout.fileno()][1]).decode(
                "utf-8", errors="replace"
            )
            stderr = bytes(streams[process.stderr.fileno()][1]).decode(
                "utf-8", errors="replace"
            )
            process.stdout.close()
            process.stderr.close()
        finally:
            if stdin_file is not None:
                stdin_file.close()
        return CommandResult(
            tuple(str(item) for item in argv),
            returncode,
            stdout,
            stderr,
            timed_out=timed_out,
            output_limited=output_limited,
        )

    def remove_exact_container(
        self, docker_binary: Path, container_name: str
    ) -> None:
        self._kill_exact_container(docker_binary, container_name)


def _common_security_argv(
    config: ControllerConfig,
    *,
    name: str,
    run_id: str,
    memory: str,
    cpus: float,
    pids: int,
) -> list[str]:
    return [
        str(config.docker_binary),
        "run",
        "--rm",
        "--name",
        name,
        "--label",
        f"kernel-autoresearch.run={run_id}",
        "--label",
        "kernel-autoresearch.project=fused-moe",
        "--init",
        "--read-only",
        "--user",
        f"{config.container_uid}:{config.container_gid}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(pids),
        "--memory",
        memory,
        "--cpus",
        str(cpus),
    ]


def proposer_argv(
    config: ControllerConfig,
    *,
    name: str,
    run_id: str,
    opencode_config: Path,
) -> list[str]:
    model = resolve_opencode_model(config.opencode_model)
    argv = _common_security_argv(
        config,
        name=name,
        run_id=run_id,
        memory=config.proposer_memory,
        cpus=config.proposer_cpus,
        pids=PROPOSER_PID_LIMIT,
    )
    argv.extend(
        [
            # Keep Docker stdin open so the pinned OpenCode CLI can consume the
            # proposal prompt without placing its contents in argv or env.
            "--interactive",
            "--network",
            "bridge",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
            "--tmpfs",
            (
                f"/home/opencode:rw,nosuid,nodev,size=256m,"
                f"mode=700,uid={config.container_uid},gid={config.container_gid}"
            ),
            "--mount",
            f"type=bind,src={opencode_config},dst=/opencode/config.json,readonly",
            "--mount",
            (
                f"type=bind,src={config.deepseek_key_file},"
                "dst=/run/secrets/deepseek_api_key,readonly"
            ),
            "--env",
            "HOME=/home/opencode",
            "--env",
            "XDG_CONFIG_HOME=/home/opencode/.config",
            "--env",
            "XDG_DATA_HOME=/home/opencode/.local/share",
            "--env",
            "XDG_CACHE_HOME=/home/opencode/.cache",
            "--env",
            "OPENCODE_CONFIG=/opencode/config.json",
            "--env",
            "OPENCODE_DISABLE_AUTOUPDATE=1",
            "--env",
            "OPENCODE_DISABLE_CLAUDE_CODE=1",
            "--env",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS=1",
            "--env",
            "OPENCODE_DISABLE_LSP_DOWNLOAD=1",
            "--env",
            "OPENCODE_DISABLE_MODELS_FETCH=1",
            "--env",
            (
                f"{OPENCODE_OUTPUT_TOKEN_MAX_ENV}="
                f"{model.request_output_token_cap}"
            ),
            "--workdir",
            "/home/opencode",
            "--entrypoint",
            "opencode",
            config.proposer_image,
            "--pure",
            "run",
            "--format",
            "json",
            "--model",
            config.opencode_model,
            "--agent",
            "kernel-proposer",
        ]
    )
    return argv


def _evaluator_base_argv(
    config: ControllerConfig,
    *,
    name: str,
    run_id: str,
    cache_dir: Path,
) -> list[str]:
    framework_dir = (
        config.controller_dir
        / "framework"
        / config.resolved_framework_git_commit
    )
    argv = _common_security_argv(
        config,
        name=name,
        run_id=run_id,
        memory=config.evaluator_memory,
        cpus=config.evaluator_cpus,
        pids=EVALUATOR_PID_LIMIT,
    )
    argv.extend(
        [
            "--network",
            "none",
            "--group-add",
            str(config.video_gid),
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=4g,mode=1777",
            "--tmpfs",
            (
                f"/home/mw:rw,nosuid,nodev,size=512m,"
                f"mode=700,uid={config.container_uid},gid={config.container_gid}"
            ),
            "--mount",
            (
                f"type=bind,src={framework_dir},"
                "dst=/workspace,readonly"
            ),
            "--mount",
            (
                f"type=bind,src={cache_dir},"
                "dst=/evaluator-cache"
            ),
            "--env",
            "HOME=/home/mw",
            "--env",
            "TRITON_CACHE_DIR=/evaluator-cache/triton",
            "--workdir",
            "/workspace",
        ]
    )
    for device in config.gpu_devices:
        argv.extend(["--device", f"{device}:{device}"])
    argv.extend(
        [
            "--entrypoint",
            "/opt/conda/bin/python",
            config.evaluator_image,
        ]
    )
    return argv


def evaluator_argv(
    config: ControllerConfig,
    *,
    name: str,
    run_id: str,
    candidate_path: Path,
    suite: str,
    baseline_path: Path | None,
    cache_dir: Path,
    request_identity_path: Path | None = None,
    backend: str = "c500",
    evaluation_protocol_id: str | None = None,
) -> list[str]:
    argv = _evaluator_base_argv(
        config, name=name, run_id=run_id, cache_dir=cache_dir
    )
    insertion = argv.index("--entrypoint")
    mounts = [
        "--mount",
        f"type=bind,src={candidate_path},dst=/candidate/kernel.py,readonly",
    ]
    if baseline_path is not None:
        mounts.extend(
            [
                "--mount",
                f"type=bind,src={baseline_path},dst=/baseline/kernel.py,readonly",
            ]
        )
    if request_identity_path is not None:
        mounts.extend(
            [
                "--mount",
                (
                    f"type=bind,src={request_identity_path},"
                    f"dst={EVALUATOR_REQUEST_IDENTITY_CONTAINER_PATH},readonly"
                ),
            ]
        )
    argv[insertion:insertion] = mounts
    argv.extend(
        [
            "-m",
            "kernel_research",
            "evaluate-raw",
            "--backend",
            backend,
            "--suite",
            suite,
            "--candidate",
            "/candidate/kernel.py",
        ]
    )
    if baseline_path is not None:
        argv.extend(["--baseline", "/baseline/kernel.py"])
    if request_identity_path is not None:
        argv.extend(
            [
                "--request-identity",
                EVALUATOR_REQUEST_IDENTITY_CONTAINER_PATH,
            ]
        )
    if evaluation_protocol_id is not None:
        argv.extend(["--evaluation-protocol", evaluation_protocol_id])
    return argv


def evaluator_doctor_argv(
    config: ControllerConfig, *, name: str, run_id: str, cache_dir: Path
) -> list[str]:
    argv = _evaluator_base_argv(
        config, name=name, run_id=run_id, cache_dir=cache_dir
    )
    argv.extend(
        [
            "-m",
            "kernel_research",
            "doctor",
            "--backend",
            "c500",
            "--timeout",
            "180",
        ]
    )
    return argv


def scoring_baseline_probe_argv(
    config: ControllerConfig, *, name: str, run_id: str, cache_dir: Path
) -> list[str]:
    """Build the fixed evaluator-container argv for one baseline probe."""

    argv = _evaluator_base_argv(
        config, name=name, run_id=run_id, cache_dir=cache_dir
    )
    argv.extend(
        [
            "-m",
            "kernel_research",
            "score-baseline-probe",
        ]
    )
    return argv


def write_opencode_config(path: Path, config: ControllerConfig) -> None:
    """Write the no-tool, bounded-step OpenCode configuration."""

    model = resolve_opencode_model(config.opencode_model)
    value: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "model": config.opencode_model,
        "default_agent": "kernel-proposer",
        "share": "disabled",
        "autoupdate": False,
        "snapshot": False,
        "plugin": [],
        "mcp": {},
        "formatter": False,
        "lsp": False,
        "enabled_providers": ["deepseek"],
        "tools": {
            "bash": False,
            "read": False,
            "write": False,
            "edit": False,
            "glob": False,
            "grep": False,
            "list": False,
            "task": False,
            "todowrite": False,
            "question": False,
            "webfetch": False,
            "websearch": False,
            "lsp": False,
            "skill": False,
        },
        "permission": {"*": "deny"},
        "provider": {
            "deepseek": {
                "options": {
                    "apiKey": "{file:/run/secrets/deepseek_api_key}",
                    "baseURL": "https://api.deepseek.com",
                },
                "models": {
                    model.provider_id: {
                        "name": model.display_name,
                        "limit": {
                            "context": model.context_tokens,
                            "output": model.output_tokens,
                        },
                    }
                },
            }
        },
        "agent": {
            "kernel-proposer": {
                "description": "One-shot Fused MoE kernel proposal generator",
                "mode": "primary",
                "steps": OPENCODE_PROPOSER_STEPS,
                "model": config.opencode_model,
                "permission": {"*": "deny"},
                "reasoningEffort": model.reasoning_effort,
                "thinking": {"type": "enabled"},
                "prompt": (
                    "Return only the strict ProposalV1 JSON requested by the "
                    "user. Never call a tool."
                ),
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def parse_json_output(output: str) -> dict[str, Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"container did not emit one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("container output must be a JSON object")
    return value
