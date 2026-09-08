from __future__ import annotations

import asyncio
import importlib.util
import json
import io
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
import unittest
import uuid
from unittest import mock
from types import SimpleNamespace

from kernel_research.console.config import GatewayConfig
from kernel_research.console import gateway_cli
from kernel_research.console.protocol import AGENT_PROTOCOL_DIGEST, RuntimeIdentityV1
from kernel_research.console.transport import SSHAgentTransport

_FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
_CONSOLE_EXTRA_REQUIRED = "Mac Console Gateway tests require .[console]"
if _FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient

    from kernel_research.console.gateway import (
        GatewayState,
        SessionRegistry,
        SnapshotEventBroker,
        _paginate,
        create_app,
    )
else:
    TestClient = None
    GatewayState = None
    SessionRegistry = None
    SnapshotEventBroker = None
    _paginate = None
    create_app = None


def _snapshot(*, commit: str = "a" * 40) -> dict[str, object]:
    identity = RuntimeIdentityV1(
        git_commit=commit,
        expected_git_commit=commit,
        config_digest="sha256:" + "1" * 64,
        deployment_evidence_digest="sha256:" + "2" * 64,
        namespace_id="sha256:" + "3" * 64,
        execution_environment_digest="sha256:" + "4" * 64,
        profiler_activation_profile_digest="sha256:" + "5" * 64,
        scoring_shadow_profile_digest="sha256:" + "6" * 64,
        controller_schema_version=3,
        history_schema_version=3,
        campaign_schema_version=1,
        agent_protocol_digest=AGENT_PROTOCOL_DIGEST,
    )
    return {
        "schema_version": 1,
        "status": "STABLE",
        "runtime_identity": identity.to_dict(),
        "cursor": {"controller_event_id": 1},
        "observed_at": "2026-08-24T00:00:00Z",
        "source_digests": {},
        "data": {
            "runs": [{"id": "run-1", "status": "RUNNING"}],
            "iterations": [],
            "evaluation_attempts": [],
            "experiments": [{"id": 1, "status": "SUCCESS"}],
            "experiment_relations": [],
            "campaigns": [{"id": "campaign-1", "status": "CREATED"}],
            "child_runs": [],
            "resource_leases": [],
            "budget_actions": [],
            "soak_generations": [],
            "soak_violations": [],
        },
    }


class _FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((operation, dict(payload)))
        if operation == "snapshot":
            if "artifact_id" in payload:
                return {
                    "scientific_artifact": {
                        "artifact_id": payload["artifact_id"],
                        "manifest": {"format": "source-bundle-v1"},
                        "source": "def kernel(): pass\n",
                    }
                }
            return {"snapshot": _snapshot()}
        if operation == "prepare":
            return {
                "prepared_operation": {
                    "schema_version": 1,
                    "operation_id": payload["operation_id"],
                    "kind": payload["kind"],
                    "operation_digest": "sha256:" + "8" * 64,
                    "runtime_identity_digest": payload["runtime_identity_digest"],
                    "prepared_at": "2026-08-24T00:00:00Z",
                    "expires_epoch": 9999999999.0,
                    "confirmation_phrase": "启动自治 Run",
                    "impact": {"gpu_possible": True},
                }
            }
        if operation in {"execute", "reconcile"}:
            return {
                "operation_receipt": {
                    "schema_version": 1,
                    "operation_id": payload["operation_id"],
                    "kind": "RUN_START",
                    "operation_digest": "sha256:" + "8" * 64,
                    "status": "EXECUTING",
                    "observed_at": "2026-08-24T00:00:01Z",
                    "domain_identity": {"run_id": "console-run-1"},
                    "result": None,
                    "problem": None,
                }
            }
        raise AssertionError(operation)


class ConsoleConfigAndTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config_value(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "ssh_binary": "/usr/bin/ssh",
            "ssh_target": "mx@node01",
            "remote_python": "/home/mx/miniforge3/bin/python3.13",
            "remote_admin_manifest": "/home/mx/autoresearch-runtime/gpu1/admin.json",
            "data_dir": str(self.root / "data"),
            "poll_interval_ms": 2000,
        }

    def test_config_requires_0600_strict_shape_and_safe_paths(self) -> None:
        path = self.root / "config.json"
        path.write_text(json.dumps(self._config_value()), encoding="utf-8")
        path.chmod(0o600)
        config = GatewayConfig.load(path)
        self.assertEqual(config.ssh_target, "mx@node01")
        self.assertEqual(stat.S_IMODE(config.ensure_data_dir().stat().st_mode), 0o700)
        path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "0600"):
            GatewayConfig.load(path)
        path.chmod(0o600)
        value = self._config_value()
        value["ssh_target"] = "mx@node01;touch /tmp/pwned"
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ssh_target"):
            GatewayConfig.load(path)

    def test_transport_uses_fixed_ssh_argv_and_exact_response_identity(self) -> None:
        config = GatewayConfig(
            ssh_binary=Path("/usr/bin/ssh"),
            ssh_target="mx@node01",
            remote_python="/home/mx/miniforge3/bin/python3.13",
            remote_admin_manifest="/home/mx/autoresearch-runtime/gpu1/admin.json",
            data_dir=self.root / "data",
        )
        transport = SSHAgentTransport(config)
        observed: list[object] = []

        def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            observed.extend(argv)
            request = json.loads(bytes(kwargs["input"]).decode("utf-8"))
            response = {
                "schema_version": 1,
                "request_id": request["request_id"],
                "status": "SUCCESS",
                "payload": {"snapshot": _snapshot()},
                "problem": None,
            }
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(response).encode("utf-8") + b"\n", b""
            )

        with mock.patch("subprocess.run", side_effect=run):
            result = transport.call("snapshot", {"limit": 10})
        self.assertIn("BatchMode=yes", observed)
        self.assertIn("ClearAllForwardings=yes", observed)
        self.assertNotIn("SELECT", " ".join(str(item) for item in observed))
        self.assertEqual(result["snapshot"]["status"], "STABLE")

    def test_transport_rejects_output_bounds_identity_and_remote_errors(self) -> None:
        config = GatewayConfig(
            ssh_binary=Path("/usr/bin/ssh"),
            ssh_target="mx@node01",
            remote_python="/home/mx/miniforge3/bin/python3.13",
            remote_admin_manifest="/home/mx/autoresearch-runtime/gpu1/admin.json",
            data_dir=self.root / "data",
        )
        transport = SSHAgentTransport(config)

        def response(
            argv: list[str],
            *,
            input: bytes,
            stdout: object,
            stderr: object,
            timeout: float,
            check: bool,
        ) -> subprocess.CompletedProcess[bytes]:
            del stdout, stderr, timeout, check
            request = json.loads(input)
            value = {
                "schema_version": 1,
                "request_id": request["request_id"],
                "status": "ERROR",
                "payload": None,
                "problem": {"code": "REMOTE_BLOCKED"},
            }
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(value).encode() + b"\n", b""
            )

        with mock.patch("subprocess.run", side_effect=response):
            with self.assertRaisesRegex(RuntimeError, "REMOTE_BLOCKED"):
                transport.call("snapshot", {})
        cases = (
            subprocess.CompletedProcess([], 1, b"", b""),
            subprocess.CompletedProcess([], 0, b"{}\n{}\n", b""),
            subprocess.CompletedProcess([], 0, b"not-json\n", b""),
            subprocess.CompletedProcess([], 0, b"{}\n", b""),
            subprocess.CompletedProcess([], 0, b"{}\n", b"x" * (64 * 1024 + 1)),
        )
        for selected in cases:
            with self.subTest(returncode=selected.returncode, stdout=selected.stdout[:8]):
                with mock.patch("subprocess.run", return_value=selected):
                    with self.assertRaises((RuntimeError, ValueError)):
                        transport.call("snapshot", {})
        with (
            mock.patch("kernel_research.console.transport.MAX_AGENT_RESPONSE_BYTES", 8),
            mock.patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, b"123456789", b""),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "size bound"):
                transport.call("snapshot", {})

    def test_transport_subscription_reuses_one_fixed_ssh_process(self) -> None:
        config = GatewayConfig(
            ssh_binary=Path("/usr/bin/ssh"),
            ssh_target="mx@node01",
            remote_python="/home/mx/miniforge3/bin/python3.13",
            remote_admin_manifest="/home/mx/autoresearch-runtime/gpu1/admin.json",
            data_dir=self.root / "data",
        )
        transport = SSHAgentTransport(config)
        read_descriptor, write_descriptor = os.pipe()

        class Input(io.BytesIO):
            def flush(selected) -> None:
                request = json.loads(selected.getvalue())
                response = {
                    "schema_version": 1,
                    "request_id": request["request_id"],
                    "status": "SUCCESS",
                    "payload": {"snapshot": _snapshot()},
                    "problem": None,
                }
                os.write(write_descriptor, json.dumps(response).encode() + b"\n")
                os.close(write_descriptor)

        class Process:
            def __init__(self) -> None:
                self.stdin = Input()
                self.stdout = os.fdopen(read_descriptor, "rb", buffering=0)
                self.returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = 0

            def wait(self, timeout: float) -> int:
                del timeout
                self.returncode = 0
                return 0

            def kill(self) -> None:
                self.returncode = -9

        process = Process()
        observed: dict[str, object] = {}

        def popen(argv: list[str], **kwargs: object) -> Process:
            observed.update(kwargs)
            self.assertIn("BatchMode=yes", argv)
            return process

        with mock.patch("subprocess.Popen", side_effect=popen):
            stream = transport.subscribe()
            value = next(stream)
            self.assertEqual(value["snapshot"]["status"], "STABLE")
            stream.close()
        self.assertIs(observed["shell"], False)
        self.assertIs(observed["start_new_session"], True)
        self.assertEqual(process.returncode, 0)

    def test_config_rejects_duplicate_nan_remote_alias_and_poll_drift(self) -> None:
        path = self.root / "strict.json"
        invalid_values = (
            b'{"schema_version":1,"schema_version":1}',
            b'{"schema_version":NaN}',
        )
        for raw in invalid_values:
            path.write_bytes(raw)
            path.chmod(0o600)
            with self.assertRaises(ValueError):
                GatewayConfig.load(path)
        value = self._config_value()
        value["remote_python"] = "/home/mx/../bin/python"
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "canonical"):
            GatewayConfig.load(path)
        value = self._config_value()
        value["poll_interval_ms"] = True
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "poll_interval"):
            GatewayConfig.load(path)

    @unittest.skipUnless(_FASTAPI_AVAILABLE, _CONSOLE_EXTRA_REQUIRED)
    def test_snapshot_cache_fails_closed_on_runtime_identity_change(self) -> None:
        from kernel_research.console.gateway import SnapshotCache

        transport = _FakeTransport()
        cache = SnapshotCache(transport, ttl_seconds=0)
        self.assertEqual(cache.get()["status"], "STABLE")
        changed = _snapshot(commit="b" * 40)

        def changed_call(operation: str, payload: dict[str, object]) -> dict[str, object]:
            del operation, payload
            return {"snapshot": changed}

        transport.call = changed_call  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "restart handshake"):
            cache.get(force=True)

    @unittest.skipUnless(_FASTAPI_AVAILABLE, _CONSOLE_EXTRA_REQUIRED)
    def test_snapshot_cache_rejects_tampered_identity_and_protocol(self) -> None:
        from kernel_research.console.gateway import SnapshotCache

        for field, value in (
            ("runtime_identity_digest", "sha256:" + "9" * 64),
            ("agent_protocol_digest", "sha256:" + "8" * 64),
        ):
            with self.subTest(field=field):
                snapshot = _snapshot()
                snapshot["runtime_identity"] = dict(snapshot["runtime_identity"])
                snapshot["runtime_identity"][field] = value
                transport = _FakeTransport()
                transport.call = lambda operation, payload, snapshot=snapshot: {  # type: ignore[method-assign]
                    "snapshot": snapshot
                }
                with self.assertRaisesRegex(RuntimeError, "identity is invalid"):
                    SnapshotCache(transport, ttl_seconds=0).get(force=True)


@unittest.skipUnless(_FASTAPI_AVAILABLE, _CONSOLE_EXTRA_REQUIRED)
class ConsoleGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name).resolve()
        self.static = root / "dist"
        (self.static / "assets").mkdir(parents=True)
        (self.static / "index.html").write_text(
            '<html><head><meta name="csp-nonce" content="__CSP_NONCE__"></head><body>Console</body></html>',
            encoding="utf-8",
        )
        self.transport = _FakeTransport()
        self.sessions = SessionRegistry("b" * 43)
        self.state = GatewayState(
            transport=self.transport,
            static_dir=self.static,
            sessions=self.sessions,
        )
        self.client = TestClient(create_app(self.state), base_url="http://127.0.0.1")

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def _bootstrap(self) -> str:
        response = self.client.post(
            "/api/v1/session/bootstrap",
            json={"schema_version": 1, "token": "b" * 43},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return str(response.json()["data"]["csrf_token"])

    def _mutation_headers(self, csrf: str) -> dict[str, str]:
        return {
            "Origin": "http://127.0.0.1",
            "Sec-Fetch-Site": "same-origin",
            "X-CSRF-Token": csrf,
        }

    def test_health_and_spa_have_strict_security_headers(self) -> None:
        health = self.client.get("/api/v1/health")
        self.assertEqual(health.status_code, 200)
        self.assertIn("default-src 'none'", health.headers["content-security-policy"])
        self.assertEqual(health.headers["x-frame-options"], "DENY")
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertNotIn("__CSP_NONCE__", page.text)
        self.assertIn('meta name="csp-nonce"', page.text)
        self.assertIn("script-src 'nonce-", page.headers["content-security-policy"])

    def test_runtime_requires_one_time_session_and_returns_snapshot(self) -> None:
        self.assertEqual(self.client.get("/api/v1/runtime").status_code, 401)
        self._bootstrap()
        second = self.client.post(
            "/api/v1/session/bootstrap",
            json={"schema_version": 1, "token": "b" * 43},
        )
        self.assertEqual(second.status_code, 401)
        runtime = self.client.get("/api/v1/runtime")
        self.assertEqual(runtime.status_code, 200)
        self.assertEqual(runtime.json()["data"]["snapshot"]["status"], "STABLE")
        self.assertEqual(self.client.get("/api/v1/tasks").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/experiments").status_code, 200)

    def test_mutation_requires_origin_fetch_metadata_and_csrf(self) -> None:
        csrf = self._bootstrap()
        url = "/api/v1/operations/RUN_START/prepare"
        operation_id = str(uuid.uuid4())
        body = {
            "schema_version": 1,
            "operation_id": operation_id,
            "runtime_identity_digest": _snapshot()["runtime_identity"]["runtime_identity_digest"],
            "parameters": {"profile": "pro", "proposal_only": False},
        }
        self.assertEqual(self.client.post(url, json=body).status_code, 403)
        response = self.client.post(
            url,
            json=body,
            headers={
                "Origin": "http://127.0.0.1",
                "Sec-Fetch-Site": "same-origin",
                "X-CSRF-Token": csrf,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        prepared = response.json()["data"]["prepared_operation"]
        confirm = self.client.post(
            f"/api/v1/operations/{operation_id}/confirm",
            json={
                "schema_version": 1,
                "operation_digest": prepared["operation_digest"],
                "confirmation_phrase": prepared["confirmation_phrase"],
            },
            headers={
                "Origin": "http://127.0.0.1",
                "Sec-Fetch-Site": "same-origin",
                "X-CSRF-Token": csrf,
            },
        )
        self.assertEqual(confirm.status_code, 200, confirm.text)
        status = self.client.get(f"/api/v1/operations/{operation_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["data"]["operation"]["status"], "EXECUTING")

    def test_invalid_host_is_rejected_before_routing(self) -> None:
        response = self.client.get(
            "/api/v1/health", headers={"Host": "attacker.example"}
        )
        self.assertEqual(response.status_code, 400)

    def test_all_read_models_and_exact_artifact_route_are_session_bound(self) -> None:
        self._bootstrap()
        expected_ok = (
            "/api/v1/profiles",
            "/api/v1/tasks",
            "/api/v1/tasks/run/run-1",
            "/api/v1/campaigns",
            "/api/v1/campaigns/campaign-1",
            "/api/v1/resources",
            "/api/v1/soak",
            "/api/v1/audit",
            "/api/v1/drafts",
            "/api/v1/artifacts/source-bundle-v1:fixture",
        )
        for path in expected_ok:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.client.get("/api/v1/tasks/run/missing").status_code, 404
        )
        self.assertEqual(
            self.client.get("/api/v1/experiments/missing").status_code, 404
        )
        self.assertEqual(
            self.client.get("/api/v1/campaigns/missing").status_code, 404
        )
        self.assertEqual(self.client.get("/api/v1/no-file.js").status_code, 404)

    def test_draft_contract_and_http_json_bounds_fail_closed(self) -> None:
        csrf = self._bootstrap()
        headers = self._mutation_headers(csrf)
        draft_id = str(uuid.uuid4())
        draft = {
            "schema_version": 1,
            "task_kind": "AUTONOMOUS_RUN",
            "title": "Pro proposal-only",
            "values": {"profile": "pro", "proposal_only": True},
            "created_at": "2026-08-24T00:00:00Z",
            "updated_at": "2026-08-24T00:00:00Z",
        }
        saved = self.client.put(
            f"/api/v1/drafts/{draft_id}", json=draft, headers=headers
        )
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(
            self.client.get("/api/v1/drafts").json()["data"]["drafts"][0]["draft_id"],
            draft_id,
        )
        malformed = self.client.put(
            f"/api/v1/drafts/{draft_id}",
            content=b'{"schema_version":NaN}',
            headers={**headers, "Content-Type": "application/json"},
        )
        self.assertEqual(malformed.status_code, 400)
        mismatch = self.client.put(
            f"/api/v1/drafts/{draft_id}", json={"schema_version": 1}, headers=headers
        )
        self.assertEqual(mismatch.status_code, 400)
        too_large = self.client.post(
            "/api/v1/session/bootstrap",
            content=b"{}",
            headers={"Content-Length": str(1024 * 1024 + 1)},
        )
        self.assertEqual(too_large.status_code, 413)

    def test_session_expiry_and_unstable_snapshot_disable_writes(self) -> None:
        csrf = self._bootstrap()
        with mock.patch(
            "kernel_research.console.gateway._now",
            return_value=time.monotonic() + 13 * 60 * 60,
        ):
            self.assertEqual(
                self.client.get("/api/v1/runtime").status_code,
                401,
            )

        # Use a fresh app because the bootstrap token and session are one-use.
        self.tearDown()
        self.setUp()
        csrf = self._bootstrap()
        unstable = _snapshot()
        unstable["status"] = "CHANGING"
        with mock.patch.object(self.state.cache, "get", return_value=unstable):
            response = self.client.post(
                "/api/v1/operations/RUN_START/prepare",
                json={
                    "schema_version": 1,
                    "operation_id": str(uuid.uuid4()),
                    "runtime_identity_digest": _snapshot()["runtime_identity"]["runtime_identity_digest"],
                    "parameters": {"profile": "pro", "proposal_only": False},
                },
                headers=self._mutation_headers(csrf),
            )
        self.assertEqual(response.status_code, 409)


@unittest.skipUnless(_FASTAPI_AVAILABLE, _CONSOLE_EXTRA_REQUIRED)
class ConsoleGatewayCliTests(unittest.TestCase):
    def test_gateway_cli_binds_loopback_random_port_and_never_shells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = SimpleNamespace(
                ensure_data_dir=lambda: root,
                poll_interval_ms=2000,
            )
            listener_calls: list[object] = []

            class Listener:
                def setsockopt(self, *args: object) -> None:
                    listener_calls.append(args)

                def bind(self, address: object) -> None:
                    listener_calls.append(("bind", address))

                def listen(self, backlog: int) -> None:
                    listener_calls.append(("listen", backlog))

                def getsockname(self) -> tuple[str, int]:
                    return ("127.0.0.1", 43210)

            class Server:
                def __init__(self, config: object) -> None:
                    self.config = config

                def run(self, *, sockets: list[object]) -> None:
                    self.sockets = sockets

            state = SimpleNamespace(
                sessions=SimpleNamespace(bootstrap_token="bootstrap-token")
            )
            output = io.StringIO()
            with (
                mock.patch.object(gateway_cli.GatewayConfig, "load", return_value=config),
                mock.patch.object(gateway_cli, "SSHAgentTransport", return_value=object()),
                mock.patch.object(gateway_cli, "ConsoleLocalStore", return_value=object()),
                mock.patch("kernel_research.console.gateway.GatewayState", return_value=state),
                mock.patch("kernel_research.console.gateway.create_app", return_value=object()),
                mock.patch.object(gateway_cli.socket, "socket", return_value=Listener()),
                mock.patch("uvicorn.Config", return_value=object()),
                mock.patch("uvicorn.Server", Server),
                mock.patch("sys.stdout", output),
            ):
                self.assertEqual(
                    gateway_cli.main(
                        ["--config", str(root / "config.json"), "--no-browser"]
                    ),
                    0,
                )
            self.assertIn("CONSOLE_URL=http://127.0.0.1:43210/", output.getvalue())
            self.assertIn(("bind", ("127.0.0.1", 0)), listener_calls)
            self.assertEqual(stat.S_IMODE((root / "gateway.lock").stat().st_mode), 0o600)

    def test_gateway_cli_rejects_invalid_port_before_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid TCP port"):
            gateway_cli.main(["--port", "65536", "--no-browser"])


@unittest.skipUnless(_FASTAPI_AVAILABLE, _CONSOLE_EXTRA_REQUIRED)
class ConsoleSharedSseTests(unittest.IsolatedAsyncioTestCase):
    async def test_tabs_share_one_sampler_and_receive_same_snapshot(self) -> None:
        cache = mock.MagicMock()
        cache.get.return_value = _snapshot()
        cache.transport = SimpleNamespace()
        broker = SnapshotEventBroker(cache, poll_interval_ms=1000)
        broker.poll_interval = 0.05

        class Request:
            headers: dict[str, str] = {}

            async def is_disconnected(self) -> bool:
                return False

        first = broker.stream(Request())
        second = broker.stream(Request())
        first_value, second_value = await asyncio.gather(
            first.__anext__(), second.__anext__()
        )
        self.assertEqual(first_value, second_value)
        self.assertIn("event: snapshot", first_value)
        self.assertEqual(cache.get.call_count, 1)
        await first.aclose()
        await second.aclose()
        await asyncio.sleep(0)

    async def test_sampler_emits_reset_without_fabricating_snapshot(self) -> None:
        cache = mock.MagicMock()
        cache.get.side_effect = RuntimeError("identity changed")
        cache.transport = SimpleNamespace()
        broker = SnapshotEventBroker(cache, poll_interval_ms=1000)

        class Request:
            headers: dict[str, str] = {}

            async def is_disconnected(self) -> bool:
                return False

        stream = broker.stream(Request())
        value = await stream.__anext__()
        self.assertIn("event: reset", value)
        self.assertNotIn("event: snapshot", value)
        await stream.aclose()
        await asyncio.sleep(0)

    def test_subscription_identity_change_requires_gateway_restart(self) -> None:
        cache = mock.MagicMock()
        broker = SnapshotEventBroker(cache, poll_interval_ms=1000)
        broker._verify_runtime_identity(_snapshot())
        changed = _snapshot(commit="b" * 40)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            broker._verify_runtime_identity(changed)
        self.assertEqual(broker._last_snapshot_event, "")


@unittest.skipUnless(_FASTAPI_AVAILABLE, _CONSOLE_EXTRA_REQUIRED)
class ConsoleKeysetPaginationTests(unittest.TestCase):
    def test_cursor_binds_section_runtime_and_high_water(self) -> None:
        snapshot = _snapshot()
        items = [{"id": index} for index in (3, 2, 1)]
        first, info = _paginate(
            snapshot,
            section="experiments",
            items=items,
            keys=["3", "2", "1"],
            limit=2,
            after=None,
        )
        self.assertEqual([item["id"] for item in first], [3, 2])
        self.assertIsInstance(info["next_after"], str)
        second, final = _paginate(
            snapshot,
            section="experiments",
            items=items,
            keys=["3", "2", "1"],
            limit=2,
            after=info["next_after"],
        )
        self.assertEqual(second, [{"id": 1}])
        self.assertIsNone(final["next_after"])
        changed = _snapshot()
        changed["cursor"] = {"controller_event_id": 2}
        with self.assertRaisesRegex(ValueError, "another snapshot"):
            _paginate(
                changed,
                section="experiments",
                items=items,
                keys=["3", "2", "1"],
                limit=2,
                after=info["next_after"],
            )
        with self.assertRaisesRegex(ValueError, "cursor"):
            _paginate(
                snapshot,
                section="experiments",
                items=items,
                keys=["3", "2", "1"],
                limit=2,
                after="not-a-valid-keyset-token",
            )

    def test_duplicate_keys_and_unbounded_limits_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "not unique"):
            _paginate(
                _snapshot(),
                section="tasks",
                items=[{"id": "same"}, {"id": "same"}],
                keys=["same", "same"],
                limit=50,
                after=None,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            _paginate(
                _snapshot(),
                section="tasks",
                items=[],
                keys=[],
                limit=101,
                after=None,
            )


if __name__ == "__main__":
    unittest.main()
