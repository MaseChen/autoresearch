from __future__ import annotations

from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest import mock

from kernel_research.autorun.deployment import DeploymentBaselinePin
from kernel_research.console import agent as agent_module
from kernel_research.console.agent import handle_request, serve
from kernel_research.console.protocol import (
    AGENT_PROTOCOL_DIGEST,
    CompositeEventCursorV1,
    ConsoleAgentRequestV1,
    ConsoleAgentResponseV1,
    ConsoleSnapshotV1,
    OperationReceiptV1,
    PreparedOperationV1,
    ProblemV1,
    RuntimeIdentityV1,
    strict_json_loads,
)
from kernel_research.console.read_model import (
    _CAMPAIGN_COLUMNS,
    _CONTROLLER_COLUMNS,
    _HISTORY_COLUMNS,
    ConsolePaths,
    ConsoleReadModel,
    ReadOnlyDatabase,
)
from kernel_research.platform.artifacts import ArtifactId
from kernel_research.platform.canonical import canonical_sha256
from kernel_research.platform.identity import BaselineRef, ExecutionEnvironmentDigest
from kernel_research.platform.profiles import LEGACY_RESEARCH_NAMESPACE
from kernel_research.platform.proposal import CandidateBundle
from kernel_research.scoring_shadow import unavailable_scoring_shadow_report
from test_autorun import SEED
from test_scoring_shadow import (
    INCUMBENT_HASH,
    operation_binding,
    qualified_candidate_measurement,
    successful_full_result,
)
from kernel_research.scoring_shadow import (
    project_scoring_shadow_report,
    scoring_shadow_profile_snapshot,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_schema(
    path: Path,
    *,
    version: int,
    columns: dict[str, frozenset[str]],
) -> None:
    connection = sqlite3.connect(path)
    try:
        for table, names in columns.items():
            definitions = []
            for name in sorted(names):
                kind = "INTEGER" if name in {
                    "id",
                    "iteration_id",
                    "history_experiment_id",
                    "replicate_index",
                    "child_index",
                    "fencing_epoch",
                    "generation_id",
                    "generation_index",
                    "promotable",
                    "stop_requested",
                    "valid_candidates",
                    "revision_index",
                    "reserved_candidates",
                    "reserved_wall_ms",
                    "reserved_gpu_ms",
                    "reserved_tokens",
                    "reserved_cost_microusd",
                    "actual_candidates",
                    "actual_wall_ms",
                    "actual_gpu_ms",
                    "actual_tokens",
                    "actual_cost_microusd",
                    "byte_size",
                } else "TEXT"
                definitions.append(f'"{name}" {kind}')
            connection.execute(
                f'CREATE TABLE "{table}" ({", ".join(definitions)})'
            )
        connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
    finally:
        connection.close()


def _identity(commit: str = "a" * 40) -> RuntimeIdentityV1:
    digest = "sha256:" + "1" * 64
    return RuntimeIdentityV1(
        git_commit=commit,
        expected_git_commit=commit,
        config_digest=digest,
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


class ConsoleProtocolTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_nan_nul_and_oversize(self) -> None:
        for raw, message in (
            ('{"x":1,"x":2}', "repeats"),
            ('{"x":NaN}', "constant"),
            (b'{"x":"\x00"}', "NUL"),
            (b"{}", "size bound"),
        ):
            with self.subTest(raw=raw):
                kwargs = {"max_bytes": 1} if raw == b"{}" else {}
                with self.assertRaisesRegex(ValueError, message):
                    strict_json_loads(raw, **kwargs)

    def test_cursor_and_runtime_identity_round_trip(self) -> None:
        identity = _identity()
        cursor = CompositeEventCursorV1(
            controller_event_id=1,
            history_experiment_id=2,
            runtime_identity_digest=identity.digest,
        )
        self.assertEqual(
            CompositeEventCursorV1.from_value(cursor.to_dict()), cursor
        )
        self.assertEqual(identity.to_dict()["runtime_identity_digest"], identity.digest)
        self.assertEqual(RuntimeIdentityV1.from_value(identity.to_dict()), identity)
        tampered = identity.to_dict()
        tampered["runtime_identity_digest"] = "sha256:" + "9" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            RuntimeIdentityV1.from_value(tampered)
        drifted = identity.to_dict()
        drifted["agent_protocol_digest"] = "sha256:" + "8" * 64
        drifted.pop("runtime_identity_digest")
        drifted["runtime_identity_digest"] = canonical_sha256(
            {key: value for key, value in drifted.items() if key != "runtime_identity_digest"}
        )
        with self.assertRaisesRegex(ValueError, "protocol digest"):
            RuntimeIdentityV1.from_value(drifted)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            CompositeEventCursorV1(controller_event_id=-1)

    def test_agent_contract_rejects_unknown_operations_and_shapes(self) -> None:
        request = ConsoleAgentRequestV1.from_value(
            {
                "schema_version": 1,
                "request_id": "request-1",
                "operation": "snapshot",
                "payload": {"limit": 5},
            }
        )
        self.assertEqual(request.operation, "snapshot")
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            ConsoleAgentRequestV1("request-2", "shell", {})
        with self.assertRaisesRegex(ValueError, "requires only payload"):
            ConsoleAgentResponseV1(
                request_id="request-3", status="SUCCESS", payload=None
            )
        problem = ProblemV1("FAIL_CLOSED", "Failed", "No authority")
        response = ConsoleAgentResponseV1(
            request_id="request-4", status="ERROR", problem=problem
        )
        self.assertEqual(response.to_dict()["problem"]["code"], "FAIL_CLOSED")

    def test_agent_import_is_dependency_light(self) -> None:
        source = """
import sys
import kernel_research.console.agent
blocked = sorted(set(sys.modules) & {'numpy', 'torch', 'triton', 'fastapi', 'uvicorn'})
if blocked:
    raise SystemExit(','.join(blocked))
"""
        process = subprocess.run(
            [sys.executable, "-c", source],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)


class ConsoleReadModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _paths(self) -> ConsolePaths:
        state = self.root / "state"
        controller = self.root / "controller"
        checkpoint = self.root / "checkpoints"
        campaign_dir = self.root / "campaign"
        for directory in (state, controller, checkpoint, campaign_dir):
            directory.mkdir(mode=0o700)
        controller_db = controller / "controller.sqlite3"
        history_db = state / "history.sqlite3"
        campaign_db = campaign_dir / "campaign.sqlite3"
        _create_schema(
            controller_db,
            version=3,
            columns=dict(_CONTROLLER_COLUMNS),
        )
        _create_schema(history_db, version=3, columns=dict(_HISTORY_COLUMNS))
        _create_schema(campaign_db, version=1, columns=dict(_CAMPAIGN_COLUMNS))
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        environment = ExecutionEnvironmentDigest.resolved(
            evaluator_image_digest="sha256:" + "1" * 64,
            toolchain_digest="sha256:" + "2" * 64,
            framework_digest="sha256:" + "3" * 64,
            operator_abi_digest="sha256:" + "4" * 64,
            build_flags_digest="sha256:" + "5" * 64,
        )
        candidate = "a" * 64
        baseline = BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=ArtifactId.source_sha256(candidate),
            source="deployment",
            revision="fixture",
            execution_environment=environment,
        )
        pin = DeploymentBaselinePin.create(
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            baseline_ref=baseline,
            candidate_hash=candidate,
            git_commit=commit,
            primary_experiment_uid=None,
            confirmation_experiment_uid="fixture-confirmation",
            confirmation_experiment_id=1,
            parent_baseline_ref=None,
            execution_environment=environment,
        )
        pin_path = self.root / "deployment-baseline.json"
        pin_path.write_text(json.dumps(pin.to_dict()), encoding="utf-8")
        pin_path.chmod(0o600)
        manifest = self.root / "admin.json"
        manifest.write_text("{}", encoding="utf-8")
        manifest.chmod(0o600)
        configs = []
        for index in range(5):
            path = self.root / f"config-{index}.json"
            path.write_text("{}", encoding="utf-8")
            path.chmod(0o600)
            configs.append(path)
        return ConsolePaths(
            manifest_path=manifest,
            repository_dir=Path(__file__).resolve().parents[1],
            runtime_root=self.root,
            state_dir=state,
            controller_dir=controller,
            checkpoint_dir=checkpoint,
            controller_db=controller_db,
            history_db=history_db,
            campaign_db=campaign_db,
            deployment_pin=pin_path,
            config_paths=tuple(configs),
            expected_git_commit=commit,
        )

    def test_read_only_database_rejects_schema_and_column_drift(self) -> None:
        path = self.root / "minimal.sqlite3"
        _create_schema(path, version=3, columns={"runs": frozenset({"id"})})
        database = ReadOnlyDatabase(
            path,
            expected_version=3,
            required_columns={"runs": frozenset({"id", "status"})},
        )
        with self.assertRaisesRegex(ValueError, "missing columns"):
            database.connect()
        self.assertFalse(Path(str(path) + "-wal").exists())

    def test_snapshot_is_stable_bounded_and_has_zero_database_side_effects(self) -> None:
        paths = self._paths()
        before = {
            path: (path.stat().st_size, _sha(path), stat.S_IMODE(path.stat().st_mode))
            for path in (paths.controller_db, paths.history_db, paths.campaign_db)
        }
        model = ConsoleReadModel(paths)
        health = model.health(deep=True)
        snapshot = model.snapshot(limit=10)
        self.assertEqual(health["status"], "READY")
        self.assertEqual(snapshot.status, "STABLE")
        self.assertEqual(snapshot.identity.git_commit, paths.expected_git_commit)
        self.assertEqual(snapshot.data["runs"], [])
        after = {
            path: (path.stat().st_size, _sha(path), stat.S_IMODE(path.stat().st_mode))
            for path in before
        }
        self.assertEqual(after, before)
        for path in before:
            self.assertFalse(Path(str(path) + "-wal").exists())
            self.assertFalse(Path(str(path) + "-shm").exists())

    def test_snapshot_projects_shadow_score_without_exposing_full_result(self) -> None:
        paths = self._paths()
        report = unavailable_scoring_shadow_report(
            reason="PROFILE_NOT_FROZEN",
            profile=None,
            candidate_hash="a" * 64,
            incumbent_history_experiment_id=210,
            incumbent_candidate_hash="b" * 64,
        )
        with closing(sqlite3.connect(paths.history_db)) as connection, connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    id, status, candidate_hash, suite, result_json
                ) VALUES (1, 'SUCCESS', ?, 'full', ?)
                """,
                ("a" * 64, json.dumps({"objective_scoring": report})),
            )
        before = _sha(paths.history_db)
        snapshot = ConsoleReadModel(paths).snapshot(limit=10)
        self.assertEqual(_sha(paths.history_db), before)
        experiment = snapshot.data["experiments"][0]
        self.assertEqual(experiment["xpuoj_proxy_status"], "UNAVAILABLE")
        self.assertEqual(
            experiment["xpuoj_proxy_reason"], "PROFILE_NOT_FROZEN"
        )
        self.assertIsNone(experiment["xpuoj_proxy_score"])
        self.assertFalse(experiment["xpuoj_proxy_promotion_authority"])
        self.assertNotIn("result", experiment)
        self.assertNotIn("objective_scoring", experiment)

        measurement = qualified_candidate_measurement()
        available = project_scoring_shadow_report(
            successful_full_result(),
            suite="full",
            profile=scoring_shadow_profile_snapshot(),
            incumbent_history_experiment_id=210,
            incumbent_candidate_hash=INCUMBENT_HASH,
            candidate_measurement=measurement,
            candidate_framework_git_commit="c" * 40,
            **operation_binding(measurement),
        )
        with closing(sqlite3.connect(paths.history_db)) as connection, connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    id, status, candidate_hash, suite, result_json
                ) VALUES (2, 'SUCCESS', ?, 'full', ?)
                """,
                ("a" * 64, json.dumps({"objective_scoring": available})),
            )
        projected = ConsoleReadModel(paths).snapshot(limit=10).data["experiments"][0]
        self.assertEqual(projected["xpuoj_proxy_status"], "AVAILABLE")
        self.assertEqual(projected["xpuoj_proxy_score"], 50.0)
        serialized = json.dumps(projected, sort_keys=True)
        for private_field in (
            "candidate_measurement",
            "environment_snapshot_digest",
            "candidate_anchor_measurement",
            "paired_safety_measurement",
            '"cases"',
            '"rounds"',
        ):
            self.assertNotIn(private_field, serialized)

    def test_scientific_artifact_requires_verified_success_bundle(self) -> None:
        paths = self._paths()
        bundle = CandidateBundle.single_file(content=SEED)
        relative = Path("objects") / "fixture-bundle"
        object_path = paths.state_dir / relative
        object_path.parent.mkdir(mode=0o700)
        object_path.write_bytes(bundle.bundle_bytes)
        object_path.chmod(0o600)
        with closing(sqlite3.connect(paths.history_db)) as connection, connection:
            connection.execute(
                """
                INSERT INTO candidate_artifacts (
                    artifact_id, artifact_kind, content_sha256, object_path,
                    byte_size, manifest_json
                ) VALUES (?, 'source_bundle_v1', ?, ?, ?, ?)
                """,
                (
                    str(bundle.artifact_id),
                    hashlib.sha256(bundle.bundle_bytes).hexdigest(),
                    relative.as_posix(),
                    len(bundle.bundle_bytes),
                    json.dumps(bundle.manifest),
                ),
            )
            connection.execute(
                "INSERT INTO experiments (id, artifact_id, status) VALUES (1, ?, 'SUCCESS')",
                (str(bundle.artifact_id),),
            )
        model = ConsoleReadModel(paths)
        result = model.scientific_artifact(str(bundle.artifact_id))
        self.assertEqual(result["source"], SEED)
        self.assertEqual(result["manifest"], bundle.manifest)
        with self.assertRaisesRegex(ValueError, "V2 bundle"):
            model.scientific_artifact(str(ArtifactId.for_source_text(SEED)))
        object_path.write_bytes(bundle.bundle_bytes + b"\n")
        with self.assertRaisesRegex(ValueError, "metadata"):
            model.scientific_artifact(str(bundle.artifact_id))


class _FakeReadModel:
    def __init__(self) -> None:
        self.identity = _identity()

    def runtime_identity(self) -> RuntimeIdentityV1:
        return self.identity

    def health(self, *, deep: bool = False) -> dict[str, object]:
        return {"schema_version": 1, "status": "READY", "deep": deep}

    def snapshot(self, *, limit: int = 50) -> ConsoleSnapshotV1:
        return ConsoleSnapshotV1(
            status="STABLE",
            identity=self.identity,
            cursor=CompositeEventCursorV1(
                runtime_identity_digest=self.identity.digest
            ),
            observed_at="2026-08-24T00:00:00Z",
            data={"limit": limit},
        )

    def scientific_artifact(self, artifact_id: object) -> dict[str, object]:
        return {"artifact_id": artifact_id, "source": SEED}


class _FakeOperations:
    def prepare(self, payload: dict[str, object]) -> PreparedOperationV1:
        return PreparedOperationV1(
            operation_id=str(payload["operation_id"]),
            kind=str(payload["kind"]),
            operation_digest="sha256:" + "8" * 64,
            runtime_identity_digest="sha256:" + "1" * 64,
            prepared_at="2026-08-24T00:00:00Z",
            expires_epoch=9999999999.0,
            confirmation_phrase="确认",
            impact={"gpu_possible": True},
        )

    def execute(self, payload: dict[str, object]) -> OperationReceiptV1:
        return self._receipt(payload, "EXECUTING")

    def reconcile(self, payload: dict[str, object]) -> OperationReceiptV1:
        return self._receipt(payload, "SUCCEEDED")

    def _receipt(
        self, payload: dict[str, object], status: str
    ) -> OperationReceiptV1:
        return OperationReceiptV1(
            operation_id=str(payload["operation_id"]),
            kind="RUN_START",
            operation_digest=str(
                payload.get("operation_digest", "sha256:" + "8" * 64)
            ),
            status=status,
            observed_at="2026-08-24T00:00:01Z",
            domain_identity={"run_id": "console-run"},
        )


class ConsoleAgentTests(unittest.TestCase):
    def test_handshake_and_snapshot_are_strict(self) -> None:
        model = _FakeReadModel()
        handshake = handle_request(
            ConsoleAgentRequestV1("request-1", "handshake", {}), model=model
        )
        self.assertEqual(handshake.status, "SUCCESS")
        self.assertEqual(handshake.payload["health"]["status"], "READY")
        snapshot = handle_request(
            ConsoleAgentRequestV1("request-2", "snapshot", {"limit": 7}),
            model=model,
        )
        self.assertEqual(snapshot.payload["snapshot"]["data"]["limit"], 7)
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            handle_request(
                ConsoleAgentRequestV1("request-3", "snapshot", {"sql": "SELECT"}),
                model=model,
            )

    def test_serve_returns_one_bounded_error_then_valid_response(self) -> None:
        invalid = b'{"schema_version":1,"schema_version":1}\n'
        valid = json.dumps(
            {
                "schema_version": 1,
                "request_id": "request-4",
                "operation": "handshake",
                "payload": {},
            },
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        output = io.BytesIO()
        result = serve(
            model=_FakeReadModel(),
            input_stream=io.BytesIO(invalid + valid),
            output_stream=output,
        )
        self.assertEqual(result, 0)
        responses = [strict_json_loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([item["status"] for item in responses], ["ERROR", "SUCCESS"])
        self.assertEqual(responses[0]["request_id"], "invalid-request")

    def test_artifact_and_write_dispatch_are_exact(self) -> None:
        model = _FakeReadModel()
        operations = _FakeOperations()
        artifact = handle_request(
            ConsoleAgentRequestV1(
                "request-artifact",
                "snapshot",
                {"artifact_id": "source-bundle-v1:fixture"},
            ),
            model=model,
        )
        self.assertEqual(
            artifact.payload["scientific_artifact"]["source"], SEED
        )
        with self.assertRaisesRegex(ValueError, "may not be combined"):
            handle_request(
                ConsoleAgentRequestV1(
                    "request-bad-artifact",
                    "snapshot",
                    {"artifact_id": "fixture", "limit": 1},
                ),
                model=model,
            )
        operation_id = str(uuid.uuid4())
        prepared = handle_request(
            ConsoleAgentRequestV1(
                "request-prepare",
                "prepare",
                {
                    "operation_id": operation_id,
                    "kind": "RUN_START",
                    "runtime_identity_digest": model.identity.digest,
                    "parameters": {"profile": "pro", "proposal_only": True},
                },
            ),
            model=model,
            operations=operations,
        )
        self.assertEqual(prepared.payload["prepared_operation"]["kind"], "RUN_START")
        executed = handle_request(
            ConsoleAgentRequestV1(
                "request-execute",
                "execute",
                {
                    "operation_id": operation_id,
                    "operation_digest": "sha256:" + "8" * 64,
                    "confirmation_phrase": "确认",
                },
            ),
            model=model,
            operations=operations,
        )
        self.assertEqual(executed.payload["operation_receipt"]["status"], "EXECUTING")
        reconciled = handle_request(
            ConsoleAgentRequestV1(
                "request-reconcile",
                "reconcile",
                {"operation_id": operation_id},
            ),
            model=model,
            operations=operations,
        )
        self.assertEqual(reconciled.payload["operation_receipt"]["status"], "SUCCEEDED")
        with self.assertRaisesRegex(ValueError, "not configured"):
            handle_request(
                ConsoleAgentRequestV1(
                    "request-no-writes", "reconcile", {"operation_id": operation_id}
                ),
                model=model,
            )

    def test_agent_main_reports_bounded_startup_failure(self) -> None:
        output = io.BytesIO()
        fake_stdout = SimpleNamespace(buffer=output)
        with (
            mock.patch.object(
                agent_module.ConsolePaths,
                "from_admin_manifest",
                side_effect=ValueError("bad manifest"),
            ),
            mock.patch.object(agent_module.sys, "stdout", fake_stdout),
        ):
            self.assertEqual(
                agent_module.main(["--admin-manifest", "/tmp/missing-admin.json"]),
                2,
            )
        response = strict_json_loads(output.getvalue())
        self.assertEqual(response["status"], "ERROR")
        self.assertEqual(response["problem"]["code"], "CONSOLE_REQUEST_REJECTED")

    def test_agent_main_constructs_dependency_light_service(self) -> None:
        paths = object()
        model = _FakeReadModel()
        operations = _FakeOperations()
        with (
            mock.patch.object(
                agent_module.ConsolePaths, "from_admin_manifest", return_value=paths
            ),
            mock.patch.object(agent_module, "ConsoleReadModel", return_value=model),
            mock.patch(
                "kernel_research.console.operations.ConsoleOperationService",
                return_value=operations,
            ),
            mock.patch.object(agent_module, "serve", return_value=0) as selected,
        ):
            self.assertEqual(
                agent_module.main(["--admin-manifest", "/tmp/admin.json"]), 0
            )
        self.assertIs(selected.call_args.kwargs["model"], model)
        self.assertIs(selected.call_args.kwargs["operations"], operations)

    def test_subscribe_streams_until_the_ssh_consumer_disconnects(self) -> None:
        request = json.dumps(
            {
                "schema_version": 1,
                "request_id": "subscribe-1",
                "operation": "subscribe",
                "payload": {"limit": 10},
            },
            separators=(",", ":"),
        ).encode() + b"\n"

        class Output(io.BytesIO):
            writes = 0

            def write(selected, value: bytes) -> int:
                selected.writes += 1
                if selected.writes > 1:
                    raise BrokenPipeError
                return super().write(value)

        output = Output()
        with mock.patch.object(agent_module.time, "sleep", return_value=None):
            self.assertEqual(
                serve(
                    model=_FakeReadModel(),
                    input_stream=io.BytesIO(request),
                    output_stream=output,
                ),
                0,
            )
        response = strict_json_loads(output.getvalue())
        self.assertEqual(response["payload"]["snapshot"]["status"], "STABLE")


if __name__ == "__main__":
    unittest.main()
