from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import uuid
from unittest import mock

from kernel_research.autorun.admin import AdminManifest
from kernel_research.console.local_store import ConsoleLocalStore
from kernel_research.console import operations as operation_module
from kernel_research.console.operations import ConsoleOperationService
from kernel_research.console.protocol import (
    DraftV1,
    ManualEvaluationRequestV1,
    OperationReceiptV1,
    PreparedOperationV1,
)
from kernel_research.console.read_model import ConsoleReadModel
from kernel_research.console import worker as worker_module
from kernel_research.console import write_service
from kernel_research.platform.artifacts import ArtifactId
from kernel_research.platform.identity import BaselineRef, ExecutionEnvironmentDigest
from kernel_research.platform.profiles import CURRENT_RESEARCH_NAMESPACE
from kernel_research.platform.proposal import CandidateBundle

from test_autorun import SEED
import test_console_v1 as console_fixtures


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ConsoleLocalStoreTests(unittest.TestCase):
    def test_drafts_candidate_cas_receipts_and_audit_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            store = ConsoleLocalStore(root)
            self.assertEqual(stat.S_IMODE(store.database.stat().st_mode), 0o600)
            bundle = CandidateBundle.single_file(content=SEED)
            self.assertEqual(
                store.store_candidate_bundle(bundle), str(bundle.artifact_id)
            )
            self.assertEqual(
                store.store_candidate_bundle(bundle), str(bundle.artifact_id)
            )
            timestamp = _now()
            draft = DraftV1(
                draft_id=str(uuid.uuid4()),
                task_kind="MANUAL_EVALUATION",
                title="CURRENT candidate",
                values={"candidate_artifact_id": str(bundle.artifact_id)},
                created_at=timestamp,
                updated_at=timestamp,
            )
            store.put_draft(draft)
            self.assertEqual(store.list_drafts()[0]["draft_digest"], draft.digest)
            operation_id = str(uuid.uuid4())
            prepared = PreparedOperationV1(
                operation_id=operation_id,
                kind="MANUAL_EVALUATION_START",
                operation_digest="sha256:" + "1" * 64,
                runtime_identity_digest="sha256:" + "2" * 64,
                prepared_at=timestamp,
                expires_epoch=9999999999.0,
                confirmation_phrase="启动手工 CURRENT 评测",
                impact={"gpu_possible": True},
            )
            store.record_prepared(prepared)
            receipt = OperationReceiptV1(
                operation_id=operation_id,
                kind=prepared.kind,
                operation_digest=prepared.operation_digest,
                status="EXECUTING",
                observed_at=timestamp,
                domain_identity={"run_id": "console-run"},
            )
            store.record_receipt(receipt)
            self.assertEqual(store.get_operation(operation_id)["status"], "EXECUTING")
            audit = list(reversed(store.audit_records()))
            self.assertIsNone(audit[0]["previous_digest"])
            for previous, current in zip(audit, audit[1:]):
                self.assertEqual(current["previous_digest"], previous["entry_digest"])

    def test_local_store_rejects_collision_and_status_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ConsoleLocalStore(Path(temporary).resolve())
            timestamp = _now()
            operation_id = str(uuid.uuid4())
            prepared = PreparedOperationV1(
                operation_id,
                "RUN_START",
                "sha256:" + "1" * 64,
                "sha256:" + "2" * 64,
                timestamp,
                9999999999.0,
                "启动自治 Run",
                {},
            )
            store.record_prepared(prepared)
            terminal = OperationReceiptV1(
                operation_id,
                prepared.kind,
                prepared.operation_digest,
                "FAILED",
                timestamp,
                {},
            )
            store.record_receipt(terminal)
            with self.assertRaisesRegex(ValueError, "state"):
                store.record_receipt(
                    OperationReceiptV1(
                        operation_id,
                        prepared.kind,
                        prepared.operation_digest,
                        "EXECUTING",
                        timestamp,
                        {},
                    )
                )


class ConsoleRemoteOperationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = console_fixtures.ConsoleReadModelTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.paths = self.fixture._paths()
        self.model = ConsoleReadModel(self.paths)
        self.service = ConsoleOperationService(self.paths, self.model)

    def test_manual_prepare_is_immutable_and_stale_identity_fails(self) -> None:
        operation_id = str(uuid.uuid4())
        bundle = CandidateBundle.single_file(content=SEED)
        request = ManualEvaluationRequestV1(
            operation_id=operation_id,
            candidate=bundle,
            runtime_identity_digest=self.model.runtime_identity().digest,
        )
        payload = {
            "operation_id": operation_id,
            "kind": "MANUAL_EVALUATION_START",
            "runtime_identity_digest": request.runtime_identity_digest,
            "parameters": {"candidate": bundle.to_dict()},
        }
        prepared = self.service.prepare(payload)
        replay = self.service.prepare(payload)
        self.assertEqual(replay, prepared)
        self.assertEqual(
            prepared.impact["unknown_outcome_replay"], "FORBIDDEN"
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            self.service.prepare(
                {**payload, "runtime_identity_digest": "sha256:" + "0" * 64}
            )

    def test_execute_publishes_detached_identity_without_shell(self) -> None:
        operation_id = str(uuid.uuid4())
        runtime_identity = self.model.runtime_identity()
        identity = runtime_identity.digest
        prepared = self.service.prepare(
            {
                "operation_id": operation_id,
                "kind": "MANUAL_EVALUATION_START",
                "runtime_identity_digest": identity,
                "parameters": {
                    "candidate": CandidateBundle.single_file(content=SEED).to_dict()
                },
            }
        )
        observed: list[object] = []

        class Process:
            pid = os.getpid()

        def popen(argv: list[str], **kwargs: object) -> Process:
            observed.extend(argv)
            self.assertIs(kwargs["shell"], False)
            self.assertIs(kwargs["start_new_session"], True)
            return Process()

        with (
            mock.patch(
                "kernel_research.console.operations._host_python",
                return_value=Path(sys.executable),
            ),
            mock.patch.object(
                self.model,
                "runtime_identity",
                return_value=runtime_identity,
            ),
            mock.patch("subprocess.Popen", side_effect=popen),
        ):
            receipt = self.service.execute(
                {
                    "operation_id": operation_id,
                    "operation_digest": prepared.operation_digest,
                    "confirmation_phrase": prepared.confirmation_phrase,
                }
            )
        self.assertEqual(receipt.status, "EXECUTING")
        self.assertIn("kernel_research.console.worker", observed)
        self.assertNotIn("sh", observed)
        self.assertEqual(
            self.service.reconcile({"operation_id": operation_id}).status,
            "EXECUTING",
        )

    def _prepare_run(self, **overrides: object) -> PreparedOperationV1:
        operation_id = str(uuid.uuid4())
        parameters: dict[str, object] = {
            "profile": "pro",
            "proposal_only": False,
        }
        parameters.update(overrides)
        return self.service.prepare(
            {
                "operation_id": operation_id,
                "kind": "RUN_START",
                "runtime_identity_digest": self.model.runtime_identity().digest,
                "parameters": parameters,
            }
        )

    def test_prepare_and_execute_fail_closed_before_domain_actions(self) -> None:
        identity = self.model.runtime_identity()
        with self.assertRaisesRegex(ValueError, "fields"):
            self.service.prepare({"kind": "RUN_START"})
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            self.service.prepare(
                {
                    "operation_id": str(uuid.uuid4()),
                    "kind": "SHELL",
                    "runtime_identity_digest": identity.digest,
                    "parameters": {},
                }
            )
        with self.assertRaisesRegex(ValueError, "canonical UUID"):
            self.service._path("NOT-A-UUID")
        with mock.patch.object(
            self.model, "health", return_value={"status": "CHANGING"}
        ):
            with self.assertRaisesRegex(ValueError, "not READY"):
                self._prepare_run()

        prepared = self._prepare_run()
        payload = {
            "operation_id": prepared.operation_id,
            "operation_digest": prepared.operation_digest,
            "confirmation_phrase": "wrong",
        }
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.service.execute(payload)
        with mock.patch.object(
            self.model,
            "runtime_identity",
            return_value=replace(identity, git_commit="b" * 40),
        ):
            with self.assertRaisesRegex(ValueError, "runtime identity changed"):
                self.service.execute(
                    {
                        **payload,
                        "confirmation_phrase": prepared.confirmation_phrase,
                    }
                )

        failure = self._prepare_run()
        with (
            mock.patch(
                "kernel_research.console.operations._host_python",
                return_value=Path(sys.executable),
            ),
            mock.patch.object(
                self.model, "runtime_identity", return_value=identity
            ),
            mock.patch("subprocess.Popen", side_effect=OSError("blocked")),
        ):
            receipt = self.service.execute(
                {
                    "operation_id": failure.operation_id,
                    "operation_digest": failure.operation_digest,
                    "confirmation_phrase": failure.confirmation_phrase,
                }
            )
        self.assertEqual(receipt.status, "FAILED")
        self.assertEqual(receipt.problem["code"], "DETACHED_EXECUTOR_START_FAILED")
        self.assertEqual(
            self.service.reconcile({"operation_id": failure.operation_id}), receipt
        )

    def test_expiry_collision_and_dead_executor_are_durable(self) -> None:
        prepared = self._prepare_run()
        with mock.patch("time.time", return_value=prepared.expires_epoch + 1):
            expired = self.service.execute(
                {
                    "operation_id": prepared.operation_id,
                    "operation_digest": prepared.operation_digest,
                    "confirmation_phrase": prepared.confirmation_phrase,
                }
            )
        self.assertEqual(expired.status, "EXPIRED")
        self.assertEqual(
            self.service.execute(
                {
                    "operation_id": prepared.operation_id,
                    "operation_digest": prepared.operation_digest,
                    "confirmation_phrase": prepared.confirmation_phrase,
                }
            ),
            expired,
        )

        collision = self._prepare_run()
        with self.assertRaisesRegex(ValueError, "another intent"):
            self.service.prepare(
                {
                    "operation_id": collision.operation_id,
                    "kind": "RUN_START",
                    "runtime_identity_digest": self.model.runtime_identity().digest,
                    "parameters": {"profile": "flash", "proposal_only": False},
                }
            )

        dead = self._prepare_run()
        path = self.service._path(dead.operation_id)
        record = operation_module._read_record(path)
        record.update({"state": "EXECUTING", "pid": 99999999})
        operation_module._atomic_json(path, record)
        with mock.patch("os.kill", side_effect=ProcessLookupError):
            unknown = self.service.reconcile({"operation_id": dead.operation_id})
        self.assertEqual(unknown.status, "UNKNOWN_OUTCOME")
        self.assertEqual(
            self.service.reconcile({"operation_id": dead.operation_id}), unknown
        )
        with self.assertRaisesRegex(ValueError, "fields"):
            self.service.reconcile({"operation_id": dead.operation_id, "sql": "x"})

    def test_fixed_domain_dtos_validate_every_mutating_family(self) -> None:
        budget = {
            "candidates": 5,
            "wall_ms": 1000,
            "gpu_ms": 1000,
            "tokens": 1000,
            "cost_microusd": 1000,
        }
        valid = {
            "RUN_START": {"profile": "pro", "proposal_only": True},
            "RUN_STOP": {"profile": "flash", "run_id": "run-1"},
            "RUN_RESUME": {"profile": "pro", "run_id": "run-1"},
            "CAMPAIGN_CREATE": {
                "mode": "DISCOVERY",
                "profile": "pro",
                "budget": budget,
            },
            "CAMPAIGN_START": {"campaign_id": "campaign-1"},
            "CAMPAIGN_PAUSE": {"campaign_id": "campaign-1", "reason": "operator"},
            "CAMPAIGN_RESUME": {"campaign_id": "campaign-1", "profile": "pro"},
            "CAMPAIGN_CHILD_EXECUTE": {
                "campaign_id": "campaign-1",
                "profile": "flash",
                "max_candidates": 2,
                "max_wall_seconds": 60,
            },
            "CAMPAIGN_LINEAGE_ADVANCE": {
                "campaign_id": "campaign-1",
                "child_id": "child-1",
            },
            "BENCHMARK_INIT": {
                "profile": "pro",
                "arms": ["pro", "flash"],
                "repetitions": 2,
                "budget": budget,
            },
            "BENCHMARK_EXECUTE": {"campaign_id": "campaign-1", "profile": "flash"},
        }
        for kind, parameters in valid.items():
            with self.subTest(kind=kind):
                self.assertEqual(
                    operation_module._validate_domain_parameters(kind, parameters),
                    parameters,
                )
        invalid = (
            ("RUN_START", []),
            ("RUN_START", {"profile": "raw", "proposal_only": False}),
            ("RUN_START", {"profile": "pro", "proposal_only": 1}),
            ("RUN_STOP", {"profile": "pro", "run_id": "../run"}),
            ("CAMPAIGN_CREATE", {**valid["CAMPAIGN_CREATE"], "mode": "BENCHMARK"}),
            ("CAMPAIGN_CREATE", {**valid["CAMPAIGN_CREATE"], "budget": {**budget, "gpu_ms": 0}}),
            ("CAMPAIGN_CREATE", {**valid["CAMPAIGN_CREATE"], "budget": {**budget, "extra": 1}}),
            ("BENCHMARK_INIT", {**valid["BENCHMARK_INIT"], "arms": ["pro", "pro"]}),
            ("BENCHMARK_INIT", {**valid["BENCHMARK_INIT"], "arms": [{}, "flash"]}),
            ("BENCHMARK_INIT", {**valid["BENCHMARK_INIT"], "repetitions": True}),
            ("CAMPAIGN_CHILD_EXECUTE", {**valid["CAMPAIGN_CHILD_EXECUTE"], "max_candidates": 0}),
            ("CAMPAIGN_PAUSE", {**valid["CAMPAIGN_PAUSE"], "reason": "bad\nreason"}),
            ("CAMPAIGN_LINEAGE_ADVANCE", {**valid["CAMPAIGN_LINEAGE_ADVANCE"], "child_id": "../child"}),
        )
        for kind, parameters in invalid:
            with self.subTest(kind=kind, parameters=parameters):
                with self.assertRaises(ValueError):
                    operation_module._validate_domain_parameters(kind, parameters)


class ConsoleWorkerAndFacadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = console_fixtures.ConsoleReadModelTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.paths = self.fixture._paths()
        self.model = ConsoleReadModel(self.paths)
        self.service = ConsoleOperationService(self.paths, self.model)

    def _worker_record(self) -> tuple[str, Path]:
        operation_id = str(uuid.uuid4())
        prepared = self.service.prepare(
            {
                "operation_id": operation_id,
                "kind": "RUN_START",
                "runtime_identity_digest": self.model.runtime_identity().digest,
                "parameters": {"profile": "pro", "proposal_only": False},
            }
        )
        path = self.service._path(operation_id)
        record = operation_module._read_record(path)
        record.update({"state": "EXECUTING", "pid": os.getpid()})
        operation_module._atomic_json(path, record)
        self.assertEqual(prepared.operation_id, operation_id)
        return operation_id, path

    def test_worker_classifies_success_failure_unknown_and_interrupt(self) -> None:
        outcomes: tuple[tuple[object, str], ...] = (
            (({"run_id": "run-1"}, {"status": "SUCCESS"}), "SUCCEEDED"),
            (({"run_id": "run-1"}, {"status": "FAILED"}), "FAILED"),
            (({"run_id": "run-1"}, {"status": "UNKNOWN_GPU_OUTCOME"}), "UNKNOWN_OUTCOME"),
            (ValueError("bad domain"), "FAILED"),
            (KeyboardInterrupt(), "UNKNOWN_OUTCOME"),
        )
        for outcome, expected in outcomes:
            with self.subTest(expected=expected):
                operation_id, path = self._worker_record()
                effect = outcome if isinstance(outcome, BaseException) else None
                returned = None if effect is not None else outcome
                with (
                    mock.patch("resource.setrlimit"),
                    mock.patch.object(
                        worker_module.ConsolePaths,
                        "from_admin_manifest",
                        return_value=self.paths,
                    ),
                    mock.patch(
                        "kernel_research.console.worker.ConsoleReadModel",
                        return_value=self.model,
                    ),
                    mock.patch.object(
                        worker_module.AdminManifest,
                        "load",
                        return_value=mock.MagicMock(spec=AdminManifest),
                    ),
                    mock.patch(
                        "kernel_research.console.write_service.execute_domain_operation",
                        return_value=returned,
                        side_effect=effect,
                    ),
                ):
                    self.assertEqual(
                        worker_module.run_worker(self.paths.manifest_path, operation_id),
                        0,
                    )
                record = operation_module._read_record(path)
                self.assertEqual(record["receipt"]["status"], expected)

    def test_write_facade_uses_only_fixed_controller_and_campaign_calls(self) -> None:
        runtime_root = self.paths.runtime_root
        manifest = SimpleNamespace(
            runtime_root=runtime_root,
            repository_dir=self.paths.repository_dir,
            pro_config=runtime_root / "pro.json",
            flash_config=runtime_root / "flash.json",
        )
        calls: list[list[str]] = []

        def campaign(argv: list[str]) -> dict[str, object]:
            calls.append(argv)
            return {"status": "SUCCESS"}

        controller = mock.MagicMock()
        controller.start_console_operation.return_value = {
            "id": "console-run",
            "status": "RUNNING",
        }
        controller.stop.return_value = {"status": "STOPPED"}
        controller.resume.return_value = {"status": "RUNNING"}
        common = {"profile": "pro"}
        with (
            mock.patch.object(write_service, "_config", return_value=object()),
            mock.patch.object(write_service, "ResearchController", return_value=controller),
            mock.patch.object(write_service, "_call_campaign", side_effect=campaign),
            mock.patch.object(
                write_service,
                "_proposer_ref",
                return_value={"kind": "proposer", "id": "fixture", "revision": "v1"},
            ),
        ):
            identity, result = write_service.execute_domain_operation(
                manifest=manifest,
                kind="RUN_START",
                operation_id="operation-1",
                parameters={**common, "proposal_only": True},
            )
            self.assertEqual(identity["run_id"], "console-run")
            self.assertEqual(result["status"], "RUNNING")
            write_service.execute_domain_operation(
                manifest=manifest,
                kind="RUN_STOP",
                operation_id="operation-2",
                parameters={**common, "run_id": "run-1"},
            )
            write_service.execute_domain_operation(
                manifest=manifest,
                kind="RUN_RESUME",
                operation_id="operation-3",
                parameters={**common, "run_id": "run-1"},
            )
            fixed_campaigns = (
                ("CAMPAIGN_START", {"campaign_id": "campaign-1"}),
                ("CAMPAIGN_PAUSE", {"campaign_id": "campaign-1", "reason": "operator"}),
                ("CAMPAIGN_RESUME", {"campaign_id": "campaign-1", "profile": "pro"}),
                (
                    "CAMPAIGN_CHILD_EXECUTE",
                    {
                        "campaign_id": "campaign-1",
                        "profile": "pro",
                        "max_candidates": 2,
                        "max_wall_seconds": 60,
                    },
                ),
                (
                    "CAMPAIGN_LINEAGE_ADVANCE",
                    {"campaign_id": "campaign-1", "child_id": "child-1"},
                ),
                ("BENCHMARK_EXECUTE", {"campaign_id": "campaign-1", "profile": "pro"}),
            )
            for index, (kind, parameters) in enumerate(fixed_campaigns, 4):
                write_service.execute_domain_operation(
                    manifest=manifest,
                    kind=kind,
                    operation_id=f"operation-{index}",
                    parameters=parameters,
                )
        flattened = " ".join(" ".join(call) for call in calls)
        self.assertNotIn("--image", flattened)
        self.assertNotIn("--device", flattened)
        self.assertNotIn("--timeout", flattened)
        self.assertIn("console-lineage:", flattened)

    def test_campaign_creation_and_benchmark_freeze_current_seed(self) -> None:
        environment = ExecutionEnvironmentDigest.resolved(
            evaluator_image_digest="sha256:" + "1" * 64,
            toolchain_digest="sha256:" + "2" * 64,
            framework_digest="sha256:" + "3" * 64,
            operator_abi_digest="sha256:" + "4" * 64,
            build_flags_digest="sha256:" + "5" * 64,
        )
        baseline = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=ArtifactId.source_sha256("a" * 64),
            source="campaign",
            revision="deployment-seed-fixture",
            execution_environment=environment,
        )
        manifest = SimpleNamespace(
            runtime_root=self.paths.runtime_root,
            repository_dir=self.paths.repository_dir,
            pro_config=self.paths.runtime_root / "pro.json",
            flash_config=self.paths.runtime_root / "flash.json",
        )
        budget = {
            "candidates": 20,
            "wall_ms": 100000,
            "gpu_ms": 50000,
            "tokens": 100000,
            "cost_microusd": 100000,
        }
        calls: list[list[str]] = []
        with (
            mock.patch.object(write_service, "_config", return_value=object()),
            mock.patch.object(write_service, "_campaign_baseline", return_value=baseline),
            mock.patch.object(
                write_service,
                "_call_campaign",
                side_effect=lambda argv: calls.append(argv) or {"status": "SUCCESS"},
            ),
        ):
            created = write_service.execute_domain_operation(
                manifest=manifest,
                kind="CAMPAIGN_CREATE",
                operation_id="operation-create",
                parameters={"mode": "DISCOVERY", "profile": "pro", "budget": budget},
            )
            benchmark = write_service.execute_domain_operation(
                manifest=manifest,
                kind="BENCHMARK_INIT",
                operation_id="operation-benchmark",
                parameters={
                    "profile": "pro",
                    "arms": ["pro", "flash"],
                    "repetitions": 2,
                    "budget": budget,
                },
            )
        self.assertTrue(created[0]["campaign_id"].startswith("console-discovery-"))
        self.assertTrue(benchmark[0]["campaign_id"].startswith("console-benchmark-"))
        self.assertEqual(calls[0][0], "create")
        self.assertEqual(calls[1][:2], ["benchmark", "init"])

    def test_campaign_command_envelope_is_strict(self) -> None:
        def emit_error(argv: list[str]) -> int:
            del argv
            print("bounded failure", file=sys.stderr)
            return 2

        with mock.patch.object(write_service.campaign_cli, "main", side_effect=emit_error):
            with self.assertRaisesRegex(ValueError, "bounded failure"):
                write_service._call_campaign(["status"])
        with mock.patch.object(
            write_service.campaign_cli,
            "main",
            side_effect=lambda argv: print("not-json") or 0,
        ):
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                write_service._call_campaign(["status"])
        with mock.patch.object(
            write_service.campaign_cli,
            "main",
            side_effect=lambda argv: print('{"status":"FAILED"}') or 0,
        ):
            with self.assertRaisesRegex(ValueError, "invalid envelope"):
                write_service._call_campaign(["status"])


if __name__ == "__main__":
    unittest.main()
