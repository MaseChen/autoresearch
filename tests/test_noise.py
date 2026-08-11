from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from kernel_research.autorun.controller import ResearchController
from kernel_research.autorun.models import ControllerConfig
from kernel_research.autorun.runtime import CommandResult
from kernel_research.autorun.store import ControllerStore
from kernel_research.campaign import BudgetAmount, CampaignStore
from kernel_research.constants import (
    CURRENT_C500_CASE_IDS,
    CURRENT_C500_EVALUATION_PROTOCOL_ID,
)
from kernel_research.cli import main as cli_main
from kernel_research.evaluation import record_noise_external_result
from kernel_research.history import ExperimentRecord
from kernel_research.history import HistoryStore
from kernel_research.noise import (
    NoiseEvidenceScope,
    SQLITE_MAX_INT,
    build_noise_report,
    format_noise_report_table,
)
from kernel_research.platform import (
    ArtifactId,
    BaselineRef,
    CURRENT_RESEARCH_NAMESPACE,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
)
from kernel_research.platform.proposal import CandidateBundle


ROOT = Path(__file__).resolve().parents[1]
HASH = "a" * 64
SOURCE = (ROOT / "kernel.py").read_text(encoding="utf-8")


def resolved_environment(seed: str = "1") -> ExecutionEnvironmentDigest:
    return ExecutionEnvironmentDigest.resolved(
        evaluator_image_digest="sha256:" + seed * 64,
        toolchain_digest="sha256:" + "2" * 64,
        framework_digest="sha256:" + "3" * 64,
        operator_abi_digest="sha256:" + "4" * 64,
        build_flags_digest="sha256:" + "5" * 64,
    )


def controller_config(root: Path) -> tuple[ControllerConfig, Path]:
    root = root.resolve()
    repository = root / "repo"
    state = root / "state"
    controller = root / "controller"
    checkpoints = root / "checkpoints"
    cache = root / "cache"
    for directory in (repository, state, controller, checkpoints, cache):
        directory.mkdir(parents=True, exist_ok=True)
    secret = root / "deepseek-key"
    secret.write_text("fixture", encoding="utf-8")
    secret.chmod(0o600)
    docker = root / "docker"
    docker.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    docker.chmod(0o700)
    config = ControllerConfig(
        repository_dir=repository,
        state_dir=state,
        controller_dir=controller,
        checkpoint_dir=checkpoints,
        docker_binary=docker,
        proposer_image="fixture/proposer@sha256:" + "1" * 64,
        evaluator_image="fixture/evaluator@sha256:" + "2" * 64,
        deepseek_key_file=secret,
        gpu_devices=(
            Path("/dev/mxcd"),
            Path("/dev/dri/card2"),
            Path("/dev/dri/renderD129"),
        ),
        evaluator_cache_dir=cache,
        expected_git_commit="8" * 40,
        expected_kernel_hash=hashlib.sha256(SOURCE.encode("utf-8")).hexdigest(),
        evaluator_timeout_sec=30.0,
        container_uid=os.getuid(),
        container_gid=os.getgid(),
        acknowledge_gpu_passthrough_risk=True,
    )
    config_path = root / "controller.json"
    config_path.write_text(
        json.dumps(config.redacted_dict()), encoding="utf-8"
    )
    return ControllerConfig.load(config_path), config_path


def mount_source(argv: tuple[str, ...] | list[str], destination: str) -> Path:
    marker = f"dst={destination}"
    for value in argv:
        if value.startswith("type=bind,") and marker in value:
            source = next(
                part.removeprefix("src=")
                for part in value.split(",")
                if part.startswith("src=")
            )
            return Path(source)
    raise AssertionError(f"missing Docker bind mount for {destination}")


class NoiseDockerRunner:
    def __init__(
        self,
        *,
        status: str = "SUCCESS",
        mutate_echo: bool = False,
        interrupt: bool = False,
    ) -> None:
        self.status = status
        self.mutate_echo = mutate_echo
        self.interrupt = interrupt
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: object, **_kwargs: object) -> CommandResult:
        command = tuple(str(value) for value in argv)  # type: ignore[arg-type]
        self.calls.append(command)
        if self.interrupt:
            raise KeyboardInterrupt
        identity_path = mount_source(command, "/request/identity.json")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if self.mutate_echo:
            identity["replicate_index"] = int(identity["replicate_index"]) + 1
        result = raw_noise_result(identity_path, status=self.status)
        result["request_identity"] = identity
        return CommandResult(
            argv=command,
            returncode=0,
            stdout=json.dumps(result),
            stderr="",
        )

    def remove_exact_container(self, *_args: object) -> None:
        return None


def strict_baseline_record(
    *, environment: ExecutionEnvironmentDigest | None = None
) -> ExperimentRecord:
    execution_environment = environment or resolved_environment()
    artifact = ArtifactId.source_sha256(HASH)
    parent = BaselineRef.create(
        namespace=CURRENT_RESEARCH_NAMESPACE,
        artifact_id=artifact,
        source="deployment",
        revision="strict-parent",
        execution_environment=execution_environment,
    )
    identity = ExperimentIdentity.create(
        experiment_uid="00000000-0000-4000-8000-ffffffffffff",
        namespace=CURRENT_RESEARCH_NAMESPACE,
        mode="DISCOVERY",
        candidate_artifact_id=artifact,
        parent_artifact_id=artifact,
        baseline=parent,
        execution_environment=execution_environment,
        stage="confirmation",
        suite="full",
        replicate_kind="confirmation",
        run_id="strict-baseline",
        iteration=0,
    )
    value = record(99, 1.0)
    return replace(
        value,
        schema_version=3,
        status="SUCCESS",
        promotable=True,
        namespace_id=identity.namespace_id,
        artifact_id=str(artifact),
        experiment_uid=identity.experiment_uid,
        condition_digest=identity.condition_digest,
        replicate_kind="confirmation",
        replicate_index=0,
        identity=identity.to_dict(),
        result={
            "status": "SUCCESS",
            "evaluation_protocol_id": CURRENT_C500_EVALUATION_PROTOCOL_ID,
            "request_identity": identity.to_dict(),
            "environment": dict(value.environment),
        },
    )


def record(experiment_id: int, speed: float, environment: dict[str, object] | None = None) -> ExperimentRecord:
    candidate = [100.0 / speed + ((index % 3) - 1) * 0.01 for index in range(30)]
    baseline = [100.0 + ((index % 3) - 1) * 0.01 for index in range(30)]
    return ExperimentRecord(
        id=experiment_id,
        schema_version=2,
        created_at=f"2026-08-0{experiment_id}T00:00:00Z",
        candidate_hash=HASH,
        git_commit=None,
        backend="c500",
        suite="full",
        status="SUCCESS",
        promotable=False,
        duplicate_of_id=None,
        aggregate_score=1.0,
        note="noise",
        environment=environment or {"device": "C500", "driver_version": "1"},
        error_summary=None,
        artifact_path=f"artifacts/{HASH}.py",
        result={
            "promotion": {"phase": "remeasurement"},
            "benchmark_config": {"measurement_rounds": 3, "samples_per_round": 10},
            "cases": [
                {
                    "case_id": name,
                    "latency_samples_us": candidate,
                    "baseline_latency_samples_us": baseline,
                }
                for name in ("a", "b", "c", "d")
            ],
        },
        case_measurements=(),
    )


def strict_record(
    experiment_id: int,
    speed: float,
    *,
    replicate_index: int | None = None,
    environment: ExecutionEnvironmentDigest | None = None,
    prompt_digest: str | None = None,
    replicate_kind: str = "noise",
) -> ExperimentRecord:
    execution_environment = environment or resolved_environment()
    artifact = ArtifactId.source_sha256(HASH)
    baseline = BaselineRef.create(
        namespace=CURRENT_RESEARCH_NAMESPACE,
        artifact_id=artifact,
        source="deployment",
        revision="noise-baseline-1",
        execution_environment=execution_environment,
    )
    index = experiment_id if replicate_index is None else replicate_index
    identity = ExperimentIdentity.create(
        experiment_uid=f"00000000-0000-4000-8000-{experiment_id:012x}",
        namespace=CURRENT_RESEARCH_NAMESPACE,
        mode="DISCOVERY",
        candidate_artifact_id=artifact,
        parent_artifact_id=artifact,
        baseline=baseline,
        execution_environment=execution_environment,
        stage=replicate_kind.upper(),
        suite="full",
        replicate_kind=replicate_kind,
        replicate_index=index,
        prompt_digest=prompt_digest,
        run_id="noise-collection",
        iteration=experiment_id,
    )
    legacy = record(experiment_id, speed)
    result = dict(legacy.result)
    result.pop("promotion", None)
    result.update(
        {
            "eligible_for_promotion": False,
            "evaluation_protocol_id": CURRENT_C500_EVALUATION_PROTOCOL_ID,
            "request_identity": identity.to_dict(),
            "environment": dict(legacy.environment),
            "evidence": {
                "schema_version": 1,
                "role": replicate_kind,
                "promotion_eligible": False,
            },
        }
    )
    return replace(
        legacy,
        schema_version=3,
        promotable=False,
        namespace_id=identity.namespace_id,
        artifact_id=str(artifact),
        experiment_uid=identity.experiment_uid,
        condition_digest=identity.condition_digest,
        replicate_kind=replicate_kind,
        replicate_index=index,
        baseline_experiment_uid="00000000-0000-4000-8000-ffffffffffff",
        identity=identity.to_dict(),
        result=result,
    )


def production_baseline(
    root: Path,
    *,
    environment: ExecutionEnvironmentDigest | None = None,
) -> ExperimentRecord:
    execution_environment = environment or resolved_environment()
    bundle = CandidateBundle.single_file(content=SOURCE)
    baseline_ref = BaselineRef.create(
        namespace=CURRENT_RESEARCH_NAMESPACE,
        artifact_id=bundle.artifact_id,
        source="deployment",
        revision="reviewed-noise-baseline",
        execution_environment=(
            execution_environment if execution_environment.is_resolved else None
        ),
    )
    identity = ExperimentIdentity.create(
        experiment_uid="10000000-0000-4000-8000-000000000001",
        namespace=CURRENT_RESEARCH_NAMESPACE,
        mode="DISCOVERY",
        candidate_artifact_id=bundle.artifact_id,
        parent_artifact_id=bundle.artifact_id,
        baseline=baseline_ref,
        execution_environment=baseline_ref.execution_environment,
        stage="confirmation",
        suite="full",
        replicate_kind="confirmation",
        run_id="reviewed-noise-baseline",
        iteration=0,
    )
    with HistoryStore(root / "history.sqlite3", root) as history:
        history.ensure_namespace(
            CURRENT_RESEARCH_NAMESPACE.namespace_id,
            CURRENT_RESEARCH_NAMESPACE.to_dict(),
        )
        history.store_candidate_bundle(bundle)
        history.store_candidate_artifact(
            SOURCE,
            artifact_id=str(
                ArtifactId.source_sha256(
                    hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
                )
            ),
            artifact_kind="source_text_v1",
            manifest={
                "format": "python_source_v1",
                "entrypoint": "kernel.py",
                "media_type": "text/x-python",
            },
        )
        return history.record_experiment(
            candidate_source=SOURCE,
            backend="c500",
            suite="full",
            status="SUCCESS",
            promotable=True,
            aggregate_score=1.0,
            identity=identity,
            environment={"device": "C500"},
            result={
                "status": "SUCCESS",
                "evaluation_protocol_id": CURRENT_C500_EVALUATION_PROTOCOL_ID,
                "request_identity": identity.to_dict(),
                "environment": {"device": "C500"},
            },
        )


def raw_noise_result(
    request_identity_path: str | Path,
    *,
    status: str = "SUCCESS",
) -> dict[str, object]:
    identity = json.loads(Path(request_identity_path).read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "command": "evaluate-raw",
        "backend": "c500",
        "suite": "full",
        "candidate_hash": source_hash,
        "baseline_candidate_hash": source_hash,
        "evaluation_protocol_id": CURRENT_C500_EVALUATION_PROTOCOL_ID,
        "request_identity": identity,
        "status": status,
        "eligible_for_promotion": True,
        "aggregate_score": 9.0,
        "promotion": {"phase": "confirmation", "confirmed": True},
        "cases": (
            [
                {
                    "case_id": case_id,
                    "status": "SUCCESS",
                    "matched_ratio": 1.0,
                    "latency_samples_us": [100.0] * 30,
                    "baseline_latency_samples_us": [100.0] * 30,
                }
                for case_id in CURRENT_C500_CASE_IDS["full"]
            ]
            if status == "SUCCESS"
            else []
        ),
        "environment": {"device": "C500"},
        **({} if status == "SUCCESS" else {"error": "GPU hard fault"}),
    }


class NoiseReportTests(unittest.TestCase):
    def test_run_level_null_distribution_and_order_diagnostics(self) -> None:
        records = [record(index, 0.999 + index * 0.0002) for index in range(1, 11)]
        report = build_noise_report(records, candidate_hash=HASH)
        self.assertEqual(report["independent_run_count"], 10)
        self.assertTrue(report["sufficient_independent_runs"])
        self.assertTrue(report["environment"]["consistent"])
        self.assertTrue(report["p99_below_promotion_threshold"])
        self.assertTrue(report["statistical_gate_passed"])
        self.assertFalse(report["automatic_promotion_allowed"])
        self.assertFalse(report["current_protocol_activation_allowed"])
        self.assertEqual(report["authority"], "LEGACY_READ_ONLY")
        self.assertEqual(len(report["runs"]), 10)
        table = format_noise_report_table(report)
        self.assertIn("10-run minimum satisfied: true", table)
        self.assertIn("p99 below promotion threshold: true", table)

    def test_filters_non_remeasurement_and_detects_environment_drift(self) -> None:
        other = replace(
            record(3, 1.0),
            result={"promotion": {"phase": "primary"}, "cases": []},
        )
        report = build_noise_report(
            [
                record(1, 1.0),
                record(2, 1.001, {"device": "C500", "driver_version": "2"}),
                other,
            ],
            candidate_hash=HASH,
        )
        self.assertEqual(report["independent_run_count"], 2)
        self.assertFalse(report["environment"]["consistent"])
        self.assertFalse(report["sufficient_independent_runs"])
        self.assertFalse(report["automatic_promotion_allowed"])

    def test_rejects_missing_or_unpaired_evidence(self) -> None:
        with self.assertRaises(ValueError):
            build_noise_report([], candidate_hash=HASH)
        broken = record(1, 1.0)
        broken.result["cases"][0]["baseline_latency_samples_us"] = [1.0]
        with self.assertRaises(ValueError):
            build_noise_report([broken], candidate_hash=HASH)

    def test_current_scope_is_exact_and_can_activate_only_with_ten_noise_runs(self) -> None:
        anchor = strict_record(1, 0.9992)
        baseline = strict_baseline_record()
        scope = NoiseEvidenceScope.from_anchor(
            anchor, baseline_record=baseline
        )
        records = [
            strict_record(index, 0.999 + index * 0.00005)
            for index in range(1, 11)
        ]
        # Same source hash, but a different condition family and execution
        # environment, must not enter the cohort.
        records.append(
            strict_record(
                20,
                1.5,
                prompt_digest="sha256:" + "9" * 64,
            )
        )
        records.append(strict_record(21, 1.5, environment=resolved_environment("6")))
        records.append(strict_record(22, 1.5, replicate_kind="primary"))

        report = build_noise_report(
            records, scope=scope, baseline_record=baseline
        )

        self.assertEqual(report["authority"], "CURRENT_NOISE_GATE")
        self.assertEqual(report["independent_run_count"], 10)
        self.assertEqual(report["scope"]["namespace_id"], scope.namespace_id)
        self.assertEqual(
            report["scope"]["evaluation_protocol"]["digest"],
            scope.evaluation_protocol.digest,
        )
        self.assertEqual(
            report["scope"]["condition_family_digest"],
            scope.condition_family_digest,
        )
        self.assertEqual(
            report["scope"]["execution_environment"]["digest"],
            scope.execution_environment.digest,
        )
        self.assertTrue(report["current_protocol_activation_allowed"])
        self.assertTrue(report["automatic_promotion_allowed"])
        self.assertEqual(
            {run["replicate_index"] for run in report["runs"]},
            set(range(1, 11)),
        )
        all_drifted = [
            replace(
                row,
                environment={"device": "C500", "driver_version": "DIFFERENT"},
                result={
                    **row.result,
                    "environment": {
                        "device": "C500",
                        "driver_version": "DIFFERENT",
                    },
                },
            )
            for row in records[:10]
        ]
        drifted_report = build_noise_report(
            all_drifted,
            scope=scope,
            baseline_record=baseline,
        )
        self.assertTrue(drifted_report["environment"]["consistent"])
        self.assertFalse(
            drifted_report["environment"]["matches_frozen_baseline"]
        )
        self.assertFalse(drifted_report["automatic_promotion_allowed"])
        self.assertIn(
            "differs from the frozen baseline",
            drifted_report["automatic_promotion_block_reason"],
        )

    def test_v2_rejects_duplicate_index_and_malformed_noise_authority(self) -> None:
        anchor = strict_record(1, 1.0)
        baseline = strict_baseline_record()
        scope = NoiseEvidenceScope.from_anchor(
            anchor, baseline_record=baseline
        )
        duplicate = strict_record(2, 1.0, replicate_index=1)
        with self.assertRaisesRegex(ValueError, "replicate_index"):
            build_noise_report(
                [anchor, duplicate],
                scope=scope,
                baseline_record=baseline,
            )

        malformed = replace(
            anchor,
            promotable=True,
            result={**anchor.result, "eligible_for_promotion": True},
        )
        with self.assertRaisesRegex(ValueError, "non-promotable"):
            build_noise_report(
                [malformed], scope=scope, baseline_record=baseline
            )

    def test_trusted_noise_recording_and_history_query_never_promote(self) -> None:
        source_hash = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
        environment = resolved_environment()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = production_baseline(root, environment=environment)
            artifact = ArtifactId.parse(baseline.artifact_id)
            baseline_ref = BaselineRef.create(
                namespace=CURRENT_RESEARCH_NAMESPACE,
                artifact_id=artifact,
                source="deployment",
                revision=f"history-{baseline.id}",
                execution_environment=environment,
            )
            noise_identity = ExperimentIdentity.create(
                namespace=CURRENT_RESEARCH_NAMESPACE,
                mode="DISCOVERY",
                candidate_artifact_id=artifact,
                parent_artifact_id=artifact,
                baseline=baseline_ref,
                execution_environment=environment,
                stage="NOISE",
                suite="full",
                replicate_kind="noise",
                replicate_index=1,
                history_cutoff=baseline.id,
                run_id="noise-run",
                iteration=1,
            )
            request_path = root / "request.json"
            request_path.write_text(
                json.dumps(noise_identity.to_dict()), encoding="utf-8"
            )
            raw = raw_noise_result(request_path)

            evidence = record_noise_external_result(
                candidate_source=SOURCE,
                result=raw,
                state_dir=root,
                note="noise:1",
                identity=noise_identity,
                baseline_experiment_id=baseline.id,
            )

            self.assertFalse(evidence.promotable)
            self.assertIsNone(evidence.aggregate_score)
            self.assertNotIn("promotion", evidence.result)
            self.assertEqual(evidence.result["evidence"]["role"], "noise")
            self.assertFalse(
                evidence.result["evidence"]["promotion_eligible"]
            )
            with HistoryStore(root / "history.sqlite3", root) as history:
                queried = history.list_noise_experiments(
                    namespace_id=noise_identity.namespace_id,
                    artifact_id=str(artifact),
                    baseline_experiment_uid=baseline.experiment_uid,
                )
                self.assertEqual([row.experiment_uid for row in queried], [
                    noise_identity.experiment_uid
                ])

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = cli_main(
                    [
                        "noise-report",
                        "--state-dir",
                        str(root),
                        "--namespace-id",
                        noise_identity.namespace_id,
                        "--anchor-experiment-uid",
                        noise_identity.experiment_uid,
                        "--protocol-digest",
                        noise_identity.evaluation_protocol.digest,
                        "--artifact-id",
                        str(artifact),
                        "--baseline-experiment-uid",
                        baseline.experiment_uid,
                        "--format",
                        "json",
                    ]
                )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["authority"], "CURRENT_NOISE_GATE")
            self.assertEqual(payload["independent_run_count"], 1)
            self.assertFalse(payload["current_protocol_activation_allowed"])

    def test_noise_collect_cli_creates_first_anchor_and_rejects_duplicate_index(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, config_path = controller_config(root)
            candidate = root / "kernel.py"
            candidate.write_bytes(SOURCE.encode("utf-8"))
            environment = ResearchController(config)._resolved_execution_environment(
                CURRENT_RESEARCH_NAMESPACE
            )
            baseline = production_baseline(
                config.state_dir, environment=environment
            )
            runner = NoiseDockerRunner()

            argv = [
                "noise-collect",
                "--config",
                str(config_path),
                "--candidate",
                str(candidate),
                "--namespace-id",
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
                "--baseline-experiment-uid",
                baseline.experiment_uid,
                "--replicate-index",
                "1",
            ]
            output = io.StringIO()
            with patch(
                "kernel_research.cli.ResearchController",
                side_effect=lambda loaded: ResearchController(
                    loaded, runner=runner
                ),
            ), patch(
                "kernel_research.evaluation.raw_evaluate"
            ) as host_evaluator, redirect_stdout(output):
                self.assertEqual(cli_main(argv), 0)
                errors = io.StringIO()
                with redirect_stderr(errors):
                    self.assertEqual(cli_main(argv), 2)

            host_evaluator.assert_not_called()
            self.assertEqual(len(runner.calls), 1)
            docker_argv = runner.calls[0]
            self.assertIn("evaluate-raw", docker_argv)
            self.assertIn("none", docker_argv)
            self.assertIn("--read-only", docker_argv)
            self.assertEqual(
                mount_source(docker_argv, "/candidate/kernel.py").read_bytes(),
                SOURCE.encode("utf-8"),
            )
            self.assertIn("duplicate collection is forbidden", errors.getvalue())
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["command"], "noise-collect")
            self.assertEqual(payload["replicate_kind"], "noise")
            self.assertEqual(payload["replicate_index"], 1)
            self.assertFalse(payload["promotable"])
            self.assertEqual(payload["artifact_id"], baseline.artifact_id)
            with HistoryStore(
                config.state_dir / "history.sqlite3", config.state_dir
            ) as history:
                anchor = history.get_experiment_by_uid(payload["experiment_uid"])
                assert anchor is not None
                scope = NoiseEvidenceScope.from_anchor(
                    anchor, baseline_record=baseline
                )
                self.assertTrue(scope.is_current_protocol)
                self.assertEqual(scope.baseline_experiment_uid, baseline.experiment_uid)
            with ControllerStore(
                config.controller_dir / "controller.sqlite3"
            ) as store:
                attempt = store.get_evaluation_attempt_by_uid(
                    payload["experiment_uid"]
                )
                assert attempt is not None
                self.assertEqual(attempt["status"], "SUCCEEDED")

    def test_noise_collect_records_hard_outcome_once_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = controller_config(root)
            candidate = root / "kernel.py"
            candidate.write_text(SOURCE, encoding="utf-8")
            environment = ResearchController(config)._resolved_execution_environment(
                CURRENT_RESEARCH_NAMESPACE
            )
            baseline = production_baseline(
                config.state_dir, environment=environment
            )
            runner = NoiseDockerRunner(status="CRASH")
            controller = ResearchController(config, runner=runner)
            record_value = controller.collect_noise(
                candidate_path=candidate,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                baseline_experiment_uid=baseline.experiment_uid,
                replicate_index=7,
            )
            self.assertEqual(record_value.status, "CRASH")
            with self.assertRaisesRegex(
                Exception, "trusted GPU doctor and manual recovery"
            ):
                controller.collect_noise(
                    candidate_path=candidate,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    baseline_experiment_uid=baseline.experiment_uid,
                    replicate_index=8,
                )
            self.assertEqual(len(runner.calls), 1)
            with HistoryStore(
                config.state_dir / "history.sqlite3", config.state_dir
            ) as history:
                attempts = history.list_noise_experiments(
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    artifact_id=baseline.artifact_id,
                    baseline_experiment_uid=baseline.experiment_uid,
                )
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0].status, "CRASH")
            self.assertFalse(attempts[0].promotable)

    def test_noise_collect_fails_before_gpu_for_legacy_or_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = controller_config(root)
            candidate = root / "kernel.py"
            candidate.write_text(SOURCE + "# drift\n", encoding="utf-8")
            environment = ResearchController(config)._resolved_execution_environment(
                CURRENT_RESEARCH_NAMESPACE
            )
            baseline = production_baseline(
                config.state_dir, environment=environment
            )
            runner = NoiseDockerRunner()
            controller = ResearchController(config, runner=runner)
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                controller.collect_noise(
                    candidate_path=candidate,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    baseline_experiment_uid=baseline.experiment_uid,
                    replicate_index=2,
                )
            candidate.write_text(SOURCE, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CURRENT namespace"):
                controller.collect_noise(
                    candidate_path=candidate,
                    namespace_id="sha256:" + "0" * 64,
                    baseline_experiment_uid=baseline.experiment_uid,
                    replicate_index=2,
                )
            with self.assertRaisesRegex(ValueError, "SQLite INTEGER"):
                controller.collect_noise(
                    candidate_path=candidate,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    baseline_experiment_uid=baseline.experiment_uid,
                    replicate_index=SQLITE_MAX_INT + 1,
                )
            self.assertEqual(runner.calls, [])

    def test_noise_collect_rejects_non_exact_evaluator_echo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = controller_config(root)
            candidate = root / "kernel.py"
            candidate.write_text(SOURCE, encoding="utf-8")
            environment = ResearchController(config)._resolved_execution_environment(
                CURRENT_RESEARCH_NAMESPACE
            )
            baseline = production_baseline(
                config.state_dir, environment=environment
            )
            runner = NoiseDockerRunner(mutate_echo=True)
            controller = ResearchController(config, runner=runner)
            with self.assertRaisesRegex(Exception, "identity mismatch"):
                controller.collect_noise(
                    candidate_path=candidate,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    baseline_experiment_uid=baseline.experiment_uid,
                    replicate_index=3,
                )
            self.assertEqual(len(runner.calls), 1)
            with HistoryStore(
                config.state_dir / "history.sqlite3", config.state_dir
            ) as history:
                self.assertEqual(
                    history.list_noise_experiments(
                        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                        artifact_id=baseline.artifact_id,
                        baseline_experiment_uid=baseline.experiment_uid,
                    ),
                    [],
                )

    def test_noise_collect_interruption_is_durable_unknown_and_blocks_new_index(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = controller_config(root)
            candidate = root / "kernel.py"
            candidate.write_text(SOURCE, encoding="utf-8")
            environment = ResearchController(config)._resolved_execution_environment(
                CURRENT_RESEARCH_NAMESPACE
            )
            baseline = production_baseline(
                config.state_dir, environment=environment
            )
            runner = NoiseDockerRunner(interrupt=True)
            controller = ResearchController(config, runner=runner)
            with self.assertRaises(KeyboardInterrupt):
                controller.collect_noise(
                    candidate_path=candidate,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    baseline_experiment_uid=baseline.experiment_uid,
                    replicate_index=4,
                )
            with ControllerStore(
                config.controller_dir / "controller.sqlite3"
            ) as store:
                attempts = store.list_evaluation_attempts_by_replicate_kind(
                    "noise"
                )
                self.assertEqual(len(attempts), 1)
                self.assertEqual(attempts[0]["status"], "UNKNOWN_OUTCOME")
                self.assertEqual(
                    store.get_run(str(attempts[0]["run_id"]))["status"],
                    "HARD_FAILED",
                )
            with self.assertRaisesRegex(
                Exception, "trusted GPU doctor and manual recovery"
            ):
                controller.collect_noise(
                    candidate_path=candidate,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    baseline_experiment_uid=baseline.experiment_uid,
                    replicate_index=5,
                )
            self.assertEqual(len(runner.calls), 1)

    def test_noise_collect_rechecks_campaign_fence_immediately_before_runner(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = controller_config(root)
            candidate = root / "kernel.py"
            candidate.write_text(SOURCE, encoding="utf-8")
            environment = ResearchController(config)._resolved_execution_environment(
                CURRENT_RESEARCH_NAMESPACE
            )
            baseline = production_baseline(
                config.state_dir, environment=environment
            )
            runner = NoiseDockerRunner()
            controller = ResearchController(config, runner=runner)
            evaluator = controller.evaluator
            original_cache = evaluator._cache_dir  # type: ignore[attr-defined]

            def create_racing_campaign(**kwargs: object) -> Path:
                cache_path = original_cache(**kwargs)
                campaign_ref = BaselineRef.create(
                    namespace=CURRENT_RESEARCH_NAMESPACE,
                    artifact_id=baseline.artifact_id,
                    source="campaign",
                    revision="noise-race-seed",
                    execution_environment=environment,
                )
                database = root.resolve() / "campaign" / "campaign.sqlite3"
                with CampaignStore(database) as campaign:
                    campaign.create_campaign(
                        campaign_id="racing-campaign",
                        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                        mode="DISCOVERY",
                        snapshot={"fixture": "race-before-runner"},
                        budget_limit=BudgetAmount(
                            candidates=1,
                            wall_ms=60_000,
                            gpu_ms=30_000,
                            tokens=1,
                            cost_microusd=1,
                        ),
                        initial_baseline_ref=campaign_ref,
                        initial_policy_snapshot={"policy": "fixture"},
                    )
                return cache_path

            with patch.object(
                evaluator, "_cache_dir", side_effect=create_racing_campaign
            ), self.assertRaisesRegex(Exception, "Campaign can resume"):
                controller.collect_noise(
                    candidate_path=candidate,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    baseline_experiment_uid=baseline.experiment_uid,
                    replicate_index=6,
                )
            self.assertEqual(runner.calls, [])
            with ControllerStore(
                config.controller_dir / "controller.sqlite3"
            ) as store:
                attempt = store.list_evaluation_attempts_by_replicate_kind(
                    "noise"
                )[0]
                self.assertEqual(attempt["status"], "UNKNOWN_OUTCOME")

    def test_noise_collect_rejects_quarantined_campaign_lease_before_intent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = controller_config(root)
            candidate = root / "kernel.py"
            candidate.write_text(SOURCE, encoding="utf-8")
            environment = ResearchController(config)._resolved_execution_environment(
                CURRENT_RESEARCH_NAMESPACE
            )
            baseline = production_baseline(
                config.state_dir, environment=environment
            )
            campaign_ref = BaselineRef.create(
                namespace=CURRENT_RESEARCH_NAMESPACE,
                artifact_id=baseline.artifact_id,
                source="campaign",
                revision="quarantine-seed",
                execution_environment=environment,
            )
            database = root.resolve() / "campaign" / "campaign.sqlite3"
            with CampaignStore(database) as campaign:
                campaign.create_campaign(
                    campaign_id="quarantined-campaign",
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    mode="DISCOVERY",
                    snapshot={"fixture": "quarantine"},
                    budget_limit=BudgetAmount(
                        candidates=1,
                        wall_ms=60_000,
                        gpu_ms=30_000,
                        tokens=1,
                        cost_microusd=1,
                    ),
                    initial_baseline_ref=campaign_ref,
                    initial_policy_snapshot={"policy": "fixture"},
                )
                campaign.start_campaign("quarantined-campaign")
                lease = campaign.acquire_resource(
                    "quarantined-campaign",
                    resource_id="gpu1",
                    ttl_seconds=60,
                )
                campaign.release_resource(
                    lease, quarantine=True, reason="fixture hard fault"
                )
                campaign.connection.execute(
                    "UPDATE campaigns SET status = 'CANCELLED' WHERE id = ?",
                    ("quarantined-campaign",),
                )
                campaign.connection.commit()
            runner = NoiseDockerRunner()
            controller = ResearchController(config, runner=runner)
            with self.assertRaisesRegex(Exception, "QUARANTINED"):
                controller.collect_noise(
                    candidate_path=candidate,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    baseline_experiment_uid=baseline.experiment_uid,
                    replicate_index=9,
                )
            self.assertEqual(runner.calls, [])
            self.assertFalse(
                (config.controller_dir / "controller.sqlite3").exists()
            )


if __name__ == "__main__":
    unittest.main()
