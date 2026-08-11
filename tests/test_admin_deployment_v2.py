from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from kernel_research.autorun import admin
from kernel_research.autorun.controller import ResearchController
from kernel_research.autorun.deployment import (
    DEPLOYMENT_BASELINE_FILENAME,
    MAX_DEPLOYMENT_BASELINE_BYTES,
    DeploymentBaselinePin,
    deployment_runtime_root,
)
from kernel_research.autorun.errors import ControlledRuntimeError
from kernel_research.autorun.models import ControllerConfig
from kernel_research.autorun.store import ControllerStore
from kernel_research.history import HistoryStore
from kernel_research.platform import (
    ArtifactId,
    BaselineRef,
    CandidateBundle,
    CURRENT_RESEARCH_NAMESPACE,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
    LEGACY_RESEARCH_NAMESPACE,
)

from test_autorun import SEED, SEED_HASH
from test_autorun_admin import AdminFixture, _command


def _measurements() -> tuple[dict[str, object], ...]:
    return (
        {
            "name": "v2-full-case",
            "matched_ratio": 1.0,
            "passed": True,
            "raw_samples": [90.0] * 30,
            "baseline_samples": [100.0] * 30,
        },
    )


def _source_manifest() -> dict[str, str]:
    return {
        "format": "python_source_v1",
        "entrypoint": "kernel.py",
        "media_type": "text/x-python",
    }


def _valid_deployment_pin() -> DeploymentBaselinePin:
    environment = ExecutionEnvironmentDigest.resolved(
        evaluator_image_digest="sha256:" + "1" * 64,
        toolchain_digest="sha256:" + "2" * 64,
        framework_digest="sha256:" + "3" * 64,
        operator_abi_digest="sha256:" + "4" * 64,
        build_flags_digest="sha256:" + "5" * 64,
    )
    baseline = BaselineRef.create(
        namespace=CURRENT_RESEARCH_NAMESPACE,
        artifact_id=CandidateBundle.single_file(
            content=SEED + "\n# deployment pin\n"
        ).artifact_id,
        source="deployment",
        revision="deployment-pin-test",
        execution_environment=environment,
    )
    parent = BaselineRef.create(
        namespace=CURRENT_RESEARCH_NAMESPACE,
        artifact_id=CandidateBundle.single_file(content=SEED).artifact_id,
        source="deployment",
        revision="deployment-pin-parent",
        execution_environment=environment,
    )
    return DeploymentBaselinePin.create(
        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
        baseline_ref=baseline,
        candidate_hash=admin._sha256_bytes(
            (SEED + "\n# deployment pin\n").encode("utf-8")
        ),
        git_commit="a" * 40,
        primary_experiment_uid="00000000-0000-4000-8000-000000000010",
        confirmation_experiment_uid="00000000-0000-4000-8000-000000000011",
        confirmation_experiment_id=11,
        parent_baseline_ref=parent,
        execution_environment=environment,
    )


class CurrentEvidenceFixture:
    def __init__(
        self,
        fixture: AdminFixture,
        manifest,
        candidate: str,
        *,
        resolved: bool = True,
        mixed_confirmation_environment: bool = False,
        add_confirmation_relation: bool = True,
        target_environment_matches: bool = True,
    ) -> None:
        self.fixture = fixture
        self.manifest = manifest
        self.candidate = candidate
        self.candidate_hash = admin._sha256_bytes(candidate.encode("utf-8"))

        configured = ControllerConfig.load(manifest.pro_config)
        self.framework_commit = configured.resolved_framework_git_commit
        target_config = configured
        if not target_environment_matches:
            target_config = replace(
                configured,
                framework_git_commit="f" * 40,
            )
        resolved_environment = ResearchController(
            target_config
        )._resolved_execution_environment(CURRENT_RESEARCH_NAMESPACE)
        environment = (
            resolved_environment
            if resolved
            else ExecutionEnvironmentDigest.legacy_unknown(
                scope={"test": "current-relabel"}
            )
        )
        confirmation_environment = environment
        if mixed_confirmation_environment and resolved:
            confirmation_environment = ExecutionEnvironmentDigest.resolved(
                evaluator_image_digest=environment.evaluator_image_digest,
                toolchain_digest=environment.toolchain_digest,
                framework_digest=environment.framework_digest,
                operator_abi_digest=environment.operator_abi_digest,
                build_flags_digest="sha256:" + "f" * 64,
            )
        self.environment = environment

        with HistoryStore(
            fixture.state / "history.sqlite3", state_dir=fixture.state
        ) as history:
            history.ensure_namespace(
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
                CURRENT_RESEARCH_NAMESPACE.to_dict(),
            )
            baseline_bundle = CandidateBundle.single_file(content=SEED)
            history.store_candidate_bundle(baseline_bundle)
            history.store_candidate_artifact(
                SEED,
                artifact_id=str(ArtifactId.source_sha256(SEED_HASH)),
                artifact_kind="source_text_v1",
                manifest=_source_manifest(),
            )
            baseline_ref = BaselineRef.create(
                namespace=CURRENT_RESEARCH_NAMESPACE,
                artifact_id=baseline_bundle.artifact_id,
                source="deployment",
                revision="reviewed-current-seed",
                execution_environment=(environment if resolved else None),
            )
            baseline_identity = ExperimentIdentity.create(
                experiment_uid="00000000-0000-4000-8000-000000000001",
                namespace=CURRENT_RESEARCH_NAMESPACE,
                mode="DISCOVERY",
                candidate_artifact_id=baseline_bundle.artifact_id,
                parent_artifact_id=baseline_bundle.artifact_id,
                baseline=baseline_ref,
                execution_environment=baseline_ref.execution_environment,
                stage="confirmation",
                suite="full",
                replicate_kind="confirmation",
                run_id="reviewed-current-seed",
                iteration=0,
            )
            baseline = history.record_experiment(
                candidate_source=SEED,
                backend="c500",
                suite="full",
                status="SUCCESS",
                promotable=True,
                aggregate_score=1.0,
                identity=baseline_identity,
                result={
                    "status": "SUCCESS",
                    "promotion": {
                        "phase": "baseline",
                        "reason": "initial_correct_baseline",
                        "confirmed": True,
                    },
                },
                case_measurements=_measurements(),
            )
            parent = BaselineRef.create(
                namespace=CURRENT_RESEARCH_NAMESPACE,
                artifact_id=baseline_bundle.artifact_id,
                source="deployment",
                revision=f"history-{baseline.id}",
                execution_environment=(environment if resolved else None),
            )

            candidate_bundle = CandidateBundle.single_file(content=candidate)
            self.bundle_id = str(candidate_bundle.artifact_id)
            bundle_record = history.store_candidate_bundle(candidate_bundle)
            self.bundle_path = fixture.state / bundle_record.object_path
            source_record = history.store_candidate_artifact(
                candidate,
                artifact_id=str(ArtifactId.source_sha256(self.candidate_hash)),
                artifact_kind="source_text_v1",
                manifest=_source_manifest(),
            )
            self.source_path = fixture.state / source_record.object_path
            primary_identity = ExperimentIdentity.create(
                experiment_uid="00000000-0000-4000-8000-000000000002",
                namespace=CURRENT_RESEARCH_NAMESPACE,
                mode="DISCOVERY",
                candidate_artifact_id=candidate_bundle.artifact_id,
                parent_artifact_id=baseline_bundle.artifact_id,
                baseline=parent,
                execution_environment=parent.execution_environment,
                stage="full_primary",
                suite="full",
                replicate_kind="primary",
                run_id="current-discovery",
                iteration=1,
            )
            self.primary = history.record_experiment(
                candidate_source=candidate,
                backend="c500",
                suite="full",
                status="SUCCESS",
                promotable=False,
                aggregate_score=1.02,
                identity=primary_identity,
                baseline_experiment_uid=baseline.experiment_uid,
                result={
                    "status": "SUCCESS",
                    "promotion": {
                        "phase": "primary",
                        "reason": "confirmation_required",
                        "baseline_experiment_id": baseline.id,
                        "baseline_candidate_hash": SEED_HASH,
                        "confirmed": False,
                        "decision": {
                            "promoted": False,
                            "needs_confirmation": True,
                            "reason": "confirmation_required",
                        },
                    },
                },
                case_measurements=_measurements(),
            )
            confirmation_parent = BaselineRef.create(
                namespace=CURRENT_RESEARCH_NAMESPACE,
                artifact_id=baseline_bundle.artifact_id,
                source="deployment",
                revision=f"history-{baseline.id}",
                execution_environment=(
                    confirmation_environment if resolved else None
                ),
            )
            confirmation_identity = ExperimentIdentity.create(
                experiment_uid="00000000-0000-4000-8000-000000000003",
                namespace=CURRENT_RESEARCH_NAMESPACE,
                mode="DISCOVERY",
                candidate_artifact_id=candidate_bundle.artifact_id,
                parent_artifact_id=baseline_bundle.artifact_id,
                baseline=confirmation_parent,
                execution_environment=confirmation_parent.execution_environment,
                stage="confirmation",
                suite="full",
                replicate_kind="confirmation",
                run_id="current-discovery",
                iteration=1,
            )
            self.confirmation = history.record_experiment(
                candidate_source=candidate,
                backend="c500",
                suite="full",
                status="SUCCESS",
                promotable=True,
                aggregate_score=1.02,
                identity=confirmation_identity,
                baseline_experiment_uid=baseline.experiment_uid,
                result={
                    "status": "SUCCESS",
                    "promotion": {
                        "phase": "confirmation",
                        "reason": "promoted",
                        "baseline_experiment_id": baseline.id,
                        "baseline_candidate_hash": SEED_HASH,
                        "primary_experiment_id": self.primary.id,
                        "confirmed": True,
                        "decision": {
                            "promoted": True,
                            "needs_confirmation": False,
                            "reason": "promoted",
                        },
                    },
                },
                case_measurements=_measurements(),
            )
            if add_confirmation_relation:
                history.add_experiment_relation(
                    source_experiment_uid=self.primary.experiment_uid,
                    target_experiment_uid=self.confirmation.experiment_uid,
                    relation_type="confirmation_of",
                    metadata={"baseline_experiment_id": baseline.id},
                )

        # Match the production order: immutable evidence exists before the
        # operator reviews and commits the candidate source.
        (fixture.repo / "kernel.py").write_text(candidate, encoding="utf-8")
        _command("git", "add", "kernel.py", cwd=fixture.repo)
        _command("git", "commit", "-m", "current V2 candidate", cwd=fixture.repo)
        self.commit = _command("git", "rev-parse", "HEAD", cwd=fixture.repo)


class AdminDeploymentV2Tests(unittest.TestCase):
    def test_deployment_pin_is_not_replaced_by_later_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            with mock.patch.object(admin, "_active_containers", return_value=[]):
                admin.adopt_baseline(
                    manifest,
                    candidate_hash=SEED_HASH,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    doctor=False,
                )

            pin_path = fixture.runtime / DEPLOYMENT_BASELINE_FILENAME
            pin = DeploymentBaselinePin.load(pin_path)
            config = ControllerConfig.load(manifest.pro_config)
            environment = ResearchController(
                config
            )._resolved_execution_environment(LEGACY_RESEARCH_NAMESPACE)
            artifact_id = ArtifactId.source_sha256(SEED_HASH)
            baseline_ref = BaselineRef.create(
                namespace=LEGACY_RESEARCH_NAMESPACE,
                artifact_id=artifact_id,
                source="deployment",
                revision="qualification-after-published-pin",
                execution_environment=environment,
            )
            identity = ExperimentIdentity.create(
                experiment_uid="00000000-0000-4000-8000-000000000099",
                namespace=LEGACY_RESEARCH_NAMESPACE,
                mode="DISCOVERY",
                candidate_artifact_id=artifact_id,
                parent_artifact_id=artifact_id,
                baseline=baseline_ref,
                execution_environment=environment,
                stage="baseline_qualification",
                suite="full",
                replicate_kind="qualification",
                run_id="qualification-after-published-pin",
                iteration=0,
            )
            with HistoryStore(
                fixture.state / "history.sqlite3", state_dir=fixture.state
            ) as history:
                qualification = history.record_experiment(
                    candidate_source=SEED,
                    backend="c500",
                    suite="full",
                    status="SUCCESS",
                    promotable=True,
                    aggregate_score=1.0,
                    identity=identity,
                    result={
                        "status": "SUCCESS",
                        "request_identity": identity.to_dict(),
                        "promotion": {
                            "phase": "baseline",
                            "reason": "execution_environment_requalified",
                            "confirmed": True,
                        },
                    },
                    case_measurements=_measurements(),
                )
            self.assertGreater(
                qualification.id, pin.confirmation_experiment_id
            )

            (fixture.repo / "README.md").write_text(
                "control-plane update\n", encoding="utf-8"
            )
            _command("git", "add", "README.md", cwd=fixture.repo)
            _command("git", "commit", "-m", "control update", cwd=fixture.repo)

            report = admin._identity(
                manifest,
                require_config_commit=False,
                expected_kernel_hash=SEED_HASH,
            )
            self.assertEqual(
                report["baseline_experiment_id"],
                pin.confirmation_experiment_id,
            )
            self.assertNotEqual(
                report["baseline_experiment_id"], qualification.id
            )

            for name, confirmation_id, confirmation_uid in (
                (
                    "id",
                    qualification.id,
                    pin.confirmation_experiment_uid,
                ),
                (
                    "uid",
                    pin.confirmation_experiment_id,
                    qualification.experiment_uid,
                ),
            ):
                with self.subTest(tampered=name):
                    tampered = DeploymentBaselinePin.create(
                        namespace_id=pin.namespace_id,
                        baseline_ref=pin.baseline_ref,
                        candidate_hash=pin.candidate_hash,
                        git_commit=pin.git_commit,
                        primary_experiment_uid=pin.primary_experiment_uid,
                        confirmation_experiment_uid=confirmation_uid,
                        confirmation_experiment_id=confirmation_id,
                        parent_baseline_ref=pin.parent_baseline_ref,
                        execution_environment=pin.execution_environment,
                    )
                    pin_path.write_text(
                        json.dumps(tampered.to_dict()), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        ControlledRuntimeError, "ID/UID"
                    ):
                        admin._identity(
                            manifest,
                            require_config_commit=False,
                            expected_kernel_hash=SEED_HASH,
                        )

    def test_current_adoption_writes_atomic_pin_and_ordinary_run_uses_current(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            evidence = CurrentEvidenceFixture(
                fixture,
                manifest,
                SEED + "\n# current accepted candidate\n",
            )
            with mock.patch.object(admin, "_active_containers", return_value=[]):
                report = admin.adopt_baseline(
                    manifest,
                    candidate_hash=evidence.candidate_hash,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    doctor=False,
                )
            pin_path = fixture.runtime / DEPLOYMENT_BASELINE_FILENAME
            self.assertEqual(stat.S_IMODE(pin_path.stat().st_mode), 0o600)
            pin = DeploymentBaselinePin.load(pin_path)
            self.assertEqual(pin.namespace_id, CURRENT_RESEARCH_NAMESPACE.namespace_id)
            self.assertTrue(pin.execution_environment.is_resolved)
            self.assertEqual(pin.baseline_ref.source, "deployment")
            self.assertEqual(pin.baseline_ref.artifact_id.value, evidence.bundle_id)
            self.assertEqual(report["evidence_digest"], pin.evidence_digest)
            self.assertEqual(
                report["framework_git_commit"], evidence.framework_commit
            )
            self.assertEqual(report["deployment_git_commit"], evidence.commit)
            adopted_config = ControllerConfig.load(manifest.pro_config)
            self.assertEqual(
                adopted_config.resolved_framework_git_commit,
                evidence.framework_commit,
            )
            self.assertEqual(adopted_config.expected_git_commit, evidence.commit)
            self.assertEqual(
                admin._identity(manifest)["baseline_experiment_id"],
                evidence.confirmation.id,
            )

            config = ControllerConfig.load(manifest.pro_config)
            controller = ResearchController(config)

            def finish_immediately(run_id: str, *, proposal_only: bool = False):
                del proposal_only
                with ControllerStore(controller.controller_db) as store:
                    return store.get_run(run_id)

            with (
                mock.patch.object(
                    controller,
                    "doctor",
                    return_value={"status": "SUCCESS", "errors": []},
                ),
                mock.patch.object(
                    controller, "_run_loop", side_effect=finish_immediately
                ),
            ):
                run = controller.start(proposal_only=True)
            self.assertEqual(run["namespace_id"], CURRENT_RESEARCH_NAMESPACE.namespace_id)
            self.assertEqual(run["baseline_ref"], pin.baseline_ref.to_dict())
            self.assertTrue(run["workflow_snapshot"]["scientifically_comparable"])
            self.assertNotIn("compatibility_mode", run["workflow_snapshot"])

    def test_adoption_commit_bridge_rejects_extra_paths_and_extra_commits(self) -> None:
        for mutation, message in (
            ("extra-path", "change exactly kernel.py"),
            ("extra-commit", "directly descend"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = AdminFixture(Path(temporary))
                manifest = fixture.bootstrap()
                evidence = CurrentEvidenceFixture(
                    fixture,
                    manifest,
                    SEED + f"\n# bridge {mutation}\n",
                )
                (fixture.repo / "README.md").write_text(
                    f"{mutation}\n", encoding="utf-8"
                )
                _command("git", "add", "README.md", cwd=fixture.repo)
                if mutation == "extra-path":
                    _command(
                        "git", "commit", "--amend", "--no-edit", cwd=fixture.repo
                    )
                else:
                    _command(
                        "git", "commit", "-m", "unrelated follow-up", cwd=fixture.repo
                    )
                with (
                    mock.patch.object(
                        admin, "_active_containers", return_value=[]
                    ),
                    self.assertRaisesRegex(ControlledRuntimeError, message),
                ):
                    admin.adopt_baseline(
                        manifest,
                        candidate_hash=evidence.candidate_hash,
                        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                        doctor=False,
                    )
    def test_bundle_object_is_authoritative_and_corruption_fails_closed(self) -> None:
        for failure in ("missing", "tampered", "source-tampered"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                fixture = AdminFixture(Path(temporary))
                manifest = fixture.bootstrap()
                evidence = CurrentEvidenceFixture(
                    fixture,
                    manifest,
                    SEED + f"\n# bundle {failure}\n",
                )
                if failure == "missing":
                    evidence.bundle_path.unlink()
                elif failure == "source-tampered":
                    evidence.source_path.write_bytes(b"tampered source")
                else:
                    evidence.bundle_path.write_bytes(b"{}")
                with (
                    mock.patch.object(admin, "_active_containers", return_value=[]),
                    self.assertRaisesRegex(
                        ControlledRuntimeError,
                        "bundle CAS object|entrypoint source CAS object",
                    ),
                ):
                    admin.adopt_baseline(
                        manifest,
                        candidate_hash=evidence.candidate_hash,
                        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                        doctor=False,
                    )
                self.assertFalse(
                    (fixture.runtime / DEPLOYMENT_BASELINE_FILENAME).exists()
                )

    def test_current_rejects_unknown_mixed_or_unlinked_evidence(self) -> None:
        variants = {
            "LEGACY_UNKNOWN": {"resolved": False},
            "does not prove one V2 promotion": {
                "mixed_confirmation_environment": True
            },
            "immutable primary-to-confirmation link": {
                "add_confirmation_relation": False
            },
            "adoption target execution environment mismatch": {
                "target_environment_matches": False
            },
        }
        for message, options in variants.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                fixture = AdminFixture(Path(temporary))
                manifest = fixture.bootstrap()
                evidence = CurrentEvidenceFixture(
                    fixture,
                    manifest,
                    SEED + f"\n# invalid {message}\n",
                    **options,
                )
                with (
                    mock.patch.object(admin, "_active_containers", return_value=[]),
                    self.assertRaisesRegex(ControlledRuntimeError, message),
                ):
                    admin.adopt_baseline(
                        manifest,
                        candidate_hash=evidence.candidate_hash,
                        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                        doctor=False,
                    )

    def test_namespace_is_explicit_and_legacy_grandfather_remains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            candidate = SEED + "\n# legacy grandfather\n"
            candidate_hash = admin._sha256_bytes(candidate.encode("utf-8"))
            (fixture.repo / "kernel.py").write_text(candidate, encoding="utf-8")
            _command("git", "add", "kernel.py", cwd=fixture.repo)
            _command("git", "commit", "-m", "legacy candidate", cwd=fixture.repo)
            fixture._record(candidate, candidate_hash, score=2.0, phase="confirmation")
            with mock.patch.object(admin, "_active_containers", return_value=[]):
                legacy = admin.adopt_baseline(
                    manifest,
                    candidate_hash=candidate_hash,
                    doctor=False,
                )
            self.assertEqual(
                legacy["namespace_id"],
                admin.LEGACY_RESEARCH_NAMESPACE.namespace_id,
            )
            pin = DeploymentBaselinePin.load(
                fixture.runtime / DEPLOYMENT_BASELINE_FILENAME
            )
            self.assertFalse(pin.execution_environment.is_resolved)

            with (
                mock.patch.object(admin, "_active_containers", return_value=[]),
                self.assertRaisesRegex(
                    ControlledRuntimeError, "deployment namespace"
                ),
            ):
                admin.adopt_baseline(
                    manifest,
                    candidate_hash=candidate_hash,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    doctor=False,
                )

            parsed = admin.build_parser().parse_args(
                [
                    "adopt-baseline",
                    "--manifest",
                    str(fixture.manifest_path),
                    "--candidate-hash",
                    candidate_hash,
                    "--namespace",
                    CURRENT_RESEARCH_NAMESPACE.namespace_id,
                ]
            )
            self.assertEqual(parsed.namespace_id, CURRENT_RESEARCH_NAMESPACE.namespace_id)
            with self.assertRaisesRegex(ValueError, "exact built-in"):
                admin._trusted_namespace("CURRENT")

    def test_pin_loader_rejects_loose_mode_duplicates_nan_size_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            candidate = SEED + "\n# pin parser\n"
            candidate_hash = admin._sha256_bytes(candidate.encode("utf-8"))
            (fixture.repo / "kernel.py").write_text(candidate, encoding="utf-8")
            _command("git", "add", "kernel.py", cwd=fixture.repo)
            _command("git", "commit", "-m", "pin parser", cwd=fixture.repo)
            fixture._record(candidate, candidate_hash, score=2.0, phase="confirmation")
            with mock.patch.object(admin, "_active_containers", return_value=[]):
                admin.adopt_baseline(
                    manifest, candidate_hash=candidate_hash, doctor=False
                )
            source = fixture.runtime / DEPLOYMENT_BASELINE_FILENAME
            valid = source.read_bytes()

            cases = {
                "mode": valid,
                "repeats JSON key": valid.replace(
                    b"{\n",
                    b'{\n  "schema_version": 1,\n',
                    1,
                ),
                "invalid JSON constant": valid.replace(
                    f'"confirmation_experiment_id": '.encode("ascii")
                    + str(
                        DeploymentBaselinePin.load(source).confirmation_experiment_id
                    ).encode("ascii"),
                    b'"confirmation_experiment_id": NaN',
                    1,
                ),
                "size bound": b"x" * (MAX_DEPLOYMENT_BASELINE_BYTES + 1),
            }
            for message, content in cases.items():
                with self.subTest(message=message):
                    path = fixture.runtime / f"invalid-{message.replace(' ', '-')}.json"
                    path.write_bytes(content)
                    path.chmod(0o644 if message == "mode" else 0o600)
                    with self.assertRaisesRegex(ValueError, message):
                        DeploymentBaselinePin.load(path)

            link = fixture.runtime / "pin-link.json"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "symlink"):
                DeploymentBaselinePin.load(link)

    def test_deployment_runtime_root_requires_three_sibling_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(
                deployment_runtime_root(
                    state_dir=root / "state",
                    controller_dir=root / "controller",
                    checkpoint_dir=root / "checkpoints",
                ),
                root,
            )
            with self.assertRaisesRegex(ValueError, "share one"):
                deployment_runtime_root(
                    state_dir=root / "state",
                    controller_dir=root / "controller",
                    checkpoint_dir=root / "other" / "checkpoints",
                )

    def test_deployment_runtime_root_rejects_relative_unresolvable_and_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, "absolute path"):
                deployment_runtime_root(
                    state_dir=Path("state"),
                    controller_dir=root / "controller",
                    checkpoint_dir=root / "checkpoints",
                )
            with (
                mock.patch.object(
                    Path,
                    "resolve",
                    side_effect=RuntimeError("resolution loop"),
                ),
                self.assertRaisesRegex(ValueError, "resolved canonically"),
            ):
                deployment_runtime_root(
                    state_dir=root / "state",
                    controller_dir=root / "controller",
                    checkpoint_dir=root / "checkpoints",
                )
        with self.assertRaisesRegex(ValueError, "filesystem root"):
            deployment_runtime_root(
                state_dir=Path("/state"),
                controller_dir=Path("/controller"),
                checkpoint_dir=Path("/checkpoints"),
            )

    def test_pin_value_contract_rejects_shape_schema_and_bad_uids(self) -> None:
        valid = _valid_deployment_pin().to_dict()
        cases: tuple[tuple[str, object, str], ...] = (
            ("not-object", [], "JSON object"),
            (
                "unknown-field",
                {**valid, "untrusted": True},
                "unknown fields",
            ),
            (
                "missing-field",
                {key: value for key, value in valid.items() if key != "git_commit"},
                "missing fields",
            ),
            (
                "boolean-schema",
                {**valid, "schema_version": True},
                "schema_version must be 1",
            ),
            (
                "empty-primary-uid",
                {**valid, "primary_experiment_uid": ""},
                "primary_experiment_uid",
            ),
            (
                "oversized-primary-uid",
                {**valid, "primary_experiment_uid": "x" * 257},
                "primary_experiment_uid",
            ),
            (
                "nul-confirmation-uid",
                {**valid, "confirmation_experiment_uid": "bad\x00uid"},
                "confirmation_experiment_uid",
            ),
            (
                "typed-confirmation-uid",
                {**valid, "confirmation_experiment_uid": 7},
                "confirmation_experiment_uid",
            ),
            (
                "malformed-baseline",
                {**valid, "baseline_ref": []},
                "baseline reference",
            ),
            (
                "malformed-parent",
                {**valid, "parent_baseline_ref": []},
                "baseline reference",
            ),
            (
                "malformed-environment",
                {**valid, "execution_environment": {}},
                "execution environment fields",
            ),
        )
        for name, value, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, message
            ):
                DeploymentBaselinePin.from_value(value)

    def test_pin_object_rejects_invalid_identity_parent_environment_and_digest(self) -> None:
        pin = _valid_deployment_pin()
        other_namespace = "sha256:" + "f" * 64
        other_environment = ExecutionEnvironmentDigest.resolved(
            evaluator_image_digest=pin.execution_environment.evaluator_image_digest,
            toolchain_digest=pin.execution_environment.toolchain_digest,
            framework_digest=pin.execution_environment.framework_digest,
            operator_abi_digest=pin.execution_environment.operator_abi_digest,
            build_flags_digest="sha256:" + "e" * 64,
        )
        campaign_baseline = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=pin.baseline_ref.artifact_id,
            source="campaign",
            revision="wrong-owner",
            execution_environment=pin.execution_environment,
        )
        foreign_baseline = BaselineRef.create(
            namespace=other_namespace,
            artifact_id=pin.baseline_ref.artifact_id,
            source="deployment",
            revision="wrong-namespace",
            execution_environment=pin.execution_environment,
        )
        foreign_parent = BaselineRef.create(
            namespace=other_namespace,
            artifact_id=pin.parent_baseline_ref.artifact_id,
            source="deployment",
            revision="wrong-parent-namespace",
            execution_environment=pin.execution_environment,
        )
        mismatched_environment_baseline = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=pin.baseline_ref.artifact_id,
            source="deployment",
            revision="wrong-environment",
            execution_environment=other_environment,
        )
        cases = (
            ("baseline-type", {"baseline_ref": {}}, "must be a BaselineRef"),
            (
                "baseline-source",
                {"baseline_ref": campaign_baseline},
                "source must be deployment",
            ),
            (
                "baseline-namespace",
                {"baseline_ref": foreign_baseline},
                "another namespace",
            ),
            (
                "candidate-hash",
                {"candidate_hash": "A" * 64},
                "candidate_hash",
            ),
            ("git-object", {"git_commit": "a" * 39}, "Git object ID"),
            (
                "same-uids",
                {"primary_experiment_uid": pin.confirmation_experiment_uid},
                "must differ",
            ),
            (
                "confirmation-id-bool",
                {"confirmation_experiment_id": True},
                "positive integer",
            ),
            (
                "parent-type",
                {"parent_baseline_ref": "not-a-reference"},
                "BaselineRef or null",
            ),
            (
                "parent-namespace",
                {"parent_baseline_ref": foreign_parent},
                "parent baseline belongs",
            ),
            (
                "environment-type",
                {"execution_environment": {}},
                "ExecutionEnvironmentDigest",
            ),
            (
                "environment-mismatch",
                {
                    "baseline_ref": mismatched_environment_baseline,
                    "execution_environment": pin.execution_environment,
                },
                "execution environments differ",
            ),
            (
                "digest-shape",
                {"evidence_digest": "not-a-digest"},
                "evidence_digest",
            ),
            (
                "digest-mismatch",
                {"evidence_digest": "sha256:" + "0" * 64},
                "evidence_digest mismatch",
            ),
        )
        for name, changes, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, message
            ):
                replace(pin, **changes)

    def test_pin_loader_rejects_stat_file_kind_and_read_failures(self) -> None:
        pin = _valid_deployment_pin()
        raw = (json.dumps(pin.to_dict(), sort_keys=True) + "\n").encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            missing = root / "missing.json"
            with self.assertRaisesRegex(ValueError, "could not stat"):
                DeploymentBaselinePin.load(missing)

            directory = root / "pin-directory"
            directory.mkdir(mode=0o700)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                DeploymentBaselinePin.load(directory)

            empty = root / "empty.json"
            empty.write_bytes(b"")
            empty.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "size bound"):
                DeploymentBaselinePin.load(empty)

            valid = root / "valid.json"
            valid.write_bytes(raw)
            valid.chmod(0o600)
            with (
                mock.patch.object(
                    Path, "read_bytes", side_effect=PermissionError("denied")
                ),
                self.assertRaisesRegex(ValueError, "could not read"),
            ):
                DeploymentBaselinePin.load(valid)
            with (
                mock.patch.object(Path, "read_bytes", return_value=raw + b"x"),
                self.assertRaisesRegex(ValueError, "changed while being read"),
            ):
                DeploymentBaselinePin.load(valid)
            original_stat = Path.stat

            def fail_following_stat(path, *args, **kwargs):
                if kwargs.get("follow_symlinks") is False:
                    return original_stat(path, *args, **kwargs)
                raise OSError("denied")

            with (
                mock.patch.object(Path, "stat", new=fail_following_stat),
                self.assertRaisesRegex(ValueError, "could not stat"),
            ):
                DeploymentBaselinePin.load(valid)

            invalid_utf8 = root / "invalid-utf8.json"
            invalid_utf8.write_bytes(b"\xff")
            invalid_utf8.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "could not read"):
                DeploymentBaselinePin.load(invalid_utf8)

            invalid_json = root / "invalid-json.json"
            invalid_json.write_bytes(b"{")
            invalid_json.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "could not read"):
                DeploymentBaselinePin.load(invalid_json)

            real_dir = root / "real"
            real_dir.mkdir()
            nested_pin = real_dir / "pin.json"
            nested_pin.write_bytes(raw)
            nested_pin.chmod(0o600)
            linked_dir = root / "linked"
            linked_dir.symlink_to(real_dir, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                DeploymentBaselinePin.load(linked_dir / "pin.json")

    def test_resolved_pin_rebinds_deployment_but_not_framework_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            evidence = CurrentEvidenceFixture(
                fixture,
                manifest,
                SEED + "\n# commit-bound evidence\n",
            )
            with mock.patch.object(admin, "_active_containers", return_value=[]):
                admin.adopt_baseline(
                    manifest,
                    candidate_hash=evidence.candidate_hash,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    doctor=False,
                )
            old_config = ControllerConfig.load(manifest.pro_config)
            old_pin = DeploymentBaselinePin.load(
                fixture.runtime / DEPLOYMENT_BASELINE_FILENAME
            )
            (fixture.repo / "README.md").write_text(
                "new framework commit\n", encoding="utf-8"
            )
            _command("git", "add", "README.md", cwd=fixture.repo)
            _command("git", "commit", "-m", "framework change", cwd=fixture.repo)
            new_commit = _command("git", "rev-parse", "HEAD", cwd=fixture.repo)
            with self.assertRaisesRegex(
                ControlledRuntimeError, "cannot cross a framework commit"
            ):
                admin._deployment_pin_for_commit(
                    old_config,
                    git_commit=new_commit,
                    framework_git_commit=new_commit,
                )
            with mock.patch.object(
                admin, "_active_containers", return_value=[]
            ):
                admin.sync(manifest)
            updated = ControllerConfig.load(manifest.pro_config)
            updated_pin = DeploymentBaselinePin.load(
                fixture.runtime / DEPLOYMENT_BASELINE_FILENAME
            )
            self.assertEqual(updated.expected_git_commit, new_commit)
            self.assertEqual(
                updated.resolved_framework_git_commit,
                old_config.resolved_framework_git_commit,
            )
            self.assertEqual(updated_pin.git_commit, new_commit)
            self.assertEqual(
                updated_pin.execution_environment,
                old_pin.execution_environment,
            )

    def test_current_start_rereads_authoritative_bundle_after_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            evidence = CurrentEvidenceFixture(
                fixture,
                manifest,
                SEED + "\n# post-adoption CAS check\n",
            )
            with mock.patch.object(admin, "_active_containers", return_value=[]):
                admin.adopt_baseline(
                    manifest,
                    candidate_hash=evidence.candidate_hash,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    doctor=False,
                )
            evidence.bundle_path.write_bytes(b"tampered after adoption")
            controller = ResearchController(
                ControllerConfig.load(manifest.pro_config)
            )
            with self.assertRaisesRegex(
                ControlledRuntimeError, "bundle/source CAS object is corrupted"
            ):
                controller._best()

    def test_current_start_rejects_pin_formal_config_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            evidence = CurrentEvidenceFixture(
                fixture,
                manifest,
                SEED + "\n# config mismatch after adoption\n",
            )
            with mock.patch.object(admin, "_active_containers", return_value=[]):
                admin.adopt_baseline(
                    manifest,
                    candidate_hash=evidence.candidate_hash,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    doctor=False,
                )
            config = replace(
                ControllerConfig.load(manifest.pro_config),
                expected_kernel_hash="f" * 64,
            )
            controller = ResearchController(config)
            with (
                mock.patch.object(
                    controller,
                    "doctor",
                    return_value={"status": "SUCCESS", "errors": []},
                ),
                self.assertRaisesRegex(
                    ControlledRuntimeError, "differs from formal config identity"
                ),
            ):
                controller.start(proposal_only=True)

    def test_current_start_detects_bundle_manifest_and_source_disagreement(self) -> None:
        mutations = ("manifest", "bundle-size", "source-size", "source-content")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = AdminFixture(Path(temporary))
                manifest = fixture.bootstrap()
                evidence = CurrentEvidenceFixture(
                    fixture,
                    manifest,
                    SEED + f"\n# post-adoption {mutation}\n",
                )
                with mock.patch.object(admin, "_active_containers", return_value=[]):
                    admin.adopt_baseline(
                        manifest,
                        candidate_hash=evidence.candidate_hash,
                        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                        doctor=False,
                    )
                source_id = str(
                    ArtifactId.source_sha256(evidence.candidate_hash)
                )
                with HistoryStore(
                    fixture.state / "history.sqlite3", state_dir=fixture.state
                ) as history:
                    if mutation == "manifest":
                        history._connection.execute(
                            """
                            UPDATE candidate_artifacts
                            SET manifest_json = ?
                            WHERE artifact_id = ?
                            """,
                            (json.dumps({"entrypoint": "wrong.py"}), evidence.bundle_id),
                        )
                    elif mutation == "bundle-size":
                        history._connection.execute(
                            """
                            UPDATE candidate_artifacts
                            SET byte_size = byte_size + 1
                            WHERE artifact_id = ?
                            """,
                            (evidence.bundle_id,),
                        )
                    elif mutation == "source-size":
                        history._connection.execute(
                            """
                            UPDATE candidate_artifacts
                            SET byte_size = byte_size + 1
                            WHERE artifact_id = ?
                            """,
                            (source_id,),
                        )
                    else:
                        altered = evidence.candidate.encode("utf-8") + b"# altered\n"
                        evidence.source_path.write_bytes(altered)
                        history._connection.execute(
                            """
                            UPDATE candidate_artifacts
                            SET content_sha256 = ?, byte_size = ?
                            WHERE artifact_id = ?
                            """,
                            (
                                admin._sha256_bytes(altered),
                                len(altered),
                                source_id,
                            ),
                        )
                controller = ResearchController(
                    ControllerConfig.load(manifest.pro_config)
                )
                with self.assertRaisesRegex(
                    ControlledRuntimeError,
                    "bundle, manifest, entrypoint, and source CAS disagree",
                ):
                    controller.start(proposal_only=True)


if __name__ == "__main__":
    unittest.main()
