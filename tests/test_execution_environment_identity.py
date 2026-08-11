from __future__ import annotations

from dataclasses import replace
import unittest

from kernel_research.platform import (
    ArtifactId,
    BaselineRef,
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
    LEGACY_RESEARCH_NAMESPACE,
    RunBudget,
    RunSpecV2,
    resolve_run_spec,
)


IMAGE_A = "registry.invalid/evaluator@sha256:" + "1" * 64
IMAGE_B = "registry.invalid/evaluator@sha256:" + "2" * 64
ARTIFACT = ArtifactId.source_sha256("a" * 64)


def _ref(kind: str, profile_id: str, revision: str):
    return BUILTIN_PROFILE_REGISTRY.get(
        kind=kind, profile_id=profile_id, revision=revision
    ).ref


def _bindings() -> dict:
    return {
        "evaluator_image": IMAGE_A,
        "toolchain": {"compiler": "metax-triton", "version": "1.0"},
        "framework": {"git_commit": "b" * 40},
        "operator_abi": {
            "profile": LEGACY_RESEARCH_NAMESPACE.operator.to_dict(),
            "abi": "run-kernel-v1",
        },
        "build_flags": ["--opt-level=3"],
    }


def _environment(bindings: dict | None = None) -> ExecutionEnvironmentDigest:
    return ExecutionEnvironmentDigest.from_bindings(**(bindings or _bindings()))


def _baseline(environment: ExecutionEnvironmentDigest) -> BaselineRef:
    return BaselineRef.create(
        namespace=LEGACY_RESEARCH_NAMESPACE,
        artifact_id=ARTIFACT,
        source="deployment",
        revision="deployment-test",
        execution_environment=environment,
    )


def _spec(environment: ExecutionEnvironmentDigest) -> RunSpecV2:
    namespace = LEGACY_RESEARCH_NAMESPACE
    return RunSpecV2(
        deployment=_ref("deployment", "legacy-local-c500", "v1"),
        operator=namespace.operator,
        language=namespace.language,
        evaluator=namespace.evaluator,
        evaluation_protocol=namespace.evaluation_protocol,
        promotion_policy=namespace.promotion_policy,
        proposer=_ref("proposer", "opencode-deepseek-v4-pro", "v1"),
        baseline=_baseline(environment),
        budget=RunBudget(),
    )


def _identity(
    environment: ExecutionEnvironmentDigest, *, uid_suffix: int
) -> ExperimentIdentity:
    return ExperimentIdentity.create(
        experiment_uid=(
            "00000000-0000-4000-8000-" + f"{uid_suffix:012d}"
        ),
        namespace=LEGACY_RESEARCH_NAMESPACE,
        mode="DISCOVERY",
        candidate_artifact_id=ArtifactId.source_sha256("c" * 64),
        parent_artifact_id=ARTIFACT,
        baseline=_baseline(environment),
        execution_environment=environment,
        stage="QUICK",
        suite="quick",
        run_id="run-environment-test",
        iteration=1,
    )


class ExecutionEnvironmentIdentityTests(unittest.TestCase):
    def _mutations(self):
        return {
            "evaluator image": lambda value: value.update(
                evaluator_image=IMAGE_B
            ),
            "toolchain": lambda value: value.update(
                toolchain={"compiler": "metax-triton", "version": "2.0"}
            ),
            "framework": lambda value: value.update(
                framework={"git_commit": "d" * 40}
            ),
            "operator ABI": lambda value: value.update(
                operator_abi={
                    "profile": LEGACY_RESEARCH_NAMESPACE.operator.to_dict(),
                    "abi": "run-kernel-v2",
                }
            ),
            "build flags": lambda value: value.update(
                build_flags=["--opt-level=0"]
            ),
        }

    def test_all_execution_dimensions_isolate_condition_and_baseline(self) -> None:
        original_bindings = _bindings()
        original_environment = _environment(original_bindings)
        original_identity = _identity(original_environment, uid_suffix=1)

        for name, mutate in self._mutations().items():
            with self.subTest(binding=name):
                changed_bindings = _bindings()
                mutate(changed_bindings)
                changed_environment = _environment(changed_bindings)
                self.assertNotEqual(
                    original_environment.digest, changed_environment.digest
                )
                changed_identity = _identity(
                    changed_environment, uid_suffix=2
                )
                self.assertNotEqual(
                    original_identity.condition_digest,
                    changed_identity.condition_digest,
                )
                with self.assertRaisesRegex(
                    ValueError, "execution environment mismatch"
                ):
                    ExperimentIdentity.create(
                        namespace=LEGACY_RESEARCH_NAMESPACE,
                        mode="DISCOVERY",
                        candidate_artifact_id=ArtifactId.source_sha256(
                            "c" * 64
                        ),
                        parent_artifact_id=ARTIFACT,
                        baseline=_baseline(original_environment),
                        execution_environment=changed_environment,
                        stage="QUICK",
                        suite="quick",
                        run_id="run-mismatch",
                        iteration=1,
                    )
                with self.assertRaisesRegex(
                    ValueError, "execution environment mismatch"
                ):
                    resolve_run_spec(
                        _spec(original_environment),
                        runtime_bindings={
                            "execution_environment": changed_bindings
                        },
                    )

    def test_resolved_snapshot_freezes_and_audits_environment(self) -> None:
        environment = _environment()
        snapshot = resolve_run_spec(
            _spec(environment),
            runtime_bindings={"execution_environment": _bindings()},
        )
        value = snapshot.to_dict()
        self.assertTrue(value["scientifically_comparable"])
        self.assertEqual(
            value["execution_environment"], environment.to_dict()
        )
        self.assertEqual(snapshot.spec.baseline.to_dict()["schema_version"], 2)
        self.assertEqual(
            BaselineRef.from_value(snapshot.spec.baseline.to_dict()),
            snapshot.spec.baseline,
        )
        identity = _identity(environment, uid_suffix=3)
        self.assertEqual(identity.to_dict()["schema_version"], 2)
        self.assertEqual(ExperimentIdentity.from_value(identity.to_dict()), identity)

        with self.assertRaisesRegex(
            ValueError, "missing runtime execution_environment"
        ):
            resolve_run_spec(_spec(environment))
        bad_definitions = {
            key: dict(value) for key, value in snapshot.definitions.items()
        }
        bad_definitions["language"]["config"] = {
            **bad_definitions["language"]["config"],
            "toolchain_fingerprint": "silently-changed",
        }
        with self.assertRaisesRegex(ValueError, "digest"):
            replace(snapshot, definitions=bad_definitions)

    def test_legacy_unknown_is_compatible_but_never_comparable(self) -> None:
        baseline = BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=ARTIFACT,
            source="deployment",
            revision="legacy",
        )
        self.assertEqual(baseline.to_dict()["schema_version"], 1)
        self.assertEqual(
            baseline.execution_environment.status, "LEGACY_UNKNOWN"
        )
        self.assertFalse(baseline.is_scientifically_comparable)
        with self.assertRaisesRegex(ValueError, "not scientifically comparable"):
            baseline.require_environment(baseline.execution_environment)

        identity = ExperimentIdentity.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            mode="DISCOVERY",
            candidate_artifact_id=ARTIFACT,
            parent_artifact_id=ARTIFACT,
            baseline=baseline,
            stage="SMOKE",
            suite="smoke",
            run_id="legacy-run",
            iteration=0,
        )
        self.assertEqual(identity.to_dict()["schema_version"], 1)
        self.assertFalse(identity.is_scientifically_comparable)
        self.assertEqual(ExperimentIdentity.from_value(identity.to_dict()), identity)
        snapshot = resolve_run_spec(
            _spec(baseline.execution_environment),
            runtime_bindings={"credential_ref": "legacy-key-file"},
        )
        self.assertFalse(snapshot.to_dict()["scientifically_comparable"])
        self.assertEqual(
            snapshot.to_dict()["execution_environment"]["status"],
            "LEGACY_UNKNOWN",
        )

        current = CURRENT_RESEARCH_NAMESPACE
        current_spec = replace(
            _spec(baseline.execution_environment),
            operator=current.operator,
            language=current.language,
            evaluator=current.evaluator,
            evaluation_protocol=current.evaluation_protocol,
            promotion_policy=current.promotion_policy,
            baseline=BaselineRef.create(
                namespace=current,
                artifact_id=ARTIFACT,
                source="deployment",
                revision="current-unresolved",
            ),
        )
        with self.assertRaisesRegex(ValueError, "only the legacy namespace"):
            resolve_run_spec(current_spec)

    def test_unpinned_image_and_incomplete_environment_fail_closed(self) -> None:
        bindings = _bindings()
        bindings["evaluator_image"] = "registry.invalid/evaluator:latest"
        with self.assertRaisesRegex(ValueError, "pinned"):
            _environment(bindings)
        bindings = _bindings()
        del bindings["framework"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            resolve_run_spec(
                _spec(_environment()),
                runtime_bindings={"execution_environment": bindings},
            )


if __name__ == "__main__":
    unittest.main()
