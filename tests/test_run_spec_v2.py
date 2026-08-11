from __future__ import annotations

from dataclasses import replace
import unittest

from kernel_research.platform.artifacts import ArtifactId
from kernel_research.platform.identity import BaselineRef
from kernel_research.platform.profiles import (
    BUILTIN_PROFILE_REGISTRY,
    LEGACY_RESEARCH_NAMESPACE,
)
from kernel_research.platform.run_spec import (
    RunBudget,
    RunSpecV2,
    resolve_run_spec,
)


def ref(kind: str, profile_id: str, revision: str):
    return BUILTIN_PROFILE_REGISTRY.get(
        kind=kind, profile_id=profile_id, revision=revision
    ).ref


def spec(*, mode: str = "DISCOVERY") -> RunSpecV2:
    namespace = LEGACY_RESEARCH_NAMESPACE
    return RunSpecV2(
        deployment=ref("deployment", "legacy-local-c500", "v1"),
        operator=namespace.operator,
        language=namespace.language,
        evaluator=namespace.evaluator,
        evaluation_protocol=namespace.evaluation_protocol,
        promotion_policy=namespace.promotion_policy,
        proposer=ref("proposer", "opencode-deepseek-v4-pro", "v1"),
        baseline=BaselineRef.create(
            namespace=namespace,
            artifact_id=ArtifactId.source_sha256("1" * 64),
            source="deployment",
            revision="deployment-test",
        ),
        budget=RunBudget(),
        mode=mode,
        cohort_id="cohort-a" if mode == "BENCHMARK" else None,
        history_cutoff=12 if mode == "BENCHMARK" else None,
    )


class RunSpecV2Tests(unittest.TestCase):
    def test_resolution_is_complete_and_digest_is_stable(self) -> None:
        first = resolve_run_spec(
            spec(), runtime_bindings={"credential_ref": "deepseek-key-file"}
        )
        second = resolve_run_spec(
            spec(), runtime_bindings={"credential_ref": "deepseek-key-file"}
        )
        self.assertEqual(first.snapshot_digest, second.snapshot_digest)
        value = first.to_dict()
        self.assertEqual(value["namespace"]["namespace_id"], spec().namespace.namespace_id)
        self.assertEqual(
            set(value["resolved_profiles"]),
            {
                "deployment",
                "operator",
                "language",
                "evaluator",
                "evaluation_protocol",
                "promotion_policy",
                "proposer",
            },
        )

    def test_budget_cannot_expand_single_run_safety(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            RunBudget(max_candidates=6)
        with self.assertRaisesRegex(ValueError, "stop after"):
            RunBudget(stop_after_promotion=False)

    def test_baseline_and_benchmark_cutoff_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "cohort_id"):
            replace(spec(), mode="BENCHMARK")
        wrong = BaselineRef.create(
            namespace="sha256:" + "f" * 64,
            artifact_id=ArtifactId.source_sha256("1" * 64),
            source="deployment",
            revision="wrong",
        )
        with self.assertRaisesRegex(ValueError, "different research namespace"):
            replace(spec(), baseline=wrong)

    def test_secret_values_are_forbidden_but_refs_are_allowed(self) -> None:
        resolve_run_spec(spec(), runtime_bindings={"credential_ref": "key-file"})
        with self.assertRaisesRegex(ValueError, "secret content"):
            resolve_run_spec(spec(), runtime_bindings={"api_key": "plaintext"})
        with self.assertRaisesRegex(ValueError, "secret content"):
            resolve_run_spec(spec(), runtime_bindings={"bearer_token": "plaintext"})

    def test_orchestration_identifiers_and_cutoff_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "history_cutoff"):
            replace(spec(), history_cutoff={"after": 12})
        with self.assertRaisesRegex(ValueError, "cohort_id"):
            replace(
                spec(mode="BENCHMARK"),
                cohort_id="contains whitespace",
            )
        with self.assertRaisesRegex(ValueError, "both be set"):
            replace(
                spec(),
                campaign=ref("campaign", "bounded-discovery", "v1"),
            )


if __name__ == "__main__":
    unittest.main()
