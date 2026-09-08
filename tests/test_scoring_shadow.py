from __future__ import annotations

import copy
import unittest

from kernel_research.scoring_shadow import (
    project_scoring_shadow_report,
    public_scoring_shadow_summary,
    require_scoring_shadow_profile,
    scoring_shadow_profile_snapshot,
    unavailable_scoring_shadow_report,
)
from kernel_research.device_timing import DeviceEventMeasurement, DeviceEventRound
from kernel_research.platform.canonical import canonical_sha256
from kernel_research.scoring_candidate_measurement import (
    SCORING_CANDIDATE_WORKER_REVISION,
    qualified_scoring_probe_environment_snapshot,
)


CANDIDATE_HASH = "a" * 64
INCUMBENT_HASH = "b" * 64


def successful_full_result() -> dict[str, object]:
    profile = scoring_shadow_profile_snapshot()
    baselines = profile["scoring_baseline"]["case_baseline_ms"]
    return {
        "status": "SUCCESS",
        "suite": "full",
        "candidate_hash": CANDIDATE_HASH,
        "baseline_candidate_hash": INCUMBENT_HASH,
        "cases": [
            {
                "case_id": case_id,
                "status": "PASS",
                "matched_ratio": 1.0,
                "p50_us": latency_ms * 1000.0,
                "baseline_p50_us": latency_ms * 1100.0,
            }
            for case_id, latency_ms in baselines.items()
        ],
    }


def _measurement(candidate: float, incumbent: float) -> dict[str, object]:
    return DeviceEventMeasurement(
        rounds=tuple(
            DeviceEventRound(
                round_index=index,
                order="AB" if index % 2 == 0 else "BA",
                candidate_latency_ms=candidate,
                incumbent_latency_ms=incumbent,
            )
            for index in range(6)
        )
    ).to_dict()


def qualified_candidate_measurement() -> dict[str, object]:
    profile = scoring_shadow_profile_snapshot()
    cases = []
    for case_id, latency_ms in profile["scoring_baseline"][
        "case_baseline_ms"
    ].items():
        anchor = _measurement(latency_ms, latency_ms)
        paired = _measurement(latency_ms, latency_ms * 1.1)
        cases.append(
            {
                "case_id": case_id,
                "candidate_matched_ratio": 1.0,
                "candidate_fresh_output_ratio": 1.0,
                "candidate_anchor_median_ms": latency_ms,
                "candidate_anchor_measurement": anchor,
                "paired_candidate_matched_ratio": 1.0,
                "paired_incumbent_matched_ratio": 1.0,
                "paired_candidate_fresh_output_ratio": 1.0,
                "paired_incumbent_fresh_output_ratio": 1.0,
                "paired_candidate_median_ms": latency_ms,
                "paired_incumbent_median_ms": latency_ms * 1.1,
                "paired_safety_measurement": paired,
            }
        )
    environment = qualified_scoring_probe_environment_snapshot()
    return {
        "schema_version": 1,
        "command": "score-candidate-probe",
        "status": "QUALIFIED",
        "gpu_state": "COMPLETED",
        "completion_trusted": True,
        "phase": "complete",
        "protocol_id": "xpuoj-th0-proxy-v1",
        "worker_revision": SCORING_CANDIDATE_WORKER_REVISION,
        "scoring_framework_git_commit": "c" * 40,
        "scoring_profile_digest": profile["digest"],
        "candidate_hash": CANDIDATE_HASH,
        "incumbent_hash": INCUMBENT_HASH,
        "measurement_contract": profile["candidate_measurement_contract"],
        "timing_protocol": profile["candidate_measurement_contract"][
            "timing_protocol"
        ],
        "environment": environment,
        "environment_snapshot_digest": canonical_sha256(environment),
        "cases": cases,
    }


def operation_binding(measurement: dict[str, object]) -> dict[str, object]:
    operation_digest = canonical_sha256(
        {"operation": "test-candidate-shadow", "candidate": CANDIDATE_HASH}
    )
    return {
        "candidate_operation_id": (
            "score-candidate-" + operation_digest.removeprefix("sha256:")[:24]
        ),
        "candidate_operation_digest": operation_digest,
        "candidate_operation_result_digest": canonical_sha256(measurement),
        "candidate_operation_status": measurement["status"],
    }


class ScoringShadowProfileTests(unittest.TestCase):
    def test_profile_freezes_qualified_identity_without_promotion_authority(self) -> None:
        profile = scoring_shadow_profile_snapshot()

        self.assertEqual(profile["status"], "QUALIFIED")
        self.assertEqual(profile["mode"], "SHADOW_ONLY")
        self.assertFalse(profile["promotion_authority"])
        self.assertEqual(
            profile["qualification_operation_id"],
            "score-baseline-d7423c19f914afc877afc77b",
        )
        self.assertEqual(
            profile["qualification_evidence_manifest_sha256"],
            "sha256:4511014045b0172a40a68e156a7ce45efbb5c109199998f06148c1d6d2bced11",
        )
        self.assertEqual(
            profile["candidate_measurement_contract"]["worker_source_sha256"],
            "sha256:ee9a54a74b127e0a5f416a878d20b5f83f92c24b0929404b7344bcf5c251f3ef",
        )
        self.assertEqual(
            profile["candidate_measurement_contract"]["digest"],
            "sha256:be6c7d41da9b27c218d5c45d3693c2b6bf8958bff0930278b7c0689d59570517",
        )
        self.assertEqual(
            profile["digest"],
            "sha256:ff6601831d141e4c632d815f9bd965eaadf95a16842c311fec9ac3c09509ecb2",
        )
        self.assertEqual(
            require_scoring_shadow_profile(profile), profile
        )

    def test_profile_drift_is_rejected(self) -> None:
        profile = scoring_shadow_profile_snapshot()
        profile["promotion_authority"] = True
        with self.assertRaisesRegex(ValueError, "qualified activation"):
            require_scoring_shadow_profile(profile)


class ScoringShadowProjectionTests(unittest.TestCase):
    def test_successful_full_result_has_objective_and_paired_channels(self) -> None:
        report = project_scoring_shadow_report(
            successful_full_result(),
            suite="full",
            profile=scoring_shadow_profile_snapshot(),
            incumbent_history_experiment_id=210,
            incumbent_candidate_hash=INCUMBENT_HASH,
            candidate_measurement=qualified_candidate_measurement(),
            candidate_framework_git_commit="c" * 40,
            **operation_binding(qualified_candidate_measurement()),
        )

        self.assertEqual(report["status"], "AVAILABLE")
        self.assertIsNone(report["reason"])
        self.assertFalse(report["promotion_authority"])
        self.assertAlmostEqual(report["objective"]["objective_score"], 50.0)
        self.assertAlmostEqual(
            report["paired_safety"]["aggregate_speedup"], 1.1
        )
        self.assertEqual(report["paired_safety"]["worst_case_regression"], 0.0)
        self.assertEqual(len(report["objective"]["case_scores"]), 4)

        public = public_scoring_shadow_summary(report)
        self.assertEqual(public["status"], "AVAILABLE")
        self.assertEqual(public["objective_score"], 50.0)
        self.assertEqual(public["objective_delta_from_parity"], 0.0)
        self.assertAlmostEqual(public["paired_aggregate_speedup"], 1.1)
        self.assertNotIn("case_scores", public)
        self.assertNotIn("per_case_speedups", public)
        public_json = str(public)
        self.assertNotIn("candidate_measurement", public_json)
        self.assertNotIn("environment", public_json)
        self.assertNotIn("rounds", public_json)

    def test_non_full_and_failed_results_are_explicitly_unavailable(self) -> None:
        profile = scoring_shadow_profile_snapshot()
        quick = project_scoring_shadow_report(
            {"status": "SUCCESS", "candidate_hash": CANDIDATE_HASH},
            suite="quick",
            profile=profile,
            incumbent_history_experiment_id=210,
            incumbent_candidate_hash=INCUMBENT_HASH,
        )
        failed = project_scoring_shadow_report(
            {"status": "CRASH", "candidate_hash": CANDIDATE_HASH},
            suite="full",
            profile=profile,
            incumbent_history_experiment_id=210,
            incumbent_candidate_hash=INCUMBENT_HASH,
        )

        self.assertEqual(quick["reason"], "SUITE_NOT_FULL")
        self.assertEqual(failed["reason"], "RESULT_NOT_SUCCESS")
        for report in (quick, failed):
            self.assertEqual(report["status"], "UNAVAILABLE")
            self.assertIsNone(report["objective"])
            self.assertIsNone(report["paired_safety"])
            self.assertNotEqual(report.get("objective_score"), 0)

    def test_correctness_failure_is_not_scored(self) -> None:
        result = successful_full_result()
        result["cases"][0]["matched_ratio"] = 0.5
        report = project_scoring_shadow_report(
            result,
            suite="full",
            profile=scoring_shadow_profile_snapshot(),
            incumbent_history_experiment_id=210,
            incumbent_candidate_hash=INCUMBENT_HASH,
        )
        self.assertEqual(report["status"], "UNAVAILABLE")
        self.assertEqual(report["reason"], "CORRECTNESS_NOT_QUALIFIED")

    def test_successful_full_evidence_fails_closed_on_identity_or_case_drift(self) -> None:
        profile = scoring_shadow_profile_snapshot()
        wrong_baseline = successful_full_result()
        wrong_baseline["baseline_candidate_hash"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "frozen incumbent"):
            project_scoring_shadow_report(
                wrong_baseline,
                suite="full",
                profile=profile,
                incumbent_history_experiment_id=210,
                incumbent_candidate_hash=INCUMBENT_HASH,
                candidate_measurement=qualified_candidate_measurement(),
                candidate_framework_git_commit="c" * 40,
                **operation_binding(qualified_candidate_measurement()),
            )

        missing_measurement_case = qualified_candidate_measurement()
        missing_measurement_case["cases"] = missing_measurement_case["cases"][:-1]
        with self.assertRaisesRegex(ValueError, "case IDs"):
            project_scoring_shadow_report(
                successful_full_result(),
                suite="full",
                profile=profile,
                incumbent_history_experiment_id=210,
                incumbent_candidate_hash=INCUMBENT_HASH,
                candidate_measurement=missing_measurement_case,
                candidate_framework_git_commit="c" * 40,
                **operation_binding(missing_measurement_case),
            )

    def test_legacy_host_timer_fields_cannot_create_an_absolute_score(self) -> None:
        report = project_scoring_shadow_report(
            successful_full_result(),
            suite="full",
            profile=scoring_shadow_profile_snapshot(),
            incumbent_history_experiment_id=210,
            incumbent_candidate_hash=INCUMBENT_HASH,
        )
        self.assertEqual(report["status"], "UNAVAILABLE")
        self.assertEqual(
            report["reason"], "CANDIDATE_MEASUREMENT_NOT_COLLECTED"
        )
        self.assertIsNone(report["objective"])

    def test_unqualified_probe_is_bounded_and_not_scored(self) -> None:
        qualified = qualified_candidate_measurement()
        unqualified = {
            key: value
            for key, value in qualified.items()
            if key
            not in {"environment", "environment_snapshot_digest"}
        }
        unqualified.update(
            {
                "status": "UNQUALIFIED",
                "gpu_state": "COMPLETED",
                "completion_trusted": True,
                "phase": "correctness",
                "error": "RuntimeError: correctness failed",
                "traceback": "private diagnostic",
                "cases": [],
                "container_exit_code": 2,
            }
        )
        report = project_scoring_shadow_report(
            successful_full_result(),
            suite="full",
            profile=scoring_shadow_profile_snapshot(),
            incumbent_history_experiment_id=210,
            incumbent_candidate_hash=INCUMBENT_HASH,
            candidate_measurement=unqualified,
            candidate_framework_git_commit="c" * 40,
            **operation_binding(unqualified),
        )
        self.assertEqual(report["status"], "UNAVAILABLE")
        self.assertEqual(report["reason"], "CANDIDATE_MEASUREMENT_UNQUALIFIED")
        self.assertNotIn("traceback", report["candidate_measurement"])
        self.assertIsNone(report["objective"])
        self.assertEqual(
            public_scoring_shadow_summary(report)["status"], "UNAVAILABLE"
        )

    def test_unavailable_report_is_digest_bound_and_legacy_is_not_reinterpreted(self) -> None:
        report = unavailable_scoring_shadow_report(
            reason="PROFILE_NOT_FROZEN",
            profile=None,
            candidate_hash=CANDIDATE_HASH,
            incumbent_history_experiment_id=210,
            incumbent_candidate_hash=INCUMBENT_HASH,
        )
        self.assertEqual(report["status"], "UNAVAILABLE")
        self.assertIsNone(report["profile_digest"])
        self.assertTrue(str(report["digest"]).startswith("sha256:"))

        legacy = public_scoring_shadow_summary(None)
        self.assertEqual(legacy["status"], "UNAVAILABLE")
        self.assertEqual(legacy["reason"], "NOT_RECORDED")
        self.assertIsNone(legacy["objective_score"])

        tampered = copy.deepcopy(report)
        tampered["status"] = "AVAILABLE"
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            public_scoring_shadow_summary(tampered)

    def test_report_identity_and_unavailable_reason_fail_closed(self) -> None:
        for field, value in (
            ("candidate_hash", "A" * 64),
            ("incumbent_candidate_hash", "short"),
            ("incumbent_history_experiment_id", True),
        ):
            arguments = {
                "reason": "PROFILE_NOT_FROZEN",
                "profile": None,
                "candidate_hash": CANDIDATE_HASH,
                "incumbent_history_experiment_id": 210,
                "incumbent_candidate_hash": INCUMBENT_HASH,
            }
            arguments[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                unavailable_scoring_shadow_report(**arguments)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            unavailable_scoring_shadow_report(
                reason="UNKNOWN_REASON",
                profile=None,
                candidate_hash=CANDIDATE_HASH,
                incumbent_history_experiment_id=210,
                incumbent_candidate_hash=INCUMBENT_HASH,
            )

    def test_candidate_measurement_tampering_is_rejected(self) -> None:
        mutations = (
            ("identity", lambda value: value.update(candidate_hash="c" * 64)),
            (
                "environment",
                lambda value: value.update(environment_snapshot_digest="sha256:" + "0" * 64),
            ),
            (
                "ratio",
                lambda value: value["cases"][0].update(candidate_matched_ratio=0.5),
            ),
            (
                "anchor",
                lambda value: value["cases"][0].update(candidate_anchor_median_ms=999.0),
            ),
            (
                "paired",
                lambda value: value["cases"][0].update(paired_incumbent_median_ms=999.0),
            ),
            ("extra", lambda value: value.update(untrusted=True)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                measurement_value = qualified_candidate_measurement()
                mutate(measurement_value)
                with self.assertRaises(ValueError):
                    project_scoring_shadow_report(
                        successful_full_result(),
                        suite="full",
                        profile=scoring_shadow_profile_snapshot(),
                        incumbent_history_experiment_id=210,
                        incumbent_candidate_hash=INCUMBENT_HASH,
                        candidate_measurement=measurement_value,
                        candidate_framework_git_commit="c" * 40,
                        **operation_binding(measurement_value),
                    )

    def test_public_projection_rejects_structural_drift(self) -> None:
        report = project_scoring_shadow_report(
            successful_full_result(),
            suite="full",
            profile=scoring_shadow_profile_snapshot(),
            incumbent_history_experiment_id=210,
            incumbent_candidate_hash=INCUMBENT_HASH,
            candidate_measurement=qualified_candidate_measurement(),
            candidate_framework_git_commit="c" * 40,
            **operation_binding(qualified_candidate_measurement()),
        )
        variants = []
        missing = copy.deepcopy(report)
        missing.pop("candidate_measurement")
        variants.append(missing)
        wrong_authority = copy.deepcopy(report)
        wrong_authority["promotion_authority"] = True
        wrong_authority["digest"] = canonical_sha256(
            {key: value for key, value in wrong_authority.items() if key != "digest"}
        )
        variants.append(wrong_authority)
        wrong_measurement = copy.deepcopy(report)
        wrong_measurement["candidate_measurement"]["environment"] = {"drift": True}
        wrong_measurement["digest"] = canonical_sha256(
            {key: value for key, value in wrong_measurement.items() if key != "digest"}
        )
        variants.append(wrong_measurement)
        self_consistent_environment_drift = copy.deepcopy(report)
        drifted_measurement = self_consistent_environment_drift[
            "candidate_measurement"
        ]
        drifted_measurement["environment"] = {"drift": True}
        drifted_measurement["environment_snapshot_digest"] = canonical_sha256(
            drifted_measurement["environment"]
        )
        self_consistent_environment_drift["candidate_measurement_digest"] = (
            canonical_sha256(drifted_measurement)
        )
        self_consistent_environment_drift["digest"] = canonical_sha256(
            {
                key: value
                for key, value in self_consistent_environment_drift.items()
                if key != "digest"
            }
        )
        variants.append(self_consistent_environment_drift)
        activation_drift = copy.deepcopy(report)
        activation_drift["profile_digest"] = "sha256:" + "0" * 64
        activation_drift["digest"] = canonical_sha256(
            {key: value for key, value in activation_drift.items() if key != "digest"}
        )
        variants.append(activation_drift)
        for value in variants:
            with self.assertRaises(ValueError):
                public_scoring_shadow_summary(value)


if __name__ == "__main__":
    unittest.main()
