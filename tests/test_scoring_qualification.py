from __future__ import annotations

import copy
import unittest

from kernel_research.compiled_reference import (
    SCORING_COMPILER_CONFIG,
    scoring_reference_source_sha256,
)
from kernel_research.device_timing import device_event_protocol_snapshot
from kernel_research.device_timing import DeviceEventMeasurement, DeviceEventRound
from kernel_research.scoring_qualification import (
    MAX_RELATIVE_MAD,
    ScoringBaselineQualification,
    aggregate_scoring_baseline_probes,
    validate_anchor_drift,
)
from kernel_research.scoring_measurement import (
    MAX_AGGREGATE_PROBE_DEVIATION_POINTS,
    MAX_AGGREGATE_SCORE_MAD_POINTS,
    MAX_CASE_RELATIVE_MAD,
    scoring_baseline_measurement_contract_snapshot,
)


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
SCORING_COMMIT = "c" * 40


def measurement(candidate: float, incumbent: float) -> dict:
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


def probe(offset: float = 0.0) -> dict:
    cases = []
    for case_id, base in (("decode", 1.0), ("prefill", 10.0)):
        cases.append(
            {
                "case_id": case_id,
                "matched_ratio": 1.0,
                "eager_matched_ratio": 1.0,
                "compiler_first_invocation_seconds": 1.0,
                "compiled_to_eager_ratio": 0.9,
                "anchor_median_ms": base + offset,
                "anchor_measurement": measurement(
                    base + offset, base + offset
                ),
                "performance_measurement": measurement(0.9, 1.0),
            }
        )
    return {
        "schema_version": 1,
        "command": "score-baseline-probe",
        "status": "QUALIFIED",
        "protocol_id": "xpuoj-th0-proxy-v1",
        "scoring_framework_git_commit": SCORING_COMMIT,
        "reference_source_sha256": scoring_reference_source_sha256(),
        "compiler_config": dict(SCORING_COMPILER_CONFIG),
        "measurement_contract": scoring_baseline_measurement_contract_snapshot(),
        "timing_protocol": device_event_protocol_snapshot(),
        "environment_snapshot_digest": DIGEST_A,
        "cases": cases,
    }


def scale_anchor(probe_value: dict, scale: float) -> dict:
    value = copy.deepcopy(probe_value)
    for case in value["cases"]:
        anchor = float(case["anchor_median_ms"]) * scale
        case["anchor_median_ms"] = anchor
        case["anchor_measurement"] = measurement(anchor, anchor)
    return value


class ScoringQualificationTests(unittest.TestCase):
    def test_ten_stable_probes_create_frozen_descriptor_and_envelope(self) -> None:
        probes = [probe(index * 0.0001) for index in range(10)]
        qualification = aggregate_scoring_baseline_probes(
            probes,
            environment_digest=DIGEST_A,
            evaluator_profile_digest=DIGEST_B,
            scoring_framework_git_commit=SCORING_COMMIT,
        )
        self.assertTrue(qualification.qualified)
        self.assertEqual(qualification.reason, "qualified")
        self.assertEqual(len(qualification.probe_digests), 10)
        self.assertEqual(qualification.schema_version, 2)
        self.assertIsNotNone(qualification.aggregate_score_envelope)
        self.assertLessEqual(
            qualification.aggregate_score_envelope["mad"],
            MAX_AGGREGATE_SCORE_MAD_POINTS,
        )
        self.assertAlmostEqual(
            qualification.descriptor.case_baseline_ms["decode"], 1.00045
        )
        self.assertRegex(qualification.digest, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(
            ScoringBaselineQualification.from_value(qualification.to_dict()),
            qualification,
        )
        validate_anchor_drift(
            qualification,
            {
                case: envelope["p50"]
                for case, envelope in qualification.case_envelopes_ms.items()
            },
        )

    def test_serialized_qualification_rejects_tampered_derived_fields(self) -> None:
        qualification = aggregate_scoring_baseline_probes(
            [probe(index * 0.0001) for index in range(10)],
            environment_digest=DIGEST_A,
            evaluator_profile_digest=DIGEST_B,
            scoring_framework_git_commit=SCORING_COMMIT,
        )
        tampered = copy.deepcopy(qualification.to_dict())
        tampered["case_envelopes_ms"]["decode"]["p50"] = 2.0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            ScoringBaselineQualification.from_value(tampered)

        tampered = copy.deepcopy(qualification.to_dict())
        tampered["probe_digests"][0] = DIGEST_B
        with self.assertRaisesRegex(ValueError, "digest or fields differ"):
            ScoringBaselineQualification.from_value(tampered)

        tampered = copy.deepcopy(qualification.to_dict())
        tampered["aggregate_score_envelope"]["mad"] = 1.0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            ScoringBaselineQualification.from_value(tampered)

        tampered = copy.deepcopy(qualification.to_dict())
        tampered["qualification_stability"]["maximum_case_relative_mad"] = 2.0
        with self.assertRaisesRegex(ValueError, "stability contract"):
            ScoringBaselineQualification.from_value(tampered)

        tampered = copy.deepcopy(qualification.to_dict())
        tampered["aggregate_score_envelope"][
            "maximum_probe_deviation_points"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            ScoringBaselineQualification.from_value(tampered)

    def test_unstable_probes_remain_unqualified(self) -> None:
        probes = [probe(0.0) for _ in range(5)] + [probe(0.2) for _ in range(5)]
        qualification = aggregate_scoring_baseline_probes(
            probes,
            environment_digest=DIGEST_A,
            evaluator_profile_digest=DIGEST_B,
            scoring_framework_git_commit=SCORING_COMMIT,
        )
        self.assertFalse(qualification.qualified)
        self.assertEqual(qualification.reason, "case_relative_mad_exceeded")
        with self.assertRaisesRegex(ValueError, "not active"):
            validate_anchor_drift(
                qualification, {"decode": 1.0, "prefill": 10.0}
            )

    def test_aggregate_score_stability_is_an_independent_gate(self) -> None:
        probes = [
            scale_anchor(probe(), scale)
            for scale in ([0.991] * 5 + [1.009] * 5)
        ]
        qualification = aggregate_scoring_baseline_probes(
            probes,
            environment_digest=DIGEST_A,
            evaluator_profile_digest=DIGEST_B,
            scoring_framework_git_commit=SCORING_COMMIT,
        )
        self.assertTrue(
            all(
                envelope["relative_mad"] <= MAX_CASE_RELATIVE_MAD
                for envelope in qualification.case_envelopes_ms.values()
            )
        )
        self.assertGreater(
            qualification.aggregate_score_envelope["mad"],
            MAX_AGGREGATE_SCORE_MAD_POINTS,
        )
        self.assertFalse(qualification.qualified)
        self.assertEqual(qualification.reason, "aggregate_score_mad_exceeded")

    def test_one_aggregate_outlier_fails_closed_even_with_zero_mad(self) -> None:
        probes = [probe() for _ in range(9)] + [scale_anchor(probe(), 1.05)]
        qualification = aggregate_scoring_baseline_probes(
            probes,
            environment_digest=DIGEST_A,
            evaluator_profile_digest=DIGEST_B,
            scoring_framework_git_commit=SCORING_COMMIT,
        )
        self.assertTrue(
            all(
                envelope["relative_mad"] <= MAX_CASE_RELATIVE_MAD
                for envelope in qualification.case_envelopes_ms.values()
            )
        )
        self.assertEqual(qualification.aggregate_score_envelope["mad"], 0.0)
        self.assertGreater(
            qualification.aggregate_score_envelope[
                "maximum_probe_deviation_points"
            ],
            MAX_AGGREGATE_PROBE_DEVIATION_POINTS,
        )
        self.assertFalse(qualification.qualified)
        self.assertEqual(
            qualification.reason,
            "aggregate_score_probe_deviation_exceeded",
        )

    def test_schema_v1_qualification_remains_verifiable(self) -> None:
        current = aggregate_scoring_baseline_probes(
            [probe(index * 0.0001) for index in range(10)],
            environment_digest=DIGEST_A,
            evaluator_profile_digest=DIGEST_B,
            scoring_framework_git_commit=SCORING_COMMIT,
        )
        legacy = ScoringBaselineQualification(
            schema_version=1,
            descriptor=current.descriptor,
            probe_digests=current.probe_digests,
            case_envelopes_ms=current.case_envelopes_ms,
            qualified=True,
            reason="qualified",
        )
        self.assertEqual(
            ScoringBaselineQualification.from_value(legacy.to_dict()),
            legacy,
        )
        self.assertEqual(MAX_RELATIVE_MAD, 0.005)

    def test_probe_identity_and_anchor_drift_fail_closed(self) -> None:
        probes = [probe(index * 0.0001) for index in range(10)]
        tampered = copy.deepcopy(probes)
        tampered[-1]["environment_snapshot_digest"] = DIGEST_B
        with self.assertRaisesRegex(ValueError, "identity drift"):
            aggregate_scoring_baseline_probes(
                tampered,
                environment_digest=DIGEST_A,
                evaluator_profile_digest=DIGEST_B,
                scoring_framework_git_commit=SCORING_COMMIT,
            )
        tampered = copy.deepcopy(probes)
        tampered[-1]["measurement_contract"]["anchor_pairing"] = "eager"
        with self.assertRaisesRegex(ValueError, "measurement contract"):
            aggregate_scoring_baseline_probes(
                tampered,
                environment_digest=DIGEST_A,
                evaluator_profile_digest=DIGEST_B,
                scoring_framework_git_commit=SCORING_COMMIT,
            )
        tampered = copy.deepcopy(probes)
        tampered[-1]["cases"][0]["anchor_median_ms"] = 2.0
        with self.assertRaisesRegex(ValueError, "anchor median"):
            aggregate_scoring_baseline_probes(
                tampered,
                environment_digest=DIGEST_A,
                evaluator_profile_digest=DIGEST_B,
                scoring_framework_git_commit=SCORING_COMMIT,
            )
        tampered = copy.deepcopy(probes)
        tampered[-1]["cases"][0]["performance_measurement"] = measurement(
            1.02, 1.0
        )
        tampered[-1]["cases"][0]["compiled_to_eager_ratio"] = 1.02
        with self.assertRaisesRegex(ValueError, "compiled-to-eager"):
            aggregate_scoring_baseline_probes(
                tampered,
                environment_digest=DIGEST_A,
                evaluator_profile_digest=DIGEST_B,
                scoring_framework_git_commit=SCORING_COMMIT,
            )
        tampered = copy.deepcopy(probes)
        tampered[-1]["cases"][0]["matched_ratio"] = 0.98
        with self.assertRaisesRegex(ValueError, "correctness proof"):
            aggregate_scoring_baseline_probes(
                tampered,
                environment_digest=DIGEST_A,
                evaluator_profile_digest=DIGEST_B,
                scoring_framework_git_commit=SCORING_COMMIT,
            )
        qualification = aggregate_scoring_baseline_probes(
            probes,
            environment_digest=DIGEST_A,
            evaluator_profile_digest=DIGEST_B,
            scoring_framework_git_commit=SCORING_COMMIT,
        )
        with self.assertRaisesRegex(ValueError, "SCORING_ENVIRONMENT_DRIFT"):
            validate_anchor_drift(
                qualification, {"decode": 2.0, "prefill": 10.00045}
            )

    def test_wrong_probe_count_or_unqualified_probe_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ten"):
            aggregate_scoring_baseline_probes(
                [probe()] * 9,
                environment_digest=DIGEST_A,
                evaluator_profile_digest=DIGEST_B,
                scoring_framework_git_commit=SCORING_COMMIT,
            )
        probes = [probe() for _ in range(10)]
        probes[4]["status"] = "UNQUALIFIED"
        with self.assertRaisesRegex(ValueError, "not qualified"):
            aggregate_scoring_baseline_probes(
                probes,
                environment_digest=DIGEST_A,
                evaluator_profile_digest=DIGEST_B,
                scoring_framework_git_commit=SCORING_COMMIT,
            )


if __name__ == "__main__":
    unittest.main()
