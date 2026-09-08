from __future__ import annotations

import math
import unittest

from kernel_research.objective_scoring import (
    ObjectiveScoreReport,
    ScoreNoiseCalibration,
    ScoringBaselineDescriptor,
    anchored_case_score,
    evaluate_objective_promotion,
)


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64


class AnchoredScoreTests(unittest.TestCase):
    def test_baseline_parity_is_fifty_and_anchor_is_one_hundred(self) -> None:
        self.assertEqual(anchored_case_score(10.0, 10.0, 0.0), 50.0)
        self.assertEqual(anchored_case_score(2.0, 10.0, 2.0), 100.0)
        self.assertGreater(anchored_case_score(1.0, 10.0, 2.0), 100.0)

    def test_th0_score_is_continuous_and_not_rounded(self) -> None:
        actual = anchored_case_score(9.5, 10.0)
        self.assertAlmostEqual(actual, 100.0 * 10.0 / 19.5)
        self.assertNotEqual(actual, round(actual))

    def test_invalid_latencies_and_anchor_fail_closed(self) -> None:
        for candidate in (0, -1, math.inf, math.nan, "bad"):
            with self.subTest(candidate=candidate):
                with self.assertRaises((TypeError, ValueError)):
                    anchored_case_score(candidate, 10.0)
        with self.assertRaisesRegex(ValueError, "must exceed"):
            anchored_case_score(1.0, 2.0, 2.0)

    def test_report_uses_equal_weight_arithmetic_mean(self) -> None:
        report = ObjectiveScoreReport.calculate(
            {"decode": 10.0, "prefill": 5.0},
            {"decode": 10.0, "prefill": 10.0},
        )
        self.assertEqual([case.case_id for case in report.case_scores], ["decode", "prefill"])
        self.assertAlmostEqual(report.objective_score, (50.0 + 200.0 / 3.0) / 2.0)
        self.assertEqual(report.status, "AVAILABLE")
        self.assertEqual(report.mode, "PROXY_CURRENT_MACA")
        self.assertEqual(
            ObjectiveScoreReport.from_value(report.to_dict()), report
        )

    def test_report_requires_identical_case_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical"):
            ObjectiveScoreReport.calculate({"a": 1.0}, {"b": 1.0})
        with self.assertRaisesRegex(TypeError, "strings"):
            ObjectiveScoreReport.calculate({1: 1.0}, {1: 1.0})  # type: ignore[dict-item]
        report = ObjectiveScoreReport.calculate({"a": 1.0}, {"a": 1.0}).to_dict()
        report["objective_score"] = 51.0
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            ObjectiveScoreReport.from_value(report)


class ScoringIdentityTests(unittest.TestCase):
    def test_baseline_digest_is_stable_and_canonical(self) -> None:
        first = ScoringBaselineDescriptor(
            environment_digest=DIGEST_A,
            evaluator_profile_digest=DIGEST_B,
            reference_source_sha256=DIGEST_C,
            compiler_backend="inductor",
            compiler_config={"dynamic": False, "fullgraph": True, "mode": "default"},
            case_baseline_ms={"prefill": 10.0, "decode": 1.0},
        )
        second = ScoringBaselineDescriptor(
            environment_digest=DIGEST_A,
            evaluator_profile_digest=DIGEST_B,
            reference_source_sha256=DIGEST_C,
            compiler_backend="inductor",
            compiler_config={"mode": "default", "fullgraph": True, "dynamic": False},
            case_baseline_ms={"decode": 1.0, "prefill": 10.0},
        )
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.to_dict()["digest"], first.digest)
        self.assertEqual(
            ScoringBaselineDescriptor.from_value(first.to_dict()).digest,
            first.digest,
        )
        with self.assertRaises(TypeError):
            first.case_baseline_ms["decode"] = 2.0  # type: ignore[index]

    def test_baseline_rejects_untrusted_compiler_and_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "environment_digest"):
            ScoringBaselineDescriptor(
                environment_digest="bad",
                evaluator_profile_digest=DIGEST_B,
                reference_source_sha256=DIGEST_C,
                compiler_backend="inductor",
                compiler_config={},
                case_baseline_ms={"case": 1.0},
            )
        with self.assertRaisesRegex(ValueError, "inductor"):
            ScoringBaselineDescriptor(
                environment_digest=DIGEST_A,
                evaluator_profile_digest=DIGEST_B,
                reference_source_sha256=DIGEST_C,
                compiler_backend="eager",
                compiler_config={},
                case_baseline_ms={"case": 1.0},
            )


class NoiseAndPromotionTests(unittest.TestCase):
    def calibration(self, value: float = 0.10) -> ScoreNoiseCalibration:
        return ScoreNoiseCalibration(
            null_deltas=tuple(value for _ in range(20)),
            scoring_baseline_digest=DIGEST_A,
        )

    def test_noise_threshold_has_visible_floor_and_requires_independent_runs(self) -> None:
        calibration = self.calibration()
        self.assertEqual(calibration.score_threshold, 0.25)
        self.assertTrue(calibration.qualified)
        self.assertEqual(
            ScoreNoiseCalibration.from_value(calibration.to_dict()), calibration
        )
        with self.assertRaisesRegex(ValueError, "20"):
            ScoreNoiseCalibration(
                null_deltas=(0.0,) * 19,
                scoring_baseline_digest=DIGEST_A,
            )
        tampered = calibration.to_dict()
        tampered["score_threshold"] = 0.30
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ScoreNoiseCalibration.from_value(tampered)

    def test_noisy_calibration_cannot_authorize_promotion(self) -> None:
        calibration = self.calibration(0.60)
        self.assertFalse(calibration.qualified)
        with self.assertRaisesRegex(ValueError, "not qualified"):
            evaluate_objective_promotion(
                primary_candidate_score=51.0,
                primary_incumbent_score=50.0,
                primary_incumbent_latencies_ms={"a": 10.0},
                primary_candidate_latencies_ms={"a": 9.0},
                calibration=calibration,
            )

    def test_primary_requests_confirmation_and_complete_confirmation_promotes(self) -> None:
        arguments = {
            "primary_candidate_score": 50.5,
            "primary_incumbent_score": 50.0,
            "primary_incumbent_latencies_ms": {"a": 10.0, "b": 10.0},
            "primary_candidate_latencies_ms": {"a": 9.8, "b": 9.8},
            "calibration": self.calibration(),
        }
        primary = evaluate_objective_promotion(**arguments)
        self.assertTrue(primary.needs_confirmation)
        self.assertFalse(primary.promoted)
        confirmed = evaluate_objective_promotion(
            **arguments,
            confirmation_candidate_score=50.4,
            confirmation_incumbent_score=50.0,
            confirmation_incumbent_latencies_ms={"a": 10.0, "b": 10.0},
            confirmation_candidate_latencies_ms={"a": 9.8, "b": 9.8},
        )
        self.assertTrue(confirmed.promoted)
        self.assertEqual(confirmed.reason, "promoted")

    def test_score_gain_cannot_override_case_or_aggregate_regression(self) -> None:
        case_regression = evaluate_objective_promotion(
            primary_candidate_score=51.0,
            primary_incumbent_score=50.0,
            primary_incumbent_latencies_ms={"a": 10.0, "b": 10.0},
            primary_candidate_latencies_ms={"a": 9.0, "b": 10.4},
            calibration=self.calibration(),
        )
        self.assertEqual(case_regression.reason, "per_case_regression_exceeded")
        aggregate_regression = evaluate_objective_promotion(
            primary_candidate_score=51.0,
            primary_incumbent_score=50.0,
            primary_incumbent_latencies_ms={"a": 10.0, "b": 10.0},
            primary_candidate_latencies_ms={"a": 9.9, "b": 10.2},
            calibration=self.calibration(),
            max_case_regression=0.03,
        )
        self.assertEqual(aggregate_regression.reason, "paired_aggregate_regression")

    def test_partial_confirmation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete set"):
            evaluate_objective_promotion(
                primary_candidate_score=50.5,
                primary_incumbent_score=50.0,
                primary_incumbent_latencies_ms={"a": 10.0},
                primary_candidate_latencies_ms={"a": 9.8},
                calibration=self.calibration(),
                confirmation_candidate_score=50.5,
            )


if __name__ == "__main__":
    unittest.main()
