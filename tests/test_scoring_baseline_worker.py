from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from kernel_research.scoring_baseline_worker import run_scoring_baseline_probe

SCORING_COMMIT = "d" * 40

class FakeMeasurement:
    candidate_median_ms = 1.0
    incumbent_median_ms = 2.0

    def to_dict(self):
        return {
            "candidate_median_ms": self.candidate_median_ms,
            "incumbent_median_ms": self.incumbent_median_ms,
        }


class ScoringBaselineWorkerTests(unittest.TestCase):
    def patches(self):
        health = SimpleNamespace(
            status="SUCCESS", environment={"device": "cuda:0", "torch": "test"}
        )
        output = SimpleNamespace(device="cuda:0")
        tensors = {
            "a": SimpleNamespace(device="cuda:0"),
            "b_col_major": object(),
            "scale_a": object(),
            "scale_b": object(),
            "moe_weights": object(),
            "expert_ids": object(),
        }
        return (
            mock.patch(
                "kernel_research.backends.C500Backend.doctor",
                return_value=health,
            ),
            mock.patch(
                "kernel_research.backends._copy_dataset_to_device",
                return_value=tensors,
            ),
            mock.patch(
                "kernel_research.backends._clone_readonly_inputs",
                return_value={"a": object()},
            ),
            mock.patch(
                "kernel_research.backends._torch_reference",
                return_value=output,
            ),
            mock.patch(
                "kernel_research.backends._matched_ratio", return_value=1.0
            ),
            mock.patch(
                "kernel_research.backends._readonly_inputs_unchanged",
                return_value=(True, None),
            ),
            mock.patch("kernel_research.backends._synchronize"),
            mock.patch(
                "kernel_research.cases.generate_case", return_value=object()
            ),
            mock.patch(
                "kernel_research.scoring_baseline_worker.make_scoring_reference",
                return_value=lambda *args: output,
            ),
            mock.patch(
                "kernel_research.scoring_baseline_worker.compile_scoring_reference",
                return_value=lambda *args: output,
            ),
            mock.patch(
                "kernel_research.scoring_baseline_worker.benchmark_device_event_interleaved",
                return_value=FakeMeasurement(),
            ),
        )

    def test_probe_returns_qualified_bounded_evidence(self) -> None:
        stack = self.patches()
        for patcher in stack:
            patcher.start()
            self.addCleanup(patcher.stop)
        result = run_scoring_baseline_probe(
            scoring_framework_git_commit=SCORING_COMMIT,
            torch_module=object(),
            case_specs=(SimpleNamespace(name="case-a"),),
        )
        self.assertEqual(result["status"], "QUALIFIED")
        self.assertEqual(result["cases"][0]["case_id"], "case-a")
        self.assertEqual(
            result["cases"][0]["measurement"]["candidate_median_ms"], 1.0
        )
        self.assertTrue(result["compiler_proof"]["fullgraph_fail_closed"])

    def test_probe_failure_is_unqualified_not_a_score(self) -> None:
        with mock.patch(
            "kernel_research.backends.C500Backend.doctor",
            side_effect=RuntimeError("device unavailable"),
        ):
            result = run_scoring_baseline_probe(
                scoring_framework_git_commit=SCORING_COMMIT,
                torch_module=object(),
                case_specs=(SimpleNamespace(name="case-a"),),
            )
        self.assertEqual(result["status"], "UNQUALIFIED")
        self.assertIn("device unavailable", result["error"])
        self.assertEqual(result["cases"], [])
        self.assertNotIn("objective_score", result)

    def test_scoring_framework_commit_is_required_and_echoed(self) -> None:
        with self.assertRaisesRegex(ValueError, "scoring_framework_git_commit"):
            run_scoring_baseline_probe(
                scoring_framework_git_commit="invalid",
                torch_module=object(),
                case_specs=(SimpleNamespace(name="case-a"),),
            )


if __name__ == "__main__":
    unittest.main()
