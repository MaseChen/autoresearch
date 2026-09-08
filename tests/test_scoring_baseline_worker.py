from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest import mock

from kernel_research.device_timing import DeviceEventMeasurement, DeviceEventRound
from kernel_research.scoring_baseline_worker import run_scoring_baseline_probe
from kernel_research.scoring_measurement import (
    scoring_baseline_measurement_contract_snapshot,
)

SCORING_COMMIT = "d" * 40

def measurement(candidate: float, incumbent: float) -> DeviceEventMeasurement:
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
    )


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
                "kernel_research.scoring_baseline_worker."
                "benchmark_device_event_interleaved",
                side_effect=(
                    measurement(1.0, 1.0),
                    measurement(2.0, 2.0),
                    measurement(1.0, 2.0),
                    measurement(2.0, 4.0),
                ),
            ),
        )

    def test_probe_returns_qualified_bounded_evidence(self) -> None:
        with ExitStack() as stack:
            started = [stack.enter_context(patcher) for patcher in self.patches()]
            result = run_scoring_baseline_probe(
                scoring_framework_git_commit=SCORING_COMMIT,
                torch_module=object(),
                case_specs=(
                    SimpleNamespace(name="case-a"),
                    SimpleNamespace(name="case-b"),
                ),
            )
        self.assertEqual(result["status"], "QUALIFIED")
        self.assertEqual(result["cases"][0]["case_id"], "case-a")
        self.assertEqual(
            result["cases"][0]["anchor_median_ms"], 1.0
        )
        self.assertEqual(
            result["measurement_contract"],
            scoring_baseline_measurement_contract_snapshot(),
        )
        self.assertEqual(result["cases"][0]["compiled_to_eager_ratio"], 0.5)
        self.assertTrue(result["compiler_proof"]["fullgraph_fail_closed"])
        benchmark = started[-1]
        self.assertEqual(benchmark.call_count, 4)
        self.assertEqual(started[1].call_count, 4)
        self.assertEqual(started[7].call_count, 4)
        anchor_calls = benchmark.call_args_list[:2]
        performance_calls = benchmark.call_args_list[2:]
        self.assertTrue(
            all(call.args[1] is call.args[3] for call in anchor_calls)
        )
        self.assertTrue(
            all(call.args[1] is not call.args[3] for call in performance_calls)
        )

    def test_probe_fails_closed_for_each_measurement_proof(self) -> None:
        cases = (
            ("no-cases", lambda mocks: None, ()),
            (
                "doctor",
                lambda mocks: setattr(
                    mocks[0].return_value, "status", "UNAVAILABLE"
                ),
                (SimpleNamespace(name="case-a"),),
            ),
            (
                "compiled-correctness",
                lambda mocks: setattr(mocks[4], "return_value", 0.98),
                (SimpleNamespace(name="case-a"),),
            ),
            (
                "device",
                lambda mocks: setattr(
                    mocks[9].return_value(*()), "device", "cuda:1"
                ),
                (SimpleNamespace(name="case-a"),),
            ),
            (
                "anchor-mutation",
                lambda mocks: setattr(
                    mocks[5], "return_value", (False, "a")
                ),
                (SimpleNamespace(name="case-a"),),
            ),
            (
                "eager-correctness",
                lambda mocks: setattr(
                    mocks[4], "side_effect", (1.0, 1.0, 0.98)
                ),
                (SimpleNamespace(name="case-a"),),
            ),
            (
                "compiled-regression",
                lambda mocks: setattr(
                    mocks[-1],
                    "side_effect",
                    (measurement(1.0, 1.0), measurement(1.02, 1.0)),
                ),
                (SimpleNamespace(name="case-a"),),
            ),
            (
                "performance-mutation",
                lambda mocks: setattr(
                    mocks[5],
                    "side_effect",
                    ((True, None), (False, "b_col_major")),
                ),
                (SimpleNamespace(name="case-a"),),
            ),
        )
        for label, configure, case_specs in cases:
            with self.subTest(label=label), ExitStack() as stack:
                started = [
                    stack.enter_context(patcher) for patcher in self.patches()
                ]
                started[-1].side_effect = (
                    measurement(1.0, 1.0),
                    measurement(1.0, 2.0),
                )
                configure(started)
                result = run_scoring_baseline_probe(
                    scoring_framework_git_commit=SCORING_COMMIT,
                    torch_module=object(),
                    case_specs=case_specs,
                )
                self.assertEqual(result["status"], "UNQUALIFIED")
                self.assertEqual(result["cases"], [])

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
