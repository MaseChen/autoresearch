from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from kernel_research.device_timing import DeviceEventMeasurement, DeviceEventRound
from kernel_research.scoring_candidate_measurement import (
    qualified_scoring_probe_environment_snapshot,
    scoring_candidate_measurement_contract_snapshot,
)
from kernel_research.scoring_candidate_worker import run_scoring_candidate_probe


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


class ScoringCandidateWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.candidate = root / "candidate.py"
        self.incumbent = root / "incumbent.py"
        self.candidate.write_text("def run_kernel(*args):\n    return None\n", encoding="utf-8")
        self.incumbent.write_text("def run_kernel(*args):\n    return None\n", encoding="utf-8")
        from kernel_research.contract import validate_candidate

        self.candidate_hash = validate_candidate(self.candidate).sha256
        self.incumbent_hash = validate_candidate(self.incumbent).sha256

    def patches(self):
        health = SimpleNamespace(
            status="SUCCESS",
            environment=qualified_scoring_probe_environment_snapshot(),
        )
        tensors = {
            "arguments": (object(), object()),
            "out": object(),
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
                return_value=object(),
            ),
            mock.patch(
                "kernel_research.backends._matched_ratio",
                return_value=1.0,
            ),
            mock.patch(
                "kernel_research.backends._fresh_output_check",
                return_value=1.0,
            ),
            mock.patch(
                "kernel_research.backends._readonly_inputs_unchanged",
                return_value=(True, None),
            ),
            mock.patch("kernel_research.backends._synchronize"),
            mock.patch("kernel_research.cases.generate_case", return_value=object()),
            mock.patch(
                "kernel_research.scoring_candidate_worker._run_kernel",
                side_effect=(lambda *args: None, lambda *args: None),
            ),
            mock.patch(
                "kernel_research.scoring_candidate_worker."
                "benchmark_device_event_interleaved",
                side_effect=(
                    measurement(1.0, 1.0),
                    measurement(2.0, 2.0),
                    measurement(0.9, 1.0),
                    measurement(1.8, 2.0),
                ),
            ),
            mock.patch("kernel_research.scoring_candidate_worker._release_case"),
        )

    def run_probe(self, **overrides):
        values = {
            "candidate_path": self.candidate,
            "incumbent_path": self.incumbent,
            "expected_candidate_hash": self.candidate_hash,
            "expected_incumbent_hash": self.incumbent_hash,
            "scoring_framework_git_commit": "c" * 40,
            "scoring_profile_digest": "sha256:" + "d" * 64,
            "expected_measurement_contract_digest": (
                scoring_candidate_measurement_contract_snapshot()["digest"]
            ),
            "torch_module": SimpleNamespace(empty_like=lambda _value: object()),
            "case_specs": (
                SimpleNamespace(name="case-a"),
                SimpleNamespace(name="case-b"),
            ),
        }
        values.update(overrides)
        return run_scoring_candidate_probe(**values)

    def test_probe_uses_two_memory_bounded_phases_and_exact_echo(self) -> None:
        with ExitStack() as stack:
            started = [stack.enter_context(patch) for patch in self.patches()]
            result = self.run_probe()

        self.assertEqual(result["status"], "QUALIFIED")
        self.assertEqual(result["gpu_state"], "COMPLETED")
        self.assertTrue(result["completion_trusted"])
        self.assertEqual(result["phase"], "complete")
        self.assertEqual(result["candidate_hash"], self.candidate_hash)
        self.assertEqual(result["incumbent_hash"], self.incumbent_hash)
        self.assertEqual(len(result["cases"]), 2)
        self.assertEqual(result["cases"][0]["candidate_anchor_median_ms"], 1.0)
        self.assertEqual(result["cases"][0]["paired_candidate_median_ms"], 0.9)
        self.assertEqual(result["cases"][0]["paired_incumbent_median_ms"], 1.0)
        self.assertEqual(started[1].call_count, 4)
        self.assertEqual(started[-1].call_count, 4)
        self.assertEqual(started[-2].call_count, 4)
        anchor_calls = started[-2].call_args_list[:2]
        paired_calls = started[-2].call_args_list[2:]
        self.assertTrue(all(call.args[1] is call.args[3] for call in anchor_calls))
        self.assertTrue(all(call.args[1] is not call.args[3] for call in paired_calls))

    def test_hash_or_contract_drift_fails_before_gpu(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate source hash"):
            self.run_probe(expected_candidate_hash="f" * 64)
        with self.assertRaisesRegex(ValueError, "contract digest"):
            self.run_probe(
                expected_measurement_contract_digest="sha256:" + "f" * 64
            )

    def test_runtime_failures_are_unqualified_without_score(self) -> None:
        with mock.patch(
            "kernel_research.backends.C500Backend.doctor",
            side_effect=RuntimeError("device unavailable"),
        ):
            result = self.run_probe()
        self.assertEqual(result["status"], "UNQUALIFIED")
        self.assertEqual(result["gpu_state"], "NOT_STARTED")
        self.assertTrue(result["completion_trusted"])
        self.assertEqual(result["cases"], [])
        self.assertIn("device unavailable", result["error"])
        self.assertNotIn("objective_score", result)

    def test_correctness_failure_prevents_any_trusted_measurement(self) -> None:
        with ExitStack() as stack:
            started = [stack.enter_context(patch) for patch in self.patches()]
            started[4].return_value = 0.5
            result = self.run_probe()
        self.assertEqual(result["status"], "UNQUALIFIED")
        self.assertEqual(result["cases"], [])
        self.assertEqual(started[-2].call_count, 0)

    def test_each_runtime_proof_failure_is_unqualified(self) -> None:
        variants = (
            (
                "doctor",
                lambda mocks: setattr(mocks[0].return_value, "status", "FAILED"),
            ),
            (
                "fresh-output",
                lambda mocks: setattr(mocks[5], "return_value", 0.5),
            ),
            (
                "input-mutation",
                lambda mocks: setattr(mocks[6], "return_value", (False, "a")),
            ),
            (
                "paired-correctness",
                lambda mocks: setattr(
                    mocks[4], "side_effect", (1.0, 1.0, 0.5, 1.0)
                ),
            ),
        )
        for label, configure in variants:
            with self.subTest(label=label), ExitStack() as stack:
                started = [stack.enter_context(patch) for patch in self.patches()]
                configure(started)
                result = self.run_probe()
                self.assertEqual(result["status"], "UNQUALIFIED")
                self.assertEqual(result["cases"], [])

    def test_ratio_bool_nan_and_above_one_fail_closed(self) -> None:
        for value in (True, float("nan"), 1.0001):
            with self.subTest(value=value), ExitStack() as stack:
                started = [stack.enter_context(patch) for patch in self.patches()]
                started[4].return_value = value
                result = self.run_probe()
                self.assertEqual(result["status"], "UNQUALIFIED")
                self.assertTrue(result["completion_trusted"])

    def test_exception_during_gpu_call_is_unknown(self) -> None:
        with ExitStack() as stack:
            started = [stack.enter_context(patch) for patch in self.patches()]
            started[1].side_effect = RuntimeError("copy interrupted")
            result = self.run_probe()
        self.assertEqual(result["status"], "UNKNOWN_OUTCOME")
        self.assertEqual(result["gpu_state"], "STARTED")
        self.assertFalse(result["completion_trusted"])
        self.assertIn("anchor:case-a:copy", result["phase"])

    def test_environment_drift_is_known_before_gpu(self) -> None:
        with ExitStack() as stack:
            started = [stack.enter_context(patch) for patch in self.patches()]
            started[0].return_value.environment = {"device": "cuda:0", "drift": True}
            result = self.run_probe()
        self.assertEqual(result["status"], "UNQUALIFIED")
        self.assertEqual(result["gpu_state"], "NOT_STARTED")
        self.assertTrue(result["completion_trusted"])

    def test_empty_case_set_and_invalid_identities_are_rejected(self) -> None:
        with ExitStack() as stack:
            for patch in self.patches():
                stack.enter_context(patch)
            result = self.run_probe(case_specs=())
        self.assertEqual(result["status"], "UNQUALIFIED")
        for field, value in (
            ("expected_candidate_hash", "bad"),
            ("expected_incumbent_hash", "bad"),
            ("scoring_framework_git_commit", "bad"),
            ("scoring_profile_digest", "bad"),
            ("expected_measurement_contract_digest", "bad"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_probe(**{field: value})


if __name__ == "__main__":
    unittest.main()
