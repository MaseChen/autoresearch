from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from kernel_research.backends import (
    C500Backend,
    MEASUREMENT_ROUNDS,
    MockBackend,
    SAMPLES_PER_ROUND,
    STATUS_COMPILE_ERROR,
    STATUS_CRASH,
    STATUS_MOCK_VALIDATED,
    STATUS_UNSUPPORTED_ENV,
    WARMUP_ITERATIONS,
    _benchmark_interleaved,
    _classify_first_invocation_error,
    _parse_driver_version,
    _parse_maca_version,
    _mx_smi_fingerprint,
)
from kernel_research.contract import KERNEL_PARAMETERS, validate_candidate
from kernel_research.constants import (
    CURRENT_C500_EVALUATION_PROTOCOL_ID,
    LEGACY_C500_EVALUATION_PROTOCOL_ID,
)


SIDE_EFFECT_CANDIDATE = """\
from pathlib import Path
Path(__file__).with_suffix('.imported').write_text('candidate was imported')

def run_kernel(a, b_col_major, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk, out):
    raise AssertionError('mock must not execute candidates')
"""


class MockBackendTests(unittest.TestCase):
    def test_mock_is_deterministic_static_only_and_has_no_performance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "kernel.py"
            candidate.write_text(SIDE_EFFECT_CANDIDATE, encoding="utf-8")
            marker = candidate.with_suffix(".imported")

            first = MockBackend().evaluate(candidate, suite="smoke")
            second = MockBackend().evaluate(candidate, suite="smoke")

            self.assertEqual(first, second)
            self.assertEqual(first.status, STATUS_MOCK_VALIDATED)
            self.assertFalse(first.eligible_for_promotion)
            self.assertIsNone(first.aggregate_score)
            self.assertEqual(first.cases, ())
            self.assertFalse(marker.exists(), "mock imported candidate source")

            payload = first.to_dict()
            self.assertFalse(payload["environment"]["candidate_imported"])
            self.assertFalse(payload["environment"]["candidate_executed"])
            self.assertFalse(payload["environment"]["performance_measurement"])
            self.assertEqual(
                payload["environment"]["evaluation_protocol_id"],
                CURRENT_C500_EVALUATION_PROTOCOL_ID,
            )
            self.assertEqual(payload["benchmark_config"]["warmup_iterations"], 0)
            self.assertEqual(payload["benchmark_config"]["measurement_rounds"], 0)
            self.assertNotIn("latency", json.dumps(payload).lower())
            json.dumps(payload)

    def test_mock_accepts_only_registered_protocol_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "kernel.py"
            candidate.write_text(SIDE_EFFECT_CANDIDATE, encoding="utf-8")
            legacy = MockBackend().evaluate(
                candidate,
                suite="quick",
                evaluation_protocol_id=LEGACY_C500_EVALUATION_PROTOCOL_ID,
            )
            self.assertEqual(
                legacy.environment["evaluation_protocol_id"],
                LEGACY_C500_EVALUATION_PROTOCOL_ID,
            )
            with self.assertRaisesRegex(
                ValueError, "unsupported evaluation protocol"
            ):
                MockBackend().evaluate(
                    candidate,
                    evaluation_protocol_id="unregistered-protocol",
                )

    def test_contract_error_also_does_not_import_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "invalid.py"
            candidate.write_text(
                "raise AssertionError('not imported')\n"
                "def run_kernel(wrong):\n    pass\n",
                encoding="utf-8",
            )
            result = MockBackend().evaluate(candidate)

        self.assertEqual(result.status, "CONTRACT_ERROR")
        self.assertFalse(result.eligible_for_promotion)
        self.assertIn("positional parameters", result.error or "")

    def test_kernel_seed_has_public_contract_without_importing_dependencies(self) -> None:
        candidate = Path(__file__).resolve().parents[1] / "kernel.py"
        validation = validate_candidate(candidate)
        self.assertTrue(validation.valid, validation.errors)

        tree = ast.parse(candidate.read_text(encoding="utf-8"))
        run_kernel = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_kernel"
        )
        self.assertEqual(
            tuple(argument.arg for argument in run_kernel.args.args),
            KERNEL_PARAMETERS,
        )


class C500BackendTests(unittest.TestCase):
    def test_protocol_reaches_suite_lookup_and_unknown_fails_before_doctor(
        self,
    ) -> None:
        health = SimpleNamespace(
            status="SUCCESS",
            error=None,
            environment={"device": "cuda:0"},
            benchmark_config={},
        )
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "kernel.py"
            candidate.write_text(SIDE_EFFECT_CANDIDATE, encoding="utf-8")
            backend = C500Backend()
            with (
                mock.patch.object(backend, "doctor", return_value=health),
                mock.patch(
                    "kernel_research.backends._load_c500_runtime",
                    return_value=((object(), object()), None),
                ),
                mock.patch(
                    "kernel_research.cases.get_suite",
                    side_effect=RuntimeError("suite lookup sentinel"),
                ) as get_suite,
            ):
                result = backend.evaluate(
                    candidate,
                    suite="quick",
                    evaluation_protocol_id=LEGACY_C500_EVALUATION_PROTOCOL_ID,
                )

            self.assertEqual(result.status, STATUS_CRASH)
            get_suite.assert_called_once_with(
                "quick",
                evaluation_protocol_id=LEGACY_C500_EVALUATION_PROTOCOL_ID,
            )

            with (
                mock.patch.object(backend, "doctor") as doctor,
                self.assertRaisesRegex(
                    ValueError, "unsupported evaluation protocol"
                ),
            ):
                backend.evaluate(
                    candidate,
                    evaluation_protocol_id="unregistered-protocol",
                )
            doctor.assert_not_called()

    def test_doctor_closes_cleanly_when_torch_is_missing(self) -> None:
        real_import = importlib.import_module

        def missing_torch(name: str, package: str | None = None):
            if name == "torch":
                raise ModuleNotFoundError("test fixture: no torch")
            return real_import(name, package)

        with mock.patch(
            "kernel_research.backends.importlib.import_module",
            side_effect=missing_torch,
        ):
            result = C500Backend().doctor()

        self.assertEqual(result.status, STATUS_UNSUPPORTED_ENV)
        self.assertFalse(result.eligible_for_promotion)
        self.assertEqual(result.environment["missing_dependency"], "torch")
        self.assertIn("sdk_version", result.environment)
        self.assertIn("driver_version", result.environment)
        self.assertIn("no torch", result.error or "")
        json.dumps(result.to_dict())

    def test_real_benchmark_configuration_is_warm_cache_and_raw(self) -> None:
        real_import = importlib.import_module

        def missing_torch(name: str, package: str | None = None):
            if name == "torch":
                raise ModuleNotFoundError("test fixture")
            return real_import(name, package)

        with mock.patch(
            "kernel_research.backends.importlib.import_module",
            side_effect=missing_torch,
        ):
            config = C500Backend().doctor().benchmark_config

        self.assertEqual(config["warmup_iterations"], WARMUP_ITERATIONS)
        self.assertEqual(config["measurement_rounds"], MEASUREMENT_ROUNDS)
        self.assertEqual(config["samples_per_round"], SAMPLES_PER_ROUND)
        self.assertEqual(
            config["total_samples_per_case"],
            MEASUREMENT_ROUNDS * SAMPLES_PER_ROUND,
        )
        self.assertFalse(config["l2_flush"])
        self.assertIn("warm-cache", config["cache_policy"])

    def test_doctor_rejects_a_non_c500_accelerator(self) -> None:
        class FakeAccelerator:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 1

            @staticmethod
            def get_device_name(index: int) -> str:
                if index != 0:
                    raise IndexError(index)
                return "NVIDIA H100"

        fake_torch = mock.Mock(__version__="test", cuda=FakeAccelerator())
        fake_triton = mock.Mock(__version__="test")

        with mock.patch(
            "kernel_research.backends._load_c500_runtime",
            return_value=((fake_torch, fake_triton), None),
        ):
            result = C500Backend().doctor()

        self.assertEqual(result.status, STATUS_UNSUPPORTED_ENV)
        self.assertIn("not a MetaX C500", result.error or "")

    def test_doctor_requires_the_isolated_compile_probe_to_pass(self) -> None:
        class FakeAccelerator:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 1

            @staticmethod
            def get_device_name(index: int) -> str:
                return "MetaX C500"

        fake_torch = SimpleNamespace(
            __version__="vendor-test",
            __file__="/vendor/torch/__init__.py",
            version=SimpleNamespace(maca="3.7.1.3"),
            cuda=FakeAccelerator(),
        )
        fake_triton = SimpleNamespace(
            __version__="vendor-test",
            __file__="/vendor/triton/__init__.py",
        )
        seen: list[str] = []
        probe_module = ModuleType("kernel_research.c500_probe")
        probe_module.run_probe = seen.append  # type: ignore[attr-defined]

        with (
            mock.patch(
                "kernel_research.backends._load_c500_runtime",
                return_value=((fake_torch, fake_triton), None),
            ),
            mock.patch(
                "kernel_research.backends._mx_smi_fingerprint",
                return_value={
                    "mx_smi_version": "fixture",
                    "mx_smi_driver_version": "3.8.30",
                    "mx_smi_maca_version": "3.7.1.5",
                    "mx_smi_probe_error": None,
                },
            ),
            mock.patch.dict(
                sys.modules, {"kernel_research.c500_probe": probe_module}
            ),
        ):
            not_run = C500Backend().doctor()
            passed = C500Backend().doctor(compile_probe=True)

            def fail_probe(device: str) -> None:
                raise RuntimeError(f"probe failed on {device}")

            probe_module.run_probe = fail_probe  # type: ignore[attr-defined]
            failed = C500Backend().doctor(compile_probe=True)

        self.assertEqual(passed.status, "SUCCESS")
        self.assertEqual(seen, ["cuda:0"])
        self.assertIsNone(not_run.environment["compile_probe_passed"])
        self.assertEqual(not_run.environment["compile_probe_status"], "NOT_RUN")
        self.assertTrue(passed.environment["compile_probe_passed"])
        self.assertEqual(passed.environment["compile_probe_status"], "PASSED")
        self.assertEqual(passed.environment["mctriton_version"], "vendor-test")
        self.assertEqual(passed.environment["driver_version"], "3.8.30")
        self.assertEqual(passed.environment["system_maca_version"], "3.7.1.5")
        self.assertFalse(passed.environment["sdk_system_maca_version_match"])
        self.assertEqual(failed.status, STATUS_UNSUPPORTED_ENV)
        self.assertFalse(failed.environment["compile_probe_passed"])
        self.assertEqual(failed.environment["compile_probe_status"], "FAILED")
        self.assertIn("probe failed on cuda:0", failed.error or "")
        for field in (
            "sdk_version",
            "sdk_version_source",
            "sdk_probe_error",
            "system_maca_version",
            "system_maca_version_source",
            "system_maca_probe_error",
            "driver_version",
            "driver_version_source",
            "driver_probe_error",
        ):
            self.assertIn(field, passed.environment)

    def test_driver_version_parser_uses_an_explicit_driver_field(self) -> None:
        self.assertEqual(
            _parse_driver_version("Device 0\nDriver Version : 3.4.5\n"),
            "3.4.5",
        )
        self.assertIsNone(_parse_driver_version("mx-smi version 9.9"))

    def test_real_mx_smi_status_text_exposes_driver_and_maca_versions(self) -> None:
        status_text = """\
mx-smi  version: 2.3.1

=================== MetaX System Management Interface Log ===================
| MX-SMI 2.3.1                       Kernel Mode Driver Version: 3.8.30           |
| MACA Version: 3.7.1.5              BIOS Version: 1.33.5.0                       |
"""
        self.assertEqual(_parse_driver_version(status_text), "3.8.30")
        self.assertEqual(_parse_maca_version(status_text), "3.7.1.5")

    def test_mx_smi_fingerprint_uses_the_supported_no_argument_status_call(
        self,
    ) -> None:
        calls: list[list[str]] = []
        status_text = (
            "| MX-SMI 2.3.1 Kernel Mode Driver Version: 3.8.30 |\n"
            "| MACA Version: 3.7.1.5 BIOS Version: 1.33.5.0 |"
        )

        def fake_run(command: list[str], **kwargs):
            calls.append(command)
            stdout = "mx-smi  version: 2.3.1" if len(command) == 2 else status_text
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        with (
            mock.patch(
                "kernel_research.backends.shutil.which",
                return_value="/usr/bin/mx-smi",
            ),
            mock.patch(
                "kernel_research.backends.subprocess.run",
                side_effect=fake_run,
            ),
        ):
            fingerprint = _mx_smi_fingerprint()

        self.assertEqual(
            calls,
            [["/usr/bin/mx-smi", "--version"], ["/usr/bin/mx-smi"]],
        )
        self.assertEqual(fingerprint["mx_smi_driver_version"], "3.8.30")
        self.assertEqual(fingerprint["mx_smi_maca_version"], "3.7.1.5")
        self.assertEqual(fingerprint["mx_smi_status_exit_code"], 0)
        self.assertIsNone(fingerprint["mx_smi_probe_error"])

    def test_first_invocation_error_classification(self) -> None:
        compilation_error = type("CompilationError", (RuntimeError,), {})
        accelerator_error = type("AcceleratorError", (RuntimeError,), {})

        self.assertEqual(
            _classify_first_invocation_error(
                compilation_error("mcTriton compilation failed")
            ),
            STATUS_COMPILE_ERROR,
        )
        self.assertEqual(
            _classify_first_invocation_error(
                AssertionError(
                    "num_warps must be a power of 2 or greater than 0 "
                    "and less than or equal to 16"
                )
            ),
            STATUS_COMPILE_ERROR,
        )
        self.assertEqual(
            _classify_first_invocation_error(
                accelerator_error("device launch failed")
            ),
            STATUS_CRASH,
        )
        self.assertEqual(
            _classify_first_invocation_error(
                RuntimeError(
                    "CUDA error: an illegal memory access was encountered; "
                    "Compile with TORCH_USE_CUDA_DSA"
                )
            ),
            STATUS_CRASH,
        )
        self.assertEqual(
            _classify_first_invocation_error(
                RuntimeError("Xnack exception: ATU address translation error")
            ),
            STATUS_CRASH,
        )
        self.assertEqual(
            _classify_first_invocation_error(ValueError("unexpected launch failure")),
            STATUS_CRASH,
        )

    def test_interleaved_measurement_keeps_equal_sample_counts(self) -> None:
        calls = {"candidate": 0, "baseline": 0}

        def candidate(*arguments) -> None:
            calls["candidate"] += len(arguments)

        def baseline(*arguments) -> None:
            calls["baseline"] += len(arguments)

        candidate_samples, baseline_samples = _benchmark_interleaved(
            object(), candidate, (1,), baseline, (1,)
        )
        expected_calls = WARMUP_ITERATIONS + MEASUREMENT_ROUNDS * SAMPLES_PER_ROUND
        self.assertEqual(calls, {"candidate": expected_calls, "baseline": expected_calls})
        self.assertEqual(len(candidate_samples), 30)
        self.assertEqual(len(baseline_samples), 30)
        self.assertTrue(all(sample > 0 for sample in candidate_samples))
        self.assertTrue(all(sample > 0 for sample in baseline_samples))


if __name__ == "__main__":
    unittest.main()
