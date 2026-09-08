from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from kernel_research.autorun.controller import ResearchController
from kernel_research.autorun.models import ControllerConfig
from kernel_research.autorun.runtime import CommandResult
from kernel_research.compiled_reference import (
    SCORING_COMPILER_CONFIG,
    scoring_reference_source_sha256,
)
from kernel_research.device_timing import device_event_protocol_snapshot


DIGEST_A = "sha256:" + "a" * 64


def make_config(root: Path) -> ControllerConfig:
    repository = root / "repo"
    repository.mkdir()
    (repository / "kernel.py").write_text("def run_kernel():\n    pass\n", encoding="utf-8")
    key = root / "key"
    key.write_text("unused", encoding="utf-8")
    return ControllerConfig(
        repository_dir=repository,
        state_dir=root / "state",
        controller_dir=root / "controller",
        checkpoint_dir=root / "checkpoints",
        docker_binary=Path("/usr/bin/docker"),
        proposer_image="registry/proposer@sha256:" + "b" * 64,
        evaluator_image="registry/evaluator@sha256:" + "c" * 64,
        deepseek_key_file=key,
        gpu_devices=(Path("/dev/a"), Path("/dev/b"), Path("/dev/c")),
        evaluator_cache_dir=root / "cache",
        expected_git_commit="d" * 40,
        expected_kernel_hash="e" * 64,
        framework_git_commit="d" * 40,
        acknowledge_gpu_passthrough_risk=True,
    )


def probe(index: int) -> dict:
    offset = index * 0.0001
    cases = []
    for case_id, base in (("decode", 1.0), ("prefill", 10.0)):
        cases.append(
            {
                "case_id": case_id,
                "measurement": {
                    "candidate_median_ms": base + offset,
                },
            }
        )
    environment = {"device": "cuda:0", "version": "test"}
    from kernel_research.platform.canonical import canonical_sha256

    return {
        "schema_version": 1,
        "command": "score-baseline-probe",
        "status": "QUALIFIED",
        "protocol_id": "xpuoj-th0-proxy-v1",
        "reference_source_sha256": scoring_reference_source_sha256(),
        "compiler_config": dict(SCORING_COMPILER_CONFIG),
        "timing_protocol": device_event_protocol_snapshot(),
        "environment": environment,
        "environment_snapshot_digest": canonical_sha256(environment),
        "cases": cases,
    }


class ProbeRunner:
    def __init__(
        self,
        *,
        timeout: bool = False,
        unqualified: bool = False,
        identity_mismatch: bool = False,
    ) -> None:
        self.calls = 0
        self.timeout = timeout
        self.unqualified = unqualified
        self.identity_mismatch = identity_mismatch

    def run(self, argv, **_kwargs):
        index = self.calls
        self.calls += 1
        payload = probe(index)
        returncode = 0
        if self.unqualified:
            payload["status"] = "UNQUALIFIED"
            payload["error"] = "fullgraph compilation failed"
            returncode = 2
        if self.identity_mismatch:
            payload["compiler_config"] = {
                **payload["compiler_config"],
                "fullgraph": False,
            }
        return CommandResult(
            argv=tuple(argv),
            returncode=returncode,
            stdout=json.dumps(payload),
            stderr="",
            timed_out=self.timeout,
        )


class ScoringControllerTests(unittest.TestCase):
    def controller(self, root: Path, runner: ProbeRunner) -> ResearchController:
        controller = ResearchController(make_config(root), runner=runner)
        self.patchers = [
            mock.patch.object(controller, "_verify_repository", return_value={}),
            mock.patch.object(
                controller, "_prepare_framework", return_value=root / "framework"
            ),
            mock.patch.object(
                controller,
                "_inspect_image",
                return_value={"available": True},
            ),
            mock.patch.object(
                controller,
                "_resolved_execution_environment",
                return_value=SimpleNamespace(digest=DIGEST_A),
            ),
            mock.patch.object(ControllerConfig, "validate_host", return_value=[]),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return controller

    def test_ten_probe_operation_is_private_durable_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = ProbeRunner()
            controller = self.controller(root, runner)
            result = controller.qualify_scoring_baseline()
            self.assertEqual(result["status"], "QUALIFIED")
            self.assertFalse(result["idempotent"])
            self.assertEqual(runner.calls, 10)
            operation = Path(controller.config.controller_dir) / "scoring-baseline" / result["operation_id"]
            self.assertEqual(operation.stat().st_mode & 0o777, 0o700)
            self.assertEqual((operation / "state.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(list(operation.glob("probe-*.receipt.json"))), 10)
            repeated = controller.qualify_scoring_baseline()
            self.assertTrue(repeated["idempotent"])
            self.assertEqual(runner.calls, 10)

    def test_idempotent_replay_reproves_receipts_and_private_object(self) -> None:
        for target in ("receipt", "private-object"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                runner = ProbeRunner()
                controller = self.controller(root, runner)
                result = controller.qualify_scoring_baseline()
                operation = (
                    controller.config.controller_dir
                    / "scoring-baseline"
                    / result["operation_id"]
                )
                if target == "receipt":
                    receipt = operation / "probe-00.receipt.json"
                    payload = json.loads(receipt.read_text(encoding="utf-8"))
                    payload["cases"][0]["measurement"]["candidate_median_ms"] = 2.0
                    receipt.write_text(json.dumps(payload), encoding="utf-8")
                    expected = "receipt digest mismatch"
                else:
                    object_digest = result["private_object_id"].removeprefix(
                        "sha256:"
                    )
                    object_path = (
                        controller.config.controller_dir
                        / "objects"
                        / "sha256"
                        / object_digest[:2]
                        / object_digest[2:]
                    )
                    object_path.write_bytes(b"tampered\n")
                    expected = "private qualification object mismatch"
                with self.assertRaisesRegex(Exception, expected):
                    controller.qualify_scoring_baseline()
                self.assertEqual(runner.calls, 10)

    def test_terminal_state_and_operation_directory_identity_are_reproved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = ProbeRunner(unqualified=True)
            controller = self.controller(root, runner)
            result = controller.qualify_scoring_baseline()
            operation = (
                controller.config.controller_dir
                / "scoring-baseline"
                / result["operation_id"]
            )
            state = json.loads((operation / "state.json").read_text(encoding="utf-8"))
            state["reason"] = "tampered"
            (operation / "state.json").write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "state differs"):
                controller.qualify_scoring_baseline()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = ProbeRunner(unqualified=True)
            controller = self.controller(root, runner)
            result = controller.qualify_scoring_baseline()
            scoring_root = controller.config.controller_dir / "scoring-baseline"
            operation = scoring_root / result["operation_id"]
            mismatched = scoring_root / ("score-baseline-" + "f" * 24)
            operation.rename(mismatched)
            with self.assertRaisesRegex(Exception, "directory identity mismatch"):
                controller.qualify_scoring_baseline()

    def test_timeout_leaves_unknown_marker_and_forbids_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = ProbeRunner(timeout=True)
            controller = self.controller(root, runner)
            with self.assertRaisesRegex(Exception, "outcome is unknown"):
                controller.qualify_scoring_baseline()
            state_paths = list(
                (controller.config.controller_dir / "scoring-baseline").glob(
                    "*/state.json"
                )
            )
            self.assertEqual(len(state_paths), 1)
            self.assertEqual(
                json.loads(state_paths[0].read_text(encoding="utf-8"))["status"],
                "UNKNOWN_OUTCOME",
            )
            with self.assertRaisesRegex(Exception, "unresolved GPU outcome"):
                controller.qualify_scoring_baseline()
            self.assertEqual(runner.calls, 1)

    def test_known_compile_failure_is_terminal_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = ProbeRunner(unqualified=True)
            controller = self.controller(root, runner)
            result = controller.qualify_scoring_baseline()
            self.assertEqual(result["status"], "UNQUALIFIED")
            self.assertIn("fullgraph compilation failed", result["reason"])
            self.assertEqual(runner.calls, 1)
            self.assertTrue(controller.qualify_scoring_baseline()["idempotent"])
            self.assertEqual(runner.calls, 1)

    def test_identity_echo_mismatch_is_unknown_and_not_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = ProbeRunner(identity_mismatch=True)
            controller = self.controller(root, runner)
            with self.assertRaisesRegex(Exception, "outcome is unknown"):
                controller.qualify_scoring_baseline()
            self.assertEqual(runner.calls, 1)
            state = next(
                (controller.config.controller_dir / "scoring-baseline").glob(
                    "*/state.json"
                )
            )
            self.assertEqual(
                json.loads(state.read_text(encoding="utf-8"))["status"],
                "UNKNOWN_OUTCOME",
            )


if __name__ == "__main__":
    unittest.main()
