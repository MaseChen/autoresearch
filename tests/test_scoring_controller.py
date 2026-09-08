from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from kernel_research.autorun.controller import (
    SCORING_ABANDONED_UNKNOWN_STATUS,
    SCORING_OOM_ERROR,
    SCORING_OOM_MEASUREMENT_DIGEST,
    SCORING_OOM_OPERATION_DIGEST,
    SCORING_OOM_OPERATION_ID,
    SCORING_OOM_SOURCE_COMMIT,
    ResearchController,
)
from kernel_research.autorun.models import ControllerConfig
from kernel_research.autorun.runtime import CommandResult
from kernel_research.compiled_reference import (
    SCORING_COMPILER_CONFIG,
    scoring_reference_source_sha256,
)
from kernel_research.device_timing import (
    DeviceEventMeasurement,
    DeviceEventRound,
    device_event_protocol_snapshot,
)
from kernel_research.scoring_measurement import (
    scoring_baseline_measurement_contract_snapshot,
)
from kernel_research.platform.canonical import canonical_sha256


DIGEST_A = "sha256:" + "a" * 64
PRODUCTION_EVALUATOR = (
    "registry.cn-shanghai.aliyuncs.com/kcr-3rd/kesci_kernel_lab@sha256:"
    "5f1da890360acc5a81438d0e35a80079b34f30d1fa64fc954fef2e9a1ae45b64"
)
PRODUCTION_FRAMEWORK_COMMIT = "58c8b7e36fbff4bb08c7636b391f72b33d0ef0da"


def rewrite_unknown_as_legacy_operation(operation: Path) -> Path:
    intent_path = operation / "intent.json"
    state_path = operation / "state.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    intent.pop("scoring_framework_git_commit")
    state.pop("scoring_framework_git_commit")
    intent.pop("measurement_contract")
    state.pop("measurement_contract")
    material = {
        key: value
        for key, value in intent.items()
        if key
        not in {
            "operation_id",
            "operation_digest",
            "status",
            "active_probe_index",
            "completed_probes",
        }
    }
    digest = canonical_sha256(material)
    operation_id = "score-baseline-" + digest.removeprefix("sha256:")[:24]
    for value in (intent, state):
        value["operation_id"] = operation_id
        value["operation_digest"] = digest
    rewritten = operation.parent / operation_id
    operation.rename(rewritten)
    (rewritten / "intent.json").write_text(
        json.dumps(intent, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (rewritten / "state.json").write_text(
        json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return rewritten


def make_config(
    root: Path,
    *,
    expected_git_commit: str = "d" * 40,
    framework_git_commit: str | None = None,
    evaluator_image: str = "registry/evaluator@sha256:" + "c" * 64,
) -> ControllerConfig:
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
        evaluator_image=evaluator_image,
        deepseek_key_file=key,
        gpu_devices=(Path("/dev/a"), Path("/dev/b"), Path("/dev/c")),
        evaluator_cache_dir=root / "cache",
        expected_git_commit=expected_git_commit,
        expected_kernel_hash="e" * 64,
        framework_git_commit=framework_git_commit or expected_git_commit,
        acknowledge_gpu_passthrough_risk=True,
    )


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


def probe(index: int) -> dict:
    offset = index * 0.0001
    cases = []
    for case_id, base in (("decode", 1.0), ("prefill", 10.0)):
        cases.append(
            {
                "case_id": case_id,
                "matched_ratio": 1.0,
                "eager_matched_ratio": 1.0,
                "compiler_first_invocation_seconds": 1.0,
                "compiled_to_eager_ratio": 0.5,
                "anchor_median_ms": base + offset,
                "anchor_measurement": measurement(
                    base + offset, base + offset
                ),
                "performance_measurement": measurement(1.0, 2.0),
            }
        )
    environment = {"device": "cuda:0", "version": "test"}
    from kernel_research.platform.canonical import canonical_sha256

    return {
        "schema_version": 1,
        "command": "score-baseline-probe",
        "status": "QUALIFIED",
        "protocol_id": "xpuoj-th0-proxy-v1",
        "scoring_framework_git_commit": "d" * 40,
        "reference_source_sha256": scoring_reference_source_sha256(),
        "compiler_config": dict(SCORING_COMPILER_CONFIG),
        "measurement_contract": scoring_baseline_measurement_contract_snapshot(),
        "timing_protocol": device_event_protocol_snapshot(),
        "environment": environment,
        "environment_snapshot_digest": canonical_sha256(environment),
        "cases": cases,
    }


def write_oom_incident(controller: ResearchController) -> Path:
    measurement_contract = scoring_baseline_measurement_contract_snapshot()
    measurement_contract.pop("digest")
    measurement_contract.pop("case_memory_policy")
    measurement_contract["digest"] = canonical_sha256(measurement_contract)
    assert measurement_contract["digest"] == SCORING_OOM_MEASUREMENT_DIGEST
    material = {
        "schema_version": 1,
        "operation": "score-baseline-qualify",
        "expected_git_commit": SCORING_OOM_SOURCE_COMMIT,
        "framework_git_commit": PRODUCTION_FRAMEWORK_COMMIT,
        "scoring_framework_git_commit": SCORING_OOM_SOURCE_COMMIT,
        "evaluator_image": PRODUCTION_EVALUATOR,
        "reference_source_sha256": scoring_reference_source_sha256(),
        "timing_protocol": device_event_protocol_snapshot(),
        "measurement_contract": measurement_contract,
        "probe_count": 10,
    }
    assert canonical_sha256(material) == SCORING_OOM_OPERATION_DIGEST
    intent = {
        **material,
        "operation_id": SCORING_OOM_OPERATION_ID,
        "operation_digest": SCORING_OOM_OPERATION_DIGEST,
        "status": "RUNNING",
        "active_probe_index": None,
        "completed_probes": 0,
    }
    unknown = {
        **intent,
        "status": "UNKNOWN_OUTCOME",
        "active_probe_index": 0,
        "error": SCORING_OOM_ERROR,
    }
    operation = (
        controller.config.controller_dir
        / "scoring-baseline"
        / SCORING_OOM_OPERATION_ID
    )
    operation.mkdir(parents=True)
    values = {
        "intent.json": intent,
        "state.json": unknown,
        "probe-00.receipt.json": {
            "container_exit_code": 137,
            "error": SCORING_OOM_ERROR,
            "status": "UNKNOWN_OUTCOME",
        },
    }
    for name, value in values.items():
        (operation / name).write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    (operation / "probe-00.stdout.json").write_bytes(b"")
    (operation / "probe-00.stderr.txt").write_bytes(b"")
    return operation


class ProbeRunner:
    def __init__(
        self,
        *,
        timeout: bool = False,
        unqualified: bool = False,
        identity_mismatch: bool = False,
        pre_gpu_cli_rejection: bool = False,
    ) -> None:
        self.calls = 0
        self.timeout = timeout
        self.unqualified = unqualified
        self.identity_mismatch = identity_mismatch
        self.pre_gpu_cli_rejection = pre_gpu_cli_rejection

    def run(self, argv, **_kwargs):
        index = self.calls
        self.calls += 1
        if self.pre_gpu_cli_rejection:
            return CommandResult(
                argv=tuple(argv),
                returncode=2,
                stdout="",
                stderr=(
                    "usage: kernel-research [-h] {doctor}\n"
                    "kernel-research: error: argument command: invalid choice: "
                    "'score-baseline-probe' (choose from 'doctor')\n"
                ),
            )
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
    def controller(
        self, root: Path, runner: ProbeRunner, **config_kwargs
    ) -> ResearchController:
        controller = ResearchController(
            make_config(root, **config_kwargs), runner=runner
        )
        self.patchers = [
            mock.patch.object(controller, "_verify_repository", return_value={}),
            mock.patch.object(
                controller,
                "_prepare_framework_snapshot",
                return_value=root / "framework",
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

    @staticmethod
    def recovery_doctor(commit: str) -> dict:
        return {
            "schema_version": 1,
            "command": "doctor",
            "status": "SUCCESS",
            "identity": {"git_commit": commit},
            "c500_probe": {
                "status": "SUCCESS",
                "environment": {"compile_probe_status": "PASSED"},
            },
        }

    def test_exact_oom_is_abandoned_after_one_fresh_doctor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery_commit = "f" * 40
            runner = ProbeRunner()
            controller = self.controller(
                root,
                runner,
                expected_git_commit=recovery_commit,
                framework_git_commit=PRODUCTION_FRAMEWORK_COMMIT,
                evaluator_image=PRODUCTION_EVALUATOR,
            )
            operation = write_oom_incident(controller)
            doctor_result = self.recovery_doctor(recovery_commit)
            with (
                mock.patch.object(
                    controller, "_verify_scoring_oom_recovery_commit"
                ),
                mock.patch.object(controller, "_assert_scoring_host_idle"),
                mock.patch.object(
                    controller, "_scoring_container_present", return_value=False
                ),
                mock.patch.object(
                    controller, "doctor", return_value=doctor_result
                ) as doctor,
            ):
                final = controller.finalize_scoring_unknown_oom()
                repeated = controller.finalize_scoring_unknown_oom()
                controller._assert_no_unresolved_scoring_baseline()

            self.assertEqual(final["status"], SCORING_ABANDONED_UNKNOWN_STATUS)
            self.assertEqual(final["original_status"], "UNKNOWN_OUTCOME")
            self.assertEqual(final["qualification_effect"], "none")
            self.assertEqual(final["deployment_effect"], "none")
            self.assertFalse(final["replay_permitted"])
            self.assertFalse(final["idempotent"])
            self.assertTrue(repeated["idempotent"])
            doctor.assert_called_once_with(
                require_secret=False,
                run_id="score-baseline-oom-recovery",
            )
            self.assertEqual(runner.calls, 0)
            unknown = json.loads(
                (operation / "unknown.json").read_text(encoding="utf-8")
            )
            state = json.loads(
                (operation / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(unknown["status"], "UNKNOWN_OUTCOME")
            self.assertEqual(state["status"], SCORING_ABANDONED_UNKNOWN_STATUS)

    def test_oom_abandonment_rejects_source_evidence_drift(self) -> None:
        for target in ("receipt", "raw-output"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                recovery_commit = "f" * 40
                controller = self.controller(
                    root,
                    ProbeRunner(),
                    expected_git_commit=recovery_commit,
                    framework_git_commit=PRODUCTION_FRAMEWORK_COMMIT,
                    evaluator_image=PRODUCTION_EVALUATOR,
                )
                operation = write_oom_incident(controller)
                if target == "receipt":
                    receipt = operation / "probe-00.receipt.json"
                    payload = json.loads(receipt.read_text(encoding="utf-8"))
                    payload["container_exit_code"] = 0
                    receipt.write_text(json.dumps(payload), encoding="utf-8")
                else:
                    (operation / "probe-00.stderr.txt").write_text(
                        "untrusted\n", encoding="utf-8"
                    )
                with (
                    mock.patch.object(
                        controller, "_verify_scoring_oom_recovery_commit"
                    ),
                    mock.patch.object(controller, "_assert_scoring_host_idle"),
                    mock.patch.object(
                        controller,
                        "_scoring_container_present",
                        return_value=False,
                    ),
                    mock.patch.object(controller, "doctor") as doctor,
                    self.assertRaisesRegex(
                        Exception,
                        "source evidence|raw output contract",
                    ),
                ):
                    controller.finalize_scoring_unknown_oom()
                doctor.assert_not_called()

    def test_failed_oom_recovery_doctor_never_terminalizes_or_replays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery_commit = "f" * 40
            controller = self.controller(
                root,
                ProbeRunner(),
                expected_git_commit=recovery_commit,
                framework_git_commit=PRODUCTION_FRAMEWORK_COMMIT,
                evaluator_image=PRODUCTION_EVALUATOR,
            )
            operation = write_oom_incident(controller)
            failed_doctor = {
                "status": "FAILED",
                "identity": {"git_commit": recovery_commit},
                "c500_probe": None,
            }
            with (
                mock.patch.object(
                    controller, "_verify_scoring_oom_recovery_commit"
                ),
                mock.patch.object(controller, "_assert_scoring_host_idle"),
                mock.patch.object(
                    controller, "_scoring_container_present", return_value=False
                ),
                mock.patch.object(
                    controller, "doctor", return_value=failed_doctor
                ) as doctor,
                self.assertRaisesRegex(Exception, "doctor failed"),
            ):
                controller.finalize_scoring_unknown_oom()
            state = json.loads(
                (operation / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "UNKNOWN_OUTCOME")
            self.assertFalse((operation / "final.json").exists())
            self.assertEqual(doctor.call_count, 1)
            with (
                mock.patch.object(
                    controller, "_verify_scoring_oom_recovery_commit"
                ),
                mock.patch.object(controller, "_assert_scoring_host_idle"),
                mock.patch.object(
                    controller, "_scoring_container_present", return_value=False
                ),
                mock.patch.object(controller, "doctor") as repeated_doctor,
                self.assertRaisesRegex(Exception, "doctor failed"),
            ):
                controller.finalize_scoring_unknown_oom()
            repeated_doctor.assert_not_called()

    def test_saved_successful_oom_doctor_resumes_without_doctor_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery_commit = "f" * 40
            controller = self.controller(
                root,
                ProbeRunner(),
                expected_git_commit=recovery_commit,
                framework_git_commit=PRODUCTION_FRAMEWORK_COMMIT,
                evaluator_image=PRODUCTION_EVALUATOR,
            )
            operation = write_oom_incident(controller)
            material, unknown, receipt = controller._verify_scoring_oom_source(
                operation, state_name="state.json"
            )
            recovery_intent = controller._scoring_oom_recovery_intent(
                recovery_commit=recovery_commit,
                unknown=unknown,
                receipt=receipt,
            )
            (operation / "unknown.json").write_text(
                json.dumps(unknown, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            (operation / "recovery-intent.json").write_text(
                json.dumps(
                    recovery_intent, sort_keys=True, separators=(",", ":")
                )
                + "\n",
                encoding="utf-8",
            )
            (operation / "recovery-doctor.json").write_text(
                json.dumps(
                    self.recovery_doctor(recovery_commit),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    controller, "_verify_scoring_oom_recovery_commit"
                ),
                mock.patch.object(controller, "_assert_scoring_host_idle"),
                mock.patch.object(
                    controller, "_scoring_container_present", return_value=False
                ),
                mock.patch.object(controller, "doctor") as doctor,
            ):
                final = controller.finalize_scoring_unknown_oom()
            self.assertEqual(final["status"], SCORING_ABANDONED_UNKNOWN_STATUS)
            doctor.assert_not_called()
            self.assertEqual(material["operation"], "score-baseline-qualify")

    def test_oom_recovery_commit_requires_exact_parent_and_path_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self.controller(
                root,
                ProbeRunner(),
                expected_git_commit="f" * 40,
                framework_git_commit=PRODUCTION_FRAMEWORK_COMMIT,
                evaluator_image=PRODUCTION_EVALUATOR,
            )
            with (
                mock.patch(
                    "kernel_research.autorun.controller._run_git",
                    return_value="0" * 40,
                ),
                self.assertRaisesRegex(Exception, "direct incident child"),
            ):
                controller._verify_scoring_oom_recovery_commit()
            with (
                mock.patch(
                    "kernel_research.autorun.controller._run_git",
                    side_effect=(SCORING_OOM_SOURCE_COMMIT, "kernel.py"),
                ),
                self.assertRaisesRegex(Exception, "unauthorized path"),
            ):
                controller._verify_scoring_oom_recovery_commit()

    def test_exact_argparse_rejection_can_be_finalized_without_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = ProbeRunner(pre_gpu_cli_rejection=True)
            controller = self.controller(root, runner)
            with self.assertRaisesRegex(Exception, "outcome is unknown"):
                controller.qualify_scoring_baseline()
            operation = next(
                (controller.config.controller_dir / "scoring-baseline").iterdir()
            )
            rewrite_unknown_as_legacy_operation(operation)
            with mock.patch.object(
                controller, "_scoring_container_present", return_value=False
            ):
                final = controller.finalize_scoring_pre_gpu_failure()
                repeated = controller.finalize_scoring_pre_gpu_failure()
            self.assertEqual(final["status"], "UNQUALIFIED")
            self.assertEqual(
                final["failure_classification"],
                "KNOWN_PRE_GPU_CLI_REJECTION",
            )
            self.assertFalse(final["idempotent"])
            self.assertTrue(repeated["idempotent"])
            self.assertEqual(runner.calls, 1)

    def test_pre_gpu_finalizer_rejects_stderr_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = ProbeRunner(pre_gpu_cli_rejection=True)
            controller = self.controller(root, runner)
            with self.assertRaisesRegex(Exception, "outcome is unknown"):
                controller.qualify_scoring_baseline()
            stderr = next(
                (controller.config.controller_dir / "scoring-baseline").glob(
                    "*/probe-00.stderr.txt"
                )
            )
            stderr.write_text("usage: unrelated\n", encoding="utf-8")
            with (
                mock.patch.object(
                    controller, "_scoring_container_present", return_value=False
                ),
                self.assertRaisesRegex(Exception, "argparse proof"),
            ):
                controller.finalize_scoring_pre_gpu_failure()

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
                    payload["cases"][0]["anchor_measurement"][
                        "candidate_median_ms"
                    ] = 2.0
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
