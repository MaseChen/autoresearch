from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from types import SimpleNamespace
from unittest import mock

from kernel_research.autorun.controller import (
    ActionBudgetExhausted,
    DockerEvaluator,
    ResearchController,
    UnknownGPUOutcome,
    gpu_lock,
)
from kernel_research.autorun.models import ControllerConfig
from kernel_research.autorun.runtime import CommandResult
from kernel_research.device_timing import device_event_protocol_snapshot
from kernel_research.platform.canonical import canonical_sha256
from kernel_research.scoring_candidate_measurement import (
    SCORING_CANDIDATE_WORKER_REVISION,
    scoring_candidate_measurement_contract_snapshot,
)
from kernel_research.scoring_shadow import scoring_shadow_profile_snapshot


class CandidateProbeRunner:
    def __init__(self, *, status="QUALIFIED", returncode=0, **flags):
        self.status = status
        self.returncode = returncode
        self.flags = flags
        self.calls = []

    def run(self, argv, **kwargs):
        self.calls.append((tuple(argv), kwargs))
        profile = scoring_shadow_profile_snapshot()
        environment = {"device": "cuda:0"}
        payload = {
            "schema_version": 1,
            "command": "score-candidate-probe",
            "status": self.status,
            "gpu_state": (
                "STARTED"
                if self.status == "UNKNOWN_OUTCOME"
                else "COMPLETED"
            ),
            "completion_trusted": self.status != "UNKNOWN_OUTCOME",
            "phase": "complete" if self.status == "QUALIFIED" else "test-phase",
            "protocol_id": "xpuoj-th0-proxy-v1",
            "worker_revision": SCORING_CANDIDATE_WORKER_REVISION,
            "scoring_framework_git_commit": "d" * 40,
            "scoring_profile_digest": profile["digest"],
            "candidate_hash": "a" * 64,
            "incumbent_hash": "b" * 64,
            "measurement_contract": scoring_candidate_measurement_contract_snapshot(),
            "timing_protocol": device_event_protocol_snapshot(),
            "environment": environment,
            "environment_snapshot_digest": canonical_sha256(environment),
            "cases": [],
        }
        if self.status == "UNQUALIFIED":
            payload.pop("environment")
            payload.pop("environment_snapshot_digest")
            payload["error"] = "correctness failed"
            payload["traceback"] = "trace"
        elif self.status == "UNKNOWN_OUTCOME":
            payload.pop("environment")
            payload.pop("environment_snapshot_digest")
            payload["error"] = "worker interrupted"
            payload["traceback"] = "trace"
        return CommandResult(
            argv=tuple(argv),
            returncode=self.returncode,
            stdout=json.dumps(payload),
            stderr=str(self.flags.get("stderr", "")),
            timed_out=bool(self.flags.get("timed_out", False)),
            output_limited=bool(self.flags.get("output_limited", False)),
        )


class ScoringCandidateDockerEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        repository = root / "repo"
        repository.mkdir()
        key = root / "key"
        key.write_text("unused", encoding="utf-8")
        self.candidate = root / "candidate.py"
        self.incumbent = root / "incumbent.py"
        self.candidate.write_text("candidate\n", encoding="utf-8")
        self.incumbent.write_text("incumbent\n", encoding="utf-8")
        self.config = ControllerConfig(
            repository_dir=repository,
            state_dir=root / "state",
            controller_dir=root / "controller",
            checkpoint_dir=root / "checkpoints",
            docker_binary=Path("/usr/bin/docker"),
            proposer_image="registry/proposer@sha256:" + "c" * 64,
            evaluator_image="registry/evaluator@sha256:" + "e" * 64,
            deepseek_key_file=key,
            gpu_devices=(),
            evaluator_cache_dir=root / "cache",
            expected_git_commit="d" * 40,
            expected_kernel_hash="f" * 64,
            framework_git_commit="d" * 40,
        )
        self.result_dir = root / "result"

    def run_probe(self, runner, before=None):
        evaluator = DockerEvaluator(
            self.config,
            controller_dir=self.config.controller_dir,
            runner=runner,
            before_container_start=before,
        )
        return evaluator.scoring_candidate_probe(
            operation_id="score-candidate-" + "1" * 24,
            result_dir=self.result_dir,
            candidate_path=self.candidate,
            incumbent_path=self.incumbent,
            candidate_hash="a" * 64,
            incumbent_hash="b" * 64,
            scoring_profile=scoring_shadow_profile_snapshot(),
        )

    def test_probe_preserves_output_and_accepts_exact_qualified_exit(self):
        runner = CandidateProbeRunner()
        guarded = []
        result = self.run_probe(runner, lambda *args: guarded.append(args))

        self.assertEqual(result["status"], "QUALIFIED")
        self.assertEqual(result["container_exit_code"], 0)
        self.assertTrue((self.result_dir / "candidate-probe.stdout.json").is_file())
        self.assertTrue((self.result_dir / "candidate-probe.stderr.txt").is_file())
        self.assertEqual(len(runner.calls), 1)
        argv, kwargs = runner.calls[0]
        self.assertIn("score-candidate-probe", argv)
        self.assertEqual(kwargs["container_name"], "kar-score-candidate-111111111111")
        self.assertEqual(guarded[0][1], "scoring-candidate")

    def test_unqualified_is_known_only_with_exit_two(self):
        result = self.run_probe(CandidateProbeRunner(status="UNQUALIFIED", returncode=2))
        self.assertEqual(result["status"], "UNQUALIFIED")
        self.assertEqual(result["container_exit_code"], 2)
        mismatch = self.run_probe(CandidateProbeRunner(status="UNQUALIFIED", returncode=0))
        self.assertEqual(mismatch["status"], "UNKNOWN_OUTCOME")

    def test_timeout_output_limit_and_fatal_marker_are_unknown(self):
        variants = (
            CandidateProbeRunner(timed_out=True),
            CandidateProbeRunner(output_limited=True),
            CandidateProbeRunner(stderr="illegal memory access"),
        )
        for runner in variants:
            with self.subTest(flags=runner.flags):
                result = self.run_probe(runner)
                self.assertEqual(result["status"], "UNKNOWN_OUTCOME")

    def test_worker_unknown_requires_exact_exit_and_inflight_evidence(self):
        result = self.run_probe(
            CandidateProbeRunner(status="UNKNOWN_OUTCOME", returncode=3)
        )
        self.assertEqual(result["status"], "UNKNOWN_OUTCOME")
        self.assertEqual(result["gpu_state"], "STARTED")
        mismatch = self.run_probe(
            CandidateProbeRunner(status="UNKNOWN_OUTCOME", returncode=2)
        )
        self.assertEqual(mismatch["status"], "UNKNOWN_OUTCOME")
        self.assertIn("status/exit", mismatch["error"])


class ScoringCandidateDurabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        repository = root / "repo"
        repository.mkdir()
        key = root / "key"
        key.write_text("unused", encoding="utf-8")
        self.candidate = root / "candidate.py"
        self.incumbent = root / "incumbent.py"
        self.candidate.write_text("candidate\n", encoding="utf-8")
        self.incumbent.write_text("incumbent\n", encoding="utf-8")
        import hashlib

        self.candidate_hash = hashlib.sha256(self.candidate.read_bytes()).hexdigest()
        self.incumbent_hash = hashlib.sha256(self.incumbent.read_bytes()).hexdigest()
        config = ControllerConfig(
            repository_dir=repository,
            state_dir=root / "state",
            controller_dir=root / "controller",
            checkpoint_dir=root / "checkpoints",
            docker_binary=Path("/usr/bin/docker"),
            proposer_image="registry/proposer@sha256:" + "c" * 64,
            evaluator_image="registry/evaluator@sha256:" + "e" * 64,
            deepseek_key_file=key,
            gpu_devices=(),
            evaluator_cache_dir=root / "cache",
            expected_git_commit="d" * 40,
            expected_kernel_hash="f" * 64,
            framework_git_commit="d" * 40,
        )
        self.controller = ResearchController(config)
        prepare = mock.patch.object(
            self.controller,
            "_prepare_framework_snapshot",
            return_value=root / "framework",
        )
        prepare.start()
        self.addCleanup(prepare.stop)
        self.profile = scoring_shadow_profile_snapshot()
        self.run = {
            "id": "run-1",
            "workflow_snapshot": {
                "objective_scoring_profile": self.profile,
                "runtime_binding": {"expected_git_commit": "d" * 40},
            },
        }
        self.baseline = SimpleNamespace(
            id=210, candidate_hash=self.incumbent_hash
        )
        self.identity = SimpleNamespace(
            experiment_uid="experiment-1", stage="full_primary"
        )
        self.iteration = {
            "id": 1,
            "iteration_index": 1,
            "candidate_hash": self.candidate_hash,
        }
        self.raw = {
            "status": "SUCCESS",
            "candidate_hash": self.candidate_hash,
            "baseline_candidate_hash": self.incumbent_hash,
            "cases": [{"matched_ratio": 1.0}],
        }

    def invoke(self):
        return self.controller._run_scoring_candidate_shadow(
            run=self.run,
            baseline=self.baseline,
            identity=self.identity,
            iteration=self.iteration,
            candidate_path=self.candidate,
            raw=self.raw,
            suite="full",
        )

    def test_terminal_measurement_is_digest_bound_and_reconciled_without_replay(self):
        result = {"status": "UNQUALIFIED", "error": "correctness failed"}
        with (
            mock.patch.object(
                self.controller,
                "_experiment_source_path",
                return_value=self.incumbent,
            ),
            mock.patch.object(
                self.controller.evaluator,
                "scoring_candidate_probe",
                return_value=result,
            ) as probe,
        ):
            first = self.invoke()
            second = self.invoke()
        self.assertEqual(first["result"], result)
        self.assertEqual(second, first)
        probe.assert_called_once()
        operation = (
            self.controller.config.controller_dir
            / "runs/run-1/results/001/full_primary.scoring-shadow"
        )
        final = json.loads((operation / "final.json").read_text(encoding="utf-8"))
        self.assertEqual(final["result_digest"], canonical_sha256(result))
        self.assertEqual(json.loads((operation / "state.json").read_text()), final)
        self.assertEqual(json.loads((operation / "receipt.json").read_text()), final)

    def test_receipt_repairs_final_and_state_without_replay(self):
        result = {"status": "UNQUALIFIED", "error": "known failure"}
        with (
            mock.patch.object(
                self.controller,
                "_experiment_source_path",
                return_value=self.incumbent,
            ),
            mock.patch.object(
                self.controller.evaluator,
                "scoring_candidate_probe",
                return_value=result,
            ) as probe,
        ):
            first = self.invoke()
            operation = (
                self.controller.config.controller_dir
                / "runs/run-1/results/001/full_primary.scoring-shadow"
            )
            (operation / "final.json").unlink()
            (operation / "state.json").unlink()
            second = self.invoke()
        self.assertEqual(second, first)
        self.assertEqual(
            json.loads((operation / "final.json").read_text()),
            json.loads((operation / "receipt.json").read_text()),
        )
        probe.assert_called_once()

    def test_prelaunch_rejection_is_known_and_never_replayed(self):
        with (
            mock.patch.object(
                self.controller,
                "_experiment_source_path",
                return_value=self.incumbent,
            ),
            mock.patch.object(
                self.controller.evaluator,
                "scoring_candidate_probe",
                side_effect=ActionBudgetExhausted("deadline"),
            ) as probe,
        ):
            first = self.invoke()
            second = self.invoke()
        self.assertEqual(first["status"], "LAUNCH_REJECTED")
        self.assertEqual(first["result"]["gpu_state"], "NOT_STARTED")
        self.assertTrue(first["result"]["completion_trusted"])
        self.assertEqual(second, first)
        probe.assert_called_once()

    def test_interruption_after_launch_boundary_is_unknown(self):
        def interrupt(**_kwargs):
            context = self.controller._scoring_candidate_context
            assert context is not None
            launching = {**context["state"], "status": "LAUNCHING"}
            Path(context["state_path"]).write_text(
                json.dumps(launching), encoding="utf-8"
            )
            context["state"] = launching
            context["status"] = "LAUNCHING"
            context["launch_started"] = True
            raise KeyboardInterrupt()

        with (
            mock.patch.object(
                self.controller,
                "_experiment_source_path",
                return_value=self.incumbent,
            ),
            mock.patch.object(
                self.controller.evaluator,
                "scoring_candidate_probe",
                side_effect=interrupt,
            ) as probe,
            self.assertRaisesRegex(UnknownGPUOutcome, "after launch"),
        ):
            self.invoke()
        with self.assertRaises(UnknownGPUOutcome):
            self.invoke()
        probe.assert_called_once()

    def test_unresolved_candidate_state_fences_other_gpu_work(self):
        state_dir = (
            self.controller.config.controller_dir
            / "runs/run-x/results/001/full_primary.scoring-shadow"
        )
        state_dir.mkdir(parents=True)
        (state_dir / "state.json").write_text(
            json.dumps({"status": "UNKNOWN_OUTCOME"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(Exception, "unresolved GPU outcome"):
            self.controller._assert_no_unresolved_scoring_candidates()

    def test_unresolved_candidate_state_fences_the_shared_gpu_lock(self):
        state_dir = (
            self.controller.config.controller_dir
            / "runs/run-x/results/001/full_primary.scoring-shadow"
        )
        state_dir.mkdir(parents=True)
        (state_dir / "state.json").write_text(
            json.dumps({"status": "UNKNOWN_OUTCOME"}), encoding="utf-8"
        )

        with self.assertRaisesRegex(
            Exception, "fenced by an unresolved scoring candidate outcome"
        ):
            with gpu_lock(self.controller.config.controller_dir / "gpu1.lock"):
                self.fail("the shared GPU lock must not admit new work")

    def test_unknown_result_is_durable_and_never_replayed(self):
        with (
            mock.patch.object(
                self.controller,
                "_experiment_source_path",
                return_value=self.incumbent,
            ),
            mock.patch.object(
                self.controller.evaluator,
                "scoring_candidate_probe",
                return_value={"status": "UNKNOWN_OUTCOME", "error": "timeout"},
            ) as probe,
            self.assertRaisesRegex(UnknownGPUOutcome, "timeout"),
        ):
            self.invoke()
        with self.assertRaisesRegex(UnknownGPUOutcome, "timeout"):
            self.invoke()
        probe.assert_called_once()
        operation = (
            self.controller.config.controller_dir
            / "runs/run-1/results/001/full_primary.scoring-shadow"
        )
        state = json.loads((operation / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "UNKNOWN_OUTCOME")

    def test_last_moment_guard_reproves_state_sources_run_and_attempt(self):
        state_path = Path(self.temporary.name) / "state.json"
        state = {
            "status": "PREPARING",
            "operation_id": "score-candidate-" + "1" * 24,
        }
        state_path.write_text(json.dumps(state), encoding="utf-8")
        context = {
            **state,
            "state_path": str(state_path),
            "state": state,
            "candidate_path": str(self.candidate),
            "incumbent_path": str(self.incumbent),
            "candidate_hash": self.candidate_hash,
            "incumbent_hash": self.incumbent_hash,
            "run_id": "run-1",
            "experiment_uid": "experiment-1",
            "stage": "full_primary",
            "iteration_id": 1,
            "container_name": "kar-score-candidate-111111111111",
            "launch_started": False,
        }
        store = mock.MagicMock()
        store.get_run.return_value = {
            "id": "run-1",
            "status": "RUNNING",
            "stop_requested": 0,
            "deadline_epoch": 9999999999.0,
        }
        store.get_evaluation_attempt_by_uid.return_value = {
            "status": "RUNNING",
            "stage": "full_primary",
        }
        manager = mock.MagicMock()
        manager.__enter__.return_value = store
        with (
            mock.patch(
                "kernel_research.autorun.controller.ControllerStore",
                return_value=manager,
            ),
            mock.patch.object(self.controller, "_assert_no_unresolved_scoring_baseline"),
            mock.patch.object(self.controller, "_assert_no_unresolved_scoring_candidates"),
            mock.patch.object(self.controller, "_authorize_external_action"),
            mock.patch.object(self.controller, "_verify_repository") as verify,
        ):
            self.controller._authorize_scoring_candidate_probe(
                context,
                operation_id=state["operation_id"],
                configured_timeout=float(self.controller.config.evaluator_timeout_sec),
            )
        verify.assert_called_once_with(allowed_best_hashes={self.incumbent_hash})

        self.assertTrue(context["launch_started"])
        self.assertEqual(context["status"], "LAUNCHING")
        with self.assertRaisesRegex(Exception, "active intent"):
            self.controller._authorize_scoring_candidate_probe(
                context,
                operation_id=state["operation_id"],
                configured_timeout=float(self.controller.config.evaluator_timeout_sec),
            )


if __name__ == "__main__":
    unittest.main()
