from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from kernel_research.autorun import cli as autorun_cli
from kernel_research.autorun import controller as controller_module
from kernel_research.autorun import runtime as autorun_runtime
from kernel_research.autorun.controller import DockerEvaluator, ResearchController
from kernel_research.autorun.errors import ControlledRuntimeError
from kernel_research.autorun.models import ProposalV1
from kernel_research.autorun.runtime import CommandResult, CommandRunner
from kernel_research.autorun.states import IterationStatus, RunStatus, Stage
from kernel_research.autorun.store import ControllerStore
from kernel_research.backends import (
    BackendResult,
    C500Backend,
    STATUS_CRASH,
    STATUS_PRECISION_FAILED,
    STATUS_SUCCESS,
)
from kernel_research.contract import CandidateError
from kernel_research.history import HistoryStore
from kernel_research import policy_worker
from kernel_research.policy_worker import _bounded_result
from kernel_research.research_policy import ResearchPolicyResult

from test_autorun import (
    FakeEvaluator,
    NoopRunner,
    SEED,
    SEED_HASH,
    StaticProposer,
    _baseline,
    _config,
    _proposal_value,
    _with_different_block_size_n,
)


class C500OrchestrationTests(unittest.TestCase):
    def _run(
        self,
        root: Path,
        *,
        baseline: bool = True,
        candidate_kernel=None,
        baseline_kernel=None,
        matched_ratio: float = 1.0,
        benchmark_error: Exception | None = None,
    ):
        candidate_path = root / "candidate.py"
        candidate_path.write_text(SEED, encoding="utf-8")
        baseline_path = root / "baseline.py"
        baseline_path.write_text(SEED, encoding="utf-8")
        candidate_kernel = candidate_kernel or (lambda *_args: None)
        baseline_kernel = baseline_kernel or (lambda *_args: None)
        modules = [
            SimpleNamespace(run_kernel=candidate_kernel),
            SimpleNamespace(run_kernel=baseline_kernel),
        ]
        fake_torch = SimpleNamespace(empty_like=lambda _value: object())
        health = BackendResult(
            status=STATUS_SUCCESS,
            eligible_for_promotion=False,
            candidate_hash="",
            environment={"device": "cuda:0", "device_name": "MetaX C500"},
            benchmark_config={"mode": "fixture"},
            backend="c500",
        )
        tensor_counter = iter(range(100))

        def tensors(*_args):
            marker = next(tensor_counter)
            out = object()
            return {
                "out": out,
                "arguments": (marker, marker + 1, out),
            }

        progress: list[tuple[str, str | None, str]] = []
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(C500Backend, "doctor", return_value=health)
            )
            stack.enter_context(
                mock.patch(
                    "kernel_research.backends._load_c500_runtime",
                    return_value=((fake_torch, object()), None),
                )
            )
            stack.enter_context(
                mock.patch(
                    "kernel_research.cases.get_suite",
                    return_value=(
                        SimpleNamespace(name="fixture-a"),
                        SimpleNamespace(name="fixture-b"),
                    ),
                )
            )
            stack.enter_context(
                mock.patch(
                    "kernel_research.cases.generate_case",
                    return_value=object(),
                )
            )
            stack.enter_context(
                mock.patch(
                    "kernel_research.backends._import_candidate",
                    side_effect=modules,
                )
            )
            stack.enter_context(
                mock.patch(
                    "kernel_research.backends._copy_dataset_to_device",
                    side_effect=tensors,
                )
            )
            stack.enter_context(
                mock.patch(
                    "kernel_research.backends._clone_readonly_inputs",
                    return_value={},
                )
            )
            stack.enter_context(
                mock.patch(
                    "kernel_research.backends._torch_reference",
                    return_value=object(),
                )
            )
            stack.enter_context(
                mock.patch("kernel_research.backends._synchronize")
            )
            stack.enter_context(
                mock.patch(
                    "kernel_research.backends._matched_ratio",
                    return_value=matched_ratio,
                )
            )
            stack.enter_context(
                mock.patch(
                    "kernel_research.backends._fresh_output_check",
                    return_value=matched_ratio,
                )
            )
            stack.enter_context(
                mock.patch(
                    "kernel_research.backends._readonly_inputs_unchanged",
                    return_value=(True, None),
                )
            )
            if benchmark_error is None:
                stack.enter_context(
                    mock.patch(
                        "kernel_research.backends._benchmark_candidate",
                        return_value=[10.0] * 30,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "kernel_research.backends._benchmark_interleaved",
                        return_value=([10.0] * 30, [12.0] * 30),
                    )
                )
            else:
                stack.enter_context(
                    mock.patch(
                        "kernel_research.backends._benchmark_candidate",
                        side_effect=benchmark_error,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "kernel_research.backends._benchmark_interleaved",
                        side_effect=benchmark_error,
                    )
                )
            result = C500Backend().evaluate(
                candidate_path,
                suite="smoke",
                baseline_path=baseline_path if baseline else None,
                progress_callback=lambda *values: progress.append(values),
            )
        return result, progress

    def test_complete_candidate_and_baseline_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, progress = self._run(Path(temporary), baseline=True)
        self.assertEqual(result.status, STATUS_SUCCESS)
        self.assertEqual(len(result.cases), 2)
        self.assertTrue(all(len(case.latency_samples_us) == 30 for case in result.cases))
        self.assertTrue(
            all(len(case.baseline_latency_samples_us) == 30 for case in result.cases)
        )
        self.assertIn(
            ("compile", "fixture-a", "candidate-first-invocation"),
            progress,
        )
        self.assertIn(
            ("compile", "fixture-a", "baseline-first-invocation"),
            progress,
        )

    def test_candidate_only_and_failure_branches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = self._run(Path(temporary), baseline=False)
        self.assertEqual(result.status, STATUS_SUCCESS)
        self.assertEqual(result.cases[0].baseline_latency_samples_us, ())

        with tempfile.TemporaryDirectory() as temporary:
            result, _ = self._run(
                Path(temporary),
                candidate_kernel=lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("illegal memory access")
                ),
            )
        self.assertEqual(result.status, STATUS_CRASH)
        self.assertIn("candidate first invocation", result.error)

        with tempfile.TemporaryDirectory() as temporary:
            result, _ = self._run(
                Path(temporary), matched_ratio=0.5
            )
        self.assertEqual(result.status, STATUS_PRECISION_FAILED)

        with tempfile.TemporaryDirectory() as temporary:
            result, _ = self._run(
                Path(temporary),
                baseline=False,
                benchmark_error=RuntimeError("timer failed"),
            )
        self.assertEqual(result.status, STATUS_CRASH)
        self.assertIn("timing failed", result.error)


class ControllerPublicSurfaceTests(unittest.TestCase):
    def _controller(self, root: Path):
        config = _config(root, max_candidates=1)
        (config.repository_dir / "program.md").write_text(
            "Only kernel.py.", encoding="utf-8"
        )
        _baseline(config.state_dir)
        source = _with_different_block_size_n(SEED)
        proposal = ProposalV1.from_value(
            _proposal_value(source), expected_parent_hash=SEED_HASH
        )
        runner = NoopRunner()
        controller = ResearchController(
            config,
            evaluator=FakeEvaluator(),
            runner=runner,  # type: ignore[arg-type]
            proposer_factory=lambda *_args: StaticProposer(proposal),
        )
        return controller, config, runner

    def test_public_start_status_stop_and_resume_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, config, runner = self._controller(Path(temporary))
            with mock.patch.object(
                controller,
                "doctor",
                return_value={"status": "SUCCESS", "errors": []},
            ):
                result = controller.start()
            self.assertEqual(result["status"], "PROMOTED")
            status = controller.status(result["id"])
            self.assertEqual(status["status"], "PROMOTED")
            self.assertEqual(controller.resume(result["id"])["status"], "PROMOTED")

            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id="stop-run",
                    deadline_epoch=time.time() + 3600,
                    config=config.redacted_dict(),
                    initial_best_hash=SEED_HASH,
                )
                iteration = store.create_iteration("stop-run", 1, SEED_HASH)
                store.update_iteration(
                    iteration["id"], active_container="exact-stop-container"
                )
            stopped = controller.stop("stop-run")
            self.assertTrue(stopped["stop_requested"])
            self.assertEqual(runner.removed, ["exact-stop-container"])
            with ControllerStore(controller.controller_db) as store:
                self.assertIsNone(
                    store.latest_iteration("stop-run")["active_container"]
                )

    def test_status_without_database_and_doctor_branches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, _config_value, _runner = self._controller(
                Path(temporary)
            )
            controller.controller_db.unlink(missing_ok=True)
            self.assertEqual(controller.status()["status"], "NO_RUNS")

            with (
                mock.patch.object(
                    type(controller.config), "validate_host", return_value=[]
                ),
                mock.patch.object(
                    controller,
                    "_verify_repository",
                    return_value={"git_commit": "fixture"},
                ),
                mock.patch.object(controller, "_prepare_framework"),
                mock.patch.object(
                    controller,
                    "_inspect_image",
                    return_value={
                        "available": True,
                        "repo_digests": "[]",
                        "error": None,
                    },
                ),
                mock.patch.object(
                    controller.evaluator,
                    "doctor",
                    return_value={
                        "status": "SUCCESS",
                        "environment": {
                            "compile_probe_status": "PASSED"
                        },
                    },
                ),
            ):
                doctor = controller.doctor()
            self.assertEqual(doctor["status"], "SUCCESS")
            self.assertTrue(doctor["gpu_passthrough_risk_acknowledged"])
            self.assertEqual(
                doctor["proposer_model"], "deepseek/deepseek-v4-pro"
            )
            self.assertEqual(doctor["proposer_reasoning_effort"], "max")
            self.assertEqual(
                doctor["proposer_declared_output_tokens"], 384_000
            )
            self.assertEqual(
                doctor["proposer_request_output_token_cap"], 384_000
            )

    def test_resume_rejects_opencode_model_drift_in_both_directions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller, pro_config, _runner = self._controller(root)
            pairs = (
                (
                    "deepseek/deepseek-v4-pro",
                    "deepseek/deepseek-v4-flash",
                ),
                (
                    "deepseek/deepseek-v4-flash",
                    "deepseek/deepseek-v4-pro",
                ),
            )
            for index, (stored_model, resumed_model) in enumerate(pairs, 1):
                run_id = f"model-drift-{index}"
                stored_config = replace(
                    pro_config, opencode_model=stored_model
                )
                with ControllerStore(controller.controller_db) as store:
                    store.create_run(
                        run_id=run_id,
                        deadline_epoch=time.time() + 3600,
                        config=stored_config.redacted_dict(),
                        initial_best_hash=SEED_HASH,
                    )
                resumed = ResearchController(
                    replace(pro_config, opencode_model=resumed_model),
                    evaluator=FakeEvaluator(),
                )
                with self.assertRaisesRegex(
                    ControlledRuntimeError,
                    "resume config changed immutable fields: opencode_model",
                ):
                    resumed.resume(run_id)

    def test_repository_identity_and_framework_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller, config, _runner = self._controller(root)
            repository = root / "trusted-repo"
            repository.mkdir()
            (repository / "kernel.py").write_text(SEED, encoding="utf-8")
            shutil.copytree(
                Path(__file__).resolve().parents[1] / "kernel_research",
                repository / "kernel_research",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            env = {
                **dict(os.environ),
                "GIT_AUTHOR_NAME": "Fixture",
                "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                "GIT_COMMITTER_NAME": "Fixture",
                "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
            }
            subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-m", "fixture"],
                cwd=repository,
                check=True,
                capture_output=True,
                env=env,
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            config = replace(
                config,
                repository_dir=repository,
                expected_git_commit=commit,
            )
            controller = ResearchController(
                config, evaluator=FakeEvaluator()
            )
            identity = controller._verify_repository()
            self.assertEqual(identity["kernel_hash"], SEED_HASH)
            first = controller._prepare_framework()
            second = controller._prepare_framework()
            self.assertEqual(first, second)
            self.assertTrue((first / "kernel_research").is_dir())

    def test_failed_checkpoint_removes_its_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller, config, _runner = self._controller(root)
            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id="checkpoint-failure",
                    deadline_epoch=time.time() + 60,
                    config=config.redacted_dict(),
                    initial_best_hash=SEED_HASH,
                )
            with (
                mock.patch(
                    "kernel_research.autorun.controller._sqlite_summary",
                    side_effect=RuntimeError("integrity fixture"),
                ),
                self.assertRaisesRegex(RuntimeError, "integrity fixture"),
            ):
                controller.checkpoint("checkpoint-failure")
            self.assertEqual(list(config.checkpoint_dir.iterdir()), [])

    def test_preflight_identity_and_image_failures_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            controller = ResearchController(config, evaluator=FakeEvaluator())
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                controller._best()
            with HistoryStore(
                config.state_dir / "history.sqlite3",
                state_dir=config.state_dir,
            ):
                pass
            with self.assertRaisesRegex(RuntimeError, "no accepted"):
                controller._best()
            _baseline(config.state_dir)
            best = controller._best()

            with (
                mock.patch.object(
                    controller_module,
                    "_run_git",
                    return_value="0" * 40,
                ),
                self.assertRaisesRegex(RuntimeError, "HEAD"),
            ):
                controller._verify_repository()
            with (
                mock.patch.object(
                    controller_module,
                    "_run_git",
                    side_effect=[config.expected_git_commit, "dirty"],
                ),
                self.assertRaisesRegex(RuntimeError, "tracked"),
            ):
                controller._verify_repository()
            with (
                mock.patch.object(
                    controller_module,
                    "_run_git",
                    side_effect=[config.expected_git_commit, ""],
                ),
                mock.patch.object(
                    controller_module,
                    "_sha256_file",
                    return_value="0" * 64,
                ),
                self.assertRaisesRegex(RuntimeError, "kernel.py"),
            ):
                controller._verify_repository()
            with (
                mock.patch.object(
                    controller_module,
                    "_run_git",
                    side_effect=[config.expected_git_commit, ""],
                ),
                mock.patch.object(
                    controller_module,
                    "_sha256_file",
                    return_value=config.expected_kernel_hash,
                ),
                self.assertRaisesRegex(RuntimeError, "not allowed"),
            ):
                controller._verify_repository(
                    allowed_best_hashes={"0" * 64}
                )
            (config.state_dir / best.artifact_path).unlink()
            with (
                mock.patch.object(
                    controller_module,
                    "_run_git",
                    side_effect=[config.expected_git_commit, ""],
                ),
                mock.patch.object(
                    controller_module,
                    "_sha256_file",
                    return_value=config.expected_kernel_hash,
                ),
                self.assertRaisesRegex(RuntimeError, "missing or corrupted"),
            ):
                controller._verify_repository()

            with mock.patch(
                "kernel_research.autorun.controller.subprocess.run",
                side_effect=OSError("docker unavailable"),
            ):
                image = controller._inspect_image(config.proposer_image)
            self.assertFalse(image["available"])
            self.assertIn("docker unavailable", image["error"])

    def test_framework_doctor_candidate_and_hard_failure_branches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller, config, _runner = self._controller(root)
            package = config.repository_dir / "kernel_research"
            package.mkdir()
            (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            environment = {
                **dict(os.environ),
                "GIT_AUTHOR_NAME": "Fixture",
                "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                "GIT_COMMITTER_NAME": "Fixture",
                "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
            }
            subprocess.run(
                ["git", "init"],
                cwd=config.repository_dir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "add", "."], cwd=config.repository_dir, check=True
            )
            subprocess.run(
                ["git", "commit", "-m", "framework fixture"],
                cwd=config.repository_dir,
                check=True,
                capture_output=True,
                env=environment,
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=config.repository_dir,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            config = replace(
                config,
                expected_git_commit=commit,
                framework_git_commit=commit,
            )
            controller = ResearchController(config, evaluator=FakeEvaluator())
            controller._prepare_framework()
            (package / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
            # Working-tree drift cannot change the commit-materialized view.
            controller._prepare_framework()
            snapshot = (
                config.controller_dir
                / "framework"
                / commit
                / "kernel_research"
                / "module.py"
            )
            self.assertEqual(snapshot.read_text(encoding="utf-8"), "VALUE = 1\n")
            snapshot.chmod(0o664)
            controller._prepare_framework()
            self.assertEqual(snapshot.stat().st_mode & 0o777, 0o644)
            snapshot.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "snapshot hash"):
                controller._prepare_framework()
            snapshot_root = snapshot.parents[1]
            shutil.rmtree(snapshot_root)
            snapshot_root.symlink_to(config.repository_dir, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "contains a symlink"):
                controller._prepare_framework()

            empty_state = root / "missing-state"
            config_without_state = replace(config, state_dir=empty_state)
            no_state = ResearchController(
                config_without_state, evaluator=FakeEvaluator()
            )
            no_state._prepare_runtime_dirs()
            self.assertFalse(empty_state.exists())

            with (
                mock.patch.object(
                    type(config), "validate_host", return_value=[]
                ),
                mock.patch.object(
                    controller, "_verify_repository", return_value={}
                ),
                mock.patch.object(controller, "_prepare_framework"),
                mock.patch.object(
                    controller,
                    "_inspect_image",
                    return_value={
                        "available": False,
                        "repo_digests": "",
                        "error": "missing",
                    },
                ),
            ):
                failed_images = controller.doctor()
            self.assertEqual(failed_images["status"], "FAILED")
            self.assertEqual(len(failed_images["errors"]), 2)

            with (
                mock.patch.object(
                    type(config), "validate_host", return_value=[]
                ),
                mock.patch.object(
                    controller, "_verify_repository", return_value={}
                ),
                mock.patch.object(controller, "_prepare_framework"),
                mock.patch.object(
                    controller,
                    "_inspect_image",
                    return_value={
                        "available": True,
                        "repo_digests": "[]",
                        "error": None,
                    },
                ),
                mock.patch.object(
                    controller.evaluator,
                    "doctor",
                    return_value={
                        "status": "SUCCESS",
                        "environment": {
                            "compile_probe_status": "FAILED"
                        },
                    },
                ),
            ):
                failed_probe = controller.doctor()
            self.assertIn(
                "C500 compile doctor did not pass",
                failed_probe["errors"],
            )

            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id="seen",
                    deadline_epoch=time.time() + 60,
                    config={},
                    initial_best_hash=SEED_HASH,
                )
                iteration = store.create_iteration("seen", 1, SEED_HASH)
                store.accept_candidate(
                    iteration["id"],
                    candidate_hash="f" * 64,
                    hypothesis="fixture",
                    rationale="fixture",
                    candidate_path="/candidate.py",
                )
                self.assertTrue(
                    controller._candidate_seen(
                        store,
                        run_id="seen",
                        candidate_hash="f" * 64,
                    )
                )
            self.assertTrue(
                controller._hard_failure(
                    SimpleNamespace(status="CRASH", error_summary=None)
                )
            )

    def test_budget_reasons_and_terminal_run_close_without_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, config, _runner = self._controller(Path(temporary))
            run = {
                "stop_requested": True,
                "deadline_epoch": time.time() + 60,
                "valid_candidates": 0,
                "consecutive_failures": 0,
            }
            self.assertEqual(
                controller._terminal_budget_reason(
                    run, has_pending_iteration=False
                ),
                "operator requested stop",
            )
            run["stop_requested"] = False
            run["deadline_epoch"] = time.time() - 1
            self.assertEqual(
                controller._terminal_budget_reason(
                    run, has_pending_iteration=False
                ),
                "six-hour wall-clock budget exhausted",
            )
            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id="terminal",
                    deadline_epoch=time.time() + 60,
                    config=config.redacted_dict(),
                    initial_best_hash=SEED_HASH,
                )
                store.update_run("terminal", status=RunStatus.STOPPED)
            result = controller._run_loop("terminal")
            self.assertEqual(result["status"], "STOPPED")


class QueueRunner:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[str, ...], dict]] = []
        self.removed: list[str] = []

    def run(self, argv, **kwargs):
        self.calls.append(
            (tuple(str(value) for value in argv), dict(kwargs))
        )
        return self.results.pop(0)

    def remove_exact_container(
        self, _docker_binary: Path, container_name: str
    ) -> None:
        self.removed.append(container_name)


class DockerEvaluatorFailureMatrixTests(unittest.TestCase):
    def _evaluator(
        self, root: Path, results: list[CommandResult]
    ) -> tuple[DockerEvaluator, QueueRunner, Path]:
        config = _config(root)
        runner = QueueRunner(results)
        evaluator = DockerEvaluator(
            config,
            controller_dir=config.controller_dir,
            runner=runner,  # type: ignore[arg-type]
        )
        candidate = root / "candidate.py"
        candidate.write_text(SEED, encoding="utf-8")
        return evaluator, runner, candidate

    def test_doctor_closes_every_command_failure_mode(self) -> None:
        success = json.dumps(
            {
                "status": "SUCCESS",
                "environment": {"compile_probe_status": "PASSED"},
            }
        )
        fixtures = (
            (
                CommandResult((), -15, "", "", timed_out=True),
                "TIMEOUT",
                "timed out",
            ),
            (
                CommandResult((), -15, "", "", output_limited=True),
                "CRASH",
                "exceeded",
            ),
            (
                CommandResult((), 1, "not-json", "diagnostic"),
                "CRASH",
                "invalid output",
            ),
            (
                CommandResult((), 9, success, ""),
                "CRASH",
                "exited 9",
            ),
            (
                CommandResult((), 0, success, ""),
                "SUCCESS",
                None,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            evaluator, runner, _candidate = self._evaluator(
                Path(temporary), [item[0] for item in fixtures]
            )
            for index, (_command, status, message) in enumerate(fixtures):
                with self.subTest(index=index):
                    result = evaluator.doctor(run_id=f"run-{index}")
                    self.assertEqual(result["status"], status)
                    if message is not None:
                        self.assertIn(message, result["error"])
            self.assertEqual(len(runner.calls), len(fixtures))

    def test_evaluate_closes_container_and_gpu_failure_modes(self) -> None:
        success = json.dumps(
            {"status": "SUCCESS", "cases": [], "environment": {}}
        )
        fixtures = (
            (
                CommandResult((), -15, "", "", timed_out=True),
                "TIMEOUT",
                "timed out",
            ),
            (
                CommandResult((), -15, "", "", output_limited=True),
                "CRASH",
                "exceeded",
            ),
            (
                CommandResult((), 137, "", ""),
                "CRASH",
                "137",
            ),
            (
                CommandResult((), 1, "not-json", ""),
                "CRASH",
                "invalid output",
            ),
            (
                CommandResult(
                    (),
                    1,
                    success,
                    "kernel causes ATU address translation error",
                ),
                "CRASH",
                "fatal GPU",
            ),
            (
                CommandResult((), 7, success, ""),
                "CRASH",
                "exited 7",
            ),
            (
                CommandResult((), 0, success, ""),
                "SUCCESS",
                None,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            evaluator, runner, candidate = self._evaluator(
                Path(temporary), [item[0] for item in fixtures]
            )
            for index, (_command, status, message) in enumerate(fixtures):
                with self.subTest(index=index):
                    result = evaluator.evaluate(
                        candidate_path=candidate,
                        suite="smoke",
                        baseline_path=None,
                        run_id="failure-matrix",
                        iteration_index=index + 1,
                        stage=f"stage-{index}",
                    )
                    self.assertEqual(result["status"], status)
                    if message is not None:
                        self.assertIn(message, result["error"])
            self.assertEqual(len(runner.calls), len(fixtures))

    def test_cache_manifest_is_trusted_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator, _runner, candidate = self._evaluator(root, [])
            candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
            evaluator._cache_dir(candidate_hash=candidate_hash)
            manifest = next(
                evaluator.controller_dir.rglob(f"{candidate_hash}.json")
            )
            manifest.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unreadable"):
                evaluator._cache_dir(candidate_hash=candidate_hash)
            manifest.write_text(
                json.dumps({"candidate_hash": "wrong"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                evaluator._cache_dir(candidate_hash=candidate_hash)


class RuntimeAndStoreBranchTests(unittest.TestCase):
    def test_runtime_validation_cleanup_and_json_helpers(self) -> None:
        runner = CommandRunner()
        with self.assertRaises(ValueError):
            runner.run(
                [sys.executable, "-c", "pass"],
                input_text=None,
                timeout_sec=0,
                max_output_bytes=1,
            )
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired(["fixture"], 2),
            0,
        ]
        CommandRunner._kill_client(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()

        with mock.patch(
            "kernel_research.autorun.runtime.subprocess.run",
            side_effect=OSError("docker unavailable"),
        ) as cleanup:
            CommandRunner._kill_exact_container(Path("/docker"), "exact")
        self.assertEqual(cleanup.call_count, 2)

        self.assertEqual(
            autorun_runtime.parse_json_output('{"status":"SUCCESS"}')[
                "status"
            ],
            "SUCCESS",
        )
        for output in ("not-json", "[]"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                autorun_runtime.parse_json_output(output)

        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            cache = Path(temporary) / "cache"
            argv = autorun_runtime.evaluator_doctor_argv(
                config,
                name="doctor",
                run_id="full-run-id",
                cache_dir=cache,
            )
            self.assertIn("doctor", argv)
            self.assertIn("180", argv)

    def test_store_negative_and_transactional_branches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "controller.sqlite3"
            with ControllerStore(path) as store:
                self.assertIsNone(store.latest_run())
                with self.assertRaisesRegex(ValueError, "unknown run"):
                    store.get_run("missing")
                store.create_run(
                    run_id="run",
                    deadline_epoch=time.time() + 60,
                    config={},
                    initial_best_hash=SEED_HASH,
                )
                store.update_run("run", status=RunStatus.RUNNING)
                store.update_run_with_event("run", "NOOP", {})
                self.assertEqual(store.update_run("run")["status"], "RUNNING")
                with self.assertRaisesRegex(ValueError, "unsupported run"):
                    store.update_run("run", invalid=True)
                store.request_stop("run")
                iteration = store.create_iteration("run", 1, SEED_HASH)
                self.assertIsNone(store.get_iteration_by_index("run", 2))
                with self.assertRaisesRegex(ValueError, "unknown iteration"):
                    store.get_iteration(999)
                self.assertFalse(store.candidate_seen(SEED_HASH))
                store.accept_candidate(
                    iteration["id"],
                    candidate_hash=SEED_HASH,
                    hypothesis="fixture",
                    rationale="fixture",
                    candidate_path="/candidate.py",
                )
                self.assertTrue(store.candidate_seen(SEED_HASH))
                with self.assertRaisesRegex(ValueError, "only be accepted"):
                    store.accept_candidate(
                        iteration["id"],
                        candidate_hash=SEED_HASH,
                        hypothesis="fixture",
                        rationale="fixture",
                        candidate_path="/candidate.py",
                    )
                with self.assertRaisesRegex(
                    ValueError, "unsupported iteration"
                ):
                    store.update_iteration(iteration["id"], invalid=True)
                current = store.update_iteration(iteration["id"])
                self.assertEqual(current["stage"], Stage.POLICY.value)
                with self.assertRaisesRegex(ValueError, "must be COMPLETED"):
                    store.update_iteration(
                        iteration["id"], stage=Stage.DONE
                    )
                with self.assertRaisesRegex(ValueError, "must be DONE"):
                    store.update_iteration(
                        iteration["id"],
                        status=IterationStatus.COMPLETED,
                    )
                store.update_iteration_with_event(
                    iteration["id"],
                    "ITERATION_DONE",
                    {},
                    stage=Stage.DONE,
                    status=IterationStatus.COMPLETED,
                )
                store.update_run_with_event(
                    "run",
                    "RUN_STOPPED",
                    {},
                    status=RunStatus.STOPPED,
                )
                with self.assertRaisesRegex(
                    ValueError, "terminal run status"
                ):
                    store.update_run("run", status=RunStatus.RUNNING)
                self.assertGreaterEqual(len(store.list_events("run")), 5)

            newer = Path(temporary) / "newer.sqlite3"
            connection = sqlite3.connect(newer)
            connection.execute("PRAGMA user_version = 99")
            connection.close()
            with self.assertRaisesRegex(RuntimeError, "newer"):
                ControllerStore(newer)


class PolicyWorkerMainTests(unittest.TestCase):
    class _BinaryStream:
        def __init__(self, value: bytes = b"") -> None:
            self.buffer = io.BytesIO(value)

        def write(self, value: str) -> int:
            return self.buffer.write(value.encode("utf-8"))

        def flush(self) -> None:
            return None

    def _run_main(
        self,
        source: bytes,
        *,
        validator=None,
    ) -> tuple[int, dict]:
        stdin = self._BinaryStream(source)
        stdout = self._BinaryStream()
        contexts = [
            mock.patch.object(policy_worker, "_set_limits"),
            mock.patch.object(policy_worker.sys, "stdin", stdin),
            mock.patch.object(policy_worker.sys, "stdout", stdout),
        ]
        if validator is not None:
            contexts.append(
                mock.patch.object(
                    policy_worker,
                    "validate_research_candidate",
                    side_effect=validator,
                )
            )
        with ExitStack() as stack:
            for context in contexts:
                stack.enter_context(context)
            result = policy_worker.main()
        return result, json.loads(stdout.buffer.getvalue())

    def test_policy_worker_valid_invalid_encoding_and_source_limit(self) -> None:
        status, value = self._run_main(SEED.encode("utf-8"))
        self.assertEqual(status, 0)
        self.assertTrue(value["valid"])
        self.assertEqual(value["sha256"], SEED_HASH)

        status, value = self._run_main(b"\xff")
        self.assertEqual(status, 0)
        self.assertEqual(value["errors"][0]["code"], "POLICY_RESOURCE_LIMIT")
        self.assertEqual(value["errors"][0]["message"], "UnicodeDecodeError")

        status, value = self._run_main(b"x" * (256 * 1024 + 1))
        self.assertEqual(status, 0)
        self.assertIn("exceeds", value["errors"][0]["message"])

        status, value = self._run_main(
            SEED.encode("utf-8"), validator=MemoryError("fixture")
        )
        self.assertEqual(status, 0)
        self.assertEqual(value["errors"][0]["message"], "MemoryError")

    def test_policy_worker_resource_limit_setup(self) -> None:
        with mock.patch.object(
            policy_worker.resource, "setrlimit"
        ) as setrlimit:
            policy_worker._set_limits()
        self.assertEqual(setrlimit.call_count, 3)

        def reject_address_space(kind, _limit):
            if kind == policy_worker.resource.RLIMIT_AS:
                raise ValueError("unsupported")

        with (
            mock.patch.object(
                policy_worker.resource,
                "setrlimit",
                side_effect=reject_address_space,
            ),
            mock.patch.object(policy_worker.sys, "platform", "darwin"),
        ):
            policy_worker._set_limits()
        with (
            mock.patch.object(
                policy_worker.resource,
                "setrlimit",
                side_effect=reject_address_space,
            ),
            mock.patch.object(policy_worker.sys, "platform", "linux"),
            self.assertRaises(ValueError),
        ):
            policy_worker._set_limits()


class CliAndWorkerTests(unittest.TestCase):
    def test_autorun_cli_handlers_and_json_errors(self) -> None:
        fake = mock.Mock()
        fake.doctor.return_value = {"status": "SUCCESS"}
        fake.start.return_value = {"status": "PROPOSAL_READY"}
        fake.resume.return_value = {"status": "PROMOTED"}
        fake.status.return_value = {"status": "NO_RUNS"}
        fake.stop.return_value = {"status": "RUNNING"}
        fake.checkpoint.return_value = {"status": "SUCCESS"}
        commands = (
            (["doctor", "--config", "/tmp/config"], 0),
            (["start", "--config", "/tmp/config", "--proposal-only"], 0),
            (
                ["resume", "--config", "/tmp/config", "--run-id", "run"],
                0,
            ),
            (["status", "--config", "/tmp/config"], 0),
            (["stop", "--config", "/tmp/config", "--run-id", "run"], 0),
            (
                ["checkpoint", "--config", "/tmp/config", "--run-id", "run"],
                0,
            ),
        )
        with mock.patch(
            "kernel_research.autorun.cli._controller", return_value=fake
        ):
            for argv, expected in commands:
                with self.subTest(argv=argv), redirect_stdout(io.StringIO()):
                    self.assertEqual(autorun_cli.main(argv), expected)

        with (
            mock.patch(
                "kernel_research.autorun.cli._controller",
                side_effect=sqlite3.DatabaseError("broken"),
            ),
            redirect_stderr(io.StringIO()) as error,
        ):
            self.assertEqual(
                autorun_cli.main(["status", "--config", "/tmp/config"]),
                2,
            )
        self.assertEqual(json.loads(error.getvalue())["status"], "FAILED")

        with mock.patch(
            "kernel_research.autorun.cli._controller",
            side_effect=RuntimeError("programming defect"),
        ):
            with self.assertRaisesRegex(RuntimeError, "programming defect"):
                autorun_cli.main(["status", "--config", "/tmp/config"])

    def test_autorun_start_resume_and_status_compact_json(self) -> None:
        fake = mock.Mock()
        terminal = {"id": "run", "status": "BUDGET_EXHAUSTED"}
        full_status = {
            "schema_version": 1,
            "command": "status",
            "status": "BUDGET_EXHAUSTED",
            "run": {
                "id": "run",
                "status": "BUDGET_EXHAUSTED",
                "valid_candidates": 5,
                "consecutive_failures": 0,
                "stop_reason": "valid candidate budget exhausted",
                "stop_requested": False,
                "initial_best_hash": "a" * 64,
                "final_best_hash": None,
                "created_at": "created",
                "updated_at": "updated",
                "config": {
                    "max_candidates": 5,
                    "max_hours": 6.0,
                    "opencode_model": "deepseek/deepseek-v4-flash",
                    "deepseek_key_file": "/secret/path",
                },
                "preflight": {
                    "large": True,
                    "proposer_reasoning_effort": "max",
                    "proposer_declared_output_tokens": 384_000,
                    "proposer_request_output_token_cap": 384_000,
                },
            },
            "iterations": [
                {
                    "iteration_index": 1,
                    "stage": "DONE",
                    "status": "COMPLETED",
                    "outcome": "FULL_REJECTED",
                    "candidate_hash": "b" * 64,
                    "hypothesis": "fixture",
                    "experiment_ids": {"full_primary": 1},
                    "active_container": None,
                    "error": None,
                    "result_summary": {
                        "relative_speedup_vs_accepted": 0.9
                    },
                }
            ],
        }
        fake.start.return_value = terminal
        fake.resume.return_value = terminal
        fake.status.return_value = full_status
        commands = (
            (
                ["start", "--config", "/tmp/config", "--format", "compact"],
                "start",
            ),
            (
                [
                    "resume",
                    "--config",
                    "/tmp/config",
                    "--run-id",
                    "run",
                    "--format",
                    "compact",
                ],
                "resume",
            ),
            (
                [
                    "status",
                    "--config",
                    "/tmp/config",
                    "--run-id",
                    "run",
                    "--format",
                    "compact",
                ],
                "status",
            ),
        )
        with mock.patch(
            "kernel_research.autorun.cli._controller", return_value=fake
        ):
            for argv, expected_command in commands:
                with self.subTest(argv=argv), redirect_stdout(
                    io.StringIO()
                ) as output:
                    self.assertEqual(autorun_cli.main(argv), 0)
                value = json.loads(output.getvalue())
                self.assertEqual(value["format"], "compact")
                self.assertEqual(value["command"], expected_command)
                self.assertNotIn("config", value["run"])
                self.assertNotIn("preflight", value["run"])
                self.assertEqual(
                    value["run"]["proposer_model"],
                    "deepseek/deepseek-v4-flash",
                )
                self.assertEqual(
                    value["run"]["proposer_reasoning_effort"], "max"
                )
                self.assertEqual(
                    value["run"]["proposer_declared_output_tokens"],
                    384_000,
                )
                self.assertEqual(
                    value["run"]["proposer_request_output_token_cap"],
                    384_000,
                )
                self.assertEqual(
                    value["iterations"][0]["result_summary"][
                        "relative_speedup_vs_accepted"
                    ],
                    0.9,
                )

        full_status["run"]["preflight"] = {"legacy": True}
        with (
            mock.patch(
                "kernel_research.autorun.cli._controller", return_value=fake
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(
                autorun_cli.main(
                    ["status", "--config", "/tmp/config", "--format", "compact"]
                ),
                0,
            )
        legacy = json.loads(output.getvalue())
        self.assertIsNone(legacy["run"]["proposer_reasoning_effort"])
        self.assertIsNone(
            legacy["run"]["proposer_declared_output_tokens"]
        )
        self.assertIsNone(
            legacy["run"]["proposer_request_output_token_cap"]
        )

    def test_policy_worker_error_bounding(self) -> None:
        result = ResearchPolicyResult(
            source=SEED,
            sha256=SEED_HASH,
            errors=tuple(
                CandidateError("E", f"error-{index}")
                for index in range(150)
            ),
        )
        bounded = _bounded_result(result)
        self.assertEqual(len(bounded.errors), 101)
        self.assertEqual(bounded.errors[-1].code, "POLICY_ERROR_LIMIT")

    def test_c500_probe_run_logic_with_fake_runtime(self) -> None:
        fake_tl = ModuleType("triton.language")
        fake_tl.constexpr = object()
        fake_triton = ModuleType("triton")
        fake_triton.__path__ = []  # type: ignore[attr-defined]

        class DecoratedKernel:
            def __getitem__(self, _grid):
                return lambda *_args, **_kwargs: None

        fake_triton.jit = lambda _function: DecoratedKernel()
        fake_torch = ModuleType("torch")
        fake_torch.float32 = object()
        with mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "triton": fake_triton,
                "triton.language": fake_tl,
            },
        ):
            sys.modules.pop("kernel_research.c500_probe", None)
            probe = importlib.import_module("kernel_research.c500_probe")

        input_tensor = object()
        output_tensor = object()
        accelerator = SimpleNamespace(
            is_available=lambda: True,
            synchronize=mock.Mock(),
        )
        probe.torch = SimpleNamespace(
            float32=object(),
            zeros=lambda *_args, **_kwargs: input_tensor,
            empty_like=lambda _value: output_tensor,
            ones_like=lambda _value: object(),
            equal=lambda *_args: True,
            cuda=accelerator,
            maca=SimpleNamespace(is_available=lambda: False),
        )
        launch = mock.MagicMock()
        launch.__getitem__.return_value = lambda *_args, **_kwargs: None
        probe._copy_probe = launch
        probe.run_probe("cuda:0")
        accelerator.synchronize.assert_called_once_with()

        probe.torch.equal = lambda *_args: False
        with self.assertRaisesRegex(RuntimeError, "incorrect"):
            probe.run_probe("cuda:0")


class KernelWrapperTests(unittest.TestCase):
    def test_seed_wrapper_validates_and_launches_with_fixed_contract(self) -> None:
        class Tensor:
            def __init__(self, shape, dtype, strides, device="cuda:0"):
                self.shape = shape
                self.dtype = dtype
                self._strides = strides
                self.device = device
                self.ndim = len(shape)

            def stride(self, index):
                return self._strides[index]

        class Kernel:
            def __init__(self):
                self.launches = []

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    self.launches.append((grid, args, kwargs))

                return launch

        kernel = Kernel()
        fake_torch = ModuleType("torch")
        fake_torch.Tensor = Tensor
        fake_torch.int8 = "int8"
        fake_torch.int32 = "int32"
        fake_torch.float32 = "float32"
        fake_torch.bfloat16 = "bfloat16"
        fake_tl = ModuleType("triton.language")
        fake_tl.constexpr = object()
        fake_triton = ModuleType("triton")
        fake_triton.__path__ = []  # type: ignore[attr-defined]
        fake_triton.cdiv = lambda value, block: (value + block - 1) // block
        fake_triton.jit = lambda _function: kernel
        with mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "triton": fake_triton,
                "triton.language": fake_tl,
            },
        ):
            spec = importlib.util.spec_from_file_location(
                "kernel_fixture", Path(__file__).resolve().parents[1] / "kernel.py"
            )
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)
        a = Tensor((256, 224), "int8", (224, 1))
        b = Tensor((4, 128, 224), "int8", (128 * 224, 224, 1))
        scale_a = Tensor((256,), "float32", (1,))
        scale_b = Tensor((4, 128), "float32", (128, 1))
        weights = Tensor((256,), "float32", (1,))
        token_ids = Tensor((256,), "int32", (1,))
        expert_ids = Tensor((2,), "int32", (1,))
        out = Tensor((256, 128), "bfloat16", (128, 1))
        module.run_kernel(
            a,
            b,
            scale_a,
            scale_b,
            weights,
            token_ids,
            expert_ids,
            8,
            out,
        )
        grid, _args, kwargs = kernel.launches[-1]
        self.assertEqual(grid, (2, 1))
        self.assertEqual(kwargs["BLOCK_SIZE_N"], 128)
        self.assertEqual(kwargs["BLOCK_SIZE_K"], 128)
        self.assertFalse(kwargs["COLUMN_MAJOR"])
        self.assertEqual(kwargs["num_warps"], 16)
        self.assertEqual(kwargs["num_stages"], 2)

        prefill_a = Tensor((32768, 7168), "int8", (7168, 1))
        prefill_b = Tensor(
            (256, 4096, 7168),
            "int8",
            (4096 * 7168, 7168, 1),
        )
        prefill_scale_a = Tensor((32768,), "float32", (1,))
        prefill_scale_b = Tensor((256, 4096), "float32", (4096, 1))
        prefill_weights = Tensor((32768,), "float32", (1,))
        prefill_token_ids = Tensor((32768,), "int32", (1,))
        prefill_expert_ids = Tensor((256,), "int32", (1,))
        prefill_out = Tensor((32768, 4096), "bfloat16", (4096, 1))
        module.run_kernel(
            prefill_a,
            prefill_b,
            prefill_scale_a,
            prefill_scale_b,
            prefill_weights,
            prefill_token_ids,
            prefill_expert_ids,
            8,
            prefill_out,
        )
        grid, _args, kwargs = kernel.launches[-1]
        self.assertEqual(grid, (32, 256))
        self.assertTrue(kwargs["COLUMN_MAJOR"])
        with self.assertRaisesRegex(ValueError, "topk"):
            module.run_kernel(
                a,
                b,
                scale_a,
                scale_b,
                weights,
                token_ids,
                expert_ids,
                7,
                out,
            )


if __name__ == "__main__":
    unittest.main()
