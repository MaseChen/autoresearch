from __future__ import annotations

from contextlib import redirect_stdout
import io
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from kernel_research.cli import (
    _apply_c500_promotion,
    _history_cases,
    evaluate_and_record,
    main,
)
from kernel_research.history import HistoryStore
from kernel_research.executor import (
    _close_queue,
    _doctor_child_entry,
    _stop_process,
    _wait_for_evaluation,
)


VALID_SOURCE = """\
def run_kernel(a, b_col_major, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk, out):
    out_marker = out
"""


def _measured_result(candidate_us: float, baseline_us: float) -> dict:
    cases = []
    for name in ("gate-decode", "gate-prefill", "down-decode", "down-prefill"):
        cases.append(
            {
                "case_id": name,
                "status": "SUCCESS",
                "matched_ratio": 1.0,
                "latency_samples_us": [candidate_us] * 30,
                "baseline_latency_samples_us": [baseline_us] * 30,
                "p50_us": candidate_us,
                "baseline_p50_us": baseline_us,
            }
        )
    return {
        "status": "SUCCESS",
        "eligible_for_promotion": False,
        "aggregate_score": None,
        "cases": cases,
        "environment": {},
    }


def _case_phase_timeout_fixture(progress) -> None:
    progress.put(
        {"phase": "case", "case_id": "fixture-case", "stage": "benchmark"}
    )
    time.sleep(60.0)


def _stubborn_grandchild_fixture(pid_output, ready_path: str) -> None:
    os.setsid()
    source = (
        "import pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({ready_path!r}).write_text('ready'); "
        "time.sleep(60.0)"
    )
    grandchild = subprocess.Popen([sys.executable, "-c", source])
    deadline = time.monotonic() + 5.0
    while not Path(ready_path).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    pid_output.put(grandchild.pid)
    time.sleep(60.0)


def _orphaning_worker_fixture(ready_path: str, pid_path: str) -> None:
    os.setsid()
    source = (
        "import pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({ready_path!r}).write_text('ready'); "
        "time.sleep(60.0)"
    )
    grandchild = subprocess.Popen([sys.executable, "-c", source])
    deadline = time.monotonic() + 5.0
    while not Path(ready_path).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    Path(pid_path).write_text(str(grandchild.pid), encoding="ascii")
    os._exit(17)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class IsolatedEvaluationTests(unittest.TestCase):
    def _write(self, directory: Path, name: str, source: str) -> Path:
        path = directory / name
        path.write_text(source, encoding="utf-8")
        return path

    def test_contract_crash_timeout_and_recovery_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._write(root, "valid.py", VALID_SOURCE)
            invalid = self._write(root, "invalid.py", "def run_kernel(:\n")
            crash_fixture = self._write(
                root, "crash.py", VALID_SOURCE + "\nFIXTURE = 'crash'\n"
            )
            timeout_fixture = self._write(
                root, "timeout.py", VALID_SOURCE + "\nFIXTURE = 'timeout'\n"
            )

            state = root / "state"
            first = evaluate_and_record(
                valid, backend="mock", suite="smoke", state_dir=state, note="one"
            )
            contract = evaluate_and_record(
                invalid, backend="mock", suite="smoke", state_dir=state, note="two"
            )
            crashed = evaluate_and_record(
                crash_fixture,
                backend="mock",
                suite="smoke",
                state_dir=state,
                note="three",
                test_fault="crash",
            )
            timed_out = evaluate_and_record(
                timeout_fixture,
                backend="mock",
                suite="smoke",
                state_dir=state,
                note="four",
                timeout_sec=0.05,
                test_fault="timeout",
            )
            recovered = evaluate_and_record(
                valid, backend="mock", suite="smoke", state_dir=state, note="five"
            )

            self.assertEqual(first["status"], "MOCK_VALIDATED")
            self.assertFalse(first["eligible_for_promotion"])
            self.assertEqual(first["cases"], [])
            self.assertEqual(contract["status"], "CONTRACT_ERROR")
            self.assertEqual(crashed["status"], "CRASH")
            self.assertEqual(timed_out["status"], "TIMEOUT")
            self.assertEqual(timed_out["environment"]["timeout_mode"], "global-override")
            self.assertEqual(timed_out["environment"]["timeout_phase"], "mock")
            self.assertEqual(recovered["status"], "MOCK_VALIDATED")

            with HistoryStore(state / "history.sqlite3", state_dir=state) as history:
                records = history.list_experiments()

            self.assertEqual(len(records), 5)
            self.assertEqual([record.status for record in records], [
                "MOCK_VALIDATED",
                "CONTRACT_ERROR",
                "CRASH",
                "TIMEOUT",
                "MOCK_VALIDATED",
            ])
            self.assertTrue(records[-1].duplicate)
            self.assertEqual([record.note for record in records], [
                "one", "two", "three", "four", "five"
            ])
            self.assertEqual(
                len(list((state / "artifacts").glob("*.py"))), 4
            )

        # Give multiprocessing resource trackers a moment to settle, then make
        # sure the framework did not leave a worker behind.
        time.sleep(0.05)
        leaked = [
            child
            for child in multiprocessing.active_children()
            if child.name.startswith("kernel-research-")
        ]
        self.assertEqual(leaked, [])

    def test_c500_watchdog_enforces_a_case_deadline_and_names_the_case(self) -> None:
        context = multiprocessing.get_context("spawn")
        progress = context.Queue(maxsize=4)
        process = context.Process(
            target=_case_phase_timeout_fixture,
            args=(progress,),
            name="kernel-research-watchdog-fixture",
        )
        process.start()
        try:
            with (
                mock.patch(
                    "kernel_research.executor.DEFAULT_C500_COMPILE_TIMEOUT_SEC",
                    2.0,
                ),
                mock.patch(
                    "kernel_research.executor.DEFAULT_C500_CASE_TIMEOUT_SEC",
                    0.05,
                ),
            ):
                completed, watchdog = _wait_for_evaluation(
                    process,
                    progress,
                    backend="c500",
                    explicit_timeout_sec=None,
                )
            self.assertFalse(completed)
            self.assertEqual(watchdog["mode"], "c500-phased")
            self.assertEqual(watchdog["phase"], "case")
            self.assertEqual(watchdog["case_id"], "fixture-case")
            self.assertEqual(watchdog["stage"], "benchmark")
        finally:
            _stop_process(process)
            _close_queue(progress)

    @unittest.skipUnless(
        hasattr(os, "setsid") and hasattr(os, "killpg"),
        "POSIX process groups are required",
    )
    def test_worker_cleanup_kills_a_sigterm_ignoring_grandchild(self) -> None:
        context = multiprocessing.get_context("spawn")
        pid_output = context.Queue(maxsize=1)
        grandchild_pid: int | None = None
        with tempfile.TemporaryDirectory() as temporary:
            ready_path = str(Path(temporary) / "grandchild-ready")
            process = context.Process(
                target=_stubborn_grandchild_fixture,
                args=(pid_output, ready_path),
                name="kernel-research-process-group-fixture",
            )
            process.start()
            try:
                grandchild_pid = int(pid_output.get(timeout=10.0))
                self.assertTrue(Path(ready_path).exists())
                self.assertTrue(_pid_exists(grandchild_pid))

                _stop_process(process)
                deadline = time.monotonic() + 5.0
                while _pid_exists(grandchild_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(
                    _pid_exists(grandchild_pid),
                    "compiler-like grandchild survived worker cleanup",
                )
            finally:
                _stop_process(process)
                if grandchild_pid is not None and _pid_exists(grandchild_pid):
                    try:
                        os.kill(grandchild_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                _close_queue(pid_output)

    def test_direct_worker_without_owned_group_reaches_kill_fallback(self) -> None:
        class StubbornProcess:
            pid = None

            def __init__(self) -> None:
                self.alive = True
                self.terminated = False
                self.killed = False
                self.joins: list[float | None] = []

            def is_alive(self) -> bool:
                return self.alive

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.killed = True
                self.alive = False

            def join(self, timeout=None) -> None:
                self.joins.append(timeout)

        process = StubbornProcess()
        _stop_process(process)  # type: ignore[arg-type]
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertFalse(process.alive)
        self.assertEqual(process.joins, [2.0, 2.0])

    @unittest.skipUnless(
        hasattr(os, "setsid") and hasattr(os, "killpg"),
        "POSIX process groups are required",
    )
    def test_cached_ownership_cleans_orphans_after_worker_exit(self) -> None:
        context = multiprocessing.get_context("spawn")
        grandchild_pid: int | None = None
        with tempfile.TemporaryDirectory() as temporary:
            ready_path = str(Path(temporary) / "orphan-ready")
            pid_path = Path(temporary) / "orphan-pid"
            process = context.Process(
                target=_orphaning_worker_fixture,
                args=(ready_path, str(pid_path)),
                name="kernel-research-orphan-fixture",
            )
            process.start()
            try:
                process.join(timeout=10.0)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 17)
                grandchild_pid = int(pid_path.read_text(encoding="ascii"))
                self.assertTrue(_pid_exists(grandchild_pid))

                _stop_process(process, owned_group_id=process.pid)
                deadline = time.monotonic() + 5.0
                while _pid_exists(grandchild_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(
                    _pid_exists(grandchild_pid),
                    "orphaned compiler-like process survived cached-PGID cleanup",
                )
            finally:
                _stop_process(process, owned_group_id=process.pid)
                if grandchild_pid is not None and _pid_exists(grandchild_pid):
                    try:
                        os.kill(grandchild_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_isolated_c500_doctor_requests_the_compile_probe(self) -> None:
        output: queue.Queue[dict] = queue.Queue()
        ownership = mock.Mock()
        doctor_result = mock.Mock()
        doctor_result.to_dict.return_value = {"status": "SUCCESS"}
        with (
            mock.patch(
                "kernel_research.executor._isolate_process_group",
                return_value=123,
            ),
            mock.patch(
                "kernel_research.executor._redirect_child_stdout"
            ),
            mock.patch(
                "kernel_research.backends.C500Backend.doctor",
                return_value=doctor_result,
            ) as doctor,
        ):
            _doctor_child_entry(  # type: ignore[arg-type]
                output, ownership, "c500"
            )

        doctor.assert_called_once_with(compile_probe=True)
        ownership.send.assert_called_once_with(123)
        ownership.close.assert_called_once_with()
        self.assertEqual(output.get_nowait(), {"status": "SUCCESS"})

    def test_cli_evaluate_and_history_emit_machine_readable_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self._write(root, "kernel.py", VALID_SOURCE)
            state = root / ".autoresearch"

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "evaluate",
                        "--backend",
                        "mock",
                        "--suite",
                        "smoke",
                        "--candidate",
                        str(candidate),
                        "--state-dir",
                        str(state),
                        "--note",
                        "integration",
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["status"], "MOCK_VALIDATED")
            self.assertFalse(payload["eligible_for_promotion"])
            self.assertEqual(payload["note"], "integration")

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    ["history", "--format", "json", "--state-dir", str(state)]
                )
            rows = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["note"], "integration")

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["doctor", "--backend", "mock"])
            doctor = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(doctor["status"], "MOCK_VALIDATED")
            self.assertEqual(doctor["command"], "doctor")

    def test_non_full_c500_success_is_validation_only(self) -> None:
        result = {
            "status": "SUCCESS",
            "eligible_for_promotion": False,
            "aggregate_score": 123.0,
            "cases": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            with HistoryStore(state / "history.sqlite3", state_dir=state) as history:
                _apply_c500_promotion(
                    result,
                    history=history,
                    best=None,
                    candidate_hash="fixture",
                    suite="quick",
                )

        self.assertFalse(result["eligible_for_promotion"])
        self.assertIsNone(result["aggregate_score"])
        self.assertEqual(result["promotion"]["phase"], "validation")

    def test_c500_baseline_primary_and_confirmation_form_an_accepted_chain(self) -> None:
        baseline_source = VALID_SOURCE + "\nBASELINE = True\n"
        candidate_source = VALID_SOURCE + "\nCANDIDATE = True\n"
        candidate_hash = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            with HistoryStore(state / "history.sqlite3", state_dir=state) as history:
                baseline_result = _measured_result(100.0, 100.0)
                baseline_result["eligible_for_promotion"] = True
                baseline_result["aggregate_score"] = 1.0
                baseline_result["promotion"] = {
                    "phase": "baseline",
                    "reason": "initial_correct_baseline",
                }
                baseline_record = history.record_experiment(
                    candidate_source=baseline_source,
                    status="SUCCESS",
                    backend="c500",
                    suite="full",
                    promotable=True,
                    aggregate_score=1.0,
                    case_measurements=_history_cases(baseline_result),
                    result=baseline_result,
                )

                primary = _measured_result(99.0, 100.0)
                _apply_c500_promotion(
                    primary,
                    history=history,
                    best=baseline_record,
                    candidate_hash=candidate_hash,
                    suite="full",
                )
                self.assertFalse(primary["eligible_for_promotion"])
                self.assertEqual(primary["promotion"]["phase"], "primary")
                primary_record = history.record_experiment(
                    candidate_source=candidate_source,
                    candidate_hash=candidate_hash,
                    status="SUCCESS",
                    backend="c500",
                    suite="full",
                    promotable=False,
                    aggregate_score=primary["aggregate_score"],
                    case_measurements=_history_cases(primary),
                    result=primary,
                )
                self.assertEqual(
                    primary_record.case_measurements[0].baseline_samples,
                    (100.0,) * 30,
                )

                confirmation = _measured_result(198.0, 200.0)
                _apply_c500_promotion(
                    confirmation,
                    history=history,
                    best=baseline_record,
                    candidate_hash=candidate_hash,
                    suite="full",
                )
                self.assertTrue(confirmation["eligible_for_promotion"])
                self.assertEqual(
                    confirmation["promotion"]["phase"], "confirmation"
                )
                confirmed_record = history.record_experiment(
                    candidate_source=candidate_source,
                    candidate_hash=candidate_hash,
                    status="SUCCESS",
                    backend="c500",
                    suite="full",
                    promotable=True,
                    aggregate_score=confirmation["aggregate_score"],
                    case_measurements=_history_cases(confirmation),
                    result=confirmation,
                )

                best = history.get_best(backend="c500", suite="full")
                self.assertIsNotNone(best)
                self.assertEqual(best.id, confirmed_record.id)
                self.assertGreater(best.aggregate_score or 0.0, 1.0)

    def test_failed_confirmation_consumes_primary_before_a_retry(self) -> None:
        baseline_source = VALID_SOURCE + "\nBASELINE = 'consume-test'\n"
        candidate_source = VALID_SOURCE + "\nCANDIDATE = 'consume-test'\n"
        candidate_hash = hashlib.sha256(candidate_source.encode()).hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            with HistoryStore(state / "history.sqlite3", state_dir=state) as history:
                baseline_result = _measured_result(100.0, 100.0)
                baseline_record = history.record_experiment(
                    candidate_source=baseline_source,
                    status="SUCCESS",
                    backend="c500",
                    suite="full",
                    promotable=True,
                    aggregate_score=1.0,
                    result=baseline_result,
                )

                primary = _measured_result(99.0, 100.0)
                _apply_c500_promotion(
                    primary,
                    history=history,
                    best=baseline_record,
                    candidate_hash=candidate_hash,
                    suite="full",
                )
                primary_record = history.record_experiment(
                    candidate_source=candidate_source,
                    candidate_hash=candidate_hash,
                    status="SUCCESS",
                    backend="c500",
                    suite="full",
                    aggregate_score=primary["aggregate_score"],
                    result=primary,
                )

                failed = _measured_result(105.0, 100.0)
                _apply_c500_promotion(
                    failed,
                    history=history,
                    best=baseline_record,
                    candidate_hash=candidate_hash,
                    suite="full",
                )
                self.assertEqual(failed["promotion"]["phase"], "confirmation")
                self.assertFalse(failed["eligible_for_promotion"])
                history.record_experiment(
                    candidate_source=candidate_source,
                    candidate_hash=candidate_hash,
                    status="SUCCESS",
                    backend="c500",
                    suite="full",
                    aggregate_score=failed["aggregate_score"],
                    result=failed,
                )

                third = _measured_result(99.0, 100.0)
                _apply_c500_promotion(
                    third,
                    history=history,
                    best=baseline_record,
                    candidate_hash=candidate_hash,
                    suite="full",
                )

                self.assertEqual(primary_record.result["promotion"]["phase"], "primary")
                self.assertEqual(third["promotion"]["phase"], "primary")
                self.assertFalse(third["eligible_for_promotion"])


if __name__ == "__main__":
    unittest.main()
