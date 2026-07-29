from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from kernel_research.autorun.controller import (
    ControllerSignal,
    DockerEvaluator,
    ResearchController,
)
from kernel_research.autorun.models import ProposalV1
from kernel_research.autorun.opencode import OpenCodeProposer
from kernel_research.autorun.proposal import (
    ProposalRequest,
    parse_opencode_ndjson,
)
from kernel_research.autorun.runtime import CommandResult, CommandRunner
from kernel_research.autorun.states import Stage
from kernel_research.autorun.store import ControllerStore
from kernel_research.evaluation import raw_evaluate, record_external_result
from kernel_research.history import HistoryStore
from kernel_research.research_policy import (
    validate_research_candidate,
    validate_research_candidate_bounded,
)

from test_autorun import (
    FULL_CASES,
    NoopRunner,
    SEED,
    SEED_HASH,
    StaticProposer,
    _baseline,
    _config,
    _proposal_value,
    _with_different_block_size_n,
)


class RecordingCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.killed: list[tuple[str, str]] = []

    def _kill_exact_container(
        self, docker_binary: Path, container_name: str
    ) -> None:
        self.killed.append((str(docker_binary), container_name))


class CommandRunnerHardeningTests(unittest.TestCase):
    def test_high_rate_output_is_bounded_and_killed_immediately(self) -> None:
        runner = RecordingCommandRunner()
        started = time.monotonic()
        result = runner.run(
            [
                sys.executable,
                "-c",
                "import os\n"
                "chunk=b'x'*65536\n"
                "while True: os.write(1, chunk)\n",
            ],
            input_text=None,
            timeout_sec=10,
            max_output_bytes=2048,
            container_name="exact-container",
            docker_binary=Path("/bin/echo"),
        )
        self.assertTrue(result.output_limited)
        self.assertFalse(result.timed_out)
        self.assertLess(time.monotonic() - started, 3)
        self.assertLessEqual(
            len(result.stdout.encode("utf-8"))
            + len(result.stderr.encode("utf-8")),
            2048,
        )
        self.assertEqual(runner.killed, [("/bin/echo", "exact-container")])

    def test_timeout_partial_utf8_and_stdin_do_not_deadlock(self) -> None:
        timed = CommandRunner().run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            input_text=None,
            timeout_sec=0.05,
            max_output_bytes=1024,
        )
        self.assertTrue(timed.timed_out)

        partial = CommandRunner().run(
            [
                sys.executable,
                "-c",
                "import os; os.write(1, b'good\\xf0\\x9f')",
            ],
            input_text=None,
            timeout_sec=2,
            max_output_bytes=1024,
        )
        self.assertEqual(partial.returncode, 0)
        self.assertIn("good", partial.stdout)
        self.assertIn("\ufffd", partial.stdout)

        echoed = CommandRunner().run(
            [
                sys.executable,
                "-c",
                "import sys; data=sys.stdin.buffer.read(); "
                "sys.stdout.buffer.write(data[::-1])",
            ],
            input_text="abcdef",
            timeout_sec=2,
            max_output_bytes=1024,
        )
        self.assertEqual(echoed.stdout, "fedcba")

        closed_pipes = CommandRunner().run(
            [
                sys.executable,
                "-c",
                "import os,time; os.close(1); os.close(2); time.sleep(30)",
            ],
            input_text=None,
            timeout_sec=0.05,
            max_output_bytes=1024,
        )
        self.assertTrue(closed_pipes.timed_out)


class PolicyHardeningTests(unittest.TestCase):
    def test_int64_product_must_flow_to_the_b_load(self) -> None:
        swapped = SEED.replace(
            "expert_id.to(tl.int64) * stride_be",
            "stride_be * expert_id.to(tl.int64)",
        )
        self.assertTrue(validate_research_candidate(swapped).valid)

        variants = (
            SEED.replace(
                "expert_id.to(tl.int64) * stride_be",
                "(expert_id * stride_be).to(tl.int64)",
            ),
            SEED.replace(
                "expert_id.to(tl.int64) * stride_be",
                "expert_id * stride_be",
            ).replace(
                "offs_m =",
                "unrelated = expert_id.to(tl.int64) * 1\n    offs_m =",
            ),
            SEED.replace(
                "b_expert_ptr = b_ptr + "
                "expert_id.to(tl.int64) * stride_be",
                "unused = expert_id.to(tl.int64) * stride_be\n"
                "    b_expert_ptr = b_ptr + expert_id * stride_be",
            ),
        )
        for source in variants:
            with self.subTest(source_hash=hashlib.sha256(source.encode()).hexdigest()):
                codes = {
                    error.code
                    for error in validate_research_candidate(source).errors
                }
                self.assertIn("INT64_B_BASE_REQUIRED", codes)

    def test_bounded_worker_handles_pathological_source_and_timeout(self) -> None:
        signature = (
            "def run_kernel(a, b_col_major, scale_a, scale_b, moe_weights, "
            "token_ids, expert_ids, topk, out):\n    value = "
        )
        source = signature + "+".join(["1"] * 100_000) + "\n"
        self.assertLess(len(source.encode("utf-8")), 256 * 1024)
        result = validate_research_candidate_bounded(source)
        self.assertFalse(result.valid)
        self.assertTrue(result.errors)

        with mock.patch(
            "kernel_research.research_policy.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["policy"], 5),
        ):
            timed = validate_research_candidate_bounded(SEED)
        self.assertEqual(timed.errors[0].code, "POLICY_RESOURCE_LIMIT")

        def excessive_output(*_args, stdout, stderr, **_kwargs):
            stdout.write(b"x" * (64 * 1024 + 1))
            stderr.write(b"diagnostic")
            return subprocess.CompletedProcess(["policy"], 0)

        with mock.patch(
            "kernel_research.research_policy.subprocess.run",
            side_effect=excessive_output,
        ):
            excessive = validate_research_candidate_bounded(SEED)
        self.assertEqual(
            excessive.errors[0].code, "POLICY_RESOURCE_LIMIT"
        )
        self.assertIn("output limit", excessive.errors[0].message)

    def test_evaluate_raw_rechecks_policy_before_executor(self) -> None:
        unsafe = SEED.replace(
            "expert_id.to(tl.int64) * stride_be",
            "expert_id * stride_be",
        )
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "kernel.py"
            candidate.write_text(unsafe, encoding="utf-8")
            with mock.patch(
                "kernel_research.evaluation.evaluate_isolated"
            ) as evaluate:
                result = raw_evaluate(
                    candidate, backend="mock", suite="smoke"
                )
        evaluate.assert_not_called()
        self.assertEqual(result["status"], "CONTRACT_ERROR")
        self.assertIn("INT64_B_BASE_REQUIRED", result["error"])


class FixedRunner:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = list(results)
        self.argv: list[tuple[str, ...]] = []
        self.removed: list[str] = []

    def run(self, argv, **_kwargs):
        self.argv.append(tuple(str(value) for value in argv))
        return self.results.pop(0)

    def remove_exact_container(
        self, _docker_binary: Path, container_name: str
    ) -> None:
        self.removed.append(container_name)


class AdapterAndCacheTests(unittest.TestCase):
    def test_ndjson_chunks_are_concatenated_without_separators(self) -> None:
        proposal_text = json.dumps(_proposal_value())
        chunks = [
            proposal_text[index : index + 7]
            for index in range(0, len(proposal_text), 7)
        ]
        raw = "\n".join(
            json.dumps({"type": "text", "part": {"text": chunk}})
            for chunk in chunks
        )
        proposal = parse_opencode_ndjson(
            raw, expected_parent_hash=SEED_HASH
        )
        self.assertEqual(proposal.candidate_hash, SEED_HASH)
        two_responses = "\n".join(
            json.dumps(
                {"type": "text", "part": {"text": proposal_text}}
            )
            for _ in range(2)
        )
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_opencode_ndjson(
                two_responses, expected_parent_hash=SEED_HASH
            )

    def test_opencode_adapter_uses_full_run_id_and_redacts_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            proposal_text = json.dumps(_proposal_value())
            stdout = json.dumps(
                {"type": "text", "part": {"text": proposal_text}}
            )
            runner = FixedRunner(
                [
                    CommandResult(
                        argv=(),
                        returncode=0,
                        stdout=stdout,
                        stderr="provider diagnostic secret",
                    )
                ]
            )
            run_id = "a" * 32
            adapter = OpenCodeProposer(
                config,
                run_id=run_id,
                iteration_index=1,
                run_dir=root / "run",
                runner=runner,  # type: ignore[arg-type]
            )
            request = ProposalRequest(
                parent_candidate_hash=SEED_HASH,
                accepted_kernel=SEED,
                program_markdown="Only kernel.py.",
                environment={},
                accepted_case_p50_us={},
                recent_experiments=(),
                session_feedback=(),
            )
            proposal = adapter.propose(request)
            self.assertEqual(proposal.candidate_hash, SEED_HASH)
            self.assertIn(run_id, adapter.container_name)
            self.assertNotIn(
                "secret",
                adapter.stderr_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "[REDACTED_SECRET]",
                adapter.stderr_path.read_text(encoding="utf-8"),
            )

    def test_evaluator_cache_is_namespaced_by_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            payload = json.dumps(
                {
                    "status": "PRECISION_FAILED",
                    "cases": [],
                    "environment": {},
                    "error": "fixture",
                }
            )
            runner = FixedRunner(
                [
                    CommandResult((), 0, payload, ""),
                    CommandResult((), 0, payload, ""),
                ]
            )
            evaluator = DockerEvaluator(
                config,
                controller_dir=config.controller_dir,
                runner=runner,  # type: ignore[arg-type]
            )
            candidate_a = root / "a.py"
            candidate_b = root / "b.py"
            candidate_a.write_text(SEED, encoding="utf-8")
            candidate_b.write_text(
                _with_different_block_size_n(SEED),
                encoding="utf-8",
            )
            for index, candidate in enumerate((candidate_a, candidate_b), 1):
                evaluator.evaluate(
                    candidate_path=candidate,
                    suite="smoke",
                    baseline_path=None,
                    run_id="b" * 32,
                    iteration_index=index,
                    stage="smoke",
                )
            first = " ".join(runner.argv[0])
            second = " ".join(runner.argv[1])
            hash_a = hashlib.sha256(candidate_a.read_bytes()).hexdigest()
            hash_b = hashlib.sha256(candidate_b.read_bytes()).hexdigest()
            self.assertIn(hash_a, first)
            self.assertNotIn(hash_b, first)
            self.assertIn(hash_b, second)
            self.assertNotIn(hash_a, second)
            manifests = list(
                (config.controller_dir / "cache-manifests").rglob("*.json")
            )
            self.assertEqual(len(manifests), 2)


class StoreAndRecoveryTests(unittest.TestCase):
    def test_illegal_transition_and_state_event_transaction_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            with ControllerStore(database) as store:
                store.connection.execute(
                    """
                    CREATE TRIGGER fail_run_created
                    BEFORE INSERT ON events
                    WHEN NEW.event = 'RUN_CREATED'
                    BEGIN
                        SELECT RAISE(ABORT, 'fixture event failure');
                    END
                    """
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    store.create_run(
                        run_id="rollback",
                        deadline_epoch=time.time() + 60,
                        config={},
                        initial_best_hash=SEED_HASH,
                    )
                count = store.connection.execute(
                    "SELECT COUNT(*) FROM runs WHERE id = 'rollback'"
                ).fetchone()[0]
                self.assertEqual(count, 0)
                store.connection.execute("DROP TRIGGER fail_run_created")
                store.create_run(
                    run_id="run",
                    deadline_epoch=time.time() + 60,
                    config={},
                    initial_best_hash=SEED_HASH,
                )
                iteration = store.create_iteration("run", 1, SEED_HASH)
                with self.assertRaisesRegex(ValueError, "illegal"):
                    store.update_iteration(
                        iteration["id"], stage=Stage.QUICK
                    )

    def test_v1_migration_rejects_unknown_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            with ControllerStore(database) as store:
                store.create_run(
                    run_id="run",
                    deadline_epoch=time.time() + 60,
                    config={},
                    initial_best_hash=SEED_HASH,
                )
            connection = sqlite3.connect(database)
            triggers = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
            for (name,) in triggers:
                connection.execute(f'DROP TRIGGER "{name}"')
            connection.execute(
                "UPDATE runs SET status = 'UNKNOWN' WHERE id = 'run'"
            )
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(RuntimeError, "unknown state"):
                ControllerStore(database)

    def test_controller_signal_stops_and_cleans_exact_container(self) -> None:
        class SignalEvaluator:
            def container_name(self, run_id, iteration_index, stage):
                return f"container-{run_id}-{iteration_index}-{stage}"

            def doctor(self, *, run_id):
                return {"status": "SUCCESS"}

            def evaluate(self, **_values):
                raise ControllerSignal(signal.SIGTERM)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, max_candidates=1)
            (config.repository_dir / "program.md").write_text(
                "Only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)
            proposal_source = _with_different_block_size_n(SEED)
            proposal = ProposalV1.from_value(
                _proposal_value(proposal_source),
                expected_parent_hash=SEED_HASH,
            )
            runner = NoopRunner()
            controller = ResearchController(
                config,
                evaluator=SignalEvaluator(),  # type: ignore[arg-type]
                runner=runner,  # type: ignore[arg-type]
                proposer_factory=lambda *_args: StaticProposer(proposal),
            )
            run_id = "c" * 32
            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id=run_id,
                    deadline_epoch=time.time() + 60,
                    config=config.redacted_dict(),
                    initial_best_hash=SEED_HASH,
                )
            result = controller._run_loop(run_id)
            self.assertEqual(result["status"], "STOPPED")
            self.assertEqual(
                runner.removed,
                [f"container-{run_id}-1-smoke"],
            )
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
                events = store.list_events(run_id)
            self.assertIsNone(iteration["active_container"])
            self.assertTrue(
                any(event["event"] == "CONTROLLER_SIGNAL" for event in events)
            )

    def test_gpu_acknowledgement_blocks_candidate_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(
                _config(Path(temporary)),
                acknowledge_gpu_passthrough_risk=False,
            )
            controller = ResearchController(config)
            with self.assertRaisesRegex(
                RuntimeError, "acknowledge_gpu_passthrough_risk"
            ):
                controller.start()


class ExternalResultValidationTests(unittest.TestCase):
    def _raw(self) -> dict:
        return {
            "schema_version": 1,
            "command": "evaluate-raw",
            "backend": "c500",
            "suite": "smoke",
            "candidate_hash": SEED_HASH,
            "baseline_candidate_hash": None,
            "status": "SUCCESS",
            "cases": [
                {
                    "case_id": name,
                    "status": "SUCCESS",
                    "matched_ratio": 1.0,
                    "latency_samples_us": [10.0] * 30,
                }
                for name in ("smoke_gate_up", "smoke_down")
            ],
            "environment": {},
            "error": None,
        }

    def test_negative_raw_result_paths_do_not_record(self) -> None:
        mutators = (
            lambda value: value.update(candidate_hash="0" * 64),
            lambda value: value.update(schema_version=2),
            lambda value: value.update(command="evaluate"),
            lambda value: value["cases"].reverse(),
            lambda value: value["cases"][0].update(
                latency_samples_us=[10.0] * 29
            ),
            lambda value: value["cases"][0].update(matched_ratio=0.5),
        )
        for index, mutate in enumerate(mutators):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory() as temporary:
                    raw = self._raw()
                    mutate(raw)
                    with self.assertRaises(ValueError):
                        record_external_result(
                            candidate_source=SEED,
                            result=raw,
                            backend="c500",
                            suite="smoke",
                            state_dir=temporary,
                            note="negative",
                        )
                    history_path = Path(temporary) / "history.sqlite3"
                    if history_path.exists():
                        with HistoryStore(
                            history_path, state_dir=temporary
                        ) as history:
                            self.assertEqual(history.list_experiments(), [])


if __name__ == "__main__":
    unittest.main()
