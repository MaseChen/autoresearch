from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
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
from kernel_research.autorun.model_catalog import OPENCODE_MODEL_SPECS
from kernel_research.autorun.models import ProposalV1
from kernel_research.autorun.opencode import OpenCodeProposer
from kernel_research.autorun.proposal import (
    APPEND_FINAL_OBJECT_BRACE,
    DROP_EXACT_TRAILING_QUOTE_BRACE,
    ProposalFormatError,
    ProposalRequest,
    ProposerConstraintRetryExhaustedError,
    ProposerFormatRetryExhaustedError,
    ProposerOutputTokenLimitError,
    ProposerStepLimitError,
    build_prompt,
    parse_opencode_ndjson,
    parse_opencode_ndjson_result,
)
from kernel_research.autorun.runtime import CommandResult, CommandRunner
from kernel_research.autorun.states import Stage
from kernel_research.autorun.store import (
    ControllerStore,
    _V2_TO_V3_MIGRATION_CAPABILITY,
)
from kernel_research.autorun.summary import (
    compact_status_payload,
    feedback_for_iteration,
    summarize_result,
)
from kernel_research.constants import (
    MAX_FEEDBACK_CANDIDATES,
    MAX_FEEDBACK_ERROR_CHARS,
)
from kernel_research.evaluation import raw_evaluate, record_external_result
from kernel_research.history import HistoryStore
from kernel_research.research_policy import (
    validate_research_candidate,
    validate_research_candidate_bounded,
)

from test_autorun import (
    FakeEvaluator,
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

STEP_LIMIT_FIXTURE = (
    Path(__file__).with_name("fixtures") / "opencode_step_limit_005.ndjson"
)
OUTPUT_LIMIT_FIXTURE = (
    Path(__file__).with_name("fixtures")
    / "opencode_flash_output_limit.ndjson"
)


def _proposal_ndjson(
    text: str,
    *,
    reason: str = "stop",
    reasoning_tokens: int = 18_963,
    output_tokens: int = 3_844,
) -> str:
    return "\n".join(
        json.dumps(event)
        for event in (
            {"type": "step_start", "part": {"type": "step-start"}},
            {"type": "text", "part": {"text": text}},
            {
                "type": "step_finish",
                "part": {
                    "reason": reason,
                    "tokens": {
                        "total": reasoning_tokens + output_tokens + 11_830,
                        "input": 11_830,
                        "reasoning": reasoning_tokens,
                        "output": output_tokens,
                    },
                },
            },
        )
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
    def test_triton_jit_rejects_plain_module_globals_only_inside_kernel(
        self,
    ) -> None:
        self.assertTrue(validate_research_candidate(SEED).valid)
        unsafe = SEED.replace(
            "offs_m = pid_m * 128 + tl.arange(0, 128)",
            "offs_m = pid_m * _EXPERT_TILE_ROWS + "
            "tl.arange(0, _EXPERT_TILE_ROWS)",
        )
        result = validate_research_candidate(unsafe)
        self.assertFalse(result.valid)
        errors = [
            error
            for error in result.errors
            if error.code == "TRITON_NON_CONSTEXPR_GLOBAL"
        ]
        self.assertEqual(len(errors), 1)
        self.assertIn("_EXPERT_TILE_ROWS", errors[0].message)

        constexpr = unsafe.replace(
            "_EXPERT_TILE_ROWS = 128",
            "_EXPERT_TILE_ROWS: tl.constexpr = 128",
        )
        self.assertTrue(validate_research_candidate(constexpr).valid)

    def test_c500_num_warps_is_literal_and_bounded(self) -> None:
        pattern = re.compile(r"(?m)^(?P<indent>\s*)num_warps=[^,\n]+,$")

        def with_num_warps(value: str) -> str:
            updated, count = pattern.subn(
                rf"\g<indent>num_warps={value},", SEED
            )
            self.assertEqual(count, 1)
            return updated

        self.assertTrue(validate_research_candidate(with_num_warps("8")).valid)
        for value in ("0", "3", "32", "topk"):
            with self.subTest(num_warps=value):
                result = validate_research_candidate(
                    with_num_warps(value)
                )
                self.assertFalse(result.valid)
                self.assertIn(
                    "C500_NUM_WARPS_INVALID",
                    {error.code for error in result.errors},
                )

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
        unsafe_variants = (
            (
                SEED.replace(
                    "expert_id.to(tl.int64) * stride_be",
                    "expert_id * stride_be",
                ),
                "INT64_B_BASE_REQUIRED",
            ),
            (
                re.sub(
                    r"(?m)^(?P<indent>\s*)num_warps=[^,\n]+,$",
                    r"\g<indent>num_warps=32,",
                    SEED,
                ),
                "C500_NUM_WARPS_INVALID",
            ),
            (
                SEED.replace(
                    "offs_m = pid_m * 128 + tl.arange(0, 128)",
                    "offs_m = pid_m * _EXPERT_TILE_ROWS + "
                    "tl.arange(0, _EXPERT_TILE_ROWS)",
                ),
                "TRITON_NON_CONSTEXPR_GLOBAL",
            ),
        )
        for unsafe, error_code in unsafe_variants:
            with self.subTest(error_code=error_code):
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
                self.assertIn(error_code, result["error"])


class FixedRunner:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = list(results)
        self.argv: list[tuple[str, ...]] = []
        self.kwargs: list[dict[str, object]] = []
        self.removed: list[str] = []

    def run(self, argv, **kwargs):
        self.argv.append(tuple(str(value) for value in argv))
        self.kwargs.append(dict(kwargs))
        return self.results.pop(0)

    def remove_exact_container(
        self, _docker_binary: Path, container_name: str
    ) -> None:
        self.removed.append(container_name)


class AdapterAndCacheTests(unittest.TestCase):
    def test_bounded_transport_recovery_is_strict_and_semantic(self) -> None:
        valid_text = json.dumps(_proposal_value(), separators=(",", ":"))
        strict = parse_opencode_ndjson_result(
            _proposal_ndjson(valid_text),
            expected_parent_hash=SEED_HASH,
        )
        self.assertIsNone(strict.transport_recovery)

        cases = (
            (
                valid_text[:-1],
                APPEND_FINAL_OBJECT_BRACE,
                18_963,
                3_844,
            ),
            (
                valid_text + '"}',
                DROP_EXACT_TRAILING_QUOTE_BRACE,
                29_537,
                3_480,
            ),
        )
        for text, expected_recovery, reasoning, output in cases:
            with self.subTest(recovery=expected_recovery):
                result = parse_opencode_ndjson_result(
                    _proposal_ndjson(
                        text,
                        reasoning_tokens=reasoning,
                        output_tokens=output,
                    ),
                    expected_parent_hash=SEED_HASH,
                )
                self.assertEqual(
                    result.transport_recovery, expected_recovery
                )
                self.assertEqual(result.proposal.kernel_source, SEED)
                self.assertEqual(result.proposal.candidate_hash, SEED_HASH)

        semantic_errors = []
        invalid_parent = _proposal_value()
        invalid_parent["parent_candidate_hash"] = "0" * 64
        semantic_errors.append((invalid_parent, "parent_candidate_hash"))
        unknown_field = _proposal_value()
        unknown_field["unknown"] = True
        semantic_errors.append((unknown_field, "unknown fields"))
        empty_source = _proposal_value()
        empty_source["kernel_source"] = ""
        semantic_errors.append((empty_source, "kernel_source"))
        for value, message in semantic_errors:
            with self.subTest(semantic=message), self.assertRaisesRegex(
                ValueError, message
            ):
                parse_opencode_ndjson_result(
                    _proposal_ndjson(
                        json.dumps(value, separators=(",", ":"))[:-1]
                    ),
                    expected_parent_hash=SEED_HASH,
                )

    def test_transport_recovery_rejects_non_eof_and_unsafe_contexts(
        self,
    ) -> None:
        valid_text = json.dumps(_proposal_value(), separators=(",", ":"))
        unsafe = (
            _proposal_ndjson(valid_text[:-1], reason="length"),
            _proposal_ndjson(f"```json\n{valid_text[:-1]}\n```"),
            _proposal_ndjson(valid_text + "x"),
            json.dumps({"type": "text", "part": {"text": valid_text[:-1]}}),
            _proposal_ndjson(valid_text.replace('"hypothesis":', '"hypothesis"')),
        )
        expected_errors = (
            ProposerOutputTokenLimitError,
            ProposalFormatError,
            ProposalFormatError,
            ProposalFormatError,
            ProposalFormatError,
        )
        for index, (raw, error_type) in enumerate(
            zip(unsafe, expected_errors), 1
        ):
            with self.subTest(index=index), self.assertRaises(error_type):
                parse_opencode_ndjson_result(
                    raw,
                    expected_parent_hash=SEED_HASH,
                    configured_output_token_cap=384_000,
                )

        earlier_length = json.dumps(
            {
                "type": "step_finish",
                "part": {
                    "reason": "length",
                    "tokens": {"reasoning": 32_000, "output": 0},
                },
            }
        ) + "\n" + _proposal_ndjson(valid_text[:-1])
        result = parse_opencode_ndjson_result(
            earlier_length,
            expected_parent_hash=SEED_HASH,
            configured_output_token_cap=384_000,
        )
        self.assertEqual(
            result.transport_recovery, APPEND_FINAL_OBJECT_BRACE
        )

    def test_adapter_recovers_first_or_retry_attempt_without_mutating_raw(
        self,
    ) -> None:
        valid_text = json.dumps(_proposal_value(), separators=(",", ":"))
        recovered_outputs = (
            (
                [_proposal_ndjson(valid_text[:-1])],
                APPEND_FINAL_OBJECT_BRACE,
                1,
            ),
            (
                [
                    json.dumps(
                        {"type": "text", "part": {"text": "not JSON"}}
                    ),
                    _proposal_ndjson(
                        valid_text + '"}',
                        reasoning_tokens=29_537,
                        output_tokens=3_480,
                    ),
                ],
                DROP_EXACT_TRAILING_QUOTE_BRACE,
                2,
            ),
        )
        for index, (outputs, recovery, call_count) in enumerate(
            recovered_outputs, 1
        ):
            with self.subTest(
                recovery=recovery
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = _config(root)
                runner = FixedRunner(
                    [CommandResult((), 0, output, "") for output in outputs]
                )
                adapter = OpenCodeProposer(
                    config,
                    run_id=str(index) * 32,
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
                self.assertEqual(len(runner.argv), call_count)
                self.assertEqual(
                    adapter.attempts[-1]["transport_recovery"], recovery
                )
                self.assertEqual(adapter.attempts[-1]["outcome"], "SUCCESS")
                self.assertEqual(
                    adapter.raw_path.read_text(encoding="utf-8"),
                    outputs[-1],
                )

    def test_controller_audits_recovery_before_proposal_only_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            (config.repository_dir / "program.md").write_text(
                "Only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)
            source = _with_different_block_size_n(SEED)
            text = json.dumps(
                _proposal_value(source), separators=(",", ":")
            )[:-1]
            raw = _proposal_ndjson(text)
            runner = FixedRunner([CommandResult((), 0, raw, "")])
            evaluator = FakeEvaluator()
            controller = ResearchController(
                config,
                evaluator=evaluator,
                proposer_factory=lambda run_id, index, run_dir: OpenCodeProposer(
                    config,
                    run_id=run_id,
                    iteration_index=index,
                    run_dir=run_dir,
                    runner=runner,  # type: ignore[arg-type]
                ),
            )
            run_id = "transport-recovery-run"
            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id=run_id,
                    deadline_epoch=time.time() + 3600,
                    config=config.redacted_dict(),
                    initial_best_hash=SEED_HASH,
                )
            result = controller._run_loop(run_id, proposal_only=True)
            self.assertEqual(result["status"], "PROPOSAL_READY")
            self.assertEqual(evaluator.stages, [])
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
                events = store.list_events(run_id)
            self.assertEqual(iteration["outcome"], "PROPOSAL_VALIDATED")
            recovery = next(
                event
                for event in events
                if event["event"] == "PROPOSER_TRANSPORT_RECOVERY"
            )
            expected_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            self.assertEqual(
                recovery["details"]["candidate_hash"], expected_hash
            )
            self.assertEqual(
                recovery["details"]["attempts"],
                [
                    {
                        "attempt": 1,
                        "transport_recovery": APPEND_FINAL_OBJECT_BRACE,
                        "raw_output_path": iteration["raw_output_path"],
                    }
                ],
            )
            self.assertEqual(
                Path(iteration["raw_output_path"]).read_text(encoding="utf-8"),
                raw,
            )

    def test_opencode_adapter_uses_selected_model_without_fallback(self) -> None:
        request = ProposalRequest(
            parent_candidate_hash=SEED_HASH,
            accepted_kernel=SEED,
            program_markdown="Only kernel.py.",
            environment={},
            accepted_case_p50_us={},
            recent_experiments=(),
            session_feedback=(),
        )
        valid = json.dumps(
            {
                "type": "text",
                "part": {"text": json.dumps(_proposal_value())},
            }
        )
        for index, qualified_id in enumerate(OPENCODE_MODEL_SPECS, 1):
            with self.subTest(qualified_id=qualified_id), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = replace(
                    _config(root), opencode_model=qualified_id
                )
                runner = FixedRunner([CommandResult((), 0, valid, "")])
                adapter = OpenCodeProposer(
                    config,
                    run_id=f"{index}" * 32,
                    iteration_index=1,
                    run_dir=root / "run",
                    runner=runner,  # type: ignore[arg-type]
                )
                self.assertEqual(
                    adapter.propose(request).candidate_hash, SEED_HASH
                )
                argv = runner.argv[0]
                self.assertEqual(
                    argv[argv.index("--model") + 1], qualified_id
                )

                malformed = json.dumps(
                    {"type": "text", "part": {"text": "not JSON"}}
                )
                retry_runner = FixedRunner(
                    [
                        CommandResult((), 0, malformed, ""),
                        CommandResult((), 0, valid, ""),
                    ]
                )
                retry_adapter = OpenCodeProposer(
                    config,
                    run_id=f"r{index}" * 16,
                    iteration_index=2,
                    run_dir=root / "retry-run",
                    runner=retry_runner,  # type: ignore[arg-type]
                )
                self.assertEqual(
                    retry_adapter.propose(request).candidate_hash,
                    SEED_HASH,
                )
                self.assertEqual(len(retry_runner.argv), 2)
                self.assertTrue(
                    all(
                        argv[argv.index("--model") + 1] == qualified_id
                        for argv in retry_runner.argv
                    )
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flash = "deepseek/deepseek-v4-flash"
            config = replace(_config(root), opencode_model=flash)
            runner = FixedRunner(
                [CommandResult((), 1, "", "transient network failure")]
            )
            adapter = OpenCodeProposer(
                config,
                run_id="f" * 32,
                iteration_index=1,
                run_dir=root / "run",
                runner=runner,  # type: ignore[arg-type]
            )
            with self.assertRaisesRegex(
                RuntimeError, "OpenCode container exited with code 1"
            ):
                adapter.propose(request)
            self.assertEqual(len(runner.argv), 1)
            argv = runner.argv[0]
            self.assertEqual(argv[argv.index("--model") + 1], flash)

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

    def test_real_step_limit_ndjson_and_heading_variants_are_classified(
        self,
    ) -> None:
        fixture_events = [
            json.loads(line)
            for line in STEP_LIMIT_FIXTURE.read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        text_event = next(
            event for event in fixture_events if event["type"] == "text"
        )
        details = "\n\nSanitized OpenCode fallback details."
        headings = (
            "Maximum steps for this agent have been reached",
            "CRITICAL – MAXIMUM STEPS REACHED",
            "CRITICAL – Maximum steps reached",
        )
        for heading in headings:
            with self.subTest(heading=heading):
                events = json.loads(json.dumps(fixture_events))
                candidate_text_event = next(
                    event for event in events if event["type"] == "text"
                )
                candidate_text_event["part"]["text"] = heading + details
                raw = "\n".join(
                    json.dumps(event, ensure_ascii=False) for event in events
                )
                with self.assertRaisesRegex(
                    ProposerStepLimitError,
                    r"^PROPOSER_STEP_LIMIT:.*3-step proposal budget$",
                ):
                    parse_opencode_ndjson(
                        raw, expected_parent_hash=SEED_HASH
                    )
        self.assertIn(
            "Maximum steps for this agent have been reached",
            text_event["part"]["text"],
        )

    def test_step_limit_classification_preserves_strict_json_errors(
        self,
    ) -> None:
        raw = json.dumps(
            {"type": "text", "part": {"text": "not JSON"}}
        )
        with self.assertRaises(ValueError) as caught:
            parse_opencode_ndjson(raw, expected_parent_hash=SEED_HASH)
        self.assertNotIsInstance(caught.exception, ProposerStepLimitError)
        self.assertIn("valid JSON", str(caught.exception))

        source = (
            SEED
            + "\n# Maximum steps for this agent have been reached in a comment.\n"
        )
        value = _proposal_value(source)
        proposal_raw = json.dumps(
            {
                "type": "text",
                "part": {"text": json.dumps(value)},
            }
        )
        proposal = parse_opencode_ndjson(
            proposal_raw, expected_parent_hash=SEED_HASH
        )
        self.assertEqual(proposal.kernel_source, source)

    def test_flash_output_token_limit_is_classified_without_format_retry(
        self,
    ) -> None:
        raw = OUTPUT_LIMIT_FIXTURE.read_text(encoding="utf-8")
        with self.assertRaisesRegex(
            ProposerOutputTokenLimitError,
            (
                r"^PROPOSER_OUTPUT_TOKEN_LIMIT:.*reasoning=32000, output=0, "
                r"total=43211, configured_cap=384000\)$"
            ),
        ):
            parse_opencode_ndjson(
                raw,
                expected_parent_hash=SEED_HASH,
                configured_output_token_cap=384_000,
            )

        for token_limit in (65_536, 384_000):
            with self.subTest(token_limit=token_limit):
                value = json.loads(raw.splitlines()[-1])
                value["part"]["tokens"].update(
                    {
                        "total": token_limit + 100,
                        "reasoning": token_limit,
                        "output": 0,
                    }
                )
                with self.assertRaisesRegex(
                    ProposerOutputTokenLimitError,
                    f"reasoning={token_limit}, output=0",
                ):
                    parse_opencode_ndjson(
                        json.dumps(value),
                        expected_parent_hash=SEED_HASH,
                        configured_output_token_cap=384_000,
                    )

        partial = raw.replace(
            '{"type":"step_finish"',
            '{"type":"text","part":{"text":"{\\\"schema_version\\\":1"}}\n'
            '{"type":"step_finish"',
        )
        with self.assertRaises(ProposerOutputTokenLimitError):
            parse_opencode_ndjson(partial, expected_parent_hash=SEED_HASH)

        semantic = json.dumps({"type": "text", "part": {"text": "{}"}})
        with self.assertRaises(ProposerOutputTokenLimitError):
            parse_opencode_ndjson(
                semantic + "\n" + raw.splitlines()[-1],
                expected_parent_hash=SEED_HASH,
            )

        complete = json.dumps(
            {"type": "text", "part": {"text": json.dumps(_proposal_value())}}
        )
        length = raw.splitlines()[-1]
        proposal = parse_opencode_ndjson(
            complete + "\n" + length,
            expected_parent_hash=SEED_HASH,
        )
        self.assertEqual(proposal.candidate_hash, SEED_HASH)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            runner = FixedRunner([CommandResult((), 0, raw, "")])
            adapter = OpenCodeProposer(
                config,
                run_id="l" * 32,
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
            with self.assertRaises(ProposerOutputTokenLimitError):
                adapter.propose(request)
            self.assertEqual(len(runner.argv), 1)
            self.assertEqual(adapter.attempts[0]["outcome"], "PROPOSAL_ERROR")

    def test_step_limit_adapter_archives_redacted_output_before_raising(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            fixture = STEP_LIMIT_FIXTURE.read_text(encoding="utf-8")
            runner = FixedRunner(
                [
                    CommandResult(
                        argv=(),
                        returncode=0,
                        stdout=fixture,
                        stderr="provider diagnostic secret",
                    )
                ]
            )
            adapter = OpenCodeProposer(
                config,
                run_id="b" * 32,
                iteration_index=5,
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
            with self.assertRaisesRegex(
                ProposerStepLimitError, r"^PROPOSER_STEP_LIMIT:"
            ):
                adapter.propose(request)
            self.assertEqual(
                adapter.raw_path.read_text(encoding="utf-8"), fixture
            )
            stderr = adapter.stderr_path.read_text(encoding="utf-8")
            self.assertNotIn("secret", stderr)
            self.assertIn("[REDACTED_SECRET]", stderr)
            self.assertEqual(len(runner.argv), 1)

    def test_format_retry_succeeds_with_shared_budgets_and_full_audit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            malformed = json.dumps(
                {
                    "type": "text",
                    "part": {
                        "text": '{"schema_version":1,"kernel_source":"'
                    },
                }
            )
            valid = json.dumps(
                {
                    "type": "text",
                    "part": {"text": json.dumps(_proposal_value())},
                }
            )
            runner = FixedRunner(
                [
                    CommandResult((), 0, malformed, "secret first"),
                    CommandResult((), 0, valid, "secret second"),
                ]
            )
            guarded_timeouts: list[float] = []
            adapter = OpenCodeProposer(
                config,
                run_id="c" * 32,
                iteration_index=1,
                run_dir=root / "run",
                runner=runner,  # type: ignore[arg-type]
                before_container_start=guarded_timeouts.append,
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
            with mock.patch(
                "kernel_research.autorun.opencode.time.monotonic",
                side_effect=[1000.0, 1000.0, 1100.0],
            ):
                proposal = adapter.propose(request)
            self.assertEqual(proposal.candidate_hash, SEED_HASH)
            self.assertEqual(len(runner.argv), 2)
            self.assertEqual(len(guarded_timeouts), 2)
            self.assertGreater(guarded_timeouts[0], guarded_timeouts[1])
            self.assertTrue(
                all(adapter.container_name in argv for argv in runner.argv)
            )
            self.assertEqual(
                [item["outcome"] for item in adapter.attempts],
                ["FORMAT_ERROR", "SUCCESS"],
            )
            first_raw = root / "run" / "raw" / "001.ndjson"
            retry_raw = root / "run" / "raw" / "001.retry-1.ndjson"
            retry_prompt = root / "run" / "prompts" / "001.retry-1.txt"
            self.assertTrue(first_raw.is_file())
            self.assertTrue(retry_raw.is_file())
            self.assertIn("FORMAT RETRY", retry_prompt.read_text(encoding="utf-8"))
            for stderr_path in (
                root / "run" / "raw" / "001.stderr.txt",
                root / "run" / "raw" / "001.retry-1.stderr.txt",
            ):
                stderr = stderr_path.read_text(encoding="utf-8")
                self.assertNotIn("secret", stderr)
                self.assertIn("[REDACTED_SECRET]", stderr)
            first_limit = int(runner.kwargs[0]["max_output_bytes"])
            second_limit = int(runner.kwargs[1]["max_output_bytes"])
            used = len(malformed.encode("utf-8")) + len(
                "secret first".encode("utf-8")
            )
            self.assertEqual(first_limit, config.proposer_max_output_bytes)
            self.assertEqual(second_limit, first_limit - used)
            self.assertEqual(
                float(runner.kwargs[0]["timeout_sec"]), 1200.0
            )
            self.assertEqual(
                float(runner.kwargs[1]["timeout_sec"]), 1100.0
            )

    def test_constraint_retry_succeeds_with_shared_budgets_and_audit(
        self,
    ) -> None:
        for field, limit in (("hypothesis", 1000), ("rationale", 8000)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = _config(root)
                overlong = _proposal_value()
                overlong[field] = "x" * (limit + 1)
                valid = _proposal_value()
                outputs = (
                    _proposal_ndjson(json.dumps(overlong)),
                    _proposal_ndjson(json.dumps(valid)),
                )
                runner = FixedRunner(
                    [
                        CommandResult((), 0, outputs[0], "secret first"),
                        CommandResult((), 0, outputs[1], "secret second"),
                    ]
                )
                adapter = OpenCodeProposer(
                    config,
                    run_id="1" * 32,
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
                with mock.patch(
                    "kernel_research.autorun.opencode.time.monotonic",
                    side_effect=[1000.0, 1000.0, 1100.0],
                ):
                    proposal = adapter.propose(request)

                self.assertEqual(proposal.candidate_hash, SEED_HASH)
                self.assertEqual(len(runner.argv), 2)
                self.assertEqual(
                    [item["outcome"] for item in adapter.attempts],
                    ["CONSTRAINT_ERROR", "SUCCESS"],
                )
                first = adapter.attempts[0]
                self.assertEqual(first["field"], field)
                self.assertEqual(first["actual_length"], limit + 1)
                self.assertEqual(first["hard_limit"], limit)
                self.assertEqual(
                    first["error_code"], f"{field.upper()}_TOO_LONG"
                )
                self.assertEqual(
                    adapter.attempts[1]["retry_trigger"],
                    "FIELD_LENGTH_ERROR",
                )
                retry_prompt = (
                    root / "run" / "prompts" / "001.retry-1.txt"
                ).read_text(encoding="utf-8")
                self.assertIn("COMPLIANCE RETRY", retry_prompt)
                self.assertIn(f"contained {limit + 1} characters", retry_prompt)
                self.assertIn("at most 600 characters", retry_prompt)
                self.assertIn("at most 6000 characters", retry_prompt)
                self.assertNotIn("x" * (limit + 1), retry_prompt)
                for stderr_path in (
                    root / "run" / "raw" / "001.stderr.txt",
                    root / "run" / "raw" / "001.retry-1.stderr.txt",
                ):
                    stderr = stderr_path.read_text(encoding="utf-8")
                    self.assertNotIn("secret", stderr)
                    self.assertIn("[REDACTED_SECRET]", stderr)
                first_limit = int(runner.kwargs[0]["max_output_bytes"])
                second_limit = int(runner.kwargs[1]["max_output_bytes"])
                used = len(outputs[0].encode("utf-8")) + len(
                    "secret first".encode("utf-8")
                )
                self.assertEqual(
                    second_limit,
                    first_limit - used,
                )
                self.assertEqual(
                    float(runner.kwargs[1]["timeout_sec"]), 1100.0
                )

    def test_constraint_and_format_retries_share_two_attempts(self) -> None:
        malformed = json.dumps(
            {"type": "text", "part": {"text": "not JSON"}}
        )
        overlong_hypothesis = _proposal_value()
        overlong_hypothesis["hypothesis"] = "x" * 1001
        overlong_rationale = _proposal_value()
        overlong_rationale["rationale"] = "x" * 8001
        cases = (
            (
                _proposal_ndjson(json.dumps(overlong_hypothesis)),
                _proposal_ndjson(json.dumps(overlong_rationale)),
                ProposerConstraintRetryExhaustedError,
                "FIELD_LENGTH_ERROR",
            ),
            (
                malformed,
                _proposal_ndjson(json.dumps(overlong_hypothesis)),
                ProposerConstraintRetryExhaustedError,
                "FORMAT_ERROR",
            ),
            (
                _proposal_ndjson(json.dumps(overlong_hypothesis)),
                malformed,
                ProposerFormatRetryExhaustedError,
                "FIELD_LENGTH_ERROR",
            ),
        )
        for first, second, error_type, retry_trigger in cases:
            with self.subTest(
                error=error_type.__name__, trigger=retry_trigger
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = _config(root)
                runner = FixedRunner(
                    [
                        CommandResult((), 0, first, ""),
                        CommandResult((), 0, second, ""),
                    ]
                )
                adapter = OpenCodeProposer(
                    config,
                    run_id="2" * 32,
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
                with self.assertRaises(error_type):
                    adapter.propose(request)
                self.assertEqual(len(runner.argv), 2)
                self.assertEqual(
                    adapter.attempts[1]["retry_trigger"], retry_trigger
                )

        semantic = _proposal_value()
        semantic["parent_candidate_hash"] = "0" * 64
        overlong_with_empty_source = _proposal_value()
        overlong_with_empty_source["hypothesis"] = "x" * 1001
        overlong_with_empty_source["kernel_source"] = ""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            runner = FixedRunner(
                [
                    CommandResult(
                        (),
                        0,
                        _proposal_ndjson(json.dumps(overlong_hypothesis)),
                        "",
                    ),
                    CommandResult(
                        (), 0, _proposal_ndjson(json.dumps(semantic)), ""
                    ),
                ]
            )
            adapter = OpenCodeProposer(
                config,
                run_id="3" * 32,
                iteration_index=1,
                run_dir=root / "run",
                runner=runner,  # type: ignore[arg-type]
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                adapter.propose(request)
            self.assertEqual(len(runner.argv), 2)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            runner = FixedRunner(
                [
                    CommandResult(
                        (),
                        0,
                        _proposal_ndjson(
                            json.dumps(overlong_with_empty_source)
                        ),
                        "",
                    )
                ]
            )
            adapter = OpenCodeProposer(
                config,
                run_id="4" * 32,
                iteration_index=1,
                run_dir=root / "run",
                runner=runner,  # type: ignore[arg-type]
            )
            with self.assertRaisesRegex(ValueError, "kernel_source"):
                adapter.propose(request)
            self.assertEqual(len(runner.argv), 1)

    def test_format_retry_exhaustion_and_non_format_failures_do_not_retry(
        self,
    ) -> None:
        malformed = json.dumps(
            {"type": "text", "part": {"text": "not JSON"}}
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            runner = FixedRunner(
                [
                    CommandResult((), 0, malformed, ""),
                    CommandResult((), 0, malformed, ""),
                ]
            )
            adapter = OpenCodeProposer(
                config,
                run_id="d" * 32,
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
            with self.assertRaisesRegex(
                ProposerFormatRetryExhaustedError,
                "^PROPOSER_FORMAT_RETRY_EXHAUSTED:",
            ):
                adapter.propose(request)
            self.assertEqual(len(runner.argv), 2)

        semantic = _proposal_value()
        semantic["parent_candidate_hash"] = "0" * 64
        unknown = _proposal_value()
        unknown["unknown"] = True
        empty_source = _proposal_value()
        empty_source["kernel_source"] = ""
        oversized_source = _proposal_value()
        oversized_source["kernel_source"] = "x" * (256 * 1024 + 1)
        non_retry_outputs = (
            json.dumps({"type": "tool_use", "name": "bash"}),
            json.dumps({"type": "error", "message": "provider failed"}),
            _proposal_ndjson(json.dumps(semantic)),
            _proposal_ndjson(json.dumps(unknown)),
            _proposal_ndjson(json.dumps(empty_source)),
            _proposal_ndjson(json.dumps(oversized_source)),
        )
        for index, output in enumerate(non_retry_outputs, 1):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = _config(root)
                runner = FixedRunner([CommandResult((), 0, output, "")])
                adapter = OpenCodeProposer(
                    config,
                    run_id="e" * 32,
                    iteration_index=index,
                    run_dir=root / "run",
                    runner=runner,  # type: ignore[arg-type]
                )
                with self.assertRaises(ValueError):
                    adapter.propose(request)
                self.assertEqual(len(runner.argv), 1)

        runtime_failures = (
            CommandResult((), -15, "", "", timed_out=True),
            CommandResult((), -9, "limited", "", output_limited=True),
            CommandResult((), 1, "", "container failed"),
        )
        for index, command_result in enumerate(runtime_failures, 1):
            with self.subTest(
                runtime=index
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = _config(root)
                runner = FixedRunner([command_result])
                adapter = OpenCodeProposer(
                    config,
                    run_id="f" * 32,
                    iteration_index=index,
                    run_dir=root / "run",
                    runner=runner,  # type: ignore[arg-type]
                )
                with self.assertRaises(RuntimeError):
                    adapter.propose(request)
                self.assertEqual(len(runner.argv), 1)

    def test_format_retry_exhaustion_counts_one_controller_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            (config.repository_dir / "program.md").write_text(
                "Only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)
            malformed = json.dumps(
                {"type": "text", "part": {"text": "not JSON"}}
            )
            runner = FixedRunner(
                [
                    CommandResult((), 0, malformed, ""),
                    CommandResult((), 0, malformed, ""),
                ]
            )
            evaluator = FakeEvaluator()
            controller = ResearchController(
                config,
                evaluator=evaluator,
                proposer_factory=lambda run_id, index, run_dir: OpenCodeProposer(
                    config,
                    run_id=run_id,
                    iteration_index=index,
                    run_dir=run_dir,
                    runner=runner,  # type: ignore[arg-type]
                ),
            )
            run_id = "format-retry-run"
            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id=run_id,
                    deadline_epoch=time.time() + 3600,
                    config=config.redacted_dict(),
                    initial_best_hash=SEED_HASH,
                )
            result = controller._run_loop(run_id, proposal_only=True)
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["consecutive_failures"], 1)
            self.assertEqual(result["valid_candidates"], 0)
            self.assertEqual(evaluator.stages, [])
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
                events = store.list_events(run_id)
            self.assertEqual(iteration["outcome"], "PROPOSER_ERROR")
            self.assertIsNone(iteration["candidate_hash"])
            self.assertEqual(iteration["experiment_ids"], {})
            self.assertIn(
                "PROPOSER_FORMAT_RETRY_EXHAUSTED",
                iteration["error"],
            )
            self.assertTrue(
                any(
                    event["event"] == "PROPOSER_FORMAT_RETRY"
                    for event in events
                )
            )
            self.assertTrue(
                str(iteration["raw_output_path"]).endswith(
                    "001.retry-1.ndjson"
                )
            )

    def test_controller_audits_constraint_retry_without_gpu_work(
        self,
    ) -> None:
        scenarios = ((False, "PROPOSAL_READY"), (True, "FAILED"))
        for exhaust, expected_status in scenarios:
            with self.subTest(
                exhaust=exhaust
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = _config(root)
                (config.repository_dir / "program.md").write_text(
                    "Only kernel.py.", encoding="utf-8"
                )
                _baseline(config.state_dir)
                source = _with_different_block_size_n(SEED)
                first = _proposal_value(source=source)
                first["hypothesis"] = "x" * 1001
                second = _proposal_value(source=source)
                if exhaust:
                    second["rationale"] = "x" * 8001
                runner = FixedRunner(
                    [
                        CommandResult(
                            (), 0, _proposal_ndjson(json.dumps(first)), ""
                        ),
                        CommandResult(
                            (), 0, _proposal_ndjson(json.dumps(second)), ""
                        ),
                    ]
                )
                evaluator = FakeEvaluator()
                controller = ResearchController(
                    config,
                    evaluator=evaluator,
                    proposer_factory=lambda run_id, index, run_dir: OpenCodeProposer(
                        config,
                        run_id=run_id,
                        iteration_index=index,
                        run_dir=run_dir,
                        runner=runner,  # type: ignore[arg-type]
                    ),
                )
                run_id = f"constraint-retry-{'fail' if exhaust else 'pass'}"
                with ControllerStore(controller.controller_db) as store:
                    store.create_run(
                        run_id=run_id,
                        deadline_epoch=time.time() + 3600,
                        config=config.redacted_dict(),
                        initial_best_hash=SEED_HASH,
                    )
                result = controller._run_loop(run_id, proposal_only=True)
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(evaluator.stages, [])
                with ControllerStore(controller.controller_db) as store:
                    iteration = store.latest_iteration(run_id)
                    events = store.list_events(run_id)
                retry = next(
                    event
                    for event in events
                    if event["event"] == "PROPOSER_CONSTRAINT_RETRY"
                )
                self.assertFalse(
                    any(
                        event["event"] == "PROPOSER_FORMAT_RETRY"
                        for event in events
                    )
                )
                attempts = retry["details"]["attempts"]
                self.assertEqual(len(attempts), 2)
                self.assertEqual(
                    attempts[0]["error_code"], "HYPOTHESIS_TOO_LONG"
                )
                self.assertNotIn("x" * 1001, json.dumps(retry["details"]))
                self.assertEqual(
                    attempts[1]["retry_trigger"], "FIELD_LENGTH_ERROR"
                )
                self.assertEqual(iteration["experiment_ids"], {})
                if exhaust:
                    self.assertEqual(result["consecutive_failures"], 1)
                    self.assertEqual(result["valid_candidates"], 0)
                    self.assertIsNone(iteration["candidate_hash"])
                    self.assertIn(
                        "PROPOSER_CONSTRAINT_RETRY_EXHAUSTED",
                        iteration["error"],
                    )
                else:
                    expected_hash = hashlib.sha256(
                        source.encode("utf-8")
                    ).hexdigest()
                    self.assertEqual(result["valid_candidates"], 1)
                    self.assertEqual(iteration["candidate_hash"], expected_hash)
                    self.assertEqual(iteration["outcome"], "PROPOSAL_VALIDATED")

    def test_output_token_limit_counts_once_and_never_calls_evaluator(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            (config.repository_dir / "program.md").write_text(
                "Only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)
            runner = FixedRunner(
                [
                    CommandResult(
                        (),
                        0,
                        OUTPUT_LIMIT_FIXTURE.read_text(encoding="utf-8"),
                        "",
                    )
                ]
            )
            evaluator = FakeEvaluator()
            controller = ResearchController(
                config,
                evaluator=evaluator,
                proposer_factory=lambda run_id, index, run_dir: OpenCodeProposer(
                    config,
                    run_id=run_id,
                    iteration_index=index,
                    run_dir=run_dir,
                    runner=runner,  # type: ignore[arg-type]
                ),
            )
            run_id = "output-token-limit-run"
            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id=run_id,
                    deadline_epoch=time.time() + 3600,
                    config=config.redacted_dict(),
                    initial_best_hash=SEED_HASH,
                )
            result = controller._run_loop(run_id, proposal_only=True)
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["consecutive_failures"], 1)
            self.assertEqual(result["valid_candidates"], 0)
            self.assertEqual(evaluator.stages, [])
            self.assertEqual(len(runner.argv), 1)
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
            self.assertEqual(iteration["outcome"], "PROPOSER_ERROR")
            self.assertIsNone(iteration["candidate_hash"])
            self.assertIn("PROPOSER_OUTPUT_TOKEN_LIMIT:", iteration["error"])

    def test_format_retry_enforces_shared_timeout_and_output_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            request = ProposalRequest(
                parent_candidate_hash=SEED_HASH,
                accepted_kernel=SEED,
                program_markdown="Only kernel.py.",
                environment={},
                accepted_case_p50_us={},
                recent_experiments=(),
                session_feedback=(),
            )
            timeout_runner = FixedRunner([])
            timeout_adapter = OpenCodeProposer(
                config,
                run_id="1" * 32,
                iteration_index=1,
                run_dir=root / "timeout",
                runner=timeout_runner,  # type: ignore[arg-type]
            )
            with (
                mock.patch(
                    "kernel_research.autorun.opencode.time.monotonic",
                    side_effect=[0.0, config.proposer_timeout_sec + 1],
                ),
                self.assertRaisesRegex(
                    RuntimeError, "shared configured timeout"
                ),
            ):
                timeout_adapter.propose(request)
            self.assertEqual(timeout_runner.argv, [])

            malformed = json.dumps(
                {"type": "text", "part": {"text": "not JSON"}}
            )
            output_config = replace(
                config,
                proposer_max_output_bytes=len(malformed.encode("utf-8")),
            )
            output_runner = FixedRunner(
                [CommandResult((), 0, malformed, "")]
            )
            output_adapter = OpenCodeProposer(
                output_config,
                run_id="2" * 32,
                iteration_index=1,
                run_dir=root / "output",
                runner=output_runner,  # type: ignore[arg-type]
            )
            with self.assertRaisesRegex(
                RuntimeError, "shared output byte limit"
            ):
                output_adapter.propose(request)
            self.assertEqual(len(output_runner.argv), 1)

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


class FeedbackAndStatusTests(unittest.TestCase):
    @staticmethod
    def _rejected_result(*, error: str | None = None) -> dict:
        return {
            "status": "SUCCESS" if error is None else "COMPILE_ERROR",
            "aggregate_score": 1.7656064778560598,
            "error": error,
            "latency_samples_us": [999.0] * 30,
            "promotion": {
                "phase": "rejected",
                "reason": "aggregate_speedup_below_threshold",
                "global_score": 1.7656064778560598,
                "decision": {
                    "reason": "aggregate_speedup_below_threshold",
                    "aggregate_speedup": 0.8363341873244841,
                    "worst_case_regression": 0.22573751310775947,
                    "per_case_speedups": {
                        "full_decode_down": 0.8530472848274107,
                        "full_decode_gate_up": 0.827830862551063,
                        "full_prefill_down": 0.8491846975816244,
                        "full_prefill_gate_up": 0.8158353556991007,
                    },
                },
            },
            "cases": [
                {
                    "case_id": "full_decode_down",
                    "status": "SUCCESS",
                    "matched_ratio": 1.0,
                    "p50_us": 117.0,
                    "baseline_p50_us": 100.0,
                    "latency_samples_us": [117.0] * 30,
                    "baseline_latency_samples_us": [100.0] * 30,
                    "error": error,
                }
            ],
        }

    @staticmethod
    def _complete_candidate(
        store: ControllerStore,
        *,
        run_id: str,
        index: int,
        candidate_hash: str,
        outcome: str,
        result: dict,
        error: str | None = None,
    ) -> None:
        iteration = store.create_iteration(run_id, index, SEED_HASH)
        iteration = store.accept_candidate(
            int(iteration["id"]),
            candidate_hash=candidate_hash,
            hypothesis=f"hypothesis-{index}",
            rationale="rationale-must-not-enter-feedback",
            candidate_path=f"/runtime/{candidate_hash}.py",
        )
        store.update_iteration(
            int(iteration["id"]),
            status="COMPLETED",
            stage="DONE",
            outcome=outcome,
            result=result,
            error=error,
        )

    def test_cross_run_feedback_is_candidate_level_bounded_and_restart_safe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            (config.repository_dir / "program.md").write_text(
                "Only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)
            previous_run = "previous-run"
            current_run = "current-run"
            long_error = "compile-detail-" + "x" * (
                MAX_FEEDBACK_ERROR_CHARS + 100
            )
            hashes = [
                hashlib.sha256(f"candidate-{index}".encode()).hexdigest()
                for index in range(MAX_FEEDBACK_CANDIDATES + 1)
            ]
            with ControllerStore(
                config.controller_dir / "controller.sqlite3"
            ) as store:
                for run_id in (previous_run, current_run):
                    store.create_run(
                        run_id=run_id,
                        deadline_epoch=time.time() + 3600,
                        config=config.redacted_dict(),
                        initial_best_hash=SEED_HASH,
                    )
                for index, candidate_hash in enumerate(hashes, 1):
                    is_latest = index == len(hashes)
                    self._complete_candidate(
                        store,
                        run_id=previous_run,
                        index=index,
                        candidate_hash=candidate_hash,
                        outcome=(
                            "COMPILE_ERROR" if is_latest else "FULL_REJECTED"
                        ),
                        result=self._rejected_result(
                            error=long_error if is_latest else None
                        ),
                        error=long_error if is_latest else None,
                    )
                self._complete_candidate(
                    store,
                    run_id=previous_run,
                    index=len(hashes) + 1,
                    candidate_hash=hashes[-1],
                    outcome="COMPILE_ERROR",
                    result=self._rejected_result(error=long_error),
                    error=long_error,
                )
                proposal_only = store.create_iteration(
                    previous_run, len(hashes) + 2, SEED_HASH
                )
                proposal_only = store.accept_candidate(
                    int(proposal_only["id"]),
                    candidate_hash="f" * 64,
                    hypothesis="proposal-only",
                    rationale="not scientific",
                    candidate_path="/runtime/proposal-only.py",
                )
                store.update_iteration(
                    int(proposal_only["id"]),
                    status="COMPLETED",
                    stage="DONE",
                    outcome="PROPOSAL_VALIDATED",
                    result={"status": "SUCCESS"},
                )
                for index in range(1, MAX_FEEDBACK_CANDIDATES + 2):
                    proposer_error = store.create_iteration(
                        current_run, index, SEED_HASH
                    )
                    store.update_iteration(
                        int(proposer_error["id"]),
                        status="COMPLETED",
                        stage="DONE",
                        outcome="PROPOSER_ERROR",
                        error=f"malformed JSON {index}",
                    )

            controller = ResearchController(
                config,
                evaluator=FakeEvaluator(),
            )
            request = controller._proposal_request(current_run)
            self.assertEqual(
                len(request.recent_experiments), MAX_FEEDBACK_CANDIDATES
            )
            candidate_hashes = [
                item["candidate_hash"]
                for item in request.recent_experiments
            ]
            self.assertEqual(len(candidate_hashes), len(set(candidate_hashes)))
            self.assertNotIn(hashes[0], candidate_hashes)
            self.assertNotIn("f" * 64, candidate_hashes)
            newest = request.recent_experiments[-1]
            summary = newest["result_summary"]
            self.assertEqual(summary["status"], "COMPILE_ERROR")
            self.assertEqual(
                summary["global_normalized_score"],
                1.7656064778560598,
            )
            self.assertEqual(
                summary["relative_speedup_vs_accepted"],
                0.8363341873244841,
            )
            self.assertLessEqual(
                len(summary["error"]), MAX_FEEDBACK_ERROR_CHARS
            )
            self.assertEqual(
                len(request.session_feedback), MAX_FEEDBACK_CANDIDATES
            )
            self.assertEqual(
                request.session_feedback[0]["iteration_index"], 2
            )
            prompt = build_prompt(request)
            self.assertIn('"recent_candidate_feedback"', prompt)
            self.assertNotIn('"recent_experiments"', prompt)
            self.assertNotIn("rationale-must-not-enter-feedback", prompt)
            self.assertNotIn("latency_samples_us", prompt)
            self.assertIn("one concise sentence", prompt)
            self.assertIn("Target at most 600 characters", prompt)
            self.assertIn("Target rationale at most 6000 characters", prompt)
            with ControllerStore(controller.controller_db) as store:
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    store.list_recent_scientific_iterations(
                        exclude_run_id=current_run,
                        limit=-1,
                    )

    def test_agent_feedback_redacts_holdout_cases_but_keeps_host_summary(self) -> None:
        iteration = {
            "run_id": "run",
            "iteration_index": 1,
            "result": {
                "status": "SUCCESS",
                "promotion": {
                    "decision": {
                        "per_case_speedups": {
                            "public": 1.1,
                            "secret-holdout": 0.9,
                        }
                    }
                },
                "cases": [
                    {"case_id": "public", "status": "SUCCESS"},
                    {
                        "case_id": "secret-holdout",
                        "status": "SUCCESS",
                    },
                ],
            },
        }
        host_summary = summarize_result(iteration["result"])
        self.assertEqual(len(host_summary["cases"]), 2)
        feedback = feedback_for_iteration(
            iteration, hidden_case_ids=frozenset({"secret-holdout"})
        )
        result = feedback["result_summary"]
        self.assertEqual(
            [case["case_id"] for case in result["cases"]], ["public"]
        )
        self.assertEqual(result["per_case_speedups"], {"public": 1.1})

    def test_status_result_summary_and_compact_projection_are_unambiguous(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            _baseline(config.state_dir)
            run_id = "status-run"
            candidate_hash = hashlib.sha256(b"status-candidate").hexdigest()
            with ControllerStore(
                config.controller_dir / "controller.sqlite3"
            ) as store:
                store.create_run(
                    run_id=run_id,
                    deadline_epoch=time.time() + 3600,
                    config=config.redacted_dict(),
                    initial_best_hash=SEED_HASH,
                )
                self._complete_candidate(
                    store,
                    run_id=run_id,
                    index=1,
                    candidate_hash=candidate_hash,
                    outcome="FULL_REJECTED",
                    result=self._rejected_result(),
                )
                active = store.create_iteration(run_id, 2, SEED_HASH)
                store.update_iteration(
                    int(active["id"]),
                    active_container="exact-active-container",
                )

            controller = ResearchController(
                config,
                evaluator=FakeEvaluator(),
            )
            status = controller.status(run_id)
            completed = status["iterations"][0]
            summary = completed["result_summary"]
            self.assertNotIn("result", completed)
            self.assertEqual(
                summary["global_normalized_score"],
                1.7656064778560598,
            )
            self.assertEqual(
                summary["relative_speedup_vs_accepted"],
                0.8363341873244841,
            )
            self.assertNotIn("latency_samples_us", json.dumps(summary))
            self.assertEqual(
                status["iterations"][1]["result_summary"]["status"], None
            )

            compact = compact_status_payload(status)
            self.assertEqual(compact["format"], "compact")
            self.assertNotIn("config", compact["run"])
            self.assertNotIn("preflight", compact["run"])
            self.assertEqual(
                compact["run"]["proposer_model"],
                "deepseek/deepseek-v4-pro",
            )
            self.assertEqual(
                compact["iterations"][1]["active_container"],
                "exact-active-container",
            )
            self.assertEqual(
                compact["iterations"][0]["result_summary"],
                summary,
            )
            confirmation_result = self._rejected_result()
            decision = confirmation_result["promotion"]["decision"]
            decision["confirmation_speedup"] = 0.82
            decision["confirmation_worst_case_regression"] = 0.18
            confirmation = summarize_result(confirmation_result)
            self.assertEqual(
                confirmation["primary_speedup_vs_accepted"],
                0.8363341873244841,
            )
            self.assertEqual(
                confirmation["relative_speedup_vs_accepted"], 0.82
            )
            self.assertEqual(
                confirmation["conservative_speedup_vs_accepted"], 0.82
            )
            self.assertEqual(confirmation["worst_case_regression"], 0.18)


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
                ControllerStore(
                    database,
                    _v2_to_v3_migration_capability=(
                        _V2_TO_V3_MIGRATION_CAPABILITY
                    ),
                )

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
