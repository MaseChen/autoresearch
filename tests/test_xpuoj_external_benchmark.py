from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_autorun import (
    FULL_CASES as FULL_CASE_NAMES,
    FakeEvaluator,
    NoopRunner,
    SEED,
    SEED_HASH,
    StaticProposer,
    _config,
    _proposal_value,
    _with_different_block_size_n,
)
from kernel_research.autorun.cli import build_parser
from kernel_research.autorun.controller import ResearchController
from kernel_research.autorun.models import ProposalV1
from kernel_research.autorun.store import ControllerStore
from kernel_research.cases import FULL_CASES, get_suite
from kernel_research.constants import (
    CURRENT_C500_EVALUATION_PROTOCOL_ID,
    XPUOJ_C500_EVALUATION_PROTOCOL_ID,
)
from kernel_research.history import HistoryStore
from kernel_research.platform.identity import BaselineRef, ExperimentIdentity
from kernel_research.platform.legacy import resolve_builtin_target
from kernel_research.platform.profiles import (
    CURRENT_RESEARCH_NAMESPACE,
    XPUOJ_BENCHMARK_NAMESPACE,
)
from kernel_research.platform.proposal import CandidateBundle


class XpuojExternalBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _external_receipt() -> str:
        rows = [
            "Accepted#141440",
            "Score84",
            "=== SPJ Report Testcase #1 Pass: OK ===",
            "=== SPJ Report Testcase #1 Pass: OK ===",
        ]
        rows.extend(
            f"=== SPJ Report Testcase #{case_id} Pass: OK ==="
            for case_id in (2, 3, 4)
        )
        return "\n".join(rows) + "\n"

    @staticmethod
    def _record_local_proof(controller: ResearchController) -> int:
        environment = controller._resolved_execution_environment(
            CURRENT_RESEARCH_NAMESPACE
        )
        bundle = CandidateBundle.single_file(content=SEED)
        baseline = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=bundle.artifact_id,
            source="campaign",
            revision="fixture-local-four-case-proof",
            execution_environment=environment,
        )
        identity = ExperimentIdentity.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            mode="DISCOVERY",
            candidate_artifact_id=bundle.artifact_id,
            parent_artifact_id=bundle.artifact_id,
            baseline=baseline,
            execution_environment=environment,
            stage="full_primary",
            suite="full",
            replicate_kind="primary",
            run_id="fixture-local-proof",
            iteration=1,
        )
        with HistoryStore(
            controller.history_db,
            state_dir=controller.config.state_dir,
        ) as history:
            history.ensure_namespace(
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
                CURRENT_RESEARCH_NAMESPACE.to_dict(),
            )
            history.store_candidate_bundle(bundle)
            record = history.record_experiment(
                candidate_source=SEED,
                status="SUCCESS",
                backend="c500",
                suite="full",
                promotable=True,
                aggregate_score=1.0,
                identity=identity,
                result={
                    "status": "SUCCESS",
                    "evaluation_protocol_id": (
                        CURRENT_C500_EVALUATION_PROTOCOL_ID
                    ),
                },
                case_measurements=(
                    {
                        "name": name,
                        "matched_ratio": 1.0,
                        "passed": True,
                        "raw_samples": [100.0] * 30,
                    }
                    for name in FULL_CASE_NAMES
                ),
            )
        return record.id

    def _register_fixture_seed(
        self,
        controller: ResearchController,
        root: Path,
    ):
        proof_id = self._record_local_proof(controller)
        candidate = root / "external-kernel.py"
        receipt = root / "xpuoj-result.txt"
        candidate.write_text(SEED, encoding="utf-8")
        receipt.write_text(self._external_receipt(), encoding="utf-8")
        with mock.patch.multiple(
            "kernel_research.autorun.controller",
            XPUOJ_EXTERNAL_BASELINE_SOURCE_SHA256=SEED_HASH,
            XPUOJ_EXTERNAL_BASELINE_RECEIPT_SHA256=hashlib.sha256(
                receipt.read_bytes()
            ).hexdigest(),
            XPUOJ_LOCAL_FOUR_CASE_PROOF_SHA256=SEED_HASH,
        ):
            first = controller.register_xpuoj_external_baseline(
                candidate_path=candidate,
                receipt_path=receipt,
                proof_experiment_id=proof_id,
            )
            replay = controller.register_xpuoj_external_baseline(
                candidate_path=candidate,
                receipt_path=receipt,
                proof_experiment_id=proof_id,
            )
        self.assertEqual(first.id, replay.id)
        return first

    def test_profile_freezes_only_the_four_official_cases(self) -> None:
        target = resolve_builtin_target(XPUOJ_BENCHMARK_NAMESPACE)
        self.assertEqual(
            [stage.stage_id for stage in target.protocol.stages],
            ["PROPOSE", "POLICY", "FULL_PRIMARY", "CONFIRMATION"],
        )
        self.assertEqual(
            target.protocol.suite_cases,
            {"full": tuple(case.name for case in FULL_CASES)},
        )
        self.assertEqual(
            get_suite(
                "full",
                evaluation_protocol_id=XPUOJ_C500_EVALUATION_PROTOCOL_ID,
            ),
            FULL_CASES,
        )
        with self.assertRaisesRegex(ValueError, "unknown suite"):
            get_suite(
                "smoke",
                evaluation_protocol_id=XPUOJ_C500_EVALUATION_PROTOCOL_ID,
            )

    def test_cli_exposes_no_free_protocol_case_or_image_arguments(self) -> None:
        parser = build_parser()
        register = parser.parse_args(
            [
                "register-xpuoj-baseline",
                "--config",
                "config.json",
                "--candidate",
                "kernel.py",
                "--receipt",
                "receipt.txt",
                "--proof-experiment-id",
                "294",
            ]
        )
        self.assertEqual(register.proof_experiment_id, 294)
        start = parser.parse_args(
            [
                "start-xpuoj-benchmark",
                "--config",
                "config.json",
                "--run-id",
                "run-1",
                "--baseline-experiment-id",
                "300",
            ]
        )
        self.assertEqual(start.baseline_experiment_id, 300)
        for forbidden in ("--protocol", "--case", "--image", "--device", "--timeout"):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "start-xpuoj-benchmark",
                        "--config",
                        "config.json",
                        "--run-id",
                        "run-1",
                        "--baseline-experiment-id",
                        "300",
                        forbidden,
                        "caller-value",
                    ]
                )

    def test_registration_rejects_untrusted_source_before_state_access(self) -> None:
        controller = object.__new__(ResearchController)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "kernel.py"
            receipt = root / "receipt.txt"
            candidate.write_text("def run_kernel():\n    pass\n", encoding="utf-8")
            receipt.write_text("Accepted#141440\nScore84\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source identity mismatch"):
                controller.register_xpuoj_external_baseline(
                    candidate_path=candidate,
                    receipt_path=receipt,
                    proof_experiment_id=294,
                )

    def test_registration_is_idempotent_and_keeps_external_authority_separate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            controller = ResearchController(_config(root))
            record = self._register_fixture_seed(controller, root)
            self.assertEqual(
                record.namespace_id,
                XPUOJ_BENCHMARK_NAMESPACE.namespace_id,
            )
            self.assertFalse(record.promotable)
            self.assertTrue(
                record.result["evidence"]["benchmark_baseline_eligible"]
            )
            self.assertFalse(record.result["evidence"]["promotion_authority"])
            with HistoryStore(
                controller.history_db,
                state_dir=controller.config.state_dir,
            ) as history:
                namespace = history.get_namespace(
                    XPUOJ_BENCHMARK_NAMESPACE.namespace_id
                )
                self.assertIsNotNone(namespace)
                self.assertEqual(
                    history.read_candidate_artifact(
                        "source-sha256-v1:" + SEED_HASH
                    ),
                    SEED.encode("utf-8"),
                )

    def test_benchmark_run_executes_only_full_primary_and_confirmation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = _config(root, max_candidates=1)
            (config.repository_dir / "program.md").write_text(
                "Change only kernel.py.", encoding="utf-8"
            )
            candidate_source = _with_different_block_size_n(SEED)
            proposal = ProposalV1.from_value(
                _proposal_value(candidate_source),
                expected_parent_hash=SEED_HASH,
            )
            evaluator = FakeEvaluator()
            controller = ResearchController(
                config,
                evaluator=evaluator,
                runner=NoopRunner(),
                proposer_factory=lambda _run_id, _index, _run_dir: (
                    StaticProposer(proposal)
                ),
            )
            baseline = self._register_fixture_seed(controller, root)
            with (
                mock.patch(
                    "kernel_research.autorun.controller."
                    "XPUOJ_EXTERNAL_BASELINE_SOURCE_SHA256",
                    SEED_HASH,
                ),
                mock.patch.object(
                    controller,
                    "_authorize_external_action",
                    return_value=None,
                ),
                mock.patch.object(
                    controller,
                    "doctor",
                    return_value={"status": "SUCCESS", "errors": []},
                ),
            ):
                result = controller.start_xpuoj_benchmark(
                    run_id="xpuoj-fixture-run",
                    baseline_experiment_id=baseline.id,
                )
            with ControllerStore(controller.controller_db) as store:
                iterations = store.list_iterations("xpuoj-fixture-run")
                events = store.list_events("xpuoj-fixture-run")
                attempts = store.list_evaluation_attempts(
                    "xpuoj-fixture-run"
                )
            self.assertEqual(
                result["status"],
                "PROMOTED",
                {"result": result, "iterations": iterations, "events": events},
            )
            self.assertEqual(
                evaluator.stages,
                ["full_primary", "confirmation"],
            )
            self.assertEqual(
                [attempt["suite"] for attempt in attempts],
                ["full", "full"],
            )
            self.assertEqual(
                {
                    attempt["request"]["namespace"]["namespace_id"]
                    for attempt in attempts
                },
                {XPUOJ_BENCHMARK_NAMESPACE.namespace_id},
            )


if __name__ == "__main__":
    unittest.main()
