from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import uuid
from unittest import mock

from kernel_research.autorun import admin
from kernel_research.autorun.controller import ResearchController
from kernel_research.autorun.deployment import (
    DEPLOYMENT_BASELINE_FILENAME,
    DeploymentBaselinePin,
)
from kernel_research.autorun.store import ControllerStore
from kernel_research.history import HistoryStore
from kernel_research.platform.identity import BaselineRef
from kernel_research.platform.profiles import (
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
)
from kernel_research.platform.proposal import CandidateBundle

from test_autorun import (
    FakeEvaluator,
    NoopRunner,
    ProposalV1,
    SEED,
    SEED_HASH,
    StaticProposer,
    _baseline,
    _config,
    _proposal_value,
    _with_block_size_n,
    _with_different_block_size_n,
)
import test_requalification as requalification_helpers


class ConsoleManualEvaluationTests(unittest.TestCase):
    _PREFLIGHT = requalification_helpers.RequalificationTests._PREFLIGHT

    def _install_current_evidence(self, root: Path):
        legacy_config = _config(root, max_candidates=1)
        _baseline(legacy_config.state_dir)
        requalification_helpers.RequalificationTests._install_legacy_pin(
            legacy_config
        )
        deployed_source = _with_different_block_size_n(SEED)
        deployed_hash = hashlib.sha256(deployed_source.encode("utf-8")).hexdigest()
        staged = root / "deployed-candidate.py"
        staged.write_text(deployed_source, encoding="utf-8")

        legacy_evaluator = requalification_helpers.TrustedFakeDockerEvaluator()
        legacy_controller = ResearchController(
            legacy_config, evaluator=legacy_evaluator
        )
        with HistoryStore(
            legacy_config.state_dir / "history.sqlite3",
            state_dir=legacy_config.state_dir,
        ) as history:
            legacy_seed = history.get_best_for_namespace(
                LEGACY_RESEARCH_NAMESPACE.namespace_id,
                backend="c500",
                suite="full",
            )
        assert legacy_seed is not None
        with (
            mock.patch.object(
                legacy_controller, "doctor", return_value=self._PREFLIGHT
            ),
            mock.patch.object(
                legacy_controller, "_best", return_value=legacy_seed
            ),
        ):
            legacy_controller.requalify_candidate_for_adoption(
                candidate_path=staged,
                candidate_hash=deployed_hash,
                namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            )
        legacy_proof = admin._prove_adoption(
            legacy_config,
            candidate_hash=deployed_hash,
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
        )
        assert legacy_proof.primary is not None
        assert legacy_proof.parent_baseline is not None
        legacy_ref = BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=legacy_proof.confirmation.artifact_id,
            source="deployment",
            revision=f"history-{legacy_proof.confirmation.id}",
            execution_environment=legacy_proof.execution_environment,
        )
        legacy_pin = DeploymentBaselinePin.create(
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            baseline_ref=legacy_ref,
            candidate_hash=deployed_hash,
            git_commit=legacy_config.expected_git_commit,
            primary_experiment_uid=legacy_proof.primary.experiment_uid,
            confirmation_experiment_uid=(
                legacy_proof.confirmation.experiment_uid
            ),
            confirmation_experiment_id=legacy_proof.confirmation.id,
            parent_baseline_ref=legacy_proof.parent_baseline,
            execution_environment=legacy_proof.execution_environment,
        )
        pin_path = legacy_config.state_dir.parent / DEPLOYMENT_BASELINE_FILENAME
        pin_path.write_text(json.dumps(legacy_pin.to_dict()), encoding="utf-8")
        pin_path.chmod(0o600)
        (legacy_config.repository_dir / "kernel.py").write_text(
            deployed_source, encoding="utf-8"
        )
        current_config = replace(
            legacy_config, expected_kernel_hash=deployed_hash
        )
        current_controller = ResearchController(
            current_config,
            evaluator=requalification_helpers.TrustedFakeDockerEvaluator(),
        )
        with (
            mock.patch.object(
                current_controller, "doctor", return_value=self._PREFLIGHT
            ),
            mock.patch.object(
                current_controller,
                "_best",
                return_value=legacy_proof.confirmation,
            ),
        ):
            current_controller.bootstrap_current_baseline(
                candidate_path=current_config.repository_dir / "kernel.py",
                candidate_hash=deployed_hash,
            )
        current_proof = admin._prove_adoption(
            current_config,
            candidate_hash=deployed_hash,
            namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
        )
        assert current_proof.primary is not None
        assert current_proof.parent_baseline is not None
        current_ref = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=current_proof.confirmation.artifact_id,
            source="deployment",
            revision=f"history-{current_proof.confirmation.id}",
            execution_environment=current_proof.execution_environment,
        )
        current_pin = DeploymentBaselinePin.create(
            namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
            baseline_ref=current_ref,
            candidate_hash=deployed_hash,
            git_commit=current_config.expected_git_commit,
            primary_experiment_uid=current_proof.primary.experiment_uid,
            confirmation_experiment_uid=(
                current_proof.confirmation.experiment_uid
            ),
            confirmation_experiment_id=current_proof.confirmation.id,
            parent_baseline_ref=current_proof.parent_baseline,
            execution_environment=current_proof.execution_environment,
        )
        pin_path.write_text(json.dumps(current_pin.to_dict()), encoding="utf-8")
        pin_path.chmod(0o600)
        return current_config, current_pin, current_proof.confirmation, pin_path

    def test_manual_candidate_runs_current_chain_once_without_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config, pin, baseline, pin_path = self._install_current_evidence(root)
            pin_before = pin_path.read_bytes()
            with HistoryStore(
                config.state_dir / "history.sqlite3",
                state_dir=config.state_dir,
            ) as history:
                history_before = max(
                    item.id
                    for item in history.list_experiments_for_namespace(
                        CURRENT_RESEARCH_NAMESPACE.namespace_id,
                        newest_first=False,
                    )
                )

            candidate_source = _with_block_size_n(SEED, 64)
            self.assertNotEqual(candidate_source, (root / "repo/kernel.py").read_text())
            bundle = CandidateBundle.single_file(content=candidate_source)
            evaluator = requalification_helpers.TrustedFakeDockerEvaluator()
            controller = ResearchController(config, evaluator=evaluator)
            operation_id = str(uuid.uuid4())
            with (
                mock.patch.object(
                    controller, "doctor", return_value=self._PREFLIGHT
                ),
                mock.patch.object(controller, "_best", return_value=baseline),
            ):
                result = controller.evaluate_manual_candidate(
                    candidate=bundle,
                    operation_id=operation_id,
                )

            self.assertEqual(result["status"], "PROMOTED")
            self.assertEqual(
                result["namespace_id"],
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
            )
            self.assertEqual(
                evaluator.stages,
                ["smoke", "quick", "full_primary", "confirmation"],
            )
            self.assertEqual(pin_path.read_bytes(), pin_before)
            self.assertFalse((config.state_dir.parent / "campaign").exists())
            with HistoryStore(
                config.state_dir / "history.sqlite3",
                state_dir=config.state_dir,
            ) as history:
                records = history.find_by_candidate_hash(
                    hashlib.sha256(candidate_source.encode("utf-8")).hexdigest(),
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                )
                self.assertEqual(max(item.id for item in records), history_before + 4)
            self.assertEqual(
                [item.identity["stage"] for item in records],
                ["smoke", "quick", "full_primary", "confirmation"],
            )
            self.assertTrue(records[-1].promotable)
            self.assertEqual(
                records[-1].baseline_experiment_uid,
                pin.confirmation_experiment_uid,
            )
            with ControllerStore(controller.controller_db) as store:
                run = store.get_run(result["run_id"])
                snapshot = run["workflow_snapshot"]
                self.assertEqual(
                    snapshot["evidence_operation"],
                    "console-manual-evaluation-v1",
                )
                self.assertEqual(snapshot["console_operation_id"], operation_id)
                self.assertEqual(store.list_proposal_attempts(result["run_id"]), [])

            before_replay = list(evaluator.stages)
            with mock.patch.object(controller, "_best", return_value=baseline):
                replay = controller.evaluate_manual_candidate(
                    candidate=bundle,
                    operation_id=operation_id,
                )
            self.assertEqual(replay["run_id"], result["run_id"])
            self.assertEqual(evaluator.stages, before_replay)

    def test_manual_candidate_rejects_untrusted_inputs_before_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = _config(root, max_candidates=1)
            _baseline(config.state_dir)
            evaluator = requalification_helpers.TrustedFakeDockerEvaluator()
            controller = ResearchController(config, evaluator=evaluator)
            with self.assertRaisesRegex(ValueError, "canonical UUID"):
                controller.evaluate_manual_candidate(
                    candidate=CandidateBundle.single_file(content=SEED),
                    operation_id="not-a-uuid",
                )
            with self.assertRaises(ValueError):
                controller.evaluate_manual_candidate(
                    candidate={
                        "format": "source-bundle-v1",
                        "entrypoint": "kernel.py",
                        "files": [
                            {
                                "path": "kernel.py",
                                "media_type": "text/x-python",
                                "content": SEED,
                            },
                            {
                                "path": "other.py",
                                "media_type": "text/x-python",
                                "content": "pass\n",
                            },
                        ],
                    },
                    operation_id=str(uuid.uuid4()),
                )
            self.assertEqual(evaluator.stages, [])

    def test_manual_candidate_initialization_rolls_back_as_one_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config, _pin, baseline, _pin_path = self._install_current_evidence(root)
            bundle = CandidateBundle.single_file(
                content=_with_block_size_n(SEED, 64)
            )
            operation_id = str(uuid.uuid4())
            run_id = "console-manual-" + uuid.uuid5(
                uuid.NAMESPACE_URL,
                "kernel-research/console-manual-evaluation/v1/"
                f"{operation_id}/{bundle.artifact_id}",
            ).hex
            evaluator = requalification_helpers.TrustedFakeDockerEvaluator()
            controller = ResearchController(config, evaluator=evaluator)
            with (
                mock.patch.object(
                    controller, "doctor", return_value=self._PREFLIGHT
                ),
                mock.patch.object(controller, "_best", return_value=baseline),
                mock.patch.object(
                    ControllerStore,
                    "create_iteration",
                    side_effect=sqlite3.OperationalError("injected I/O failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    sqlite3.OperationalError, "injected I/O failure"
                ):
                    controller.evaluate_manual_candidate(
                        candidate=bundle,
                        operation_id=operation_id,
                    )
            with ControllerStore(controller.controller_db) as store:
                with self.assertRaisesRegex(ValueError, "unknown run id"):
                    store.get_run(run_id)
                self.assertEqual(store.list_events(run_id), [])
                self.assertEqual(store.list_iterations(run_id), [])
                self.assertEqual(store.list_evaluation_attempts(run_id), [])
            self.assertEqual(evaluator.stages, [])

    def test_console_run_start_is_deterministic_and_proposal_only_on_replay(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = _config(root, max_candidates=1)
            (config.repository_dir / "program.md").write_text(
                "Change only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)
            source = _with_different_block_size_n(SEED)
            proposal = ProposalV1.from_value(
                _proposal_value(source), expected_parent_hash=SEED_HASH
            )
            proposer_calls: list[str] = []

            def proposer_factory(run_id: str, index: int, run_dir: Path):
                del index, run_dir
                proposer_calls.append(run_id)
                return StaticProposer(proposal)

            controller = ResearchController(
                config,
                evaluator=FakeEvaluator(),
                runner=NoopRunner(),
                proposer_factory=proposer_factory,
            )
            operation_id = str(uuid.uuid4())
            with mock.patch.object(
                controller, "doctor", return_value=self._PREFLIGHT
            ):
                result = controller.start_console_operation(
                    operation_id=operation_id, proposal_only=True
                )
            self.assertEqual(result["status"], "PROPOSAL_READY")
            self.assertEqual(len(proposer_calls), 1)
            with ControllerStore(controller.controller_db) as store:
                run = store.get_run(result["id"])
                self.assertTrue(run["workflow_snapshot"]["proposal_only"])
                self.assertEqual(
                    store.list_evaluation_attempts(result["id"]), []
                )
            replay = controller.start_console_operation(
                operation_id=operation_id, proposal_only=True
            )
            self.assertEqual(replay["id"], result["id"])
            self.assertEqual(len(proposer_calls), 1)
            with self.assertRaisesRegex(ValueError, "canonical UUID"):
                controller.start_console_operation(
                    operation_id="invalid", proposal_only=True
                )


if __name__ == "__main__":
    unittest.main()
