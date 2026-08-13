from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from kernel_research.autorun import admin
from kernel_research.autorun.controller import (
    DockerEvaluator,
    ResearchController,
    UnknownGPUOutcome,
)
from kernel_research.autorun.deployment import (
    DEPLOYMENT_BASELINE_FILENAME,
    DeploymentBaselinePin,
)
from kernel_research.autorun.store import ControllerStore
from kernel_research.evaluation import record_external_result
from kernel_research.history import HistoryStore
from kernel_research.platform.profiles import LEGACY_RESEARCH_NAMESPACE
from kernel_research.platform.profiles import CURRENT_RESEARCH_NAMESPACE
from kernel_research.platform.identity import BaselineRef, ExperimentIdentity
from kernel_research.recovery import verify_checkpoint

from test_autorun import (
    FakeEvaluator,
    SEED,
    _baseline,
    _config,
    _with_different_block_size_n,
)


class TrustedFakeDockerEvaluator(DockerEvaluator):
    """Side-effect-free fixture that still crosses the production type gate."""

    def __init__(self) -> None:
        self.fixture = FakeEvaluator()
        self.before_container_start = None

    @property
    def stages(self) -> list[str]:
        return self.fixture.stages

    def container_name(
        self, run_id: str, iteration_index: int, stage: str
    ) -> str:
        return self.fixture.container_name(run_id, iteration_index, stage)

    def doctor(self, *, run_id: str) -> dict:
        return self.fixture.doctor(run_id=run_id)

    def evaluate(self, **values):
        return self.fixture.evaluate(**values)


class RequalificationTests(unittest.TestCase):
    _PREFLIGHT = {
        "schema_version": 1,
        "command": "doctor",
        "status": "SUCCESS",
        "errors": [],
    }

    @staticmethod
    def _install_legacy_pin(config) -> None:
        with HistoryStore(
            config.state_dir / "history.sqlite3",
            state_dir=config.state_dir,
        ) as history:
            baseline = history.get_best_for_namespace(
                LEGACY_RESEARCH_NAMESPACE.namespace_id,
                backend="c500",
                suite="full",
            )
        assert baseline is not None
        baseline_ref = BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=baseline.artifact_id,
            source="deployment",
            revision=f"history-{baseline.id}",
        )
        pin = DeploymentBaselinePin.create(
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            baseline_ref=baseline_ref,
            candidate_hash=baseline.candidate_hash,
            git_commit=config.expected_git_commit,
            primary_experiment_uid=None,
            confirmation_experiment_uid=baseline.experiment_uid,
            confirmation_experiment_id=baseline.id,
            parent_baseline_ref=None,
            execution_environment=baseline_ref.execution_environment,
        )
        path = config.state_dir.parent / DEPLOYMENT_BASELINE_FILENAME
        path.write_text(json.dumps(pin.to_dict()), encoding="utf-8")
        path.chmod(0o600)

    def test_legacy_baseline_is_qualified_before_normal_candidate_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = _config(root, max_candidates=1)
            _baseline(config.state_dir)
            self._install_legacy_pin(config)
            candidate = _with_different_block_size_n(SEED)
            candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            candidate_path = root / "candidate.py"
            candidate_path.write_text(candidate, encoding="utf-8")
            evaluator = TrustedFakeDockerEvaluator()
            controller = ResearchController(config, evaluator=evaluator)
            with HistoryStore(
                config.state_dir / "history.sqlite3",
                state_dir=config.state_dir,
            ) as history:
                baseline = history.get_best_for_namespace(
                    LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    backend="c500",
                    suite="full",
                )
            assert baseline is not None
            with (
                mock.patch.object(
                    controller, "doctor", return_value=self._PREFLIGHT
                ),
                mock.patch.object(controller, "_best", return_value=baseline),
            ):
                result = controller.requalify_candidate_for_adoption(
                    candidate_path=candidate_path,
                    candidate_hash=candidate_hash,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                )

            self.assertEqual(result["status"], "PROMOTED")
            self.assertEqual(
                evaluator.stages,
                [
                    "baseline_qualification",
                    "smoke",
                    "quick",
                    "full_primary",
                    "confirmation",
                ],
            )
            with HistoryStore(
                config.state_dir / "history.sqlite3",
                state_dir=config.state_dir,
            ) as history:
                qualification = history.get_experiment(
                    result["qualification_experiment_id"]
                )
                candidate_records = history.find_by_candidate_hash(
                    candidate_hash,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                )
            self.assertIsNotNone(qualification)
            assert qualification is not None
            self.assertTrue(qualification.promotable)
            self.assertEqual(
                qualification.result["promotion"]["phase"], "baseline"
            )
            self.assertTrue(
                qualification.identity["execution_environment"]["status"]
                == "RESOLVED"
            )
            self.assertEqual(len(candidate_records), 4)
            self.assertTrue(candidate_records[-1].promotable)
            self.assertEqual(
                candidate_records[-1].baseline_experiment_uid,
                qualification.experiment_uid,
            )
            proof = admin._prove_adoption(
                config,
                candidate_hash=candidate_hash,
                namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            )
            self.assertTrue(proof.execution_environment.is_resolved)
            assert proof.parent_baseline is not None
            self.assertEqual(
                proof.parent_baseline.revision,
                qualification.result["request_identity"]["baseline"][
                    "revision"
                ],
            )
            with ControllerStore(controller.controller_db) as store:
                attempts = store.list_evaluation_attempts(result["run_id"])
            self.assertEqual(len(attempts), 5)
            self.assertTrue(all(item["status"] == "SUCCEEDED" for item in attempts))
            with self.assertRaisesRegex(
                ValueError, "trusted controller capability"
            ):
                record_external_result(
                    candidate_source=SEED,
                    result=dict(qualification.result),
                    backend="c500",
                    suite="full",
                    state_dir=config.state_dir,
                    note="untrusted qualification replay",
                    identity=ExperimentIdentity.from_value(
                        qualification.identity
                    ),
                    baseline_experiment_id=baseline.id,
                )

            before = list(evaluator.stages)
            with (
                mock.patch.object(
                    controller, "doctor", return_value=self._PREFLIGHT
                ),
                mock.patch.object(controller, "_best", return_value=baseline),
            ):
                replay = controller.requalify_candidate_for_adoption(
                    candidate_path=candidate_path,
                    candidate_hash=candidate_hash,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                )
            self.assertEqual(replay["run_id"], result["run_id"])
            self.assertEqual(evaluator.stages, before)

    def test_requalification_rejects_untrusted_inputs_before_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = _config(root, max_candidates=1)
            _baseline(config.state_dir)
            self._install_legacy_pin(config)
            candidate_path = root / "candidate.py"
            candidate_path.write_text(SEED, encoding="utf-8")
            evaluator = TrustedFakeDockerEvaluator()
            controller = ResearchController(config, evaluator=evaluator)

            with self.assertRaisesRegex(ValueError, "exact LEGACY"):
                controller.requalify_candidate_for_adoption(
                    candidate_path=candidate_path,
                    candidate_hash=hashlib.sha256(
                        candidate_path.read_bytes()
                    ).hexdigest(),
                    namespace_id="sha256:" + "0" * 64,
                )
            with self.assertRaisesRegex(ValueError, "do not match"):
                controller.requalify_candidate_for_adoption(
                    candidate_path=candidate_path,
                    candidate_hash="f" * 64,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                )
            self.assertEqual(evaluator.stages, [])

    def test_durable_qualification_result_is_reconciled_without_gpu_replay(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = _config(root, max_candidates=1)
            _baseline(config.state_dir)
            self._install_legacy_pin(config)
            candidate = _with_different_block_size_n(SEED)
            candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            candidate_path = root / "candidate.py"
            candidate_path.write_text(candidate, encoding="utf-8")
            evaluator = TrustedFakeDockerEvaluator()
            controller = ResearchController(config, evaluator=evaluator)
            with HistoryStore(
                config.state_dir / "history.sqlite3",
                state_dir=config.state_dir,
            ) as history:
                baseline = history.get_best_for_namespace(
                    LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    backend="c500",
                    suite="full",
                )
            assert baseline is not None

            with (
                mock.patch.object(
                    controller, "doctor", return_value=self._PREFLIGHT
                ),
                mock.patch.object(controller, "_best", return_value=baseline),
                mock.patch(
                    "kernel_research.autorun.controller."
                    "record_baseline_qualification_result",
                    side_effect=KeyboardInterrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                controller.requalify_candidate_for_adoption(
                    candidate_path=candidate_path,
                    candidate_hash=candidate_hash,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                )
            self.assertEqual(evaluator.stages, ["baseline_qualification"])
            with ControllerStore(controller.controller_db) as store:
                attempts = store.list_evaluation_attempts_by_replicate_kind(
                    "qualification"
                )
            self.assertEqual(attempts[0]["status"], "RUNNING")

            with (
                mock.patch.object(
                    controller, "doctor", return_value=self._PREFLIGHT
                ),
                mock.patch.object(controller, "_best", return_value=baseline),
            ):
                result = controller.requalify_candidate_for_adoption(
                    candidate_path=candidate_path,
                    candidate_hash=candidate_hash,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                )
            self.assertEqual(result["status"], "PROMOTED")
            self.assertEqual(evaluator.stages.count("baseline_qualification"), 1)

    def test_interrupted_qualification_is_unknown_and_never_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = _config(root, max_candidates=1)
            _baseline(config.state_dir)
            self._install_legacy_pin(config)
            candidate = _with_different_block_size_n(SEED)
            candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            candidate_path = root / "candidate.py"
            candidate_path.write_text(candidate, encoding="utf-8")
            evaluator = TrustedFakeDockerEvaluator()
            controller = ResearchController(config, evaluator=evaluator)
            with HistoryStore(
                config.state_dir / "history.sqlite3",
                state_dir=config.state_dir,
            ) as history:
                baseline = history.get_best_for_namespace(
                    LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    backend="c500",
                    suite="full",
                )
            assert baseline is not None

            original_evaluate = evaluator.evaluate

            def interrupt_once(**values):
                evaluator.fixture.stages.append(str(values["stage"]))
                raise KeyboardInterrupt

            with (
                mock.patch.object(
                    controller, "doctor", return_value=self._PREFLIGHT
                ),
                mock.patch.object(controller, "_best", return_value=baseline),
                mock.patch.object(evaluator, "evaluate", side_effect=interrupt_once),
                self.assertRaises(KeyboardInterrupt),
            ):
                controller.requalify_candidate_for_adoption(
                    candidate_path=candidate_path,
                    candidate_hash=candidate_hash,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                )
            with ControllerStore(controller.controller_db) as store:
                attempts = store.list_evaluation_attempts_by_replicate_kind(
                    "qualification"
                )
            self.assertEqual(attempts[0]["status"], "UNKNOWN_OUTCOME")
            self.assertEqual(evaluator.stages, ["baseline_qualification"])

            evaluator.evaluate = original_evaluate
            with (
                mock.patch.object(
                    controller, "doctor", return_value=self._PREFLIGHT
                ),
                mock.patch.object(controller, "_best", return_value=baseline),
                self.assertRaisesRegex(UnknownGPUOutcome, "cannot restart"),
            ):
                controller.requalify_candidate_for_adoption(
                    candidate_path=candidate_path,
                    candidate_hash=candidate_hash,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                )
            self.assertEqual(evaluator.stages, ["baseline_qualification"])

    def test_admin_parser_exposes_only_fixed_requalification_inputs(self) -> None:
        parsed = admin.build_parser().parse_args(
            [
                "requalify-adoption",
                "--manifest",
                "/runtime/admin.json",
                "--candidate-hash",
                "a" * 64,
                "--candidate-path",
                "/runtime/candidate.py",
                "--namespace",
                LEGACY_RESEARCH_NAMESPACE.namespace_id,
            ]
        )
        self.assertEqual(parsed.command, "requalify-adoption")
        self.assertFalse(hasattr(parsed, "timeout"))
        self.assertFalse(hasattr(parsed, "image"))

    def test_current_bootstrap_repeats_the_pinned_improvement_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            legacy_config = _config(root, max_candidates=1)
            _baseline(legacy_config.state_dir)
            self._install_legacy_pin(legacy_config)
            candidate = _with_different_block_size_n(SEED)
            candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            candidate_path = root / "candidate.py"
            candidate_path.write_text(candidate, encoding="utf-8")
            legacy_evaluator = TrustedFakeDockerEvaluator()
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
                legacy_result = (
                    legacy_controller.requalify_candidate_for_adoption(
                        candidate_path=candidate_path,
                        candidate_hash=candidate_hash,
                        namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    )
                )
            self.assertEqual(legacy_result["status"], "PROMOTED")
            legacy_proof = admin._prove_adoption(
                legacy_config,
                candidate_hash=candidate_hash,
                namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            )
            assert legacy_proof.primary is not None
            assert legacy_proof.parent_baseline is not None
            deployed_ref = BaselineRef.create(
                namespace=LEGACY_RESEARCH_NAMESPACE,
                artifact_id=legacy_proof.confirmation.artifact_id,
                source="deployment",
                revision=f"history-{legacy_proof.confirmation.id}",
                execution_environment=legacy_proof.execution_environment,
            )
            deployed_pin = DeploymentBaselinePin.create(
                namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                baseline_ref=deployed_ref,
                candidate_hash=candidate_hash,
                git_commit=legacy_config.expected_git_commit,
                primary_experiment_uid=legacy_proof.primary.experiment_uid,
                confirmation_experiment_uid=(
                    legacy_proof.confirmation.experiment_uid
                ),
                confirmation_experiment_id=legacy_proof.confirmation.id,
                parent_baseline_ref=legacy_proof.parent_baseline,
                execution_environment=legacy_proof.execution_environment,
            )
            pin_path = (
                legacy_config.state_dir.parent / DEPLOYMENT_BASELINE_FILENAME
            )
            pin_path.write_text(
                json.dumps(deployed_pin.to_dict()), encoding="utf-8"
            )
            pin_path.chmod(0o600)
            pin_bytes = pin_path.read_bytes()

            deployed_kernel = legacy_config.repository_dir / "kernel.py"
            deployed_kernel.write_text(candidate, encoding="utf-8")
            current_config = replace(
                legacy_config, expected_kernel_hash=candidate_hash
            )
            with HistoryStore(
                current_config.state_dir / "history.sqlite3",
                state_dir=current_config.state_dir,
            ) as history:
                legacy_before = [
                    item.to_dict()
                    for item in history.list_experiments_for_namespace(
                        LEGACY_RESEARCH_NAMESPACE.namespace_id,
                        newest_first=False,
                    )
                ]
            current_evaluator = TrustedFakeDockerEvaluator()
            current_controller = ResearchController(
                current_config, evaluator=current_evaluator
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
                current_result = current_controller.bootstrap_current_baseline(
                    candidate_path=deployed_kernel,
                    candidate_hash=candidate_hash,
                )

            self.assertEqual(current_result["status"], "PROMOTED")
            self.assertEqual(
                current_result["namespace_id"],
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
            )
            self.assertEqual(
                current_result["evaluation_protocol"],
                CURRENT_RESEARCH_NAMESPACE.evaluation_protocol.to_dict(),
            )
            self.assertEqual(
                current_evaluator.stages,
                [
                    "baseline_qualification",
                    "smoke",
                    "quick",
                    "full_primary",
                    "confirmation",
                ],
            )
            self.assertEqual(pin_path.read_bytes(), pin_bytes)
            with HistoryStore(
                current_config.state_dir / "history.sqlite3",
                state_dir=current_config.state_dir,
            ) as history:
                qualification = history.get_experiment(
                    current_result["qualification_experiment_id"]
                )
                current_records = history.list_experiments_for_namespace(
                    CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    newest_first=False,
                )
                legacy_after = [
                    item.to_dict()
                    for item in history.list_experiments_for_namespace(
                        LEGACY_RESEARCH_NAMESPACE.namespace_id,
                        newest_first=False,
                    )
                ]
            self.assertEqual(legacy_after, legacy_before)
            self.assertIsNotNone(qualification)
            assert qualification is not None
            self.assertEqual(
                qualification.namespace_id,
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
            )
            self.assertIsNone(qualification.baseline_experiment_uid)
            self.assertEqual(
                qualification.result["evidence"]["role"],
                "current_baseline_bootstrap",
            )
            self.assertEqual(
                qualification.result["evidence"]["source_experiment_uid"],
                legacy_proof.parent_baseline.revision.removeprefix(
                    "qualification-"
                ),
            )
            self.assertEqual(len(current_records), 5)
            quick = next(
                item
                for item in current_records
                if item.identity.get("stage") == "quick"
            )
            self.assertEqual(len(quick.case_measurements), 8)
            confirmation = current_records[-1]
            self.assertTrue(confirmation.promotable)
            proof = admin._prove_adoption(
                current_config,
                candidate_hash=candidate_hash,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
            )
            self.assertEqual(proof.confirmation.id, confirmation.id)
            qualification_identity = ExperimentIdentity.from_value(
                qualification.identity
            )
            self.assertEqual(
                proof.parent_baseline, qualification_identity.baseline
            )
            with ControllerStore(current_controller.controller_db) as store:
                attempts = store.list_evaluation_attempts(current_result["run_id"])
            self.assertEqual(len(attempts), 5)
            self.assertTrue(
                all(item["status"] == "SUCCEEDED" for item in attempts)
            )
            checkpoint = Path(
                current_controller.checkpoint(current_result["run_id"])["path"]
            )
            verified = verify_checkpoint(checkpoint)
            self.assertEqual(verified["run_id"], current_result["run_id"])
            stages_before_replay = list(current_evaluator.stages)
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
                replay = current_controller.bootstrap_current_baseline(
                    candidate_path=deployed_kernel,
                    candidate_hash=candidate_hash,
                )
            self.assertEqual(replay["run_id"], current_result["run_id"])
            self.assertEqual(current_evaluator.stages, stages_before_replay)
            copied_candidate = root / "copied-deployed-kernel.py"
            copied_candidate.write_text(candidate, encoding="utf-8")
            with (
                mock.patch.object(
                    current_controller, "doctor", return_value=self._PREFLIGHT
                ) as doctor,
                mock.patch.object(
                    current_controller,
                    "_best",
                    return_value=legacy_proof.confirmation,
                ),
                self.assertRaisesRegex(
                    ValueError, "must be the deployed kernel.py"
                ),
            ):
                current_controller.bootstrap_current_baseline(
                    candidate_path=copied_candidate,
                    candidate_hash=candidate_hash,
                )
            doctor.assert_not_called()
            self.assertEqual(current_evaluator.stages, stages_before_replay)

    def test_current_bootstrap_parser_has_no_free_scientific_inputs(self) -> None:
        parsed = admin.build_parser().parse_args(
            [
                "bootstrap-current-baseline",
                "--manifest",
                "/runtime/admin.json",
                "--candidate-hash",
                "a" * 64,
            ]
        )
        self.assertEqual(parsed.command, "bootstrap-current-baseline")
        for forbidden in (
            "namespace_id",
            "candidate_path",
            "timeout",
            "image",
            "baseline",
        ):
            self.assertFalse(hasattr(parsed, forbidden), forbidden)


if __name__ == "__main__":
    unittest.main()
