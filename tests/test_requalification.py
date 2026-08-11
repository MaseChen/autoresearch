from __future__ import annotations

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
from kernel_research.platform.identity import BaselineRef, ExperimentIdentity

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


if __name__ == "__main__":
    unittest.main()
