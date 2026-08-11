from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from kernel_research.autorun.controller import (
    ControllerDataIntegrityError,
    ResearchController,
)
from kernel_research.autorun.models import ProposalV1
from kernel_research.autorun.proposal import ProposalRequest, Proposer, build_prompt
from kernel_research.autorun.store import ControllerStore
from kernel_research.constants import MAX_FEEDBACK_CANDIDATES
from kernel_research.history import HistoryStore
from kernel_research.platform.artifacts import ArtifactId
from kernel_research.platform.canonical import canonical_sha256
from kernel_research.platform.identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
)
from kernel_research.platform.profiles import (
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
)
from kernel_research.platform.proposal import CandidateBundle

from test_autorun import (
    FakeEvaluator,
    NoopRunner,
    SEED,
    SEED_HASH,
    _baseline,
    _config,
    _proposal_value,
    _with_different_block_size_n,
)


class AuditedStaticProposer(Proposer):
    """Write realistic proposer audit files while returning a fixed proposal."""

    def __init__(
        self,
        proposal: ProposalV1,
        *,
        run_dir: Path,
        iteration_index: int,
    ) -> None:
        self.proposal = proposal
        audit_dir = run_dir / "proposer-audit"
        self.prompt_path = audit_dir / f"{iteration_index:03d}.prompt.txt"
        self.raw_path = audit_dir / f"{iteration_index:03d}.ndjson"
        self.container_name = f"audited-proposer-{iteration_index}"
        self.attempts: list[dict[str, object]] = []

    def propose(self, request: ProposalRequest) -> ProposalV1:
        if request.parent_candidate_hash != self.proposal.parent_candidate_hash:
            raise AssertionError("unexpected frozen proposal parent")
        self.prompt_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_path.write_text(build_prompt(request), encoding="utf-8")
        raw = (
            json.dumps(
                {
                    "type": "text",
                    "part": {
                        "text": json.dumps(
                            _proposal_value(self.proposal.kernel_source),
                            sort_keys=True,
                        )
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        self.raw_path.write_text(raw, encoding="utf-8")
        self.attempts = [
            {
                "attempt": 1,
                "outcome": "SUCCESS",
                "raw_output_path": str(self.raw_path),
            }
        ]
        return self.proposal


class EchoFaultEvaluator(FakeEvaluator):
    def __init__(self, fault: str) -> None:
        super().__init__()
        self.fault = fault

    def evaluate(self, **values):  # type: ignore[no-untyped-def]
        raw = super().evaluate(**values)
        if self.fault == "missing":
            raw.pop("request_identity")
        elif self.fault == "mismatch":
            raw["candidate_hash"] = "f" * 64
        else:  # pragma: no cover - fixture construction guard
            raise AssertionError(self.fault)
        return raw


class ControllerV2IntegrationTests(unittest.TestCase):
    def _controller(
        self,
        root: Path,
        *,
        evaluator: FakeEvaluator | None = None,
    ) -> tuple[
        ResearchController,
        ProposalV1,
        list[AuditedStaticProposer],
    ]:
        config = _config(root, max_candidates=1)
        (config.repository_dir / "kernel.py").write_text(SEED, encoding="utf-8")
        (config.repository_dir / "program.md").write_text(
            "Change only kernel.py.", encoding="utf-8"
        )
        _baseline(config.state_dir)
        source = _with_different_block_size_n(SEED)
        proposal = ProposalV1.from_value(
            _proposal_value(source), expected_parent_hash=SEED_HASH
        )
        proposers: list[AuditedStaticProposer] = []

        def proposer_factory(
            _run_id: str, iteration_index: int, run_dir: Path
        ) -> AuditedStaticProposer:
            proposer = AuditedStaticProposer(
                proposal,
                run_dir=run_dir,
                iteration_index=iteration_index,
            )
            proposers.append(proposer)
            return proposer

        controller = ResearchController(
            config,
            evaluator=evaluator or FakeEvaluator(),
            runner=NoopRunner(),
            proposer_factory=proposer_factory,
        )
        return controller, proposal, proposers

    @staticmethod
    def _private_object_path(controller: ResearchController, object_id: str) -> Path:
        if not object_id.startswith("sha256:"):
            raise AssertionError(f"unexpected object id: {object_id}")
        digest = object_id.removeprefix("sha256:")
        return (
            controller.config.controller_dir
            / "objects"
            / "sha256"
            / digest[:2]
            / digest[2:]
        )

    @staticmethod
    def _start(controller: ResearchController) -> dict:
        with mock.patch.object(
            controller,
            "doctor",
            return_value={"status": "SUCCESS", "errors": []},
        ):
            return controller.start()

    def test_start_freezes_v2_provenance_and_links_four_bundle_experiments(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, proposal, proposers = self._controller(Path(temporary))
            run_result = self._start(controller)
            run_id = str(run_result["id"])
            expected_bundle = CandidateBundle.single_file(
                content=proposal.kernel_source
            )

            with ControllerStore(controller.controller_db) as store:
                run = store.get_run(run_id)
                iteration = store.latest_iteration(run_id)
                proposal_attempts = store.list_proposal_attempts(run_id)
                evaluation_attempts = store.list_evaluation_attempts(run_id)

            self.assertEqual(run["status"], "PROMOTED")
            self.assertEqual(run["initial_best_hash"], SEED_HASH)
            self.assertEqual(run["final_best_hash"], proposal.candidate_hash)
            snapshot = run["workflow_snapshot"]
            self.assertEqual(snapshot["schema_version"], 2)
            self.assertEqual(snapshot["mode"], "DISCOVERY")
            self.assertFalse(snapshot["scientifically_comparable"])
            self.assertEqual(
                snapshot["compatibility_mode"], "LEGACY_V1_EVIDENCE"
            )
            self.assertEqual(
                snapshot["execution_environment"]["status"],
                "LEGACY_UNKNOWN",
            )
            self.assertEqual(
                snapshot["runtime_execution_environment"]["status"],
                "RESOLVED",
            )
            self.assertEqual(
                snapshot["namespace"], LEGACY_RESEARCH_NAMESPACE.to_dict()
            )
            self.assertEqual(snapshot["baseline_ref"], run["baseline_ref"])
            self.assertEqual(snapshot["history_cutoff"], run["history_cutoff"])
            baseline_id = int(
                run["baseline_ref"]["revision"].removeprefix("history-")
            )
            self.assertEqual(run["history_cutoff"], baseline_id)
            self.assertEqual(
                snapshot["snapshot_digest"], run["resolved_config_digest"]
            )
            unsigned_snapshot = dict(snapshot)
            unsigned_snapshot.pop("snapshot_digest")
            self.assertEqual(
                canonical_sha256(unsigned_snapshot),
                run["resolved_config_digest"],
            )
            self.assertEqual(run["baseline_ref"]["source"], "deployment")
            self.assertEqual(
                run["baseline_ref"]["artifact_id"],
                str(ArtifactId.source_sha256(SEED_HASH)),
            )
            self.assertEqual(
                set(snapshot["resolved_profiles"]),
                {
                    "deployment",
                    "operator",
                    "language",
                    "evaluator",
                    "evaluation_protocol",
                    "promotion_policy",
                    "proposer",
                },
            )
            controller._validate_run_snapshot(run)

            self.assertEqual(len(proposal_attempts), 1)
            proposal_attempt = proposal_attempts[0]
            self.assertEqual(proposal_attempt["status"], "SUCCEEDED")
            self.assertTrue(proposal_attempt["attempt_uid"])
            self.assertGreaterEqual(float(proposal_attempt["latency_ms"]), 0.0)
            self.assertEqual(
                proposal_attempt["proposal_context_id"],
                canonical_sha256(proposal_attempt["request"]),
            )
            self.assertEqual(
                proposal_attempt["parent_artifact_id"],
                run["baseline_ref"]["artifact_id"],
            )
            self.assertEqual(
                proposal_attempt["proposer_profile"],
                snapshot["proposer_profile"],
            )
            self.assertEqual(
                proposal_attempt["candidate_artifact_id"],
                str(expected_bundle.artifact_id),
            )
            self.assertEqual(
                proposal_attempt["result"]["proposal"]["proposal_context_id"],
                proposal_attempt["proposal_context_id"],
            )
            self.assertNotIn(
                "content",
                proposal_attempt["result"]["proposal"]["candidate"]["files"][0],
            )

            self.assertEqual(len(proposers), 1)
            proposer = proposers[0]
            prompt_object_id = str(proposal_attempt["prompt_object_id"])
            self.assertEqual(
                proposal_attempt["request"]["prompt_object_id"],
                prompt_object_id,
            )
            prompt_object = self._private_object_path(
                controller, prompt_object_id
            )
            raw_object = self._private_object_path(
                controller, str(proposal_attempt["raw_output_object_id"])
            )
            self.assertEqual(
                prompt_object.read_bytes(), proposer.prompt_path.read_bytes()
            )
            self.assertEqual(raw_object.read_bytes(), proposer.raw_path.read_bytes())
            self.assertEqual(
                prompt_object_id,
                "sha256:" + hashlib.sha256(prompt_object.read_bytes()).hexdigest(),
            )

            self.assertIsNotNone(iteration)
            assert iteration is not None
            self.assertEqual(iteration["outcome"], "PROMOTED")
            expected_stages = (
                ("smoke", "smoke", "validation"),
                ("quick", "quick", "validation"),
                ("full_primary", "full", "primary"),
                ("confirmation", "full", "confirmation"),
            )
            self.assertEqual(
                [attempt["stage"] for attempt in evaluation_attempts],
                [stage for stage, _suite, _kind in expected_stages],
            )
            self.assertEqual(
                len({attempt["experiment_uid"] for attempt in evaluation_attempts}),
                4,
            )
            prompt_digest = "sha256:" + hashlib.sha256(
                proposer.prompt_path.read_bytes()
            ).hexdigest()

            with HistoryStore(
                controller.history_db, state_dir=controller.config.state_dir
            ) as history:
                baseline = history.get_experiment(
                    baseline_id
                )
                self.assertIsNotNone(baseline)
                bundle_record = history.get_candidate_artifact(
                    str(expected_bundle.artifact_id)
                )
                self.assertIsNotNone(bundle_record)
                self.assertEqual(
                    history.read_candidate_artifact(str(expected_bundle.artifact_id)),
                    expected_bundle.bundle_bytes,
                )
                for attempt, (stage, suite, replicate_kind) in zip(
                    evaluation_attempts, expected_stages, strict=True
                ):
                    self.assertEqual(attempt["status"], "SUCCEEDED")
                    self.assertIsNotNone(attempt["history_experiment_id"])
                    identity = ExperimentIdentity.from_value(attempt["request"])
                    self.assertEqual(identity.experiment_uid, attempt["experiment_uid"])
                    self.assertEqual(
                        identity.condition_digest, attempt["condition_digest"]
                    )
                    self.assertEqual(identity.stage, stage)
                    self.assertEqual(identity.suite, suite)
                    self.assertEqual(identity.replicate_kind, replicate_kind)
                    self.assertFalse(identity.is_scientifically_comparable)
                    self.assertEqual(identity.namespace_id, run["namespace_id"])
                    self.assertEqual(identity.prompt_digest, prompt_digest)
                    self.assertEqual(
                        str(identity.candidate_artifact_id),
                        str(expected_bundle.artifact_id),
                    )
                    self.assertEqual(
                        str(identity.parent_artifact_id),
                        run["baseline_ref"]["artifact_id"],
                    )
                    record = history.get_experiment_by_uid(
                        identity.experiment_uid
                    )
                    self.assertIsNotNone(record)
                    assert record is not None
                    self.assertEqual(record.id, attempt["history_experiment_id"])
                    self.assertEqual(
                        record.artifact_id, str(expected_bundle.artifact_id)
                    )
                    self.assertEqual(
                        iteration["experiment_ids"][stage], record.id
                    )
                    self.assertEqual(
                        record.baseline_experiment_uid,
                        None if baseline is None else baseline.experiment_uid,
                    )

                scientific_best = history.get_best_for_namespace(
                    run["namespace_id"], backend="c500", suite="full"
                )
            self.assertIsNotNone(scientific_best)
            assert scientific_best is not None
            self.assertEqual(scientific_best.candidate_hash, proposal.candidate_hash)

            # Promotion only creates evidence. The explicit deployment pin and
            # repository source remain on the original accepted baseline.
            self.assertEqual(controller._best().candidate_hash, SEED_HASH)
            self.assertEqual(
                hashlib.sha256(
                    (controller.config.repository_dir / "kernel.py").read_bytes()
                ).hexdigest(),
                SEED_HASH,
            )

    def test_controller_environment_change_and_legacy_baseline_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, _proposal, _proposers = self._controller(Path(temporary))
            completed = self._start(controller)
            with ControllerStore(controller.controller_db) as store:
                stored = store.get_run(str(completed["id"]))
                iteration = store.latest_iteration(str(completed["id"]))
            assert iteration is not None

            run = json.loads(json.dumps(stored))
            snapshot = run["workflow_snapshot"]
            actual = ExecutionEnvironmentDigest.from_value(
                snapshot["runtime_execution_environment"]
            )
            changed = ExecutionEnvironmentDigest.resolved(
                evaluator_image_digest=actual.evaluator_image_digest,
                toolchain_digest=actual.toolchain_digest,
                framework_digest="sha256:" + "f" * 64,
                operator_abi_digest=actual.operator_abi_digest,
                build_flags_digest=actual.build_flags_digest,
            )
            snapshot["runtime_execution_environment"] = changed.to_dict()
            unsigned = dict(snapshot)
            unsigned.pop("snapshot_digest")
            snapshot["snapshot_digest"] = canonical_sha256(unsigned)
            run["resolved_config_digest"] = snapshot["snapshot_digest"]
            with self.assertRaisesRegex(
                Exception, "actual evaluator execution environment differs"
            ):
                controller._validate_run_snapshot(run)

            # Rebinding a legacy accepted artifact to today's environment is
            # not evidence that it was measured there.  A comparable V2
            # snapshot can carry the binding, but baseline lookup must prove it
            # from the immutable History identity and therefore rejects this.
            run = json.loads(json.dumps(stored))
            snapshot = run["workflow_snapshot"]
            resolved_baseline = BaselineRef.create(
                namespace=LEGACY_RESEARCH_NAMESPACE,
                artifact_id=run["baseline_ref"]["artifact_id"],
                source="deployment",
                revision=run["baseline_ref"]["revision"],
                execution_environment=actual,
            )
            run["baseline_ref"] = resolved_baseline.to_dict()
            snapshot["baseline_ref"] = resolved_baseline.to_dict()
            snapshot["execution_environment"] = actual.to_dict()
            snapshot["runtime_execution_environment"] = actual.to_dict()
            snapshot["scientifically_comparable"] = True
            snapshot.pop("compatibility_mode")
            unsigned = dict(snapshot)
            unsigned.pop("snapshot_digest")
            snapshot["snapshot_digest"] = canonical_sha256(unsigned)
            run["resolved_config_digest"] = snapshot["snapshot_digest"]
            controller._validate_run_snapshot(run)
            active_identity = controller._experiment_identity(
                run=run,
                iteration=iteration,
                suite="quick",
                stage="quick",
            )
            self.assertTrue(active_identity.is_scientifically_comparable)
            self.assertEqual(active_identity.execution_environment, actual)
            with self.assertRaisesRegex(
                Exception, "no matching V2 execution evidence"
            ):
                controller._baseline_for_ref(
                    resolved_baseline,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                )

    def test_missing_or_mismatched_echo_is_unknown_and_writes_no_experiment(
        self,
    ) -> None:
        for fault in ("missing", "mismatch"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                evaluator = EchoFaultEvaluator(fault)
                controller, proposal, _proposers = self._controller(
                    Path(temporary), evaluator=evaluator
                )
                run = self._start(controller)
                run_id = str(run["id"])

                self.assertEqual(run["status"], "HARD_FAILED")
                with ControllerStore(controller.controller_db) as store:
                    iteration = store.latest_iteration(run_id)
                    attempts = store.list_evaluation_attempts(run_id)
                self.assertIsNotNone(iteration)
                assert iteration is not None
                self.assertEqual(iteration["outcome"], "UNKNOWN_GPU_OUTCOME")
                self.assertEqual(iteration["experiment_ids"], {})
                self.assertEqual(len(attempts), 1)
                self.assertEqual(attempts[0]["stage"], "smoke")
                self.assertEqual(attempts[0]["status"], "UNKNOWN_OUTCOME")
                self.assertIsNone(attempts[0]["history_experiment_id"])
                self.assertEqual(evaluator.stages, ["smoke"])
                with HistoryStore(
                    controller.history_db,
                    state_dir=controller.config.state_dir,
                ) as history:
                    self.assertEqual(
                        history.find_by_candidate_hash(
                            proposal.candidate_hash,
                            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                        ),
                        [],
                    )
                    self.assertIsNone(
                        history.get_experiment_by_uid(
                            attempts[0]["experiment_uid"]
                        )
                    )

    def test_grafted_history_uid_is_rejected_before_controller_linkage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = FakeEvaluator()
            controller, _proposal, _proposers = self._controller(
                Path(temporary), evaluator=evaluator
            )
            with mock.patch.object(
                controller,
                "doctor",
                return_value={"status": "SUCCESS", "errors": []},
            ):
                completed = controller.start(proposal_only=True)
            run_id = str(completed["id"])

            with ControllerStore(controller.controller_db) as store:
                run = store.get_run(run_id)
                iteration = store.latest_iteration(run_id)
                self.assertIsNotNone(iteration)
                assert iteration is not None
                candidate_path = Path(str(iteration["candidate_path"]))
                source = candidate_path.read_text(encoding="utf-8")
                identity = controller._experiment_identity(
                    run=run,
                    iteration=iteration,
                    suite="smoke",
                    stage="smoke",
                )
                store.create_evaluation_attempt(
                    experiment_uid=identity.experiment_uid,
                    run_id=run_id,
                    iteration_id=int(iteration["id"]),
                    stage="smoke",
                    suite="smoke",
                    replicate_kind=identity.replicate_kind,
                    replicate_index=identity.replicate_index,
                    candidate_artifact_id=str(identity.candidate_artifact_id),
                    parent_artifact_id=str(identity.parent_artifact_id),
                    baseline_ref=identity.baseline.to_dict(),
                    condition_digest=identity.condition_digest,
                    request=identity.to_dict(),
                )

                grafted_value = identity.to_dict()
                grafted_value["run_id"] = "grafted-run"
                grafted_identity = ExperimentIdentity.from_value(
                    grafted_value
                )
                self.assertEqual(
                    grafted_identity.condition_digest,
                    identity.condition_digest,
                )
                self.assertNotEqual(grafted_identity, identity)
                with HistoryStore(
                    controller.history_db,
                    state_dir=controller.config.state_dir,
                ) as history:
                    grafted = history.record_experiment(
                        candidate_source=source,
                        candidate_hash=str(iteration["candidate_hash"]),
                        backend="c500",
                        suite="smoke",
                        status="SUCCESS",
                        identity=grafted_identity,
                        result={"status": "SUCCESS"},
                    )
                self.assertEqual(grafted.experiment_uid, identity.experiment_uid)
                event_count = len(store.list_events(run_id))

                with self.assertRaisesRegex(
                    ControllerDataIntegrityError,
                    "does not exactly match",
                ):
                    controller._evaluate_stage(
                        store=store,
                        run_id=run_id,
                        iteration=iteration,
                        source=source,
                        candidate_path=candidate_path,
                        suite="smoke",
                        stage="smoke",
                    )

                attempt = store.get_evaluation_attempt_by_uid(
                    identity.experiment_uid
                )
                self.assertIsNotNone(attempt)
                assert attempt is not None
                self.assertEqual(attempt["status"], "PENDING")
                self.assertIsNone(attempt["history_experiment_id"])
                self.assertEqual(attempt["result"], {})
                self.assertEqual(
                    store.get_iteration(int(iteration["id"]))[
                        "experiment_ids"
                    ],
                    {},
                )
                self.assertEqual(len(store.list_events(run_id)), event_count)
            self.assertEqual(evaluator.stages, [])

    def test_matching_history_uid_reconciles_pending_attempt_without_evaluator(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = FakeEvaluator()
            controller, _proposal, _proposers = self._controller(
                Path(temporary), evaluator=evaluator
            )
            with mock.patch.object(
                controller,
                "doctor",
                return_value={"status": "SUCCESS", "errors": []},
            ):
                completed = controller.start(proposal_only=True)
            run_id = str(completed["id"])

            with ControllerStore(controller.controller_db) as store:
                run = store.get_run(run_id)
                iteration = store.latest_iteration(run_id)
                self.assertIsNotNone(iteration)
                assert iteration is not None
                candidate_path = Path(str(iteration["candidate_path"]))
                source = candidate_path.read_text(encoding="utf-8")
                identity = controller._experiment_identity(
                    run=run,
                    iteration=iteration,
                    suite="smoke",
                    stage="smoke",
                )
                store.create_evaluation_attempt(
                    experiment_uid=identity.experiment_uid,
                    run_id=run_id,
                    iteration_id=int(iteration["id"]),
                    stage="smoke",
                    suite="smoke",
                    replicate_kind=identity.replicate_kind,
                    replicate_index=identity.replicate_index,
                    candidate_artifact_id=str(identity.candidate_artifact_id),
                    parent_artifact_id=str(identity.parent_artifact_id),
                    baseline_ref=identity.baseline.to_dict(),
                    condition_digest=identity.condition_digest,
                    request=identity.to_dict(),
                )
                with HistoryStore(
                    controller.history_db,
                    state_dir=controller.config.state_dir,
                ) as history:
                    persisted = history.record_experiment(
                        candidate_source=source,
                        candidate_hash=str(iteration["candidate_hash"]),
                        backend="c500",
                        suite="smoke",
                        status="SUCCESS",
                        identity=identity,
                        result={"status": "SUCCESS"},
                    )

                reconciled = controller._evaluate_stage(
                    store=store,
                    run_id=run_id,
                    iteration=iteration,
                    source=source,
                    candidate_path=candidate_path,
                    suite="smoke",
                    stage="smoke",
                )
                self.assertEqual(reconciled.id, persisted.id)
                attempt = store.get_evaluation_attempt_by_uid(
                    identity.experiment_uid
                )
                self.assertIsNotNone(attempt)
                assert attempt is not None
                self.assertEqual(attempt["status"], "SUCCEEDED")
                self.assertEqual(
                    attempt["history_experiment_id"], persisted.id
                )
                self.assertEqual(
                    store.get_iteration(int(iteration["id"]))[
                        "experiment_ids"
                    ],
                    {"smoke": persisted.id},
                )
            self.assertEqual(evaluator.stages, [])

    def test_duplicate_detection_is_isolated_by_run_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, _proposal, _proposers = self._controller(Path(temporary))
            legacy_ref = BaselineRef.create(
                namespace=LEGACY_RESEARCH_NAMESPACE,
                artifact_id=ArtifactId.source_sha256(SEED_HASH),
                source="deployment",
                revision="legacy-fixture",
            )
            current_ref = BaselineRef.create(
                namespace=CURRENT_RESEARCH_NAMESPACE,
                artifact_id=ArtifactId.source_sha256(SEED_HASH),
                source="deployment",
                revision="current-fixture",
            )
            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id="legacy-run",
                    deadline_epoch=time.time() + 60,
                    config={},
                    initial_best_hash=SEED_HASH,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    workflow_snapshot={"schema_version": 2},
                    baseline_ref=legacy_ref.to_dict(),
                )
                legacy_iteration = store.create_iteration(
                    "legacy-run", 1, SEED_HASH
                )
                store.accept_candidate(
                    int(legacy_iteration["id"]),
                    candidate_hash=SEED_HASH,
                    hypothesis="legacy",
                    rationale="legacy namespace fixture",
                    candidate_path="/legacy/kernel.py",
                )
                store.create_run(
                    run_id="current-run",
                    deadline_epoch=time.time() + 60,
                    config={},
                    initial_best_hash=SEED_HASH,
                    namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    workflow_snapshot={"schema_version": 2},
                    baseline_ref=current_ref.to_dict(),
                )

                # The same bytes exist in both legacy History and another
                # Controller run, but neither crosses the namespace boundary.
                self.assertFalse(
                    controller._candidate_seen(
                        store,
                        run_id="current-run",
                        candidate_hash=SEED_HASH,
                    )
                )
                current_iteration = store.create_iteration(
                    "current-run", 1, SEED_HASH
                )
                store.accept_candidate(
                    int(current_iteration["id"]),
                    candidate_hash=SEED_HASH,
                    hypothesis="current",
                    rationale="current namespace fixture",
                    candidate_path="/current/kernel.py",
                )
                self.assertTrue(
                    controller._candidate_seen(
                        store,
                        run_id="current-run",
                        candidate_hash=SEED_HASH,
                    )
                )

    def test_recent_feedback_filters_namespace_before_the_bounded_cutoff(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, _proposal, _proposers = self._controller(Path(temporary))
            current_ref = BaselineRef.create(
                namespace=CURRENT_RESEARCH_NAMESPACE,
                artifact_id=ArtifactId.source_sha256(SEED_HASH),
                source="deployment",
                revision="current-feedback-fixture",
            )
            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id="legacy-visible",
                    deadline_epoch=time.time() + 60,
                    config={},
                    initial_best_hash=SEED_HASH,
                )
                visible = store.create_iteration(
                    "legacy-visible", 1, SEED_HASH
                )
                visible_hash = hashlib.sha256(b"legacy-visible").hexdigest()
                visible = store.accept_candidate(
                    int(visible["id"]),
                    candidate_hash=visible_hash,
                    hypothesis="same namespace evidence",
                    rationale="must survive the cutoff",
                    candidate_path="/legacy-visible/kernel.py",
                )
                controller._finish_iteration(
                    store,
                    int(visible["id"]),
                    outcome="PRECISION_FAILED",
                    result={"status": "PRECISION_FAILED"},
                )

                # These newer rows would hide the legacy row if namespace
                # filtering happened after the SQL LIMIT.
                for index in range(MAX_FEEDBACK_CANDIDATES):
                    run_id = f"current-foreign-{index}"
                    store.create_run(
                        run_id=run_id,
                        deadline_epoch=time.time() + 60,
                        config={},
                        initial_best_hash=SEED_HASH,
                        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                        workflow_snapshot={"schema_version": 2},
                        baseline_ref=current_ref.to_dict(),
                    )
                    foreign = store.create_iteration(run_id, 1, SEED_HASH)
                    foreign = store.accept_candidate(
                        int(foreign["id"]),
                        candidate_hash=hashlib.sha256(
                            f"foreign-{index}".encode("ascii")
                        ).hexdigest(),
                        hypothesis=f"foreign namespace {index}",
                        rationale="must not enter legacy feedback",
                        candidate_path=f"/foreign-{index}/kernel.py",
                    )
                    controller._finish_iteration(
                        store,
                        int(foreign["id"]),
                        outcome="PRECISION_FAILED",
                        result={"status": "PRECISION_FAILED"},
                    )

                store.create_run(
                    run_id="legacy-target",
                    deadline_epoch=time.time() + 60,
                    config={},
                    initial_best_hash=SEED_HASH,
                )

            request = controller._proposal_request("legacy-target")
            self.assertEqual(len(request.recent_experiments), 1)
            self.assertEqual(
                request.recent_experiments[0]["candidate_hash"], visible_hash
            )
            self.assertEqual(
                request.recent_experiments[0]["hypothesis"],
                "same namespace evidence",
            )


if __name__ == "__main__":
    unittest.main()
