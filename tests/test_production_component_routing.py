from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from kernel_research.autorun.controller import (
    ControllerDataIntegrityError,
    DockerEvaluator,
    _trusted_target,
)
from kernel_research.autorun.runtime import CommandResult
from kernel_research.autorun.store import ControllerStore
from kernel_research.evaluation import raw_evaluate
from kernel_research.contract import CandidateError, CandidateValidation
from kernel_research.platform.canonical import canonical_sha256
from kernel_research.platform.legacy import resolve_builtin_target
from kernel_research.platform.profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    ResearchNamespace,
)

from test_autorun import SEED, _config
import test_controller_v2_integration as controller_fixtures


ROOT = Path(__file__).resolve().parents[1]


class _CaptureRunner:
    def __init__(self) -> None:
        self.argv: tuple[str, ...] | None = None

    def run(self, argv, **_kwargs):  # type: ignore[no-untyped-def]
        self.argv = tuple(str(value) for value in argv)
        return CommandResult(
            argv=self.argv,
            returncode=3,
            stdout=json.dumps({"status": "CRASH", "error": "fixture"}),
            stderr="",
        )


class ProductionComponentRoutingTests(unittest.TestCase):
    def test_candidate_policy_merges_language_and_operator_axes(self) -> None:
        target = resolve_builtin_target(CURRENT_RESEARCH_NAMESPACE)

        class RejectingOperator:
            operator_id = target.operator.operator_id
            implementation_id = target.operator.implementation_id
            revision = target.operator.revision

            def validate_operator_candidate(self, source: str):  # type: ignore[no-untyped-def]
                return CandidateValidation(
                    source=source,
                    sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    errors=(
                        CandidateError(
                            code="OPERATOR_FIXTURE_REJECTED",
                            message="fixture operator invariant",
                        ),
                    ),
                )

        decision = replace(target, operator=RejectingOperator()).validate_candidate(
            SEED
        )
        self.assertFalse(decision.valid)
        self.assertEqual(
            decision.errors[-1].code, "OPERATOR_FIXTURE_REJECTED"
        )

    def test_controller_start_exercises_registry_and_freezes_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = controller_fixtures.ControllerV2IntegrationTests()
            controller, _proposal, _proposers = fixture._controller(
                Path(temporary)
            )
            from kernel_research.autorun import controller as controller_module

            real_resolver = controller_module._resolve_builtin_target
            with mock.patch.object(
                controller_module,
                "_resolve_builtin_target",
                wraps=real_resolver,
            ) as resolver, mock.patch.object(
                controller,
                "doctor",
                return_value={"status": "SUCCESS", "errors": []},
            ):
                terminal = controller.start()

            self.assertEqual(terminal["status"], "PROMOTED")
            self.assertGreaterEqual(resolver.call_count, 6)
            resolved_namespaces = {
                call.args[0].namespace_id for call in resolver.call_args_list
            }
            self.assertEqual(
                resolved_namespaces,
                {LEGACY_RESEARCH_NAMESPACE.namespace_id},
            )
            with ControllerStore(controller.controller_db) as store:
                run = store.get_run(str(terminal["id"]))
            snapshot = run["workflow_snapshot"]
            target = resolve_builtin_target(LEGACY_RESEARCH_NAMESPACE)
            self.assertEqual(
                snapshot["target_components"]["evaluator"]["backend"],
                target.device.evaluator_backend(),
            )
            self.assertEqual(
                snapshot["workflow"],
                [stage.stage_id for stage in target.protocol.stages],
            )

    def test_forged_profile_change_fails_even_with_recomputed_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = controller_fixtures.ControllerV2IntegrationTests()
            controller, _proposal, _proposers = fixture._controller(
                Path(temporary)
            )
            with mock.patch.object(
                controller,
                "doctor",
                return_value={"status": "SUCCESS", "errors": []},
            ):
                terminal = controller.start()
            with ControllerStore(controller.controller_db) as store:
                run = store.get_run(str(terminal["id"]))

            forged = copy.deepcopy(run)
            snapshot = forged["workflow_snapshot"]
            snapshot["resolved_profiles"]["language"]["config"][
                "entrypoint"
            ] = "other.py"
            unsigned = dict(snapshot)
            unsigned.pop("snapshot_digest")
            snapshot["snapshot_digest"] = canonical_sha256(unsigned)
            forged["resolved_config_digest"] = snapshot["snapshot_digest"]
            with self.assertRaisesRegex(
                ControllerDataIntegrityError, "exact builtin definition"
            ):
                controller._validate_run_snapshot(forged)

    def test_inactive_language_cannot_enter_production_route(self) -> None:
        inactive_language = BUILTIN_PROFILE_REGISTRY.get(
            kind="language",
            profile_id="tilelang-python",
            revision="v0-probe",
        ).ref
        inactive_namespace = ResearchNamespace(
            operator=CURRENT_RESEARCH_NAMESPACE.operator,
            language=inactive_language,
            evaluator=CURRENT_RESEARCH_NAMESPACE.evaluator,
            evaluation_protocol=(
                CURRENT_RESEARCH_NAMESPACE.evaluation_protocol
            ),
            promotion_policy=CURRENT_RESEARCH_NAMESPACE.promotion_policy,
        )
        with self.assertRaisesRegex(
            ControllerDataIntegrityError, "trusted component binding"
        ):
            _trusted_target(inactive_namespace)

    def test_docker_evaluator_argv_comes_from_current_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            candidate = root / "candidate.py"
            candidate.write_text(SEED, encoding="utf-8")
            runner = _CaptureRunner()
            evaluator = DockerEvaluator(
                config,
                controller_dir=config.controller_dir,
                runner=runner,  # type: ignore[arg-type]
            )
            identity = {
                "namespace": CURRENT_RESEARCH_NAMESPACE.to_dict(),
                "candidate_artifact_id": "source-sha256-v1:" + "a" * 64,
            }
            evaluator.evaluate(
                candidate_path=candidate,
                suite="smoke",
                baseline_path=None,
                run_id="route",
                iteration_index=1,
                stage="smoke",
                request_identity=identity,
            )
            self.assertIsNotNone(runner.argv)
            argv = runner.argv or ()
            backend_index = argv.index("--backend")
            protocol_index = argv.index("--evaluation-protocol")
            target = resolve_builtin_target(CURRENT_RESEARCH_NAMESPACE)
            self.assertEqual(
                argv[backend_index + 1], target.device.evaluator_backend()
            )
            self.assertEqual(
                argv[protocol_index + 1], target.protocol.protocol_id
            )

    def test_legacy_current_shared_fixture_contracts_are_equivalent(self) -> None:
        legacy = resolve_builtin_target(LEGACY_RESEARCH_NAMESPACE)
        current = resolve_builtin_target(CURRENT_RESEARCH_NAMESPACE)
        self.assertEqual(
            legacy.validate_candidate(SEED).to_dict(),
            current.validate_candidate(SEED).to_dict(),
        )
        self.assertEqual(
            legacy.device.evaluator_backend(),
            current.device.evaluator_backend(),
        )
        self.assertEqual(
            [stage.stage_id for stage in legacy.protocol.stages],
            [stage.stage_id for stage in current.protocol.stages],
        )
        for suite in ("smoke", "full"):
            self.assertEqual(
                legacy.protocol.expected_cases(suite),
                current.protocol.expected_cases(suite),
            )
        self.assertEqual(
            legacy.protocol.scored_cases("quick"),
            current.protocol.scored_cases("quick"),
        )

    def test_raw_evaluator_re_resolves_both_v2_protocols(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "kernel.py"
            candidate.write_text(SEED, encoding="utf-8")
            from kernel_research.platform import legacy as legacy_module

            real_resolver = legacy_module.resolve_builtin_target
            with mock.patch.object(
                legacy_module,
                "resolve_builtin_target",
                wraps=real_resolver,
            ) as resolver, mock.patch(
                "kernel_research.evaluation.evaluate_isolated",
                return_value={
                    "status": "SUCCESS",
                    "eligible_for_promotion": False,
                    "aggregate_score": None,
                    "cases": [],
                    "environment": {},
                },
            ) as isolated:
                for index, namespace in enumerate(
                    (
                        LEGACY_RESEARCH_NAMESPACE,
                        CURRENT_RESEARCH_NAMESPACE,
                    )
                ):
                    identity_path = root / f"identity-{index}.json"
                    identity_path.write_text(
                        json.dumps({"namespace": namespace.to_dict()}),
                        encoding="utf-8",
                    )
                    target = real_resolver(namespace)
                    result = raw_evaluate(
                        candidate,
                        backend=target.device.evaluator_backend(),
                        suite="smoke",
                        request_identity_path=identity_path,
                        expected_evaluation_protocol_id=(
                            target.protocol.protocol_id
                        ),
                    )
                    self.assertEqual(
                        result["evaluation_protocol_id"],
                        target.protocol.protocol_id,
                    )
                    self.assertEqual(
                        isolated.call_args.kwargs["backend"],
                        target.device.evaluator_backend(),
                    )
                    self.assertEqual(
                        isolated.call_args.kwargs["evaluation_protocol_id"],
                        target.protocol.protocol_id,
                    )
            self.assertEqual(resolver.call_count, 2)

    def test_importing_controller_does_not_import_accelerator_stacks(self) -> None:
        script = """
import sys
import kernel_research.autorun.controller
assert 'kernel_research.platform.legacy' not in sys.modules
assert 'numpy' not in sys.modules
assert 'torch' not in sys.modules
assert 'triton' not in sys.modules
print('LIGHT_CONTROLLER_IMPORT_OK')
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(
            completed.stdout.strip(), "LIGHT_CONTROLLER_IMPORT_OK"
        )


if __name__ == "__main__":
    unittest.main()
