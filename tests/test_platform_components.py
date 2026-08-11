from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest

from kernel_research.platform import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    CandidateBundle,
    TRITON_PYTHON_BUNDLE_LIMITS,
)
from kernel_research.platform.components import (
    CaseRole,
    EvaluationProtocolDefinition,
    ReplicateKind,
    StageDefinition,
    TrustedComponentRegistry,
)
from kernel_research.platform.legacy import (
    CURRENT_C500_PROTOCOL,
    CURRENT_TARGET,
    LEGACY_C500_PROTOCOL,
    LEGACY_TARGET,
    builtin_component_registry,
    resolve_builtin_target,
)


ROOT = Path(__file__).resolve().parents[1]


class ComponentContractTests(unittest.TestCase):
    def test_legacy_and_current_targets_have_distinct_protocols(self) -> None:
        self.assertEqual(CURRENT_TARGET.operator.operator_id, "fused-moe-w8a8-tn")
        self.assertEqual(CURRENT_TARGET.operator.revision, "v1")
        self.assertEqual(CURRENT_TARGET.language.language_id, "triton-python")
        self.assertEqual(CURRENT_TARGET.language.revision, "v1")
        self.assertEqual(CURRENT_TARGET.device.device_id, "metax-c500")
        self.assertEqual(CURRENT_TARGET.device.revision, "legacy-v1")
        self.assertEqual(len(LEGACY_C500_PROTOCOL.expected_cases("quick")), 4)
        self.assertNotIn(
            "quick_shadow_tiles_128_n2",
            LEGACY_C500_PROTOCOL.case_roles,
        )
        self.assertEqual(CURRENT_C500_PROTOCOL.samples_per_case, 30)
        self.assertEqual(len(CURRENT_C500_PROTOCOL.expected_cases("quick")), 8)
        self.assertEqual(
            CURRENT_C500_PROTOCOL.scored_cases("quick"),
            (
                "quick_decode_gate_up",
                "quick_prefill_gate_up",
                "quick_decode_down",
                "quick_prefill_down",
            ),
        )
        self.assertIs(
            CURRENT_C500_PROTOCOL.case_roles["quick_shadow_tiles_128_n2"],
            CaseRole.CORRECTNESS_ONLY,
        )
        self.assertIs(
            CURRENT_C500_PROTOCOL.case_roles["quick_shadow_tiles_128_n1"],
            CaseRole.HOLDOUT,
        )
        self.assertIs(
            CURRENT_C500_PROTOCOL.case_roles["quick_shadow_tiles_129_n2"],
            CaseRole.HOLDOUT,
        )
        self.assertNotEqual(
            LEGACY_RESEARCH_NAMESPACE.namespace_id,
            CURRENT_RESEARCH_NAMESPACE.namespace_id,
        )

    def test_registry_is_closed_and_rejects_dynamic_unknowns(self) -> None:
        registry = builtin_component_registry()
        self.assertTrue(registry.sealed)
        self.assertIs(
            registry.resolve(
                kind="operator", component_id="fused-moe-w8a8-tn", revision="v1"
            ),
            CURRENT_TARGET.operator,
        )
        with self.assertRaises(ValueError):
            registry.resolve(
                kind="operator", component_id="dynamic.import", revision="v1"
            )
        with self.assertRaises(RuntimeError):
            registry.register(
                kind="operator", component_id="other", revision="v1", component=object()
            )

    def test_exact_profiles_resolve_to_the_five_target_components(self) -> None:
        registry = builtin_component_registry()
        bindings = (
            (CURRENT_RESEARCH_NAMESPACE.operator, CURRENT_TARGET.operator),
            (CURRENT_RESEARCH_NAMESPACE.language, CURRENT_TARGET.language),
            (CURRENT_RESEARCH_NAMESPACE.evaluator, CURRENT_TARGET.device),
            (
                CURRENT_RESEARCH_NAMESPACE.evaluation_protocol,
                CURRENT_TARGET.protocol,
            ),
            (
                CURRENT_RESEARCH_NAMESPACE.promotion_policy,
                CURRENT_TARGET.promotion,
            ),
            (
                LEGACY_RESEARCH_NAMESPACE.evaluation_protocol,
                LEGACY_TARGET.protocol,
            ),
        )
        for ref, expected in bindings:
            with self.subTest(kind=ref.kind):
                definition = BUILTIN_PROFILE_REGISTRY.resolve(ref)
                self.assertEqual(
                    getattr(expected, "revision"), definition.ref.revision
                )
                self.assertEqual(
                    getattr(expected, "implementation_id"),
                    definition.implementation_id,
                )
                self.assertIs(registry.resolve_profile(ref), expected)
        self.assertEqual(
            resolve_builtin_target(LEGACY_RESEARCH_NAMESPACE), LEGACY_TARGET
        )
        self.assertEqual(
            resolve_builtin_target(CURRENT_RESEARCH_NAMESPACE), CURRENT_TARGET
        )
        self.assertEqual(resolve_builtin_target(), CURRENT_TARGET)

        tampered = replace(
            LEGACY_RESEARCH_NAMESPACE.language,
            digest="sha256:" + "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "digest"):
            registry.resolve_profile(tampered)

    def test_triton_adapter_owns_the_strict_single_file_bundle_policy(self) -> None:
        adapter = CURRENT_TARGET.language
        self.assertIs(adapter.bundle_limits, TRITON_PYTHON_BUNDLE_LIMITS)
        self.assertEqual(adapter.allowed_paths(), ("kernel.py",))
        bundle = CandidateBundle.single_file(content="def run_kernel(): pass\n")
        self.assertIs(bundle.validate(adapter.bundle_limits), bundle)

        two_files = {
            "format": "source_bundle_v1",
            "entrypoint": "kernel.py",
            "files": [
                {
                    "path": "kernel.py",
                    "media_type": "text/x-python",
                    "content": "def run_kernel(): pass\n",
                },
                {
                    "path": "helper.py",
                    "media_type": "text/x-python",
                    "content": "VALUE = 1\n",
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "max_files"):
            CandidateBundle.from_value(two_files, limits=adapter.bundle_limits)

    def test_platform_and_component_contract_imports_remain_dependency_light(
        self,
    ) -> None:
        script = """
import builtins
import sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'triton'}:
        raise AssertionError('forbidden import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import kernel_research.platform
import kernel_research.platform.components
assert 'kernel_research.platform.legacy' not in sys.modules
print('LIGHT_PLATFORM_IMPORT_OK')
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
            completed.stdout.strip(), "LIGHT_PLATFORM_IMPORT_OK"
        )

    def test_protocol_validation_rejects_untrusted_shape(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationProtocolDefinition(
                protocol_id="bad",
                revision="1",
                implementation_id="bad",
                stages=(
                    StageDefinition(
                        "FULL", "missing", ReplicateKind.PRIMARY, requires_baseline=True
                    ),
                ),
                suite_cases={},
                case_roles={},
                warmup_iterations=1,
                measurement_rounds=1,
                samples_per_round=1,
                interleaved_baseline=True,
            )

        registry = TrustedComponentRegistry()
        with self.assertRaises(ValueError):
            registry.register(
                kind="operator",
                component_id="bad id",
                revision="1",
                component=object(),
            )


if __name__ == "__main__":
    unittest.main()
