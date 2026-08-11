from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from kernel_research.platform.components import LanguageAdapter, OperatorPack
from kernel_research.platform.profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
)
from kernel_research.platform.proposal import (
    SOURCE_BUNDLE_V1,
    CandidateBundle,
    CandidateFile,
)
from kernel_research.platform.next_targets import (
    DESCRIPTOR_ONLY_OPERATORS,
    INACTIVE,
    INACTIVE_OPERATOR_DESCRIPTORS,
    MACA_CUDA_ABI,
    MACA_CUDA_BUILD_FLAGS,
    MACA_CUDA_BUNDLE_LIMITS,
    RAGGED_PREFILL_DESCRIPTOR,
    TILELANG_C500_DOCTOR_RECIPE,
    TILELANG_C500_MICRO_RECIPE,
    CompiledMacaBinaryRef,
    FixedProbeRecipe,
    MACACudaAdapter,
    ManualFullEvidence,
    ProbeObservation,
    ProbeStatus,
    RaggedPrefillPack,
    TileLangPythonAdapter,
    resolve_inactive_operator_descriptor,
)


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
UUID_1 = "00000000-0000-4000-8000-000000000001"
UUID_2 = "00000000-0000-4000-8000-000000000002"


def maca_source() -> str:
    return """#include "autoresearch_maca_kernel_abi_v1.h"
extern "C" int autoresearch_kernel_v1(
    const AutoresearchKernelArgsV1* args,
    AutoresearchKernelResultV1* result,
    void* stream) {
    return 0;
}
"""


def maca_bundle(source: str | None = None) -> CandidateBundle:
    return CandidateBundle(
        SOURCE_BUNDLE_V1,
        "kernel.cu",
        (
            CandidateFile(
                "kernel.cu",
                "text/x-cuda",
                maca_source() if source is None else source,
            ),
        ),
    ).validate(MACA_CUDA_BUNDLE_LIMITS)


class _MissingRunner:
    def __init__(self) -> None:
        self.ran = False

    def supports(self, capability_id: str) -> bool:
        return False

    def run_fixed(self, recipe: FixedProbeRecipe) -> ProbeObservation:
        self.ran = True
        raise AssertionError("missing capability must not execute")


class _CapturingRunner:
    def __init__(self) -> None:
        self.capabilities: list[str] = []
        self.recipes: list[FixedProbeRecipe] = []

    def supports(self, capability_id: str) -> bool:
        self.capabilities.append(capability_id)
        return True

    def run_fixed(self, recipe: FixedProbeRecipe) -> ProbeObservation:
        self.recipes.append(recipe)
        return ProbeObservation.passed(
            recipe,
            evidence_digest=SHA_A,
            image_digest=SHA_B,
            toolchain_fingerprint="tilelang-test-fixture-v1",
        )


class ProfileDescriptorTests(unittest.TestCase):
    def test_profiles_are_builtin_but_do_not_change_existing_namespaces(
        self,
    ) -> None:
        expected_profiles = {
            ("language", "tilelang-python", "v0-probe"),
            ("language", "maca-cuda", "v0-contract"),
            ("operator", "ragged-prefill", "v0-inactive"),
            ("operator", "paged-decode", "v0-inactive"),
            ("operator", "paged-prefill", "v0-inactive"),
            ("operator", "kv-cache-decode", "v0-inactive"),
            ("operator", "mla", "v0-inactive"),
        }
        for kind, profile_id, revision in expected_profiles:
            with self.subTest(profile_id=profile_id):
                definition = BUILTIN_PROFILE_REGISTRY.get(
                    kind=kind,
                    profile_id=profile_id,
                    revision=revision,
                )
                self.assertEqual(definition.config["activation_state"], INACTIVE)
                self.assertFalse(definition.config["agent_enabled"])
                self.assertFalse(definition.config["promotion_eligible"])

        self.assertEqual(
            LEGACY_RESEARCH_NAMESPACE.namespace_id,
            "sha256:ec67b4d65a8b734e336d78c4459a1f159"
            "1ba5635e50317c9be32762865e0a6b4",
        )
        self.assertEqual(
            CURRENT_RESEARCH_NAMESPACE.namespace_id,
            "sha256:ed7f99e4d7108646ff0477b118f301ed1"
            "eba4637e30105dd11c7918318dcf096",
        )

    def test_four_future_operators_are_descriptor_only(self) -> None:
        self.assertEqual(
            tuple(item.profile.id for item in DESCRIPTOR_ONLY_OPERATORS),
            ("paged-decode", "paged-prefill", "kv-cache-decode", "mla"),
        )
        for descriptor in DESCRIPTOR_ONLY_OPERATORS:
            self.assertEqual(descriptor.implementation_status, "descriptor-only")
            self.assertEqual(descriptor.activation_state, INACTIVE)
            self.assertFalse(descriptor.agent_enabled)
            self.assertFalse(descriptor.promotion_eligible)
            self.assertEqual(descriptor.abi_id, "unpublished")
            self.assertEqual(descriptor.oracle_id, "unpublished")

        ragged = INACTIVE_OPERATOR_DESCRIPTORS["ragged-prefill"]
        self.assertEqual(ragged.implementation_status, "oracle-ready")

    def test_descriptor_resolution_is_exact_and_digest_bound(self) -> None:
        ref = RAGGED_PREFILL_DESCRIPTOR.profile
        self.assertEqual(
            resolve_inactive_operator_descriptor(ref),
            RAGGED_PREFILL_DESCRIPTOR,
        )
        tampered = replace(ref, digest=SHA_A)
        with self.assertRaisesRegex(ValueError, "digest"):
            resolve_inactive_operator_descriptor(tampered)
        with self.assertRaisesRegex(ValueError, "trusted inactive"):
            resolve_inactive_operator_descriptor(CURRENT_RESEARCH_NAMESPACE.operator)

    def test_two_independent_manual_full_records_only_enable_human_review(
        self,
    ) -> None:
        descriptor = RAGGED_PREFILL_DESCRIPTOR
        first = ManualFullEvidence(
            descriptor.profile, UUID_1, SHA_A, "reviewer-a"
        )
        duplicate = replace(first, reviewer="reviewer-b")
        one = descriptor.review_manual_evidence((first, duplicate))
        self.assertEqual(one.activation_state, INACTIVE)
        self.assertEqual(one.accepted_evidence, 1)
        self.assertFalse(one.eligible_for_manual_profile_revision)

        second = ManualFullEvidence(
            descriptor.profile, UUID_2, SHA_B, "reviewer-b"
        )
        ready = descriptor.review_manual_evidence((first, second))
        self.assertEqual(ready.accepted_evidence, 2)
        self.assertTrue(ready.eligible_for_manual_profile_revision)
        self.assertEqual(ready.activation_state, INACTIVE)
        self.assertIn("publish a new profile revision", ready.reasons[-1])
        self.assertFalse(descriptor.promotion_eligible)

        with self.assertRaisesRegex(ValueError, "FULL_PRIMARY"):
            replace(first, stage_id="QUICK")
        with self.assertRaisesRegex(ValueError, "manual"):
            replace(first, execution_mode="AGENT")
        with self.assertRaisesRegex(ValueError, "active state"):
            replace(ready, activation_state="ACTIVE")


class TileLangProbeTests(unittest.TestCase):
    def test_adapter_is_inactive_and_default_probe_is_unavailable(self) -> None:
        adapter = TileLangPythonAdapter()
        self.assertIsInstance(adapter, LanguageAdapter)
        self.assertEqual(adapter.activation_state, INACTIVE)
        with (
            mock.patch(
                "socket.socket",
                side_effect=AssertionError("network must not be used"),
            ),
            mock.patch(
                "subprocess.run",
                side_effect=AssertionError("default must not spawn"),
            ),
        ):
            doctor = adapter.doctor()
            micro = adapter.micro_compile_execute_probe()
        self.assertIs(doctor.status, ProbeStatus.UNAVAILABLE)
        self.assertIs(micro.status, ProbeStatus.UNAVAILABLE)
        self.assertIn("capability", doctor.reason)
        self.assertIsNone(doctor.evidence_digest)

    def test_probe_recipes_have_fixed_argv_and_no_network(self) -> None:
        for recipe in (
            TILELANG_C500_DOCTOR_RECIPE,
            TILELANG_C500_MICRO_RECIPE,
        ):
            with self.subTest(recipe=recipe.recipe_id):
                self.assertEqual(recipe.argv_policy, "FIXED")
                self.assertFalse(recipe.network_access)
                self.assertEqual(recipe.argv[0:2], ("/usr/bin/python3", "-I"))
                self.assertEqual(recipe.device_binding, "metax-c500-exclusive")
                self.assertEqual(
                    recipe.image_binding, "pinned-image-digest-required"
                )
        with self.assertRaisesRegex(ValueError, "network"):
            replace(TILELANG_C500_DOCTOR_RECIPE, network_access=True)
        with self.assertRaisesRegex(ValueError, "dynamic argv"):
            replace(TILELANG_C500_DOCTOR_RECIPE, argv_policy="DYNAMIC")

    def test_injected_runner_receives_only_the_frozen_recipe(self) -> None:
        adapter = TileLangPythonAdapter()
        runner = _CapturingRunner()
        doctor = adapter.doctor(runner)
        micro = adapter.micro_compile_execute_probe(runner)
        self.assertIs(doctor.status, ProbeStatus.PASSED)
        self.assertIs(micro.status, ProbeStatus.PASSED)
        self.assertEqual(
            runner.recipes,
            [TILELANG_C500_DOCTOR_RECIPE, TILELANG_C500_MICRO_RECIPE],
        )
        self.assertEqual(
            runner.capabilities,
            [
                TILELANG_C500_DOCTOR_RECIPE.capability_id,
                TILELANG_C500_MICRO_RECIPE.capability_id,
            ],
        )

        missing = _MissingRunner()
        unavailable = adapter.doctor(missing)
        self.assertIs(unavailable.status, ProbeStatus.UNAVAILABLE)
        self.assertFalse(missing.ran)

    def test_probe_identity_mismatch_and_runner_failure_fail_closed(self) -> None:
        class MismatchRunner(_CapturingRunner):
            def run_fixed(self, recipe: FixedProbeRecipe) -> ProbeObservation:
                return ProbeObservation.unavailable(
                    TILELANG_C500_MICRO_RECIPE,
                    "wrong recipe",
                )

        mismatch = TileLangPythonAdapter().doctor(MismatchRunner())
        self.assertIs(mismatch.status, ProbeStatus.FAILED)
        self.assertIn("mismatched", mismatch.reason)

        class FailingRunner(_CapturingRunner):
            def run_fixed(self, recipe: FixedProbeRecipe) -> ProbeObservation:
                raise RuntimeError("raw vendor output must not escape")

        failed = TileLangPythonAdapter().doctor(FailingRunner())
        self.assertIs(failed.status, ProbeStatus.FAILED)
        self.assertNotIn("raw vendor", failed.reason)

    def test_tilelang_static_validation_is_not_hardware_validation(self) -> None:
        adapter = TileLangPythonAdapter()
        result = adapter.validate_language_candidate(
            "def run_kernel(x):\n    return x\n"
        )
        self.assertEqual(result["status"], "STATIC_ONLY")
        with self.assertRaisesRegex(ValueError, "run_kernel"):
            adapter.validate_language_candidate("def other():\n    pass\n")
        with self.assertRaisesRegex(ValueError, "valid Python"):
            adapter.validate_language_candidate("def run_kernel(:\n")


class MacaCudaContractTests(unittest.TestCase):
    def test_adapter_uses_one_strict_cuda_file_and_fixed_abi(self) -> None:
        adapter = MACACudaAdapter()
        self.assertIsInstance(adapter, LanguageAdapter)
        self.assertEqual(adapter.allowed_paths(), ("kernel.cu",))
        self.assertEqual(adapter.activation_state, INACTIVE)
        bundle = maca_bundle()
        self.assertIs(bundle.validate(adapter.bundle_limits), bundle)
        result = adapter.validate_language_candidate(maca_source())
        self.assertEqual(result["abi_digest"], MACA_CUDA_ABI.digest)

        wrong_signature = maca_source().replace(
            "AutoresearchKernelResultV1* result", "void* result"
        )
        with self.assertRaisesRegex(ValueError, "fixed ABI"):
            adapter.validate_language_candidate(wrong_signature)
        with self.assertRaisesRegex(ValueError, "fixed ABI header"):
            adapter.validate_language_candidate(
                maca_source().replace(
                    '#include "autoresearch_maca_kernel_abi_v1.h"\n', ""
                )
            )

    def test_maca_bundle_policy_rejects_wrong_layout(self) -> None:
        wrong_extension = CandidateBundle(
            SOURCE_BUNDLE_V1,
            "kernel.cpp",
            (CandidateFile("kernel.cpp", "text/x-c++", maca_source()),),
        )
        with self.assertRaisesRegex(ValueError, "entrypoint"):
            wrong_extension.validate(MACA_CUDA_BUNDLE_LIMITS)

        extra_file = CandidateBundle(
            SOURCE_BUNDLE_V1,
            "kernel.cu",
            (
                CandidateFile("kernel.cu", "text/x-cuda", maca_source()),
                CandidateFile("helper.cu", "text/x-cuda", "int helper = 1;\n"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "max_files"):
            extra_file.validate(MACA_CUDA_BUNDLE_LIMITS)

    def test_compile_and_execution_contracts_are_separate_and_bound(self) -> None:
        adapter = MACACudaAdapter()
        bundle = maca_bundle()
        compile_contract = adapter.make_compile_contract(
            bundle,
            toolchain_digest=SHA_A,
            image_digest=SHA_B,
            compile_uid=UUID_1,
        )
        retry = adapter.make_compile_contract(
            bundle,
            toolchain_digest=SHA_A,
            image_digest=SHA_B,
            compile_uid=UUID_2,
        )
        self.assertEqual(
            compile_contract.condition_digest, retry.condition_digest
        )
        self.assertEqual(compile_contract.build_flags, MACA_CUDA_BUILD_FLAGS)
        self.assertEqual(
            compile_contract.environment_id,
            "maca-cuda-compile-sandbox-v1",
        )
        self.assertFalse(compile_contract.network_access)
        self.assertNotIn("content", str(compile_contract.to_dict()))
        with self.assertRaisesRegex(ValueError, "build flags"):
            replace(compile_contract, build_flags=("-O0",))
        with self.assertRaisesRegex(ValueError, "network"):
            replace(compile_contract, network_access=True)

        binary = CompiledMacaBinaryRef.from_compile_contract(
            compile_contract,
            binary_digest=SHA_C,
        )
        execution = adapter.make_execution_contract(
            binary,
            input_digest=SHA_A,
            image_digest=SHA_B,
            case_id="ragged-prefill-smoke-1",
            execution_uid=UUID_1,
        )
        self.assertEqual(
            execution.compiled_binary.compile_condition_digest,
            compile_contract.condition_digest,
        )
        self.assertNotEqual(
            execution.condition_digest, compile_contract.condition_digest
        )
        self.assertEqual(execution.to_dict()["execution_uid"], UUID_1)
        self.assertEqual(
            execution.environment_id,
            "maca-cuda-execute-sandbox-v1",
        )
        self.assertEqual(execution.device_binding, "metax-c500-exclusive")
        self.assertFalse(execution.network_access)
        with self.assertRaisesRegex(ValueError, "compiled binary"):
            adapter.make_execution_contract(  # type: ignore[arg-type]
                bundle,
                input_digest=SHA_A,
                image_digest=SHA_B,
                case_id="ragged-prefill-smoke-1",
            )
        with self.assertRaisesRegex(ValueError, "network"):
            replace(execution, network_access=True)

    def test_adapter_identities_cannot_be_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, "identity"):
            replace(TileLangPythonAdapter(), activation_state="ACTIVE")
        with self.assertRaisesRegex(ValueError, "identity"):
            replace(MACACudaAdapter(), entrypoint="other.cu")
        with self.assertRaisesRegex(ValueError, "identity"):
            replace(RaggedPrefillPack(), revision="v1")


class RaggedPrefillTests(unittest.TestCase):
    def test_pack_is_inactive_and_owns_semantics(self) -> None:
        pack = RaggedPrefillPack()
        self.assertIsInstance(pack, OperatorPack)
        self.assertEqual(pack.activation_state, INACTIVE)
        self.assertEqual(pack.contract_summary()["abi"], "ragged-prefill-v1")
        self.assertIn("causal prefix", pack.prompt_context())
        self.assertGreaterEqual(len(pack.invariants()), 5)
        self.assertEqual(
            pack.validate_operator_candidate("def run_kernel(): pass")["status"],
            "CONTRACT_ONLY",
        )

    def test_small_causal_oracle_matches_hand_computation(self) -> None:
        import numpy as np

        pack = RaggedPrefillPack()
        q = np.array([[[1.0]], [[1.0]]])
        k = np.array([[[0.0]], [[1.0]]])
        v = np.array([[[2.0]], [[4.0]]])
        output = pack.oracle(q, k, v, [0, 2], [0, 2], scale=1.0)
        expected_second = (2.0 + 4.0 * math_e()) / (1.0 + math_e())
        np.testing.assert_allclose(
            output[:, 0, 0],
            np.array([2.0, expected_second]),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(output.dtype, np.float64)

    def test_oracle_supports_prefix_and_grouped_query_heads(self) -> None:
        import numpy as np

        q = np.ones((1, 2, 1), dtype=np.float32)
        k = np.zeros((2, 1, 1), dtype=np.float32)
        v = np.array([[[2.0]], [[4.0]]], dtype=np.float32)
        output = RaggedPrefillPack().oracle(
            q,
            k,
            v,
            [0, 1],
            [0, 2],
            causal=True,
            scale=np.float32(1.0),
        )
        np.testing.assert_allclose(output[0, :, 0], [3.0, 3.0])

    def test_oracle_rejects_invalid_ragged_boundaries_and_heads(self) -> None:
        import numpy as np

        pack = RaggedPrefillPack()
        q = np.ones((2, 2, 1))
        k = np.ones((2, 1, 1))
        v = np.ones((2, 1, 1))
        with self.assertRaisesRegex(ValueError, "monotonic"):
            pack.oracle(
                np.ones((3, 2, 1)),
                np.ones((3, 1, 1)),
                np.ones((3, 1, 1)),
                [0, 2, 1, 3],
                [0, 1, 2, 3],
            )
        with self.assertRaisesRegex(ValueError, "kv_length"):
            pack.oracle(q, k[:1], v[:1], [0, 2], [0, 1])
        with self.assertRaisesRegex(ValueError, "divisible"):
            pack.oracle(
                np.ones((1, 3, 1)),
                np.ones((1, 2, 1)),
                np.ones((1, 2, 1)),
                [0, 1],
                [0, 1],
            )
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            pack.oracle(
                np.ones((1, 1, 1), dtype=np.complex64),
                np.ones((1, 1, 1)),
                np.ones((1, 1, 1)),
                [0, 1],
                [0, 1],
            )
        with self.assertRaisesRegex(ValueError, "small-scale"):
            pack.oracle(
                np.ones((513, 1, 1)),
                np.ones((513, 1, 1)),
                np.ones((513, 1, 1)),
                [0, 513],
                [0, 513],
            )


def math_e() -> float:
    # Keeps the hand-computed expected value independent from NumPy.
    import math

    return math.e


class DependencyBoundaryTests(unittest.TestCase):
    def test_import_is_light_and_does_not_load_numpy_or_gpu_libraries(self) -> None:
        script = """
import builtins
import sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'triton', 'tilelang'}:
        raise AssertionError('forbidden import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import kernel_research.platform.next_targets
assert 'numpy' not in sys.modules
print('LIGHT_NEXT_TARGETS_IMPORT_OK')
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
            completed.stdout.strip(), "LIGHT_NEXT_TARGETS_IMPORT_OK"
        )


if __name__ == "__main__":
    unittest.main()
