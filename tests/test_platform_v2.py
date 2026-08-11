from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

from kernel_research.autorun.models import (
    ProposalV1,
    normalize_proposal_to_v2,
)
from kernel_research.autorun.proposal import (
    ProposalFormatError,
    parse_proposal_v2_text,
)
from kernel_research.platform import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    ArtifactId,
    BaselineRef,
    BundleLimits,
    CandidateBundle,
    ExperimentIdentity,
    ProfileDefinition,
    ProfileRef,
    ProposalV2,
    ResearchNamespace,
    TRITON_PYTHON_BUNDLE_LIMITS,
    canonical_json_text,
    canonical_sha256,
)


CONTEXT_ID = "sha256:" + "a" * 64
PARENT_HASH = "b" * 64
PARENT_ID = ArtifactId.source_sha256(PARENT_HASH)


def _candidate_value(content: str = "def run_kernel():\n    return None\n") -> dict:
    return {
        "format": "source_bundle_v1",
        "entrypoint": "kernel.py",
        "files": [
            {
                "path": "kernel.py",
                "media_type": "text/x-python",
                "content": content,
            }
        ],
    }


def _proposal_v2_value() -> dict:
    return {
        "schema_version": 2,
        "proposal_context_id": CONTEXT_ID,
        "parent_artifact_id": str(PARENT_ID),
        "hypothesis": "Change one tile and measure its effect.",
        "rationale": "The change is isolated and preserves the ABI.",
        "candidate": _candidate_value(),
    }


def _other_namespace() -> ResearchNamespace:
    language = ProfileDefinition.create(
        kind="language",
        profile_id="tilelang-python",
        revision="v1",
        implementation_id="tilelang-python",
        config={
            "entrypoint": "kernel.py",
            "allowed_extensions": [".py"],
            "toolchain_fingerprint": "fixture",
        },
    ).ref
    return replace(LEGACY_RESEARCH_NAMESPACE, language=language)


class CanonicalAndProfileTests(unittest.TestCase):
    def test_canonical_json_and_digest_golden(self) -> None:
        value = {
            "z": [3, -0.0],
            "a": "界",
            "b": {"y": True, "x": None},
        }
        self.assertEqual(
            canonical_json_text(value),
            '{"a":"界","b":{"x":null,"y":true},"z":[3,0]}',
        )
        self.assertEqual(
            canonical_sha256(value),
            "sha256:cf234f28e6ca07b67dba71967f05ce7f"
            "a19de02577e1665d1834f6b72b177d57",
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            canonical_json_text({"bad": float("nan")})
        with self.assertRaisesRegex(TypeError, "non-string"):
            canonical_json_text({1: "coercion is forbidden"})

    def test_protocol_namespaces_have_distinct_golden_digests(self) -> None:
        self.assertEqual(
            LEGACY_RESEARCH_NAMESPACE.namespace_id,
            "sha256:ec67b4d65a8b734e336d78c4459a1f15"
            "91ba5635e50317c9be32762865e0a6b4",
        )
        self.assertEqual(
            CURRENT_RESEARCH_NAMESPACE.namespace_id,
            "sha256:ed7f99e4d7108646ff0477b118f301ed"
            "1eba4637e30105dd11c7918318dcf096",
        )
        self.assertNotIn("proposer", LEGACY_RESEARCH_NAMESPACE.to_dict())
        self.assertNotIn("proposer", CURRENT_RESEARCH_NAMESPACE.to_dict())
        self.assertNotEqual(
            LEGACY_RESEARCH_NAMESPACE.evaluation_protocol,
            CURRENT_RESEARCH_NAMESPACE.evaluation_protocol,
        )
        self.assertNotEqual(
            LEGACY_RESEARCH_NAMESPACE.namespace_id,
            _other_namespace().namespace_id,
        )

    def test_registry_requires_exact_builtin_revision_and_digest(self) -> None:
        ref = LEGACY_RESEARCH_NAMESPACE.language
        self.assertEqual(BUILTIN_PROFILE_REGISTRY.resolve(ref).ref, ref)
        tampered = replace(ref, digest="sha256:" + "0" * 64)
        with self.assertRaisesRegex(ValueError, "digest"):
            BUILTIN_PROFILE_REGISTRY.resolve(tampered)
        unknown = ProfileRef.create(
            kind="language",
            profile_id="unknown-language",
            revision="v1",
            profile={"implementation_id": "none"},
        )
        with self.assertRaisesRegex(KeyError, "unknown built-in"):
            BUILTIN_PROFILE_REGISTRY.resolve(unknown)

        proposer_models = {
            "opencode-deepseek-v4-pro": "deepseek/deepseek-v4-pro",
            "opencode-deepseek-v4-flash": "deepseek/deepseek-v4-flash",
        }
        for profile_id, model in proposer_models.items():
            with self.subTest(profile_id=profile_id):
                definition = BUILTIN_PROFILE_REGISTRY.get(
                    kind="proposer",
                    profile_id=profile_id,
                    revision="v1",
                )
                self.assertEqual(definition.config["model"], model)
                self.assertEqual(definition.config["harness"], "opencode")
                self.assertEqual(
                    definition.config["prompt_protocol"], "proposal-v1"
                )

    def test_namespace_round_trip_rejects_tampered_digest(self) -> None:
        value = LEGACY_RESEARCH_NAMESPACE.to_dict()
        self.assertEqual(
            ResearchNamespace.from_value(value), LEGACY_RESEARCH_NAMESPACE
        )
        value["namespace_id"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            ResearchNamespace.from_value(value)


class CandidateBundleTests(unittest.TestCase):
    def test_bundle_manifest_and_artifact_digest_golden(self) -> None:
        bundle = CandidateBundle.single_file(content='print("界")\n')
        self.assertEqual(
            str(bundle.artifact_id),
            "bundle-sha256-v1:f1c9f0f4c3d93946157a4d58a3966ad"
            "93e1bec33339b0e2a98ac3c89989b18fa",
        )
        self.assertEqual(bundle.files[0].content, 'print("界")\n')
        changed = CandidateBundle.single_file(content='print("界")\r\n')
        self.assertNotEqual(changed.artifact_id, bundle.artifact_id)

    def test_file_order_is_normalized_but_duplicate_paths_fail(self) -> None:
        limits = BundleLimits(
            max_files=2,
            max_file_bytes=128,
            max_total_bytes=256,
            allowed_extensions=frozenset({".py"}),
            allowed_media_types=frozenset({"text/x-python"}),
            required_entrypoint="kernel.py",
        )
        value = _candidate_value()
        value["files"].append(
            {"path": "helpers.py", "media_type": "text/x-python", "content": "X=1\n"}
        )
        reversed_value = {**value, "files": list(reversed(value["files"]))}
        first = CandidateBundle.from_value(value, limits=limits)
        second = CandidateBundle.from_value(reversed_value, limits=limits)
        self.assertEqual(first.artifact_id, second.artifact_id)
        duplicate = _candidate_value()
        duplicate["files"].append(dict(duplicate["files"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            CandidateBundle.from_value(duplicate, limits=limits)

    def test_paths_fail_closed_before_language_execution(self) -> None:
        invalid_paths = (
            "../kernel.py",
            "dir/../kernel.py",
            "/kernel.py",
            "dir//kernel.py",
            "./kernel.py",
            "C:/kernel.py",
            "dir\\kernel.py",
        )
        for path in invalid_paths:
            with self.subTest(path=path):
                value = _candidate_value()
                value["entrypoint"] = path
                value["files"][0]["path"] = path
                with self.assertRaises(ValueError):
                    CandidateBundle.from_value(value)

    def test_language_and_system_size_limits_are_both_enforced(self) -> None:
        limits = BundleLimits(
            max_files=2,
            max_file_bytes=4,
            max_total_bytes=7,
            allowed_extensions=frozenset({".py"}),
            allowed_media_types=frozenset({"text/x-python"}),
            required_entrypoint="kernel.py",
        )
        too_large = _candidate_value("12345")
        with self.assertRaisesRegex(ValueError, "max_file_bytes"):
            CandidateBundle.from_value(too_large, limits=limits)

        too_large = _candidate_value("1234")
        too_large["files"].append(
            {"path": "helper.py", "media_type": "text/x-python", "content": "1234"}
        )
        with self.assertRaisesRegex(ValueError, "max_total_bytes"):
            CandidateBundle.from_value(too_large, limits=limits)

        wrong_extension = _candidate_value("1234")
        wrong_extension["entrypoint"] = "kernel.cu"
        wrong_extension["files"][0]["path"] = "kernel.cu"
        permissive_entrypoint = replace(limits, required_entrypoint=None)
        with self.assertRaisesRegex(ValueError, "extension"):
            CandidateBundle.from_value(
                wrong_extension, limits=permissive_entrypoint
            )

        with self.assertRaisesRegex(ValueError, "hard limit"):
            BundleLimits(max_file_bytes=1024 * 1024 + 1)


class ProposalV2Tests(unittest.TestCase):
    def test_strict_context_parent_and_unknown_fields(self) -> None:
        value = _proposal_v2_value()
        proposal = ProposalV2.from_value(
            value,
            expected_proposal_context_id=CONTEXT_ID,
            expected_parent_artifact_id=PARENT_ID,
            limits=TRITON_PYTHON_BUNDLE_LIMITS,
        )
        self.assertEqual(proposal.parent_artifact_id, PARENT_ID)
        self.assertEqual(proposal.candidate_artifact_id.tag, "bundle-sha256-v1")
        self.assertNotIn("candidate_artifact_id", proposal.to_dict())

        with self.assertRaisesRegex(ValueError, "proposal_context_id"):
            ProposalV2.from_value(
                value,
                expected_proposal_context_id="sha256:" + "0" * 64,
                expected_parent_artifact_id=PARENT_ID,
            )
        with self.assertRaisesRegex(ValueError, "parent_artifact_id"):
            ProposalV2.from_value(
                value,
                expected_proposal_context_id=CONTEXT_ID,
                expected_parent_artifact_id=ArtifactId.source_sha256("0" * 64),
            )
        unknown = dict(value, candidate_hash="0" * 64)
        with self.assertRaisesRegex(ValueError, "unknown"):
            ProposalV2.from_value(
                unknown,
                expected_proposal_context_id=CONTEXT_ID,
                expected_parent_artifact_id=PARENT_ID,
            )

    def test_text_parser_is_strict_and_supports_one_exact_fence(self) -> None:
        raw = json.dumps(_proposal_v2_value())
        proposal = parse_proposal_v2_text(
            raw,
            expected_proposal_context_id=CONTEXT_ID,
            expected_parent_artifact_id=PARENT_ID,
            limits=TRITON_PYTHON_BUNDLE_LIMITS,
        )
        fenced = parse_proposal_v2_text(
            f"```json\n{raw}\n```",
            expected_proposal_context_id=CONTEXT_ID,
            expected_parent_artifact_id=PARENT_ID,
            limits=TRITON_PYTHON_BUNDLE_LIMITS,
        )
        self.assertEqual(proposal, fenced)
        with self.assertRaises(ProposalFormatError):
            parse_proposal_v2_text(
                f"```json\n{raw}\n```\nextra",
                expected_proposal_context_id=CONTEXT_ID,
                expected_parent_artifact_id=PARENT_ID,
            )
        duplicate_key = raw.replace(
            '"schema_version": 2,',
            '"schema_version": 2, "schema_version": 2,',
            1,
        )
        with self.assertRaisesRegex(ProposalFormatError, "duplicate"):
            parse_proposal_v2_text(
                duplicate_key,
                expected_proposal_context_id=CONTEXT_ID,
                expected_parent_artifact_id=PARENT_ID,
            )

    def test_v1_normalization_is_legacy_only_and_preserves_v1_behavior(self) -> None:
        source = "def run_kernel():\n    return None\n"
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        value = {
            "schema_version": 1,
            "parent_candidate_hash": PARENT_HASH,
            "hypothesis": "Change one tile.",
            "rationale": "The change is measurable.",
            "kernel_source": source,
        }
        proposal_v1 = ProposalV1.from_value(
            value, expected_parent_hash=PARENT_HASH
        )
        self.assertEqual(proposal_v1.candidate_hash, source_hash)
        self.assertEqual(proposal_v1.to_dict()["kernel_source"], source)

        normalized = normalize_proposal_to_v2(
            proposal_v1,
            expected_proposal_context_id=CONTEXT_ID,
            expected_parent_artifact_id=PARENT_ID,
            namespace=LEGACY_RESEARCH_NAMESPACE,
        )
        self.assertEqual(normalized.proposal_context_id, CONTEXT_ID)
        self.assertEqual(normalized.parent_artifact_id, PARENT_ID)
        self.assertEqual(normalized.candidate.files[0].content, source)
        self.assertEqual(normalized.candidate_artifact_id.tag, "bundle-sha256-v1")

        with self.assertRaisesRegex(ValueError, "legacy"):
            normalize_proposal_to_v2(
                proposal_v1,
                expected_proposal_context_id=CONTEXT_ID,
                expected_parent_artifact_id=PARENT_ID,
                namespace=_other_namespace(),
            )


class ExperimentIdentityTests(unittest.TestCase):
    def _identity(
        self,
        *,
        namespace: ResearchNamespace = LEGACY_RESEARCH_NAMESPACE,
        uid: str,
        run_id: str = "run-1",
    ) -> ExperimentIdentity:
        baseline = BaselineRef.create(
            namespace=namespace,
            artifact_id=PARENT_ID,
            source="deployment",
            revision="pin-1",
        )
        proposer = BUILTIN_PROFILE_REGISTRY.get(
            kind="proposer",
            profile_id="opencode-deepseek-v4-pro",
            revision="v1",
        ).ref
        return ExperimentIdentity.create(
            experiment_uid=uid,
            namespace=namespace,
            mode="BENCHMARK",
            cohort_id="cohort-1",
            history_cutoff=42,
            candidate_artifact_id=CandidateBundle.single_file(
                content="def run_kernel(): pass\n"
            ).artifact_id,
            parent_artifact_id=PARENT_ID,
            baseline=baseline,
            stage="QUICK",
            suite="quick",
            replicate_kind="noise",
            replicate_index=3,
            proposer_profile=proposer,
            prompt_digest="sha256:" + "c" * 64,
            feedback_digest="sha256:" + "d" * 64,
            campaign_id=None,
            run_id=run_id,
            iteration=1,
        )

    def test_uid_condition_and_namespace_are_distinct_identities(self) -> None:
        first = self._identity(uid="00000000-0000-4000-8000-000000000001")
        second = self._identity(
            uid="00000000-0000-4000-8000-000000000002",
            run_id="run-2",
        )
        self.assertNotEqual(first.experiment_uid, second.experiment_uid)
        self.assertEqual(first.condition_digest, second.condition_digest)

        cross_namespace = self._identity(
            namespace=_other_namespace(),
            uid="00000000-0000-4000-8000-000000000003",
        )
        self.assertNotEqual(first.namespace_id, cross_namespace.namespace_id)
        self.assertNotEqual(
            first.condition_digest, cross_namespace.condition_digest
        )

    def test_identity_round_trip_and_tamper_detection(self) -> None:
        identity = self._identity(
            uid="00000000-0000-4000-8000-000000000004"
        )
        value = identity.to_dict()
        self.assertEqual(ExperimentIdentity.from_value(value), identity)
        value["condition_digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            ExperimentIdentity.from_value(value)

    def test_cross_namespace_baseline_is_rejected(self) -> None:
        baseline = BaselineRef.create(
            namespace=_other_namespace(),
            artifact_id=PARENT_ID,
            source="campaign",
            revision="revision-1",
        )
        with self.assertRaisesRegex(ValueError, "different"):
            ExperimentIdentity.create(
                experiment_uid="00000000-0000-4000-8000-000000000005",
                namespace=LEGACY_RESEARCH_NAMESPACE,
                mode="DISCOVERY",
                candidate_artifact_id=PARENT_ID,
                parent_artifact_id=PARENT_ID,
                baseline=baseline,
                stage="SMOKE",
                suite="smoke",
                run_id="run-1",
                iteration=1,
            )


if __name__ == "__main__":
    unittest.main()
