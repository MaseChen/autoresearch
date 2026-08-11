from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys
import unittest
from unittest import mock

from kernel_research.autorun.model_catalog import OPENCODE_MODEL_SPECS
from kernel_research.platform.artifacts import ArtifactId
from kernel_research.platform.profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    ProfileDefinition,
    ProfileRegistry,
)
from kernel_research.platform.proposers import (
    BUILTIN_PROPOSER_REGISTRY,
    MAX_PROPOSER_PROMPT_BYTES,
    CredentialRef,
    HarnessInvocation,
    ModelProfile,
    PromptProtocol,
    TrustedProposerRegistry,
)


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_ID = "sha256:" + "a" * 64
PARENT_ID = ArtifactId.source_sha256("b" * 64)
REQUEST_ID = "00000000-0000-4000-8000-000000000001"


def proposer_ref(profile_id: str):
    return BUILTIN_PROFILE_REGISTRY.get(
        kind="proposer", profile_id=profile_id, revision="v1"
    ).ref


class ProposerRegistryTests(unittest.TestCase):
    def test_all_harness_model_profiles_resolve_without_aliases(self) -> None:
        expected_transports = {
            "direct-api": "direct-api",
            "opencode": "opencode",
            "pi": "pi",
        }
        for harness_id, transport in expected_transports.items():
            for model_suffix in ("pro", "flash"):
                profile_id = f"{harness_id}-deepseek-v4-{model_suffix}"
                with self.subTest(profile_id=profile_id):
                    resolved = BUILTIN_PROPOSER_REGISTRY.resolve_profile(
                        proposer_ref(profile_id)
                    )
                    self.assertEqual(resolved.harness.harness_id, harness_id)
                    self.assertEqual(resolved.harness.revision, "v1")
                    self.assertEqual(
                        resolved.model.model_id,
                        f"deepseek/deepseek-v4-{model_suffix}",
                    )
                    invocation = resolved.harness.build_invocation(
                        model=resolved.model,
                        prompt_protocol=resolved.prompt_protocol,
                        prompt="Return one proposal.",
                        max_output_tokens=100,
                    )
                    self.assertEqual(invocation.transport, transport)

        self.assertNotIn(
            "proposer", CURRENT_RESEARCH_NAMESPACE.to_dict()
        )

        with self.assertRaisesRegex(ValueError, "unknown trusted harness"):
            BUILTIN_PROPOSER_REGISTRY.resolve_harness("opencode", "latest")
        with self.assertRaisesRegex(ValueError, "unknown trusted model"):
            BUILTIN_PROPOSER_REGISTRY.resolve_model(
                "deepseek/deepseek-v4-unknown", "v1"
            )
        with self.assertRaisesRegex(ValueError, "unknown trusted prompt"):
            BUILTIN_PROPOSER_REGISTRY.resolve_prompt("proposal-v1", "latest")

    def test_model_profiles_match_the_existing_allowlist(self) -> None:
        for qualified_id, catalog in OPENCODE_MODEL_SPECS.items():
            with self.subTest(qualified_id=qualified_id):
                model = BUILTIN_PROPOSER_REGISTRY.resolve_model(
                    qualified_id, "v1"
                )
                self.assertEqual(model.provider_model_id, catalog.provider_id)
                self.assertEqual(model.context_tokens, catalog.context_tokens)
                self.assertEqual(model.output_tokens, catalog.output_tokens)
                self.assertEqual(
                    model.request_output_token_cap,
                    catalog.request_output_token_cap,
                )
                self.assertEqual(
                    model.reasoning_effort, catalog.reasoning_effort
                )

    def test_registry_is_sealed_and_rejects_tampered_profile_digest(self) -> None:
        self.assertTrue(BUILTIN_PROPOSER_REGISTRY.sealed)
        with self.assertRaises(RuntimeError):
            BUILTIN_PROPOSER_REGISTRY.register_prompt(
                PromptProtocol("proposal-v1", "v2", 1)
            )
        ref = proposer_ref("opencode-deepseek-v4-pro")
        with self.assertRaisesRegex(ValueError, "digest"):
            BUILTIN_PROPOSER_REGISTRY.resolve_profile(
                replace(ref, digest="sha256:" + "0" * 64)
            )
        with self.assertRaisesRegex(ValueError, "proposer kind"):
            BUILTIN_PROPOSER_REGISTRY.resolve_profile(
                CURRENT_RESEARCH_NAMESPACE.language
            )

        unbound = ProfileDefinition.create(
            kind="proposer",
            profile_id="unbound",
            revision="v1",
            implementation_id="unbound",
            config={
                "harness": "opencode",
                "harness_revision": "v1",
                "model": "deepseek/deepseek-v4-pro",
                "model_revision": "v1",
                "prompt_protocol": "proposal-v1",
                "prompt_protocol_revision": "v1",
            },
        )
        unbound_registry = TrustedProposerRegistry(
            profile_registry=ProfileRegistry((unbound,))
        )
        with self.assertRaisesRegex(ValueError, "no trusted binding"):
            unbound_registry.resolve_profile(unbound.ref)


class ProposerRequestTests(unittest.TestCase):
    def request(
        self,
        profile_id: str = "opencode-deepseek-v4-pro",
        *,
        credential_ref: str = "credential-ref:v1:deepseek-primary",
        request_id: str = REQUEST_ID,
    ):
        return BUILTIN_PROPOSER_REGISTRY.create_request(
            proposer_ref(profile_id),
            proposal_context_id=CONTEXT_ID,
            parent_artifact_id=PARENT_ID,
            prompt="Return one strict ProposalV1 object.",
            credential_ref=credential_ref,
            request_id=request_id,
            timeout_sec=120,
            max_output_tokens=2048,
        )

    def test_request_generation_is_inert_and_harness_specific(self) -> None:
        with mock.patch(
            "socket.socket",
            side_effect=AssertionError("network access is forbidden"),
        ):
            direct = self.request("direct-api-deepseek-v4-pro")
            opencode = self.request("opencode-deepseek-v4-pro")
            pi = self.request("pi-deepseek-v4-pro")

        self.assertEqual(direct.invocation.transport, "direct-api")
        self.assertEqual(opencode.invocation.transport, "opencode")
        self.assertEqual(pi.invocation.transport, "pi")
        self.assertEqual(
            direct.invocation.payload["provider_model_id"],
            "deepseek-v4-pro",
        )
        self.assertEqual(
            opencode.invocation.payload["format"], "json-events"
        )
        self.assertEqual(pi.invocation.payload["session_mode"], "non-interactive")
        self.assertNotEqual(direct.condition_digest, opencode.condition_digest)
        self.assertNotEqual(opencode.condition_digest, pi.condition_digest)

    def test_provenance_is_bounded_and_credential_rotation_is_not_identity(self) -> None:
        first = self.request()
        rotated = self.request(
            credential_ref="credential-ref:v1:deepseek-rotated",
            request_id="00000000-0000-4000-8000-000000000002",
        )
        self.assertEqual(first.condition_digest, rotated.condition_digest)
        self.assertEqual(first.prompt_digest, rotated.prompt_digest)

        provenance = first.provenance_dict()
        serialized = json.dumps(provenance, sort_keys=True)
        self.assertNotIn("Return one strict", serialized)
        self.assertNotIn("messages", serialized)
        self.assertNotIn("input", serialized)
        self.assertEqual(
            provenance["credential_ref"],
            "credential-ref:v1:deepseek-primary",
        )
        self.assertEqual(
            first.condition_digest,
            "sha256:b6212abcca298745f599b103bfa45ccc86"
            "71b600c6ef7878da05efa6c1f3218c",
        )

    def test_secret_values_and_untrusted_limits_fail_closed(self) -> None:
        for value in (
            "sk-plaintext-secret",
            "deepseek-key-file",
            "credential-ref:v1:../key",
            "credential-ref:v1:key//child",
            "credential-ref:v1:key/./child",
            "credential-ref:v1:key/",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CredentialRef.parse(value)

        with self.assertRaisesRegex(ValueError, "model profile cap"):
            BUILTIN_PROPOSER_REGISTRY.create_request(
                proposer_ref("opencode-deepseek-v4-pro"),
                proposal_context_id=CONTEXT_ID,
                parent_artifact_id=PARENT_ID,
                prompt="proposal",
                credential_ref="credential-ref:v1:key",
                max_output_tokens=384_001,
            )
        with self.assertRaisesRegex(ValueError, "host limit"):
            BUILTIN_PROPOSER_REGISTRY.create_request(
                proposer_ref("opencode-deepseek-v4-pro"),
                proposal_context_id=CONTEXT_ID,
                parent_artifact_id=PARENT_ID,
                prompt="x" * (MAX_PROPOSER_PROMPT_BYTES + 1),
                credential_ref="credential-ref:v1:key",
            )
        with self.assertRaisesRegex(ValueError, "secret content"):
            HarnessInvocation(
                "opencode",
                "v1",
                "opencode",
                {"api_key": "plaintext"},
            )
        with self.assertRaisesRegex(ValueError, "secret content"):
            HarnessInvocation(
                "direct-api",
                "v1",
                "direct-api",
                {"nested": {"client_secret": "plaintext"}},
            )

    def test_request_contract_rejects_invocation_tampering(self) -> None:
        request = self.request()
        tampered = HarnessInvocation(
            "opencode",
            "v1",
            "opencode",
            {
                **request.invocation.to_dict()["payload"],
                "max_output_tokens": 1,
            },
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            replace(request, invocation=tampered)


class ProposerImportTests(unittest.TestCase):
    def test_proposer_contract_import_is_dependency_light(self) -> None:
        script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'triton'}:
        raise AssertionError('forbidden import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import kernel_research.platform.proposers
print('LIGHT_PROPOSER_IMPORT_OK')
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
            completed.stdout.strip(), "LIGHT_PROPOSER_IMPORT_OK"
        )


if __name__ == "__main__":
    unittest.main()
