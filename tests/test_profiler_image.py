from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from kernel_research.platform.canonical import canonical_sha256
from kernel_research.platform.profiles import CURRENT_RESEARCH_NAMESPACE
from kernel_research.platform.proposal import CandidateBundle
from kernel_research.profiler_contract import (
    MCTRACER_PATH,
    MCTRACER_SHA256,
    MCTRACER_VERSION,
    MCTOOLS_EXT_LITE_SHA256,
    MCTOOLS_EXT_SHA256,
    PROFILE_COLLECTION_SCHEMA_VERSION,
    PROFILER_ACTIVE,
    PROFILER_ACTIVATION_PROFILE_DIGEST,
    PROFILER_BASE_IMAGE,
    PROFILER_BUILD_PROFILE_DIGEST,
    PROFILER_IMAGE,
    PROFILER_IMAGE_REPOSITORY,
    PROFILER_WORKER_REVISION,
    profiler_activation_profile_snapshot,
    profiler_build_profile_snapshot,
    require_active_profiler,
)
import kernel_research.profiler_contract as contract
import kernel_research.profiler_worker as worker
import kernel_research.profiling as host


class ProfilerContractTests(unittest.TestCase):
    def test_submit_a_profile_is_exact_and_inactive(self) -> None:
        build = profiler_build_profile_snapshot()
        activation = profiler_activation_profile_snapshot()
        self.assertFalse(PROFILER_ACTIVE)
        self.assertEqual(build["base_image"], PROFILER_BASE_IMAGE)
        self.assertNotIn("active", build)
        self.assertNotIn("profiler_image", build)
        self.assertEqual(activation["profiler_image"], PROFILER_IMAGE)
        self.assertFalse(activation["active"])
        self.assertTrue(PROFILER_IMAGE.startswith(PROFILER_IMAGE_REPOSITORY))
        self.assertEqual(build["worker_revision"], PROFILER_WORKER_REVISION)
        self.assertEqual(
            build["collection_schema_version"],
            PROFILE_COLLECTION_SCHEMA_VERSION,
        )
        self.assertEqual(PROFILER_BUILD_PROFILE_DIGEST, canonical_sha256(build))
        self.assertEqual(
            PROFILER_ACTIVATION_PROFILE_DIGEST,
            canonical_sha256(activation),
        )
        with self.assertRaisesRegex(ValueError, "inactive"):
            require_active_profiler()

    def test_commit_a_worker_echo_survives_commit_b_activation_only_change(self) -> None:
        request = ProfilerWorkerTests._request()
        commit_a_echo = request.echo()
        commit_a_build = profiler_build_profile_snapshot()
        final_image = PROFILER_IMAGE_REPOSITORY + "@sha256:" + "a" * 64
        with (
            mock.patch.object(contract, "PROFILER_ACTIVE", True),
            mock.patch.object(contract, "PROFILER_IMAGE", final_image),
        ):
            commit_b_build = profiler_build_profile_snapshot()
            commit_b_activation = profiler_activation_profile_snapshot()
        self.assertEqual(commit_b_build, commit_a_build)
        self.assertEqual(
            canonical_sha256(commit_b_build),
            PROFILER_BUILD_PROFILE_DIGEST,
        )
        self.assertEqual(
            commit_a_echo["profiler_build_profile_digest"],
            PROFILER_BUILD_PROFILE_DIGEST,
        )
        self.assertEqual(
            commit_b_activation["build_profile_digest"],
            commit_a_echo["profiler_build_profile_digest"],
        )
        self.assertNotIn("profiler_image", commit_a_echo)
        self.assertNotIn("active", commit_a_echo)
        self.assertNotIn("profiler_activation_profile_digest", commit_a_echo)
        self.assertNotEqual(
            canonical_sha256(profiler_activation_profile_snapshot()),
            canonical_sha256(commit_b_activation),
        )
        subject = host._ProfileSubject(
            experiment_uid=request.experiment_uid,
            namespace_id=request.namespace_id,
            condition_digest=request.condition_digest,
            artifact_id=request.artifact_id,
            execution_environment_digest=request.environment_digest,
            campaign_id=request.campaign_id,
            run_id=request.run_id,
            stage="smoke",
            suite="smoke",
            replicate_kind="validation",
            candidate_object_path=Path("/state/object"),
            candidate_content_sha256=request.candidate_sha256,
        )
        with (
            mock.patch.object(host, "PROFILER_IMAGE", final_image),
            mock.patch.object(
                host,
                "PROFILER_ACTIVATION_PROFILE_DIGEST",
                canonical_sha256(commit_b_activation),
            ),
        ):
            commit_b_expected = host._expected_worker_echo(
                recipe=host.BUILTIN_PROFILE_RECIPES[request.recipe_id],
                subject=subject,
                invariant_digest=request.soak_invariant_digest,
                budget_action_key=request.budget_action_key,
                lease=None,
            )
        self.assertEqual(commit_b_expected, commit_a_echo)

    def test_build_profile_freezes_every_worker_and_runtime_axis(self) -> None:
        build = profiler_build_profile_snapshot()
        self.assertEqual(
            set(build),
            {
                "schema_version",
                "platform",
                "base_image",
                "worker_revision",
                "entrypoint",
                "image_user",
                "toolchain",
                "recipes",
                "paths",
                "limits",
                "isolation",
                "resource",
                "collection_schema_version",
                "worker_output_schema_version",
            },
        )
        self.assertEqual(build["image_user"]["uid"], 1000)
        self.assertEqual(build["image_user"]["gid"], 1000)
        self.assertEqual(build["platform"], "linux/amd64")
        self.assertEqual(build["limits"]["action_timeout_seconds"], 900.0)
        self.assertEqual(build["limits"]["toolchain_help_timeout_seconds"], 10.0)
        self.assertEqual(build["limits"]["raw_trace_bytes"], 64 * 1024 * 1024)
        self.assertEqual(
            build["limits"]["canary_raw_trace_total_bytes"],
            64 * 1024 * 1024,
        )
        self.assertEqual(build["limits"]["memory"], "4g")
        self.assertEqual(build["limits"]["cpus"], 4.0)
        self.assertEqual(build["limits"]["pids"], 128)
        self.assertTrue(build["isolation"]["read_only_root"])
        self.assertEqual(build["isolation"]["network"], "none")
        self.assertEqual(build["resource"]["resource_id"], "gpu1")
        self.assertEqual(build["resource"]["lease_ttl_seconds"], 930.0)
        self.assertEqual(build["resource"]["canary_lease_ttl_seconds"], 1830.0)
        self.assertEqual(len(build["resource"]["gpu_device_paths"]), 3)
        self.assertEqual(
            set(profiler_activation_profile_snapshot()),
            {"schema_version", "build_profile_digest", "active", "profiler_image"},
        )

    def test_activation_rejects_sentinel_and_accepts_exact_ghcr_digest(self) -> None:
        with (
            mock.patch.object(contract, "PROFILER_ACTIVE", True),
            mock.patch.object(
                contract,
                "PROFILER_IMAGE",
                PROFILER_IMAGE_REPOSITORY + "@sha256:" + "a" * 64,
            ),
        ):
            require_active_profiler()
        with (
            mock.patch.object(contract, "PROFILER_ACTIVE", True),
            mock.patch.object(
                contract,
                "PROFILER_IMAGE",
                PROFILER_IMAGE_REPOSITORY + "@sha256:" + "0" * 64,
            ),
            self.assertRaisesRegex(ValueError, "invalid"),
        ):
            require_active_profiler()

    def test_dockerfile_has_exact_offline_build_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "containers/profiler/Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"FROM --platform=linux/amd64 {PROFILER_BASE_IMAGE}", dockerfile)
        self.assertIn('ENTRYPOINT ["/opt/kernel-research/bin/bounded-profiler"]', dockerfile)
        self.assertIn("verify-toolchain", dockerfile)
        self.assertIn("org.opencontainers.image.revision", dockerfile)
        self.assertIn(
            f"USER {contract.PROFILER_IMAGE_UID}:{contract.PROFILER_IMAGE_GID}",
            dockerfile,
        )
        self.assertIn("HOME=/tmp/profile-home", dockerfile)
        self.assertIn("TRITON_CACHE_DIR=/tmp/triton-cache", dockerfile)
        lowered = dockerfile.lower()
        for forbidden in (
            "apt-get",
            "apt ",
            "pip install",
            "conda install",
            "curl ",
            "wget ",
            "git clone",
        ):
            self.assertNotIn(forbidden, lowered)
        dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
        self.assertTrue(dockerignore.startswith("**\n"))
        self.assertNotIn("README.md", dockerignore)


class ProfilerWorkerTests(unittest.TestCase):
    @staticmethod
    def _request(
        recipe_id: str = "metax-compile-metadata-v1",
    ) -> worker.WorkerRequest:
        recipe = contract.PROFILER_RECIPES[recipe_id]
        requires_gpu = bool(recipe["requires_gpu"])
        return worker.WorkerRequest(
            recipe_id=recipe_id,
            case_id=str(recipe["case_id"]),
            candidate_path=Path(worker.PROFILE_CANDIDATE_CONTAINER_PATH),
            result_path=Path(worker.PROFILE_RESULT_CONTAINER_PATH),
            outcome_path=Path(worker.PROFILE_OUTCOME_CONTAINER_PATH),
            raw_trace_path=Path(worker.PROFILE_TRACE_CONTAINER_PATH),
            experiment_uid="experiment",
            namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
            condition_digest="sha256:" + "1" * 64,
            artifact_id="source-sha256-v1:" + "2" * 64,
            candidate_sha256="2" * 64,
            environment_digest="sha256:" + "3" * 64,
            soak_invariant_digest="sha256:" + "4" * 64,
            campaign_id="campaign",
            run_id="run",
            budget_action_key="action",
            resource_id="gpu1" if requires_gpu else None,
            fencing_epoch=7 if requires_gpu else None,
        )

    @staticmethod
    def _arguments(
        *,
        recipe_id: str = "metax-compile-metadata-v1",
        artifact_id: str = "source-sha256-v1:" + "2" * 64,
        candidate_sha256: str = "2" * 64,
    ) -> SimpleNamespace:
        recipe = contract.PROFILER_RECIPES[recipe_id]
        requires_gpu = bool(recipe["requires_gpu"])
        return SimpleNamespace(
            recipe=recipe_id,
            case_id=recipe["case_id"],
            candidate=worker.PROFILE_CANDIDATE_CONTAINER_PATH,
            result=worker.PROFILE_RESULT_CONTAINER_PATH,
            outcome=worker.PROFILE_OUTCOME_CONTAINER_PATH,
            raw_trace=worker.PROFILE_TRACE_CONTAINER_PATH,
            max_result_bytes=worker.PROFILE_RESULT_LIMIT_BYTES,
            max_outcome_bytes=worker.PROFILE_OUTCOME_LIMIT_BYTES,
            max_raw_trace_bytes=worker.PROFILE_RAW_TRACE_LIMIT_BYTES,
            experiment_uid="experiment",
            namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
            condition_digest="sha256:" + "1" * 64,
            artifact_id=artifact_id,
            candidate_sha256=candidate_sha256,
            environment_digest="sha256:" + "3" * 64,
            soak_invariant_digest="sha256:" + "4" * 64,
            campaign_id="campaign",
            run_id="run",
            budget_action_key="action",
            resource_id="gpu1" if requires_gpu else None,
            fencing_epoch=7 if requires_gpu else None,
        )

    @staticmethod
    def _hash_side_effect(path, *, expected, field):
        return {"path": str(Path(path)), "sha256": expected}

    def test_exact_toolchain_and_help_signature_are_required(self) -> None:
        output = (
            f"=====Help Info=====\nVersion:\n{MCTRACER_VERSION}\nUsage:\n"
        ).encode("utf-8")
        with (
            mock.patch.object(worker, "_sha256_file", side_effect=self._hash_side_effect),
            mock.patch.object(
                worker.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    (MCTRACER_PATH, "--help"), 1, stdout=output, stderr=b""
                ),
            ),
        ):
            descriptor = worker.verify_toolchain()
        self.assertEqual(descriptor["tools"]["mctracer"]["sha256"], MCTRACER_SHA256)
        self.assertEqual(
            descriptor["profiler_build_profile_digest"],
            PROFILER_BUILD_PROFILE_DIGEST,
        )
        self.assertEqual(
            descriptor["tools"]["libmcToolsExt_lite.so"]["sha256"],
            MCTOOLS_EXT_LITE_SHA256,
        )
        self.assertEqual(
            descriptor["tools"]["libmcToolsExt.so"]["sha256"],
            MCTOOLS_EXT_SHA256,
        )
        material = dict(descriptor)
        digest = material.pop("digest")
        self.assertEqual(digest, canonical_sha256(material))

    def test_toolchain_rejects_version_exit_and_failure_marker_drift(self) -> None:
        good = (
            f"=====Help Info=====\nVersion:\n{MCTRACER_VERSION}\nUsage:\n"
        ).encode("utf-8")
        cases = (
            (0, good),
            (2, good),
            (1, good.replace(MCTRACER_VERSION.encode(), b"3.5.3.21-drift")),
            (1, good + b"execvpe: No such file or directory\n"),
        )
        for returncode, output in cases:
            with (
                self.subTest(returncode=returncode, output=output[-30:]),
                mock.patch.object(worker, "_sha256_file", side_effect=self._hash_side_effect),
                mock.patch.object(
                    worker.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        (MCTRACER_PATH, "--help"),
                        returncode,
                        stdout=output,
                        stderr=b"",
                    ),
                ),
                self.assertRaisesRegex(ValueError, "help output"),
            ):
                worker.verify_toolchain()

    def test_toolchain_probe_fails_closed_on_io_and_encoding_edges(self) -> None:
        good = (
            f"=====Help Info=====\nVersion:\n{MCTRACER_VERSION}\nUsage:\n"
        ).encode("utf-8")
        cases = (
            (subprocess.TimeoutExpired((MCTRACER_PATH,), 10), "unavailable"),
            (
                subprocess.CompletedProcess(
                    (MCTRACER_PATH,), 1, stdout=b"x" * (64 * 1024 + 1), stderr=b""
                ),
                "output limit",
            ),
            (
                subprocess.CompletedProcess(
                    (MCTRACER_PATH,), 1, stdout=good + b"\xff", stderr=b""
                ),
                "UTF-8",
            ),
        )
        for result, message in cases:
            runner_patch = (
                mock.patch.object(worker.subprocess, "run", side_effect=result)
                if isinstance(result, BaseException)
                else mock.patch.object(worker.subprocess, "run", return_value=result)
            )
            with (
                self.subTest(message=message),
                mock.patch.object(
                    worker, "_sha256_file", side_effect=self._hash_side_effect
                ),
                runner_patch,
                self.assertRaisesRegex(ValueError, message),
            ):
                worker.verify_toolchain()

    def test_strict_scalars_files_and_canonical_json_fail_closed(self) -> None:
        self.assertEqual(worker._strict_token("safe:value", "field"), "safe:value")
        self.assertEqual(
            worker._strict_digest("sha256:" + "a" * 64, "field"),
            "sha256:" + "a" * 64,
        )
        for value in (None, "", "space value", "x" * 257):
            with self.subTest(token=value), self.assertRaises(ValueError):
                worker._strict_token(value, "field")
        for value in (None, "a" * 64, "sha256:" + "g" * 64):
            with self.subTest(digest=value), self.assertRaises(ValueError):
                worker._strict_digest(value, "field")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "value.json"
            path.write_bytes(b'{"a":1}')
            self.assertEqual(
                worker._regular_bytes(
                    path, exact_path=str(path), limit=32, field="fixture"
                ),
                b'{"a":1}',
            )
            self.assertEqual(
                worker._strict_json_file(path, limit=32, field="fixture"),
                {"a": 1},
            )
            with self.assertRaisesRegex(ValueError, "fixed container path"):
                worker._regular_bytes(
                    path, exact_path=str(root / "other"), limit=32, field="fixture"
                )
            path.write_bytes(b'{ "a": 1 }')
            with self.assertRaisesRegex(ValueError, "canonical"):
                worker._strict_json_file(path, limit=32, field="fixture")
            path.write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError, "canonical"):
                worker._strict_json_file(path, limit=32, field="fixture")
            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "strict JSON"):
                worker._strict_json_file(path, limit=32, field="fixture")

    def test_sha256_file_checks_exact_regular_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "tool"
            path.write_bytes(b"tool")
            digest = hashlib.sha256(b"tool").hexdigest()
            self.assertEqual(
                worker._sha256_file(path, expected=digest, field="tool")["sha256"],
                digest,
            )
            with self.assertRaisesRegex(ValueError, "sha256"):
                worker._sha256_file(path, expected="0" * 64, field="tool")

    def test_request_contract_rejects_every_caller_controlled_axis(self) -> None:
        source = b"def run_kernel(*args):\n    return None\n"
        digest = hashlib.sha256(source).hexdigest()
        base = self._arguments(
            artifact_id="source-sha256-v1:" + digest,
            candidate_sha256=digest,
        )
        cases = (
            ("recipe", "unknown", "recipe"),
            ("case_id", "other", "case-id"),
            ("namespace_id", "sha256:" + "f" * 64, "CURRENT"),
            ("candidate", "/tmp/alias", "candidate path"),
            ("result", "/tmp/result", "result path"),
            ("outcome", "/tmp/outcome", "outcome path"),
            ("raw_trace", "/tmp/raw", "raw_trace path"),
            ("max_result_bytes", 1, "result byte"),
            ("max_outcome_bytes", 1, "outcome byte"),
            ("max_raw_trace_bytes", 1, "raw trace byte"),
            ("artifact_id", "unknown:artifact", "artifact ID"),
            ("candidate_sha256", "0" * 64, "source bytes"),
            ("resource_id", "gpu1", "must not receive"),
        )
        for field, value, message in cases:
            arguments = SimpleNamespace(**vars(base))
            setattr(arguments, field, value)
            with (
                self.subTest(field=field),
                mock.patch.object(worker, "_regular_bytes", return_value=source),
                mock.patch.object(
                    worker, "validate_candidate", return_value=SimpleNamespace(is_valid=True)
                ),
                mock.patch.object(
                    worker,
                    "validate_research_candidate_bounded",
                    return_value=SimpleNamespace(valid=True),
                ),
                self.assertRaisesRegex(ValueError, message),
            ):
                worker._validate_request(arguments)
        hardware = self._arguments(
            recipe_id="metax-hardware-counters-v1",
            artifact_id="source-sha256-v1:" + digest,
            candidate_sha256=digest,
        )
        hardware.resource_id = None
        with (
            mock.patch.object(worker, "_regular_bytes", return_value=source),
            mock.patch.object(
                worker, "validate_candidate", return_value=SimpleNamespace(is_valid=True)
            ),
            mock.patch.object(
                worker,
                "validate_research_candidate_bounded",
                return_value=SimpleNamespace(valid=True),
            ),
            self.assertRaisesRegex(ValueError, "fencing"),
        ):
            worker._validate_request(hardware)

    def test_compile_recipe_emits_manifest_and_only_whitelisted_unavailable(self) -> None:
        request = self._request()
        toolchain = {"digest": "sha256:" + "5" * 64, "tools": {}}
        metrics, raw, descriptor = worker._compile_recipe(
            request, b"candidate", toolchain
        )
        self.assertGreater(len(raw), 0)
        self.assertEqual(descriptor["format"], "canonical-json-manifest-v1")
        self.assertEqual(metrics["compiler_version_digest"]["status"], "AVAILABLE")
        for metric in ("register_count", "shared_memory_bytes", "spill_bytes"):
            self.assertEqual(
                metrics[metric],
                {
                    "status": "UNAVAILABLE",
                    "reason_code": "COUNTER_NOT_EXPOSED",
                },
            )

    def test_candidate_bundle_validation_uses_entrypoint_raw_bytes(self) -> None:
        source = "def run_kernel(*args):\n    return None\n"
        bundle = CandidateBundle.single_file(content=source)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.cas"
            path.write_bytes(bundle.bundle_bytes)
            arguments = SimpleNamespace(
                recipe="metax-compile-metadata-v1",
                case_id="smoke_gate_up",
                candidate=str(path),
                result="/output/result.json",
                outcome="/output/outcome.json",
                raw_trace="/output/raw.trace",
                max_result_bytes=64 * 1024,
                max_outcome_bytes=64 * 1024,
                max_raw_trace_bytes=64 * 1024 * 1024,
                experiment_uid="experiment",
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                condition_digest="sha256:" + "1" * 64,
                artifact_id=str(bundle.artifact_id),
                candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
                environment_digest="sha256:" + "2" * 64,
                soak_invariant_digest="sha256:" + "3" * 64,
                campaign_id="campaign",
                run_id="run",
                budget_action_key="action",
                resource_id=None,
                fencing_epoch=None,
            )
            with (
                mock.patch.object(worker, "PROFILE_CANDIDATE_CONTAINER_PATH", str(path)),
                mock.patch.object(worker, "validate_candidate") as public,
                mock.patch.object(worker, "validate_research_candidate_bounded") as policy,
            ):
                public.return_value = SimpleNamespace(is_valid=True)
                policy.return_value = SimpleNamespace(valid=True)
                request, raw = worker._validate_request(arguments)
        self.assertEqual(raw, source.encode())
        self.assertEqual(request.artifact_id, str(bundle.artifact_id))
        public.assert_called_once_with(source=source)
        policy.assert_called_once_with(source)

    def test_trace_tar_is_deterministic_and_normalizes_metadata(self) -> None:
        files = (("z/data.bin", b"two"), ("a/data.bin", b"one"))
        first = worker._deterministic_tar(files)
        second = worker._deterministic_tar(tuple(reversed(files)))
        self.assertEqual(first, second)
        self.assertGreater(len(first), 0)

    def test_trace_scan_rejects_symlinks_empty_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "trace.bin").write_bytes(b"trace")
            self.assertEqual(worker._trace_files(root), [("trace.bin", b"trace")])
            (root / "trace.bin").unlink()
            (root / "link").symlink_to(root / "missing")
            with self.assertRaisesRegex(ValueError, "symlink"):
                worker._trace_files(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "empty").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                worker._trace_files(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "large").write_bytes(b"12")
            with (
                mock.patch.object(worker, "PROFILE_TRACE_MEMBER_LIMIT_BYTES", 1),
                self.assertRaisesRegex(ValueError, "bounded regular file|size limit"),
            ):
                worker._trace_files(root)

    def test_hardware_requires_success_sentinel_and_nonempty_trace(self) -> None:
        request = SimpleNamespace(
            recipe_id="metax-hardware-counters-v1",
            case_id="quick_decode_gate_up",
            recipe={
                "metrics": (
                    ("gpu_active_percent", "number", "percent"),
                    ("achieved_occupancy_percent", "number", "percent"),
                    ("dram_bandwidth_gbps", "number", "GB/s"),
                    ("l2_hit_percent", "number", "percent"),
                )
            },
        )
        sentinel = {
            "status": "SUCCESS",
            "phase": "tracked",
            "correctness_passed": True,
            "target_completed": True,
            "warmup_launches": 0,
            "tracked_launches": 10,
            "target_success_sentinel": "METAX_PROFILE_TARGET_SUCCESS_V1",
            "case_id": "quick_decode_gate_up",
        }
        warmup_sentinel = {
            "status": "SUCCESS",
            "phase": "warmup",
            "correctness_passed": True,
            "target_completed": True,
            "warmup_launches": 10,
            "tracked_launches": 0,
            "target_success_sentinel": "METAX_PROFILE_WARMUP_SUCCESS_V1",
            "case_id": "quick_decode_gate_up",
        }
        toolchain = {"digest": "sha256:" + "4" * 64}
        with (
            mock.patch.object(
                worker,
                "_copy_candidate_to_tmp",
                return_value=Path("/tmp/candidate/kernel.py"),
            ),
            mock.patch.object(worker, "_TRACE_DIRECTORY") as trace_dir,
            mock.patch.object(worker, "_write_outcome"),
            mock.patch.object(
                worker,
                "_strict_json_file",
                side_effect=(warmup_sentinel, sentinel),
            ),
            mock.patch.object(worker, "_trace_files", return_value=[]),
            mock.patch.object(
                worker.subprocess,
                "run",
                side_effect=(
                    subprocess.CompletedProcess(
                        ("python",), 0, stdout=b"warm", stderr=b""
                    ),
                    subprocess.CompletedProcess(
                        (MCTRACER_PATH,), 1, stdout=b"ok", stderr=b""
                    ),
                ),
            ) as run_process,
        ):
            trace_dir.name = "metax-mctx"
            trace_dir.mkdir.return_value = None
            with self.assertRaisesRegex(ValueError, "no non-empty trace files"):
                # Preserve the real rejection even though the directory scan is mocked.
                with mock.patch.object(
                    worker,
                    "_deterministic_tar",
                    side_effect=ValueError("mcTracer emitted no non-empty trace files"),
                ):
                    worker._hardware_recipe(request, b"source", toolchain)
        warmup, traced = run_process.call_args_list
        warmup_argv = warmup.args[0]
        traced_argv = traced.args[0]
        self.assertNotIn(MCTRACER_PATH, warmup_argv)
        self.assertIn("warmup", warmup_argv)
        self.assertEqual(
            traced_argv[:6],
            (
                MCTRACER_PATH,
                "--mctx",
                "--odname",
                "metax-mctx",
                "--name",
                "bounded-profile",
            ),
        )
        self.assertIn("tracked", traced_argv)

    def test_hardware_accepts_only_exit_zero_or_one_with_complete_evidence(self) -> None:
        request = self._request("metax-hardware-counters-v1")
        warmup = {
            "status": "SUCCESS",
            "phase": "warmup",
            "correctness_passed": True,
            "target_completed": True,
            "warmup_launches": 10,
            "tracked_launches": 0,
            "target_success_sentinel": "METAX_PROFILE_WARMUP_SUCCESS_V1",
            "case_id": request.case_id,
        }
        tracked = {
            "status": "SUCCESS",
            "phase": "tracked",
            "correctness_passed": True,
            "target_completed": True,
            "warmup_launches": 0,
            "tracked_launches": 10,
            "target_success_sentinel": "METAX_PROFILE_TARGET_SUCCESS_V1",
            "case_id": request.case_id,
        }
        toolchain = {"digest": "sha256:" + "5" * 64}
        for exit_code in (0, 1):
            with (
                self.subTest(exit_code=exit_code),
                mock.patch.object(
                    worker,
                    "_copy_candidate_to_tmp",
                    return_value=Path("/tmp/candidate/kernel.py"),
                ),
                mock.patch.object(worker, "_TRACE_DIRECTORY") as trace_dir,
                mock.patch.object(worker, "_write_outcome") as write_outcome,
                mock.patch.object(
                    worker, "_strict_json_file", side_effect=(warmup, tracked)
                ),
                mock.patch.object(
                    worker, "_trace_files", return_value=[("trace.bin", b"trace")]
                ),
                mock.patch.object(
                    worker.subprocess,
                    "run",
                    side_effect=(
                        subprocess.CompletedProcess(
                            ("python",), 0, stdout=b"warm", stderr=b""
                        ),
                        subprocess.CompletedProcess(
                            (MCTRACER_PATH,),
                            exit_code,
                            stdout=b"trace complete",
                            stderr=b"",
                        ),
                    ),
                ),
            ):
                trace_dir.name = "metax-mctx"
                trace_dir.mkdir.return_value = None
                metrics, raw, descriptor = worker._hardware_recipe(
                    request, b"candidate", toolchain
                )
            self.assertGreater(len(raw), 0)
            self.assertEqual(descriptor["mctracer_exit_code"], exit_code)
            self.assertEqual(descriptor["file_count"], 1)
            self.assertTrue(
                all(value["status"] == "UNAVAILABLE" for value in metrics.values())
            )
            self.assertEqual(write_outcome.call_count, 2)

    def test_fixed_gpu_target_runs_only_frozen_case_and_exact_launch_counts(self) -> None:
        from kernel_research import backends, cases

        candidate = Path("/tmp/candidate/kernel.py")
        sentinel = Path("/output/target-sentinel.json")
        tensors = {"arguments": (object(),), "out": object()}
        run_kernel = mock.Mock()
        module = SimpleNamespace(run_kernel=run_kernel)
        case = SimpleNamespace(name="quick_decode_gate_up")
        validation = SimpleNamespace(valid=True, sha256="f" * 64)
        for phase, expected_launches, expected_marker in (
            ("warmup", 11, "METAX_PROFILE_WARMUP_SUCCESS_V1"),
            ("tracked", 10, "METAX_PROFILE_TARGET_SUCCESS_V1"),
        ):
            written: dict[str, object] = {}

            def capture(_path, value, **_kwargs):
                written.update(value)

            run_kernel.reset_mock()
            with (
                self.subTest(phase=phase),
                mock.patch.object(
                    worker, "validate_candidate", return_value=validation
                ),
                mock.patch.object(
                    backends, "_load_c500_runtime", return_value=((object(), object()), None)
                ),
                mock.patch.object(
                    backends,
                    "_find_accelerator",
                    return_value=({"device": "maca"}, None),
                ),
                mock.patch.object(cases, "get_suite", return_value=(case,)) as suite,
                mock.patch.object(cases, "generate_case", return_value=object()),
                mock.patch.object(
                    backends, "_copy_dataset_to_device", return_value=tensors
                ),
                mock.patch.object(
                    backends, "_clone_readonly_inputs", return_value=object()
                ),
                mock.patch.object(backends, "_torch_reference", return_value=object()),
                mock.patch.object(backends, "_import_candidate", return_value=module),
                mock.patch.object(backends, "_matched_ratio", return_value=1.0),
                mock.patch.object(
                    backends, "_readonly_inputs_unchanged", return_value=(True, None)
                ),
                mock.patch.object(backends, "_synchronize"),
                mock.patch.object(worker, "_canonical_write", side_effect=capture),
            ):
                self.assertEqual(
                    worker._run_fixed_case_target(
                        candidate, sentinel, phase=phase
                    ),
                    0,
                )
            self.assertEqual(run_kernel.call_count, expected_launches)
            self.assertEqual(written["status"], "SUCCESS")
            self.assertEqual(written["target_success_sentinel"], expected_marker)
            self.assertEqual(written["case_id"], "quick_decode_gate_up")
            suite.assert_called_once_with(
                "quick",
                evaluation_protocol_id=worker.CURRENT_C500_EVALUATION_PROTOCOL_ID,
            )

    def test_fixed_gpu_target_records_failure_without_raising(self) -> None:
        written: dict[str, object] = {}

        def capture(_path, value, **_kwargs):
            written.update(value)

        with (
            mock.patch.object(
                worker,
                "validate_candidate",
                return_value=SimpleNamespace(valid=False),
            ),
            mock.patch.object(worker, "_canonical_write", side_effect=capture),
        ):
            self.assertEqual(
                worker._run_fixed_case_target(
                    Path("/tmp/candidate/kernel.py"),
                    Path("/output/target-sentinel.json"),
                    phase="tracked",
                ),
                2,
            )
        self.assertEqual(written["status"], "FAILED")
        self.assertEqual(written["error_type"], "ValueError")
        with self.assertRaisesRegex(ValueError, "phase"):
            worker._run_fixed_case_target(
                Path("/tmp/candidate/kernel.py"),
                Path("/output/target-sentinel.json"),
                phase="other",
            )

    def test_hardware_failures_preserve_known_unknown_and_hard_boundaries(self) -> None:
        request = self._request("metax-hardware-counters-v1")
        warmup = {
            "status": "SUCCESS",
            "phase": "warmup",
            "correctness_passed": True,
            "target_completed": True,
            "warmup_launches": 10,
            "tracked_launches": 0,
            "target_success_sentinel": "METAX_PROFILE_WARMUP_SUCCESS_V1",
            "case_id": request.case_id,
        }
        tracked = {
            "status": "SUCCESS",
            "phase": "tracked",
            "correctness_passed": True,
            "target_completed": True,
            "warmup_launches": 0,
            "tracked_launches": 10,
            "target_success_sentinel": "METAX_PROFILE_TARGET_SUCCESS_V1",
            "case_id": request.case_id,
        }
        toolchain = {"digest": "sha256:" + "5" * 64}
        cases = (
            (
                (
                    subprocess.CompletedProcess(
                        ("python",), 0, stdout=b"gpu reset", stderr=b""
                    ),
                ),
                (warmup,),
                RuntimeError,
                "fatal GPU",
            ),
            (
                (
                    subprocess.CompletedProcess(
                        ("python",), 1, stdout=b"warm", stderr=b""
                    ),
                ),
                (warmup,),
                ValueError,
                "warmup",
            ),
            (
                (
                    subprocess.CompletedProcess(
                        ("python",), 0, stdout=b"warm", stderr=b""
                    ),
                    subprocess.CompletedProcess(
                        (MCTRACER_PATH,), 2, stdout=b"done", stderr=b""
                    ),
                ),
                (warmup, tracked),
                ValueError,
                "unaccepted exit",
            ),
            (
                (
                    subprocess.CompletedProcess(
                        ("python",), 0, stdout=b"warm", stderr=b""
                    ),
                    subprocess.CompletedProcess(
                        (MCTRACER_PATH,),
                        1,
                        stdout=b"execvpe: missing",
                        stderr=b"",
                    ),
                ),
                (warmup, tracked),
                ValueError,
                "failure marker",
            ),
            (
                (
                    subprocess.CompletedProcess(
                        ("python",), 0, stdout=b"warm", stderr=b""
                    ),
                    subprocess.CompletedProcess(
                        (MCTRACER_PATH,), 1, stdout=b"gpu reset", stderr=b""
                    ),
                ),
                (warmup, tracked),
                RuntimeError,
                "fatal GPU",
            ),
        )
        for process_results, sentinels, error, message in cases:
            with (
                self.subTest(message=message),
                mock.patch.object(
                    worker,
                    "_copy_candidate_to_tmp",
                    return_value=Path("/tmp/candidate/kernel.py"),
                ),
                mock.patch.object(worker, "_TRACE_DIRECTORY") as trace_dir,
                mock.patch.object(worker, "_write_outcome"),
                mock.patch.object(
                    worker, "_strict_json_file", side_effect=sentinels
                ),
                mock.patch.object(
                    worker.subprocess, "run", side_effect=process_results
                ),
                self.assertRaisesRegex(error, message),
            ):
                trace_dir.name = "metax-mctx"
                trace_dir.mkdir.return_value = None
                worker._hardware_recipe(request, b"candidate", toolchain)

    def test_run_worker_commits_compile_evidence_and_classifies_failure(self) -> None:
        request = self._request()
        toolchain = {"digest": "sha256:" + "5" * 64}
        metrics = {
            "compiler_version_digest": {
                "status": "AVAILABLE",
                "value": toolchain["digest"],
            }
        }
        descriptor = {
            "format": "canonical-json-manifest-v1",
            "sha256": "sha256:" + "6" * 64,
            "byte_size": 2,
            "file_count": 1,
        }
        with (
            mock.patch.object(Path, "mkdir"),
            mock.patch.object(
                worker, "_validate_request", return_value=(request, b"candidate")
            ),
            mock.patch.object(worker, "verify_toolchain", return_value=toolchain),
            mock.patch.object(
                worker,
                "_compile_recipe",
                return_value=(metrics, b"{}", descriptor),
            ),
            mock.patch.object(worker, "_atomic_write") as atomic_write,
            mock.patch.object(worker, "_canonical_write") as canonical_write,
            mock.patch.object(worker, "_write_outcome") as write_outcome,
        ):
            self.assertEqual(worker.run_worker(SimpleNamespace()), 0)
        atomic_write.assert_called_once()
        canonical_write.assert_called_once()
        self.assertEqual(write_outcome.call_args.kwargs["status"], "SUCCESS")

        with (
            mock.patch.object(Path, "mkdir"),
            mock.patch.object(
                worker, "_validate_request", return_value=(request, b"candidate")
            ),
            mock.patch.object(
                worker, "verify_toolchain", side_effect=ValueError("tool drift")
            ),
            mock.patch.object(
                worker,
                "_strict_json_file",
                return_value={
                    "status": "RUNNING",
                    "gpu_state": "NOT_STARTED",
                    "completion_trusted": True,
                },
            ),
            mock.patch.object(worker, "_write_outcome") as failure_outcome,
            mock.patch.object(worker.sys, "stderr"),
        ):
            self.assertEqual(worker.run_worker(SimpleNamespace()), 2)
        self.assertEqual(
            failure_outcome.call_args.kwargs["reason_code"],
            "WORKER_KNOWN_FAILURE",
        )

    def test_internal_target_cli_rejects_path_alias_before_execution(self) -> None:
        with (
            mock.patch.object(worker, "_run_fixed_case_target") as target,
            self.assertRaisesRegex(ValueError, "paths differ"),
        ):
            worker.main(
                (
                    "_hardware-target",
                    "--phase",
                    "tracked",
                    "--candidate",
                    "/tmp/alias.py",
                    "--sentinel",
                    str(worker._TARGET_SENTINEL),
                )
            )
        target.assert_not_called()


if __name__ == "__main__":
    unittest.main()
