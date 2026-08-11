from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from kernel_research.autorun import admin
from kernel_research.autorun.admin import AdminManifest
from kernel_research.autorun.errors import ControlledRuntimeError
from kernel_research.autorun.models import ControllerConfig
from kernel_research.autorun.runtime import CommandResult
from kernel_research.history import HistoryStore

from test_autorun import FULL_CASES, SEED, SEED_HASH


def _command(*argv: str, cwd: Path) -> str:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    return completed.stdout.strip()


class AdminFixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.remote = self.root / "remote.git"
        self.repo = self.root / "repo"
        self.runtime = self.root / "runtime"
        self.state = self.runtime / "state"
        self.controller = self.runtime / "controller"
        self.secret = self.root / "deepseek-key"
        self.manifest_path = self.runtime / "admin.json"
        self.pro_path = self.runtime / "autorun.pro.json"
        self.flash_path = self.runtime / "autorun.flash.json"
        self.runtime.mkdir(parents=True)
        self.state.mkdir()
        self.controller.mkdir()
        (self.runtime / "checkpoints").mkdir()
        (self.runtime / "cache").mkdir()
        self.secret.write_text("test-secret", encoding="utf-8")
        self.secret.chmod(0o600)

        self.remote.mkdir()
        _command("git", "init", "--bare", cwd=self.remote)
        self.repo.mkdir()
        _command("git", "init", "-b", "codex/fused-moe-autoresearch", cwd=self.repo)
        _command("git", "config", "user.name", "Admin Test", cwd=self.repo)
        _command("git", "config", "user.email", "admin@example.invalid", cwd=self.repo)
        (self.repo / "kernel.py").write_text(SEED, encoding="utf-8")
        (self.repo / "README.md").write_text("fixture\n", encoding="utf-8")
        _command("git", "add", "kernel.py", "README.md", cwd=self.repo)
        _command("git", "commit", "-m", "baseline", cwd=self.repo)
        _command("git", "remote", "add", "origin", str(self.remote), cwd=self.repo)
        _command(
            "git",
            "push",
            "-u",
            "origin",
            "codex/fused-moe-autoresearch",
            cwd=self.repo,
        )
        self.commit = _command("git", "rev-parse", "HEAD", cwd=self.repo)
        self._record(SEED, SEED_HASH, score=1.0, phase="confirmation")
        base = self._config_value("deepseek/deepseek-v4-pro")
        self.pro_path.write_text(json.dumps(base), encoding="utf-8")
        self.pro_path.chmod(0o600)
        flash = dict(base)
        flash["opencode_model"] = "deepseek/deepseek-v4-flash"
        self.flash_path.write_text(json.dumps(flash), encoding="utf-8")
        self.flash_path.chmod(0o600)

    def _config_value(self, model: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "repository_dir": str(self.repo),
            "state_dir": str(self.state),
            "controller_dir": str(self.controller),
            "checkpoint_dir": str(self.runtime / "checkpoints"),
            # The fixture never invokes this as Docker.  Use the current
            # interpreter because it is guaranteed to be an executable,
            # canonical absolute path on both macOS and Linux; /bin/echo is
            # non-canonical on usr-merged Ubuntu images (/bin -> /usr/bin).
            "docker_binary": str(Path(sys.executable).resolve()),
            "proposer_image": "local/proposer@sha256:" + "1" * 64,
            "evaluator_image": "local/evaluator@sha256:" + "2" * 64,
            "deepseek_key_file": str(self.secret),
            "gpu_devices": [
                "/dev/mxcd",
                "/dev/dri/card2",
                "/dev/dri/renderD129",
            ],
            "evaluator_cache_dir": str(self.runtime / "cache"),
            "expected_git_commit": self.commit,
            "expected_kernel_hash": SEED_HASH,
            "opencode_model": model,
            "max_candidates": 5,
            "max_hours": 6,
            "max_consecutive_failures": 3,
            "proposer_timeout_sec": 1200,
            "proposer_max_output_bytes": 2 * 1024 * 1024,
            "evaluator_timeout_sec": 2700,
            "stop_after_promotion": True,
            "container_uid": os.getuid(),
            "container_gid": os.getgid(),
            "video_gid": 44,
            "proposer_cpus": 2,
            "proposer_memory": "2g",
            "evaluator_cpus": 8,
            "evaluator_memory": "24g",
            "acknowledge_gpu_passthrough_risk": True,
        }

    def _record(
        self, source: str, candidate_hash: str, *, score: float, phase: str
    ) -> None:
        cases = [
            {
                "case_id": name,
                "status": "SUCCESS",
                "matched_ratio": 1.0,
                "latency_samples_us": [100.0] * 30,
                "p50_us": 100.0,
            }
            for name in FULL_CASES
        ]
        result = {
            "status": "SUCCESS",
            "aggregate_score": score,
            "eligible_for_promotion": True,
            "promotion": {
                "phase": phase,
                "reason": "promoted",
                "confirmed": True,
            },
            "cases": cases,
        }
        with HistoryStore(self.state / "history.sqlite3", self.state) as history:
            history.record_experiment(
                candidate_source=source,
                candidate_hash=candidate_hash,
                backend="c500",
                suite="full",
                status="SUCCESS",
                promotable=True,
                aggregate_score=score,
                note="admin fixture",
                result=result,
                case_measurements=(
                    {
                        "name": case["case_id"],
                        "matched_ratio": 1.0,
                        "passed": True,
                        "raw_samples": case["latency_samples_us"],
                    }
                    for case in cases
                ),
            )

    def bootstrap(self) -> AdminManifest:
        admin.bootstrap(self.manifest_path, self.pro_path, self.flash_path)
        return AdminManifest.load(self.manifest_path)


class AdminManifestAndSyncTests(unittest.TestCase):
    def test_bootstrap_generates_strict_pinned_files_and_secret_free_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            self.assertEqual(stat.S_IMODE(fixture.manifest_path.stat().st_mode), 0o600)
            for path in (
                manifest.base_config,
                manifest.pro_config,
                manifest.flash_config,
                manifest.pro_canary_config,
                manifest.flash_canary_config,
                manifest.environment_file,
            ):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            pro = json.loads(manifest.pro_config.read_text(encoding="utf-8"))
            flash = json.loads(manifest.flash_config.read_text(encoding="utf-8"))
            changed = {key for key in pro if pro.get(key) != flash.get(key)}
            self.assertEqual(changed, {"opencode_model"})
            self.assertEqual(
                json.loads(manifest.pro_canary_config.read_text())["max_candidates"], 1
            )
            environment = manifest.environment_file.read_text(encoding="utf-8")
            self.assertIn("AUTORESEARCH_FLASH_CONFIG", environment)
            self.assertIn("AUTORESEARCH_FRAMEWORK_COMMIT", environment)
            self.assertIn(fixture.commit, environment)
            self.assertNotIn("test-secret", environment)
            base = json.loads(manifest.base_config.read_text(encoding="utf-8"))
            self.assertNotIn("opencode_model", base)
            self.assertEqual(base["expected_kernel_hash"], SEED_HASH)
            self.assertEqual(base["framework_git_commit"], fixture.commit)

    def test_bootstrap_adopts_current_commit_but_not_a_new_kernel_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            old_commit = fixture.commit
            (fixture.repo / "README.md").write_text("pulled update\n", encoding="utf-8")
            _command("git", "add", "README.md", cwd=fixture.repo)
            _command("git", "commit", "-m", "pulled update", cwd=fixture.repo)
            current = _command("git", "rev-parse", "HEAD", cwd=fixture.repo)
            self.assertNotEqual(old_commit, current)
            manifest = fixture.bootstrap()
            pro = json.loads(manifest.pro_config.read_text(encoding="utf-8"))
            self.assertEqual(pro["expected_git_commit"], current)
            self.assertEqual(pro["framework_git_commit"], old_commit)
            self.assertEqual(pro["expected_kernel_hash"], SEED_HASH)

    def test_bootstrap_rejects_an_unproven_kernel_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            for path in (fixture.pro_path, fixture.flash_path):
                value = json.loads(path.read_text(encoding="utf-8"))
                value["expected_kernel_hash"] = "0" * 64
                path.write_text(json.dumps(value), encoding="utf-8")
                path.chmod(0o600)
            with self.assertRaisesRegex(ControlledRuntimeError, "input kernel pin"):
                admin.bootstrap(
                    fixture.manifest_path,
                    fixture.pro_path,
                    fixture.flash_path,
                )

    def test_manifest_rejects_unknown_fields_modes_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            value = manifest.to_dict()
            value["unknown"] = True
            fixture.manifest_path.write_text(json.dumps(value), encoding="utf-8")
            fixture.manifest_path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                AdminManifest.load(fixture.manifest_path)
            value.pop("unknown")
            fixture.manifest_path.write_text(json.dumps(value), encoding="utf-8")
            fixture.manifest_path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "mode"):
                AdminManifest.load(fixture.manifest_path)

            fixture.manifest_path.unlink()
            fixture.manifest_path.symlink_to(manifest.base_config)
            with self.assertRaisesRegex(ValueError, "canonical|symlink"):
                AdminManifest.load(fixture.manifest_path)

    def test_manifest_and_base_validation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            with self.assertRaisesRegex(ValueError, "non-empty absolute"):
                admin._canonical_path(None, "path")
            with self.assertRaisesRegex(ValueError, "must be absolute"):
                admin._canonical_path("relative", "path")

            original = manifest.to_dict()

            def rejected(mutator, message: str) -> None:
                value = json.loads(json.dumps(original))
                mutator(value)
                fixture.manifest_path.write_text(json.dumps(value), encoding="utf-8")
                fixture.manifest_path.chmod(0o600)
                with self.assertRaisesRegex(ValueError, message):
                    AdminManifest.load(fixture.manifest_path)

            rejected(lambda value: value.pop("host_python"), "missing fields")
            rejected(lambda value: value.update(schema_version=2), "schema_version")
            rejected(lambda value: value.update(expected_branch="bad branch"), "without whitespace")
            rejected(
                lambda value: value.update(flash_config=value["pro_config"]),
                "distinct",
            )
            rejected(
                lambda value: value.update(environment_file=str(fixture.root / "env.sh")),
                "runtime_root",
            )

            fixture.manifest_path.write_text(json.dumps(original), encoding="utf-8")
            fixture.manifest_path.chmod(0o600)
            base = json.loads(manifest.base_config.read_text(encoding="utf-8"))
            for mutation, message in (
                ({"unknown": True}, "unknown fields"),
                ({"opencode_model": admin.PRO_MODEL}, "must not contain"),
                ({"max_candidates": 2}, "must be 5"),
            ):
                changed = dict(base)
                changed.update(mutation)
                manifest.base_config.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    admin._load_base(manifest)
            manifest.base_config.write_text(json.dumps(base), encoding="utf-8")

            invalid = fixture.root / "invalid.json"
            invalid.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "could not read"):
                admin._read_json(invalid, "fixture")
            invalid.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                admin._read_json(invalid, "fixture")

            with self.assertRaises(ControlledRuntimeError):
                admin._run([sys.executable, "-c", "raise SystemExit(7)"])

    def test_sync_updates_commit_but_never_adopts_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            (fixture.repo / "README.md").write_text("next\n", encoding="utf-8")
            _command("git", "add", "README.md", cwd=fixture.repo)
            _command("git", "commit", "-m", "next", cwd=fixture.repo)
            head = _command("git", "rev-parse", "HEAD", cwd=fixture.repo)
            with mock.patch.object(admin, "_active_containers", return_value=[]):
                report = admin.sync(manifest)
            self.assertEqual(report["identity"]["head"], head)
            for path in admin._config_targets(manifest).values():
                value = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(value["expected_git_commit"], head)
                self.assertEqual(value["expected_kernel_hash"], SEED_HASH)

            with mock.patch.object(admin, "_active_containers", return_value=["kar-live"]):
                with self.assertRaisesRegex(ControlledRuntimeError, "containers"):
                    admin.sync(manifest)

            (fixture.repo / "kernel.py").write_text(SEED + "\n# drift\n", encoding="utf-8")
            _command("git", "add", "kernel.py", cwd=fixture.repo)
            _command("git", "commit", "-m", "unadopted kernel", cwd=fixture.repo)
            with mock.patch.object(admin, "_active_containers", return_value=[]):
                with self.assertRaisesRegex(ControlledRuntimeError, "pinned baseline"):
                    admin.sync(manifest)

    def test_atomic_publish_restores_previous_group_after_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = root / "one"
            second = root / "two"
            first.write_bytes(b"old-one")
            second.write_bytes(b"old-two")
            real_replace = os.replace
            calls = 0

            def fail_second(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publish failure")
                return real_replace(source, destination)

            with mock.patch.object(admin.os, "replace", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "injected"):
                    admin._atomic_publish(
                        {first: (b"new-one", 0o600), second: (b"new-two", 0o600)}
                    )
            self.assertEqual(first.read_bytes(), b"old-one")
            self.assertEqual(second.read_bytes(), b"old-two")


class AdminVerificationAndUpdateTests(unittest.TestCase):
    def test_static_cpu_doctor_levels_and_cpu_argv_have_no_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            inspect = subprocess.CompletedProcess([], 0, "[]", "")
            original_run = admin._run

            def inspect_images(argv, **kwargs):
                if len(argv) > 2 and argv[1:3] == ["image", "inspect"]:
                    return inspect
                return original_run(argv, **kwargs)

            with (
                mock.patch.object(ControllerConfig, "validate_host", return_value=[]),
                mock.patch.object(admin, "_active_containers", return_value=[]),
                mock.patch.object(admin, "_run", side_effect=inspect_images),
            ):
                static = admin.verify(manifest, "static")
            self.assertEqual(static["status"], "SUCCESS")
            self.assertEqual(static["static"]["models"]["flash"], admin.FLASH_MODEL)
            self.assertEqual(
                static["static"]["model_settings"]["pro"]["reasoning_effort"],
                "max",
            )
            self.assertEqual(
                static["static"]["model_settings"]["flash"][
                    "reasoning_effort"
                ],
                "max",
            )
            self.assertEqual(
                static["static"]["model_settings"]["flash"][
                    "output_tokens"
                ],
                384_000,
            )
            self.assertEqual(
                static["static"]["model_settings"]["flash"][
                    "declared_output_tokens"
                ],
                384_000,
            )
            self.assertEqual(
                static["static"]["model_settings"]["flash"][
                    "request_output_token_cap"
                ],
                384_000,
            )

            real_write = admin.write_opencode_config

            for wrong_model, wrong_effort in (
                (admin.PRO_MODEL, "high"),
                (admin.FLASH_MODEL, "high"),
            ):
                def write_wrong_effort(path, config):
                    real_write(path, config)
                    if config.opencode_model != wrong_model:
                        return
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value["agent"]["kernel-proposer"][
                        "reasoningEffort"
                    ] = wrong_effort
                    path.write_text(json.dumps(value), encoding="utf-8")

                with (
                    mock.patch.object(
                        ControllerConfig, "validate_host", return_value=[]
                    ),
                    mock.patch.object(admin, "_active_containers", return_value=[]),
                    mock.patch.object(admin, "_run", side_effect=inspect_images),
                    mock.patch.object(
                        admin,
                        "write_opencode_config",
                        side_effect=write_wrong_effort,
                    ),
                    self.assertRaisesRegex(
                        ControlledRuntimeError,
                        "OpenCode proposer boundary mismatch",
                    ),
                ):
                    admin.verify(manifest, "static")

            real_argv = admin.proposer_argv

            def argv_with_wrong_output_cap(*args, **kwargs):
                argv = real_argv(*args, **kwargs)
                expected = (
                    "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=384000"
                )
                argv[argv.index(expected)] = (
                    "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=32000"
                )
                return argv

            with (
                mock.patch.object(
                    ControllerConfig, "validate_host", return_value=[]
                ),
                mock.patch.object(admin, "_active_containers", return_value=[]),
                mock.patch.object(admin, "_run", side_effect=inspect_images),
                mock.patch.object(
                    admin,
                    "proposer_argv",
                    side_effect=argv_with_wrong_output_cap,
                ),
                self.assertRaisesRegex(
                    ControlledRuntimeError,
                    "request output token cap mismatch",
                ),
            ):
                admin.verify(manifest, "static")

            with (
                mock.patch.object(admin, "_static_checks", return_value={"ok": True}),
                mock.patch.object(admin, "_cpu_tests", return_value={"status": "PASSED"}),
                mock.patch.object(admin, "_doctor_configs", return_value={"pro": {}, "flash": {}}),
            ):
                checked = admin.verify(manifest, "doctor")
            self.assertEqual(checked["cpu"]["status"], "PASSED")
            self.assertIn("flash", checked["doctor"])

            config = ControllerConfig.load(manifest.pro_config)
            success = CommandResult((), 0, "", "")
            with mock.patch.object(admin.CommandRunner, "run", return_value=success) as run:
                cpu = admin._cpu_tests(config, run_tag="fixture")
            argv = run.call_args.args[0]
            self.assertEqual(cpu["status"], "PASSED")
            self.assertIn("--network", argv)
            self.assertIn("none", argv)
            self.assertNotIn("--device", argv)
            self.assertIn("readonly", " ".join(argv))

            for result, message in (
                (CommandResult((), 0, "", "", timed_out=True), "timed out"),
                (CommandResult((), 0, "", "", output_limited=True), "exceeded"),
                (CommandResult((), 2, "bad", "failure"), "failed"),
            ):
                with (
                    mock.patch.object(admin.CommandRunner, "run", return_value=result),
                    self.assertRaisesRegex(ControlledRuntimeError, message),
                ):
                    admin._cpu_tests(config, run_tag="failure")

            listed = (
                subprocess.CompletedProcess([], 0, "kar-one\n", ""),
                subprocess.CompletedProcess([], 0, "kar-two\nkar-one\n", ""),
            )
            with mock.patch.object(admin, "_run", side_effect=listed):
                self.assertEqual(
                    admin._active_containers(config), ["kar-one", "kar-two"]
                )

    def test_doctor_result_validation_and_active_run_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            configs = {
                "pro": ControllerConfig.load(manifest.pro_config),
                "flash": ControllerConfig.load(manifest.flash_config),
            }

            def payload(config: ControllerConfig) -> dict:
                spec = admin.OPENCODE_MODEL_SPECS[config.opencode_model]
                return {
                    "status": "SUCCESS",
                    "errors": [],
                    "proposer_model": config.opencode_model,
                    "proposer_reasoning_effort": spec.reasoning_effort,
                    "proposer_declared_output_tokens": spec.output_tokens,
                    "proposer_request_output_token_cap": (
                        spec.request_output_token_cap
                    ),
                    "c500_probe": {
                        "environment": {"compile_probe_status": "PASSED"}
                    },
                }

            controllers = [
                mock.Mock(doctor=mock.Mock(return_value=payload(configs["pro"]))),
                mock.Mock(doctor=mock.Mock(return_value=payload(configs["flash"]))),
            ]
            with mock.patch.object(admin, "ResearchController", side_effect=controllers):
                result = admin._doctor_configs(configs)
            self.assertEqual(result["flash"]["status"], "SUCCESS")
            self.assertEqual(
                result["flash"]["proposer_reasoning_effort"], "max"
            )
            self.assertEqual(
                result["flash"]["proposer_declared_output_tokens"],
                384_000,
            )
            self.assertEqual(
                result["flash"]["proposer_request_output_token_cap"],
                384_000,
            )

            failures = (
                ({"status": "FAILED", "errors": ["bad"]}, "doctor failed"),
                (
                    {
                        "status": "SUCCESS",
                        "proposer_model": admin.FLASH_MODEL,
                        "c500_probe": {"environment": {"compile_probe_status": "PASSED"}},
                    },
                    "model identity",
                ),
                (
                    {
                        "status": "SUCCESS",
                        "proposer_model": admin.PRO_MODEL,
                        "proposer_reasoning_effort": "high",
                        "proposer_declared_output_tokens": 384_000,
                        "proposer_request_output_token_cap": 384_000,
                        "c500_probe": {"environment": {"compile_probe_status": "PASSED"}},
                    },
                    "reasoning effort identity",
                ),
                (
                    {
                        "status": "SUCCESS",
                        "proposer_model": admin.PRO_MODEL,
                        "proposer_reasoning_effort": "max",
                        "proposer_declared_output_tokens": 65_536,
                        "proposer_request_output_token_cap": 384_000,
                        "c500_probe": {"environment": {"compile_probe_status": "PASSED"}},
                    },
                    "declared output identity",
                ),
                (
                    {
                        "status": "SUCCESS",
                        "proposer_model": admin.PRO_MODEL,
                        "proposer_reasoning_effort": "max",
                        "proposer_declared_output_tokens": 384_000,
                        "proposer_request_output_token_cap": 32_000,
                        "c500_probe": {"environment": {"compile_probe_status": "PASSED"}},
                    },
                    "request output cap identity",
                ),
                (
                    {
                        "status": "SUCCESS",
                        "proposer_model": admin.PRO_MODEL,
                        "proposer_reasoning_effort": "max",
                        "proposer_declared_output_tokens": 384_000,
                        "proposer_request_output_token_cap": 384_000,
                        "c500_probe": {"environment": {"compile_probe_status": "FAILED"}},
                    },
                    "compile probe",
                ),
            )
            for failed_payload, message in failures:
                with (
                    mock.patch.object(
                        admin,
                        "ResearchController",
                        return_value=mock.Mock(
                            doctor=mock.Mock(return_value=failed_payload)
                        ),
                    ),
                    self.assertRaisesRegex(ControlledRuntimeError, message),
                ):
                    admin._doctor_configs(configs)

            database = configs["pro"].controller_dir / "controller.sqlite3"
            database.touch()
            fake_store = mock.MagicMock()
            fake_store.__enter__.return_value.latest_run.return_value = {
                "id": "run-id",
                "status": "RUNNING",
            }
            with mock.patch.object(admin, "ControllerStore", return_value=fake_store):
                self.assertIn("run-id", admin._active_run_error(configs["pro"]) or "")

    def test_post_update_publishes_only_after_tests_install_and_optional_doctor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            (fixture.repo / "README.md").write_text("post update\n", encoding="utf-8")
            _command("git", "add", "README.md", cwd=fixture.repo)
            _command("git", "commit", "-m", "post", cwd=fixture.repo)
            head = _command("git", "rev-parse", "HEAD", cwd=fixture.repo)
            completed = subprocess.CompletedProcess([], 0, "", "")
            original_run = admin._run

            def install_only(argv, **kwargs):
                if "pip" in argv:
                    return completed
                return original_run(argv, **kwargs)

            with (
                mock.patch.object(admin, "_cpu_tests", return_value={"status": "PASSED"}),
                mock.patch.object(admin, "_run", side_effect=install_only) as command,
                mock.patch.object(admin, "_doctor_configs", return_value={"ok": True}),
            ):
                report = admin._post_update(manifest, doctor=True)
            self.assertEqual(report["identity"]["head"], head)
            self.assertEqual(report["doctor"], {"ok": True})
            self.assertTrue(
                any("pip" in call.args[0] for call in command.call_args_list)
            )
            self.assertEqual(
                json.loads(manifest.pro_config.read_text())["expected_git_commit"], head
            )

    def test_update_fetches_fast_forward_and_keeps_old_config_on_child_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            peer = fixture.root / "peer"
            _command(
                "git",
                "clone",
                "--branch",
                "codex/fused-moe-autoresearch",
                str(fixture.remote),
                str(peer),
                cwd=fixture.root,
            )
            _command("git", "config", "user.name", "Peer", cwd=peer)
            _command("git", "config", "user.email", "peer@example.invalid", cwd=peer)
            (peer / "README.md").write_text("upstream\n", encoding="utf-8")
            _command("git", "add", "README.md", cwd=peer)
            _command("git", "commit", "-m", "upstream", cwd=peer)
            _command("git", "push", cwd=peer)
            old_config = manifest.pro_config.read_bytes()
            original_run = admin._run

            def fail_child(argv, **kwargs):
                if "kernel_research.autorun.admin" in argv:
                    return subprocess.CompletedProcess(argv, 2, "", "child failed")
                return original_run(argv, **kwargs)

            with (
                mock.patch.object(admin, "_active_containers", return_value=[]),
                mock.patch.object(admin, "_run", side_effect=fail_child),
            ):
                with self.assertRaisesRegex(ControlledRuntimeError, "formal configs remain pinned"):
                    admin.update(manifest, doctor=False)
            self.assertNotEqual(
                _command("git", "rev-parse", "HEAD", cwd=fixture.repo), fixture.commit
            )
            self.assertEqual(manifest.pro_config.read_bytes(), old_config)

    def test_update_success_and_rejects_non_fast_forward(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            original_run = admin._run

            def successful_child(argv, **kwargs):
                if "kernel_research.autorun.admin" in argv:
                    self.assertIn("--doctor", argv)
                    descriptor = int(
                        argv[argv.index("--maintenance-lock-fd") + 1]
                    )
                    self.assertEqual(kwargs.get("pass_fds"), (descriptor,))
                    self.assertEqual(
                        admin.verify_inherited_campaign_maintenance_fence(
                            fixture.runtime, descriptor
                        ),
                        descriptor,
                    )
                    return subprocess.CompletedProcess(
                        argv, 0, json.dumps({"status": "SUCCESS"}), ""
                    )
                return original_run(argv, **kwargs)

            with (
                mock.patch.object(admin, "_active_containers", return_value=[]),
                mock.patch.object(admin, "_run", side_effect=successful_child),
            ):
                report = admin.update(manifest, doctor=True)
            self.assertEqual(report["status"], "SUCCESS")
            self.assertEqual(report["old_commit"], report["new_commit"])

            def non_fast_forward(argv, **kwargs):
                if len(argv) > 2 and argv[1:3] == ["merge-base", "--is-ancestor"]:
                    return subprocess.CompletedProcess(argv, 1, "", "")
                return original_run(argv, **kwargs)

            with (
                mock.patch.object(admin, "_active_containers", return_value=[]),
                mock.patch.object(admin, "_run", side_effect=non_fast_forward),
                self.assertRaisesRegex(ControlledRuntimeError, "fast-forward descendant"),
            ):
                admin.update(manifest, doctor=False)

    def test_adopt_baseline_requires_confirmed_history_and_updates_all_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdminFixture(Path(temporary))
            manifest = fixture.bootstrap()
            candidate = SEED + "\n# accepted candidate\n"
            candidate_hash = admin._sha256_bytes(candidate.encode("utf-8"))
            (fixture.repo / "kernel.py").write_text(candidate, encoding="utf-8")
            _command("git", "add", "kernel.py", cwd=fixture.repo)
            _command("git", "commit", "-m", "accept kernel", cwd=fixture.repo)
            fixture._record(candidate, candidate_hash, score=2.0, phase="confirmation")
            with mock.patch.object(admin, "_active_containers", return_value=[]):
                report = admin.adopt_baseline(
                    manifest, candidate_hash=candidate_hash, doctor=False
                )
            self.assertEqual(report["candidate_hash"], candidate_hash)
            for path in admin._config_targets(manifest).values():
                value = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(value["expected_kernel_hash"], candidate_hash)

    def test_cli_errors_are_stable_json(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = admin.main(["sync", "--manifest", "/not/a/manifest.json"])
        self.assertEqual(code, 2)
        value = json.loads(stderr.getvalue())
        self.assertEqual(value["status"], "FAILED")
        self.assertEqual(value["schema_version"], 1)

    def test_cli_dispatches_every_admin_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runtime = root / "runtime"
            runtime.mkdir()
            manifest = AdminManifest(
                repository_dir=root,
                runtime_root=runtime,
                base_config=runtime / "autorun.base.json",
                pro_config=runtime / "autorun.pro.json",
                flash_config=runtime / "autorun.flash.json",
                pro_canary_config=runtime / "autorun.pro.canary.json",
                flash_canary_config=runtime / "autorun.flash.canary.json",
                environment_file=runtime / "env.sh",
                expected_branch="branch",
                expected_upstream="origin/branch",
                host_python=Path(sys.executable).resolve(),
            )
            success = {"schema_version": 1, "status": "SUCCESS"}
            operations = (
                (
                    [
                        "bootstrap",
                        "--manifest",
                        str(runtime / "admin.json"),
                        "--pro-config",
                        str(runtime / "autorun.pro.json"),
                        "--flash-config",
                        str(runtime / "autorun.flash.json"),
                    ],
                    "bootstrap",
                ),
                (["sync", "--manifest", str(runtime / "admin.json")], "sync"),
                (
                    ["verify", "--manifest", str(runtime / "admin.json"), "--level", "static"],
                    "verify",
                ),
                (
                    ["update", "--manifest", str(runtime / "admin.json"), "--doctor"],
                    "update",
                ),
                (
                    [
                        "adopt-baseline",
                        "--manifest",
                        str(runtime / "admin.json"),
                        "--candidate-hash",
                        "a" * 64,
                    ],
                    "adopt_baseline",
                ),
                (
                    ["_post-update", "--manifest", str(runtime / "admin.json")],
                    "_post_update",
                ),
            )
            for argv, target in operations:
                with self.subTest(command=argv[0]):
                    output = io.StringIO()
                    with (
                        mock.patch.object(AdminManifest, "load", return_value=manifest),
                        mock.patch.object(admin, target, return_value=success),
                        mock.patch.object(admin, "_write_report"),
                        redirect_stdout(output),
                    ):
                        self.assertEqual(admin.main(argv), 0)
                    self.assertEqual(json.loads(output.getvalue())["status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
