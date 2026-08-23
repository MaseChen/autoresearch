from __future__ import annotations

import base64
from contextlib import closing
import hashlib
import io
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import kernel_research.profiling as profiling
from kernel_research.constants import CURRENT_C500_EVALUATION_PROTOCOL_ID
from kernel_research.autorun.models import ControllerConfig, GPU1_DEVICES
from kernel_research.autorun.runtime import CommandResult
from kernel_research.campaign.models import BudgetAmount, CampaignMode
from kernel_research.campaign.store import CampaignStore
from kernel_research.cli import build_parser
from kernel_research.history import HistoryStore
from kernel_research.platform.identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
)
from kernel_research.platform.profiles import CURRENT_RESEARCH_NAMESPACE
from kernel_research.platform.proposal import CandidateBundle
from kernel_research.profiling import (
    BUILTIN_PROFILE_RECIPES,
    DEFAULT_TOOL_PROBES,
    PROFILE_OUTPUT_LIMIT_BYTES,
    PROFILE_TIMEOUT_SECONDS,
    PROFILER_IMAGE,
    ProfileMetric,
    ProfileRecipe,
    ToolProbe,
    run_bounded_profile,
    run_profiling_doctor,
)


class ProfilingDoctorTests(unittest.TestCase):
    def test_host_memory_preflight_parses_exact_linux_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "meminfo"
            path.write_text(
                "MemTotal:       67108864 kB\n"
                "MemFree:         1048576 kB\n"
                "MemAvailable:   50331648 kB\n",
                encoding="ascii",
            )
            report = profiling._profile_host_memory_preflight(
                path=path,
                clock=lambda: 1_787_114_880.125,
            )
        self.assertEqual(report["kind"], "HOST_MEMORY_PREFLIGHT_V1")
        self.assertEqual(report["source"], str(path))
        self.assertEqual(report["observed_epoch_ms"], 1_787_114_880_125)
        self.assertEqual(report["mem_total_bytes"], 64 * 1024**3)
        self.assertEqual(report["mem_available_bytes"], 48 * 1024**3)
        self.assertEqual(report["required_total_bytes"], 24 * 1024**3)
        self.assertEqual(report["required_available_bytes"], 24 * 1024**3)

    def test_host_memory_preflight_fails_closed_on_malformed_or_low_state(
        self,
    ) -> None:
        fixtures = (
            (b"", "empty or oversized"),
            (b"\xff", "not ASCII"),
            (b"MemTotal: 67108864 kB\n", "incomplete"),
            (
                b"MemTotal: 67108864 kB\nMemTotal: 67108864 kB\n"
                b"MemAvailable: 50331648 kB\n",
                "repeats MemTotal",
            ),
            (
                b"MemTotal: 67108864 MB\nMemAvailable: 50331648 kB\n",
                "invalid MemTotal",
            ),
            (
                b"MemTotal: 16777216 kB\nMemAvailable: 16777216 kB\n",
                "total memory",
            ),
            (
                b"MemTotal: 67108864 kB\nMemAvailable: 16777216 kB\n",
                "available memory",
            ),
            (
                ("MemTotal: " + "9" * 30 + " kB\n"
                 "MemAvailable: 50331648 kB\n").encode("ascii"),
                "overflows MemTotal",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "meminfo"
            for content, message in fixtures:
                with self.subTest(message=message):
                    path.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, message):
                        profiling._profile_host_memory_preflight(path=path)
            path.write_bytes(b"x" * (64 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, "empty or oversized"):
                profiling._profile_host_memory_preflight(path=path)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "unavailable"):
                profiling._profile_host_memory_preflight(path=path)
        with self.assertRaisesRegex(ValueError, "absolute"):
            profiling._profile_host_memory_preflight(path=Path("meminfo"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "meminfo"
            path.write_text(
                "MemTotal: 67108864 kB\nMemAvailable: 50331648 kB\n",
                encoding="ascii",
            )
            for clock in (lambda: math.nan, lambda: True):
                with self.assertRaisesRegex(ValueError, "time"):
                    profiling._profile_host_memory_preflight(
                        path=path, clock=clock
                    )

    def test_canary_diagnostic_inventory_is_bounded_and_never_follows_symlinks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "output"
            outside = root / "outside"
            output.mkdir()
            outside.mkdir()
            (outside / "secret.txt").write_text("secret", encoding="utf-8")
            (output / "a.txt").write_text("a", encoding="utf-8")
            (output / "b.txt").write_text("b", encoding="utf-8")
            (output / "c.txt").write_text("c", encoding="utf-8")
            (output / "0escape").symlink_to(outside, target_is_directory=True)
            with mock.patch.object(
                profiling, "_CANARY_DIAGNOSTIC_FILE_LIMIT", 3
            ):
                inventory = profiling._diagnostic_file_inventory(output)
            entries = {entry["path"]: entry for entry in inventory["entries"]}
            self.assertEqual(entries["0escape"]["kind"], "SYMLINK")
            self.assertNotIn("0escape/secret.txt", entries)
            self.assertTrue(inventory["truncated"])
            self.assertFalse(inventory["scan_errors"])
            self.assertEqual(
                profiling._diagnostic_file_payload(output / "0escape")["reason_code"],
                "SYMLINK",
            )

    def test_ready_report_is_advisory_only(self) -> None:
        commands: list[tuple[str, ...]] = []

        def resolve(name: str) -> str | None:
            return f"/trusted/bin/{name}"

        def run(argv, **kwargs):
            commands.append(tuple(argv))
            return {
                "status": "AVAILABLE",
                "exit_code": 0,
                "version_output": "1.2.3",
            }

        with tempfile.TemporaryDirectory() as temporary:
            accessible = Path(temporary) / "capability"
            accessible.write_text("available\n", encoding="utf-8")
            report = run_profiling_doctor(
                tool_probes=(ToolProbe("trace", ("mcTracer",)),),
                library_paths=(accessible,),
                device_paths=(accessible,),
                executable_resolver=resolve,
                command_runner=run,
            )

        self.assertEqual(report["status"], "READY")
        self.assertTrue(report["advisory_only"])
        self.assertEqual(report["promotion_effect"], "none")
        self.assertEqual(commands, [("/trusted/bin/mcTracer", "--version")])

    def test_default_metax_trace_probe_uses_help_and_can_be_ready(self) -> None:
        commands: list[tuple[str, ...]] = []

        def resolve(name: str) -> str | None:
            if name in {"mx-smi", "mcTracer"}:
                return f"/opt/maca-3.2.1/bin/{name}"
            return None

        def run(argv, **kwargs):
            commands.append(tuple(argv))
            if Path(argv[0]).name == "mcTracer":
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    stdout=(
                        b"mcTracer Help\n"
                        b"Version 3.2.1.10-df74b02\n"
                        b"Usage: mcTracer [options] command\n"
                    ),
                    stderr=b"",
                )
            return subprocess.CompletedProcess(
                argv, 0, stdout=b"mx-smi 1.0\n", stderr=b""
            )

        with tempfile.TemporaryDirectory() as temporary:
            accessible = Path(temporary) / "capability"
            accessible.write_text("available\n", encoding="utf-8")
            with mock.patch.object(profiling.subprocess, "run", side_effect=run):
                report = run_profiling_doctor(
                    tool_probes=DEFAULT_TOOL_PROBES,
                    library_paths=(accessible,),
                    device_paths=(accessible,),
                    executable_resolver=resolve,
                )

        self.assertEqual(report["status"], "READY")
        self.assertEqual(
            commands,
            [
                ("/opt/maca-3.2.1/bin/mx-smi", "--version"),
                ("/opt/maca-3.2.1/bin/mcTracer", "--help"),
            ],
        )
        trace = next(
            item
            for item in report["tools"]
            if item["tool_id"] == "metax-trace-collector"
        )
        self.assertEqual(trace["status"], "AVAILABLE")

    def test_mctracer_output_contract_rejects_spoofs_and_drift(self) -> None:
        trace = next(
            item
            for item in DEFAULT_TOOL_PROBES
            if item.tool_id == "metax-trace-collector"
        )
        self.assertEqual(trace.version_arguments, ("--help",))
        self.assertEqual(trace.accepted_exit_codes, (1,))
        valid = (
            b"mcTracer Help\n"
            b"Version 3.2.1.10-df74b02\n"
            b"Usage: mcTracer [options] command\n"
        )
        cases = (
            (1, b"arbitrary exit-one text\n", "signature mismatch"),
            (
                1,
                valid.replace(b"3.2.1.10-df74b02", b"3.2.1.11-drifted"),
                "signature mismatch",
            ),
            (0, valid, "unaccepted status"),
            (2, valid, "unaccepted status"),
            (1, valid + b"execvpe: No such file or directory\n", "failure marker"),
        )
        for exit_code, output, reason in cases:
            with self.subTest(exit_code=exit_code, reason=reason), mock.patch.object(
                profiling.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    ("/opt/maca-3.2.1/bin/mcTracer", "--help"),
                    exit_code,
                    stdout=output,
                    stderr=b"",
                ),
            ):
                result = profiling._bounded_version_command(
                    ("/opt/maca-3.2.1/bin/mcTracer", "--help"),
                    timeout_sec=1,
                    output_limit_bytes=4096,
                    accepted_exit_codes=trace.accepted_exit_codes,
                    required_output_patterns=trace.required_output_patterns,
                    forbidden_output_markers=trace.forbidden_output_markers,
                )
            self.assertEqual(result["status"], "UNAVAILABLE")
            self.assertIn(reason, result["reason"])

    def test_missing_capabilities_have_explicit_unavailable_reasons(self) -> None:
        report = run_profiling_doctor(
            tool_probes=(ToolProbe("trace", ("mcTracer",)),),
            library_paths=(self._missing_path(),),
            device_paths=(self._missing_path(),),
            executable_resolver=lambda name: None,
        )

        self.assertEqual(report["status"], "UNAVAILABLE")
        self.assertEqual(report["tools"][0]["status"], "UNAVAILABLE")
        self.assertIn("reason", report["support_library"])
        self.assertIn("reason", report["devices"])
        self.assertEqual(len(report["reasons"]), 3)

    def test_probe_recipe_rejects_paths_as_executable_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "bare trusted names"):
            ToolProbe("trace", ("/tmp/agent-tool",))

    def test_probe_and_recipe_value_objects_reject_untrusted_shapes(self) -> None:
        for call, message in (
            (lambda: ToolProbe("bad id", ("tool",)), "tool_id"),
            (lambda: ToolProbe("trace", ()), "must not be empty"),
            (lambda: ToolProbe("trace", ("tool",), ("",)), "arguments"),
            (
                lambda: ToolProbe(
                    "trace", ("tool",), accepted_exit_codes=()
                ),
                "accepted_exit_codes",
            ),
            (
                lambda: ToolProbe(
                    "trace", ("tool",), accepted_exit_codes=(True,)
                ),
                "accepted_exit_codes",
            ),
            (
                lambda: ToolProbe(
                    "trace", ("tool",), required_output_patterns=("[",)
                ),
                "output pattern",
            ),
            (lambda: ProfileMetric("Bad", "integer", "count"), "metric_id"),
            (lambda: ProfileMetric("count", "text", "count"), "value_kind"),
            (lambda: ProfileMetric("count", "integer", ""), "unit"),
            (
                lambda: ProfileRecipe(
                    "Bad", "case", 1, False, (ProfileMetric("a", "integer", "u"),)
                ),
                "recipe_id",
            ),
            (
                lambda: ProfileRecipe(
                    "recipe", "Bad-case", 1, False, (ProfileMetric("a", "integer", "u"),)
                ),
                "case_id",
            ),
            (
                lambda: ProfileRecipe(
                    "recipe", "case", 3, False, (ProfileMetric("a", "integer", "u"),)
                ),
                "minimum correctness",
            ),
            (lambda: ProfileRecipe("recipe", "case", 1, False, ()), "metrics"),
            (
                lambda: ProfileRecipe(
                    "recipe",
                    "case",
                    1,
                    False,
                    (
                        ProfileMetric("a", "integer", "u"),
                        ProfileMetric("a", "integer", "u"),
                    ),
                ),
                "metrics",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    call()

    def test_doctor_rejects_nonpositive_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "timeout"):
            run_profiling_doctor(timeout_sec=0)
        with self.assertRaisesRegex(ValueError, "output_limit"):
            run_profiling_doctor(output_limit_bytes=0)

    def test_bounded_version_command_handles_all_terminal_outcomes(self) -> None:
        completed = subprocess.CompletedProcess(
            ("/trusted/tool", "--version"), 0, stdout=b"v1", stderr=b""
        )
        with mock.patch.object(profiling.subprocess, "run", return_value=completed):
            result = profiling._bounded_version_command(
                ("/trusted/tool", "--version"), timeout_sec=1, output_limit_bytes=16
            )
        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(result["version_output"], "v1")

        failed = subprocess.CompletedProcess(
            ("/trusted/tool", "--version"), 7, stdout=b"", stderr=b"failed"
        )
        with mock.patch.object(profiling.subprocess, "run", return_value=failed):
            result = profiling._bounded_version_command(
                ("/trusted/tool", "--version"), timeout_sec=1, output_limit_bytes=16
            )
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertIn("unaccepted", result["reason"])

        noisy = subprocess.CompletedProcess(
            ("/trusted/tool", "--version"), 0, stdout=b"too much", stderr=b"noise"
        )
        with mock.patch.object(profiling.subprocess, "run", return_value=noisy):
            result = profiling._bounded_version_command(
                ("/trusted/tool", "--version"), timeout_sec=1, output_limit_bytes=4
            )
        self.assertIn("output limit", result["reason"])

        for error, reason in (
            (subprocess.TimeoutExpired(("tool",), 1), "timed out"),
            (OSError("unavailable"), "could not start"),
        ):
            with self.subTest(reason=reason), mock.patch.object(
                profiling.subprocess, "run", side_effect=error
            ):
                result = profiling._bounded_version_command(
                    ("/trusted/tool", "--version"),
                    timeout_sec=1,
                    output_limit_bytes=16,
                )
                self.assertIn(reason, result["reason"])

    def test_strict_json_rejects_ambiguous_and_noncanonical_inputs(self) -> None:
        self.assertEqual(
            profiling._strict_json_object(b'{"a":1}', field="fixture"),
            {"a": 1},
        )
        for payload, message in (
            (b'{"a":1,"a":2}', "duplicate"),
            (b'{"a":NaN}', "non-finite"),
            (b'not-json', "strict UTF-8 JSON"),
            (b'\xff', "strict UTF-8 JSON"),
            (b'[]', "JSON object"),
        ):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, message):
                    profiling._strict_json_object(payload, field="fixture")

    def test_bounded_regular_file_rejects_missing_nonregular_and_growth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"abc")
            self.assertEqual(
                profiling._read_regular_file(source, limit=3, field="fixture"),
                b"abc",
            )
            with self.assertRaisesRegex(ValueError, "positive"):
                profiling._read_regular_file(source, limit=0, field="fixture")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                profiling._read_regular_file(
                    root / "missing", limit=10, field="fixture"
                )
            with self.assertRaisesRegex(ValueError, "regular"):
                profiling._read_regular_file(root, limit=10, field="fixture")
            with self.assertRaisesRegex(ValueError, "size limit"):
                profiling._read_regular_file(source, limit=2, field="fixture")

            one = root / "one.bin"
            one.write_bytes(b"x")
            with mock.patch.object(profiling.os, "read", return_value=b"xx"):
                with self.assertRaisesRegex(ValueError, "size limit"):
                    profiling._read_regular_file(one, limit=1, field="fixture")

            real_fstat = os.fstat
            calls = 0

            def changed_size(descriptor):
                nonlocal calls
                calls += 1
                value = real_fstat(descriptor)
                if calls == 1:
                    return value
                fields = list(value)
                fields[6] = value.st_size + 1
                return os.stat_result(fields)

            with mock.patch.object(profiling.os, "fstat", side_effect=changed_size):
                with self.assertRaisesRegex(ValueError, "changed"):
                    profiling._read_regular_file(source, limit=10, field="fixture")

    def test_cas_is_idempotent_and_rejects_collision_symlink_and_failed_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "objects"
            object_id = profiling._store_cas_object(
                root, b"evidence", field="fixture"
            )
            self.assertEqual(
                profiling._store_cas_object(root, b"evidence", field="fixture"),
                object_id,
            )
            digest = object_id.removeprefix("sha256:")
            path = root / digest[:2] / digest[2:]
            path.write_bytes(b"collisio")
            with self.assertRaisesRegex(ValueError, "collision"):
                profiling._store_cas_object(root, b"evidence", field="fixture")

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            target = base / "target"
            target.mkdir()
            linked = base / "linked"
            linked.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                profiling._store_cas_object(linked, b"evidence", field="fixture")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "objects"
            with mock.patch.object(
                profiling.os, "replace", side_effect=OSError("publish failed")
            ):
                with self.assertRaises(OSError):
                    profiling._store_cas_object(root, b"evidence", field="fixture")
            self.assertEqual(list(root.rglob("*.tmp")), [])

    @staticmethod
    def _missing_path():
        from pathlib import Path

        return Path(__file__).resolve().with_name("definitely-missing-device")


class _FakeGate:
    def __init__(
        self,
        *,
        allowed: bool = True,
        change_after: int | None = None,
        on_status=None,
    ):
        self.allowed = allowed
        self.change_after = change_after
        self.on_status = on_status
        self.status_calls = 0

    def status(self):
        self.status_calls += 1
        if self.on_status is not None:
            self.on_status(self.status_calls)
        changed = (
            self.change_after is not None
            and self.status_calls > self.change_after
        )
        digest = "sha256:" + ("b" if changed else "a") * 64
        return {
            "current_observation_status": "AVAILABLE",
            "profiling_allowed": self.allowed,
            "active_invariant_matches": True,
            "current_invariant_snapshot_digest": digest,
        }

    def profiling_allowed(self):
        return self.allowed


class _FakeProfileRunner:
    def __init__(
        self,
        *,
        metrics=None,
        timed_out=False,
        output_limited=False,
        result_overrides=None,
        returncode=0,
        changed_argv=False,
        start_error=False,
        raw_trace=b"trusted raw trace\x00\x01",
        stderr="ignored profiler stderr",
        on_run=None,
        write_outputs=True,
    ):
        self.metrics = {} if metrics is None else metrics
        self.timed_out = timed_out
        self.output_limited = output_limited
        self.result_overrides = (
            {} if result_overrides is None else result_overrides
        )
        self.returncode = returncode
        self.changed_argv = changed_argv
        self.start_error = start_error
        self.raw_trace = raw_trace
        self.stderr = stderr
        self.on_run = on_run
        self.write_outputs = write_outputs
        self.calls = []

    def run(self, argv, **kwargs):
        argv = tuple(argv)
        self.calls.append((argv, dict(kwargs)))
        if self.on_run is not None:
            self.on_run(argv)
        if self.start_error:
            raise OSError("fixture start failure")
        mounts = [
            argv[index + 1]
            for index, item in enumerate(argv[:-1])
            if item == "--mount"
        ]
        output_mount = next(item for item in mounts if "dst=/output" in item)
        source = output_mount.split("src=", 1)[1].split(",dst=", 1)[0]
        output = Path(source)
        if self.write_outputs and not self.timed_out and not self.output_limited:
            recipe = argv[argv.index("--recipe") + 1]
            case_id = argv[argv.index("--case-id") + 1]
            uid = argv[argv.index("--experiment-uid") + 1]
            complete_metrics = {
                metric.metric_id: {
                    "status": "UNAVAILABLE",
                    "reason_code": "COUNTER_NOT_EXPOSED",
                }
                for metric in BUILTIN_PROFILE_RECIPES[recipe].metrics
            }
            complete_metrics.update(self.metrics)
            result = {
                "schema_version": profiling.PROFILE_COLLECTION_API_VERSION,
                "worker_revision": profiling.PROFILER_WORKER_REVISION,
                "profiler_build_profile_digest": (
                    profiling.PROFILER_BUILD_PROFILE_DIGEST
                ),
                "recipe_id": recipe,
                "case_id": case_id,
                "experiment_uid": uid,
                "namespace_id": argv[argv.index("--namespace-id") + 1],
                "condition_digest": argv[
                    argv.index("--condition-digest") + 1
                ],
                "artifact_id": argv[argv.index("--artifact-id") + 1],
                "candidate_sha256": argv[
                    argv.index("--candidate-sha256") + 1
                ],
                "environment_digest": argv[
                    argv.index("--environment-digest") + 1
                ],
                "soak_invariant_digest": argv[
                    argv.index("--soak-invariant-digest") + 1
                ],
                "campaign_id": argv[argv.index("--campaign-id") + 1],
                "run_id": argv[argv.index("--run-id") + 1],
                "budget_action_key": argv[
                    argv.index("--budget-action-key") + 1
                ],
                "metrics": complete_metrics,
            }
            if "--resource-id" in argv:
                result["resource_id"] = argv[argv.index("--resource-id") + 1]
                result["fencing_epoch"] = int(
                    argv[argv.index("--fencing-epoch") + 1]
                )
            toolchain = {
                "schema_version": 1,
                "worker_revision": profiling.PROFILER_WORKER_REVISION,
                "profiler_build_profile_digest": (
                    profiling.PROFILER_BUILD_PROFILE_DIGEST
                ),
                "tools": {
                    "mctracer": {
                        "path": profiling.MCTRACER_PATH,
                        "sha256": profiling.MCTRACER_SHA256,
                        "version": profiling.MCTRACER_VERSION,
                    },
                    "libmcToolsExt_lite.so": {
                        "path": profiling.MCTOOLS_EXT_LITE_PATH,
                        "version": profiling.METAX_TOOLCHAIN_VERSION,
                        "sha256": profiling.MCTOOLS_EXT_LITE_SHA256,
                    },
                    "libmcToolsExt.so": {
                        "path": profiling.MCTOOLS_EXT_PATH,
                        "version": profiling.METAX_TOOLCHAIN_VERSION,
                        "sha256": profiling.MCTOOLS_EXT_SHA256,
                    },
                },
                "help_contract": {
                    "argv": [profiling.MCTRACER_PATH, "--help"],
                    "stdin": "DEVNULL",
                    "exit_code": 1,
                    "version": profiling.MCTRACER_VERSION,
                    "required_markers": ["Help Info", "Usage:"],
                },
            }
            toolchain["digest"] = profiling.canonical_sha256(toolchain)
            trace_descriptor = {
                "format": (
                    "deterministic-tar-v1"
                    if "--resource-id" in argv
                    else "canonical-json-manifest-v1"
                ),
                "sha256": "sha256:" + hashlib.sha256(self.raw_trace).hexdigest(),
                "byte_size": len(self.raw_trace),
                "file_count": 1,
            }
            if "--resource-id" in argv:
                trace_descriptor.update(
                    {
                        "source_bytes": len(self.raw_trace),
                        "mctracer_exit_code": 1,
                        "target_stdout_sha256": "sha256:" + "b" * 64,
                        "target_stderr_sha256": "sha256:" + "c" * 64,
                        "toolchain_digest": toolchain["digest"],
                    }
                )
            result["toolchain"] = toolchain
            result["trace_descriptor"] = trace_descriptor
            echo = {key: value for key, value in result.items() if key not in {
                "metrics", "toolchain", "trace_descriptor"
            }}
            outcome = {
                **echo,
                "status": "SUCCESS",
                "gpu_state": (
                    "COMPLETED" if "--resource-id" in argv else "NOT_STARTED"
                ),
                "completion_trusted": True,
                "reason_code": "EVIDENCE_COMMITTED",
            }
            result.update(self.result_overrides)
            (output / "result.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            (output / "raw.trace").write_bytes(self.raw_trace)
            (output / "outcome.json").write_text(
                json.dumps(outcome), encoding="utf-8"
            )
        return CommandResult(
            argv=(argv + ("changed",) if self.changed_argv else argv),
            returncode=self.returncode,
            stdout="ignored profiler stdout",
            stderr=self.stderr,
            timed_out=self.timed_out,
            output_limited=self.output_limited,
        )


_DEFAULT_SUBJECT_CAMPAIGN = object()


def _host_memory_preflight_fixture(
    *, total: int = 64 * 1024**3, available: int = 48 * 1024**3
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "HOST_MEMORY_PREFLIGHT_V1",
        "source": "/proc/meminfo",
        "observed_epoch_ms": 1_787_114_880_000,
        "mem_total_bytes": total,
        "mem_available_bytes": available,
        "required_total_bytes": 24 * 1024**3,
        "required_available_bytes": 24 * 1024**3,
    }


class BoundedProfilingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._active_profiler = mock.patch.object(
            profiling, "require_active_profiler", return_value=None
        )
        self._profile_euid = mock.patch.object(
            profiling, "_HOST_EFFECTIVE_UID", return_value=1000
        )
        self._profile_egid = mock.patch.object(
            profiling, "_HOST_EFFECTIVE_GID", return_value=1000
        )
        self._host_memory = mock.patch.object(
            profiling,
            "_profile_host_memory_preflight",
            side_effect=lambda: _host_memory_preflight_fixture(),
        )
        self._active_profiler.start()
        self._profile_euid.start()
        self._profile_egid.start()
        self.host_memory = self._host_memory.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.state = self.root / "state"
        self.controller = self.root / "controller"
        self.checkpoints = self.root / "checkpoints"
        self.repository = self.root / "repository"
        self.cache = self.root / "cache"
        for directory in (
            self.state,
            self.controller,
            self.checkpoints,
            self.repository,
            self.cache,
        ):
            directory.mkdir()
        docker = self.root / "docker"
        docker.write_text("fixture", encoding="utf-8")
        key = self.root / "deepseek.key"
        key.write_text("fixture", encoding="utf-8")
        self.config = ControllerConfig(
            repository_dir=self.repository,
            state_dir=self.state,
            controller_dir=self.controller,
            checkpoint_dir=self.checkpoints,
            docker_binary=docker,
            proposer_image="fixture/proposer@sha256:" + "1" * 64,
            evaluator_image="fixture/evaluator@sha256:" + "2" * 64,
            deepseek_key_file=key,
            gpu_devices=GPU1_DEVICES,
            evaluator_cache_dir=self.cache,
            expected_git_commit="3" * 40,
            expected_kernel_hash="4" * 64,
            acknowledge_gpu_passthrough_risk=True,
        )
        self.campaign_database = (
            self.root / "campaign" / "campaign.sqlite3"
        )
        self.campaign_id = "profile-campaign"
        self._create_campaign(self.campaign_id)
        self.environment = ExecutionEnvironmentDigest.resolved(
            evaluator_image_digest="sha256:" + "1" * 64,
            toolchain_digest="sha256:" + "2" * 64,
            framework_digest="sha256:" + "3" * 64,
            operator_abi_digest="sha256:" + "4" * 64,
            build_flags_digest="sha256:" + "5" * 64,
        )
        self.uid = "00000000-0000-4000-8000-000000000099"

    def test_private_bounded_ipc_contract_rejects_every_unsafe_variant(self):
        profiling._require_profile_ipc_contract()
        invalid = (
            ("PROFILE_IPC_MODE", "none"),
            ("PROFILE_IPC_MODE", "host"),
            ("PROFILE_IPC_MODE", "shareable"),
            ("PROFILE_HOST_IPC_ALLOWED", True),
            ("PROFILE_CONTAINER_IPC_SHARING_ALLOWED", True),
            ("PROFILE_SHARED_MEMORY_PATH", "/host/dev/shm"),
            ("PROFILE_SHARED_MEMORY_SIZE", "2g"),
            ("PROFILE_SHARED_MEMORY_SIZE_BYTES", 2 * 1024**3),
            ("PROFILE_SHARED_MEMORY_OWNER_UID", 1000),
            ("PROFILE_SHARED_MEMORY_OWNER_GID", 1000),
            ("PROFILE_SHARED_MEMORY_MODE", "0700"),
        )
        for field, value in invalid:
            with (
                self.subTest(field=field, value=value),
                mock.patch.object(profiling, field, value),
                self.assertRaisesRegex(ValueError, "private bounded"),
            ):
                profiling._require_profile_ipc_contract()

    def _create_campaign(
        self, campaign_id: str, *, wall_ms: int = 20_000_000, gpu_ms: int = 20_000_000
    ) -> str:
        with CampaignStore(self.campaign_database) as campaigns:
            campaigns.create_campaign(
                campaign_id=campaign_id,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                mode=CampaignMode.DISCOVERY.value,
                snapshot={"fixture": "bounded-profiling"},
                budget_limit=BudgetAmount(
                    wall_ms=wall_ms,
                    gpu_ms=gpu_ms,
                ),
                initial_artifact_id="source-sha256-v1:" + "d" * 64,
                initial_policy_snapshot={"fixture": "policy"},
            )
            campaigns.start_campaign(campaign_id)
        return campaign_id

    def tearDown(self) -> None:
        self.temporary.cleanup()
        self._host_memory.stop()
        self._profile_egid.stop()
        self._profile_euid.stop()
        self._active_profiler.stop()

    def _record(
        self,
        *,
        stage="smoke",
        suite="smoke",
        passed=True,
        campaign_id=_DEFAULT_SUBJECT_CAMPAIGN,
        experiment_uid=None,
        run_id=None,
        link_child=True,
    ):
        subject_campaign_id = (
            self.campaign_id
            if campaign_id is _DEFAULT_SUBJECT_CAMPAIGN
            else campaign_id
        )
        subject_uid = self.uid if experiment_uid is None else experiment_uid
        subject_run_id = "profile-fixture-run" if run_id is None else run_id
        source = "def kernel():\n    return 1\n"
        bundle = CandidateBundle.single_file(content=source)
        baseline = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=bundle.artifact_id,
            source="deployment",
            revision="profile-fixture",
            execution_environment=self.environment,
        )
        identity = ExperimentIdentity.create(
            experiment_uid=subject_uid,
            namespace=CURRENT_RESEARCH_NAMESPACE,
            mode="DISCOVERY",
            candidate_artifact_id=bundle.artifact_id,
            parent_artifact_id=bundle.artifact_id,
            baseline=baseline,
            execution_environment=self.environment,
            stage=stage,
            suite=suite,
            replicate_kind=("primary" if stage == "full_primary" else "validation"),
            campaign_id=subject_campaign_id,
            run_id=subject_run_id,
            iteration=1,
        )
        if subject_campaign_id is not None and link_child:
            with CampaignStore(self.campaign_database) as campaigns:
                child = campaigns.get_child_run_by_controller_run_id(
                    subject_run_id
                )
                if child is None:
                    campaigns.create_child_run(
                        subject_campaign_id,
                        proposer_profile={"id": "profile-fixture"},
                        controller_run_id=subject_run_id,
                    )
        with HistoryStore(
            self.state / "history.sqlite3", state_dir=self.state
        ) as history:
            history.ensure_namespace(
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
                CURRENT_RESEARCH_NAMESPACE.to_dict(),
            )
            history.store_candidate_bundle(bundle)
            history.record_experiment(
                candidate_source=source,
                backend="c500",
                suite=suite,
                status="SUCCESS",
                identity=identity,
                result={"status": "SUCCESS"},
                case_measurements=(
                    {
                        "name": {
                            "smoke": "smoke_gate_up",
                            "quick": "quick_decode_gate_up",
                            "full": "full_decode_gate_up",
                        }[suite],
                        "matched_ratio": 1.0 if passed else 0.0,
                        "passed": passed,
                    },
                ),
            )
        return identity

    def _record_for_campaign(
        self,
        campaign_id,
        index,
        *,
        stage="smoke",
        suite="smoke",
        passed=True,
    ):
        return self._record(
            stage=stage,
            suite=suite,
            passed=passed,
            campaign_id=campaign_id,
            experiment_uid=f"00000000-0000-4000-8000-{index:012d}",
            run_id=f"profile-fixture-run-{index}",
        )

    def _collect(
        self,
        runner,
        gate,
        *,
        recipe="metax-compile-metadata-v1",
        campaign_id=None,
        experiment_uid=None,
    ):
        return run_bounded_profile(
            self.config,
            campaign_database=self.campaign_database,
            campaign_id=self.campaign_id if campaign_id is None else campaign_id,
            gate_id="production",
            experiment_uid=(
                self.uid if experiment_uid is None else experiment_uid
            ),
            namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
            execution_environment_digest=self.environment.digest,
            recipe_id=recipe,
            runner=runner,
            _gate_factory=lambda store, gate_id, collector: gate,
        )

    def _campaign_rows(self, table, campaign_id=None):
        with closing(
            sqlite3.connect(self.campaign_database)
        ) as connection, connection:
            connection.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in connection.execute(
                    f"SELECT * FROM {table} WHERE campaign_id = ? ORDER BY rowid",
                    (self.campaign_id if campaign_id is None else campaign_id,),
                )
            ]

    def _canary_subject_fixture(self, canary_id):
        identity = self._record(stage="confirmation", suite="full")
        with closing(
            sqlite3.connect(self.state / "history.sqlite3")
        ) as connection, connection:
            row = connection.execute(
                """
                SELECT e.candidate_hash, a.object_path
                FROM experiments e JOIN candidate_artifacts a
                  ON a.artifact_id = e.artifact_id
                WHERE e.experiment_uid = ?
                """,
                (identity.experiment_uid,),
            ).fetchone()
        assert row is not None
        subject = profiling._ProfileSubject(
            experiment_uid=identity.experiment_uid,
            namespace_id=identity.namespace_id,
            condition_digest=identity.condition_digest,
            artifact_id=str(identity.candidate_artifact_id),
            execution_environment_digest=identity.execution_environment.digest,
            campaign_id=canary_id,
            run_id=canary_id + "-run",
            stage="confirmation",
            suite="full",
            replicate_kind="confirmation",
            candidate_object_path=self.state / row[1],
            candidate_content_sha256=row[0],
        )
        baseline_ref = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=identity.candidate_artifact_id,
            source="campaign",
            revision=canary_id + "-seed",
            execution_environment=self.environment,
        )
        binding = {
            "deployment_evidence_digest": "sha256:" + "d" * 64,
            "deployment_git_commit": "e" * 40,
            "deployment_candidate_hash": row[0],
            "current_confirmation_experiment_uid": identity.experiment_uid,
            "current_confirmation_identity": identity.to_dict(),
            "execution_environment": self.environment.to_dict(),
        }
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(self.campaign_database) + suffix).unlink()
            except FileNotFoundError:
                pass
        return subject, baseline_ref, binding

    def _create_a6_known_failure_canary(
        self,
        canary_id: str,
        *,
        activation_digest: str | None = None,
        omit_hardware_diagnostic: bool = False,
        settle_action: bool = True,
        release_lease: bool = True,
    ):
        with CampaignStore(self.campaign_database) as store:
            try:
                store.get_campaign(self.campaign_id)
            except KeyError:
                needs_subject_campaign = True
            else:
                needs_subject_campaign = False
        if needs_subject_campaign:
            self._create_campaign(self.campaign_id)
        subject, baseline_ref, binding = self._canary_subject_fixture(canary_id)
        snapshot = profiling._canary_snapshot(
            subject=subject,
            baseline_ref=baseline_ref,
            binding=binding,
            profiler_image=profiling._A6_KNOWN_FAILURE_IMAGE,
            profiler_build_profile_digest=(
                profiling._A6_KNOWN_FAILURE_BUILD_DIGEST
            ),
            profiler_activation_profile_digest=(
                activation_digest
                or profiling._A6_KNOWN_FAILURE_ACTIVATION_DIGEST
            ),
            worker_revision=profiling.PROFILER_WORKER_REVISION,
            recipes=tuple(profiling._A6_KNOWN_FAILURE_CLASSIFICATIONS),
        )
        action_key = "profile-image-canary-v1"
        with CampaignStore(self.campaign_database) as store:
            store.create_campaign(
                campaign_id=canary_id,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                mode="DISCOVERY",
                snapshot=snapshot,
                budget_limit=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
                initial_baseline_ref=baseline_ref,
                initial_policy_snapshot={"kind": "PROFILE_IMAGE_CANARY"},
                allow_staged_lineage=False,
            )
            store.start_campaign(canary_id)
            store.reserve_budget(
                canary_id,
                idempotency_key=action_key,
                action_kind="PROFILE_IMAGE_CANARY",
                amount=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
            )
            lease = store.acquire_resource(
                canary_id,
                resource_id="gpu1",
                ttl_seconds=1_830,
            )
            for recipe_id, classification in (
                profiling._A6_KNOWN_FAILURE_CLASSIFICATIONS.items()
            ):
                recipe = BUILTIN_PROFILE_RECIPES[recipe_id]
                intent_material = {
                    "schema_version": 2,
                    "kind": "PROFILE_IMAGE_CANARY_ATTEMPT",
                    "campaign_id": canary_id,
                    "budget_action_key": action_key,
                    "recipe_id": recipe_id,
                    "case_id": recipe.case_id,
                    "requires_gpu": recipe.requires_gpu,
                    "resource_id": lease.resource_id if recipe.requires_gpu else None,
                    "fencing_epoch": (
                        lease.fencing_epoch if recipe.requires_gpu else None
                    ),
                    "profiler_image": profiling._A6_KNOWN_FAILURE_IMAGE,
                    "profiler_build_profile_digest": (
                        profiling._A6_KNOWN_FAILURE_BUILD_DIGEST
                    ),
                    "profiler_activation_profile_digest": snapshot[
                        "profiler_activation_profile_digest"
                    ],
                    "container_name": "kar-profile-a6-known-" + recipe_id,
                    "argv": [
                        "docker",
                        "run",
                        profiling._A6_KNOWN_FAILURE_IMAGE,
                        recipe_id,
                    ],
                    "argv_digest": profiling.canonical_sha256(
                        [
                            "docker",
                            "run",
                            profiling._A6_KNOWN_FAILURE_IMAGE,
                            recipe_id,
                        ]
                    ),
                    "timeout_seconds": 900.0,
                    "expected_worker_echo": {"recipe_id": recipe_id},
                    "host_memory_preflight": _host_memory_preflight_fixture(),
                }
                attempt_id = (
                    "profile-canary-attempt-"
                    + profiling.canonical_sha256(intent_material).split(":", 1)[1]
                )
                intent = {**intent_material, "attempt_id": attempt_id}
                store.record_profile_canary_attempt_intent(
                    canary_id,
                    intent=intent,
                )
                if omit_hardware_diagnostic and recipe.requires_gpu:
                    continue
                outcome_status = (
                    "SUCCESS" if classification == "SUCCESS" else "FAILED"
                )
                returncode = 0 if classification == "SUCCESS" else 1
                reason_code = (
                    "RECIPE_SUCCEEDED"
                    if classification == "SUCCESS"
                    else "RUNNER_KNOWN_NONZERO"
                )
                private = {
                    "schema_version": 1,
                    "kind": "PROFILE_IMAGE_CANARY_PRIVATE_DIAGNOSTIC",
                    "attempt_id": attempt_id,
                    "campaign_id": canary_id,
                    "budget_action_key": action_key,
                    "recipe_id": recipe_id,
                    "classification": classification,
                    "reason_code": reason_code,
                    "command": {
                        "returncode": returncode,
                        "timed_out": False,
                        "output_limited": False,
                    },
                    "outcome": {"status": outcome_status},
                    "file_inventory": {
                        "entries": [
                            {"path": f"fixture-{index}"}
                            for index in range(6)
                        ]
                    },
                }
                private_bytes = profiling.canonical_json_text(private).encode(
                    "utf-8"
                )
                diagnostic_object_id = profiling._store_cas_object(
                    self.controller / "objects" / "sha256",
                    private_bytes,
                    field="fixture private canary diagnostic",
                )
                diagnostic = {
                    "schema_version": 1,
                    "kind": "PROFILE_IMAGE_CANARY_DIAGNOSTIC",
                    "attempt_id": attempt_id,
                    "campaign_id": canary_id,
                    "budget_action_key": action_key,
                    "recipe_id": recipe_id,
                    "diagnostic_object_id": diagnostic_object_id,
                    "diagnostic_object_bytes": len(private_bytes),
                    "classification": classification,
                    "reason_code": reason_code,
                    "returncode": returncode,
                    "timed_out": False,
                    "output_limited": False,
                    "outcome_status": outcome_status,
                    "inventory_entries": 6,
                }
                store.record_profile_canary_attempt_diagnostic(
                    canary_id,
                    diagnostic=diagnostic,
                )
            if settle_action:
                store.settle_budget(
                    canary_id,
                    idempotency_key=action_key,
                    actual=BudgetAmount(wall_ms=20_000, gpu_ms=8_000),
                )
            if release_lease:
                store.release_resource(
                    lease,
                    reason="known profile image canary failure",
                )
            store.pause_campaign(
                canary_id,
                status="PAUSED_OPERATOR",
                reason="known profile image canary failure",
            )
        return subject, baseline_ref, binding, snapshot

    def test_compile_recipe_is_fixed_advisory_and_stores_bounded_evidence(self):
        self._record()
        history_before = hashlib.sha256(
            (self.state / "history.sqlite3").read_bytes()
        ).hexdigest()
        runner = _FakeProfileRunner(
            metrics={
                "compiler_version_digest": {
                    "status": "AVAILABLE",
                    "value": "sha256:" + "9" * 64,
                },
                "shared_memory_bytes": {
                    "status": "UNAVAILABLE",
                    "reason_code": "COUNTER_NOT_EXPOSED",
                },
            }
        )
        report = self._collect(runner, _FakeGate())

        self.assertEqual(report["status"], "SUCCESS")
        self.assertEqual(report["schema_version"], 2)
        self.assertTrue(report["advisory_only"])
        self.assertEqual(report["promotion_effect"], "none")
        self.assertEqual(report["baseline_effect"], "none")
        self.assertEqual(report["profiler_image"], PROFILER_IMAGE)
        self.assertEqual(report["case_id"], "smoke_gate_up")
        self.assertEqual(report["campaign_id"], self.campaign_id)
        self.assertEqual(report["run_id"], "profile-fixture-run")
        self.assertEqual(report["subject"]["campaign_id"], self.campaign_id)
        self.assertEqual(report["subject"]["run_id"], "profile-fixture-run")
        self.assertIsNone(report["resource_lease"])
        self.assertEqual(
            report["metrics"]["register_count"]["status"], "UNAVAILABLE"
        )
        self.assertIn("reason", report["metrics"]["register_count"])
        self.assertNotIn("value", report["metrics"]["register_count"])
        self.assertNotIn("argv", report)
        self.assertNotIn("stderr", report)

        argv, kwargs = runner.calls[0]
        self.assertIn("--pull=never", argv)
        self.assertIn("--network=none", argv)
        self.assertIn("--read-only", argv)
        self.assertEqual(
            [value for value in argv if value.startswith("--ipc=")],
            ["--ipc=private"],
        )
        self.assertNotIn("--ipc=none", argv)
        self.assertNotIn("--ipc=host", argv)
        self.assertNotIn("--ipc=shareable", argv)
        self.assertEqual(argv.count("--shm-size"), 1)
        self.assertEqual(argv[argv.index("--shm-size") + 1], "1g")
        self.assertIn("no-new-privileges", argv)
        self.assertEqual(
            argv[argv.index("--user") + 1],
            f"{profiling.PROFILER_IMAGE_UID}:{profiling.PROFILER_IMAGE_GID}",
        )
        tmpfs_mounts = [
            argv[index + 1]
            for index, item in enumerate(argv[:-1])
            if item == "--tmpfs"
        ]
        self.assertEqual(
            tmpfs_mounts,
            [
                profiling.PROFILE_DOCKER_TMPFS,
                profiling.PROFILE_TRITON_CACHE_TMPFS,
            ],
        )
        self.assertIn(",noexec,", profiling.PROFILE_DOCKER_TMPFS)
        self.assertIn(",exec,", profiling.PROFILE_TRITON_CACHE_TMPFS)
        self.assertTrue(
            profiling.PROFILE_TRITON_CACHE_TMPFS.startswith(
                "/tmp/triton-cache:"
            )
        )
        self.assertIn("mode=700", profiling.PROFILE_DOCKER_TMPFS)
        self.assertIn("uid=1000", profiling.PROFILE_DOCKER_TMPFS)
        self.assertIn("gid=1000", profiling.PROFILE_DOCKER_TMPFS)
        self.assertNotIn("--device", argv)
        self.assertEqual(kwargs["timeout_sec"], PROFILE_TIMEOUT_SECONDS)
        self.assertEqual(kwargs["max_output_bytes"], PROFILE_OUTPUT_LIMIT_BYTES)
        self.assertIn("--max-raw-trace-bytes", argv)
        self.assertEqual(argv[argv.index("--memory") + 1], "24g")
        self.assertEqual(argv.count("--memory"), 1)
        self.assertNotIn("--memory-swap", argv)
        image_index = argv.index(PROFILER_IMAGE)
        self.assertGreater(image_index, argv.index("--entrypoint"))
        candidate_mount = next(
            argv[index + 1]
            for index, item in enumerate(argv[:-1])
            if item == "--mount" and "dst=/input/candidate.cas" in argv[index + 1]
        )
        self.assertTrue(candidate_mount.endswith(",readonly"))
        self.assertEqual(argv[argv.index("--campaign-id") + 1], self.campaign_id)
        self.assertEqual(
            argv[argv.index("--run-id") + 1], "profile-fixture-run"
        )
        self.assertEqual(
            argv[argv.index("--budget-action-key") + 1],
            report["budget_action_key"],
        )

        trace_digest = report["raw_trace"]["object_id"].removeprefix("sha256:")
        trace_path = (
            self.controller / "objects" / "sha256" / trace_digest[:2] / trace_digest[2:]
        )
        self.assertEqual(trace_path.read_bytes(), b"trusted raw trace\x00\x01")
        evidence_digest = report["evidence_object_id"].removeprefix("sha256:")
        evidence_path = (
            self.state / "objects" / "sha256" / evidence_digest[:2] / evidence_digest[2:]
        )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        self.assertTrue(evidence["advisory_only"])
        self.assertEqual(evidence["schema_version"], 2)
        self.assertEqual(evidence["kind"], "bounded_profiling_evidence_v2")
        self.assertEqual(
            evidence["profiler_build_profile_digest"],
            profiling.PROFILER_BUILD_PROFILE_DIGEST,
        )
        self.assertEqual(
            evidence["profiler_activation_profile_digest"],
            profiling.PROFILER_ACTIVATION_PROFILE_DIGEST,
        )
        self.assertIn("toolchain", evidence)
        self.assertIn("trace_descriptor", evidence)
        self.assertEqual(evidence["campaign_id"], self.campaign_id)
        self.assertEqual(evidence["run_id"], "profile-fixture-run")
        self.assertEqual(evidence["raw_trace"]["object_id"], report["raw_trace"]["object_id"])
        self.assertEqual(
            hashlib.sha256((self.state / "history.sqlite3").read_bytes()).hexdigest(),
            history_before,
        )
        actions = self._campaign_rows("budget_actions")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["status"], "SETTLED")
        self.assertEqual(actions[0]["reserved_wall_ms"], 900_000)
        self.assertEqual(actions[0]["reserved_gpu_ms"], 0)
        self.assertLessEqual(actions[0]["actual_wall_ms"], 900_000)
        self.assertEqual(actions[0]["actual_gpu_ms"], 0)
        replay = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "terminal budget intent"):
            self._collect(replay, _FakeGate())
        self.assertEqual(replay.calls, [])

    def test_runtime_user_drift_fails_before_campaign_or_docker(self):
        runner = _FakeProfileRunner()
        for field in ("container_uid", "container_gid"):
            with self.subTest(field=field):
                config = replace(self.config, **{field: 1001})
                with self.assertRaisesRegex(ValueError, "exact runtime user 1000:1000"):
                    run_bounded_profile(
                        config,
                        campaign_database=self.campaign_database,
                        campaign_id=self.campaign_id,
                        gate_id="production",
                        experiment_uid=self.uid,
                        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                        execution_environment_digest=self.environment.digest,
                        recipe_id="metax-compile-metadata-v1",
                        runner=runner,
                        _gate_factory=lambda store, gate_id, collector: _FakeGate(),
                    )
                with self.assertRaisesRegex(ValueError, "exact runtime user 1000:1000"):
                    profiling.run_profile_image_doctor(
                        config,
                        campaign_database=self.campaign_database,
                        campaign_id="runtime-user-canary",
                        runner=runner,
                    )
        with (
            mock.patch.object(profiling, "_HOST_EFFECTIVE_UID", return_value=0),
            self.assertRaisesRegex(
                ValueError, "trusted host process user 1000:1000"
            ),
        ):
            run_bounded_profile(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=self.campaign_id,
                gate_id="production",
                experiment_uid=self.uid,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                execution_environment_digest=self.environment.digest,
                recipe_id="metax-compile-metadata-v1",
                runner=runner,
                _gate_factory=lambda store, gate_id, collector: _FakeGate(),
            )
        self.assertEqual(runner.calls, [])
        self.assertEqual(self._campaign_rows("budget_actions"), [])

    def test_hardware_recipe_requires_quick_and_mounts_only_exact_devices(self):
        self._record(stage="quick", suite="quick")
        runner = _FakeProfileRunner(
            metrics={
                "gpu_active_percent": {"status": "AVAILABLE", "value": 0.0}
            }
        )
        report = self._collect(
            runner,
            _FakeGate(),
            recipe="metax-hardware-counters-v1",
        )
        self.assertEqual(report["metrics"]["gpu_active_percent"]["value"], 0.0)
        self.assertEqual(report["resource_lease"]["resource_id"], "gpu1")
        argv = runner.calls[0][0]
        mounted = [
            argv[index + 1]
            for index, item in enumerate(argv[:-1])
            if item == "--device"
        ]
        self.assertEqual(
            mounted,
            [f"{device}:{device}:rwm" for device in GPU1_DEVICES],
        )
        self.assertEqual(
            [value for value in argv if value.startswith("--ipc=")],
            ["--ipc=private"],
        )
        self.assertEqual(argv[argv.index("--shm-size") + 1], "1g")
        self.assertEqual(
            [
                argv[index + 1]
                for index, item in enumerate(argv[:-1])
                if item == "--tmpfs"
            ],
            [
                profiling.PROFILE_DOCKER_TMPFS,
                profiling.PROFILE_TRITON_CACHE_TMPFS,
            ],
        )
        self.assertEqual(argv[argv.index("--resource-id") + 1], "gpu1")
        self.assertEqual(
            int(argv[argv.index("--fencing-epoch") + 1]),
            report["resource_lease"]["fencing_epoch"],
        )
        actions = self._campaign_rows("budget_actions")
        self.assertEqual(actions[0]["status"], "SETTLED")
        self.assertEqual(actions[0]["reserved_gpu_ms"], 900_000)
        leases = self._campaign_rows("resource_leases")
        self.assertEqual(leases[-1]["status"], "RELEASED")
        with CampaignStore(self.campaign_database) as campaigns:
            self.assertIsNone(campaigns.get_active_resource_lease("gpu1"))

    def test_hardware_recipe_rejects_smoke_before_runner(self):
        self._record(stage="smoke", suite="smoke")
        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "at least correct quick"):
            self._collect(
                runner,
                _FakeGate(),
                recipe="metax-hardware-counters-v1",
            )
        self.assertEqual(runner.calls, [])

    def test_preexisting_active_gpu1_lease_blocks_runner_and_settles_intent(self):
        self._record(stage="quick", suite="quick")
        owner = self._create_campaign("supervisor-owner")
        with CampaignStore(self.campaign_database) as campaigns:
            existing = campaigns.acquire_resource(
                owner,
                resource_id="gpu1",
                ttl_seconds=PROFILE_TIMEOUT_SECONDS,
            )
        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "active unexpired lease"):
            self._collect(
                runner,
                _FakeGate(),
                recipe="metax-hardware-counters-v1",
            )
        self.assertEqual(runner.calls, [])
        actions = self._campaign_rows("budget_actions")
        self.assertEqual(actions[0]["status"], "SETTLED")
        with CampaignStore(self.campaign_database) as campaigns:
            current = campaigns.get_active_resource_lease("gpu1")
            self.assertEqual(current, existing)
            self.assertEqual(
                campaigns.get_campaign(self.campaign_id)["status"], "RUNNING"
            )
            campaigns.release_resource(existing)

    def test_hardware_timeout_quarantines_pauses_and_same_action_never_replays(self):
        self._record(stage="quick", suite="quick")
        runner = _FakeProfileRunner(timed_out=True)
        with self.assertRaisesRegex(ValueError, "timed out"):
            self._collect(
                runner,
                _FakeGate(),
                recipe="metax-hardware-counters-v1",
            )
        self.assertEqual(len(runner.calls), 1)
        actions = self._campaign_rows("budget_actions")
        self.assertEqual(actions[0]["status"], "RESERVED")
        leases = self._campaign_rows("resource_leases")
        self.assertEqual(leases[-1]["status"], "QUARANTINED")
        with CampaignStore(self.campaign_database) as campaigns:
            self.assertEqual(
                campaigns.get_campaign(self.campaign_id)["status"],
                "PAUSED_UNKNOWN_OUTCOME",
            )
            with self.assertRaisesRegex(ValueError, "doctor"):
                campaigns.resume_campaign(self.campaign_id)

        retry = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "replay is forbidden"):
            self._collect(
                retry,
                _FakeGate(),
                recipe="metax-hardware-counters-v1",
            )
        self.assertEqual(retry.calls, [])
        self.assertEqual(len(self._campaign_rows("budget_actions")), 1)

    def test_hardware_interruption_is_unknown_and_quarantined(self):
        self._record(stage="quick", suite="quick")

        def interrupt(_argv):
            raise KeyboardInterrupt()

        runner = _FakeProfileRunner(on_run=interrupt)
        with self.assertRaises(KeyboardInterrupt):
            self._collect(
                runner,
                _FakeGate(),
                recipe="metax-hardware-counters-v1",
            )
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(
            self._campaign_rows("budget_actions")[0]["status"], "RESERVED"
        )
        self.assertEqual(
            self._campaign_rows("resource_leases")[-1]["status"],
            "QUARANTINED",
        )

    def test_crash_left_reserved_action_is_reconciled_without_replay(self):
        self._record(stage="quick", suite="quick")
        recipe = BUILTIN_PROFILE_RECIPES["metax-hardware-counters-v1"]
        subject = profiling._read_only_history_subject(
            state_dir=self.state,
            experiment_uid=self.uid,
            namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
            execution_environment_digest=self.environment.digest,
            recipe=recipe,
        )
        action_key = profiling._profile_action_key(
            campaign_id=self.campaign_id,
            gate_id="production",
            recipe=recipe,
            subject=subject,
        )
        with CampaignStore(self.campaign_database) as campaigns:
            campaigns.reserve_budget(
                self.campaign_id,
                idempotency_key=action_key,
                action_kind="PROFILE_HARDWARE_COUNTERS",
                amount=BudgetAmount(wall_ms=900_000, gpu_ms=900_000),
            )
            campaigns.acquire_resource(
                self.campaign_id,
                resource_id="gpu1",
                ttl_seconds=PROFILE_TIMEOUT_SECONDS,
            )

        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "replay is forbidden"):
            self._collect(
                runner,
                _FakeGate(),
                recipe=recipe.recipe_id,
            )
        self.assertEqual(runner.calls, [])
        self.assertEqual(
            self._campaign_rows("budget_actions")[0]["status"], "RESERVED"
        )
        self.assertEqual(
            self._campaign_rows("resource_leases")[-1]["status"],
            "QUARANTINED",
        )

    def test_known_nonfatal_hardware_failure_settles_and_releases(self):
        self._record(stage="quick", suite="quick")
        known = _FakeProfileRunner(returncode=9)
        with self.assertRaisesRegex(ValueError, "non-zero"):
            self._collect(
                known,
                _FakeGate(),
                recipe="metax-hardware-counters-v1",
            )
        self.assertEqual(
            self._campaign_rows("budget_actions")[0]["status"],
            "SETTLED",
        )
        self.assertEqual(
            self._campaign_rows("resource_leases")[-1]["status"],
            "RELEASED",
        )
        with CampaignStore(self.campaign_database) as campaigns:
            self.assertEqual(
                campaigns.get_campaign(self.campaign_id)["status"], "RUNNING"
            )

    def test_fatal_hardware_marker_quarantines_and_hard_pauses(self):
        self._record(stage="quick", suite="quick")
        fatal = _FakeProfileRunner(
            returncode=137,
            stderr="driver: ATU fault while collecting counters",
        )
        with self.assertRaisesRegex(ValueError, "fatal GPU marker"):
            self._collect(
                fatal,
                _FakeGate(),
                recipe="metax-hardware-counters-v1",
            )
        self.assertEqual(
            self._campaign_rows("budget_actions")[0]["status"],
            "RESERVED",
        )
        self.assertEqual(
            self._campaign_rows("resource_leases")[-1]["status"],
            "QUARANTINED",
        )
        with CampaignStore(self.campaign_database) as campaigns:
            self.assertEqual(
                campaigns.get_campaign(self.campaign_id)["status"],
                "PAUSED_HARD_FAILURE",
            )

    def test_completed_hardware_persistence_failure_is_known_and_released(self):
        self._record(stage="quick", suite="quick")
        runner = _FakeProfileRunner()
        with mock.patch.object(
            profiling,
            "_store_cas_object",
            side_effect=OSError("fixture CAS persistence failure"),
        ):
            with self.assertRaisesRegex(OSError, "CAS persistence"):
                self._collect(
                    runner,
                    _FakeGate(),
                    recipe="metax-hardware-counters-v1",
                )
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(
            self._campaign_rows("budget_actions")[0]["status"], "SETTLED"
        )
        self.assertEqual(
            self._campaign_rows("resource_leases")[-1]["status"], "RELEASED"
        )
        with CampaignStore(self.campaign_database) as campaigns:
            self.assertEqual(
                campaigns.get_campaign(self.campaign_id)["status"],
                "RUNNING",
            )

    def test_budget_lease_soak_and_host_lock_drift_never_reach_runner(self):
        recipe = "metax-hardware-counters-v1"

        too_small = self._create_campaign(
            "profile-budget-too-small", wall_ms=899_999, gpu_ms=899_999
        )
        too_small_identity = self._record_for_campaign(
            too_small, 101, stage="quick", suite="quick"
        )
        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "exceeds a frozen limit"):
            self._collect(
                runner,
                _FakeGate(),
                recipe=recipe,
                campaign_id=too_small,
                experiment_uid=too_small_identity.experiment_uid,
            )
        self.assertEqual(runner.calls, [])

        lease_drift = self._create_campaign("profile-lease-drift")
        lease_identity = self._record_for_campaign(
            lease_drift, 102, stage="quick", suite="quick"
        )
        runner = _FakeProfileRunner()
        with mock.patch.object(
            CampaignStore, "get_active_resource_lease", return_value=None
        ):
            with self.assertRaisesRegex(ValueError, "lease or fencing"):
                self._collect(
                    runner,
                    _FakeGate(),
                    recipe=recipe,
                    campaign_id=lease_drift,
                    experiment_uid=lease_identity.experiment_uid,
                )
        self.assertEqual(runner.calls, [])
        self.assertEqual(
            self._campaign_rows("budget_actions", lease_drift)[0]["status"],
            "SETTLED",
        )

        soak_drift = self._create_campaign("profile-soak-drift")
        soak_identity = self._record_for_campaign(
            soak_drift, 103, stage="quick", suite="quick"
        )
        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "invariant changed"):
            self._collect(
                runner,
                _FakeGate(change_after=2),
                recipe=recipe,
                campaign_id=soak_drift,
                experiment_uid=soak_identity.experiment_uid,
            )
        self.assertEqual(runner.calls, [])
        self.assertEqual(
            self._campaign_rows("budget_actions", soak_drift)[0]["status"],
            "SETTLED",
        )

        lock_drift = self._create_campaign("profile-lock-held")
        lock_identity = self._record_for_campaign(
            lock_drift, 104, stage="quick", suite="quick"
        )
        runner = _FakeProfileRunner()
        with profiling.gpu_lock(self.controller / "gpu1.lock"):
            with self.assertRaisesRegex(ValueError, "controller lock"):
                self._collect(
                    runner,
                    _FakeGate(),
                    recipe=recipe,
                    campaign_id=lock_drift,
                    experiment_uid=lock_identity.experiment_uid,
                )
        self.assertEqual(runner.calls, [])
        self.assertEqual(
            self._campaign_rows("budget_actions", lock_drift)[0]["status"],
            "SETTLED",
        )

    def test_reserved_budget_intent_drift_is_unknown_before_runner(self):
        self._record(stage="quick", suite="quick")

        def cancel_after_reservation(status_call):
            if status_call != 3:
                return
            with closing(
                sqlite3.connect(self.campaign_database)
            ) as connection, connection:
                connection.execute(
                    """
                    UPDATE budget_actions SET status = 'CANCELLED', settled_at = 'fixture'
                    WHERE campaign_id = ? AND status = 'RESERVED'
                    """,
                    (self.campaign_id,),
                )

        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "cleanup could not be persisted"):
            self._collect(
                runner,
                _FakeGate(on_status=cancel_after_reservation),
                recipe="metax-hardware-counters-v1",
            )
        self.assertEqual(runner.calls, [])
        self.assertEqual(
            self._campaign_rows("budget_actions")[0]["status"], "CANCELLED"
        )
        self.assertEqual(
            self._campaign_rows("resource_leases")[-1]["status"],
            "QUARANTINED",
        )
        with CampaignStore(self.campaign_database) as campaigns:
            self.assertEqual(
                campaigns.get_campaign(self.campaign_id)["status"],
                "PAUSED_UNKNOWN_OUTCOME",
            )

    def test_wrong_namespace_environment_or_failed_correctness_is_closed(self):
        self._record(passed=False)
        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "passing correctness"):
            self._collect(runner, _FakeGate())
        with self.assertRaisesRegex(ValueError, "namespace"):
            run_bounded_profile(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=self.campaign_id,
                gate_id="production",
                experiment_uid=self.uid,
                namespace_id="sha256:" + "f" * 64,
                execution_environment_digest=self.environment.digest,
                recipe_id="metax-compile-metadata-v1",
                runner=runner,
                _gate_factory=lambda store, gate_id, collector: _FakeGate(),
            )
        with self.assertRaisesRegex(ValueError, "environment assertion"):
            run_bounded_profile(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=self.campaign_id,
                gate_id="production",
                experiment_uid=self.uid,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                execution_environment_digest="sha256:" + "e" * 64,
                recipe_id="metax-compile-metadata-v1",
                runner=runner,
                _gate_factory=lambda store, gate_id, collector: _FakeGate(),
            )

    def test_missing_campaign_fails_before_budget_or_runner(self):
        missing_campaign = "missing-profile-campaign"
        identity = self._record(
            campaign_id=missing_campaign,
            experiment_uid="00000000-0000-4000-8000-000000000201",
            run_id="missing-profile-run",
            link_child=False,
        )
        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "Campaign does not exist"):
            self._collect(
                runner,
                _FakeGate(),
                campaign_id=missing_campaign,
                experiment_uid=identity.experiment_uid,
            )
        self.assertEqual(runner.calls, [])

    def test_subject_must_belong_to_selected_campaign_and_child_run(self):
        campaign_b = self._create_campaign("profile-campaign-b")
        campaign_a_identity = self._record()
        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "another Campaign"):
            self._collect(
                runner,
                _FakeGate(),
                campaign_id=campaign_b,
                experiment_uid=campaign_a_identity.experiment_uid,
            )

        ordinary_identity = self._record(
            campaign_id=None,
            experiment_uid="00000000-0000-4000-8000-000000000202",
            run_id="ordinary-profile-run",
            link_child=False,
        )
        with self.assertRaisesRegex(ValueError, "ordinary Run"):
            self._collect(
                runner,
                _FakeGate(),
                experiment_uid=ordinary_identity.experiment_uid,
            )

        unlinked_identity = self._record(
            campaign_id=self.campaign_id,
            experiment_uid="00000000-0000-4000-8000-000000000203",
            run_id="unlinked-profile-run",
            link_child=False,
        )
        with self.assertRaisesRegex(ValueError, "not a Campaign child run"):
            self._collect(
                runner,
                _FakeGate(),
                experiment_uid=unlinked_identity.experiment_uid,
            )
        self.assertEqual(runner.calls, [])
        self.assertEqual(self._campaign_rows("budget_actions"), [])

    def test_soak_denial_and_invariant_change_fail_closed(self):
        self._record()
        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "does not allow profiling"):
            self._collect(runner, _FakeGate(allowed=False))
        self.assertEqual(runner.calls, [])

        runner = _FakeProfileRunner()
        with self.assertRaisesRegex(ValueError, "invariant changed"):
            self._collect(runner, _FakeGate(change_after=2))
        self.assertEqual(runner.calls, [])
        self.assertFalse((self.controller / "objects").exists())

    def test_timeout_and_output_cap_are_terminal_without_evidence(self):
        for index, (runner, message) in enumerate((
            (_FakeProfileRunner(timed_out=True), "timed out"),
            (_FakeProfileRunner(output_limited=True), "output cap"),
        )):
            with self.subTest(message=message):
                campaign_id = self._create_campaign(f"timeout-{index}")
                identity = self._record_for_campaign(campaign_id, 210 + index)
                with self.assertRaisesRegex(ValueError, message):
                    self._collect(
                        runner,
                        _FakeGate(),
                        campaign_id=campaign_id,
                        experiment_uid=identity.experiment_uid,
                    )
                if index == 0:
                    replay = _FakeProfileRunner()
                    with self.assertRaisesRegex(ValueError, "replay is forbidden"):
                        self._collect(
                            replay,
                            _FakeGate(),
                            campaign_id=campaign_id,
                            experiment_uid=identity.experiment_uid,
                        )
                    self.assertEqual(replay.calls, [])
        self.assertFalse((self.controller / "objects").exists())

    def test_runner_start_argv_returncode_and_empty_trace_fail_closed(self):
        for index, (runner, message) in enumerate((
            (_FakeProfileRunner(start_error=True), "could not start"),
            (_FakeProfileRunner(changed_argv=True), "fixed argv"),
            (_FakeProfileRunner(returncode=9), "non-zero"),
            (_FakeProfileRunner(raw_trace=b""), "must not be empty"),
        )):
            with self.subTest(message=message):
                campaign_id = self._create_campaign(f"runner-failure-{index}")
                identity = self._record_for_campaign(campaign_id, 220 + index)
                with self.assertRaisesRegex(ValueError, message):
                    self._collect(
                        runner,
                        _FakeGate(),
                        campaign_id=campaign_id,
                        experiment_uid=identity.experiment_uid,
                    )
        self.assertFalse((self.controller / "objects").exists())

    def test_runner_identity_echo_and_metric_whitelist_fail_closed(self):
        self._record()
        tampered = _FakeProfileRunner(
            result_overrides={"namespace_id": "sha256:" + "f" * 64}
        )
        with self.assertRaisesRegex(ValueError, "trusted request"):
            self._collect(tampered, _FakeGate())
        unknown = _FakeProfileRunner(
            metrics={"agent_metric": {"status": "AVAILABLE", "value": 1}}
        )
        campaign_id = self._create_campaign("unknown-metric")
        identity = self._record_for_campaign(campaign_id, 230)
        with self.assertRaisesRegex(ValueError, "non-whitelisted"):
            self._collect(
                unknown,
                _FakeGate(),
                campaign_id=campaign_id,
                experiment_uid=identity.experiment_uid,
            )
        self.assertFalse((self.controller / "objects").exists())

    def test_runner_must_echo_subject_run_identity(self):
        self._record()
        runner = _FakeProfileRunner(result_overrides={"run_id": "other-run"})
        with self.assertRaisesRegex(ValueError, "trusted request"):
            self._collect(runner, _FakeGate())
        self.assertEqual(len(runner.calls), 1)

    def test_metric_schema_and_value_validation_is_fail_closed(self):
        compile_recipe = BUILTIN_PROFILE_RECIPES["metax-compile-metadata-v1"]
        hardware_recipe = BUILTIN_PROFILE_RECIPES["metax-hardware-counters-v1"]

        def complete(recipe, values):
            result = {
                metric.metric_id: {
                    "status": "UNAVAILABLE",
                    "reason_code": "COUNTER_NOT_EXPOSED",
                }
                for metric in recipe.metrics
            }
            result.update(values)
            return result

        with self.assertRaisesRegex(ValueError, "JSON object"):
            profiling._metric_summary(compile_recipe, {"metrics": []})

        invalid_compile_metrics = (
            ({"register_count": 1}, "fixed schema"),
            (
                {
                    "register_count": {
                        "status": "UNAVAILABLE",
                        "reason_code": "AGENT_REASON",
                    }
                },
                "unknown unavailable",
            ),
            (
                {"register_count": {"status": "OTHER", "value": 1}},
                "availability",
            ),
            (
                {"register_count": {"status": "AVAILABLE", "value": True}},
                "integer",
            ),
            (
                {"register_count": {"status": "AVAILABLE", "value": -1}},
                "integer",
            ),
            (
                {
                    "compiler_version_digest": {
                        "status": "AVAILABLE",
                        "value": "not-a-digest",
                    }
                },
                "digest",
            ),
        )
        for metrics, message in invalid_compile_metrics:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    profiling._metric_summary(
                        compile_recipe,
                        {"metrics": complete(compile_recipe, metrics)},
                    )

        for value in (True, "fast", math.nan, -0.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "numeric"):
                    profiling._metric_summary(
                        hardware_recipe,
                        {
                            "metrics": complete(hardware_recipe, {
                                "gpu_active_percent": {
                                    "status": "AVAILABLE",
                                    "value": value,
                                }
                            })
                        },
                    )

        summary = profiling._metric_summary(
            compile_recipe,
            {
                "metrics": complete(compile_recipe, {
                    "register_count": {"status": "AVAILABLE", "value": 64},
                    "compiler_version_digest": {
                        "status": "AVAILABLE",
                        "value": "sha256:" + "a" * 64,
                    },
                    "spill_bytes": {
                        "status": "UNAVAILABLE",
                        "reason_code": "NOT_SUPPORTED",
                    },
                })
            },
        )
        self.assertEqual(summary["register_count"]["value"], 64)
        self.assertEqual(summary["spill_bytes"]["status"], "UNAVAILABLE")

    def test_gate_authorization_rejects_missing_invalid_and_stale_digest(self):
        class Gate:
            def __init__(self, statuses, allowed=True):
                self.statuses = list(statuses)
                self.allowed = allowed

            def status(self):
                return self.statuses.pop(0)

            def profiling_allowed(self):
                return self.allowed

        base = {
            "current_observation_status": "AVAILABLE",
            "profiling_allowed": True,
            "active_invariant_matches": True,
            "current_invariant_snapshot_digest": "sha256:" + "a" * 64,
        }
        missing = dict(base)
        missing.pop("current_invariant_snapshot_digest")
        with self.assertRaisesRegex(ValueError, "no current invariant"):
            profiling._gate_authorization(Gate((missing, missing)))

        invalid = dict(base)
        invalid["current_invariant_snapshot_digest"] = "invalid"
        with self.assertRaisesRegex(ValueError, "sha256"):
            profiling._gate_authorization(Gate((invalid, invalid)))

        second_unavailable = dict(base)
        second_unavailable["current_observation_status"] = "UNAVAILABLE"
        with self.assertRaisesRegex(ValueError, "changed during authorization"):
            profiling._gate_authorization(Gate((base, second_unavailable)))

        with self.assertRaisesRegex(ValueError, "changed during profiling"):
            profiling._gate_authorization(
                Gate((base, base)),
                expected_invariant_digest="sha256:" + "b" * 64,
            )

    def test_reserved_intent_soak_recheck_exempts_only_its_single_budget_count(self):
        digest = "sha256:" + "a" * 64
        status = {
            "current_observation_status": "AVAILABLE",
            "profiling_allowed": False,
            "active_invariant_matches": True,
            "current_invariant_snapshot_digest": digest,
        }

        class Gate:
            def status(self):
                return status

            def profiling_allowed(self):
                return False

        class Store:
            def soak_profiling_allowed(self, gate_id, invariant_digest):
                return gate_id == "production" and invariant_digest == digest

        counts = {field: 0 for field in profiling.COUNT_FIELDS}
        counts["budget_leak_count"] = 1

        class Collector:
            def __init__(self, selected_counts):
                self.selected_counts = selected_counts

            def collect(self, **_kwargs):
                return SimpleNamespace(
                    status="AVAILABLE",
                    invariant_snapshot_digest=digest,
                    counts=self.selected_counts,
                )

        profiling._gate_authorization_with_reserved_profile_intent(
            Gate(),
            collector=Collector(counts),
            store=Store(),
            gate_id="production",
            expected_invariant_digest=digest,
        )
        extra_violation = dict(counts)
        extra_violation["lease_overlap_count"] = 1
        with self.assertRaisesRegex(ValueError, "violations beyond"):
            profiling._gate_authorization_with_reserved_profile_intent(
                Gate(),
                collector=Collector(extra_violation),
                store=Store(),
                gate_id="production",
                expected_invariant_digest=digest,
            )

    def test_noncanonical_campaign_database_and_unknown_recipe_fail_closed(self):
        self._record()
        runner = _FakeProfileRunner()
        other = self.root / "campaign.sqlite3"
        other.write_bytes(self.campaign_database.read_bytes())
        with self.assertRaisesRegex(ValueError, "conventional"):
            run_bounded_profile(
                self.config,
                campaign_database=other,
                campaign_id=self.campaign_id,
                gate_id="production",
                experiment_uid=self.uid,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                execution_environment_digest=self.environment.digest,
                recipe_id="metax-compile-metadata-v1",
                runner=runner,
                _gate_factory=lambda store, gate_id, collector: _FakeGate(),
            )
        with self.assertRaisesRegex(ValueError, "reviewed built-in"):
            run_bounded_profile(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=self.campaign_id,
                gate_id="production",
                experiment_uid=self.uid,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                execution_environment_digest=self.environment.digest,
                recipe_id="agent-supplied-recipe",
                runner=runner,
                _gate_factory=lambda store, gate_id, collector: _FakeGate(),
            )
        self.assertEqual(runner.calls, [])

    def test_cli_collect_has_only_frozen_recipe_selection(self):
        args = build_parser().parse_args(
            [
                "profile",
                "collect",
                "--config",
                "/tmp/config.json",
                "--database",
                "/tmp/campaign/campaign.sqlite3",
                "--campaign-id",
                self.campaign_id,
                "--gate-id",
                "production",
                "--experiment-uid",
                self.uid,
                "--namespace-id",
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
                "--environment-digest",
                self.environment.digest,
                "--recipe",
                "metax-compile-metadata-v1",
            ]
        )
        self.assertEqual(args.profile_command, "collect")
        self.assertEqual(args.campaign_id, self.campaign_id)
        self.assertFalse(hasattr(args, "image"))
        self.assertFalse(hasattr(args, "resource_id"))
        self.assertFalse(hasattr(args, "device"))
        self.assertFalse(hasattr(args, "timeout"))
        self.assertFalse(hasattr(args, "memory"))
        self.assertFalse(hasattr(args, "memory_swap"))
        self.assertFalse(hasattr(args, "ipc"))
        self.assertFalse(hasattr(args, "shm_size"))

        with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(
            SystemExit
        ):
            build_parser().parse_args(
                [
                    "profile",
                    "collect",
                    "--config",
                    "/tmp/config.json",
                    "--database",
                    "/tmp/campaign/campaign.sqlite3",
                    "--gate-id",
                    "production",
                    "--experiment-uid",
                    self.uid,
                    "--namespace-id",
                    CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    "--environment-digest",
                    self.environment.digest,
                    "--recipe",
                    "metax-compile-metadata-v1",
                ]
            )

    def test_inactive_profile_rejects_production_before_docker(self):
        self._record()
        runner = _FakeProfileRunner()
        with (
            mock.patch.object(
                profiling,
                "require_active_profiler",
                side_effect=ValueError("profiler image is inactive"),
            ) as activation_gate,
            self.assertRaisesRegex(ValueError, "profiler image is inactive"),
        ):
            run_bounded_profile(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=self.campaign_id,
                gate_id="production",
                experiment_uid=self.uid,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                execution_environment_digest=self.environment.digest,
                recipe_id="metax-compile-metadata-v1",
            )
        activation_gate.assert_called_once_with()
        self.assertEqual(runner.calls, [])

    def test_host_memory_preflight_blocks_initial_and_prelaunch_actions(self):
        self._record()
        runner = _FakeProfileRunner()
        self.host_memory.side_effect = ValueError(
            "profile host available memory is below the frozen minimum"
        )
        with self.assertRaisesRegex(ValueError, "available memory"):
            self._collect(runner, _FakeGate())
        self.assertEqual(runner.calls, [])
        self.assertEqual(self._campaign_rows("budget_actions"), [])

        self.host_memory.reset_mock()
        self.host_memory.side_effect = [
            _host_memory_preflight_fixture(),
            ValueError("profile host available memory is below the frozen minimum"),
        ]
        with self.assertRaisesRegex(ValueError, "available memory"):
            self._collect(runner, _FakeGate())
        self.assertEqual(runner.calls, [])
        actions = self._campaign_rows("budget_actions")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["status"], "SETTLED")
        self.assertEqual(actions[0]["actual_gpu_ms"], 0)
        self.assertEqual(self._campaign_rows("resource_leases"), [])
        self.assertEqual(self.host_memory.call_count, 2)

        canary_id = "profile-image-canary-low-host-memory"
        self.host_memory.reset_mock()
        self.host_memory.side_effect = ValueError(
            "profile host available memory is below the frozen minimum"
        )
        with self.assertRaisesRegex(ValueError, "available memory"):
            profiling.run_profile_image_doctor(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
                runner=runner,
            )
        with CampaignStore(self.campaign_database) as store:
            with self.assertRaises(KeyError):
                store.get_campaign(canary_id)
        self.assertEqual(runner.calls, [])

    def test_canary_memory_drop_before_hardware_is_known_and_released(self):
        canary_id = "profile-image-canary-memory-drop"
        subject, baseline_ref, binding = self._canary_subject_fixture(canary_id)
        runner = _FakeProfileRunner()
        self.host_memory.side_effect = [
            _host_memory_preflight_fixture(),
            _host_memory_preflight_fixture(),
            ValueError("profile host available memory is below the frozen minimum"),
        ]
        with (
            mock.patch.object(
                profiling,
                "_current_deployment_profile_subject",
                return_value=(subject, baseline_ref, binding),
            ),
            self.assertRaisesRegex(ValueError, "available memory"),
        ):
            profiling.run_profile_image_doctor(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
                runner=runner,
            )
        self.assertEqual(len(runner.calls), 1)
        with CampaignStore(self.campaign_database) as store:
            self.assertEqual(
                store.get_campaign(canary_id)["status"], "PAUSED_OPERATOR"
            )
            action = store.get_budget_action(
                canary_id, idempotency_key="profile-image-canary-v1"
            )
            self.assertEqual(action["status"], "SETTLED")
            self.assertEqual(action["actual"]["gpu_ms"], 0)
            self.assertEqual(
                store.connection.execute(
                    "SELECT status FROM resource_leases WHERE campaign_id = ?",
                    (canary_id,),
                ).fetchone()[0],
                "RELEASED",
            )
            attempts = store.list_profile_canary_attempts(canary_id)
            self.assertEqual(len(attempts), 1)
            self.assertEqual(
                attempts[0]["recipe_id"], "metax-compile-metadata-v1"
            )
            self.assertEqual(attempts[0]["intent"]["schema_version"], 2)
        self.assertEqual(self.host_memory.call_count, 3)

    def test_a5_schema_one_canary_intent_remains_read_only(self):
        campaign_id = "profile-image-canary-a5-readonly"
        action_key = "profile-image-canary-v1"
        intent_material = {
            "schema_version": 1,
            "kind": "PROFILE_IMAGE_CANARY_ATTEMPT",
            "campaign_id": campaign_id,
            "budget_action_key": action_key,
            "recipe_id": "metax-compile-metadata-v1",
            "case_id": "smoke_gate_up",
            "requires_gpu": False,
            "resource_id": None,
            "fencing_epoch": None,
            "profiler_image": (
                "ghcr.io/masechen/autoresearch-metax-profiler@sha256:"
                + "a" * 64
            ),
            "profiler_build_profile_digest": "sha256:" + "b" * 64,
            "profiler_activation_profile_digest": "sha256:" + "c" * 64,
            "container_name": "kar-profile-canary-a5-readonly",
            "argv": ["docker", "run", "fixture"],
            "argv_digest": profiling.canonical_sha256(
                ["docker", "run", "fixture"]
            ),
            "timeout_seconds": 900.0,
            "expected_worker_echo": {"fixture": True},
        }
        attempt_id = (
            "profile-canary-attempt-"
            + profiling.canonical_sha256(intent_material).split(":", 1)[1]
        )
        intent = {**intent_material, "attempt_id": attempt_id}
        with CampaignStore(self.campaign_database) as store:
            store.create_campaign(
                campaign_id=campaign_id,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                mode="DISCOVERY",
                snapshot={"kind": "PROFILE_IMAGE_CANARY"},
                budget_limit=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
                initial_artifact_id="source-sha256-v1:" + "f" * 64,
                initial_policy_snapshot={"kind": "PROFILE_IMAGE_CANARY"},
            )
            store.start_campaign(campaign_id)
            store.reserve_budget(
                campaign_id,
                idempotency_key=action_key,
                action_kind="PROFILE_IMAGE_CANARY",
                amount=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
            )
            store.connection.execute(
                """
                INSERT INTO profile_canary_attempt_intents(
                    attempt_id, campaign_id, budget_action_key, recipe_id,
                    requires_gpu, resource_id, fencing_epoch, intent_digest,
                    intent_json, created_at
                ) VALUES (?, ?, ?, ?, 0, NULL, NULL, ?, ?, ?)
                """,
                (
                    attempt_id,
                    campaign_id,
                    action_key,
                    "metax-compile-metadata-v1",
                    profiling.canonical_sha256(intent),
                    json.dumps(intent, sort_keys=True, separators=(",", ":")),
                    "2026-08-19T03:21:30.000Z",
                ),
            )
            attempts = store.list_profile_canary_attempts(campaign_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["intent"]["schema_version"], 1)
        self.assertNotIn(
            "host_memory_preflight", attempts[0]["intent"]
        )

    def test_image_doctor_cli_exposes_no_runtime_profiler_controls(self):
        args = build_parser().parse_args(
            [
                "profile",
                "image-doctor",
                "--config",
                "/runtime/config.json",
                "--database",
                "/runtime/campaign/campaign.sqlite3",
                "--campaign-id",
                "profiler-canary-1",
            ]
        )
        self.assertEqual(args.profile_command, "image-doctor")
        self.assertEqual(args.campaign_id, "profiler-canary-1")
        for field in (
            "image", "candidate", "case", "device", "timeout", "mctracer",
            "memory", "memory_swap", "ipc", "shm_size",
        ):
            self.assertFalse(hasattr(args, field))

        abandon = build_parser().parse_args(
            [
                "profile",
                "image-doctor-abandon",
                "--config",
                "/runtime/config.json",
                "--database",
                "/runtime/campaign/campaign.sqlite3",
                "--campaign-id",
                "profiler-canary-1",
            ]
        )
        self.assertEqual(abandon.profile_command, "image-doctor-abandon")
        for field in (
            "image", "candidate", "case", "device", "timeout", "mctracer",
            "resource_id", "doctor_digest", "reason", "memory", "memory_swap",
            "ipc", "shm_size",
        ):
            self.assertFalse(hasattr(abandon, field))

        finalize_known = build_parser().parse_args(
            [
                "profile",
                "image-doctor-finalize-known",
                "--config",
                "/runtime/config.json",
                "--database",
                "/runtime/campaign/campaign.sqlite3",
                "--campaign-id",
                "profiler-canary-1",
            ]
        )
        self.assertEqual(
            finalize_known.profile_command,
            "image-doctor-finalize-known",
        )
        for field in (
            "image", "candidate", "case", "device", "timeout", "mctracer",
            "resource_id", "doctor_digest", "reason", "memory", "memory_swap",
            "ipc", "shm_size",
        ):
            self.assertFalse(hasattr(finalize_known, field))

    def test_canary_failure_persists_intent_and_private_diagnostic(self):
        identity = self._record(stage="quick", suite="quick")
        recipe = BUILTIN_PROFILE_RECIPES["metax-hardware-counters-v1"]
        original = profiling._read_only_history_subject(
            state_dir=self.state,
            experiment_uid=identity.experiment_uid,
            namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
            execution_environment_digest=self.environment.digest,
            recipe=recipe,
        )
        canary_id = "profile-image-canary-diagnostic"
        subject = replace(
            original,
            campaign_id=canary_id,
            run_id="profile-image-canary-diagnostic-run",
        )
        baseline = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=identity.candidate_artifact_id,
            source="campaign",
            revision="profile-image-canary-diagnostic-seed",
            execution_environment=self.environment,
        )
        snapshot = {
            "kind": "PROFILE_IMAGE_CANARY",
            "namespace_id": CURRENT_RESEARCH_NAMESPACE.namespace_id,
            "profiler_image": profiling.PROFILER_IMAGE,
            "profiler_build_profile_digest": (
                profiling.PROFILER_BUILD_PROFILE_DIGEST
            ),
            "profiler_activation_profile_digest": (
                profiling.PROFILER_ACTIVATION_PROFILE_DIGEST
            ),
            "worker_revision": profiling.PROFILER_WORKER_REVISION,
            "recipes": list(profiling.PROFILE_RECIPE_IDS),
        }
        action_key = "profile-image-canary-v1"
        observed_intent: list[dict] = []

        def observe_and_write_sentinel(argv):
            with CampaignStore(self.campaign_database) as observer:
                attempts = observer.list_profile_canary_attempts(canary_id)
                self.assertEqual(len(attempts), 1)
                self.assertIsNone(attempts[0]["diagnostic"])
                observed_intent.append(attempts[0])
            output_mount = next(
                argv[index + 1]
                for index, item in enumerate(argv[:-1])
                if item == "--mount" and "dst=/output" in argv[index + 1]
            )
            output = Path(
                output_mount.split("src=", 1)[1].split(",dst=", 1)[0]
            )
            (output / "warmup-sentinel.json").write_text(
                '{"phase":"warmup","status":"FAILED"}',
                encoding="utf-8",
            )
            (output / "warmup-process.json").write_text(
                '{"phase":"warmup","returncode":2}',
                encoding="utf-8",
            )
            (output / "tracked-process.json").write_text(
                '{"phase":"tracked","returncode":null}',
                encoding="utf-8",
            )

        runner = _FakeProfileRunner(
            returncode=17,
            stderr="bounded hardware failure details",
            on_run=observe_and_write_sentinel,
            write_outputs=False,
        )
        with CampaignStore(self.campaign_database) as store:
            store.create_campaign(
                campaign_id=canary_id,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                mode="DISCOVERY",
                snapshot=snapshot,
                budget_limit=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
                initial_baseline_ref=baseline,
                initial_policy_snapshot={"kind": "PROFILE_IMAGE_CANARY"},
                allow_staged_lineage=False,
            )
            store.start_campaign(canary_id)
            store.reserve_budget(
                canary_id,
                idempotency_key=action_key,
                action_kind="PROFILE_IMAGE_CANARY",
                amount=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
            )
            lease = store.acquire_resource(
                canary_id,
                resource_id="gpu1",
                ttl_seconds=1_830,
            )
            with self.assertRaises(profiling._CanaryWorkerFailure) as raised:
                profiling._execute_canary_recipe(
                    self.config,
                    store=store,
                    runner=runner,
                    recipe=recipe,
                    subject=subject,
                    invariant_digest=profiling.canonical_sha256(snapshot),
                    budget_action_key=action_key,
                    lease=lease,
                    container_suffix="diagnostic-fixture",
                    timeout_seconds=900,
                    host_memory_preflight=_host_memory_preflight_fixture(),
                )
            self.assertIsInstance(raised.exception, profiling._CanaryWorkerFailure)
            self.assertIn("non-zero", str(raised.exception))
            attempts = store.list_profile_canary_attempts(canary_id)
            with self.assertRaises(sqlite3.IntegrityError):
                store.connection.execute(
                    """
                    UPDATE profile_canary_attempt_intents
                    SET recipe_id = 'tampered' WHERE campaign_id = ?
                    """,
                    (canary_id,),
                )
            store.connection.rollback()
            with self.assertRaises(sqlite3.IntegrityError):
                store.connection.execute(
                    """
                    UPDATE profile_canary_attempt_diagnostics
                    SET classification = 'SUCCESS' WHERE campaign_id = ?
                    """,
                    (canary_id,),
                )
            store.connection.rollback()

        self.assertEqual(len(observed_intent), 1)
        self.assertEqual(observed_intent[0]["intent"]["schema_version"], 2)
        self.assertEqual(
            observed_intent[0]["intent"]["host_memory_preflight"],
            _host_memory_preflight_fixture(),
        )
        tampered = json.loads(json.dumps(observed_intent[0]["intent"]))
        tampered["host_memory_preflight"]["mem_available_bytes"] = 1
        material = dict(tampered)
        material.pop("attempt_id")
        tampered["attempt_id"] = (
            "profile-canary-attempt-"
            + profiling.canonical_sha256(material).split(":", 1)[1]
        )
        with CampaignStore(self.campaign_database) as store:
            with self.assertRaisesRegex(ValueError, "memory is insufficient"):
                store.record_profile_canary_attempt_intent(
                    canary_id, intent=tampered
                )
        tampered = json.loads(json.dumps(observed_intent[0]["intent"]))
        tampered["host_memory_preflight"]["required_available_bytes"] = 1
        material = dict(tampered)
        material.pop("attempt_id")
        tampered["attempt_id"] = (
            "profile-canary-attempt-"
            + profiling.canonical_sha256(material).split(":", 1)[1]
        )
        with CampaignStore(self.campaign_database) as store:
            with self.assertRaisesRegex(ValueError, "contract drifted"):
                store.record_profile_canary_attempt_intent(
                    canary_id, intent=tampered
                )
        self.assertEqual(len(attempts), 1)
        diagnostic = attempts[0]["diagnostic"]
        self.assertEqual(diagnostic["classification"], "UNKNOWN")
        self.assertEqual(diagnostic["reason_code"], "NONZERO_EXIT")
        self.assertEqual(diagnostic["returncode"], 17)
        self.assertFalse(diagnostic["timed_out"])
        object_id = diagnostic["diagnostic_object_id"]
        digest = object_id.removeprefix("sha256:")
        private_object = (
            self.controller / "objects" / "sha256" / digest[:2] / digest[2:]
        )
        payload = json.loads(private_object.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["command"]["stderr"]["content"],
            "bounded hardware failure details",
        )
        self.assertEqual(
            payload["worker_files"]["outcome.json"]["status"], "ABSENT"
        )
        self.assertEqual(
            payload["worker_files"]["warmup-sentinel.json"]["status"],
            "PRESENT",
        )
        for phase_file in ("warmup-process.json", "tracked-process.json"):
            self.assertEqual(
                payload["worker_files"][phase_file]["status"],
                "PRESENT",
            )
            self.assertGreater(
                len(
                    base64.b64decode(
                        payload["worker_files"][phase_file]["content_base64"]
                    )
                ),
                0,
            )
        self.assertIn(
            "warmup-sentinel.json",
            {entry["path"] for entry in payload["file_inventory"]["entries"]},
        )

    def test_canary_start_failure_is_known_and_diagnostic_is_durable(self):
        identity = self._record(stage="smoke", suite="smoke")
        recipe = BUILTIN_PROFILE_RECIPES["metax-compile-metadata-v1"]
        original = profiling._read_only_history_subject(
            state_dir=self.state,
            experiment_uid=identity.experiment_uid,
            namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
            execution_environment_digest=self.environment.digest,
            recipe=recipe,
        )
        campaign_id = "profile-image-canary-start-failure"
        subject = replace(
            original,
            campaign_id=campaign_id,
            run_id="profile-image-canary-start-failure-run",
        )
        baseline = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=identity.candidate_artifact_id,
            source="campaign",
            revision="profile-image-canary-start-failure-seed",
            execution_environment=self.environment,
        )
        snapshot = {
            "kind": "PROFILE_IMAGE_CANARY",
            "profiler_image": profiling.PROFILER_IMAGE,
            "profiler_build_profile_digest": (
                profiling.PROFILER_BUILD_PROFILE_DIGEST
            ),
            "profiler_activation_profile_digest": (
                profiling.PROFILER_ACTIVATION_PROFILE_DIGEST
            ),
            "recipes": list(profiling.PROFILE_RECIPE_IDS),
        }
        with CampaignStore(self.campaign_database) as store:
            store.create_campaign(
                campaign_id=campaign_id,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                mode="DISCOVERY",
                snapshot=snapshot,
                budget_limit=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
                initial_baseline_ref=baseline,
                initial_policy_snapshot={"kind": "PROFILE_IMAGE_CANARY"},
                allow_staged_lineage=False,
            )
            store.start_campaign(campaign_id)
            store.reserve_budget(
                campaign_id,
                idempotency_key="profile-image-canary-v1",
                action_kind="PROFILE_IMAGE_CANARY",
                amount=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
            )
            with self.assertRaises(profiling._CanaryWorkerFailure) as raised:
                profiling._execute_canary_recipe(
                    self.config,
                    store=store,
                    runner=_FakeProfileRunner(start_error=True),
                    recipe=recipe,
                    subject=subject,
                    invariant_digest=profiling.canonical_sha256(snapshot),
                    budget_action_key="profile-image-canary-v1",
                    lease=None,
                    container_suffix="start-failure",
                    timeout_seconds=900,
                    host_memory_preflight=_host_memory_preflight_fixture(),
                )
            attempts = store.list_profile_canary_attempts(campaign_id)

        self.assertTrue(raised.exception.known)
        self.assertFalse(raised.exception.hard)
        self.assertEqual(raised.exception.reason_code, "RUNNER_START_FAILED")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(
            attempts[0]["diagnostic"]["classification"],
            "KNOWN_FAILURE",
        )
        self.assertIsNone(attempts[0]["diagnostic"]["returncode"])
        object_id = attempts[0]["diagnostic"]["diagnostic_object_id"]
        digest = object_id.split(":", 1)[1]
        payload = json.loads(
            (
                self.controller
                / "objects"
                / "sha256"
                / digest[:2]
                / digest[2:]
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(payload["command"]["available"])
        self.assertEqual(payload["cause"]["type"], "ValueError")

        interrupted_campaign = "profile-image-canary-interrupted-before-start"
        interrupted_subject = replace(
            original,
            campaign_id=interrupted_campaign,
            run_id="profile-image-canary-interrupted-run",
        )

        def interrupt(_argv):
            raise KeyboardInterrupt("operator interruption fixture")

        with CampaignStore(self.campaign_database) as store:
            store.create_campaign(
                campaign_id=interrupted_campaign,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                mode="DISCOVERY",
                snapshot=snapshot,
                budget_limit=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
                initial_baseline_ref=baseline,
                initial_policy_snapshot={"kind": "PROFILE_IMAGE_CANARY"},
                allow_staged_lineage=False,
            )
            store.start_campaign(interrupted_campaign)
            store.reserve_budget(
                interrupted_campaign,
                idempotency_key="profile-image-canary-v1",
                action_kind="PROFILE_IMAGE_CANARY",
                amount=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
            )
            with self.assertRaises(profiling._CanaryWorkerFailure) as interrupted:
                profiling._execute_canary_recipe(
                    self.config,
                    store=store,
                    runner=_FakeProfileRunner(on_run=interrupt),
                    recipe=recipe,
                    subject=interrupted_subject,
                    invariant_digest=profiling.canonical_sha256(snapshot),
                    budget_action_key="profile-image-canary-v1",
                    lease=None,
                    container_suffix="interrupted",
                    timeout_seconds=900,
                    host_memory_preflight=_host_memory_preflight_fixture(),
                )
            interrupted_attempts = store.list_profile_canary_attempts(
                interrupted_campaign
            )

        self.assertTrue(interrupted.exception.known)
        self.assertEqual(interrupted.exception.reason_code, "RUNNER_INTERRUPTED")
        self.assertEqual(
            interrupted_attempts[0]["diagnostic"]["reason_code"],
            "RUNNER_INTERRUPTED",
        )

    def test_operator_abandon_runs_fresh_doctor_and_never_replays_old_action(self):
        identity = self._record(stage="quick", suite="quick")
        canary_id = "profile-image-canary-operator-abandon"
        baseline = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=identity.candidate_artifact_id,
            source="campaign",
            revision="profile-image-canary-abandon-seed",
            execution_environment=self.environment,
        )
        snapshot = {
            "schema_version": 1,
            "kind": "PROFILE_IMAGE_CANARY",
            "namespace_id": CURRENT_RESEARCH_NAMESPACE.namespace_id,
            "profiler_image": profiling.PROFILER_IMAGE,
            "profiler_build_profile_digest": (
                profiling.PROFILER_BUILD_PROFILE_DIGEST
            ),
            "profiler_activation_profile_digest": (
                profiling.PROFILER_ACTIVATION_PROFILE_DIGEST
            ),
            "worker_revision": profiling.PROFILER_WORKER_REVISION,
            "recipes": list(profiling.PROFILE_RECIPE_IDS),
        }
        with CampaignStore(self.campaign_database) as store:
            store.create_campaign(
                campaign_id=canary_id,
                namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                mode="DISCOVERY",
                snapshot=snapshot,
                budget_limit=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
                initial_baseline_ref=baseline,
                initial_policy_snapshot={"kind": "PROFILE_IMAGE_CANARY"},
                allow_staged_lineage=False,
            )
            store.start_campaign(canary_id)
            store.reserve_budget(
                canary_id,
                idempotency_key="profile-image-canary-v1",
                action_kind="PROFILE_IMAGE_CANARY",
                amount=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
            )
            lease = store.acquire_resource(
                canary_id, resource_id="gpu1", ttl_seconds=1_830
            )
            store.release_resource(
                lease, quarantine=True, reason="unknown image canary outcome"
            )
            store.pause_campaign(
                canary_id,
                status="PAUSED_UNKNOWN_OUTCOME",
                reason="unknown image canary outcome",
            )
        doctor = {
            "schema_version": 1,
            "kind": "CAMPAIGN_RESUME_DOCTOR",
            "status": "SUCCESS",
            "campaign_id": canary_id,
            "resource_id": "gpu1",
            "observed_epoch": profiling.time.time() + 1,
            "namespace_id": CURRENT_RESEARCH_NAMESPACE.namespace_id,
            "config_digest": profiling.canonical_sha256(
                self.config.redacted_dict()
            ),
            "execution_environment": self.environment.to_dict(),
            "doctor_result": {
                "status": "SUCCESS",
                "c500_probe": {
                    "environment": {"compile_probe_status": "PASSED"}
                },
            },
        }
        history_before = hashlib.sha256(
            (self.state / "history.sqlite3").read_bytes()
        ).hexdigest()
        with mock.patch.object(
            profiling, "trusted_resume_doctor", return_value=doctor
        ) as trusted_doctor:
            report = profiling.abandon_profile_image_canary(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
            )
        self.assertEqual(report["status"], "ABANDONED")
        self.assertFalse(report["replay_permitted"])
        self.assertEqual(report["attempts"], [])
        trusted_doctor.assert_called_once()
        self.assertEqual(
            trusted_doctor.call_args.kwargs["trusted_namespace"],
            CURRENT_RESEARCH_NAMESPACE,
        )
        self.assertEqual(
            hashlib.sha256((self.state / "history.sqlite3").read_bytes()).hexdigest(),
            history_before,
        )
        with CampaignStore(self.campaign_database) as store:
            self.assertEqual(store.get_campaign(canary_id)["status"], "CANCELLED")
            self.assertEqual(
                store.get_budget_action(
                    canary_id, idempotency_key="profile-image-canary-v1"
                )["status"],
                "RESERVED",
            )
            lease_row = store.connection.execute(
                """
                SELECT status, fencing_epoch FROM resource_leases
                WHERE campaign_id = ?
                """,
                (canary_id,),
            ).fetchone()
            self.assertEqual(dict(lease_row), {
                "status": "RELEASED",
                "fencing_epoch": lease.fencing_epoch,
            })
        with mock.patch.object(profiling, "trusted_resume_doctor") as no_replay:
            replay = profiling.abandon_profile_image_canary(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
            )
        self.assertEqual(replay["status"], "ALREADY_ABANDONED")
        no_replay.assert_not_called()

    def test_a8_cannot_reopen_the_completed_a6_known_failure_finalizer(self):
        with (
            mock.patch.object(
                profiling, "_current_deployment_profile_subject"
            ) as no_subject,
            self.assertRaisesRegex(ValueError, "exact inactive A7 build"),
        ):
            profiling.finalize_known_profile_image_canary(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id="profile-image-canary-a6-retired-finalizer",
            )
        no_subject.assert_not_called()

    @mock.patch.object(
        profiling,
        "PROFILER_BUILD_PROFILE_DIGEST",
        profiling._A7_KNOWN_FINALIZER_BUILD_DIGEST,
    )
    @mock.patch.object(profiling, "PROFILER_ACTIVE", False)
    @mock.patch.object(
        profiling,
        "PROFILER_IMAGE",
        "ghcr.io/masechen/autoresearch-metax-profiler@sha256:" + "0" * 64,
    )
    def test_known_failure_finalizer_preserves_evidence_without_gpu_or_doctor(self):
        canary_id = "profile-image-canary-a6-known-finalization"
        subject, baseline_ref, binding, _snapshot = (
            self._create_a6_known_failure_canary(canary_id)
        )
        with CampaignStore(self.campaign_database) as store:
            action_before = store.get_budget_action(
                canary_id,
                idempotency_key="profile-image-canary-v1",
            )
            lease_before = dict(
                store.connection.execute(
                    "SELECT * FROM resource_leases WHERE campaign_id = ?",
                    (canary_id,),
                ).fetchone()
            )
            attempts_before = store.list_profile_canary_attempts(canary_id)
        history_before = hashlib.sha256(
            (self.state / "history.sqlite3").read_bytes()
        ).hexdigest()
        with (
            mock.patch.object(
                profiling,
                "_current_deployment_profile_subject",
                return_value=(subject, baseline_ref, binding),
            ),
            mock.patch.object(
                profiling,
                "trusted_resume_doctor",
                side_effect=AssertionError("known finalizer must not run doctor"),
            ) as no_doctor,
            mock.patch.object(
                profiling,
                "gpu_lock",
                side_effect=AssertionError("known finalizer must not take GPU lock"),
            ) as no_gpu_lock,
        ):
            report = profiling.finalize_known_profile_image_canary(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
            )
            replay = profiling.finalize_known_profile_image_canary(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
            )
        self.assertEqual(report["status"], "FINALIZED")
        self.assertEqual(replay["status"], "ALREADY_FINALIZED")
        self.assertFalse(report["doctor_invoked"])
        self.assertFalse(report["gpu_action_invoked"])
        self.assertFalse(report["replay_permitted"])
        no_doctor.assert_not_called()
        no_gpu_lock.assert_not_called()
        finalization = report["finalization"]["finalization"]
        self.assertEqual(
            [proof["classification"] for proof in finalization["attempts"]],
            ["SUCCESS", "KNOWN_FAILURE"],
        )
        self.assertEqual(finalization["preserved_budget"]["status"], "SETTLED")
        self.assertEqual(finalization["preserved_lease"]["status"], "RELEASED")
        with CampaignStore(self.campaign_database) as store:
            self.assertEqual(store.get_campaign(canary_id)["status"], "CANCELLED")
            self.assertEqual(
                store.get_budget_action(
                    canary_id,
                    idempotency_key="profile-image-canary-v1",
                ),
                action_before,
            )
            self.assertEqual(
                dict(
                    store.connection.execute(
                        "SELECT * FROM resource_leases WHERE campaign_id = ?",
                        (canary_id,),
                    ).fetchone()
                ),
                lease_before,
            )
            self.assertEqual(
                store.list_profile_canary_attempts(canary_id),
                attempts_before,
            )
            self.assertEqual(
                store.connection.execute(
                    """
                    SELECT event_type FROM outbox
                    WHERE campaign_id = ?
                      AND event_type = 'PROFILE_IMAGE_CANARY_KNOWN_FAILURE_FINALIZED'
                    """,
                    (canary_id,),
                ).fetchone()[0],
                "PROFILE_IMAGE_CANARY_KNOWN_FAILURE_FINALIZED",
            )
            for statement in (
                """
                UPDATE profile_canary_known_finalizations
                SET prior_campaign_status = 'RUNNING'
                WHERE campaign_id = ?
                """,
                """
                DELETE FROM profile_canary_known_finalizations
                WHERE campaign_id = ?
                """,
            ):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "known finalization is immutable",
                ):
                    store.connection.execute(statement, (canary_id,))
        self.assertEqual(
            hashlib.sha256((self.state / "history.sqlite3").read_bytes()).hexdigest(),
            history_before,
        )

    @mock.patch.object(
        profiling,
        "PROFILER_BUILD_PROFILE_DIGEST",
        profiling._A7_KNOWN_FINALIZER_BUILD_DIGEST,
    )
    @mock.patch.object(profiling, "PROFILER_ACTIVE", False)
    @mock.patch.object(
        profiling,
        "PROFILER_IMAGE",
        "ghcr.io/masechen/autoresearch-metax-profiler@sha256:" + "0" * 64,
    )
    def test_known_failure_finalizer_rejects_identity_and_state_drift(self):
        with (
            mock.patch.object(profiling, "PROFILER_ACTIVE", True),
            mock.patch.object(
                profiling, "_current_deployment_profile_subject"
            ) as no_subject,
            self.assertRaisesRegex(ValueError, "exact inactive A7 build"),
        ):
            profiling.finalize_known_profile_image_canary(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id="profile-image-canary-a6-active-code",
            )
        no_subject.assert_not_called()

        with (
            mock.patch.object(
                profiling, "_current_deployment_profile_subject"
            ) as no_subject,
            self.assertRaisesRegex(ValueError, "conventional absolute path"),
        ):
            profiling.finalize_known_profile_image_canary(
                self.config,
                campaign_database=self.root / "custom-campaign.sqlite3",
                campaign_id="profile-image-canary-a6-custom-database",
            )
        no_subject.assert_not_called()

        fixtures = (
            {
                "suffix": "identity",
                "create": {"activation_digest": "sha256:" + "f" * 64},
                "message": "identity",
            },
            {
                "suffix": "diagnostic",
                "create": {"omit_hardware_diagnostic": True},
                "message": "diagnostic is unavailable",
            },
            {
                "suffix": "budget",
                "create": {"settle_action": False},
                "message": "SETTLED action",
            },
            {
                "suffix": "lease",
                "create": {"release_lease": False},
                "message": "RELEASED lease",
            },
        )
        for fixture in fixtures:
            with self.subTest(fixture=fixture["suffix"]):
                canary_id = "profile-image-canary-a6-drift-" + fixture["suffix"]
                subject, baseline_ref, binding, _snapshot = (
                    self._create_a6_known_failure_canary(
                        canary_id,
                        **fixture["create"],
                    )
                )
                with (
                    mock.patch.object(
                        profiling,
                        "_current_deployment_profile_subject",
                        return_value=(subject, baseline_ref, binding),
                    ),
                    self.assertRaisesRegex(ValueError, fixture["message"]),
                ):
                    profiling.finalize_known_profile_image_canary(
                        self.config,
                        campaign_database=self.campaign_database,
                        campaign_id=canary_id,
                    )
                with CampaignStore(self.campaign_database) as store:
                    self.assertEqual(
                        store.get_campaign(canary_id)["status"],
                        "PAUSED_OPERATOR",
                    )
                    self.assertIsNone(
                        store.get_profile_canary_known_finalization(canary_id)
                    )

        canary_id = "profile-image-canary-a6-drift-private-cas"
        subject, baseline_ref, binding, _snapshot = (
            self._create_a6_known_failure_canary(canary_id)
        )
        with CampaignStore(self.campaign_database) as store:
            object_id = store.list_profile_canary_attempts(canary_id)[0][
                "diagnostic"
            ]["diagnostic_object_id"]
        digest = object_id.split(":", 1)[1]
        private_path = (
            self.controller / "objects" / "sha256" / digest[:2] / digest[2:]
        )
        private_path.write_bytes(b"{}")
        with (
            mock.patch.object(
                profiling,
                "_current_deployment_profile_subject",
                return_value=(subject, baseline_ref, binding),
            ),
            self.assertRaisesRegex(ValueError, "digest is inconsistent"),
        ):
            profiling.finalize_known_profile_image_canary(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
            )
        with CampaignStore(self.campaign_database) as store:
            self.assertEqual(
                store.get_campaign(canary_id)["status"],
                "PAUSED_OPERATOR",
            )
            self.assertIsNone(
                store.get_profile_canary_known_finalization(canary_id)
            )

        canary_id = "profile-image-canary-a6-drift-after-finalization"
        subject, baseline_ref, binding, _snapshot = (
            self._create_a6_known_failure_canary(canary_id)
        )
        subject_patch = mock.patch.object(
            profiling,
            "_current_deployment_profile_subject",
            return_value=(subject, baseline_ref, binding),
        )
        with subject_patch:
            profiling.finalize_known_profile_image_canary(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
            )
        with CampaignStore(self.campaign_database) as store:
            store.connection.execute(
                """
                UPDATE budget_actions SET actual_wall_ms = actual_wall_ms + 1
                WHERE campaign_id = ? AND idempotency_key = ?
                """,
                (canary_id, "profile-image-canary-v1"),
            )
            store.connection.commit()
        with (
            mock.patch.object(
                profiling,
                "_current_deployment_profile_subject",
                return_value=(subject, baseline_ref, binding),
            ),
            self.assertRaisesRegex(ValueError, "replay proof differs"),
        ):
            profiling.finalize_known_profile_image_canary(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
            )

    def test_image_doctor_runs_two_recipes_without_history_or_baseline_writes(self):
        identity = self._record(stage="confirmation", suite="full")
        with closing(
            sqlite3.connect(self.state / "history.sqlite3")
        ) as connection, connection:
            row = connection.execute(
                """
                SELECT e.candidate_hash, a.object_path
                FROM experiments e JOIN candidate_artifacts a
                  ON a.artifact_id = e.artifact_id
                WHERE e.experiment_uid = ?
                """,
                (identity.experiment_uid,),
            ).fetchone()
        assert row is not None
        canary_id = "profile-image-canary-test"
        subject = profiling._ProfileSubject(
            experiment_uid=identity.experiment_uid,
            namespace_id=identity.namespace_id,
            condition_digest=identity.condition_digest,
            artifact_id=str(identity.candidate_artifact_id),
            execution_environment_digest=identity.execution_environment.digest,
            campaign_id=canary_id,
            run_id="profile-image-canary-run",
            stage="confirmation",
            suite="full",
            replicate_kind="confirmation",
            candidate_object_path=self.state / row[1],
            candidate_content_sha256=row[0],
        )
        baseline_ref = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=identity.candidate_artifact_id,
            source="campaign",
            revision="profile-image-canary-fixture",
            execution_environment=self.environment,
        )
        binding = {
            "deployment_evidence_digest": "sha256:" + "d" * 64,
            "deployment_git_commit": "e" * 40,
            "deployment_candidate_hash": row[0],
            "current_confirmation_experiment_uid": identity.experiment_uid,
            "current_confirmation_identity": identity.to_dict(),
            "execution_environment": self.environment.to_dict(),
        }
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(self.campaign_database) + suffix).unlink()
            except FileNotFoundError:
                pass
        history_before = hashlib.sha256(
            (self.state / "history.sqlite3").read_bytes()
        ).hexdigest()
        runner = _FakeProfileRunner()
        with (
            mock.patch.object(profiling, "require_active_profiler"),
            mock.patch.object(
                profiling,
                "_current_deployment_profile_subject",
                return_value=(subject, baseline_ref, binding),
            ),
        ):
            report = profiling.run_profile_image_doctor(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
                runner=runner,
            )
        self.assertEqual(report["status"], "READY")
        self.assertNotIn("host_memory_preflight", json.dumps(report))
        self.assertEqual(len(runner.calls), 2)
        self.assertTrue(
            all(
                0 < float(kwargs["timeout_sec"]) <= PROFILE_TIMEOUT_SECONDS
                for _argv, kwargs in runner.calls
            )
        )
        self.assertEqual(
            report["raw_trace_total_bytes"],
            len(runner.raw_trace) * 2,
        )
        self.assertEqual(
            [
                argv[argv.index("--recipe") + 1]
                for argv, _kwargs in runner.calls
            ],
            list(profiling.PROFILE_RECIPE_IDS),
        )
        self.assertEqual(
            hashlib.sha256((self.state / "history.sqlite3").read_bytes()).hexdigest(),
            history_before,
        )
        with CampaignStore(self.campaign_database) as store:
            campaign = store.get_campaign(canary_id)
            self.assertEqual(campaign["status"], "COMPLETED")
            attempts = store.list_profile_canary_attempts(canary_id)
            self.assertEqual(
                [attempt["recipe_id"] for attempt in attempts],
                sorted(profiling.PROFILE_RECIPE_IDS),
            )
            self.assertTrue(
                all(
                    attempt["diagnostic"]["classification"] == "SUCCESS"
                    for attempt in attempts
                )
            )
            self.assertTrue(
                all(attempt["intent"]["schema_version"] == 2 for attempt in attempts)
            )
            self.assertTrue(
                all(
                    attempt["intent"]["host_memory_preflight"]
                    == _host_memory_preflight_fixture()
                    for attempt in attempts
                )
            )
            action = store.get_budget_action(
                canary_id, idempotency_key="profile-image-canary-v1"
            )
            self.assertEqual(action["status"], "SETTLED")
            self.assertEqual(action["reserved"]["wall_ms"], 1_800_000)
            self.assertEqual(action["reserved"]["gpu_ms"], 900_000)
            self.assertEqual(
                store.connection.execute(
                    "SELECT status FROM resource_leases ORDER BY rowid DESC LIMIT 1"
                ).fetchone()[0],
                "RELEASED",
            )
        self.assertEqual(self.host_memory.call_count, 3)

    def test_image_doctor_rejects_aggregate_raw_trace_over_64_mib(self):
        identity = self._record(stage="confirmation", suite="full")
        with closing(
            sqlite3.connect(self.state / "history.sqlite3")
        ) as connection, connection:
            row = connection.execute(
                """
                SELECT e.candidate_hash, a.object_path
                FROM experiments e JOIN candidate_artifacts a
                  ON a.artifact_id = e.artifact_id
                WHERE e.experiment_uid = ?
                """,
                (identity.experiment_uid,),
            ).fetchone()
        assert row is not None
        canary_id = "profile-image-canary-aggregate-limit"
        subject = profiling._ProfileSubject(
            experiment_uid=identity.experiment_uid,
            namespace_id=identity.namespace_id,
            condition_digest=identity.condition_digest,
            artifact_id=str(identity.candidate_artifact_id),
            execution_environment_digest=identity.execution_environment.digest,
            campaign_id=canary_id,
            run_id="profile-image-canary-aggregate-run",
            stage="confirmation",
            suite="full",
            replicate_kind="confirmation",
            candidate_object_path=self.state / row[1],
            candidate_content_sha256=row[0],
        )
        baseline_ref = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=identity.candidate_artifact_id,
            source="campaign",
            revision="profile-image-canary-aggregate-fixture",
            execution_environment=self.environment,
        )
        binding = {
            "deployment_evidence_digest": "sha256:" + "d" * 64,
            "deployment_git_commit": "e" * 40,
            "deployment_candidate_hash": row[0],
            "current_confirmation_experiment_uid": identity.experiment_uid,
            "current_confirmation_identity": identity.to_dict(),
            "execution_environment": self.environment.to_dict(),
        }
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(self.campaign_database) + suffix).unlink()
            except FileNotFoundError:
                pass

        def report(_config, *, recipe, **_kwargs):
            return {
                "recipe_id": recipe.recipe_id,
                "case_id": recipe.case_id,
                "metrics": {},
                "toolchain": {},
                "trace_descriptor": {},
                "raw_trace": {
                    "object_id": "sha256:" + "f" * 64,
                    "byte_size": 6,
                    "visibility": "controller-private-cas",
                },
            }

        with (
            mock.patch.object(profiling, "require_active_profiler"),
            mock.patch.object(
                profiling,
                "_current_deployment_profile_subject",
                return_value=(subject, baseline_ref, binding),
            ),
            mock.patch.object(
                profiling, "_execute_canary_recipe", side_effect=report
            ) as execute,
            mock.patch.object(
                profiling, "PROFILE_CANARY_RAW_TRACE_TOTAL_LIMIT_BYTES", 10
            ),
            self.assertRaisesRegex(ValueError, "aggregate limit"),
        ):
            profiling.run_profile_image_doctor(
                self.config,
                campaign_database=self.campaign_database,
                campaign_id=canary_id,
                runner=_FakeProfileRunner(),
            )
        self.assertEqual(execute.call_count, 2)
        with CampaignStore(self.campaign_database) as store:
            self.assertEqual(
                store.get_campaign(canary_id)["status"],
                "PAUSED_OPERATOR",
            )
            self.assertEqual(
                store.get_budget_action(
                    canary_id,
                    idempotency_key="profile-image-canary-v1",
                )["status"],
                "SETTLED",
            )
            self.assertEqual(
                store.connection.execute(
                    "SELECT status FROM resource_leases ORDER BY rowid DESC LIMIT 1"
                ).fetchone()[0],
                "RELEASED",
            )

    def test_current_canary_uses_quick_subject_anchored_by_confirmation(self):
        source = "def kernel():\n    return 1\n"
        bundle = CandidateBundle.single_file(content=source)
        baseline = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=bundle.artifact_id,
            source="deployment",
            revision="qualified-current-parent",
            execution_environment=self.environment,
        )
        run_id = "current-bootstrap-profile-fixture"
        quick = ExperimentIdentity.create(
            experiment_uid="10000000-0000-4000-8000-000000000001",
            namespace=CURRENT_RESEARCH_NAMESPACE,
            mode="DISCOVERY",
            candidate_artifact_id=bundle.artifact_id,
            parent_artifact_id=bundle.artifact_id,
            baseline=baseline,
            execution_environment=self.environment,
            stage="quick",
            suite="quick",
            replicate_kind="validation",
            run_id=run_id,
            iteration=1,
        )
        confirmation = ExperimentIdentity.create(
            experiment_uid="10000000-0000-4000-8000-000000000002",
            namespace=CURRENT_RESEARCH_NAMESPACE,
            mode="DISCOVERY",
            candidate_artifact_id=bundle.artifact_id,
            parent_artifact_id=bundle.artifact_id,
            baseline=baseline,
            execution_environment=self.environment,
            stage="confirmation",
            suite="full",
            replicate_kind="confirmation",
            run_id=run_id,
            iteration=1,
        )
        with HistoryStore(
            self.state / "history.sqlite3", state_dir=self.state
        ) as history:
            history.ensure_namespace(
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
                CURRENT_RESEARCH_NAMESPACE.to_dict(),
            )
            history.store_candidate_bundle(bundle)
            history.record_experiment(
                candidate_source=source,
                backend="c500",
                suite="quick",
                status="SUCCESS",
                identity=quick,
                result={
                    "status": "SUCCESS",
                    "evaluation_protocol_id": (
                        CURRENT_C500_EVALUATION_PROTOCOL_ID
                    ),
                    "request_identity": quick.to_dict(),
                },
                case_measurements=(
                    {
                        "name": "quick_decode_gate_up",
                        "matched_ratio": 1.0,
                        "passed": True,
                    },
                ),
            )
            history.record_experiment(
                candidate_source=source,
                backend="c500",
                suite="full",
                status="SUCCESS",
                promotable=True,
                identity=confirmation,
                result={
                    "status": "SUCCESS",
                    "evaluation_protocol_id": (
                        CURRENT_C500_EVALUATION_PROTOCOL_ID
                    ),
                    "request_identity": confirmation.to_dict(),
                    "promotion": {
                        "phase": "confirmation",
                        "confirmed": True,
                    },
                },
                case_measurements=(
                    {
                        "name": "full_decode_gate_up",
                        "matched_ratio": 1.0,
                        "passed": True,
                    },
                ),
            )
        candidate_hash = hashlib.sha256(source.encode()).hexdigest()
        fake_controller = mock.Mock()
        fake_controller._deployment_pin.return_value = SimpleNamespace(
            candidate_hash=candidate_hash,
            execution_environment=self.environment,
            evidence_digest="sha256:" + "a" * 64,
            git_commit="b" * 40,
        )
        fake_controller._best.return_value = SimpleNamespace(
            artifact_id=str(bundle.artifact_id)
        )
        fake_controller._resolved_execution_environment.return_value = (
            self.environment
        )
        with mock.patch.object(
            profiling, "ResearchController", return_value=fake_controller
        ):
            subject, _baseline, binding = (
                profiling._current_deployment_profile_subject(
                    self.config, campaign_id="canary"
                )
            )
        self.assertEqual(subject.experiment_uid, quick.experiment_uid)
        self.assertEqual(
            binding["current_confirmation_experiment_uid"],
            confirmation.experiment_uid,
        )
        self.assertEqual(
            binding["current_quick_experiment_uid"], quick.experiment_uid
        )

    def test_builtin_recipe_registry_is_sealed(self):
        with self.assertRaises(TypeError):
            BUILTIN_PROFILE_RECIPES["agent-recipe"] = (  # type: ignore[index]
                BUILTIN_PROFILE_RECIPES["metax-compile-metadata-v1"]
            )


if __name__ == "__main__":
    unittest.main()
