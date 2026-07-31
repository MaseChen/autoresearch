from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest

from kernel_research.autorun.controller import ResearchController, gpu_lock
from kernel_research.autorun.models import ControllerConfig, ProposalV1
from kernel_research.autorun.proposal import (
    ProposalRequest,
    Proposer,
    ProposerStepLimitError,
    parse_opencode_ndjson,
    parse_proposal_text,
)
from kernel_research.autorun.runtime import (
    CommandRunner,
    evaluator_argv,
    proposer_argv,
    write_opencode_config,
)
from kernel_research.autorun.store import ControllerStore
from kernel_research.cli import main as evaluator_main
from kernel_research.constants import OPENCODE_PROPOSER_STEPS
from kernel_research.evaluation import record_external_result
from kernel_research.history import HistoryStore
from kernel_research.research_policy import validate_research_candidate


ROOT = Path(__file__).resolve().parents[1]
SEED = (ROOT / "kernel.py").read_text(encoding="utf-8")
SEED_HASH = hashlib.sha256(SEED.encode("utf-8")).hexdigest()
FULL_CASES = (
    "full_decode_gate_up",
    "full_prefill_gate_up",
    "full_decode_down",
    "full_prefill_down",
)

_BLOCK_SIZE_N_ASSIGNMENT = re.compile(
    r"(?m)^    block_size_n = [0-9]+$"
)


def _with_block_size_n(source: str, value: int) -> str:
    updated, count = _BLOCK_SIZE_N_ASSIGNMENT.subn(
        f"    block_size_n = {value}", source
    )
    if count != 1:
        raise AssertionError(
            "fixture source must contain exactly one block_size_n assignment"
        )
    return updated


def _with_different_block_size_n(source: str) -> str:
    for value in (16, 32, 64, 128, 256, 512):
        updated = _with_block_size_n(source, value)
        if updated != source:
            return updated
    raise AssertionError("could not construct a distinct fixture candidate")


def _proposal_value(source: str = SEED) -> dict:
    return {
        "schema_version": 1,
        "parent_candidate_hash": SEED_HASH,
        "hypothesis": "Change one tile parameter.",
        "rationale": "A smaller tile may improve occupancy and is directly testable.",
        "kernel_source": source,
    }


def _config(root: Path, *, max_candidates: int = 5) -> ControllerConfig:
    repository = root / "repo"
    state = root / "state"
    controller = root / "controller"
    checkpoints = root / "checkpoints"
    cache = root / "cache"
    key = root / "secret"
    docker = root / "docker"
    for directory in (repository, state, controller, checkpoints, cache):
        directory.mkdir(parents=True, exist_ok=True)
    key.write_text("secret", encoding="utf-8")
    key.chmod(0o600)
    docker.write_text("", encoding="utf-8")
    devices = tuple(root / name for name in ("mxcd", "card2", "renderD129"))
    for device in devices:
        device.write_text("", encoding="utf-8")
    return ControllerConfig(
        repository_dir=repository,
        state_dir=state,
        controller_dir=controller,
        checkpoint_dir=checkpoints,
        docker_binary=docker,
        proposer_image="local/proposer@sha256:" + "1" * 64,
        evaluator_image="local/evaluator@sha256:" + "2" * 64,
        deepseek_key_file=key,
        gpu_devices=devices,
        evaluator_cache_dir=cache,
        expected_git_commit="8" * 40,
        expected_kernel_hash=SEED_HASH,
        max_candidates=max_candidates,
        acknowledge_gpu_passthrough_risk=True,
    )


def _baseline(state: Path) -> None:
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
        "eligible_for_promotion": True,
        "aggregate_score": 1.0,
        "cases": cases,
        "promotion": {"phase": "baseline", "confirmed": True},
        "environment": {"device_name": "MetaX C500"},
    }
    with HistoryStore(state / "history.sqlite3", state_dir=state) as history:
        history.record_experiment(
            candidate_source=SEED,
            candidate_hash=SEED_HASH,
            status="SUCCESS",
            backend="c500",
            suite="full",
            promotable=True,
            aggregate_score=1.0,
            note="fixture baseline",
            result=result,
            environment=result["environment"],
            case_measurements=(
                {
                    "name": case["case_id"],
                    "matched_ratio": 1.0,
                    "passed": True,
                    "raw_samples": case["latency_samples_us"],
                    "metrics": {"p50_us": 100.0},
                }
                for case in cases
            ),
        )


class ProposalTests(unittest.TestCase):
    def test_strict_json_and_single_fence(self) -> None:
        raw = json.dumps(_proposal_value())
        proposal = parse_proposal_text(raw, expected_parent_hash=SEED_HASH)
        self.assertEqual(proposal.candidate_hash, SEED_HASH)
        fenced = f"```json\n{raw}\n```"
        self.assertEqual(
            parse_proposal_text(
                fenced, expected_parent_hash=SEED_HASH
            ).candidate_hash,
            SEED_HASH,
        )
        with self.assertRaises(ValueError):
            parse_proposal_text(
                fenced + "\nextra", expected_parent_hash=SEED_HASH
            )
        bad = _proposal_value()
        bad["unknown"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            ProposalV1.from_value(bad, expected_parent_hash=SEED_HASH)

    def test_ndjson_text_and_forbidden_events(self) -> None:
        text = json.dumps(_proposal_value())
        raw = json.dumps({"type": "text", "part": {"text": text}}) + "\n"
        self.assertEqual(
            parse_opencode_ndjson(
                raw, expected_parent_hash=SEED_HASH
            ).candidate_hash,
            SEED_HASH,
        )
        tool = json.dumps({"type": "tool_use", "name": "bash"})
        with self.assertRaisesRegex(ValueError, "tool"):
            parse_opencode_ndjson(tool, expected_parent_hash=SEED_HASH)
        error = json.dumps({"type": "error", "message": "provider failed"})
        with self.assertRaisesRegex(ValueError, "error"):
            parse_opencode_ndjson(error, expected_parent_hash=SEED_HASH)
        with self.assertRaisesRegex(ValueError, "not JSON"):
            parse_opencode_ndjson("noise", expected_parent_hash=SEED_HASH)

    def test_parent_limits_and_empty_source(self) -> None:
        value = _proposal_value()
        value["parent_candidate_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            ProposalV1.from_value(value, expected_parent_hash=SEED_HASH)
        value = _proposal_value()
        value["kernel_source"] = ""
        with self.assertRaisesRegex(ValueError, "non-empty"):
            ProposalV1.from_value(value, expected_parent_hash=SEED_HASH)
        value = _proposal_value()
        value["kernel_source"] = "x" * (256 * 1024 + 1)
        with self.assertRaisesRegex(ValueError, "256 KiB"):
            ProposalV1.from_value(value, expected_parent_hash=SEED_HASH)

    def test_proposal_field_types_and_text_limits_fail_closed(self) -> None:
        fixtures = (
            ("schema_version", True),
            ("parent_candidate_hash", "ABC"),
            ("hypothesis", ""),
            ("hypothesis", "x" * 1001),
            ("rationale", ""),
            ("rationale", "x" * 8001),
            ("kernel_source", 123),
        )
        for field, invalid in fixtures:
            with self.subTest(field=field):
                value = _proposal_value()
                value[field] = invalid
                with self.assertRaises(ValueError):
                    ProposalV1.from_value(
                        value, expected_parent_hash=SEED_HASH
                    )
        value = _proposal_value()
        del value["rationale"]
        with self.assertRaisesRegex(ValueError, "missing"):
            ProposalV1.from_value(value, expected_parent_hash=SEED_HASH)
        proposal = ProposalV1.from_value(
            _proposal_value(), expected_parent_hash=SEED_HASH
        )
        self.assertNotIn("kernel_source", proposal.to_dict(include_source=False))


class PolicyTests(unittest.TestCase):
    def test_seed_passes_and_capability_expansion_fails(self) -> None:
        self.assertTrue(validate_research_candidate(SEED).valid)
        dangerous_import = SEED.replace(
            "import torch", "import torch\nimport subprocess"
        )
        codes = {
            item.code
            for item in validate_research_candidate(dangerous_import).errors
        }
        self.assertIn("IMPORT_NOT_ALLOWED", codes)
        dangerous_call = SEED.replace(
            '"""Editable Triton seed', 'open("/tmp/x", "w")\n"""Editable Triton seed'
        )
        codes = {
            item.code for item in validate_research_candidate(dangerous_call).errors
        }
        self.assertIn("MODULE_SIDE_EFFECT", codes)
        self.assertIn("DANGEROUS_BUILTIN", codes)
        unsafe_address = SEED.replace(
            "expert_id.to(tl.int64) * stride_be", "expert_id * stride_be"
        )
        codes = {
            item.code for item in validate_research_candidate(unsafe_address).errors
        }
        self.assertIn("INT64_B_BASE_REQUIRED", codes)


class ConfigAndArgvTests(unittest.TestCase):
    def test_config_is_strict_and_requires_digest_paths_and_secret_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = _config(root)
            value = config.redacted_dict()
            value.update(
                {
                    "gpu_devices": [
                        "/dev/mxcd",
                        "/dev/dri/card2",
                        "/dev/dri/renderD129",
                    ],
                    "container_uid": 1000,
                    "container_gid": 1000,
                    "video_gid": 44,
                    "proposer_cpus": 2,
                    "proposer_memory": "2g",
                    "evaluator_cpus": 8,
                    "evaluator_memory": "24g",
                }
            )
            path = root / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            loaded = ControllerConfig.load(path)
            self.assertEqual(loaded.expected_kernel_hash, SEED_HASH)
            for field, unsafe in (
                ("max_candidates", 6),
                ("max_hours", 7),
                ("max_consecutive_failures", 4),
                ("proposer_timeout_sec", 1201),
                ("proposer_max_output_bytes", 2 * 1024 * 1024 + 1),
                ("evaluator_timeout_sec", 2701),
                ("proposer_cpus", 3),
                ("evaluator_cpus", 9),
            ):
                with self.subTest(field=field):
                    unsafe_value = dict(value)
                    unsafe_value[field] = unsafe
                    path.write_text(
                        json.dumps(unsafe_value), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "maximum"):
                        ControllerConfig.load(path)
            loaded.deepseek_key_file.chmod(0o644)
            self.assertIn(
                "DeepSeek key file mode must be exactly 0600",
                loaded.validate_host(),
            )
            value["proposer_image"] = "local/proposer:latest"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable"):
                ControllerConfig.load(path)

    def test_docker_argv_enforces_separate_capability_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = _config(root)
            opencode_config = root / "opencode.json"
            write_opencode_config(opencode_config, config)
            proposal_args = proposer_argv(
                config,
                name="proposal-name",
                run_id="run",
                opencode_config=opencode_config,
            )
            joined = " ".join(proposal_args)
            self.assertIn("--interactive", proposal_args)
            self.assertLess(
                proposal_args.index("--interactive"),
                proposal_args.index(config.proposer_image),
            )
            self.assertNotIn(str(config.repository_dir), joined)
            self.assertNotIn("/dev/mxcd", joined)
            self.assertNotIn("docker.sock", joined)
            self.assertNotIn("DEEPSEEK_API_KEY=", joined)
            self.assertIn("/run/secrets/deepseek_api_key", joined)
            self.assertIn("OPENCODE_DISABLE_DEFAULT_PLUGINS=1", joined)
            self.assertIn("OPENCODE_DISABLE_LSP_DOWNLOAD=1", joined)
            self.assertIn("OPENCODE_DISABLE_MODELS_FETCH=1", joined)
            candidate = root / "candidate.py"
            baseline = root / "baseline.py"
            candidate.write_text(SEED, encoding="utf-8")
            baseline.write_text(SEED, encoding="utf-8")
            eval_args = evaluator_argv(
                config,
                name="eval-name",
                run_id="run",
                candidate_path=candidate,
                suite="full",
                baseline_path=baseline,
                cache_dir=config.evaluator_cache_dir / "candidate-a",
            )
            joined = " ".join(eval_args)
            self.assertIn("--network none", joined)
            self.assertNotIn(str(config.state_dir), joined)
            self.assertNotIn("controller.sqlite3", joined)
            self.assertNotIn("docker.sock", joined)
            for device in config.gpu_devices:
                self.assertIn(f"{device}:{device}", joined)
            self.assertIn("/candidate/kernel.py", joined)
            self.assertIn("/baseline/kernel.py", joined)
            opencode = json.loads(opencode_config.read_text(encoding="utf-8"))
            self.assertEqual(opencode["permission"], {"*": "deny"})
            self.assertNotIn("subagent_depth", opencode)
            self.assertEqual(opencode["share"], "disabled")
            self.assertFalse(opencode["formatter"])
            self.assertFalse(opencode["lsp"])
            self.assertTrue(opencode["tools"])
            self.assertTrue(
                all(enabled is False for enabled in opencode["tools"].values())
            )
            proposer = opencode["agent"]["kernel-proposer"]
            self.assertEqual(OPENCODE_PROPOSER_STEPS, 3)
            self.assertEqual(proposer["steps"], OPENCODE_PROPOSER_STEPS)
            self.assertEqual(proposer["permission"], {"*": "deny"})

    def test_config_rejects_capability_and_path_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _config(root).redacted_dict()
            base["gpu_devices"] = [
                "/dev/mxcd",
                "/dev/dri/card2",
                "/dev/dri/renderD129",
            ]
            path = root / "config-negative.json"

            def rejected(change) -> None:
                value = dict(base)
                change(value)
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    ControllerConfig.load(path)

            fixtures = (
                lambda value: value.update(schema_version=2),
                lambda value: value.update(unknown=True),
                lambda value: value.pop("repository_dir"),
                lambda value: value.update(gpu_devices=[]),
                lambda value: value.update(expected_git_commit="bad"),
                lambda value: value.update(expected_kernel_hash="bad"),
                lambda value: value.update(stop_after_promotion=False),
                lambda value: value.update(
                    acknowledge_gpu_passthrough_risk="yes"
                ),
                lambda value: value.update(proposer_memory="3g"),
                lambda value: value.update(evaluator_memory="25g"),
                lambda value: value.update(
                    opencode_model="other/model"
                ),
                lambda value: value.update(
                    controller_dir=value["repository_dir"]
                ),
                lambda value: value.update(docker_binary="relative/docker"),
                lambda value: value.update(container_uid=-1),
            )
            for index, change in enumerate(fixtures):
                with self.subTest(index=index):
                    rejected(change)

    def test_host_validation_reports_secret_and_device_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary).resolve())
            errors = config.validate_host()
            self.assertTrue(
                any("history.sqlite3" in error for error in errors)
            )
            self.assertTrue(
                any("not executable" in error for error in errors)
            )
            self.assertTrue(
                any("not a character device" in error for error in errors)
            )

            config.deepseek_key_file.unlink()
            self.assertTrue(
                any(
                    "key file is missing" in error
                    for error in config.validate_host()
                )
            )
            config.deepseek_key_file.write_bytes(b"")
            self.assertTrue(
                any("must not be empty" in error for error in config.validate_host())
            )
            config.deepseek_key_file.write_bytes(b"x" * 513)
            self.assertTrue(
                any(
                    "unexpectedly large" in error
                    for error in config.validate_host()
                )
            )
            config.deepseek_key_file.write_bytes(b"secret\n")
            self.assertTrue(
                any("without whitespace" in error for error in config.validate_host())
            )
            config.deepseek_key_file.write_bytes(b"\x01secret")
            self.assertTrue(
                any("printable ASCII" in error for error in config.validate_host())
            )
            self.assertFalse(
                any(
                    "key file" in error
                    for error in config.validate_host(require_secret=False)
                )
            )


class RawCliAndStoreTests(unittest.TestCase):
    def test_command_runner_enforces_output_limit_after_fast_exit(self) -> None:
        result = CommandRunner().run(
            [sys.executable, "-c", "print('x' * 4096)"],
            input_text=None,
            timeout_sec=5,
            max_output_bytes=1024,
        )
        self.assertTrue(result.output_limited)
        self.assertLessEqual(len(result.stdout.encode("utf-8")), 1025)

    def test_controller_imports_without_numpy_torch_or_triton(self) -> None:
        script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'triton'}:
        raise AssertionError('forbidden import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import kernel_research.autorun.cli
print('CONTROLLER_IMPORT_OK')
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.stdout.strip(), "CONTROLLER_IMPORT_OK")

    def test_evaluate_raw_never_creates_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "kernel.py"
            candidate.write_text(SEED, encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = evaluator_main(
                    [
                        "evaluate-raw",
                        "--backend",
                        "mock",
                        "--suite",
                        "smoke",
                        "--candidate",
                        str(candidate),
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["command"], "evaluate-raw")
            self.assertFalse((root / ".autoresearch").exists())
            self.assertFalse(any(root.rglob("*.sqlite3")))

    def test_controller_store_restarts_backs_up_and_locks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "controller.sqlite3"
            with ControllerStore(database) as store:
                store.create_run(
                    run_id="run",
                    deadline_epoch=time.time() + 60,
                    config={"expected_git_commit": "x"},
                    initial_best_hash=SEED_HASH,
                )
                iteration = store.create_iteration("run", 1, SEED_HASH)
                store.update_iteration(
                    iteration["id"],
                    candidate_hash="1" * 64,
                    stage="POLICY",
                )
                store.update_iteration(iteration["id"], stage="SMOKE")
                store.backup_to(root / "backup.sqlite3")
            with ControllerStore(database) as restarted:
                self.assertEqual(
                    restarted.latest_iteration("run")["candidate_hash"], "1" * 64
                )
            with ControllerStore(root / "backup.sqlite3") as backup:
                self.assertEqual(backup.get_run("run")["status"], "RUNNING")
            lock_path = root / "gpu.lock"
            with gpu_lock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "already held"):
                    with gpu_lock(lock_path):
                        pass

    def test_checkpoint_contains_both_databases_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            _baseline(config.state_dir)
            controller = ResearchController(config, evaluator=FakeEvaluator())
            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id="checkpoint-run",
                    deadline_epoch=time.time() + 60,
                    config=config.redacted_dict(),
                    initial_best_hash=SEED_HASH,
                )
            run_dir = (
                config.controller_dir / "runs" / "checkpoint-run"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "prompt.txt").write_text(
                "fixture prompt", encoding="utf-8"
            )
            payload = controller.checkpoint("checkpoint-run")
            destination = Path(payload["path"])
            self.assertTrue((destination / "controller.sqlite3").is_file())
            self.assertTrue((destination / "history.sqlite3").is_file())
            self.assertTrue((destination / "manifest.json").is_file())
            self.assertTrue((destination / "run" / "prompt.txt").is_file())
            self.assertTrue(
                (destination / "artifacts" / f"{SEED_HASH}.py").is_file()
            )
            manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["databases"]["controller"]["integrity_check"],
                "ok",
            )
            self.assertEqual(
                manifest["databases"]["history"]["integrity_check"],
                "ok",
            )
            self.assertEqual(manifest["artifact_count"], 1)
            for file_record in manifest["files"]:
                file_path = destination / file_record["path"]
                self.assertEqual(
                    hashlib.sha256(file_path.read_bytes()).hexdigest(),
                    file_record["sha256"],
                )
            with ControllerStore(
                destination / "controller.sqlite3"
            ) as restored:
                self.assertEqual(
                    restored.get_run("checkpoint-run")["status"],
                    "RUNNING",
                )

    def test_controller_store_accepts_concurrent_wal_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            with ControllerStore(database) as store:
                store.create_run(
                    run_id="concurrent",
                    deadline_epoch=time.time() + 60,
                    config={},
                    initial_best_hash=SEED_HASH,
                )

            def write_events(worker: int) -> None:
                with ControllerStore(database) as store:
                    for index in range(10):
                        store.add_event(
                            "concurrent",
                            "CONCURRENT_FIXTURE",
                            {"worker": worker, "index": index},
                        )

            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(write_events, range(4)))
            with ControllerStore(database) as store:
                events = [
                    event
                    for event in store.list_events("concurrent")
                    if event["event"] == "CONCURRENT_FIXTURE"
                ]
            self.assertEqual(len(events), 40)


class StaticProposer(Proposer):
    def __init__(self, proposal: ProposalV1) -> None:
        self.proposal = proposal

    def propose(self, request: ProposalRequest) -> ProposalV1:
        if request.parent_candidate_hash != self.proposal.parent_candidate_hash:
            raise AssertionError("unexpected parent")
        return self.proposal


class StepLimitProposer(Proposer):
    def propose(self, request: ProposalRequest) -> ProposalV1:
        raise ProposerStepLimitError(
            "PROPOSER_STEP_LIMIT: OpenCode exhausted its configured "
            "3-step proposal budget"
        )


class FakeEvaluator:
    def __init__(self, *, interrupt_confirmation: bool = False) -> None:
        self.stages: list[str] = []
        self.interrupt_confirmation = interrupt_confirmation

    def container_name(self, run_id: str, iteration_index: int, stage: str) -> str:
        return f"fake-{run_id}-{iteration_index}-{stage}"

    def doctor(self, *, run_id: str) -> dict:
        return {
            "status": "SUCCESS",
            "environment": {"compile_probe_status": "PASSED"},
        }

    def evaluate(
        self,
        *,
        candidate_path: Path,
        suite: str,
        baseline_path: Path | None,
        run_id: str,
        iteration_index: int,
        stage: str,
    ) -> dict:
        self.stages.append(stage)
        if stage == "confirmation" and self.interrupt_confirmation:
            self.interrupt_confirmation = False
            raise KeyboardInterrupt
        suite_cases = {
            "smoke": ("smoke_gate_up", "smoke_down"),
            "quick": (
                "quick_decode_gate_up",
                "quick_prefill_gate_up",
                "quick_decode_down",
                "quick_prefill_down",
            ),
        }
        if suite != "full":
            return {
                "status": "SUCCESS",
                "cases": [
                    {
                        "case_id": name,
                        "status": "SUCCESS",
                        "matched_ratio": 1.0,
                        "latency_samples_us": [10.0] * 30,
                        "p50_us": 10.0,
                    }
                    for name in suite_cases[suite]
                ],
                "environment": {},
                "error": None,
            }
        cases = [
            {
                "case_id": name,
                "status": "SUCCESS",
                "matched_ratio": 1.0,
                "latency_samples_us": [90.0] * 30,
                "baseline_latency_samples_us": [100.0] * 30,
                "p50_us": 90.0,
                "baseline_p50_us": 100.0,
                "error": None,
            }
            for name in FULL_CASES
        ]
        return {
            "status": "SUCCESS",
            "cases": cases,
            "environment": {},
            "error": None,
        }


class RejectingEvaluator(FakeEvaluator):
    def __init__(self, status: str = "PRECISION_FAILED", error: str = "fixture") -> None:
        super().__init__()
        self.status = status
        self.error = error

    def evaluate(self, **values):
        self.stages.append(str(values["stage"]))
        return {
            "status": self.status,
            "cases": [],
            "environment": {},
            "error": self.error,
        }


class StageFailureEvaluator(FakeEvaluator):
    def __init__(self, failure_stage: str, status: str) -> None:
        super().__init__()
        self.failure_stage = failure_stage
        self.failure_status = status

    def evaluate(self, **values):
        if values["stage"] == self.failure_stage:
            self.stages.append(str(values["stage"]))
            return {
                "status": self.failure_status,
                "cases": [],
                "environment": {},
                "error": "fixture stage failure",
            }
        return super().evaluate(**values)


class VariableFullEvaluator(FakeEvaluator):
    def __init__(self, primary_us: float, confirmation_us: float) -> None:
        super().__init__()
        self.primary_us = primary_us
        self.confirmation_us = confirmation_us

    def evaluate(self, **values):
        if values["suite"] != "full":
            return super().evaluate(**values)
        stage = str(values["stage"])
        self.stages.append(stage)
        candidate_us = (
            self.confirmation_us
            if stage == "confirmation"
            else self.primary_us
        )
        return {
            "status": "SUCCESS",
            "cases": [
                {
                    "case_id": name,
                    "status": "SUCCESS",
                    "matched_ratio": 1.0,
                    "latency_samples_us": [candidate_us] * 30,
                    "baseline_latency_samples_us": [100.0] * 30,
                    "p50_us": candidate_us,
                    "baseline_p50_us": 100.0,
                    "error": None,
                }
                for name in FULL_CASES
            ],
            "environment": {},
            "error": None,
        }


class NoopRunner:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def remove_exact_container(self, docker_binary: Path, container_name: str) -> None:
        self.removed.append(container_name)


class StateMachineTests(unittest.TestCase):
    def _controller(
        self, root: Path, evaluator: FakeEvaluator
    ) -> tuple[ResearchController, ControllerConfig, ProposalV1]:
        config = _config(root, max_candidates=1)
        (config.repository_dir / "program.md").write_text(
            "Change only kernel.py.", encoding="utf-8"
        )
        _baseline(config.state_dir)
        source = _with_different_block_size_n(SEED)
        proposal = ProposalV1.from_value(
            _proposal_value(source), expected_parent_hash=SEED_HASH
        )
        controller = ResearchController(
            config,
            evaluator=evaluator,
            runner=NoopRunner(),
            proposer_factory=lambda run_id, index, run_dir: StaticProposer(
                proposal
            ),
        )
        return controller, config, proposal

    def _create_run(
        self, controller: ResearchController, config: ControllerConfig
    ) -> str:
        run_id = "a" * 32
        with ControllerStore(controller.controller_db) as store:
            store.create_run(
                run_id=run_id,
                deadline_epoch=time.time() + 3600,
                config=config.redacted_dict(),
                initial_best_hash=SEED_HASH,
            )
        return run_id

    def test_full_state_machine_stops_after_confirmed_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = FakeEvaluator()
            controller, config, proposal = self._controller(
                Path(temporary), evaluator
            )
            run_id = self._create_run(controller, config)
            result = controller._run_loop(run_id)
            self.assertEqual(result["status"], "PROMOTED")
            self.assertEqual(result["final_best_hash"], proposal.candidate_hash)
            self.assertIn(
                proposal.candidate_hash,
                controller._resume_allowed_best_hashes(run_id),
            )
            self.assertEqual(
                evaluator.stages,
                ["smoke", "quick", "full_primary", "confirmation"],
            )
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
            self.assertEqual(iteration["outcome"], "PROMOTED")
            with HistoryStore(
                config.state_dir / "history.sqlite3",
                state_dir=config.state_dir,
            ) as history:
                records = history.find_by_candidate_hash(proposal.candidate_hash)
            self.assertEqual(len(records), 4)
            self.assertTrue(records[-1].promotable)
            self.assertEqual(
                hashlib.sha256(
                    (config.repository_dir / "program.md").read_bytes()
                ).hexdigest(),
                hashlib.sha256(b"Change only kernel.py.").hexdigest(),
            )

    def test_proposal_only_stops_before_any_candidate_gpu_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = FakeEvaluator()
            controller, config, _ = self._controller(
                Path(temporary), evaluator
            )
            run_id = self._create_run(controller, config)
            result = controller._run_loop(run_id, proposal_only=True)
            self.assertEqual(result["status"], "PROPOSAL_READY")
            self.assertEqual(evaluator.stages, [])
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
            self.assertEqual(iteration["outcome"], "PROPOSAL_VALIDATED")

    def test_primary_resume_runs_same_hash_confirmation_without_reproposal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = FakeEvaluator(interrupt_confirmation=True)
            controller, config, _ = self._controller(Path(temporary), evaluator)
            run_id = self._create_run(controller, config)
            with self.assertRaises(KeyboardInterrupt):
                controller._run_loop(run_id)
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
                self.assertEqual(iteration["stage"], "CONFIRMATION")
                self.assertIsNotNone(iteration["candidate_hash"])
            result = controller._run_loop(run_id)
            self.assertEqual(result["status"], "PROMOTED")
            self.assertEqual(evaluator.stages.count("smoke"), 1)
            self.assertEqual(evaluator.stages.count("confirmation"), 2)

    def test_history_controller_non_atomic_window_is_reconciled_by_note(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = FakeEvaluator()
            controller, config, proposal = self._controller(
                Path(temporary), evaluator
            )
            run_id = self._create_run(controller, config)
            run_dir = config.controller_dir / "runs" / run_id
            candidate_path = run_dir / "candidates" / f"{proposal.candidate_hash}.py"
            candidate_path.parent.mkdir(parents=True)
            candidate_path.write_text(proposal.kernel_source, encoding="utf-8")
            with ControllerStore(controller.controller_db) as store:
                iteration = store.create_iteration(run_id, 1, SEED_HASH)
                store.update_iteration(
                    iteration["id"],
                    stage="POLICY",
                    candidate_hash=proposal.candidate_hash,
                    candidate_path=str(candidate_path),
                    hypothesis=proposal.hypothesis,
                    rationale=proposal.rationale,
                )
                store.update_iteration(iteration["id"], stage="SMOKE")
                store.update_run(run_id, valid_candidates=1)
            raw = {
                "schema_version": 1,
                "command": "evaluate-raw",
                "backend": "c500",
                "suite": "smoke",
                "candidate_hash": proposal.candidate_hash,
                "baseline_candidate_hash": None,
                "status": "SUCCESS",
                "cases": [
                    {
                        "case_id": name,
                        "status": "SUCCESS",
                        "matched_ratio": 1.0,
                        "latency_samples_us": [10.0] * 30,
                    }
                    for name in ("smoke_gate_up", "smoke_down")
                ],
                "environment": {},
                "error": None,
            }
            note = f"autorun:{run_id}:1:smoke:{proposal.candidate_hash}"
            smoke_record = record_external_result(
                candidate_source=proposal.kernel_source,
                result=raw,
                backend="c500",
                suite="smoke",
                state_dir=config.state_dir,
                note=note,
            )
            result = controller._run_loop(run_id)
            self.assertEqual(result["status"], "PROMOTED")
            self.assertNotIn("smoke", evaluator.stages)
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
            self.assertEqual(
                iteration["experiment_ids"]["smoke"], smoke_record.id
            )

    def test_five_fixture_candidates_archive_and_stop_at_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, max_candidates=5)
            (config.repository_dir / "program.md").write_text(
                "Change only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)
            sources = [
                source
                for tile in (16, 32, 64, 128, 256, 512)
                if (source := _with_block_size_n(SEED, tile)) != SEED
            ][:5]
            self.assertEqual(len(sources), 5)
            proposals = {}
            for index, source in enumerate(sources, 1):
                proposals[index] = ProposalV1.from_value(
                    _proposal_value(source), expected_parent_hash=SEED_HASH
                )
            evaluator = RejectingEvaluator()
            controller = ResearchController(
                config,
                evaluator=evaluator,
                proposer_factory=lambda run_id, index, run_dir: StaticProposer(
                    proposals[index]
                ),
            )
            run_id = self._create_run(controller, config)
            before = hashlib.sha256((ROOT / "kernel.py").read_bytes()).hexdigest()
            result = controller._run_loop(run_id)
            after = hashlib.sha256((ROOT / "kernel.py").read_bytes()).hexdigest()
            self.assertEqual(result["status"], "BUDGET_EXHAUSTED")
            self.assertEqual(before, after)
            with ControllerStore(controller.controller_db) as store:
                iterations = store.list_iterations(run_id)
            self.assertEqual(len(iterations), 5)
            self.assertTrue(
                all(item["outcome"] == "PRECISION_FAILED" for item in iterations)
            )
            self.assertEqual(evaluator.stages, ["smoke"] * 5)
            with HistoryStore(
                config.state_dir / "history.sqlite3",
                state_dir=config.state_dir,
            ) as history:
                self.assertEqual(len(history.list_experiments()), 6)

    def test_hard_gpu_fault_terminates_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = RejectingEvaluator(
                status="COMPILE_ERROR",
                error="kernel caused Xnack ATU address translation fault",
            )
            controller, config, _ = self._controller(
                Path(temporary), evaluator
            )
            run_id = self._create_run(controller, config)
            result = controller._run_loop(run_id)
            self.assertEqual(result["status"], "HARD_FAILED")
            self.assertEqual(evaluator.stages, ["smoke"])

    def test_quick_compile_failure_and_full_performance_rejection(self) -> None:
        for evaluator, expected_stages, expected_outcome in (
            (
                StageFailureEvaluator("quick", "COMPILE_ERROR"),
                ["smoke", "quick"],
                "COMPILE_ERROR",
            ),
            (
                VariableFullEvaluator(99.5, 99.5),
                ["smoke", "quick", "full_primary"],
                "FULL_REJECTED",
            ),
        ):
            with self.subTest(expected_outcome=expected_outcome):
                with tempfile.TemporaryDirectory() as temporary:
                    controller, config, _ = self._controller(
                        Path(temporary), evaluator
                    )
                    run_id = self._create_run(controller, config)
                    result = controller._run_loop(run_id)
                    self.assertEqual(result["status"], "BUDGET_EXHAUSTED")
                    self.assertEqual(evaluator.stages, expected_stages)
                    with ControllerStore(controller.controller_db) as store:
                        iteration = store.latest_iteration(run_id)
                    self.assertEqual(iteration["outcome"], expected_outcome)

    def test_failed_confirmation_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = VariableFullEvaluator(90.0, 105.0)
            controller, config, _ = self._controller(
                Path(temporary), evaluator
            )
            run_id = self._create_run(controller, config)
            result = controller._run_loop(run_id)
            self.assertEqual(result["status"], "BUDGET_EXHAUSTED")
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
            self.assertEqual(iteration["outcome"], "CONFIRMATION_REJECTED")
            self.assertEqual(
                evaluator.stages,
                ["smoke", "quick", "full_primary", "confirmation"],
            )

    def test_three_consecutive_proposer_failures_stop_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, max_candidates=5)
            (config.repository_dir / "program.md").write_text(
                "Change only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)
            evaluator = FakeEvaluator()
            controller = ResearchController(
                config,
                evaluator=evaluator,
                proposer_factory=lambda run_id, index, run_dir: StepLimitProposer(),
            )
            run_id = self._create_run(controller, config)
            result = controller._run_loop(run_id)
            self.assertEqual(result["status"], "BUDGET_EXHAUSTED")
            self.assertIn("failure budget", result["stop_reason"])
            with ControllerStore(controller.controller_db) as store:
                iterations = store.list_iterations(run_id)
            self.assertEqual(len(iterations), 3)
            self.assertTrue(
                all(item["outcome"] == "PROPOSER_ERROR" for item in iterations)
            )
            self.assertTrue(
                all(item["candidate_hash"] is None for item in iterations)
            )
            self.assertTrue(
                all("PROPOSER_STEP_LIMIT:" in item["error"] for item in iterations)
            )
            self.assertEqual(result["valid_candidates"], 0)
            self.assertEqual(evaluator.stages, [])

    def test_duplicate_accepted_hash_never_reaches_gpu_or_candidate_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, max_candidates=5)
            (config.repository_dir / "program.md").write_text(
                "Change only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)
            proposal = ProposalV1.from_value(
                _proposal_value(), expected_parent_hash=SEED_HASH
            )
            evaluator = FakeEvaluator()
            controller = ResearchController(
                config,
                evaluator=evaluator,
                proposer_factory=lambda run_id, index, run_dir: StaticProposer(
                    proposal
                ),
            )
            run_id = self._create_run(controller, config)
            result = controller._run_loop(run_id)
            self.assertEqual(result["status"], "BUDGET_EXHAUSTED")
            self.assertEqual(result["valid_candidates"], 0)
            self.assertEqual(evaluator.stages, [])
            with ControllerStore(controller.controller_db) as store:
                iterations = store.list_iterations(run_id)
            self.assertEqual([item["outcome"] for item in iterations], [
                "DUPLICATE",
                "DUPLICATE",
                "DUPLICATE",
            ])


if __name__ == "__main__":
    unittest.main()
