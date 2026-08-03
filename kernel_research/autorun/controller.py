"""Durable, staged autonomous research controller."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Protocol
import uuid

from ..constants import (
    DOCTOR_OUTPUT_LIMIT_BYTES,
    DOCTOR_TIMEOUT_SEC,
    EVALUATOR_OUTPUT_LIMIT_BYTES,
    MAX_FEEDBACK_CANDIDATES,
)
from ..evaluation import record_external_result
from ..history import ExperimentRecord, HistoryStore
from ..research_policy import validate_research_candidate_bounded
from .errors import ControlledRuntimeError
from .models import ControllerConfig
from .opencode import OpenCodeProposer
from .proposal import ProposalRequest, Proposer
from .runtime import (
    CommandRunner,
    evaluator_argv,
    evaluator_doctor_argv,
    parse_json_output,
)
from .states import RunStatus, Stage, TERMINAL_RUN_STATUSES
from .store import ControllerStore
from .summary import feedback_for_iteration, summarize_result


HARD_STATUSES = frozenset({"CRASH", "TIMEOUT", "UNSUPPORTED_ENV"})
FATAL_GPU_MARKERS = (
    "atu fault",
    "atu address translation",
    "xnack",
    "illegal memory access",
    "mcerrorillegaladdress",
)


class ProposerFailure(ControlledRuntimeError):
    pass


class ControllerSignal(KeyboardInterrupt):
    """A catchable operator signal that must close the active run."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"controller received signal {signum}")


class RawEvaluator(Protocol):
    def doctor(self, *, run_id: str) -> dict[str, Any]:
        ...

    def evaluate(
        self,
        *,
        candidate_path: Path,
        suite: str,
        baseline_path: Path | None,
        run_id: str,
        iteration_index: int,
        stage: str,
    ) -> dict[str, Any]:
        ...

    def container_name(
        self, run_id: str, iteration_index: int, stage: str
    ) -> str:
        ...


class DockerEvaluator:
    def __init__(
        self,
        config: ControllerConfig,
        *,
        controller_dir: Path,
        runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.controller_dir = controller_dir
        self.runner = runner or CommandRunner()

    def container_name(
        self, run_id: str, iteration_index: int, stage: str
    ) -> str:
        safe_stage = stage.lower().replace("_", "-")
        return f"kar-eval-{run_id}-{iteration_index:03d}-{safe_stage}"

    def _cache_dir(
        self, *, candidate_hash: str, purpose: str = "candidate"
    ) -> Path:
        evaluator_digest = self.config.evaluator_image.rsplit(
            "@sha256:", 1
        )[1]
        cache_dir = (
            self.config.evaluator_cache_dir
            / evaluator_digest
            / self.config.expected_git_commit
            / candidate_hash
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.chmod(0o700)
        manifest = {
            "schema_version": 1,
            "purpose": purpose,
            "candidate_hash": candidate_hash,
            "evaluator_image": self.config.evaluator_image,
            "framework_commit": self.config.expected_git_commit,
            "cache_dir": str(cache_dir),
        }
        manifest_dir = (
            self.controller_dir
            / "cache-manifests"
            / evaluator_digest
            / self.config.expected_git_commit
        )
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{candidate_hash}.json"
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ControlledRuntimeError(
                    f"evaluator cache manifest is unreadable: {exc}"
                ) from exc
            if existing != manifest:
                raise ControlledRuntimeError(
                    "evaluator cache manifest identity mismatch"
                )
        else:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=manifest_dir,
                prefix=f".{candidate_hash}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(
                    manifest,
                    temporary,
                    sort_keys=True,
                    ensure_ascii=False,
                    indent=2,
                )
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, manifest_path)
            manifest_path.chmod(0o600)
        return cache_dir

    def doctor(self, *, run_id: str) -> dict[str, Any]:
        name = f"kar-doctor-{run_id}"
        cache_dir = self._cache_dir(
            candidate_hash="doctor", purpose="compile-probe"
        )
        result = self.runner.run(
            evaluator_doctor_argv(
                self.config,
                name=name,
                run_id=run_id,
                cache_dir=cache_dir,
            ),
            input_text=None,
            timeout_sec=DOCTOR_TIMEOUT_SEC,
            max_output_bytes=DOCTOR_OUTPUT_LIMIT_BYTES,
            container_name=name,
            docker_binary=self.config.docker_binary,
        )
        if result.timed_out:
            return {"status": "TIMEOUT", "error": "C500 doctor timed out"}
        if result.output_limited:
            return {"status": "CRASH", "error": "C500 doctor output exceeded limit"}
        try:
            payload = parse_json_output(result.stdout)
        except ValueError as exc:
            return {
                "status": "CRASH",
                "error": f"C500 doctor produced invalid output: {exc}",
                "container_exit_code": result.returncode,
            }
        payload["container_exit_code"] = result.returncode
        if result.returncode != 0 and payload.get("status") == "SUCCESS":
            payload["status"] = "CRASH"
            payload["error"] = (
                "C500 doctor reported SUCCESS but its container exited "
                f"{result.returncode}"
            )
        return payload

    def evaluate(
        self,
        *,
        candidate_path: Path,
        suite: str,
        baseline_path: Path | None,
        run_id: str,
        iteration_index: int,
        stage: str,
    ) -> dict[str, Any]:
        name = self.container_name(run_id, iteration_index, stage)
        result_dir = (
            self.controller_dir
            / "runs"
            / run_id
            / "results"
            / f"{iteration_index:03d}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        candidate_hash = _sha256_file(candidate_path)
        cache_dir = self._cache_dir(candidate_hash=candidate_hash)
        command = self.runner.run(
            evaluator_argv(
                self.config,
                name=name,
                run_id=run_id,
                candidate_path=candidate_path,
                suite=suite,
                baseline_path=baseline_path,
                cache_dir=cache_dir,
            ),
            input_text=None,
            timeout_sec=self.config.evaluator_timeout_sec,
            max_output_bytes=EVALUATOR_OUTPUT_LIMIT_BYTES,
            container_name=name,
            docker_binary=self.config.docker_binary,
        )
        (result_dir / f"{stage}.stdout.json").write_text(
            command.stdout, encoding="utf-8"
        )
        (result_dir / f"{stage}.stderr.txt").write_text(
            command.stderr, encoding="utf-8"
        )
        (result_dir / f"{stage}.stdout.json").chmod(0o600)
        (result_dir / f"{stage}.stderr.txt").chmod(0o600)
        combined = (command.stdout + "\n" + command.stderr).lower()
        if command.timed_out:
            return {"status": "TIMEOUT", "error": "evaluator container timed out"}
        if command.output_limited:
            return {
                "status": "CRASH",
                "error": "evaluator container output exceeded limit",
            }
        if command.returncode == 137:
            return {
                "status": "CRASH",
                "error": "evaluator container exited 137 (OOM/forced kill)",
                "container_exit_code": 137,
            }
        try:
            payload = parse_json_output(command.stdout)
        except ValueError as exc:
            payload = {
                "status": "CRASH",
                "error": f"evaluator container produced invalid output: {exc}",
            }
        payload["container_exit_code"] = command.returncode
        if any(marker in combined for marker in FATAL_GPU_MARKERS):
            payload["status"] = "CRASH"
            payload["error"] = (
                "fatal GPU driver marker detected (ATU/Xnack/illegal address); "
                "unattended retry is forbidden"
            )
        elif command.returncode != 0 and payload.get("status") == "SUCCESS":
            payload["status"] = "CRASH"
            payload["error"] = (
                "evaluator reported SUCCESS but its container exited "
                f"{command.returncode}"
            )
        return payload


@contextmanager
def gpu_lock(path: Path) -> Iterator[None]:
    """Hold a cooperative, same-controller GPU lock for a complete session."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControlledRuntimeError(
                "GPU1 controller lock is already held"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix not in {".pyc", ".pyo"}
    ):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = item.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,
    )
    return completed.stdout.strip()


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(source)
    target_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(target_connection)
    finally:
        target_connection.close()
        source_connection.close()


def _sqlite_summary(
    path: Path, *, tables: tuple[str, ...]
) -> dict[str, Any]:
    connection = sqlite3.connect(path)
    try:
        integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
        integrity = [str(row[0]) for row in integrity_rows]
        if integrity != ["ok"]:
            raise ControlledRuntimeError(
                f"checkpoint database integrity failed for {path.name}: "
                + "; ".join(integrity)
            )
        counts = {
            table: int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            )
            for table in tables
        }
        return {
            "sha256": _sha256_file(path),
            "integrity_check": "ok",
            "sqlite_version": sqlite3.sqlite_version,
            "table_counts": counts,
        }
    finally:
        connection.close()


def _checkpoint_file_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "manifest.json"
    ]


class ResearchController:
    def __init__(
        self,
        config: ControllerConfig,
        *,
        evaluator: RawEvaluator | None = None,
        proposer_factory: Callable[
            [str, int, Path], Proposer
        ]
        | None = None,
        runner: CommandRunner | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.runner = runner or CommandRunner()
        self.evaluator = evaluator or DockerEvaluator(
            config, controller_dir=config.controller_dir, runner=self.runner
        )
        self.proposer_factory = proposer_factory or (
            lambda run_id, index, run_dir: OpenCodeProposer(
                config,
                run_id=run_id,
                iteration_index=index,
                run_dir=run_dir,
                runner=self.runner,
            )
        )
        self.clock = clock

    @property
    def controller_db(self) -> Path:
        return self.config.controller_dir / "controller.sqlite3"

    @property
    def history_db(self) -> Path:
        return self.config.state_dir / "history.sqlite3"

    def _best(self) -> ExperimentRecord:
        if not self.history_db.is_file():
            raise ControlledRuntimeError("history database does not exist")
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            best = history.get_best(backend="c500", suite="full")
        if best is None:
            raise ControlledRuntimeError(
                "no accepted C500 full baseline exists"
            )
        return best

    def _verify_repository(
        self, *, allowed_best_hashes: set[str] | None = None
    ) -> dict[str, Any]:
        commit = _run_git(self.config.repository_dir, "rev-parse", "HEAD")
        if commit != self.config.expected_git_commit:
            raise ControlledRuntimeError(
                f"repository HEAD {commit} does not match expected commit"
            )
        status = _run_git(
            self.config.repository_dir,
            "status",
            "--porcelain",
        )
        if status:
            raise ControlledRuntimeError(
                "trusted repository has tracked modifications"
            )
        kernel_path = self.config.repository_dir / "kernel.py"
        kernel_hash = _sha256_file(kernel_path)
        if kernel_hash != self.config.expected_kernel_hash:
            raise ControlledRuntimeError(
                "kernel.py does not match expected baseline hash"
            )
        best = self._best()
        allowed = allowed_best_hashes or {self.config.expected_kernel_hash}
        if best.candidate_hash not in allowed:
            raise ControlledRuntimeError(
                "accepted history baseline is not allowed by this controller run"
            )
        artifact = self.config.state_dir / best.artifact_path
        if not artifact.is_file() or _sha256_file(artifact) != best.candidate_hash:
            raise ControlledRuntimeError(
                "accepted history artifact is missing or corrupted"
            )
        return {
            "git_commit": commit,
            "kernel_hash": kernel_hash,
            "baseline_experiment_id": best.id,
            "baseline_hash": best.candidate_hash,
            "baseline_artifact": str(artifact),
        }

    def _prepare_framework(self) -> Path:
        """Create a commit-keyed evaluator view containing only trusted code."""

        destination = (
            self.config.controller_dir
            / "framework"
            / self.config.expected_git_commit
        )
        package_destination = destination / "kernel_research"
        if package_destination.exists():
            if _tree_hash(package_destination) != _tree_hash(
                self.config.repository_dir / "kernel_research"
            ):
                raise ControlledRuntimeError(
                    "trusted framework snapshot hash mismatch"
                )
            return destination
        source = self.config.repository_dir / "kernel_research"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".tmp-{uuid.uuid4().hex}"
        try:
            shutil.copytree(
                source,
                temporary / "kernel_research",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination

    def _prepare_runtime_dirs(self) -> None:
        for directory in (
            self.config.state_dir,
            self.config.controller_dir,
            self.config.checkpoint_dir,
            self.config.evaluator_cache_dir,
        ):
            if directory == self.config.state_dir:
                if not directory.is_dir():
                    continue
            else:
                directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o700)

    def _inspect_image(self, image: str) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                [
                    str(self.config.docker_binary),
                    "image",
                    "inspect",
                    "--format",
                    "{{json .RepoDigests}}",
                    image,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "image": image,
                "available": False,
                "repo_digests": "",
                "error": str(exc),
            }
        return {
            "image": image,
            "available": completed.returncode == 0,
            "repo_digests": completed.stdout.strip(),
            "error": completed.stderr.strip() or None,
        }

    def doctor(
        self,
        *,
        require_secret: bool = True,
        allowed_best_hashes: set[str] | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        try:
            self._prepare_runtime_dirs()
        except OSError as exc:
            errors.append(f"could not prepare controller runtime directories: {exc}")
        errors.extend(self.config.validate_host(require_secret=require_secret))
        identity: dict[str, Any] | None = None
        try:
            identity = self._verify_repository(
                allowed_best_hashes=allowed_best_hashes
            )
            self._prepare_framework()
        except (
            OSError,
            subprocess.SubprocessError,
            ControlledRuntimeError,
            ValueError,
            sqlite3.DatabaseError,
        ) as exc:
            errors.append(str(exc))
        images = {
            "proposer": self._inspect_image(self.config.proposer_image),
            "evaluator": self._inspect_image(self.config.evaluator_image),
        }
        if not images["proposer"]["available"]:
            errors.append("pinned proposer image is not present locally")
        if not images["evaluator"]["available"]:
            errors.append("pinned evaluator image is not present locally")
        probe: dict[str, Any] | None = None
        if not errors:
            probe = self.evaluator.doctor(run_id="preflight")
            if (
                probe.get("status") != "SUCCESS"
                or probe.get("environment", {}).get("compile_probe_status")
                != "PASSED"
            ):
                errors.append("C500 compile doctor did not pass")
        return {
            "schema_version": 1,
            "command": "doctor",
            "proposer_model": self.config.opencode_model,
            "status": "SUCCESS" if not errors else "FAILED",
            "errors": errors,
            "identity": identity,
            "images": images,
            "c500_probe": probe,
            "security_boundary": (
                "containers isolate ordinary agent mistakes; GPU device access "
                "is not an adversarial driver sandbox"
            ),
            "gpu_passthrough_risk_acknowledged": (
                self.config.acknowledge_gpu_passthrough_risk
            ),
        }

    def _require_gpu_risk_acknowledgement(self) -> None:
        if not self.config.acknowledge_gpu_passthrough_risk:
            raise ControlledRuntimeError(
                "GPU evaluation is disabled until "
                "acknowledge_gpu_passthrough_risk is explicitly true"
            )

    def _proposal_request(self, run_id: str) -> ProposalRequest:
        best = self._best()
        source = (self.config.state_dir / best.artifact_path).read_text(
            encoding="utf-8"
        )
        case_p50 = {
            str(case.get("case_id", case.get("name", "unknown"))): float(
                case["p50_us"]
            )
            for case in best.result.get("cases", [])
            if case.get("p50_us") is not None
        }
        with ControllerStore(self.controller_db) as store:
            recent_items = store.list_recent_scientific_iterations(
                exclude_run_id=run_id,
                limit=MAX_FEEDBACK_CANDIDATES,
            )
            recent = tuple(
                feedback_for_iteration(item)
                for item in reversed(recent_items)
            )
            completed = [
                item
                for item in store.list_iterations(run_id)
                if item["status"] != "RUNNING"
            ][-MAX_FEEDBACK_CANDIDATES:]
            feedback = tuple(feedback_for_iteration(item) for item in completed)
        environment = dict(best.environment)
        environment.update(
            {
                "accepted_experiment_id": best.id,
                "accepted_global_score": best.aggregate_score,
                "evaluator_image": self.config.evaluator_image,
            }
        )
        return ProposalRequest(
            parent_candidate_hash=best.candidate_hash,
            accepted_kernel=source,
            program_markdown=(
                self.config.repository_dir / "program.md"
            ).read_text(encoding="utf-8"),
            environment=environment,
            accepted_case_p50_us=case_p50,
            recent_experiments=recent,
            session_feedback=feedback,
        )

    def _recorded_stage(
        self, *, note: str, candidate_hash: str
    ) -> ExperimentRecord | None:
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            return history.find_by_note_candidate(
                note=note, candidate_hash=candidate_hash
            )

    def _candidate_seen(
        self, store: ControllerStore, candidate_hash: str
    ) -> bool:
        if store.candidate_seen(candidate_hash):
            return True
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            return history.has_candidate(candidate_hash)

    def _resume_allowed_best_hashes(self, run_id: str) -> set[str]:
        allowed = {self.config.expected_kernel_hash}
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            for record in history.list_autorun_experiments(
                run_id=run_id,
                backend="c500",
                suite="full",
                promotable_only=True,
            ):
                promotion = record.result.get("promotion", {})
                if (
                    promotion.get("phase") == "confirmation"
                    and promotion.get("confirmed") is True
                ):
                    allowed.add(record.candidate_hash)
        return allowed

    def _evaluate_stage(
        self,
        *,
        store: ControllerStore,
        run_id: str,
        iteration: Mapping[str, Any],
        source: str,
        candidate_path: Path,
        suite: str,
        stage: str,
    ) -> ExperimentRecord:
        candidate_hash = str(iteration["candidate_hash"])
        note = (
            f"autorun:{run_id}:{iteration['iteration_index']}:{stage}:"
            f"{candidate_hash}"
        )
        reconciled = self._recorded_stage(
            note=note, candidate_hash=candidate_hash
        )
        if reconciled is not None:
            experiment_ids = dict(iteration.get("experiment_ids") or {})
            experiment_ids[stage] = reconciled.id
            store.update_iteration_with_event(
                int(iteration["id"]),
                "STAGE_RECONCILED",
                {"stage": stage, "experiment_id": reconciled.id},
                experiment_ids=experiment_ids,
                result=dict(reconciled.result),
                active_container=None,
            )
            return reconciled
        best = self._best()
        baseline_path = (
            self.config.state_dir / best.artifact_path
            if suite == "full"
            else None
        )
        container_name = self.evaluator.container_name(
            run_id, int(iteration["iteration_index"]), stage
        )
        store.update_iteration_with_event(
            int(iteration["id"]),
            "EVALUATOR_STARTED",
            {"stage": stage, "container": container_name},
            active_container=container_name,
        )
        raw = self.evaluator.evaluate(
            candidate_path=candidate_path,
            suite=suite,
            baseline_path=baseline_path,
            run_id=run_id,
            iteration_index=int(iteration["iteration_index"]),
            stage=stage,
        )
        raw.setdefault("backend", "c500")
        raw.setdefault("suite", suite)
        raw.setdefault("candidate_hash", candidate_hash)
        raw.setdefault("schema_version", 1)
        raw.setdefault("command", "evaluate-raw")
        raw.setdefault(
            "baseline_candidate_hash",
            best.candidate_hash if suite == "full" else None,
        )
        raw_result_path = (
            self.config.controller_dir
            / "runs"
            / run_id
            / "results"
            / f"{int(iteration['iteration_index']):03d}"
            / f"{stage}.validated.json"
        )
        raw_result_path.parent.mkdir(parents=True, exist_ok=True)
        raw_result_path.write_text(
            json.dumps(raw, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raw_result_path.chmod(0o600)
        record = record_external_result(
            candidate_source=source,
            result=raw,
            backend="c500",
            suite=suite,
            state_dir=self.config.state_dir,
            note=note,
            git_revision=self.config.expected_git_commit[:12],
        )
        experiment_ids = dict(iteration.get("experiment_ids") or {})
        experiment_ids[stage] = record.id
        store.update_iteration_with_event(
            int(iteration["id"]),
            "EVALUATOR_RECORDED",
            {"stage": stage, "experiment_id": record.id},
            experiment_ids=experiment_ids,
            result=dict(record.result),
            active_container=None,
        )
        return record

    def _hard_failure(self, record: ExperimentRecord) -> bool:
        if record.status in HARD_STATUSES:
            return True
        error = (record.error_summary or "").lower()
        return any(marker in error for marker in FATAL_GPU_MARKERS)

    def _finish_iteration(
        self,
        store: ControllerStore,
        iteration_id: int,
        *,
        outcome: str,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        store.update_iteration_with_event(
            iteration_id,
            "ITERATION_FINISHED",
            {"outcome": outcome},
            status="COMPLETED",
            stage="DONE",
            outcome=outcome,
            result=dict(result or {}),
            error=error,
            active_container=None,
        )

    def _process_iteration(
        self,
        *,
        store: ControllerStore,
        run_id: str,
        iteration: dict[str, Any],
        run_dir: Path,
        proposal_only: bool = False,
    ) -> str:
        iteration_id = int(iteration["id"])
        if not iteration.get("candidate_hash"):
            request = self._proposal_request(run_id)
            proposer = self.proposer_factory(
                run_id, int(iteration["iteration_index"]), run_dir
            )
            container_name = getattr(proposer, "container_name", None)
            paths = {
                "prompt_path": str(getattr(proposer, "prompt_path", "")) or None,
                "raw_output_path": str(getattr(proposer, "raw_path", "")) or None,
            }
            store.update_iteration(
                iteration_id,
                stage="PROPOSE",
                active_container=container_name,
                **paths,
            )
            try:
                try:
                    proposal = proposer.propose(request)
                finally:
                    updated_paths = {
                        "prompt_path": (
                            str(getattr(proposer, "prompt_path", "")) or None
                        ),
                        "raw_output_path": (
                            str(getattr(proposer, "raw_path", "")) or None
                        ),
                    }
                    store.update_iteration(iteration_id, **updated_paths)
                    attempts = getattr(proposer, "attempts", ())
                    if len(attempts) > 1:
                        store.add_event(
                            run_id,
                            "PROPOSER_FORMAT_RETRY",
                            {"attempts": list(attempts)},
                            iteration_id=iteration_id,
                        )
            except Exception as exc:
                raise ProposerFailure(f"{type(exc).__name__}: {exc}") from exc
            if self._candidate_seen(store, proposal.candidate_hash):
                run = store.get_run(run_id)
                store.update_run(
                    run_id,
                    consecutive_failures=int(run["consecutive_failures"]) + 1,
                )
                self._finish_iteration(
                    store,
                    iteration_id,
                    outcome="DUPLICATE",
                    error="candidate hash was already proposed",
                )
                return "CONTINUE"
            candidate_dir = run_dir / "candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = candidate_dir / f"{proposal.candidate_hash}.py"
            candidate_path.write_text(proposal.kernel_source, encoding="utf-8")
            candidate_path.chmod(0o600)
            iteration = store.accept_candidate(
                iteration_id,
                candidate_hash=proposal.candidate_hash,
                hypothesis=proposal.hypothesis,
                rationale=proposal.rationale,
                candidate_path=str(candidate_path),
            )
        candidate_path = Path(str(iteration["candidate_path"]))
        source = candidate_path.read_text(encoding="utf-8")
        if iteration["stage"] == "POLICY":
            policy = validate_research_candidate_bounded(source)
            if not policy.valid:
                store.update_run(run_id, consecutive_failures=0)
                self._finish_iteration(
                    store,
                    iteration_id,
                    outcome="CONTRACT_ERROR",
                    result=policy.to_dict(),
                    error="; ".join(
                        f"{item.code}: {item.message}" for item in policy.errors
                    ),
                )
                if proposal_only:
                    store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": "proposal policy failure"},
                        status="FAILED",
                        stop_reason="single live proposal failed research policy",
                    )
                    return "STOP"
                return "CONTINUE"
            iteration = store.update_iteration_with_event(
                iteration_id,
                "POLICY_PASSED",
                {"candidate_hash": policy.sha256},
                stage="SMOKE",
            )
            if proposal_only:
                self._finish_iteration(
                    store,
                    iteration_id,
                    outcome="PROPOSAL_VALIDATED",
                    result=policy.to_dict(),
                )
                store.update_run_with_event(
                    run_id,
                    "RUN_FINISHED",
                    {"reason": "proposal validated"},
                    status="PROPOSAL_READY",
                    stop_reason="single live proposal passed policy without GPU evaluation",
                )
                return "STOP"

        stages = (
            ("SMOKE", "smoke"),
            ("QUICK", "quick"),
            ("FULL_PRIMARY", "full"),
            ("CONFIRMATION", "full"),
        )
        for stage, suite in stages:
            current = store.get_iteration(iteration_id)
            if current["stage"] != stage:
                continue
            record = self._evaluate_stage(
                store=store,
                run_id=run_id,
                iteration=current,
                source=source,
                candidate_path=candidate_path,
                suite=suite,
                stage=stage.lower(),
            )
            store.update_run(run_id, consecutive_failures=0)
            if bool(store.get_run(run_id)["stop_requested"]):
                self._finish_iteration(
                    store,
                    iteration_id,
                    outcome="STOPPED",
                    result=record.result,
                    error="operator requested stop during evaluation",
                )
                store.update_run_with_event(
                    run_id,
                    "RUN_FINISHED",
                    {"reason": "operator stop"},
                    status="STOPPED",
                    stop_reason="operator requested stop",
                )
                return "STOP"
            if self._hard_failure(record):
                self._finish_iteration(
                    store,
                    iteration_id,
                    outcome="HARD_FAILURE",
                    result=record.result,
                    error=record.error_summary,
                )
                store.update_run_with_event(
                    run_id,
                    "RUN_FINISHED",
                    {"reason": "hard GPU failure", "stage": stage},
                    status="HARD_FAILED",
                    stop_reason=(
                        f"{stage} produced hard status {record.status}: "
                        f"{record.error_summary or ''}"
                    ).strip(),
                )
                return "STOP"
            if record.status != "SUCCESS":
                self._finish_iteration(
                    store,
                    iteration_id,
                    outcome=record.status,
                    result=record.result,
                    error=record.error_summary,
                )
                return "CONTINUE"
            if stage == "SMOKE":
                store.update_iteration_with_event(
                    iteration_id,
                    "STAGE_ADVANCED",
                    {"from": "SMOKE", "to": "QUICK"},
                    stage="QUICK",
                )
            elif stage == "QUICK":
                store.update_iteration_with_event(
                    iteration_id,
                    "STAGE_ADVANCED",
                    {"from": "QUICK", "to": "FULL_PRIMARY"},
                    stage="FULL_PRIMARY",
                )
            elif stage == "FULL_PRIMARY":
                phase = record.result.get("promotion", {}).get("phase")
                if phase == "primary":
                    store.update_iteration_with_event(
                        iteration_id,
                        "STAGE_ADVANCED",
                        {
                            "from": "FULL_PRIMARY",
                            "to": "CONFIRMATION",
                        },
                        stage="CONFIRMATION",
                    )
                else:
                    self._finish_iteration(
                        store,
                        iteration_id,
                        outcome=f"FULL_{str(phase or 'REJECTED').upper()}",
                        result=record.result,
                    )
                    return "CONTINUE"
            else:
                promoted = bool(
                    record.result.get("promotion", {})
                    .get("decision", {})
                    .get("promoted")
                )
                self._finish_iteration(
                    store,
                    iteration_id,
                    outcome="PROMOTED" if promoted else "CONFIRMATION_REJECTED",
                    result=record.result,
                )
                if promoted:
                    store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": "confirmed promotion"},
                        status="PROMOTED",
                        stop_reason="first confirmation promotion",
                        final_best_hash=record.candidate_hash,
                    )
                    return "STOP"
                return "CONTINUE"
        return "CONTINUE"

    def _terminal_budget_reason(
        self, run: Mapping[str, Any], *, has_pending_iteration: bool
    ) -> str | None:
        if bool(run["stop_requested"]):
            return "operator requested stop"
        if self.clock() >= float(run["deadline_epoch"]):
            return "six-hour wall-clock budget exhausted"
        if (
            not has_pending_iteration
            and int(run["valid_candidates"]) >= self.config.max_candidates
        ):
            return "valid candidate budget exhausted"
        if (
            not has_pending_iteration
            and int(run["consecutive_failures"])
            >= self.config.max_consecutive_failures
        ):
            return "consecutive proposer/controller failure budget exhausted"
        return None

    def _run_loop(
        self, run_id: str, *, proposal_only: bool = False
    ) -> dict[str, Any]:
        run_dir = self.config.controller_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir.chmod(0o700)
        with ControllerStore(self.controller_db) as store:
            while True:
                run = store.get_run(run_id)
                if run["status"] in TERMINAL_RUN_STATUSES:
                    return run
                iteration = store.latest_iteration(run_id)
                if iteration and iteration.get("active_container"):
                    self.runner.remove_exact_container(
                        self.config.docker_binary,
                        str(iteration["active_container"]),
                    )
                    iteration = store.update_iteration_with_event(
                        int(iteration["id"]),
                        "RECOVERY_CONTAINER_CLEANED",
                        {"container": str(iteration["active_container"])},
                        active_container=None,
                    )
                pending = bool(
                    iteration is not None and iteration["status"] == "RUNNING"
                )
                reason = self._terminal_budget_reason(
                    run, has_pending_iteration=pending
                )
                if reason:
                    status = (
                        "STOPPED"
                        if bool(run["stop_requested"])
                        else "BUDGET_EXHAUSTED"
                    )
                    return store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": reason},
                        status=status,
                        stop_reason=reason,
                    )
                if iteration is None or iteration["status"] != "RUNNING":
                    next_index = (
                        1 if iteration is None else int(iteration["iteration_index"]) + 1
                    )
                    iteration = store.create_iteration(
                        run_id, next_index, self._best().candidate_hash
                    )
                try:
                    action = self._process_iteration(
                        store=store,
                        run_id=run_id,
                        iteration=iteration,
                        run_dir=run_dir,
                        proposal_only=proposal_only,
                    )
                except ControllerSignal as exc:
                    current = store.get_iteration(int(iteration["id"]))
                    active = current.get("active_container")
                    if active:
                        self.runner.remove_exact_container(
                            self.config.docker_binary, str(active)
                        )
                    store.update_iteration_with_event(
                        int(iteration["id"]),
                        "CONTROLLER_SIGNAL",
                        {"signum": exc.signum},
                        error=f"controller interrupted by signal {exc.signum}",
                        active_container=None,
                    )
                    return store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": "operator signal", "signum": exc.signum},
                        status=RunStatus.STOPPED.value,
                        stop_reason=f"controller interrupted by signal {exc.signum}",
                    )
                except KeyboardInterrupt:
                    store.update_iteration_with_event(
                        int(iteration["id"]),
                        "CONTROLLER_INTERRUPTED",
                        {"kind": "KeyboardInterrupt"},
                        error="controller interrupted by operator",
                    )
                    raise
                except Exception as exc:
                    current_run = store.get_run(run_id)
                    failures = int(current_run["consecutive_failures"]) + 1
                    self._finish_iteration(
                        store,
                        int(iteration["id"]),
                        outcome=(
                            "PROPOSER_ERROR"
                            if isinstance(exc, ProposerFailure)
                            else "CONTROLLER_ERROR"
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    store.update_run(
                        run_id, consecutive_failures=failures
                    )
                    if proposal_only:
                        store.update_run_with_event(
                            run_id,
                            "RUN_FINISHED",
                            {"reason": "proposal failure"},
                            status="FAILED",
                            stop_reason="single live proposal failed",
                        )
                        return store.get_run(run_id)
                    action = "CONTINUE"
                if action == "STOP":
                    return store.get_run(run_id)
                if proposal_only:
                    store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": "proposal policy failure"},
                        status="FAILED",
                        stop_reason="single live proposal did not pass policy",
                    )
                    return store.get_run(run_id)

    def start(self, *, proposal_only: bool = False) -> dict[str, Any]:
        self._prepare_runtime_dirs()
        if not proposal_only:
            self._require_gpu_risk_acknowledgement()
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            preflight = self.doctor()
            if preflight["status"] != "SUCCESS":
                raise ControlledRuntimeError(
                    "controller preflight failed: "
                    + "; ".join(preflight["errors"])
                )
            run_id = uuid.uuid4().hex
            best = self._best()
            with ControllerStore(self.controller_db) as store:
                store.create_run(
                    run_id=run_id,
                    deadline_epoch=self.clock() + self.config.max_hours * 3600,
                    config=self.config.redacted_dict(),
                    initial_best_hash=best.candidate_hash,
                    preflight=preflight,
                )
            return self._run_loop(run_id, proposal_only=proposal_only)

    def resume(self, run_id: str) -> dict[str, Any]:
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            with ControllerStore(self.controller_db) as store:
                run = store.get_run(run_id)
                if run["status"] in TERMINAL_RUN_STATUSES:
                    return run
                self._require_gpu_risk_acknowledgement()
                stored = run["config"]
                current = self.config.redacted_dict()
                if stored != current:
                    changed = sorted(
                        key
                        for key in set(stored) | set(current)
                        if stored.get(key) != current.get(key)
                    )
                    raise ControlledRuntimeError(
                        "resume config changed immutable fields: "
                        + ", ".join(changed)
                    )
                allowed_best_hashes = self._resume_allowed_best_hashes(run_id)
            preflight = self.doctor(
                allowed_best_hashes=allowed_best_hashes
            )
            if preflight["status"] != "SUCCESS":
                raise ControlledRuntimeError(
                    "controller resume preflight failed: "
                    + "; ".join(preflight["errors"])
                )
            return self._run_loop(run_id)

    def status(self, run_id: str | None = None) -> dict[str, Any]:
        if not self.controller_db.exists():
            return {
                "schema_version": 1,
                "command": "status",
                "status": "NO_RUNS",
                "run": None,
                "iterations": [],
            }
        with ControllerStore(self.controller_db) as store:
            run = store.latest_run() if run_id is None else store.get_run(run_id)
            if run is None:
                return {
                    "schema_version": 1,
                    "command": "status",
                    "status": "NO_RUNS",
                    "run": None,
                    "iterations": [],
                }
            iterations = store.list_iterations(str(run["id"]))
        for item in iterations:
            item["result_summary"] = summarize_result(
                item.get("result"),
                error=item.get("error"),
            )
            item.pop("rationale", None)
            item.pop("result", None)
        return {
            "schema_version": 1,
            "command": "status",
            "status": run["status"],
            "run": run,
            "iterations": iterations,
        }

    def stop(self, run_id: str) -> dict[str, Any]:
        with ControllerStore(self.controller_db) as store:
            run = store.request_stop(run_id)
            iteration = store.latest_iteration(run_id)
            if iteration and iteration.get("active_container"):
                self.runner.remove_exact_container(
                    self.config.docker_binary,
                    str(iteration["active_container"]),
                )
                store.update_iteration_with_event(
                    int(iteration["id"]),
                    "STOP_CONTAINER_CLEANED",
                    {"container": str(iteration["active_container"])},
                    active_container=None,
                )
        return run

    def checkpoint(self, run_id: str) -> dict[str, Any]:
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            destination = self.config.checkpoint_dir / f"{run_id}-{timestamp}"
            temporary_destination = (
                self.config.checkpoint_dir
                / f".tmp-{run_id}-{timestamp}-{uuid.uuid4().hex}"
            )
            if destination.exists():
                raise ControlledRuntimeError(
                    "checkpoint destination already exists"
                )
            temporary_destination.mkdir(parents=True)
            temporary_destination.chmod(0o700)
            try:
                with ControllerStore(self.controller_db) as store:
                    run = store.get_run(run_id)
                    store.backup_to(
                        temporary_destination / "controller.sqlite3"
                    )
                _sqlite_backup(
                    self.history_db,
                    temporary_destination / "history.sqlite3",
                )
                run_dir = self.config.controller_dir / "runs" / run_id
                if run_dir.exists():
                    shutil.copytree(
                        run_dir, temporary_destination / "run"
                    )
                artifacts = self.config.state_dir / "artifacts"
                if artifacts.exists():
                    shutil.copytree(
                        artifacts,
                        temporary_destination / "artifacts",
                    )
                controller_summary = _sqlite_summary(
                    temporary_destination / "controller.sqlite3",
                    tables=("runs", "iterations", "events"),
                )
                history_summary = _sqlite_summary(
                    temporary_destination / "history.sqlite3",
                    tables=("experiments", "case_measurements"),
                )
                controller_connection = sqlite3.connect(
                    temporary_destination / "controller.sqlite3"
                )
                try:
                    backed_up_run = controller_connection.execute(
                        "SELECT COUNT(*) FROM runs WHERE id = ?",
                        (run_id,),
                    ).fetchone()
                    if (
                        backed_up_run is None
                        or int(backed_up_run[0]) != 1
                    ):
                        raise ControlledRuntimeError(
                            "checkpoint does not contain the requested run"
                        )
                finally:
                    controller_connection.close()
                artifact_count = (
                    len(
                        list(
                            (
                                temporary_destination / "artifacts"
                            ).glob("*.py")
                        )
                    )
                    if (temporary_destination / "artifacts").exists()
                    else 0
                )
                manifest = {
                    "schema_version": 2,
                    "run_id": run_id,
                    "run_status": run["status"],
                    "config": self.config.redacted_dict(),
                    "databases": {
                        "controller": controller_summary,
                        "history": history_summary,
                    },
                    "artifact_count": artifact_count,
                    "files": _checkpoint_file_manifest(
                        temporary_destination
                    ),
                }
                (temporary_destination / "manifest.json").write_text(
                    json.dumps(
                        manifest,
                        sort_keys=True,
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_destination, destination)
                return {
                    "schema_version": 2,
                    "command": "checkpoint",
                    "run_id": run_id,
                    "path": str(destination),
                }
            finally:
                if temporary_destination.exists():
                    shutil.rmtree(temporary_destination)
