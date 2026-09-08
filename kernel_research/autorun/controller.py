"""Durable, staged autonomous research controller."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
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
    MAX_EVALUATOR_TIMEOUT_SEC,
    MAX_FEEDBACK_CANDIDATES,
)
from ..evaluation import (
    record_baseline_qualification_result,
    record_current_baseline_bootstrap_result,
    record_external_result,
    record_noise_external_result,
)
from ..history import ExperimentRecord, HistoryStore
from ..noise import SQLITE_MAX_INT, trusted_noise_baseline_source
from ..compiled_reference import scoring_reference_source_sha256
from ..device_timing import device_event_protocol_snapshot
from ..scoring_qualification import (
    SCORING_BASELINE_QUALIFICATION_RUNS,
    ScoringBaselineQualification,
    aggregate_scoring_baseline_probes,
)
from ..scoring_measurement import (
    scoring_baseline_measurement_contract_snapshot,
)
from ..platform.artifacts import ArtifactId
from ..platform.canonical import (
    canonical_json_text,
    canonical_sha256,
    require_sha256_digest,
)
from ..platform.components import TargetComponents
from ..platform.identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
)
from ..platform.proposal import (
    CandidateBundle,
    ProposalV2,
    TRITON_PYTHON_BUNDLE_LIMITS,
)
from ..platform.profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    ProfileRef,
    ResearchNamespace,
)
from .errors import ControlledRuntimeError
from .deployment import (
    DEPLOYMENT_BASELINE_FILENAME,
    DeploymentBaselinePin,
    deployment_runtime_root,
)
from .model_catalog import resolve_opencode_model
from .models import ControllerConfig
from .opencode import OpenCodeProposer
from .proposal import ProposalRequest, Proposer, build_prompt
from .runtime import (
    CommandRunner,
    evaluator_argv,
    evaluator_doctor_argv,
    parse_json_output,
    scoring_baseline_probe_argv,
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
SCORING_PRE_GPU_FAILURE_CLASS = "KNOWN_PRE_GPU_CLI_REJECTION"
SCORING_PRE_GPU_ERROR = (
    "scoring baseline evaluator produced invalid output: container did not emit "
    "one JSON object: Expecting value: line 1 column 1 (char 0)"
)
SCORING_OOM_SOURCE_COMMIT = "e099b2b415150c39c9e6d0cab7a9bded4256fafa"
SCORING_OOM_OPERATION_ID = "score-baseline-16bf9ae5a89201250c2dbb20"
SCORING_OOM_OPERATION_DIGEST = (
    "sha256:16bf9ae5a89201250c2dbb207bef22557faaf967320fdddbe0fce9ea86b51849"
)
SCORING_OOM_MEASUREMENT_DIGEST = (
    "sha256:3089ed4feee4e9a09c536ff360bf78d0845afaaf00e6dc1c0a7353e23502e4fe"
)
SCORING_OOM_FRAMEWORK_COMMIT = "58c8b7e36fbff4bb08c7636b391f72b33d0ef0da"
SCORING_OOM_EVALUATOR_IMAGE = (
    "registry.cn-shanghai.aliyuncs.com/kcr-3rd/kesci_kernel_lab@sha256:"
    "5f1da890360acc5a81438d0e35a80079b34f30d1fa64fc954fef2e9a1ae45b64"
)
SCORING_OOM_ROOT_CAUSE_MANIFEST_SHA256 = (
    "sha256:a810115e85395e073588f3d19982ee1e07ca1dac0125ce297b5cdebdcc00456b"
)
SCORING_OOM_ERROR = "scoring baseline evaluator had an untrusted GPU termination"
SCORING_OOM_FAILURE_CLASS = "CONFIRMED_SCORING_CONTAINER_MEMORY_CGROUP_OOM"
SCORING_ABANDONED_UNKNOWN_STATUS = "ABANDONED_UNKNOWN_OUTCOME"


def _resolve_builtin_target(namespace: ResearchNamespace) -> TargetComponents:
    """Lazily enter the reviewed component registry.

    Keeping the legacy adapter import behind this boundary preserves the
    controller module's dependency-light import contract (notably, importing
    the controller must not initialize NumPy/Torch/Triton).
    """

    from ..platform.legacy import resolve_builtin_target

    return resolve_builtin_target(namespace)


def _trusted_target(namespace: ResearchNamespace) -> TargetComponents:
    """Resolve and verify one complete production target binding."""

    try:
        target = _resolve_builtin_target(namespace)
        definitions = {
            "operator": BUILTIN_PROFILE_REGISTRY.resolve(namespace.operator),
            "language": BUILTIN_PROFILE_REGISTRY.resolve(namespace.language),
            "evaluator": BUILTIN_PROFILE_REGISTRY.resolve(namespace.evaluator),
            "evaluation_protocol": BUILTIN_PROFILE_REGISTRY.resolve(
                namespace.evaluation_protocol
            ),
            "promotion_policy": BUILTIN_PROFILE_REGISTRY.resolve(
                namespace.promotion_policy
            ),
        }
        backend = target.device.evaluator_backend()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ControllerDataIntegrityError(
            f"research target has no exact trusted component binding: {exc}"
        ) from exc
    declared_backend = definitions["evaluator"].config.get("backend")
    if (
        not isinstance(backend, str)
        or backend != declared_backend
        or backend != "c500"
        or target.protocol.protocol_id != namespace.evaluation_protocol.id
        or target.protocol.revision != namespace.evaluation_protocol.revision
    ):
        raise ControllerDataIntegrityError(
            "trusted target evaluator backend/protocol binding is inconsistent"
        )
    return target


def _target_binding_snapshot(target: TargetComponents) -> dict[str, Any]:
    """Canonical implementation identities persisted with each Run."""

    return {
        "operator": {
            "id": target.operator.operator_id,
            "implementation_id": target.operator.implementation_id,
            "revision": target.operator.revision,
        },
        "language": {
            "id": target.language.language_id,
            "implementation_id": target.language.implementation_id,
            "revision": target.language.revision,
            "entrypoint": target.language.entrypoint,
        },
        "evaluator": {
            "id": target.device.device_id,
            "implementation_id": target.device.implementation_id,
            "revision": target.device.revision,
            "backend": target.device.evaluator_backend(),
        },
        "evaluation_protocol": {
            "id": target.protocol.protocol_id,
            "implementation_id": target.protocol.implementation_id,
            "revision": target.protocol.revision,
        },
        "promotion_policy": {
            "id": target.promotion.policy_id,
            "implementation_id": target.promotion.implementation_id,
            "revision": target.promotion.revision,
        },
    }


class ProposerFailure(ControlledRuntimeError):
    pass


class UnknownGPUOutcome(ControlledRuntimeError):
    """An evaluator action may have touched the GPU but has no trusted result."""


class ControllerDataIntegrityError(ControlledRuntimeError):
    """Durable controller and History evidence disagree."""


class ActionBudgetExhausted(ControlledRuntimeError):
    """A trusted external action cannot fit in the remaining Run window.

    This is a normal bounded-Run terminal condition.  In particular, it is
    not a proposer/controller failure and must never consume the consecutive
    failure allowance.
    """


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
        request_identity: Mapping[str, Any] | None = None,
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
        before_container_start: (
            Callable[[str, str, float], None] | None
        ) = None,
    ) -> None:
        self.config = config
        self.controller_dir = controller_dir
        self.runner = runner or CommandRunner()
        if before_container_start is not None and not callable(
            before_container_start
        ):
            raise TypeError("before_container_start must be callable")
        self.before_container_start = before_container_start

    def container_name(
        self, run_id: str, iteration_index: int, stage: str
    ) -> str:
        safe_stage = stage.lower().replace("_", "-")
        return f"kar-eval-{run_id}-{iteration_index:03d}-{safe_stage}"

    def _cache_dir(
        self,
        *,
        candidate_hash: str,
        purpose: str = "candidate",
        request_identity: Mapping[str, Any] | None = None,
    ) -> Path:
        evaluator_digest = self.config.evaluator_image.rsplit(
            "@sha256:", 1
        )[1]
        cache_identity: dict[str, Any] = {
            "schema_version": 2,
            "purpose": purpose,
            "candidate_hash": candidate_hash,
            "candidate_artifact_id": None,
            "operator_profile": None,
            "language_profile": None,
            "evaluator_profile": None,
            "evaluation_protocol": None,
            "promotion_policy": None,
            "operator_abi_binding": None,
            "toolchain_binding": None,
            "evaluator_image": self.config.evaluator_image,
            "framework_commit": self.config.resolved_framework_git_commit,
            "build_flags": [],
        }
        if request_identity is not None:
            namespace = request_identity.get("namespace")
            if not isinstance(namespace, Mapping):
                raise ControlledRuntimeError(
                    "cache identity is missing its research namespace"
                )
            namespace_fields = {
                "operator_profile": "operator",
                "language_profile": "language",
                "evaluator_profile": "evaluator",
                "evaluation_protocol": "evaluation_protocol",
                "promotion_policy": "promotion_policy",
            }
            for cache_key, namespace_key in namespace_fields.items():
                value = namespace.get(namespace_key)
                if not isinstance(value, Mapping):
                    raise ControlledRuntimeError(
                        f"cache identity is missing {namespace_key}"
                    )
                cache_identity[cache_key] = dict(value)
            cache_identity["candidate_artifact_id"] = request_identity.get(
                "candidate_artifact_id"
            )
            cache_identity["operator_abi_binding"] = dict(
                namespace["operator"]
            )
            cache_identity["toolchain_binding"] = dict(
                namespace["language"]
            )
        compile_cache_key = canonical_sha256(cache_identity)
        cache_leaf = (
            candidate_hash
            if request_identity is None
            else compile_cache_key.removeprefix("sha256:")
        )
        cache_dir = (
            self.config.evaluator_cache_dir
            / evaluator_digest
            / self.config.resolved_framework_git_commit
            / cache_leaf
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.chmod(0o700)
        manifest = {
            **cache_identity,
            "compile_cache_key": compile_cache_key,
            "cache_dir": str(cache_dir),
        }
        manifest_dir = (
            self.controller_dir
            / "cache-manifests"
            / evaluator_digest
            / self.config.resolved_framework_git_commit
        )
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{cache_leaf}.json"
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
                prefix=f".{cache_leaf}.",
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
        if self.before_container_start is not None:
            self.before_container_start(
                run_id, "doctor", float(DOCTOR_TIMEOUT_SEC)
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

    def scoring_baseline_probe(
        self,
        *,
        operation_id: str,
        probe_index: int,
        result_dir: Path,
    ) -> dict[str, Any]:
        """Run one fixed scoring-baseline probe and preserve private output."""

        name = f"kar-score-{operation_id[-12:]}-{probe_index:02d}"
        cache_dir = self._cache_dir(
            candidate_hash=scoring_reference_source_sha256().removeprefix("sha256:"),
            purpose="xpuoj-scoring-baseline",
        )
        timeout = float(self.config.evaluator_timeout_sec)
        if self.before_container_start is not None:
            self.before_container_start(operation_id, "scoring-baseline", timeout)
        command = self.runner.run(
            scoring_baseline_probe_argv(
                self.config,
                name=name,
                run_id=operation_id,
                cache_dir=cache_dir,
            ),
            input_text=None,
            timeout_sec=timeout,
            max_output_bytes=EVALUATOR_OUTPUT_LIMIT_BYTES,
            container_name=name,
            docker_binary=self.config.docker_binary,
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = result_dir / f"probe-{probe_index:02d}.stdout.json"
        stderr_path = result_dir / f"probe-{probe_index:02d}.stderr.txt"
        _atomic_write_bytes(stdout_path, command.stdout.encode("utf-8"))
        _atomic_write_bytes(stderr_path, command.stderr.encode("utf-8"))
        combined = (command.stdout + "\n" + command.stderr).lower()
        if command.timed_out:
            return {
                "status": "UNKNOWN_OUTCOME",
                "error": "scoring baseline evaluator timed out",
                "container_exit_code": command.returncode,
            }
        if command.output_limited:
            return {
                "status": "UNKNOWN_OUTCOME",
                "error": "scoring baseline evaluator output exceeded limit",
                "container_exit_code": command.returncode,
            }
        if command.returncode in {137, -9} or any(
            marker in combined for marker in FATAL_GPU_MARKERS
        ):
            return {
                "status": "UNKNOWN_OUTCOME",
                "error": "scoring baseline evaluator had an untrusted GPU termination",
                "container_exit_code": command.returncode,
            }
        try:
            payload = parse_json_output(command.stdout)
        except ValueError as exc:
            return {
                "status": "UNKNOWN_OUTCOME",
                "error": f"scoring baseline evaluator produced invalid output: {exc}",
                "container_exit_code": command.returncode,
            }
        expected = {
            "schema_version": 1,
            "command": "score-baseline-probe",
            "protocol_id": "xpuoj-th0-proxy-v1",
            "scoring_framework_git_commit": self.config.expected_git_commit,
            "reference_source_sha256": scoring_reference_source_sha256(),
            "compiler_config": {
                "backend": "inductor",
                "fullgraph": True,
                "dynamic": False,
                "mode": "default",
            },
            "measurement_contract": (
                scoring_baseline_measurement_contract_snapshot()
            ),
            "timing_protocol": device_event_protocol_snapshot(),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return {
                "status": "UNKNOWN_OUTCOME",
                "error": "scoring baseline evaluator identity echo mismatch",
                "container_exit_code": command.returncode,
            }
        status = payload.get("status")
        trusted_exit = (
            status == "QUALIFIED" and command.returncode == 0
        ) or (status == "UNQUALIFIED" and command.returncode == 2)
        if not trusted_exit:
            return {
                "status": "UNKNOWN_OUTCOME",
                "error": "scoring baseline evaluator status/exit mismatch",
                "container_exit_code": command.returncode,
            }
        return {**payload, "container_exit_code": command.returncode}

    def evaluate(
        self,
        *,
        candidate_path: Path,
        suite: str,
        baseline_path: Path | None,
        run_id: str,
        iteration_index: int,
        stage: str,
        request_identity: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = self.container_name(run_id, iteration_index, stage)
        evaluator_backend = _trusted_target(
            LEGACY_RESEARCH_NAMESPACE
        ).device.evaluator_backend()
        evaluation_protocol_id: str | None = None
        if request_identity is not None:
            namespace_value = request_identity.get("namespace")
            try:
                namespace = ResearchNamespace.from_value(namespace_value)
            except (TypeError, ValueError) as exc:
                raise ControllerDataIntegrityError(
                    "evaluator request has an invalid research namespace"
                ) from exc
            target = _trusted_target(namespace)
            evaluator_backend = target.device.evaluator_backend()
            evaluation_protocol_id = target.protocol.protocol_id
        result_dir = (
            self.controller_dir
            / "runs"
            / run_id
            / "results"
            / f"{iteration_index:03d}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        request_identity_path: Path | None = None
        if request_identity is not None:
            request_identity_path = result_dir / f"{stage}.request-identity.json"
            _atomic_write_bytes(
                request_identity_path,
                (
                    json.dumps(
                        dict(request_identity),
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
        candidate_hash = _sha256_file(candidate_path)
        cache_dir = self._cache_dir(
            candidate_hash=candidate_hash,
            request_identity=request_identity,
        )
        if self.before_container_start is not None:
            self.before_container_start(
                run_id, "evaluator", float(self.config.evaluator_timeout_sec)
            )
        command = self.runner.run(
            evaluator_argv(
                self.config,
                name=name,
                run_id=run_id,
                candidate_path=candidate_path,
                suite=suite,
                baseline_path=baseline_path,
                cache_dir=cache_dir,
                request_identity_path=request_identity_path,
                backend=evaluator_backend,
                evaluation_protocol_id=evaluation_protocol_id,
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


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Durably publish one private audit object without partial visibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
        path.chmod(0o600)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _strict_json_object_bytes(
    raw: bytes, *, label: str, max_bytes: int
) -> dict[str, Any]:
    if len(raw) > max_bytes:
        raise ValueError(f"{label} exceeds its size limit")

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON number is forbidden: {value}")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate object keys")
            value[key] = child
        return value

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if type(decoded) is not dict:
        raise ValueError(f"{label} is not a JSON object")
    canonical_json_text(decoded)
    return decoded


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


def _run_git_blob(repository: Path, revision_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", revision_path],
        cwd=repository,
        check=False,
        capture_output=True,
        timeout=10,
        shell=False,
    )
    if completed.returncode != 0:
        raise ControlledRuntimeError(
            "could not read deployment kernel Git blob: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout


def _git_framework_files(
    repository: Path, commit: str
) -> dict[PurePosixPath, tuple[bytes, int]]:
    """Read the exact committed evaluator framework without a worktree.

    The deployment commit may advance when an operator adopts a new
    ``kernel.py``.  Evaluator code remains sourced from the independently
    frozen framework commit, so candidate bytes can never silently relabel the
    execution environment.
    """

    listed = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            commit,
            "--",
            "kernel_research",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        timeout=30,
        shell=False,
    )
    if listed.returncode != 0:
        raise ControlledRuntimeError(
            "could not read frozen framework Git tree: "
            + listed.stderr.decode("utf-8", errors="replace").strip()
        )
    files: dict[PurePosixPath, tuple[bytes, int]] = {}
    for encoded in listed.stdout.split(b"\0"):
        if not encoded:
            continue
        try:
            metadata, raw_path = encoded.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path_text = raw_path.decode("utf-8")
        except (UnicodeError, ValueError) as exc:
            raise ControlledRuntimeError(
                "frozen framework Git tree contains an invalid entry"
            ) from exc
        path = PurePosixPath(path_text)
        if (
            object_type != "blob"
            or mode not in {"100644", "100755"}
            or path.is_absolute()
            or not path.parts
            or path.parts[0] != "kernel_research"
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ControlledRuntimeError(
                "frozen framework Git tree contains an unsafe entry"
            )
        relative = PurePosixPath(*path.parts[1:])
        if not relative.parts or relative in files:
            raise ControlledRuntimeError(
                "frozen framework Git tree contains a duplicate entry"
            )
        blob = subprocess.run(
            ["git", "cat-file", "blob", object_id],
            cwd=repository,
            check=False,
            capture_output=True,
            timeout=30,
            shell=False,
        )
        if blob.returncode != 0:
            raise ControlledRuntimeError(
                "could not read frozen framework Git blob: "
                + blob.stderr.decode("utf-8", errors="replace").strip()
            )
        files[relative] = (blob.stdout, 0o755 if mode == "100755" else 0o644)
    if not files:
        raise ControlledRuntimeError("frozen framework Git tree is empty")
    return files


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


def _campaign_database_for_checkpoint(
    config: ControllerConfig, run: Mapping[str, Any]
) -> tuple[Path, str] | None:
    """Locate the one conventional Campaign DB for a Campaign-owned run."""

    snapshot = run.get("workflow_snapshot")
    campaign_id = (
        snapshot.get("campaign_id")
        if isinstance(snapshot, Mapping)
        else None
    )
    baseline_value = run.get("baseline_ref")
    baseline_source = (
        baseline_value.get("source")
        if isinstance(baseline_value, Mapping)
        else None
    )
    campaign_owned = campaign_id is not None or baseline_source == "campaign"
    if not campaign_owned:
        return None
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or baseline_source != "campaign"
    ):
        raise ControlledRuntimeError(
            "Campaign-owned run has inconsistent campaign/baseline identity"
        )
    try:
        runtime_root = Path(
            os.path.commonpath(
                [
                    str(config.state_dir.resolve()),
                    str(config.controller_dir.resolve()),
                    str(config.checkpoint_dir.resolve()),
                ]
            )
        )
    except (OSError, ValueError) as exc:
        raise ControlledRuntimeError(
            "Campaign runtime root cannot be resolved"
        ) from exc
    candidates = (
        runtime_root / "campaign.sqlite3",
        runtime_root / "campaign" / "campaign.sqlite3",
    )
    existing = [path for path in candidates if path.exists()]
    if len(existing) != 1:
        raise ControlledRuntimeError(
            "Campaign-owned checkpoint requires exactly one conventional "
            "campaign.sqlite3"
        )
    selected = existing[0]
    if selected.is_symlink() or not selected.is_file():
        raise ControlledRuntimeError(
            "Campaign database must be a regular non-symlink file"
        )
    return selected, campaign_id


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
        trusted_action_timeout: Callable[[str], float | None] | None = None,
        _deployment_pin_override: DeploymentBaselinePin | None = None,
    ) -> None:
        self.config = config
        if _deployment_pin_override is not None and not isinstance(
            _deployment_pin_override, DeploymentBaselinePin
        ):
            raise TypeError(
                "_deployment_pin_override must be a DeploymentBaselinePin"
            )
        self._deployment_pin_override = _deployment_pin_override
        self.runner = runner or CommandRunner()
        self._preflight_run_overrides: dict[str, Mapping[str, Any]] = {}
        self._campaign_resume_doctor_overrides: dict[str, dict[str, Any]] = {}
        self._scoring_baseline_context: dict[str, Any] | None = None
        self.evaluator = evaluator or DockerEvaluator(
            config,
            controller_dir=config.controller_dir,
            runner=self.runner,
            before_container_start=self._before_docker_container,
        )
        if (
            isinstance(self.evaluator, DockerEvaluator)
            and self.evaluator.before_container_start is None
        ):
            self.evaluator.before_container_start = self._before_docker_container
        if trusted_action_timeout is not None and not callable(
            trusted_action_timeout
        ):
            raise TypeError("trusted_action_timeout must be callable")
        self.trusted_action_timeout = trusted_action_timeout
        if proposer_factory is None:
            self.proposer_factory = self._production_proposer
        else:
            self.proposer_factory = proposer_factory
        self.clock = clock

    def _assert_noise_resource_available(self) -> None:
        """Fence standalone noise work from every live Campaign GPU claim."""

        from ..campaign.paths import conventional_campaign_database
        from ..campaign.store import CampaignStore

        try:
            runtime_root = deployment_runtime_root(
                state_dir=self.config.state_dir,
                controller_dir=self.config.controller_dir,
                checkpoint_dir=self.config.checkpoint_dir,
            )
            database = conventional_campaign_database(runtime_root)
        except ValueError as exc:
            raise ControllerDataIntegrityError(
                f"noise collection has no canonical runtime/Campaign path: {exc}"
            ) from exc
        if database.is_symlink():
            raise ControllerDataIntegrityError(
                "noise collection rejects a symlinked Campaign database"
            )
        if database.exists() and not database.is_file():
            raise ControllerDataIntegrityError(
                "noise collection Campaign path is not a regular file"
            )
        if not database.exists():
            return
        try:
            with CampaignStore(database) as campaign:
                active = campaign.list_campaigns(active_only=True)
                lease = campaign.connection.execute(
                    """
                    SELECT resource_id, fencing_epoch, campaign_id, status
                    FROM resource_leases
                    WHERE resource_id = 'gpu1'
                      AND status IN ('ACTIVE', 'QUARANTINED')
                    ORDER BY fencing_epoch DESC LIMIT 1
                    """
                ).fetchone()
        except (OSError, ValueError, sqlite3.DatabaseError) as exc:
            raise ControllerDataIntegrityError(
                f"noise collection cannot prove Campaign fencing state: {exc}"
            ) from exc
        if active:
            identities = ", ".join(
                f"{item['id']}:{item['status']}" for item in active
            )
            raise ControlledRuntimeError(
                "noise collection is forbidden while a Campaign can resume: "
                + identities
            )
        if lease is not None:
            raise ControlledRuntimeError(
                "noise collection is blocked by Campaign resource lease "
                f"gpu1/{int(lease['fencing_epoch'])}/{lease['status']}"
            )

    @staticmethod
    def _noise_recovery_required(reason: str) -> ControlledRuntimeError:
        return ControlledRuntimeError(
            f"{reason}; all further noise collection is blocked until a "
            "trusted GPU doctor and manual recovery have cleared the durable "
            "Controller state"
        )

    def _assert_noise_attempt_ledger_clear(
        self, store: ControllerStore, *, current_experiment_uid: str
    ) -> None:
        """Reject bypass-by-index after an unresolved or hard GPU outcome."""

        for attempt in store.list_evaluation_attempts_by_replicate_kind(
            "noise"
        ):
            if attempt["experiment_uid"] == current_experiment_uid:
                continue
            status = str(attempt["status"])
            result = attempt.get("result")
            result = result if isinstance(result, Mapping) else {}
            error = str(attempt.get("error") or "").lower()
            hard = result.get("status") in HARD_STATUSES or any(
                marker in error for marker in FATAL_GPU_MARKERS
            )
            if status == "RUNNING":
                # Holding gpu1.lock proves no still-live cooperative evaluator
                # owns the device.  A RUNNING row therefore represents a host
                # death window and must become UNKNOWN, never a replay.
                store.finish_evaluation_attempt(
                    str(attempt["experiment_uid"]),
                    status="UNKNOWN_OUTCOME",
                    result={},
                    error="controller process ended while GPU outcome was unknown",
                )
                try:
                    prior_run = store.get_run(str(attempt["run_id"]))
                    if prior_run["status"] == RunStatus.RUNNING.value:
                        store.update_run_with_event(
                            str(attempt["run_id"]),
                            "RUN_FINISHED",
                            {"reason": "unknown noise GPU outcome"},
                            status=RunStatus.HARD_FAILED.value,
                            stop_reason="unknown noise GPU outcome",
                        )
                except ValueError as exc:
                    raise ControllerDataIntegrityError(
                        "noise attempt refers to an invalid Controller Run"
                    ) from exc
                raise self._noise_recovery_required(
                    "a previous noise evaluator has UNKNOWN_GPU_OUTCOME"
                )
            if status in {"UNKNOWN_OUTCOME", "FAILED"}:
                raise self._noise_recovery_required(
                    f"a previous noise attempt is terminal {status}"
                )
            if status == "PENDING":
                raise self._noise_recovery_required(
                    "a previous durable noise intent was not completed"
                )
            if status == "SUCCEEDED" and hard:
                raise self._noise_recovery_required(
                    "a previous noise evaluation produced a hard GPU outcome"
                )

    def _before_docker_container(
        self, run_id: str, action: str, configured_timeout: float
    ) -> None:
        scoring_context = self._scoring_baseline_context
        if action == "scoring-baseline":
            self._authorize_scoring_baseline_probe(
                scoring_context,
                operation_id=run_id,
                configured_timeout=configured_timeout,
            )
            return
        if action != "doctor":
            self._assert_no_unresolved_scoring_baseline()
        run_override = self._preflight_run_overrides.get(run_id)
        campaign_doctor = self._campaign_resume_doctor_overrides.get(run_id)
        if campaign_doctor is not None:
            if action != "doctor":
                raise ControllerDataIntegrityError(
                    "Campaign resume doctor context cannot authorize another action"
                )
            self._authorize_campaign_resume_doctor(
                campaign_doctor, configured_timeout=configured_timeout
            )
            campaign_doctor["guarded"] = True
            return
        # The standalone administrative doctor is not owned by a Run.  Every
        # Campaign start/resume and every evaluator action uses its real Run ID.
        if run_id == "preflight" and run_override is None:
            return
        if run_override is None:
            with ControllerStore(self.controller_db) as store:
                guarded_run = store.get_run(run_id)
            guarded_snapshot = guarded_run.get("workflow_snapshot")
            if (
                isinstance(guarded_snapshot, Mapping)
                and guarded_snapshot.get("evidence_operation")
                in {"noise-collect", "adoption-requalification"}
            ):
                # This is deliberately the last trusted-host check before
                # DockerEvaluator invokes runner.run.  The enclosing canonical
                # gpu1.lock still prevents a Campaign child from overlapping
                # even if it acquires a lease immediately after this read.
                self._assert_noise_resource_available()
        self._authorize_external_action(
            run_id,
            action=action,
            configured_timeout=configured_timeout,
            run_override=run_override,
        )

    def _scoring_baseline_root(self) -> Path:
        return self.config.controller_dir / "scoring-baseline"

    @staticmethod
    def _read_scoring_evidence(path: Path, *, label: str) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise ControllerDataIntegrityError(f"{label} is not a regular file")
        try:
            return _strict_json_object_bytes(
                path.read_bytes(),
                label=label,
                max_bytes=EVALUATOR_OUTPUT_LIMIT_BYTES,
            )
        except (OSError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                f"{label} is unreadable: {exc}"
            ) from exc

    def _verify_scoring_oom_source(
        self, operation_dir: Path, *, state_name: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Re-prove the one archived exit-137 source without reclassifying it."""

        if operation_dir.name != SCORING_OOM_OPERATION_ID:
            raise ControllerDataIntegrityError(
                "scoring OOM recovery operation identity is not exact"
            )
        intent = self._read_scoring_evidence(
            operation_dir / "intent.json", label="scoring OOM source intent"
        )
        unknown = self._read_scoring_evidence(
            operation_dir / state_name, label="scoring OOM source state"
        )
        receipt = self._read_scoring_evidence(
            operation_dir / "probe-00.receipt.json",
            label="scoring OOM source receipt",
        )
        material_keys = {
            "schema_version",
            "operation",
            "expected_git_commit",
            "framework_git_commit",
            "scoring_framework_git_commit",
            "evaluator_image",
            "reference_source_sha256",
            "timing_protocol",
            "measurement_contract",
            "probe_count",
        }
        intent_tail = {
            "operation_id",
            "operation_digest",
            "status",
            "active_probe_index",
            "completed_probes",
        }
        if set(intent) != material_keys | intent_tail or set(unknown) != (
            material_keys | intent_tail | {"error"}
        ):
            raise ControllerDataIntegrityError(
                "scoring OOM source fields are not exact"
            )
        material = {key: intent[key] for key in material_keys}
        expected_intent = {
            **material,
            "operation_id": SCORING_OOM_OPERATION_ID,
            "operation_digest": SCORING_OOM_OPERATION_DIGEST,
            "status": "RUNNING",
            "active_probe_index": None,
            "completed_probes": 0,
        }
        expected_unknown = {
            **expected_intent,
            "status": "UNKNOWN_OUTCOME",
            "active_probe_index": 0,
            "error": SCORING_OOM_ERROR,
        }
        measurement_contract = material.get("measurement_contract")
        if (
            intent != expected_intent
            or unknown != expected_unknown
            or canonical_sha256(material) != SCORING_OOM_OPERATION_DIGEST
            or material["schema_version"] != 1
            or material["operation"] != "score-baseline-qualify"
            or material["expected_git_commit"] != SCORING_OOM_SOURCE_COMMIT
            or material["scoring_framework_git_commit"]
            != SCORING_OOM_SOURCE_COMMIT
            or material["framework_git_commit"]
            != SCORING_OOM_FRAMEWORK_COMMIT
            or material["evaluator_image"] != SCORING_OOM_EVALUATOR_IMAGE
            or material["reference_source_sha256"]
            != scoring_reference_source_sha256()
            or material["timing_protocol"] != device_event_protocol_snapshot()
            or not isinstance(measurement_contract, Mapping)
            or measurement_contract.get("digest")
            != SCORING_OOM_MEASUREMENT_DIGEST
            or material["probe_count"] != SCORING_BASELINE_QUALIFICATION_RUNS
            or receipt
            != {
                "container_exit_code": 137,
                "error": SCORING_OOM_ERROR,
                "status": "UNKNOWN_OUTCOME",
            }
        ):
            raise ControllerDataIntegrityError(
                "scoring OOM source evidence is inconsistent"
            )
        for name in ("probe-00.stdout.json", "probe-00.stderr.txt"):
            path = operation_dir / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != 0:
                raise ControllerDataIntegrityError(
                    "scoring OOM raw output contract is not exact"
                )
        return material, unknown, receipt

    def _verify_scoring_oom_recovery_commit(self) -> None:
        """Allow only the direct reviewed child that fixes this incident."""

        commit = self.config.expected_git_commit
        if (
            self.config.resolved_framework_git_commit
            != SCORING_OOM_FRAMEWORK_COMMIT
            or self.config.evaluator_image != SCORING_OOM_EVALUATOR_IMAGE
        ):
            raise ControllerDataIntegrityError(
                "scoring OOM recovery runtime identity has drifted"
            )
        if _run_git(self.config.repository_dir, "rev-parse", "HEAD^") != (
            SCORING_OOM_SOURCE_COMMIT
        ):
            raise ControllerDataIntegrityError(
                "scoring OOM recovery must be the direct incident child"
            )
        changed = set(
            _run_git(
                self.config.repository_dir,
                "diff",
                "--name-only",
                SCORING_OOM_SOURCE_COMMIT,
                commit,
                "--",
            ).splitlines()
        )
        allowed = {
            "docs/xpuoj-scoring-v2/IMPLEMENTATION.md",
            "docs/xpuoj-scoring-v2/PROGRESS.md",
            "docs/xpuoj-scoring-v2/RESEARCH.md",
            "kernel_research/autorun/controller.py",
            "kernel_research/cli.py",
            "kernel_research/scoring_baseline_worker.py",
            "kernel_research/scoring_measurement.py",
            "tests/test_scoring_baseline_worker.py",
            "tests/test_scoring_cli.py",
            "tests/test_scoring_controller.py",
            "tests/test_scoring_measurement.py",
        }
        required = {
            "kernel_research/autorun/controller.py",
            "kernel_research/cli.py",
            "kernel_research/scoring_baseline_worker.py",
            "kernel_research/scoring_measurement.py",
        }
        if not required <= changed or not changed <= allowed:
            raise ControllerDataIntegrityError(
                "scoring OOM recovery changed an unauthorized path"
            )

    @staticmethod
    def _scoring_oom_recovery_intent(
        *,
        recovery_commit: str,
        unknown: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "command": "score-baseline-abandon-unknown-oom",
            "source_operation_id": SCORING_OOM_OPERATION_ID,
            "source_operation_digest": SCORING_OOM_OPERATION_DIGEST,
            "source_unknown_digest": canonical_sha256(unknown),
            "source_receipt_digest": canonical_sha256(receipt),
            "root_cause_manifest_sha256": (
                SCORING_OOM_ROOT_CAUSE_MANIFEST_SHA256
            ),
            "recovery_commit": recovery_commit,
            "status": "RUNNING",
            "replay_permitted": False,
        }

    @staticmethod
    def _scoring_oom_final(
        *,
        material: Mapping[str, Any],
        unknown: Mapping[str, Any],
        receipt: Mapping[str, Any],
        recovery_intent: Mapping[str, Any],
        doctor: Mapping[str, Any],
        recovery_commit: str,
    ) -> dict[str, Any]:
        return {
            **material,
            "operation_id": SCORING_OOM_OPERATION_ID,
            "operation_digest": SCORING_OOM_OPERATION_DIGEST,
            "status": SCORING_ABANDONED_UNKNOWN_STATUS,
            "original_status": "UNKNOWN_OUTCOME",
            "failure_classification": SCORING_OOM_FAILURE_CLASS,
            "reason": "operator abandoned confirmed memory-cgroup OOM",
            "source_unknown_digest": canonical_sha256(unknown),
            "source_receipt_digest": canonical_sha256(receipt),
            "root_cause_manifest_sha256": (
                SCORING_OOM_ROOT_CAUSE_MANIFEST_SHA256
            ),
            "recovery_commit": recovery_commit,
            "recovery_intent_digest": canonical_sha256(recovery_intent),
            "doctor_evidence_digest": canonical_sha256(doctor),
            "doctor_invoked": True,
            "replay_permitted": False,
            "qualification_effect": "none",
            "deployment_effect": "none",
        }

    def _verify_scoring_oom_abandonment(
        self, operation_dir: Path
    ) -> dict[str, Any]:
        expected_names = {
            "intent.json",
            "state.json",
            "probe-00.receipt.json",
            "probe-00.stdout.json",
            "probe-00.stderr.txt",
            "unknown.json",
            "recovery-intent.json",
            "recovery-doctor.json",
            "final.json",
        }
        entries = list(operation_dir.iterdir())
        if (
            {entry.name for entry in entries} != expected_names
            or any(entry.is_symlink() or not entry.is_file() for entry in entries)
        ):
            raise ControllerDataIntegrityError(
                "scoring OOM abandonment file inventory is not exact"
            )
        material, unknown, receipt = self._verify_scoring_oom_source(
            operation_dir, state_name="unknown.json"
        )
        recovery_intent = self._read_scoring_evidence(
            operation_dir / "recovery-intent.json",
            label="scoring OOM recovery intent",
        )
        doctor = self._read_scoring_evidence(
            operation_dir / "recovery-doctor.json",
            label="scoring OOM recovery doctor",
        )
        final = self._read_scoring_evidence(
            operation_dir / "final.json", label="scoring OOM final evidence"
        )
        state = self._read_scoring_evidence(
            operation_dir / "state.json", label="scoring OOM terminal state"
        )
        recovery_commit = final.get("recovery_commit")
        if (
            not isinstance(recovery_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", recovery_commit) is None
        ):
            raise ControllerDataIntegrityError(
                "scoring OOM recovery commit is invalid"
            )
        expected_recovery_intent = self._scoring_oom_recovery_intent(
            recovery_commit=recovery_commit,
            unknown=unknown,
            receipt=receipt,
        )
        expected_final = self._scoring_oom_final(
            material=material,
            unknown=unknown,
            receipt=receipt,
            recovery_intent=recovery_intent,
            doctor=doctor,
            recovery_commit=recovery_commit,
        )
        doctor_identity = doctor.get("identity")
        doctor_probe = doctor.get("c500_probe")
        if (
            recovery_intent != expected_recovery_intent
            or final != expected_final
            or state != expected_final
            or doctor.get("status") != "SUCCESS"
            or not isinstance(doctor_identity, Mapping)
            or doctor_identity.get("git_commit") != recovery_commit
            or not isinstance(doctor_probe, Mapping)
            or doctor_probe.get("status") != "SUCCESS"
            or not isinstance(doctor_probe.get("environment"), Mapping)
            or doctor_probe["environment"].get("compile_probe_status")
            != "PASSED"
        ):
            raise ControllerDataIntegrityError(
                "scoring OOM abandonment evidence is inconsistent"
            )
        return final

    @staticmethod
    def _validate_scoring_oom_doctor(
        doctor: Mapping[str, Any], *, recovery_commit: str
    ) -> None:
        doctor_identity = doctor.get("identity")
        doctor_probe = doctor.get("c500_probe")
        if (
            doctor.get("status") != "SUCCESS"
            or not isinstance(doctor_identity, Mapping)
            or doctor_identity.get("git_commit") != recovery_commit
            or not isinstance(doctor_probe, Mapping)
            or doctor_probe.get("status") != "SUCCESS"
            or not isinstance(doctor_probe.get("environment"), Mapping)
            or doctor_probe["environment"].get("compile_probe_status")
            != "PASSED"
        ):
            raise ControlledRuntimeError(
                "scoring OOM recovery doctor failed; replay is forbidden"
            )

    def _verify_scoring_baseline_final(
        self,
        *,
        operation_dir: Path,
        operation_material: Mapping[str, Any],
        operation_id: str,
        operation_digest: str,
    ) -> dict[str, Any]:
        """Re-prove a terminal qualification and its private evidence graph."""

        final = self._read_scoring_evidence(
            operation_dir / "final.json", label="scoring baseline final evidence"
        )
        state = self._read_scoring_evidence(
            operation_dir / "state.json", label="scoring baseline state"
        )
        if canonical_json_text(final) != canonical_json_text(state):
            raise ControllerDataIntegrityError(
                "scoring baseline terminal state differs from final evidence"
            )
        for key, expected in operation_material.items():
            if final.get(key) != expected:
                raise ControllerDataIntegrityError(
                    f"scoring baseline final evidence changed {key}"
                )
        if (
            final.get("operation_id") != operation_id
            or final.get("operation_digest") != operation_digest
            or final.get("status") not in {"QUALIFIED", "UNQUALIFIED"}
        ):
            raise ControllerDataIntegrityError(
                "scoring baseline final evidence identity mismatch"
            )

        common_keys = set(operation_material) | {
            "operation_id",
            "operation_digest",
            "status",
        }
        receipt_paths = sorted(operation_dir.glob("probe-*.receipt.json"))
        if any(
            path.is_symlink()
            or not path.is_file()
            or path.name != f"probe-{index:02d}.receipt.json"
            for index, path in enumerate(receipt_paths)
        ):
            raise ControllerDataIntegrityError(
                "scoring baseline receipt sequence is not exact"
            )
        receipts = [
            self._read_scoring_evidence(
                path, label=f"scoring baseline receipt {index}"
            )
            for index, path in enumerate(receipt_paths)
        ]

        qualification_value = final.get("qualification")
        if qualification_value is not None:
            if set(final) != common_keys | {"qualification", "private_object_id"}:
                raise ControllerDataIntegrityError(
                    "scoring baseline qualified final fields are not exact"
                )
            if not isinstance(qualification_value, Mapping):
                raise ControllerDataIntegrityError(
                    "scoring baseline qualification is not an object"
                )
            try:
                qualification = ScoringBaselineQualification.from_value(
                    qualification_value
                )
                object_id = require_sha256_digest(
                    final.get("private_object_id"), field="private_object_id"
                )
            except (TypeError, ValueError) as exc:
                raise ControllerDataIntegrityError(
                    f"scoring baseline qualification is invalid: {exc}"
                ) from exc
            expected_status = (
                "QUALIFIED" if qualification.qualified else "UNQUALIFIED"
            )
            if final["status"] != expected_status:
                raise ControllerDataIntegrityError(
                    "scoring baseline qualification and final status disagree"
                )
            if len(receipts) != SCORING_BASELINE_QUALIFICATION_RUNS:
                raise ControllerDataIntegrityError(
                    "scoring baseline qualification has an incomplete receipt set"
                )
            if tuple(canonical_sha256(receipt) for receipt in receipts) != (
                qualification.probe_digests
            ):
                raise ControllerDataIntegrityError(
                    "scoring baseline receipt digest mismatch"
                )
            qualification_bytes = (
                canonical_json_text(qualification.to_dict()) + "\n"
            ).encode("utf-8")
            raw_digest = object_id.removeprefix("sha256:")
            object_path = (
                self.config.controller_dir
                / "objects"
                / "sha256"
                / raw_digest[:2]
                / raw_digest[2:]
            )
            if (
                object_path.is_symlink()
                or not object_path.is_file()
                or hashlib.sha256(qualification_bytes).hexdigest() != raw_digest
                or object_path.read_bytes() != qualification_bytes
            ):
                raise ControllerDataIntegrityError(
                    "scoring baseline private qualification object mismatch"
                )
            return final

        if final["status"] != "UNQUALIFIED" or "private_object_id" in final:
            raise ControllerDataIntegrityError(
                "scoring baseline terminal evidence is incomplete"
            )
        if "failed_probe_index" in final:
            if set(final) != common_keys | {"failed_probe_index", "reason"}:
                raise ControllerDataIntegrityError(
                    "scoring baseline failed-probe final fields are not exact"
                )
            failed_index = final["failed_probe_index"]
            if (
                type(failed_index) is not int
                or not 0 <= failed_index < SCORING_BASELINE_QUALIFICATION_RUNS
                or len(receipts) != failed_index + 1
                or any(
                    receipt.get("status") != "QUALIFIED"
                    for receipt in receipts[:-1]
                )
                or receipts[-1].get("status") != "UNQUALIFIED"
            ):
                raise ControllerDataIntegrityError(
                    "scoring baseline failed-probe receipts are inconsistent"
                )
        else:
            if set(final) != common_keys | {"reason"}:
                raise ControllerDataIntegrityError(
                    "scoring baseline aggregation final fields are not exact"
                )
            if (
                len(receipts) != SCORING_BASELINE_QUALIFICATION_RUNS
                or any(receipt.get("status") != "QUALIFIED" for receipt in receipts)
            ):
                raise ControllerDataIntegrityError(
                    "scoring baseline aggregation receipts are inconsistent"
                )
        if not isinstance(final.get("reason"), str) or not final["reason"]:
            raise ControllerDataIntegrityError(
                "scoring baseline unqualified reason is invalid"
            )
        return final

    def _assert_no_unresolved_scoring_baseline(self) -> None:
        root = self._scoring_baseline_root()
        if not root.exists():
            return
        if root.is_symlink() or not root.is_dir():
            raise ControllerDataIntegrityError(
                "scoring baseline evidence root is not a regular directory"
            )
        for operation in sorted(root.iterdir()):
            if operation.is_symlink() or not operation.is_dir():
                raise ControllerDataIntegrityError(
                    "scoring baseline evidence contains an invalid object"
                )
            state_path = operation / "state.json"
            if not state_path.is_file() or state_path.is_symlink():
                raise ControllerDataIntegrityError(
                    "scoring baseline operation has no trusted state"
                )
            try:
                state = _strict_json_object_bytes(
                    state_path.read_bytes(),
                    label="scoring baseline state",
                    max_bytes=EVALUATOR_OUTPUT_LIMIT_BYTES,
                )
            except (OSError, ValueError) as exc:
                raise ControllerDataIntegrityError(
                    f"scoring baseline state is unreadable: {exc}"
                ) from exc
            status = state.get("status")
            try:
                operation_digest = require_sha256_digest(
                    state.get("operation_digest"), field="operation_digest"
                )
            except ValueError as exc:
                raise ControllerDataIntegrityError(
                    "scoring baseline operation digest is invalid"
                ) from exc
            expected_operation_id = (
                "score-baseline-" + operation_digest.removeprefix("sha256:")[:24]
            )
            if (
                state.get("operation_id") != expected_operation_id
                or operation.name != expected_operation_id
            ):
                raise ControllerDataIntegrityError(
                    "scoring baseline operation directory identity mismatch"
                )
            if status not in {
                "RUNNING",
                "UNKNOWN_OUTCOME",
                "QUALIFIED",
                "UNQUALIFIED",
                SCORING_ABANDONED_UNKNOWN_STATUS,
            }:
                raise ControllerDataIntegrityError(
                    "scoring baseline operation has an invalid status"
                )
            if status in {"RUNNING", "UNKNOWN_OUTCOME"}:
                raise ControlledRuntimeError(
                    "a scoring baseline probe has an unresolved GPU outcome"
                )
            if status == SCORING_ABANDONED_UNKNOWN_STATUS:
                self._verify_scoring_oom_abandonment(operation)

    def _assert_scoring_host_idle(self) -> None:
        self._assert_noise_resource_available()
        with ControllerStore(self.controller_db) as store:
            running = store.connection.execute(
                "SELECT id FROM runs WHERE status = 'RUNNING' LIMIT 1"
            ).fetchone()
            unresolved = store.connection.execute(
                """
                SELECT experiment_uid FROM evaluation_attempts
                WHERE status IN ('PENDING', 'RUNNING', 'UNKNOWN_OUTCOME')
                LIMIT 1
                """
            ).fetchone()
        if running is not None:
            raise ControlledRuntimeError(
                "scoring baseline qualification requires no active Controller Run"
            )
        if unresolved is not None:
            raise ControlledRuntimeError(
                "scoring baseline qualification is blocked by an unresolved evaluator"
            )

    def _scoring_container_present(self, name: str) -> bool:
        try:
            completed = subprocess.run(
                [
                    str(self.config.docker_binary),
                    "ps",
                    "-a",
                    "--filter",
                    f"name=^/{name}$",
                    "--format",
                    "{{.ID}}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ControllerDataIntegrityError(
                f"scoring baseline container inventory failed: {exc}"
            ) from exc
        if completed.returncode != 0 or completed.stderr.strip():
            raise ControllerDataIntegrityError(
                "scoring baseline container inventory is unavailable"
            )
        return bool(completed.stdout.strip())

    def finalize_scoring_unknown_oom(self) -> dict[str, Any]:
        """Abandon the exact archived OOM after one fresh trusted doctor."""

        self._prepare_runtime_dirs()
        self._verify_repository()
        self._verify_scoring_oom_recovery_commit()
        root = self._scoring_baseline_root()
        operation_dir = root / SCORING_OOM_OPERATION_ID
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            self._assert_scoring_host_idle()
            if operation_dir.is_symlink() or not operation_dir.is_dir():
                raise ControllerDataIntegrityError(
                    "scoring OOM operation directory is unavailable"
                )
            state = self._read_scoring_evidence(
                operation_dir / "state.json", label="scoring OOM state"
            )
            if state.get("status") == SCORING_ABANDONED_UNKNOWN_STATUS:
                final = self._verify_scoring_oom_abandonment(operation_dir)
                return {**final, "idempotent": True}
            base_names = {
                "intent.json",
                "state.json",
                "probe-00.receipt.json",
                "probe-00.stdout.json",
                "probe-00.stderr.txt",
            }
            intent_names = base_names | {"unknown.json", "recovery-intent.json"}
            doctor_names = intent_names | {"recovery-doctor.json"}
            final_names = doctor_names | {"final.json"}
            entries = list(operation_dir.iterdir())
            names = {entry.name for entry in entries}
            if names not in (
                base_names,
                intent_names,
                doctor_names,
                final_names,
            ) or any(entry.is_symlink() or not entry.is_file() for entry in entries):
                raise ControllerDataIntegrityError(
                    "scoring OOM recovery has an incomplete prior attempt"
                )
            state_name = "state.json" if names == base_names else "unknown.json"
            material, unknown, receipt = self._verify_scoring_oom_source(
                operation_dir, state_name=state_name
            )
            if names != base_names:
                current_state = self._read_scoring_evidence(
                    operation_dir / "state.json", label="scoring OOM state"
                )
                if current_state != unknown:
                    raise ControllerDataIntegrityError(
                        "scoring OOM source state changed during recovery"
                    )
            source_container = (
                f"kar-score-{SCORING_OOM_OPERATION_ID[-12:]}-00"
            )
            if self._scoring_container_present(source_container):
                raise ControlledRuntimeError(
                    "scoring OOM source container is still present"
                )
            recovery_commit = self.config.expected_git_commit
            recovery_intent = self._scoring_oom_recovery_intent(
                recovery_commit=recovery_commit,
                unknown=unknown,
                receipt=receipt,
            )
            if names == base_names:
                _atomic_write_bytes(
                    operation_dir / "unknown.json",
                    (canonical_json_text(unknown) + "\n").encode("utf-8"),
                )
                _atomic_write_bytes(
                    operation_dir / "recovery-intent.json",
                    (canonical_json_text(recovery_intent) + "\n").encode("utf-8"),
                )
            else:
                stored_intent = self._read_scoring_evidence(
                    operation_dir / "recovery-intent.json",
                    label="scoring OOM recovery intent",
                )
                if stored_intent != recovery_intent:
                    raise ControllerDataIntegrityError(
                        "scoring OOM recovery intent is inconsistent"
                    )
                recovery_intent = stored_intent
            doctor_run_id = "score-baseline-oom-recovery"
            if names in (base_names, intent_names):
                if names == intent_names:
                    raise ControlledRuntimeError(
                        "scoring OOM recovery doctor completion is uncertain; "
                        "replay is forbidden"
                    )
                doctor = self.doctor(
                    require_secret=False,
                    run_id=doctor_run_id,
                )
                _atomic_write_bytes(
                    operation_dir / "recovery-doctor.json",
                    (canonical_json_text(doctor) + "\n").encode("utf-8"),
                )
            else:
                doctor = self._read_scoring_evidence(
                    operation_dir / "recovery-doctor.json",
                    label="scoring OOM recovery doctor",
                )
            self._validate_scoring_oom_doctor(
                doctor, recovery_commit=recovery_commit
            )
            doctor_container = f"kar-doctor-{doctor_run_id}"
            if self._scoring_container_present(doctor_container):
                raise ControlledRuntimeError(
                    "scoring OOM recovery doctor container is still present"
                )
            self._verify_repository()
            self._assert_scoring_host_idle()
            final = self._scoring_oom_final(
                material=material,
                unknown=unknown,
                receipt=receipt,
                recovery_intent=recovery_intent,
                doctor=doctor,
                recovery_commit=recovery_commit,
            )
            if names == final_names:
                stored_final = self._read_scoring_evidence(
                    operation_dir / "final.json", label="scoring OOM final evidence"
                )
                if stored_final != final:
                    raise ControllerDataIntegrityError(
                        "scoring OOM final evidence is inconsistent"
                    )
            else:
                _atomic_write_bytes(
                    operation_dir / "final.json",
                    (canonical_json_text(final) + "\n").encode("utf-8"),
                )
            _atomic_write_bytes(
                operation_dir / "state.json",
                (canonical_json_text(final) + "\n").encode("utf-8"),
            )
            self._verify_scoring_oom_abandonment(operation_dir)
            return {**final, "idempotent": False}

    def finalize_scoring_pre_gpu_failure(self) -> dict[str, Any]:
        """Terminalize the one proven argparse rejection without GPU replay."""

        self._prepare_runtime_dirs()
        self._verify_repository()
        root = self._scoring_baseline_root()
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            self._assert_scoring_host_idle()
            if root.is_symlink() or not root.is_dir():
                raise ControllerDataIntegrityError(
                    "scoring baseline evidence root is unavailable"
                )
            unknown: list[tuple[Path, dict[str, Any]]] = []
            finalized: list[dict[str, Any]] = []
            for operation in sorted(root.iterdir()):
                if operation.is_symlink() or not operation.is_dir():
                    raise ControllerDataIntegrityError(
                        "scoring baseline evidence contains an invalid object"
                    )
                state = self._read_scoring_evidence(
                    operation / "state.json", label="scoring baseline state"
                )
                if state.get("status") == "UNKNOWN_OUTCOME":
                    unknown.append((operation, state))
                elif (
                    state.get("status") == "UNQUALIFIED"
                    and state.get("failure_classification")
                    == SCORING_PRE_GPU_FAILURE_CLASS
                ):
                    finalized.append(state)
            if not unknown:
                if len(finalized) == 1:
                    operation_id = str(finalized[0].get("operation_id", ""))
                    operation_dir = root / operation_id
                    final = self._read_scoring_evidence(
                        operation_dir / "final.json",
                        label="scoring baseline pre-GPU final",
                    )
                    if canonical_json_text(final) != canonical_json_text(
                        finalized[0]
                    ):
                        raise ControllerDataIntegrityError(
                            "pre-GPU scoring final differs from terminal state"
                        )
                    return {**final, "idempotent": True}
                raise ControlledRuntimeError(
                    "there is no unique pre-GPU scoring failure to finalize"
                )
            if len(unknown) != 1:
                raise ControlledRuntimeError(
                    "pre-GPU scoring finalizer requires exactly one unknown operation"
                )
            operation_dir, state = unknown[0]
            if (operation_dir / "final.json").exists() or (
                operation_dir / "final.json"
            ).is_symlink():
                raise ControllerDataIntegrityError(
                    "unknown scoring operation already has final evidence"
                )
            expected_names = {
                "intent.json",
                "state.json",
                "probe-00.receipt.json",
                "probe-00.stdout.json",
                "probe-00.stderr.txt",
            }
            entries = list(operation_dir.iterdir())
            if (
                {entry.name for entry in entries} != expected_names
                or any(entry.is_symlink() or not entry.is_file() for entry in entries)
            ):
                raise ControllerDataIntegrityError(
                    "pre-GPU scoring operation file inventory is not exact"
                )
            intent = self._read_scoring_evidence(
                operation_dir / "intent.json", label="scoring baseline intent"
            )
            material_keys = {
                "schema_version",
                "operation",
                "expected_git_commit",
                "framework_git_commit",
                "evaluator_image",
                "reference_source_sha256",
                "timing_protocol",
                "probe_count",
            }
            optional_material_keys = {
                "scoring_framework_git_commit",
                "measurement_contract",
            }
            present_optional = optional_material_keys & set(intent)
            allowed_optional_sets = {
                frozenset(),
                frozenset({"scoring_framework_git_commit"}),
                frozenset(optional_material_keys),
            }
            if frozenset(present_optional) not in allowed_optional_sets:
                raise ControllerDataIntegrityError(
                    "pre-GPU scoring intent has an invalid generation"
                )
            material_keys |= present_optional
            intent_tail = {
                "operation_id",
                "operation_digest",
                "status",
                "active_probe_index",
                "completed_probes",
            }
            state_tail = intent_tail | {"error"}
            if set(intent) != material_keys | intent_tail or set(state) != (
                material_keys | state_tail
            ):
                raise ControllerDataIntegrityError(
                    "pre-GPU scoring intent/state fields are not exact"
                )
            material = {key: intent[key] for key in material_keys}
            operation_digest = canonical_sha256(material)
            operation_id = (
                "score-baseline-"
                + operation_digest.removeprefix("sha256:")[:24]
            )
            expected_intent = {
                **material,
                "operation_id": operation_id,
                "operation_digest": operation_digest,
                "status": "RUNNING",
                "active_probe_index": None,
                "completed_probes": 0,
            }
            expected_state = {
                **expected_intent,
                "status": "UNKNOWN_OUTCOME",
                "active_probe_index": 0,
                "error": SCORING_PRE_GPU_ERROR,
            }
            if (
                intent != expected_intent
                or state != expected_state
                or operation_dir.name != operation_id
                or material["schema_version"] != 1
                or material["operation"] != "score-baseline-qualify"
                or material["expected_git_commit"]
                != self.config.expected_git_commit
                or material["framework_git_commit"]
                != self.config.resolved_framework_git_commit
                or (
                    "scoring_framework_git_commit" in material
                    and material["scoring_framework_git_commit"]
                    != self.config.expected_git_commit
                )
                or (
                    "measurement_contract" in material
                    and material["measurement_contract"]
                    != scoring_baseline_measurement_contract_snapshot()
                )
                or material["evaluator_image"] != self.config.evaluator_image
                or material["reference_source_sha256"]
                != scoring_reference_source_sha256()
                or material["timing_protocol"] != device_event_protocol_snapshot()
                or material["probe_count"] != SCORING_BASELINE_QUALIFICATION_RUNS
            ):
                raise ControllerDataIntegrityError(
                    "pre-GPU scoring operation identity is not exact"
                )
            receipt_paths = sorted(operation_dir.glob("probe-*.receipt.json"))
            if [path.name for path in receipt_paths] != [
                "probe-00.receipt.json"
            ]:
                raise ControllerDataIntegrityError(
                    "pre-GPU scoring operation has unexpected receipts"
                )
            receipt = self._read_scoring_evidence(
                receipt_paths[0], label="scoring baseline pre-GPU receipt"
            )
            if receipt != {
                "container_exit_code": 2,
                "error": SCORING_PRE_GPU_ERROR,
                "status": "UNKNOWN_OUTCOME",
            }:
                raise ControllerDataIntegrityError(
                    "pre-GPU scoring receipt does not prove argparse rejection"
                )
            stdout_path = operation_dir / "probe-00.stdout.json"
            stderr_path = operation_dir / "probe-00.stderr.txt"
            if (
                stdout_path.is_symlink()
                or not stdout_path.is_file()
                or stdout_path.stat().st_size != 0
                or stderr_path.is_symlink()
                or not stderr_path.is_file()
                or not 0 < stderr_path.stat().st_size <= 64 * 1024
            ):
                raise ControllerDataIntegrityError(
                    "pre-GPU scoring raw output contract is invalid"
                )
            try:
                stderr = stderr_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ControllerDataIntegrityError(
                    "pre-GPU scoring stderr cannot be read"
                ) from exc
            lowered = stderr.lower()
            if (
                not stderr.startswith("usage: kernel-research ")
                or "argument command: invalid choice: 'score-baseline-probe'"
                not in stderr
                or "traceback" in lowered
                or any(marker in lowered for marker in FATAL_GPU_MARKERS)
            ):
                raise ControllerDataIntegrityError(
                    "pre-GPU scoring stderr lacks the fixed argparse proof"
                )
            container_name = f"kar-score-{operation_id[-12:]}-00"
            if self._scoring_container_present(container_name):
                raise ControlledRuntimeError(
                    "pre-GPU scoring container is still present"
                )
            final = {
                **material,
                "operation_id": operation_id,
                "operation_digest": operation_digest,
                "status": "UNQUALIFIED",
                "reason": "fixed evaluator CLI rejected before GPU handler dispatch",
                "failure_classification": SCORING_PRE_GPU_FAILURE_CLASS,
            }
            _atomic_write_bytes(
                operation_dir / "final.json",
                (canonical_json_text(final) + "\n").encode("utf-8"),
            )
            _atomic_write_bytes(
                operation_dir / "state.json",
                (canonical_json_text(final) + "\n").encode("utf-8"),
            )
            return {**final, "idempotent": False}

    def _authorize_scoring_baseline_probe(
        self,
        context: Mapping[str, Any] | None,
        *,
        operation_id: str,
        configured_timeout: float,
    ) -> None:
        if (
            context is None
            or context.get("operation_id") != operation_id
            or context.get("status") != "RUNNING"
            or type(context.get("probe_index")) is not int
            or configured_timeout != float(self.config.evaluator_timeout_sec)
        ):
            raise ControllerDataIntegrityError(
                "scoring baseline Docker launch has no exact active intent"
            )
        state_path = Path(str(context["state_path"]))
        try:
            state = _strict_json_object_bytes(
                state_path.read_bytes(),
                label="scoring baseline intent",
                max_bytes=EVALUATOR_OUTPUT_LIMIT_BYTES,
            )
        except (OSError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                f"scoring baseline intent cannot be re-read: {exc}"
            ) from exc
        if (
            state.get("status") != "RUNNING"
            or state.get("operation_id") != operation_id
            or state.get("active_probe_index") != context["probe_index"]
        ):
            raise ControllerDataIntegrityError(
                "scoring baseline durable intent changed before Docker"
            )
        self._assert_scoring_host_idle()
        self._verify_repository()

    def qualify_scoring_baseline(self) -> dict[str, Any]:
        """Run ten private, non-promotable scoring-baseline probes."""

        if not isinstance(self.evaluator, DockerEvaluator):
            raise ControlledRuntimeError(
                "scoring baseline qualification requires DockerEvaluator"
            )
        self._prepare_runtime_dirs()
        self._require_gpu_risk_acknowledgement()
        host_errors = self.config.validate_host(require_secret=False)
        if host_errors:
            raise ControlledRuntimeError("; ".join(host_errors))
        self._verify_repository()
        self._prepare_framework_snapshot(self.config.expected_git_commit)
        image = self._inspect_image(self.config.evaluator_image)
        if not image["available"]:
            raise ControlledRuntimeError("pinned evaluator image is unavailable")
        operation_material = {
            "schema_version": 1,
            "operation": "score-baseline-qualify",
            "expected_git_commit": self.config.expected_git_commit,
            "framework_git_commit": self.config.resolved_framework_git_commit,
            "scoring_framework_git_commit": self.config.expected_git_commit,
            "evaluator_image": self.config.evaluator_image,
            "reference_source_sha256": scoring_reference_source_sha256(),
            "timing_protocol": device_event_protocol_snapshot(),
            "measurement_contract": (
                scoring_baseline_measurement_contract_snapshot()
            ),
            "probe_count": SCORING_BASELINE_QUALIFICATION_RUNS,
        }
        operation_digest = canonical_sha256(operation_material)
        operation_id = "score-baseline-" + operation_digest.removeprefix("sha256:")[:24]
        root = self._scoring_baseline_root()
        operation_dir = root / operation_id
        state_path = operation_dir / "state.json"
        final_path = operation_dir / "final.json"

        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            self._assert_no_unresolved_scoring_baseline()
            self._assert_scoring_host_idle()
            if final_path.exists():
                final = self._verify_scoring_baseline_final(
                    operation_dir=operation_dir,
                    operation_material=operation_material,
                    operation_id=operation_id,
                    operation_digest=operation_digest,
                )
                return {**final, "idempotent": True}
            if operation_dir.exists():
                raise ControlledRuntimeError(
                    "scoring baseline operation is incomplete; GPU replay is forbidden"
                )
            root.mkdir(parents=True, exist_ok=True)
            root.chmod(0o700)
            operation_dir.mkdir(mode=0o700)
            intent = {
                **operation_material,
                "operation_id": operation_id,
                "operation_digest": operation_digest,
                "status": "RUNNING",
                "active_probe_index": None,
                "completed_probes": 0,
            }
            _atomic_write_bytes(
                operation_dir / "intent.json",
                (canonical_json_text(intent) + "\n").encode("utf-8"),
            )
            _atomic_write_bytes(
                state_path,
                (canonical_json_text(intent) + "\n").encode("utf-8"),
            )
            probes: list[dict[str, Any]] = []
            for probe_index in range(SCORING_BASELINE_QUALIFICATION_RUNS):
                state = {
                    **intent,
                    "active_probe_index": probe_index,
                    "completed_probes": probe_index,
                }
                _atomic_write_bytes(
                    state_path,
                    (canonical_json_text(state) + "\n").encode("utf-8"),
                )
                self._scoring_baseline_context = {
                    "operation_id": operation_id,
                    "probe_index": probe_index,
                    "status": "RUNNING",
                    "state_path": str(state_path),
                }
                try:
                    result = self.evaluator.scoring_baseline_probe(
                        operation_id=operation_id,
                        probe_index=probe_index,
                        result_dir=operation_dir,
                    )
                finally:
                    self._scoring_baseline_context = None
                _atomic_write_bytes(
                    operation_dir / f"probe-{probe_index:02d}.receipt.json",
                    (canonical_json_text(result) + "\n").encode("utf-8"),
                )
                if result.get("status") == "UNKNOWN_OUTCOME":
                    unknown = {
                        **state,
                        "status": "UNKNOWN_OUTCOME",
                        "error": result.get("error"),
                    }
                    _atomic_write_bytes(
                        state_path,
                        (canonical_json_text(unknown) + "\n").encode("utf-8"),
                    )
                    raise ControlledRuntimeError(
                        "scoring baseline GPU outcome is unknown; replay is forbidden"
                    )
                if result.get("status") != "QUALIFIED":
                    final = {
                        **operation_material,
                        "operation_id": operation_id,
                        "operation_digest": operation_digest,
                        "status": "UNQUALIFIED",
                        "failed_probe_index": probe_index,
                        "reason": result.get("error", "probe did not qualify"),
                    }
                    _atomic_write_bytes(
                        final_path,
                        (canonical_json_text(final) + "\n").encode("utf-8"),
                    )
                    _atomic_write_bytes(
                        state_path,
                        (canonical_json_text(final) + "\n").encode("utf-8"),
                    )
                    return {**final, "idempotent": False}
                probes.append(result)
            runtime_environment = self._resolved_execution_environment(
                CURRENT_RESEARCH_NAMESPACE
            )
            try:
                qualification = aggregate_scoring_baseline_probes(
                    probes,
                    environment_digest=runtime_environment.digest,
                    evaluator_profile_digest=CURRENT_RESEARCH_NAMESPACE.evaluator.digest,
                    scoring_framework_git_commit=self.config.expected_git_commit,
                )
            except (TypeError, ValueError) as exc:
                final = {
                    **operation_material,
                    "operation_id": operation_id,
                    "operation_digest": operation_digest,
                    "status": "UNQUALIFIED",
                    "reason": f"probe aggregation failed: {exc}",
                }
                _atomic_write_bytes(
                    final_path,
                    (canonical_json_text(final) + "\n").encode("utf-8"),
                )
                _atomic_write_bytes(
                    state_path,
                    (canonical_json_text(final) + "\n").encode("utf-8"),
                )
                return {**final, "idempotent": False}
            qualification_bytes = (
                canonical_json_text(qualification.to_dict()) + "\n"
            ).encode("utf-8")
            object_id = self._store_private_object(qualification_bytes)
            final = {
                **operation_material,
                "operation_id": operation_id,
                "operation_digest": operation_digest,
                "status": "QUALIFIED" if qualification.qualified else "UNQUALIFIED",
                "qualification": qualification.to_dict(),
                "private_object_id": object_id,
            }
            _atomic_write_bytes(
                final_path,
                (canonical_json_text(final) + "\n").encode("utf-8"),
            )
            _atomic_write_bytes(
                state_path,
                (canonical_json_text(final) + "\n").encode("utf-8"),
            )
            return {**final, "idempotent": False}

    def campaign_resume_doctor(
        self,
        *,
        campaign_database: Path,
        campaign_id: str,
        namespace_id: str,
        campaign_snapshot_digest: str,
        resource_id: str,
        quarantine_fencing_epoch: int,
        run_id: str,
    ) -> dict[str, Any]:
        """Run an administrative doctor under an exact quarantine fence.

        This is intentionally not a synthetic Run.  The Docker launch hook
        re-reads the paused Campaign and quarantined resource immediately
        before the fixed evaluator doctor container starts.
        """

        database_input = Path(campaign_database)
        if database_input.is_symlink():
            raise ControllerDataIntegrityError(
                "Campaign resume doctor database must not be a symlink"
            )
        database = database_input.resolve()
        if not database.is_file():
            raise ControllerDataIntegrityError(
                "Campaign resume doctor database must be a regular file"
            )
        if (
            not isinstance(campaign_id, str)
            or not campaign_id
            or not isinstance(resource_id, str)
            or not resource_id
            or type(quarantine_fencing_epoch) is not int
            or quarantine_fencing_epoch <= 0
            or not isinstance(run_id, str)
            or not run_id.startswith("campaign-resume-")
        ):
            raise ControllerDataIntegrityError(
                "Campaign resume doctor identity is invalid"
            )
        if run_id in self._campaign_resume_doctor_overrides:
            raise ControllerDataIntegrityError(
                "Campaign resume doctor context is already active"
            )
        context: dict[str, Any] = {
            "campaign_database": str(database),
            "campaign_id": campaign_id,
            "namespace_id": namespace_id,
            "campaign_snapshot_digest": campaign_snapshot_digest,
            "resource_id": resource_id,
            "quarantine_fencing_epoch": quarantine_fencing_epoch,
            "doctor_timeout": float(DOCTOR_TIMEOUT_SEC),
            "guarded": False,
        }
        self._campaign_resume_doctor_overrides[run_id] = context
        try:
            result = self.doctor(
                require_secret=True,
                allowed_best_hashes=None,
                run_id=run_id,
            )
            if isinstance(self.evaluator, DockerEvaluator) and not context["guarded"]:
                raise ControllerDataIntegrityError(
                    "Campaign resume doctor container bypassed its quarantine guard"
                )
            return result
        finally:
            self._campaign_resume_doctor_overrides.pop(run_id, None)

    @staticmethod
    def _authorize_campaign_resume_doctor(
        context: Mapping[str, Any], *, configured_timeout: float
    ) -> None:
        try:
            timeout = float(configured_timeout)
            expected_timeout = float(context["doctor_timeout"])
            database = Path(str(context["campaign_database"]))
            campaign_id = str(context["campaign_id"])
            namespace_id = str(context["namespace_id"])
            snapshot_digest = str(context["campaign_snapshot_digest"])
            resource_id = str(context["resource_id"])
            fencing_epoch = int(context["quarantine_fencing_epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                "Campaign resume doctor context is malformed"
            ) from exc
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or timeout != expected_timeout
        ):
            raise ControllerDataIntegrityError(
                "Campaign resume doctor timeout differs from the trusted bound"
            )
        try:
            connection = sqlite3.connect(
                database.as_uri() + "?mode=ro", uri=True, timeout=5
            )
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA query_only = ON")
                connection.execute("BEGIN")
                campaign = connection.execute(
                    """
                    SELECT status, namespace_id, snapshot_digest
                    FROM campaigns WHERE id = ?
                    """,
                    (campaign_id,),
                ).fetchone()
                latest = connection.execute(
                    """
                    SELECT resource_id, campaign_id, fencing_epoch, status
                    FROM resource_leases
                    WHERE campaign_id = ? AND status = 'QUARANTINED'
                    ORDER BY rowid DESC LIMIT 1
                    """,
                    (campaign_id,),
                ).fetchone()
                active = connection.execute(
                    """
                    SELECT COUNT(*) FROM resource_leases
                    WHERE resource_id = ? AND status = 'ACTIVE'
                    """,
                    (resource_id,),
                ).fetchone()[0]
            finally:
                connection.rollback()
                connection.close()
        except sqlite3.DatabaseError as exc:
            raise ControllerDataIntegrityError(
                f"Campaign resume doctor fence cannot be read: {exc}"
            ) from exc
        if (
            campaign is None
            or campaign["status"]
            not in {"PAUSED_HARD_FAILURE", "PAUSED_UNKNOWN_OUTCOME"}
            or campaign["namespace_id"] != namespace_id
            or campaign["snapshot_digest"] != snapshot_digest
            or latest is None
            or latest["resource_id"] != resource_id
            or latest["campaign_id"] != campaign_id
            or int(latest["fencing_epoch"]) != fencing_epoch
            or latest["status"] != "QUARANTINED"
            or int(active) != 0
        ):
            raise ControllerDataIntegrityError(
                "Campaign resume doctor quarantine fence is stale or mismatched"
            )

    def _production_proposer(
        self, run_id: str, index: int, run_dir: Path
    ) -> Proposer:
        return OpenCodeProposer(
            self.config,
            run_id=run_id,
            iteration_index=index,
            run_dir=run_dir,
            runner=self.runner,
            before_container_start=lambda timeout: (
                self._authorize_external_action(
                    run_id,
                    action="proposer",
                    configured_timeout=timeout,
                )
            ),
        )

    @property
    def controller_db(self) -> Path:
        return self.config.controller_dir / "controller.sqlite3"

    @property
    def history_db(self) -> Path:
        return self.config.state_dir / "history.sqlite3"

    def _selected_action_timeout(
        self, action: str, configured_timeout: float | None
    ) -> float | None:
        """Return the trusted upper bound for one imminent external action.

        Production Docker adapters provide their configured timeout.  Injected
        unit adapters remain side-effect free and compatible by default; tests
        that model an external action opt in with ``trusted_action_timeout``.
        """

        selected = (
            self.trusted_action_timeout(action)
            if self.trusted_action_timeout is not None
            else configured_timeout
        )
        if selected is None:
            return None
        if (
            isinstance(selected, bool)
            or not isinstance(selected, (int, float))
            or not math.isfinite(float(selected))
            or float(selected) <= 0
        ):
            raise ControllerDataIntegrityError(
                f"trusted {action} timeout must be finite and positive"
            )
        return float(selected)

    def _campaign_action_deadline(
        self,
        run: Mapping[str, Any],
        *,
        now: float,
    ) -> float | None:
        """Re-read the complete Campaign ownership/fencing boundary.

        A workflow snapshot says which durable records must own this action;
        the live Campaign database says whether they still do.  The read-only
        transaction prevents mixing rows from different fencing moments.
        """

        snapshot = run.get("workflow_snapshot")
        baseline_value = run.get("baseline_ref")
        baseline_source = (
            baseline_value.get("source")
            if isinstance(baseline_value, Mapping)
            else None
        )
        campaign_id = (
            snapshot.get("campaign_id")
            if isinstance(snapshot, Mapping)
            else None
        )
        campaign_owned = campaign_id is not None or baseline_source == "campaign"
        if not campaign_owned:
            return None
        if not isinstance(snapshot, Mapping):
            raise ControllerDataIntegrityError(
                "Campaign-owned action has no frozen workflow snapshot"
            )
        child_value = snapshot.get("campaign_child")
        lease_value = snapshot.get("resource_lease")
        if (
            not isinstance(campaign_id, str)
            or not campaign_id
            or baseline_source != "campaign"
            or not isinstance(child_value, Mapping)
            or not isinstance(lease_value, Mapping)
        ):
            raise ControllerDataIntegrityError(
                "Campaign-owned action has incomplete child/lease identity"
            )
        try:
            child_id = int(child_value["child_id"])
            child_index = int(child_value["child_index"])
            controller_run_id = str(child_value["controller_run_id"])
            baseline_revision_id = str(
                child_value.get(
                    "baseline_revision_id",
                    baseline_value.get("revision")
                    if isinstance(baseline_value, Mapping)
                    else "",
                )
            )
            resource_id = str(lease_value["resource_id"])
            fencing_epoch = int(lease_value["fencing_epoch"])
            frozen_expiry = float(lease_value["expires_epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                "Campaign child/lease snapshot is invalid"
            ) from exc
        if (
            child_id <= 0
            or child_index <= 0
            or controller_run_id != run.get("id")
            or not baseline_revision_id
            or not resource_id
            or fencing_epoch <= 0
            or not math.isfinite(frozen_expiry)
        ):
            raise ControllerDataIntegrityError(
                "Campaign child/lease snapshot is inconsistent"
            )
        try:
            located = _campaign_database_for_checkpoint(self.config, run)
        except ControlledRuntimeError as exc:
            raise ControllerDataIntegrityError(str(exc)) from exc
        if located is None or located[1] != campaign_id:
            raise ControllerDataIntegrityError(
                "Campaign database does not match the frozen campaign"
            )
        campaign_path, _ = located
        try:
            connection = sqlite3.connect(
                campaign_path.as_uri() + "?mode=ro",
                uri=True,
                timeout=5,
            )
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA query_only = ON")
                connection.execute("BEGIN")
                campaign = connection.execute(
                    """
                    SELECT status, namespace_id, mode, snapshot_digest
                    FROM campaigns WHERE id = ?
                    """,
                    (campaign_id,),
                ).fetchone()
                child = connection.execute(
                    """
                    SELECT id, child_index, campaign_id, controller_run_id,
                           baseline_revision_id, status, max_wall_seconds
                    FROM child_runs WHERE id = ?
                    """,
                    (child_id,),
                ).fetchone()
                lease = connection.execute(
                    """
                    SELECT resource_id, campaign_id, fencing_epoch, status,
                           expires_epoch
                    FROM resource_leases
                    WHERE resource_id = ? AND fencing_epoch = ?
                    """,
                    (resource_id, fencing_epoch),
                ).fetchone()
                active = connection.execute(
                    """
                    SELECT fencing_epoch FROM resource_leases
                    WHERE resource_id = ? AND status = 'ACTIVE'
                    """,
                    (resource_id,),
                ).fetchone()
            finally:
                connection.close()
        except (OSError, sqlite3.DatabaseError) as exc:
            raise ControllerDataIntegrityError(
                f"Campaign fencing state cannot be read: {exc}"
            ) from exc
        budget = snapshot.get("budget")
        expected_wall = (
            budget.get("max_wall_seconds")
            if isinstance(budget, Mapping)
            else None
        )
        if (
            campaign is None
            or str(campaign["status"]) != "RUNNING"
            or str(campaign["namespace_id"]) != str(run.get("namespace_id"))
            or str(campaign["mode"]) != str(snapshot.get("mode"))
            or str(campaign["snapshot_digest"])
            != str(snapshot.get("campaign_snapshot_digest"))
            or child is None
            or int(child["child_index"]) != child_index
            or str(child["campaign_id"]) != campaign_id
            or str(child["controller_run_id"]) != controller_run_id
            or str(child["baseline_revision_id"]) != baseline_revision_id
            or str(child["status"]) != "RUNNING"
            or type(expected_wall) is not int
            or int(child["max_wall_seconds"]) != expected_wall
            or lease is None
            or str(lease["resource_id"]) != resource_id
            or str(lease["campaign_id"]) != campaign_id
            or int(lease["fencing_epoch"]) != fencing_epoch
            or str(lease["status"]) != "ACTIVE"
            or active is None
            or int(active["fencing_epoch"]) != fencing_epoch
        ):
            raise ControllerDataIntegrityError(
                "Campaign, child, or resource fencing ownership changed"
            )
        live_expiry = float(lease["expires_epoch"])
        if (
            not math.isfinite(live_expiry)
            or live_expiry < frozen_expiry
        ):
            raise ControllerDataIntegrityError(
                "Campaign resource lease expiry regressed or is invalid"
            )
        child_deadline_value = snapshot.get("child_deadline_epoch")
        if child_deadline_value is None:
            child_deadline = float(run["deadline_epoch"])
        elif (
            isinstance(child_deadline_value, bool)
            or not isinstance(child_deadline_value, (int, float))
            or not math.isfinite(float(child_deadline_value))
        ):
            raise ControllerDataIntegrityError(
                "Campaign child deadline snapshot is invalid"
            )
        else:
            child_deadline = float(child_deadline_value)
        if live_expiry <= now:
            raise ActionBudgetExhausted(
                "Campaign resource lease expired before the next trusted action"
            )
        return min(live_expiry, child_deadline)

    def _authorize_external_action(
        self,
        run_id: str,
        *,
        action: str,
        configured_timeout: float | None,
        run_override: Mapping[str, Any] | None = None,
    ) -> None:
        """Fail before launch unless the full trusted timeout still fits."""

        timeout = self._selected_action_timeout(action, configured_timeout)
        if timeout is None:
            return
        now = float(self.clock())
        if not math.isfinite(now):
            raise ControllerDataIntegrityError(
                "controller clock returned a non-finite epoch"
            )
        if run_override is None:
            with ControllerStore(self.controller_db) as store:
                run = store.get_run(run_id)
        else:
            run = dict(run_override)
        if str(run.get("id")) != run_id:
            raise ControllerDataIntegrityError(
                "external action does not match its frozen Run"
            )
        try:
            deadline = float(run["deadline_epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                "Run deadline is missing or invalid"
            ) from exc
        if not math.isfinite(deadline):
            raise ControllerDataIntegrityError("Run deadline is not finite")
        campaign_deadline = self._campaign_action_deadline(run, now=now)
        if campaign_deadline is not None:
            deadline = min(deadline, campaign_deadline)
        remaining = deadline - now
        if remaining < timeout:
            raise ActionBudgetExhausted(
                f"trusted {action} timeout ({timeout:.3f}s) does not fit "
                f"the remaining Run/Campaign window ({max(remaining, 0.0):.3f}s)"
            )

    def _deployment_pin(self) -> DeploymentBaselinePin | None:
        try:
            runtime_root = deployment_runtime_root(
                state_dir=self.config.state_dir,
                controller_dir=self.config.controller_dir,
                checkpoint_dir=self.config.checkpoint_dir,
            )
        except ValueError as exc:
            raise ControllerDataIntegrityError(
                f"formal config has no unique deployment runtime root: {exc}"
            ) from exc
        if self._deployment_pin_override is not None:
            pin = self._deployment_pin_override
        else:
            path = runtime_root / DEPLOYMENT_BASELINE_FILENAME
            if not path.exists() and not path.is_symlink():
                return None
            try:
                pin = DeploymentBaselinePin.load(path)
            except ValueError as exc:
                raise ControllerDataIntegrityError(
                    f"deployment baseline pin is invalid: {exc}"
                ) from exc
        trusted = {
            LEGACY_RESEARCH_NAMESPACE.namespace_id: LEGACY_RESEARCH_NAMESPACE,
            CURRENT_RESEARCH_NAMESPACE.namespace_id: CURRENT_RESEARCH_NAMESPACE,
        }.get(pin.namespace_id)
        if trusted is None:
            raise ControllerDataIntegrityError(
                "deployment baseline pin is not an exact built-in namespace"
            )
        if (
            pin.candidate_hash != self.config.expected_kernel_hash
            or pin.git_commit != self.config.expected_git_commit
        ):
            raise ControllerDataIntegrityError(
                "deployment baseline pin differs from formal config identity"
            )
        if trusted == CURRENT_RESEARCH_NAMESPACE:
            if not pin.execution_environment.is_resolved:
                raise ControllerDataIntegrityError(
                    "CURRENT deployment baseline requires resolved V2 evidence"
                )
            try:
                self._resolved_execution_environment(trusted).require_match(
                    pin.execution_environment,
                    context="deployment baseline",
                )
            except ValueError as exc:
                raise ControllerDataIntegrityError(str(exc)) from exc
        elif pin.execution_environment.is_resolved:
            try:
                self._resolved_execution_environment(trusted).require_match(
                    pin.execution_environment,
                    context="deployment baseline",
                )
            except ValueError as exc:
                raise ControllerDataIntegrityError(str(exc)) from exc
        return pin

    def _verify_deployment_bundle(self, record: ExperimentRecord) -> None:
        """Re-read the authoritative bundle and source CAS before every start."""

        try:
            artifact_id = ArtifactId.parse(record.artifact_id)
        except ValueError as exc:
            raise ControllerDataIntegrityError(
                "deployment History artifact ID is invalid"
            ) from exc
        if artifact_id.is_legacy_source:
            # The one-cycle V1 path is checked by ``_experiment_source_path``
            # during repository doctor.  V2 bundle identity has the additional
            # authoritative manifest object proven below.
            return
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            bundle_record = history.get_candidate_artifact(record.artifact_id)
            source_id = str(ArtifactId.source_sha256(record.candidate_hash))
            source_record = history.get_candidate_artifact(source_id)
            if (
                bundle_record is None
                or bundle_record.artifact_kind != "source_bundle_v1"
                or source_record is None
                or source_record.artifact_kind != "source_text_v1"
            ):
                raise ControllerDataIntegrityError(
                    "deployment bundle/source CAS records are missing"
                )
            for name, candidate in (
                ("bundle", bundle_record),
                ("entrypoint source", source_record),
            ):
                raw_path = self.config.state_dir / candidate.object_path
                current = raw_path
                while True:
                    if current.is_symlink():
                        raise ControllerDataIntegrityError(
                            f"deployment {name} CAS path traverses a symlink"
                        )
                    if current.parent == current:
                        break
                    current = current.parent
                if not raw_path.is_file():
                    raise ControllerDataIntegrityError(
                        f"deployment {name} CAS object is missing"
                    )
            try:
                bundle_bytes = history.read_candidate_artifact(
                    record.artifact_id
                )
                source_bytes = history.read_candidate_artifact(source_id)
            except (KeyError, RuntimeError) as exc:
                raise ControllerDataIntegrityError(
                    "deployment bundle/source CAS object is corrupted"
                ) from exc
        try:
            bundle = CandidateBundle.from_value(
                json.loads(bundle_bytes.decode("utf-8")),
                limits=TRITON_PYTHON_BUNDLE_LIMITS,
            )
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                "deployment bundle CAS payload is invalid"
            ) from exc
        entrypoint = next(
            item for item in bundle.files if item.path == bundle.entrypoint
        )
        if (
            bundle.bundle_bytes != bundle_bytes
            or str(bundle.artifact_id) != record.artifact_id
            or dict(bundle_record.manifest) != bundle.manifest
            or bundle_record.byte_size != len(bundle_bytes)
            or source_record.byte_size != len(source_bytes)
            or source_bytes != entrypoint.content_bytes
            or hashlib.sha256(source_bytes).hexdigest()
            != record.candidate_hash
        ):
            raise ControllerDataIntegrityError(
                "deployment bundle, manifest, entrypoint, and source CAS disagree"
            )

    def _best(self) -> ExperimentRecord:
        """Resolve the explicit deployment pin, never History's global best."""

        if not self.history_db.is_file():
            raise ControlledRuntimeError("history database does not exist")
        pin = self._deployment_pin()
        if pin is not None:
            kernel_path = self.config.repository_dir / "kernel.py"
            try:
                git_head = _run_git(
                    self.config.repository_dir, "rev-parse", "HEAD"
                )
                committed_kernel = _run_git_blob(
                    self.config.repository_dir, "HEAD:kernel.py"
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ControllerDataIntegrityError(
                    f"deployment Git identity cannot be read: {exc}"
                ) from exc
            if (
                kernel_path.is_symlink()
                or not kernel_path.is_file()
                or git_head != pin.git_commit
                or _sha256_file(kernel_path) != pin.candidate_hash
                or hashlib.sha256(committed_kernel).hexdigest()
                != pin.candidate_hash
            ):
                raise ControllerDataIntegrityError(
                    "deployment pin, Git HEAD, and kernel.py bytes disagree"
                )
            record = self._baseline_for_ref(
                pin.baseline_ref, namespace_id=pin.namespace_id
            )
            promotion = record.result.get("promotion")
            if (
                record.id != pin.confirmation_experiment_id
                or record.experiment_uid != pin.confirmation_experiment_uid
                or record.candidate_hash != pin.candidate_hash
                or record.status != "SUCCESS"
                or not record.promotable
                or not isinstance(promotion, Mapping)
                or promotion.get("phase") != "confirmation"
                or promotion.get("confirmed") is not True
            ):
                raise ControllerDataIntegrityError(
                    "deployment pin no longer proves its confirmation experiment"
                )
            if pin.execution_environment.is_resolved:
                try:
                    identity = ExperimentIdentity.from_value(dict(record.identity))
                except (TypeError, ValueError) as exc:
                    raise ControllerDataIntegrityError(
                        "deployment confirmation has invalid V2 identity"
                    ) from exc
                decision = promotion.get("decision")
                trusted_namespace = {
                    LEGACY_RESEARCH_NAMESPACE.namespace_id: (
                        LEGACY_RESEARCH_NAMESPACE
                    ),
                    CURRENT_RESEARCH_NAMESPACE.namespace_id: (
                        CURRENT_RESEARCH_NAMESPACE
                    ),
                }[pin.namespace_id]
                if (
                    identity.namespace != trusted_namespace
                    or str(identity.candidate_artifact_id) != record.artifact_id
                    or identity.experiment_uid != record.experiment_uid
                    or identity.stage != "confirmation"
                    or identity.suite != "full"
                    or identity.replicate_kind != "confirmation"
                    or identity.execution_environment != pin.execution_environment
                    or identity.baseline != pin.parent_baseline_ref
                    or not record.case_measurements
                    or any(
                        case.passed is not True
                        or case.matched_ratio is None
                        or float(case.matched_ratio) < 1.0
                        for case in record.case_measurements
                    )
                    or not isinstance(decision, Mapping)
                    or decision.get("promoted") is not True
                    or decision.get("needs_confirmation") is not False
                    or type(promotion.get("primary_experiment_id")) is not int
                    or pin.primary_experiment_uid is None
                    or pin.parent_baseline_ref is None
                ):
                    raise ControllerDataIntegrityError(
                        "deployment confirmation does not retain its V2 promotion proof"
                    )
                with HistoryStore(
                    self.history_db, state_dir=self.config.state_dir
                ) as history:
                    primary = history.get_experiment(
                        int(promotion["primary_experiment_id"])
                    )
                    relations = (
                        []
                        if primary is None
                        else history.list_experiment_relations(
                            primary.experiment_uid,
                            direction="source",
                            relation_type="confirmation_of",
                        )
                    )
                try:
                    primary_identity = (
                        None
                        if primary is None
                        else ExperimentIdentity.from_value(dict(primary.identity))
                    )
                except (TypeError, ValueError) as exc:
                    raise ControllerDataIntegrityError(
                        "deployment primary has invalid V2 identity"
                    ) from exc
                primary_promotion = (
                    None if primary is None else primary.result.get("promotion")
                )
                primary_decision = (
                    primary_promotion.get("decision")
                    if isinstance(primary_promotion, Mapping)
                    else None
                )
                confirmation_baseline_id = promotion.get(
                    "baseline_experiment_id"
                )
                if (
                    primary is None
                    or primary.experiment_uid != pin.primary_experiment_uid
                    or primary.namespace_id != pin.namespace_id
                    or primary.artifact_id != record.artifact_id
                    or primary.status != "SUCCESS"
                    or primary.promotable
                    or primary_identity is None
                    or primary_identity.namespace != trusted_namespace
                    or str(primary_identity.candidate_artifact_id)
                    != record.artifact_id
                    or primary_identity.experiment_uid
                    != primary.experiment_uid
                    or primary_identity.run_id != identity.run_id
                    or primary_identity.iteration != identity.iteration
                    or primary_identity.stage != "full_primary"
                    or primary_identity.suite != "full"
                    or primary_identity.replicate_kind != "primary"
                    or primary_identity.baseline != pin.parent_baseline_ref
                    or primary_identity.execution_environment
                    != pin.execution_environment
                    or not isinstance(primary_promotion, Mapping)
                    or primary_promotion.get("phase") != "primary"
                    or type(confirmation_baseline_id) is not int
                    or primary_promotion.get("baseline_experiment_id")
                    != confirmation_baseline_id
                    or primary_promotion.get("baseline_candidate_hash")
                    != promotion.get("baseline_candidate_hash")
                    or not isinstance(primary_decision, Mapping)
                    or primary_decision.get("needs_confirmation") is not True
                    or primary_decision.get("promoted") is not False
                    or len(
                        [
                            relation
                            for relation in relations
                            if relation.target_experiment_uid
                            == record.experiment_uid
                            and dict(relation.metadata)
                            == {
                                "baseline_experiment_id": (
                                    confirmation_baseline_id
                                )
                            }
                        ]
                    )
                    != 1
                ):
                    raise ControllerDataIntegrityError(
                        "deployment pin primary/confirmation link is broken"
                    )
            self._verify_deployment_bundle(record)
            return record
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            records = history.find_by_candidate_hash(
                self.config.expected_kernel_hash,
                namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            )
        best = next(
            (
                record
                for record in reversed(records)
                if record.backend == "c500"
                and record.suite == "full"
                and record.status == "SUCCESS"
                and record.promotable
            ),
            None,
        )
        if best is None:
            raise ControlledRuntimeError(
                "deployment baseline pin has no accepted C500 full evidence"
            )
        return best

    def _proposer_profile(self) -> ProfileRef:
        for definition in BUILTIN_PROFILE_REGISTRY.definitions():
            if (
                definition.ref.kind == "proposer"
                and definition.config.get("harness") == "opencode"
                and definition.config.get("model") == self.config.opencode_model
            ):
                return definition.ref
        raise ControlledRuntimeError(
            "selected proposer model has no trusted V2 profile"
        )

    def _history_cutoff(self) -> int:
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            records = history.list_all_experiments(
                limit=1, newest_first=True
            )
        return 0 if not records else int(records[0].id)

    def _baseline_ref(self, baseline: ExperimentRecord) -> BaselineRef:
        pin = self._deployment_pin()
        if pin is not None:
            if (
                baseline.id != pin.confirmation_experiment_id
                or baseline.artifact_id != str(pin.baseline_ref.artifact_id)
            ):
                raise ControllerDataIntegrityError(
                    "deployment baseline record differs from its pin"
                )
            return pin.baseline_ref
        return BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=ArtifactId.parse(baseline.artifact_id),
            source="deployment",
            revision=f"history-{baseline.id}",
        )

    def _resolved_execution_environment(
        self, namespace: ResearchNamespace
    ) -> ExecutionEnvironmentDigest:
        """Bind scientific execution to the exact trusted-host runtime.

        The Triton toolchain is part of the pinned evaluator image.  Its
        independent vendor version was not recorded by legacy evidence, so the
        resolved binding names the exact image bytes plus the reviewed language
        profile instead of inventing a version string.
        """

        language = BUILTIN_PROFILE_REGISTRY.resolve(namespace.language)
        operator = BUILTIN_PROFILE_REGISTRY.resolve(namespace.operator)
        evaluator = BUILTIN_PROFILE_REGISTRY.resolve(namespace.evaluator)
        operator_abi = operator.config.get("abi")
        if not isinstance(operator_abi, str) or not operator_abi:
            raise ControllerDataIntegrityError(
                "operator profile has no frozen ABI binding"
            )
        return ExecutionEnvironmentDigest.from_bindings(
            evaluator_image=self.config.evaluator_image,
            toolchain={
                "binding_basis": "pinned-evaluator-image",
                "language_profile": namespace.language.to_dict(),
                "evaluator_profile": namespace.evaluator.to_dict(),
                "language_implementation": language.implementation_id,
                "evaluator_implementation": evaluator.implementation_id,
                "evaluator_image": self.config.evaluator_image,
            },
            framework={
                "binding_basis": "verified-git-commit",
                "git_commit": self.config.resolved_framework_git_commit,
            },
            operator_abi={
                "operator_profile": namespace.operator.to_dict(),
                "abi": operator_abi,
            },
            # The current Triton adapter supplies no configurable compiler
            # flags.  The ordered empty vector is therefore the actual binding.
            build_flags=(),
        )

    def _store_private_object(self, content: bytes) -> str:
        """Store proposer audit bytes in the controller-private CAS."""

        digest = hashlib.sha256(content).hexdigest()
        path = (
            self.config.controller_dir
            / "objects"
            / "sha256"
            / digest[:2]
            / digest[2:]
        )
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise ControllerDataIntegrityError(
                    f"controller CAS collision for sha256:{digest}"
                )
            return f"sha256:{digest}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{digest}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
            path.chmod(0o600)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return f"sha256:{digest}"

    def _workflow_snapshot(
        self,
        *,
        baseline: ExperimentRecord,
        history_cutoff: int,
        namespace: ResearchNamespace | None = None,
        baseline_ref: BaselineRef | None = None,
    ) -> dict[str, Any]:
        """Resolve one deployment-owned controller path once at run start."""

        namespace = namespace or LEGACY_RESEARCH_NAMESPACE
        trusted_namespace = {
            LEGACY_RESEARCH_NAMESPACE.namespace_id: LEGACY_RESEARCH_NAMESPACE,
            CURRENT_RESEARCH_NAMESPACE.namespace_id: CURRENT_RESEARCH_NAMESPACE,
        }.get(namespace.namespace_id)
        if trusted_namespace is None or trusted_namespace != namespace:
            raise ControllerDataIntegrityError(
                "ordinary runs support only exact built-in LEGACY/CURRENT targets"
            )
        namespace = trusted_namespace
        target = _trusted_target(namespace)
        proposer = self._proposer_profile()
        deployment = BUILTIN_PROFILE_REGISTRY.get(
            kind="deployment",
            profile_id="legacy-local-c500",
            revision="v1",
        ).ref
        references = {
            "deployment": deployment,
            "operator": namespace.operator,
            "language": namespace.language,
            "evaluator": namespace.evaluator,
            "evaluation_protocol": namespace.evaluation_protocol,
            "promotion_policy": namespace.promotion_policy,
            "proposer": proposer,
        }
        baseline_ref = baseline_ref or self._baseline_ref(baseline)
        if (
            baseline_ref.source != "deployment"
            or baseline_ref.namespace_id != namespace.namespace_id
            or str(baseline_ref.artifact_id) != baseline.artifact_id
        ):
            raise ControllerDataIntegrityError(
                "ordinary Run baseline is not the frozen deployment baseline"
            )
        runtime_environment = self._resolved_execution_environment(namespace)
        comparable = baseline_ref.execution_environment.is_resolved
        if comparable:
            try:
                baseline_ref.require_environment(runtime_environment)
            except ValueError as exc:
                raise ControllerDataIntegrityError(str(exc)) from exc
        elif namespace != LEGACY_RESEARCH_NAMESPACE:
            raise ControllerDataIntegrityError(
                "CURRENT ordinary Run cannot use LEGACY_UNKNOWN baseline evidence"
            )
        snapshot = {
            "schema_version": 2,
            "mode": "DISCOVERY",
            "namespace": namespace.to_dict(),
            "deployment_profile": deployment.to_dict(),
            "proposer_profile": proposer.to_dict(),
            "resolved_profiles": {
                name: BUILTIN_PROFILE_REGISTRY.resolve(reference).to_dict()
                for name, reference in references.items()
            },
            "target_components": _target_binding_snapshot(target),
            "baseline_ref": baseline_ref.to_dict(),
            "execution_environment": (
                baseline_ref.execution_environment.to_dict()
            ),
            "runtime_execution_environment": runtime_environment.to_dict(),
            "scientifically_comparable": comparable,
            "history_cutoff": history_cutoff,
            "workflow": [stage.stage_id for stage in target.protocol.stages],
            "budget": {
                "max_candidates": self.config.max_candidates,
                "max_wall_seconds": int(self.config.max_hours * 3600),
                "max_consecutive_failures": self.config.max_consecutive_failures,
                "stop_after_promotion": self.config.stop_after_promotion,
            },
            "runtime_binding": self.config.redacted_dict(),
        }
        if not comparable:
            snapshot["compatibility_mode"] = "LEGACY_V1_EVIDENCE"
        return {**snapshot, "snapshot_digest": canonical_sha256(snapshot)}

    def _validate_run_snapshot(self, run: Mapping[str, Any]) -> None:
        snapshot = run.get("workflow_snapshot")
        if not isinstance(snapshot, Mapping):
            raise ControllerDataIntegrityError(
                "run workflow snapshot is not an object"
            )
        if snapshot.get("schema_version") == 0:
            return
        if snapshot.get("schema_version") != 2:
            raise ControllerDataIntegrityError(
                "run workflow snapshot has an unsupported schema"
            )
        material = dict(snapshot)
        claimed = material.pop("snapshot_digest", None)
        computed = canonical_sha256(material)
        if (
            claimed != computed
            or run.get("resolved_config_digest") != computed
        ):
            raise ControllerDataIntegrityError(
                "run workflow snapshot digest does not match its frozen identity"
            )
        namespace = self._run_namespace(run)
        target = _trusted_target(namespace)
        frozen_target_components = snapshot.get("target_components")
        if (
            frozen_target_components is not None
            and frozen_target_components != _target_binding_snapshot(target)
        ):
            raise ControllerDataIntegrityError(
                "run target component binding differs from the trusted registry"
            )
        if snapshot.get("workflow") != [
            stage.stage_id for stage in target.protocol.stages
        ]:
            raise ControllerDataIntegrityError(
                "run workflow differs from its trusted evaluation protocol"
            )
        if snapshot.get("baseline_ref") != run.get("baseline_ref"):
            raise ControllerDataIntegrityError(
                "run baseline differs from its workflow snapshot"
            )
        try:
            baseline_ref = BaselineRef.from_value(run["baseline_ref"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                "run baseline reference is invalid"
            ) from exc
        if baseline_ref.namespace_id != namespace.namespace_id:
            raise ControllerDataIntegrityError(
                "run baseline belongs to another namespace"
            )
        scientific_value = snapshot.get("execution_environment")
        try:
            scientific_environment = (
                baseline_ref.execution_environment
                if scientific_value is None
                else ExecutionEnvironmentDigest.from_value(scientific_value)
            )
            actual_environment = self._resolved_execution_environment(namespace)
            runtime_value = snapshot.get("runtime_execution_environment")
            frozen_runtime_environment = (
                actual_environment
                if runtime_value is None
                else ExecutionEnvironmentDigest.from_value(runtime_value)
            )
        except (TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                "run execution environment snapshot is invalid"
            ) from exc
        if frozen_runtime_environment != actual_environment:
            raise ControllerDataIntegrityError(
                "actual evaluator execution environment differs from the "
                "frozen run snapshot"
            )
        comparable = snapshot.get("scientifically_comparable", False)
        if type(comparable) is not bool:
            raise ControllerDataIntegrityError(
                "run scientific comparability marker is invalid"
            )
        if comparable:
            if scientific_value is None or runtime_value is None:
                raise ControllerDataIntegrityError(
                    "comparable V2 run is missing its execution environment"
                )
            try:
                actual_environment.require_match(
                    scientific_environment, context="run snapshot"
                )
                baseline_ref.require_environment(scientific_environment)
            except ValueError as exc:
                raise ControllerDataIntegrityError(str(exc)) from exc
            if snapshot.get("compatibility_mode") is not None:
                raise ControllerDataIntegrityError(
                    "comparable V2 run cannot enable legacy compatibility"
                )
        else:
            if namespace != LEGACY_RESEARCH_NAMESPACE:
                raise ControllerDataIntegrityError(
                    "LEGACY_UNKNOWN execution evidence is restricted to the "
                    "legacy namespace"
                )
            if (
                scientific_environment.is_resolved
                or baseline_ref.execution_environment.is_resolved
                or scientific_environment != baseline_ref.execution_environment
            ):
                raise ControllerDataIntegrityError(
                    "non-comparable run cannot reuse a resolved or mismatched "
                    "baseline environment"
                )
            compatibility_mode = snapshot.get("compatibility_mode")
            if compatibility_mode not in {None, "LEGACY_V1_EVIDENCE"}:
                raise ControllerDataIntegrityError(
                    "legacy run has an unknown compatibility mode"
                )
        if snapshot.get("history_cutoff") != run.get("history_cutoff"):
            raise ControllerDataIntegrityError(
                "run History cutoff differs from its workflow snapshot"
            )
        if snapshot.get("runtime_binding") != run.get("config"):
            raise ControllerDataIntegrityError(
                "run runtime binding differs from its workflow snapshot"
            )
        resolved = snapshot.get("resolved_profiles")
        if not isinstance(resolved, Mapping):
            raise ControllerDataIntegrityError(
                "run snapshot is missing resolved profiles"
            )
        deployment_value = snapshot.get("deployment_profile")
        if not isinstance(deployment_value, Mapping):
            raise ControllerDataIntegrityError(
                "run snapshot is missing its deployment profile"
            )
        try:
            deployment_profile = ProfileRef.from_value(deployment_value)
        except ValueError as exc:
            raise ControllerDataIntegrityError(
                "run deployment profile snapshot is invalid"
            ) from exc
        expected_refs = {
            "operator": namespace.operator,
            "language": namespace.language,
            "evaluator": namespace.evaluator,
            "evaluation_protocol": namespace.evaluation_protocol,
            "promotion_policy": namespace.promotion_policy,
            "proposer": self._run_proposer_profile(run),
            "deployment": deployment_profile,
        }
        for name, reference in expected_refs.items():
            value = resolved.get(name)
            if not isinstance(value, Mapping):
                raise ControllerDataIntegrityError(
                    f"run snapshot is missing resolved {name} profile"
                )
            try:
                trusted_definition = BUILTIN_PROFILE_REGISTRY.resolve(reference)
            except ValueError as exc:
                raise ControllerDataIntegrityError(
                    f"resolved {name} profile is not a trusted builtin"
                ) from exc
            if dict(value) != trusted_definition.to_dict():
                raise ControllerDataIntegrityError(
                    f"resolved {name} profile differs from its exact builtin definition"
                )

    def _run_namespace(self, run: Mapping[str, Any]) -> ResearchNamespace:
        snapshot = run.get("workflow_snapshot")
        if isinstance(snapshot, Mapping) and isinstance(
            snapshot.get("namespace"), Mapping
        ):
            namespace = ResearchNamespace.from_value(snapshot["namespace"])
            if namespace.namespace_id != run["namespace_id"]:
                raise ControllerDataIntegrityError(
                    "run namespace does not match its workflow snapshot"
                )
            return namespace
        if run["namespace_id"] != LEGACY_RESEARCH_NAMESPACE.namespace_id:
            raise ControllerDataIntegrityError(
                "non-legacy run is missing its namespace snapshot"
            )
        return LEGACY_RESEARCH_NAMESPACE

    def _run_proposer_profile(self, run: Mapping[str, Any]) -> ProfileRef:
        snapshot = run.get("workflow_snapshot")
        if isinstance(snapshot, Mapping):
            value = snapshot.get("proposer_profile")
            if isinstance(value, Mapping):
                try:
                    return ProfileRef.from_value(value)
                except ValueError as exc:
                    raise ControllerDataIntegrityError(
                        "run proposer profile snapshot is invalid"
                    ) from exc
        if isinstance(snapshot, Mapping) and snapshot.get("schema_version") == 0:
            return self._proposer_profile()
        raise ControllerDataIntegrityError(
            "run is missing its frozen proposer profile"
        )

    def _run_baseline(self, run_id: str) -> ExperimentRecord:
        with ControllerStore(self.controller_db) as store:
            run = store.get_run(run_id)
        try:
            baseline_ref = BaselineRef.from_value(run["baseline_ref"])
        except (TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                f"run baseline reference is invalid: {exc}"
            ) from exc
        if baseline_ref.namespace_id != run["namespace_id"]:
            raise ControllerDataIntegrityError(
                "run baseline belongs to another namespace"
            )
        return self._baseline_for_ref(
            baseline_ref, namespace_id=str(run["namespace_id"])
        )

    def _baseline_for_ref(
        self, baseline_ref: BaselineRef, *, namespace_id: str
    ) -> ExperimentRecord:
        """Resolve one explicit baseline without consulting global best state."""

        if baseline_ref.namespace_id != namespace_id:
            raise ControllerDataIntegrityError(
                "baseline reference belongs to another namespace"
            )
        revision = baseline_ref.revision
        preferred_id = (
            int(revision.removeprefix("history-"))
            if revision.startswith("history-")
            and revision.removeprefix("history-").isdigit()
            else None
        )
        preferred_uid = (
            revision.removeprefix("qualification-")
            if revision.startswith("qualification-")
            and revision.removeprefix("qualification-")
            else None
        )
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            record = (
                history.get_experiment(preferred_id)
                if preferred_id is not None
                else (
                    history.get_experiment_by_uid(preferred_uid)
                    if preferred_uid is not None
                    else None
                )
            )
            if record is None and preferred_id is None and preferred_uid is None:
                matches = history.list_experiments_for_namespace(
                    namespace_id,
                    artifact_id=str(baseline_ref.artifact_id),
                    status="SUCCESS",
                    newest_first=True,
                )
                record = next(
                    (candidate for candidate in matches if candidate.promotable),
                    None,
                )
        if (
            record is None
            or record.namespace_id != namespace_id
            or record.artifact_id != str(baseline_ref.artifact_id)
            or not record.promotable
        ):
            raise ControllerDataIntegrityError(
                "frozen run baseline cannot be proven from History"
            )
        if baseline_ref.execution_environment.is_resolved:
            try:
                baseline_identity = ExperimentIdentity.from_value(
                    dict(record.identity)
                )
                baseline_ref.require_environment(
                    baseline_identity.execution_environment
                )
            except (TypeError, ValueError) as exc:
                raise ControllerDataIntegrityError(
                    "resolved baseline has no matching V2 execution evidence "
                    "in History"
                ) from exc
        return record

    def _experiment_source(self, record: ExperimentRecord) -> str:
        """Read an experiment entrypoint regardless of source/bundle identity."""

        if record.artifact_id.startswith("source-sha256-v1:"):
            path = self.config.state_dir / record.artifact_path
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ControllerDataIntegrityError(
                    "accepted history artifact is missing or corrupted"
                ) from exc
        else:
            source_artifact_id = str(
                ArtifactId.source_sha256(record.candidate_hash)
            )
            with HistoryStore(
                self.history_db, state_dir=self.config.state_dir
            ) as history:
                try:
                    source = history.read_candidate_artifact(
                        source_artifact_id
                    ).decode("utf-8")
                except (KeyError, RuntimeError, UnicodeError) as exc:
                    raise ControllerDataIntegrityError(
                        "accepted history artifact is missing or corrupted: "
                        "bundle has no verified entrypoint source"
                    ) from exc
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != record.candidate_hash:
            raise ControllerDataIntegrityError(
                "baseline entrypoint source hash does not match History"
            )
        return source

    def _experiment_source_path(self, record: ExperimentRecord) -> Path:
        """Return a verified, immutable source object usable by the evaluator."""

        source = self._experiment_source(record)
        source_bytes = source.encode("utf-8")
        source_artifact_id = str(ArtifactId.source_sha256(record.candidate_hash))
        current_manifest = {
            "format": "python_source_v1",
            "entrypoint": "kernel.py",
            "media_type": "text/x-python",
        }
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            artifact = history.get_candidate_artifact(source_artifact_id)
            if artifact is None:
                artifact = history.store_candidate_artifact(
                    source_bytes,
                    artifact_id=source_artifact_id,
                    artifact_kind="source_text_v1",
                    manifest=current_manifest,
                )
            else:
                legacy_manifest = {
                    "format": "legacy_python_source_v1",
                    "entrypoint": Path(artifact.object_path).name,
                }
                supported_metadata = (
                    artifact.artifact_kind == "source_text_v1"
                    and artifact.manifest == current_manifest
                ) or (
                    artifact.artifact_kind == "legacy_source_v1"
                    and artifact.manifest == legacy_manifest
                )
                if not supported_metadata:
                    raise ControllerDataIntegrityError(
                        "baseline source artifact metadata is unsupported"
                    )
                try:
                    persisted = history.read_candidate_artifact(
                        source_artifact_id
                    )
                except (KeyError, RuntimeError) as exc:
                    raise ControllerDataIntegrityError(
                        "baseline source artifact is missing or corrupted"
                    ) from exc
                if persisted != source_bytes:
                    raise ControllerDataIntegrityError(
                        "baseline source artifact bytes differ from History"
                    )
        path = self.config.state_dir / artifact.object_path
        if not path.is_file() or _sha256_file(path) != record.candidate_hash:
            raise ControllerDataIntegrityError(
                "materialized baseline entrypoint is missing or corrupted"
            )
        return path

    def _verify_repository(
        self, *, allowed_best_hashes: set[str] | None = None
    ) -> dict[str, Any]:
        commit = _run_git(self.config.repository_dir, "rev-parse", "HEAD")
        if commit != self.config.expected_git_commit:
            raise ControlledRuntimeError(
                f"repository HEAD {commit} does not match expected commit"
            )
        framework_commit = self.config.resolved_framework_git_commit
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
        artifact = self._experiment_source_path(best)
        framework_check = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                framework_commit,
                commit,
            ],
            cwd=self.config.repository_dir,
            check=False,
            capture_output=True,
            timeout=10,
            shell=False,
        )
        if framework_check.returncode != 0:
            raise ControlledRuntimeError(
                "framework_git_commit is not an ancestor of the deployment commit"
            )
        return {
            "git_commit": commit,
            "framework_git_commit": framework_commit,
            "kernel_hash": kernel_hash,
            "baseline_experiment_id": best.id,
            "baseline_hash": best.candidate_hash,
            "baseline_artifact": str(artifact),
        }

    def _prepare_framework_snapshot(self, framework_commit: str) -> Path:
        """Materialize one exact, immutable package tree from Git blobs."""

        if not re.fullmatch(r"[0-9a-f]{40}", framework_commit):
            raise ControllerDataIntegrityError(
                "framework snapshot commit must be 40 lowercase hex digits"
            )
        destination = (
            self.config.controller_dir
            / "framework"
            / framework_commit
        )
        package_destination = destination / "kernel_research"
        expected = _git_framework_files(
            self.config.repository_dir, framework_commit
        )
        if package_destination.exists():
            if (
                destination.is_symlink()
                or package_destination.is_symlink()
                or any(
                    path.is_symlink()
                    for path in package_destination.rglob("*")
                )
            ):
                raise ControlledRuntimeError(
                    "trusted framework snapshot contains a symlink"
                )
            actual = {
                PurePosixPath(path.relative_to(package_destination).as_posix()): (
                    path.read_bytes(),
                    path.stat().st_mode & 0o777,
                )
                for path in package_destination.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            }
            if actual != expected:
                actual_bytes = {
                    relative: content
                    for relative, (content, _mode) in actual.items()
                }
                expected_bytes = {
                    relative: content
                    for relative, (content, _mode) in expected.items()
                }
                if actual_bytes != expected_bytes:
                    raise ControlledRuntimeError(
                        "trusted framework snapshot hash mismatch"
                    )
                # Snapshots created before framework/deployment identity was
                # split used copytree(), which preserved host group-write
                # bits even though Git records only executable vs regular.
                # Normalize only after proving the complete path set and all
                # file bytes exactly match the frozen Git tree.
                for relative, (_content, mode) in expected.items():
                    package_destination.joinpath(*relative.parts).chmod(mode)
                normalized = {
                    relative: (
                        content,
                        package_destination.joinpath(
                            *relative.parts
                        ).stat().st_mode
                        & 0o777,
                    )
                    for relative, (content, _mode) in actual.items()
                }
                if normalized != expected:
                    raise ControlledRuntimeError(
                        "trusted framework snapshot mode normalization failed"
                    )
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".tmp-{uuid.uuid4().hex}"
        try:
            package_temporary = temporary / "kernel_research"
            package_temporary.mkdir(parents=True)
            for relative, (content, mode) in expected.items():
                target = package_temporary.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                target.chmod(mode)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination

    def _prepare_framework(self) -> Path:
        """Materialize the frozen scientific evaluator framework commit."""

        return self._prepare_framework_snapshot(
            self.config.resolved_framework_git_commit
        )

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
        run_id: str = "preflight",
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
            probe = self.evaluator.doctor(run_id=run_id)
            if (
                probe.get("status") != "SUCCESS"
                or probe.get("environment", {}).get("compile_probe_status")
                != "PASSED"
            ):
                errors.append("C500 compile doctor did not pass")
        proposer_model = resolve_opencode_model(self.config.opencode_model)
        return {
            "schema_version": 1,
            "command": "doctor",
            "proposer_model": self.config.opencode_model,
            "proposer_reasoning_effort": proposer_model.reasoning_effort,
            "proposer_declared_output_tokens": proposer_model.output_tokens,
            "proposer_request_output_token_cap": (
                proposer_model.request_output_token_cap
            ),
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
        best = self._run_baseline(run_id)
        source = self._experiment_source(best)
        case_p50 = {
            str(case.get("case_id", case.get("name", "unknown"))): float(
                case["p50_us"]
            )
            for case in best.result.get("cases", [])
            if case.get("p50_us") is not None
        }
        with ControllerStore(self.controller_db) as store:
            run = store.get_run(run_id)
            target = _trusted_target(self._run_namespace(run))
            holdout_case_ids = frozenset(
                case_id
                for case_id, role in target.protocol.case_roles.items()
                if role.value == "holdout"
            )
            snapshot = run.get("workflow_snapshot")
            mode = (
                str(snapshot.get("mode", "DISCOVERY"))
                if isinstance(snapshot, Mapping)
                else "DISCOVERY"
            )
            cutoff = run.get("history_cutoff") if mode == "BENCHMARK" else None
            if mode == "BENCHMARK" and type(cutoff) is not int:
                raise ControllerDataIntegrityError(
                    "benchmark run has no numeric frozen History cutoff"
                )
            if mode == "BENCHMARK":
                campaign_snapshot = (
                    snapshot.get("campaign_snapshot")
                    if isinstance(snapshot, Mapping)
                    else None
                )
                frozen_feedback = (
                    campaign_snapshot.get("feedback_snapshot")
                    if isinstance(campaign_snapshot, Mapping)
                    else None
                )
                expected_digest = (
                    snapshot.get("feedback_snapshot_digest")
                    if isinstance(snapshot, Mapping)
                    else None
                )
                if not isinstance(frozen_feedback, list) or any(
                    not isinstance(item, Mapping) for item in frozen_feedback
                ):
                    raise ControllerDataIntegrityError(
                        "benchmark run has no frozen feedback snapshot"
                    )
                recent_value = json.loads(
                    canonical_json_text(
                        [dict(item) for item in frozen_feedback]
                    )
                )
                if canonical_sha256(recent_value) != expected_digest:
                    raise ControllerDataIntegrityError(
                        "benchmark feedback snapshot digest is inconsistent"
                    )
                for item in recent_value:
                    summary = item.get("result_summary")
                    if not isinstance(summary, Mapping):
                        continue
                    cases = summary.get("cases")
                    if isinstance(cases, list) and any(
                        isinstance(case, Mapping)
                        and case.get("case_id") in holdout_case_ids
                        for case in cases
                    ):
                        raise ControllerDataIntegrityError(
                            "benchmark feedback exposes a hidden holdout case"
                        )
                    for field in (
                        "per_case_speedups",
                        "confirmation_case_speedups",
                    ):
                        values = summary.get(field)
                        if isinstance(values, Mapping) and any(
                            case_id in values for case_id in holdout_case_ids
                        ):
                            raise ControllerDataIntegrityError(
                                "benchmark feedback exposes hidden holdout metrics"
                            )
                recent = tuple(recent_value)
            else:
                recent_items = store.list_recent_scientific_iterations(
                    exclude_run_id=run_id,
                    limit=MAX_FEEDBACK_CANDIDATES,
                    namespace_id=str(run["namespace_id"]),
                    history_cutoff=cutoff,
                )
                recent = tuple(
                    feedback_for_iteration(
                        item, hidden_case_ids=holdout_case_ids
                    )
                    for item in reversed(recent_items)
                )
            completed = (
                []
                if mode == "BENCHMARK"
                else [
                    item
                    for item in store.list_iterations(run_id)
                    if item["status"] != "RUNNING"
                ][-MAX_FEEDBACK_CANDIDATES:]
            )
            feedback = tuple(
                feedback_for_iteration(
                    item, hidden_case_ids=holdout_case_ids
                )
                for item in completed
            )
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
        self, *, note: str, candidate_hash: str, namespace_id: str
    ) -> ExperimentRecord | None:
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            return history.find_by_note_candidate(
                note=note,
                candidate_hash=candidate_hash,
                namespace_id=namespace_id,
            )

    def _candidate_seen(
        self,
        store: ControllerStore,
        *,
        run_id: str,
        candidate_hash: str,
    ) -> bool:
        run = store.get_run(run_id)
        namespace_id = str(run["namespace_id"])
        if store.candidate_seen_in_namespace(namespace_id, candidate_hash):
            return True
        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            return history.has_candidate_in_namespace(
                namespace_id, candidate_hash
            )

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

    def _experiment_identity(
        self,
        *,
        run: Mapping[str, Any],
        iteration: Mapping[str, Any],
        suite: str,
        stage: str,
        experiment_uid: str | None = None,
    ) -> ExperimentIdentity:
        snapshot = run.get("workflow_snapshot")
        snapshot = snapshot if isinstance(snapshot, Mapping) else {}
        proposer_value = snapshot.get("proposer_profile")
        proposer_profile = (
            ProfileRef.from_value(proposer_value)
            if isinstance(proposer_value, Mapping)
            else self._run_proposer_profile(run)
        )
        prompt_digest = None
        prompt_path_value = iteration.get("prompt_path")
        if isinstance(prompt_path_value, str) and prompt_path_value:
            prompt_path = Path(prompt_path_value)
            if prompt_path.is_file():
                prompt_digest = "sha256:" + _sha256_file(prompt_path)
        cutoff_value = run.get("history_cutoff")
        history_cutoff = (
            cutoff_value
            if isinstance(cutoff_value, (str, int))
            and not isinstance(cutoff_value, bool)
            else None
        )
        feedback_value = snapshot.get("feedback_snapshot_digest")
        feedback_digest = (
            str(feedback_value)
            if isinstance(feedback_value, str)
            else None
        )
        replicate_kind = {
            "full_primary": "primary",
            "confirmation": "confirmation",
        }.get(stage, "validation")
        baseline_ref = BaselineRef.from_value(run["baseline_ref"])
        environment_value = snapshot.get("execution_environment")
        try:
            execution_environment = (
                baseline_ref.execution_environment
                if environment_value is None
                else ExecutionEnvironmentDigest.from_value(environment_value)
            )
        except ValueError as exc:
            raise ControllerDataIntegrityError(
                "run execution environment identity is invalid"
            ) from exc
        if snapshot.get("scientifically_comparable", False):
            try:
                self._resolved_execution_environment(
                    self._run_namespace(run)
                ).require_match(
                    execution_environment, context="experiment"
                )
                baseline_ref.require_environment(execution_environment)
            except ValueError as exc:
                raise ControllerDataIntegrityError(str(exc)) from exc
        is_v2_snapshot = snapshot.get("schema_version") == 2
        candidate_artifact_id = ArtifactId.source_sha256(
            str(iteration["candidate_hash"])
        )
        parent_artifact_id = ArtifactId.source_sha256(
            str(iteration["parent_hash"])
        )
        if is_v2_snapshot:
            candidate_path_value = iteration.get("candidate_path")
            if not isinstance(candidate_path_value, str):
                raise ControllerDataIntegrityError(
                    "V2 iteration is missing its candidate bundle entrypoint"
                )
            candidate_source = Path(candidate_path_value).read_text(
                encoding="utf-8"
            )
            target = _trusted_target(self._run_namespace(run))
            candidate_artifact_id = CandidateBundle.single_file(
                content=candidate_source,
                path=target.language.entrypoint,
                limits=target.language.bundle_limits,
            ).artifact_id
            parent_artifact_id = baseline_ref.artifact_id
        return ExperimentIdentity.create(
            experiment_uid=experiment_uid,
            namespace=self._run_namespace(run),
            mode=str(snapshot.get("mode", "DISCOVERY")),
            candidate_artifact_id=candidate_artifact_id,
            parent_artifact_id=parent_artifact_id,
            baseline=baseline_ref,
            execution_environment=execution_environment,
            stage=stage,
            suite=suite,
            replicate_kind=replicate_kind,
            replicate_index=0,
            proposer_profile=proposer_profile,
            prompt_digest=prompt_digest,
            feedback_digest=feedback_digest,
            cohort_id=(
                str(snapshot["cohort_id"])
                if snapshot.get("cohort_id") is not None
                else None
            ),
            history_cutoff=history_cutoff,
            campaign_id=(
                str(snapshot["campaign_id"])
                if snapshot.get("campaign_id") is not None
                else None
            ),
            run_id=str(run["id"]),
            iteration=int(iteration["iteration_index"]),
        )

    @staticmethod
    def _validate_evaluator_echo(
        raw: Mapping[str, Any],
        *,
        identity: ExperimentIdentity,
        candidate_hash: str,
        suite: str,
        baseline_hash: str | None,
        backend: str,
    ) -> None:
        expected = {
            "schema_version": 1,
            "command": "evaluate-raw",
            "backend": backend,
            "suite": suite,
            "candidate_hash": candidate_hash,
            "baseline_candidate_hash": baseline_hash,
            "evaluation_protocol_id": identity.evaluation_protocol.id,
            "request_identity": identity.to_dict(),
        }
        missing = [key for key in expected if key not in raw]
        if missing:
            raise UnknownGPUOutcome(
                "evaluator result omitted trusted identity fields: "
                + ", ".join(sorted(missing))
            )
        mismatched = [
            key for key, value in expected.items() if raw.get(key) != value
        ]
        if mismatched:
            raise UnknownGPUOutcome(
                "evaluator result identity mismatch: "
                + ", ".join(sorted(mismatched))
            )

    @staticmethod
    def _link_iteration_record(
        store: ControllerStore,
        *,
        iteration: Mapping[str, Any],
        stage: str,
        record: ExperimentRecord,
        event: str,
    ) -> None:
        experiment_ids = dict(iteration.get("experiment_ids") or {})
        experiment_ids[stage] = record.id
        store.update_iteration_with_event(
            int(iteration["id"]),
            event,
            {
                "stage": stage,
                "experiment_id": record.id,
                "experiment_uid": record.experiment_uid,
            },
            experiment_ids=experiment_ids,
            result=dict(record.result),
            active_container=None,
        )

    @staticmethod
    def _validate_uid_reconciliation(
        *,
        persisted: ExperimentRecord,
        identity: ExperimentIdentity,
        attempt: Mapping[str, Any],
        candidate_hash: str,
        backend: str,
        suite: str,
        stage: str,
    ) -> None:
        """Prove a cross-store UID hit before mutating Controller state."""

        try:
            persisted_identity = ExperimentIdentity.from_value(
                dict(persisted.identity)
            )
        except (TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                "persisted History experiment identity is invalid"
            ) from exc
        canonical_identity = identity.to_dict()
        if (
            persisted_identity != identity
            or dict(persisted.identity) != canonical_identity
        ):
            raise ControllerDataIntegrityError(
                "persisted History experiment identity does not exactly match "
                "the Controller attempt"
            )

        expected_attempt = {
            "experiment_uid": identity.experiment_uid,
            "stage": identity.stage,
            "suite": identity.suite,
            "replicate_kind": identity.replicate_kind,
            "replicate_index": identity.replicate_index,
            "candidate_artifact_id": str(identity.candidate_artifact_id),
            "parent_artifact_id": str(identity.parent_artifact_id),
            "baseline_ref": identity.baseline.to_dict(),
            "condition_digest": identity.condition_digest,
            "request": canonical_identity,
        }
        attempt_mismatches = sorted(
            key
            for key, expected in expected_attempt.items()
            if attempt.get(key) != expected
        )
        if attempt_mismatches:
            raise ControllerDataIntegrityError(
                "evaluation attempt columns do not match its trusted identity: "
                + ", ".join(attempt_mismatches)
            )

        scalar_checks = {
            "experiment_uid": (
                persisted.experiment_uid,
                identity.experiment_uid,
            ),
            "namespace_id": (persisted.namespace_id, identity.namespace_id),
            "candidate_hash": (persisted.candidate_hash, candidate_hash),
            "artifact_id": (
                persisted.artifact_id,
                str(identity.candidate_artifact_id),
            ),
            "condition_digest": (
                persisted.condition_digest,
                identity.condition_digest,
            ),
            "backend": (persisted.backend, backend),
            "suite": (persisted.suite, suite),
            "replicate_kind": (
                persisted.replicate_kind,
                identity.replicate_kind,
            ),
            "replicate_index": (
                persisted.replicate_index,
                identity.replicate_index,
            ),
            "status": (persisted.status, persisted.result.get("status")),
            "stage": (identity.stage, stage),
        }
        scalar_mismatches = sorted(
            key
            for key, (stored, expected) in scalar_checks.items()
            if stored != expected
        )
        if scalar_mismatches:
            raise ControllerDataIntegrityError(
                "persisted History experiment columns do not match the trusted "
                "identity/attempt: "
                + ", ".join(scalar_mismatches)
            )

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
        run = store.get_run(run_id)
        target = _trusted_target(self._run_namespace(run))
        evaluator_backend = target.device.evaluator_backend()
        workflow_snapshot = run.get("workflow_snapshot")
        if (
            isinstance(workflow_snapshot, Mapping)
            and workflow_snapshot.get("schema_version") == 0
        ):
            reconciled = self._recorded_stage(
                note=note,
                candidate_hash=candidate_hash,
                namespace_id=str(run["namespace_id"]),
            )
            if reconciled is not None:
                self._link_iteration_record(
                    store,
                    iteration=iteration,
                    stage=stage,
                    record=reconciled,
                    event="STAGE_RECONCILED_LEGACY_NOTE",
                )
                return reconciled
        attempts = [
            attempt
            for attempt in store.list_evaluation_attempts(
                run_id, iteration_id=int(iteration["id"])
            )
            if attempt["stage"] == stage
        ]
        if len(attempts) > 1:
            raise ControllerDataIntegrityError(
                f"iteration has multiple attempts for stage {stage}"
            )
        attempt = attempts[0] if attempts else None
        if attempt is None:
            identity = self._experiment_identity(
                run=run,
                iteration=iteration,
                suite=suite,
                stage=stage,
            )
            attempt = store.create_evaluation_attempt(
                experiment_uid=identity.experiment_uid,
                run_id=run_id,
                iteration_id=int(iteration["id"]),
                stage=stage,
                suite=suite,
                replicate_kind=identity.replicate_kind,
                replicate_index=identity.replicate_index,
                candidate_artifact_id=str(identity.candidate_artifact_id),
                parent_artifact_id=str(identity.parent_artifact_id),
                baseline_ref=identity.baseline.to_dict(),
                condition_digest=identity.condition_digest,
                request=identity.to_dict(),
            )
        else:
            try:
                identity = ExperimentIdentity.from_value(attempt["request"])
            except (TypeError, ValueError) as exc:
                raise ControllerDataIntegrityError(
                    f"persisted evaluation identity is invalid: {exc}"
                ) from exc
            if (
                identity.experiment_uid != attempt["experiment_uid"]
                or identity.condition_digest != attempt["condition_digest"]
            ):
                raise ControllerDataIntegrityError(
                    "evaluation attempt identity columns do not match its snapshot"
                )

        with HistoryStore(
            self.history_db, state_dir=self.config.state_dir
        ) as history:
            persisted = history.get_experiment_by_uid(identity.experiment_uid)
        if persisted is not None:
            self._validate_uid_reconciliation(
                persisted=persisted,
                identity=identity,
                attempt=attempt,
                candidate_hash=candidate_hash,
                backend=evaluator_backend,
                suite=suite,
                stage=stage,
            )
            if attempt["status"] == "PENDING":
                store.start_evaluation_attempt(identity.experiment_uid)
                attempt = store.get_evaluation_attempt_by_uid(
                    identity.experiment_uid
                )
            if attempt is not None and attempt["status"] == "RUNNING":
                store.finish_evaluation_attempt(
                    identity.experiment_uid,
                    status="SUCCEEDED",
                    result=dict(persisted.result),
                    error=persisted.error_summary,
                )
            store.link_history_experiment(identity.experiment_uid, persisted.id)
            self._link_iteration_record(
                store,
                iteration=iteration,
                stage=stage,
                record=persisted,
                event="STAGE_RECONCILED_BY_UID",
            )
            return persisted

        if attempt["status"] == "UNKNOWN_OUTCOME":
            raise UnknownGPUOutcome(
                f"stage {stage} has an unknown GPU outcome and cannot restart"
            )
        if attempt["status"] in {"SUCCEEDED", "FAILED"}:
            raise ControllerDataIntegrityError(
                f"terminal stage {stage} has no matching History evidence"
            )

        best = self._run_baseline(run_id)
        baseline_path = (
            self._experiment_source_path(best) if suite == "full" else None
        )
        container_name = self.evaluator.container_name(
            run_id, int(iteration["iteration_index"]), stage
        )
        raw_result_path = (
            self.config.controller_dir
            / "runs"
            / run_id
            / "results"
            / f"{int(iteration['iteration_index']):03d}"
            / f"{stage}.validated.json"
        )
        raw: dict[str, Any]
        if attempt["status"] == "RUNNING":
            if not raw_result_path.is_file():
                store.finish_evaluation_attempt(
                    identity.experiment_uid,
                    status="UNKNOWN_OUTCOME",
                    result={},
                    error="controller restarted without a complete evaluator result",
                )
                raise UnknownGPUOutcome(
                    f"stage {stage} was interrupted after GPU execution began"
                )
            try:
                loaded = json.loads(raw_result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                store.finish_evaluation_attempt(
                    identity.experiment_uid,
                    status="UNKNOWN_OUTCOME",
                    result={},
                    error=f"persisted evaluator result is unreadable: {exc}",
                )
                raise UnknownGPUOutcome(
                    f"stage {stage} has unreadable persisted GPU evidence"
                ) from exc
            if not isinstance(loaded, dict):
                raise ControllerDataIntegrityError(
                    "persisted evaluator result is not an object"
                )
            raw = loaded
        else:
            if not isinstance(self.evaluator, DockerEvaluator):
                self._authorize_external_action(
                    run_id,
                    action="evaluator",
                    configured_timeout=None,
                )
            store.start_evaluation_attempt(identity.experiment_uid)
            store.update_iteration_with_event(
                int(iteration["id"]),
                "EVALUATOR_STARTED",
                {
                    "stage": stage,
                    "container": container_name,
                    "experiment_uid": identity.experiment_uid,
                },
                active_container=container_name,
            )
            try:
                raw = self.evaluator.evaluate(
                    candidate_path=candidate_path,
                    suite=suite,
                    baseline_path=baseline_path,
                    run_id=run_id,
                    iteration_index=int(iteration["iteration_index"]),
                    stage=stage,
                    request_identity=identity.to_dict(),
                )
            except (ActionBudgetExhausted, ControllerDataIntegrityError) as exc:
                store.finish_evaluation_attempt(
                    identity.experiment_uid,
                    status="FAILED",
                    result={"launch_skipped": True},
                    error=str(exc),
                )
                store.update_iteration_with_event(
                    int(iteration["id"]),
                    "EVALUATOR_LAUNCH_REJECTED",
                    {
                        "stage": stage,
                        "experiment_uid": identity.experiment_uid,
                        "reason": type(exc).__name__,
                    },
                    active_container=None,
                )
                raise
            except BaseException as exc:
                store.finish_evaluation_attempt(
                    identity.experiment_uid,
                    status="UNKNOWN_OUTCOME",
                    result={},
                    error=f"{type(exc).__name__}: evaluator action was interrupted",
                )
                raise

        try:
            self._validate_evaluator_echo(
                raw,
                identity=identity,
                candidate_hash=candidate_hash,
                suite=suite,
                baseline_hash=(best.candidate_hash if suite == "full" else None),
                backend=evaluator_backend,
            )
        except UnknownGPUOutcome as exc:
            store.finish_evaluation_attempt(
                identity.experiment_uid,
                status="UNKNOWN_OUTCOME",
                result=dict(raw),
                error=str(exc),
            )
            store.update_iteration(
                int(iteration["id"]), active_container=None
            )
            raise
        try:
            encoded_raw = (
                json.dumps(
                    raw,
                    sort_keys=True,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            _atomic_write_bytes(raw_result_path, encoded_raw)
        except (OSError, TypeError, ValueError) as exc:
            store.finish_evaluation_attempt(
                identity.experiment_uid,
                status="UNKNOWN_OUTCOME",
                result={},
                error=f"evaluator evidence could not be persisted: {exc}",
            )
            store.update_iteration(
                int(iteration["id"]), active_container=None
            )
            raise UnknownGPUOutcome(
                f"stage {stage} produced non-durable evaluator evidence"
            ) from exc
        try:
            if identity.replicate_kind == "noise":
                record = record_noise_external_result(
                    candidate_source=source,
                    result=raw,
                    state_dir=self.config.state_dir,
                    note=note,
                    git_revision=self.config.expected_git_commit[:12],
                    identity=identity,
                    baseline_experiment_id=best.id,
                )
            else:
                record = record_external_result(
                    candidate_source=source,
                    result=raw,
                    backend=evaluator_backend,
                    suite=suite,
                    state_dir=self.config.state_dir,
                    note=note,
                    git_revision=self.config.expected_git_commit[:12],
                    identity=identity,
                    baseline_experiment_id=best.id,
                )
        except Exception as exc:
            store.finish_evaluation_attempt(
                identity.experiment_uid,
                status="FAILED",
                result=dict(raw),
                error=f"{type(exc).__name__}: {exc}",
            )
            store.update_iteration(
                int(iteration["id"]), active_container=None
            )
            raise ControllerDataIntegrityError(
                f"trusted evaluator evidence could not be recorded: {exc}"
            ) from exc
        store.finish_evaluation_attempt(
            identity.experiment_uid,
            status="SUCCEEDED",
            result=dict(record.result),
            error=record.error_summary,
        )
        store.link_history_experiment(identity.experiment_uid, record.id)
        self._link_iteration_record(
            store,
            iteration=iteration,
            stage=stage,
            record=record,
            event="EVALUATOR_RECORDED",
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
        run = store.get_run(run_id)
        self._validate_run_snapshot(run)
        target = _trusted_target(self._run_namespace(run))
        if not iteration.get("candidate_hash"):
            request = self._proposal_request(run_id)
            namespace = self._run_namespace(run)
            baseline_ref = BaselineRef.from_value(run["baseline_ref"])
            proposer_profile = self._run_proposer_profile(run)
            prompt_protocol = {
                "id": "proposal-v1",
                "revision": "v1",
                "digest": canonical_sha256(
                    {
                        "id": "proposal-v1",
                        "revision": "v1",
                        "format": "single-json-object",
                    }
                ),
            }
            prompt_bytes = build_prompt(request).encode("utf-8")
            prompt_object_id = self._store_private_object(prompt_bytes)
            request_snapshot = {
                "schema_version": 1,
                "namespace_id": namespace.namespace_id,
                "baseline_ref": baseline_ref.to_dict(),
                "history_cutoff": run.get("history_cutoff"),
                "parent_candidate_hash": request.parent_candidate_hash,
                "accepted_kernel_digest": "sha256:"
                + hashlib.sha256(request.accepted_kernel.encode("utf-8")).hexdigest(),
                "program_digest": "sha256:"
                + hashlib.sha256(request.program_markdown.encode("utf-8")).hexdigest(),
                "environment_digest": canonical_sha256(dict(request.environment)),
                "recent_feedback_digest": canonical_sha256(
                    [dict(item) for item in request.recent_experiments]
                ),
                "session_feedback_digest": canonical_sha256(
                    [dict(item) for item in request.session_feedback]
                ),
                "prompt_object_id": prompt_object_id,
                "proposer_profile": proposer_profile.to_dict(),
                "prompt_protocol": prompt_protocol,
            }
            workflow_snapshot = run.get("workflow_snapshot")
            if (
                isinstance(workflow_snapshot, Mapping)
                and workflow_snapshot.get("mode") == "BENCHMARK"
                and (
                    request_snapshot["recent_feedback_digest"]
                    != workflow_snapshot.get("feedback_snapshot_digest")
                    or request_snapshot["session_feedback_digest"]
                    != canonical_sha256([])
                )
            ):
                raise ControllerDataIntegrityError(
                    "benchmark prompt feedback differs from its frozen snapshot"
                )
            proposal_context_id = canonical_sha256(request_snapshot)
            existing_attempts = store.list_proposal_attempts(
                run_id, iteration_id=iteration_id
            )
            if existing_attempts:
                last_attempt = existing_attempts[-1]
                if last_attempt["status"] == "RUNNING":
                    store.finish_proposal_attempt(
                        int(last_attempt["id"]),
                        status="UNKNOWN_OUTCOME",
                        result={},
                        error=(
                            "controller restarted while proposer action was in flight"
                        ),
                    )
                    raise ProposerFailure(
                        "persisted proposer attempt has an unknown outcome"
                    )
                raise ControllerDataIntegrityError(
                    "terminal proposer attempt has no accepted iteration candidate"
                )
            proposal_attempt = store.create_proposal_attempt(
                run_id=run_id,
                iteration_id=iteration_id,
                attempt_uid=str(uuid.uuid4()),
                proposal_context_id=proposal_context_id,
                parent_artifact_id=str(baseline_ref.artifact_id),
                proposer_profile=proposer_profile.to_dict(),
                prompt_protocol=prompt_protocol,
                request=request_snapshot,
                prompt_object_id=prompt_object_id,
            )
            proposer = self.proposer_factory(
                run_id, int(iteration["iteration_index"]), run_dir
            )
            if isinstance(proposer, OpenCodeProposer):
                # An explicitly injected production adapter still receives the
                # per-attempt guard; unit proposers remain untouched unless the
                # trusted timeout hook below opts them into external semantics.
                if proposer.before_container_start is None:
                    proposer.before_container_start = lambda timeout: (
                        self._authorize_external_action(
                            run_id,
                            action="proposer",
                            configured_timeout=timeout,
                        )
                    )
            guard_injected_proposer = (
                not isinstance(proposer, OpenCodeProposer)
                and self.trusted_action_timeout is not None
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
            proposal_started = time.monotonic()
            try:
                try:
                    if guard_injected_proposer:
                        self._authorize_external_action(
                            run_id,
                            action="proposer",
                            configured_timeout=None,
                        )
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
                        retry_trigger = attempts[1].get("retry_trigger")
                        retry_event = {
                            "FORMAT_ERROR": "PROPOSER_FORMAT_RETRY",
                            "FIELD_LENGTH_ERROR": "PROPOSER_CONSTRAINT_RETRY",
                        }.get(retry_trigger)
                        if retry_event is None:
                            raise AssertionError(
                                "proposal retry is missing a known trigger"
                            )
                        store.add_event(
                            run_id,
                            retry_event,
                            {"attempts": list(attempts)},
                            iteration_id=iteration_id,
                        )
            except (ActionBudgetExhausted, ControllerDataIntegrityError) as exc:
                store.finish_proposal_attempt(
                    int(proposal_attempt["id"]),
                    status="FAILED",
                    result={
                        "attempts": list(getattr(proposer, "attempts", ())),
                        "launch_skipped": True,
                    },
                    error=str(exc),
                    latency_ms=(time.monotonic() - proposal_started) * 1000,
                )
                raise
            except Exception as exc:
                raw_path_value = getattr(proposer, "raw_path", None)
                raw_object_id = None
                if isinstance(raw_path_value, Path) and raw_path_value.is_file():
                    raw_object_id = self._store_private_object(
                        raw_path_value.read_bytes()
                    )
                store.finish_proposal_attempt(
                    int(proposal_attempt["id"]),
                    status="FAILED",
                    result={"attempts": list(getattr(proposer, "attempts", ()))},
                    error=f"{type(exc).__name__}: {exc}",
                    raw_output_object_id=raw_object_id,
                    latency_ms=(time.monotonic() - proposal_started) * 1000,
                )
                raise ProposerFailure(f"{type(exc).__name__}: {exc}") from exc
            except BaseException as exc:
                store.finish_proposal_attempt(
                    int(proposal_attempt["id"]),
                    status="UNKNOWN_OUTCOME",
                    result={"attempts": list(getattr(proposer, "attempts", ()))},
                    error=f"{type(exc).__name__}: proposer action was interrupted",
                    latency_ms=(time.monotonic() - proposal_started) * 1000,
                )
                raise
            try:
                bundle = CandidateBundle.single_file(
                    content=proposal.kernel_source,
                    path=target.language.entrypoint,
                    limits=target.language.bundle_limits,
                )
                proposal_v2 = ProposalV2.create(
                    proposal_context_id=proposal_context_id,
                    parent_artifact_id=baseline_ref.artifact_id,
                    hypothesis=proposal.hypothesis,
                    rationale=proposal.rationale,
                    candidate=bundle,
                )
                raw_path_value = getattr(proposer, "raw_path", None)
                raw_object_id = None
                if isinstance(raw_path_value, Path) and raw_path_value.is_file():
                    raw_object_id = self._store_private_object(
                        raw_path_value.read_bytes()
                    )
                with HistoryStore(
                    self.history_db, state_dir=self.config.state_dir
                ) as history:
                    if run.get("workflow_snapshot", {}).get("schema_version") == 2:
                        history.store_candidate_bundle(bundle)
                    history.store_candidate_artifact(
                        proposal.kernel_source,
                        artifact_id=str(
                            ArtifactId.source_sha256(proposal.candidate_hash)
                        ),
                        artifact_kind="source_text_v1",
                        manifest={
                            "format": "python_source_v1",
                            "entrypoint": target.language.entrypoint,
                            "media_type": "text/x-python",
                        },
                    )
                store.finish_proposal_attempt(
                    int(proposal_attempt["id"]),
                    status="SUCCEEDED",
                    result={
                        "proposal": proposal_v2.to_dict(include_content=False),
                        "attempts": list(getattr(proposer, "attempts", ())),
                    },
                    raw_output_object_id=raw_object_id,
                    candidate_artifact_id=str(bundle.artifact_id),
                    latency_ms=(time.monotonic() - proposal_started) * 1000,
                )
            except Exception as exc:
                persisted_attempt = store.get_proposal_attempt(
                    int(proposal_attempt["id"])
                )
                if persisted_attempt["status"] == "RUNNING":
                    store.finish_proposal_attempt(
                        int(proposal_attempt["id"]),
                        status="FAILED",
                        result={
                            "attempts": list(
                                getattr(proposer, "attempts", ())
                            )
                        },
                        error=f"{type(exc).__name__}: {exc}",
                        latency_ms=(time.monotonic() - proposal_started) * 1000,
                    )
                raise ControllerDataIntegrityError(
                    f"proposal evidence could not be persisted: {exc}"
                ) from exc
            recoveries = [
                {
                    "attempt": attempt.get("attempt"),
                    "transport_recovery": attempt.get(
                        "transport_recovery"
                    ),
                    "raw_output_path": attempt.get("raw_output_path"),
                }
                for attempt in attempts
                if attempt.get("transport_recovery") is not None
            ]
            if recoveries:
                store.add_event(
                    run_id,
                    "PROPOSER_TRANSPORT_RECOVERY",
                    {
                        "candidate_hash": proposal.candidate_hash,
                        "attempts": recoveries,
                    },
                    iteration_id=iteration_id,
                )
            if self._candidate_seen(
                store,
                run_id=run_id,
                candidate_hash=proposal.candidate_hash,
            ):
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
            policy = target.validate_candidate(source)
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

        stages = tuple(
            (definition.stage_id, definition.suite_id)
            for definition in target.protocol.stages
            if definition.suite_id is not None
        )
        for stage, suite in stages:
            run = store.get_run(run_id)
            self._validate_run_snapshot(run)
            stage_target = _trusted_target(self._run_namespace(run))
            if stage_target != target:
                raise ControllerDataIntegrityError(
                    "trusted target components changed within a bounded Run"
                )
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
                self._validate_run_snapshot(run)
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
                        run_id,
                        next_index,
                        self._run_baseline(run_id).candidate_hash,
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
                    current = store.get_iteration(int(iteration["id"]))
                    active = current.get("active_container")
                    if active:
                        self.runner.remove_exact_container(
                            self.config.docker_binary, str(active)
                        )
                    store.update_iteration_with_event(
                        int(iteration["id"]),
                        "CONTROLLER_INTERRUPTED",
                        {"kind": "KeyboardInterrupt"},
                        error="controller interrupted by operator",
                        active_container=None,
                    )
                    raise
                except UnknownGPUOutcome as exc:
                    self._finish_iteration(
                        store,
                        int(iteration["id"]),
                        outcome="UNKNOWN_GPU_OUTCOME",
                        error=str(exc),
                    )
                    return store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {
                            "reason": "unknown GPU outcome",
                            "quarantine_candidate": iteration.get(
                                "candidate_hash"
                            ),
                        },
                        status=RunStatus.HARD_FAILED.value,
                        stop_reason=(
                            "GPU execution outcome is unknown; candidate "
                            "must be quarantined and device doctor must pass"
                        ),
                    )
                except ActionBudgetExhausted as exc:
                    # No external container was started for the rejected
                    # action.  Close normally without consuming the
                    # consecutive proposer/controller failure allowance.
                    self._finish_iteration(
                        store,
                        int(iteration["id"]),
                        outcome="BUDGET_EXHAUSTED",
                        error=str(exc),
                    )
                    return store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {
                            "reason": "trusted action cannot fit deadline",
                            "detail": str(exc),
                        },
                        status=RunStatus.BUDGET_EXHAUSTED.value,
                        stop_reason=str(exc),
                    )
                except ControllerDataIntegrityError as exc:
                    self._finish_iteration(
                        store,
                        int(iteration["id"]),
                        outcome="DATA_INTEGRITY_FAILURE",
                        error=str(exc),
                    )
                    return store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": "controller data integrity failure"},
                        status=RunStatus.HARD_FAILED.value,
                        stop_reason=str(exc),
                    )
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

    def start_from_snapshot(
        self,
        *,
        run_id: str,
        workflow_snapshot: Mapping[str, Any],
        baseline_ref: BaselineRef | Mapping[str, Any],
        history_cutoff: str | int | None,
    ) -> dict[str, Any]:
        """Start one Campaign-owned run from an already-resolved snapshot.

        This is intentionally narrower than ``start``: the caller supplies a
        stable run ID and an explicit Campaign baseline, while this trusted
        host still owns preflight, the six-hour/single-run caps, persistence,
        and the existing state machine.  It never adopts the baseline into Git
        or consults History for a replacement parent.
        """

        safe_characters = frozenset(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        )
        if (
            not isinstance(run_id, str)
            or not 1 <= len(run_id) <= 96
            or not (run_id[0].isascii() and run_id[0].isalnum())
            or any(character not in safe_characters for character in run_id)
        ):
            raise ControllerDataIntegrityError(
                "bounded run_id must be a safe ASCII container identifier"
            )
        if not isinstance(workflow_snapshot, Mapping):
            raise ControllerDataIntegrityError(
                "bounded workflow snapshot must be an object"
            )
        snapshot = dict(workflow_snapshot)
        try:
            target_namespace = ResearchNamespace.from_value(
                snapshot.get("namespace")
            )
        except (TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                "bounded workflow snapshot has an invalid namespace"
            ) from exc
        trusted_targets = {
            LEGACY_RESEARCH_NAMESPACE.namespace_id: LEGACY_RESEARCH_NAMESPACE,
            CURRENT_RESEARCH_NAMESPACE.namespace_id: CURRENT_RESEARCH_NAMESPACE,
        }
        trusted_namespace = trusted_targets.get(target_namespace.namespace_id)
        if trusted_namespace is None or target_namespace != trusted_namespace:
            raise ControllerDataIntegrityError(
                "bounded runs support only exact built-in LEGACY/CURRENT targets"
            )
        target_namespace = trusted_namespace
        try:
            frozen_baseline = (
                baseline_ref
                if isinstance(baseline_ref, BaselineRef)
                else BaselineRef.from_value(dict(baseline_ref))
            )
        except (TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                f"bounded baseline reference is invalid: {exc}"
            ) from exc
        if frozen_baseline.source != "campaign":
            raise ControllerDataIntegrityError(
                "bounded runs require a Campaign-owned baseline reference"
            )
        if (
            frozen_baseline.namespace_id
            != target_namespace.namespace_id
        ):
            raise ControllerDataIntegrityError(
                "bounded baseline belongs to another target namespace"
            )
        if isinstance(history_cutoff, bool) or not (
            history_cutoff is None
            or isinstance(history_cutoff, str)
            or (isinstance(history_cutoff, int) and history_cutoff >= 0)
        ):
            raise ControllerDataIntegrityError(
                "bounded History cutoff must be null, a string, or a non-negative integer"
            )
        digest = snapshot.get("snapshot_digest")
        if not isinstance(digest, str):
            raise ControllerDataIntegrityError(
                "bounded workflow snapshot is missing its digest"
            )
        expected_budget = {
            "max_candidates": self.config.max_candidates,
            # ``max_hours`` is a float for legacy CLI compatibility.  A
            # Campaign supplies an integral seconds limit, so use nearest
            # integral seconds instead of truncating a harmless binary-float
            # representation such as 114.99999999999999.
            "max_wall_seconds": round(self.config.max_hours * 3600),
            "max_consecutive_failures": self.config.max_consecutive_failures,
            "stop_after_promotion": self.config.stop_after_promotion,
        }
        if snapshot.get("budget") != expected_budget:
            raise ControllerDataIntegrityError(
                "bounded workflow budget differs from ControllerConfig"
            )
        child_deadline_value = snapshot.get("child_deadline_epoch")
        if (
            isinstance(child_deadline_value, bool)
            or not isinstance(child_deadline_value, (int, float))
            or not math.isfinite(float(child_deadline_value))
        ):
            raise ControllerDataIntegrityError(
                "bounded workflow snapshot has no finite child deadline"
            )
        deadline_epoch = float(child_deadline_value)
        prospective_run = {
            "id": run_id,
            "deadline_epoch": deadline_epoch,
            "namespace_id": frozen_baseline.namespace_id,
            "resolved_config_digest": digest,
            "workflow_snapshot": snapshot,
            "baseline_ref": frozen_baseline.to_dict(),
            "history_cutoff": history_cutoff,
            "config": self.config.redacted_dict(),
        }
        self._validate_run_snapshot(prospective_run)
        baseline = self._baseline_for_ref(
            frozen_baseline,
            namespace_id=target_namespace.namespace_id,
        )

        self._prepare_runtime_dirs()
        self._require_gpu_risk_acknowledgement()
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            with ControllerStore(self.controller_db) as store:
                existing = store.connection.execute(
                    "SELECT 1 FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                if existing is not None:
                    raise ControllerDataIntegrityError(
                        "bounded run_id already has a durable Controller intent"
                    )
            try:
                if not isinstance(self.evaluator, DockerEvaluator):
                    self._authorize_external_action(
                        run_id,
                        action="doctor",
                        configured_timeout=None,
                        run_override=prospective_run,
                    )
                self._preflight_run_overrides[run_id] = prospective_run
                try:
                    preflight = self.doctor(
                        allowed_best_hashes={
                            self.config.expected_kernel_hash,
                            baseline.candidate_hash,
                        },
                        run_id=run_id,
                    )
                finally:
                    self._preflight_run_overrides.pop(run_id, None)
            except ActionBudgetExhausted as exc:
                skipped_preflight = {
                    "status": "SKIPPED_BUDGET_EXHAUSTED",
                    "errors": [str(exc)],
                }
                with ControllerStore(self.controller_db) as store:
                    store.create_run(
                        run_id=run_id,
                        deadline_epoch=deadline_epoch,
                        config=self.config.redacted_dict(),
                        initial_best_hash=baseline.candidate_hash,
                        preflight=skipped_preflight,
                        namespace_id=target_namespace.namespace_id,
                        resolved_config_digest=digest,
                        workflow_snapshot=snapshot,
                        baseline_ref=frozen_baseline.to_dict(),
                        history_cutoff=history_cutoff,
                    )
                    return store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {
                            "reason": "preflight cannot fit child deadline",
                            "detail": str(exc),
                        },
                        status=RunStatus.BUDGET_EXHAUSTED.value,
                        stop_reason=str(exc),
                    )
            if preflight["status"] != "SUCCESS":
                raise ControlledRuntimeError(
                    "controller bounded preflight failed: "
                    + "; ".join(preflight["errors"])
                )
            with ControllerStore(self.controller_db) as store:
                store.create_run(
                    run_id=run_id,
                    deadline_epoch=deadline_epoch,
                    config=self.config.redacted_dict(),
                    initial_best_hash=baseline.candidate_hash,
                    preflight=preflight,
                    namespace_id=target_namespace.namespace_id,
                    resolved_config_digest=digest,
                    workflow_snapshot=snapshot,
                    baseline_ref=frozen_baseline.to_dict(),
                    history_cutoff=history_cutoff,
                )
            return self._run_loop(run_id)

    def _start_new_run(
        self,
        *,
        run_id: str,
        proposal_only: bool,
        snapshot_extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._prepare_runtime_dirs()
        if not proposal_only:
            self._require_gpu_risk_acknowledgement()
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            deployment_pin = self._deployment_pin()
            namespace = (
                LEGACY_RESEARCH_NAMESPACE
                if deployment_pin is None
                else {
                    LEGACY_RESEARCH_NAMESPACE.namespace_id: (
                        LEGACY_RESEARCH_NAMESPACE
                    ),
                    CURRENT_RESEARCH_NAMESPACE.namespace_id: (
                        CURRENT_RESEARCH_NAMESPACE
                    ),
                }[deployment_pin.namespace_id]
            )
            _trusted_target(namespace)
            preflight = self.doctor()
            if preflight["status"] != "SUCCESS":
                raise ControlledRuntimeError(
                    "controller preflight failed: "
                    + "; ".join(preflight["errors"])
                )
            best = self._best()
            history_cutoff = self._history_cutoff()
            baseline_ref = (
                self._baseline_ref(best)
                if deployment_pin is None
                else deployment_pin.baseline_ref
            )
            workflow_snapshot = self._workflow_snapshot(
                baseline=best,
                history_cutoff=history_cutoff,
                namespace=namespace,
                baseline_ref=baseline_ref,
            )
            if snapshot_extra is not None:
                material = dict(workflow_snapshot)
                material.pop("snapshot_digest", None)
                material.update(dict(snapshot_extra))
                workflow_snapshot = {
                    **material,
                    "snapshot_digest": canonical_sha256(material),
                }
            with ControllerStore(self.controller_db) as store:
                store.create_run(
                    run_id=run_id,
                    deadline_epoch=self.clock() + self.config.max_hours * 3600,
                    config=self.config.redacted_dict(),
                    initial_best_hash=best.candidate_hash,
                    preflight=preflight,
                    namespace_id=namespace.namespace_id,
                    resolved_config_digest=str(
                        workflow_snapshot["snapshot_digest"]
                    ),
                    workflow_snapshot=workflow_snapshot,
                    baseline_ref=baseline_ref.to_dict(),
                    history_cutoff=history_cutoff,
                )
            return self._run_loop(run_id, proposal_only=proposal_only)

    def start(self, *, proposal_only: bool = False) -> dict[str, Any]:
        return self._start_new_run(
            run_id=uuid.uuid4().hex,
            proposal_only=proposal_only,
        )

    def start_console_operation(
        self, *, operation_id: str, proposal_only: bool = False
    ) -> dict[str, Any]:
        """Start one deterministic Console-owned Run without duplicate launch."""

        try:
            selected = uuid.UUID(operation_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("operation_id must be a canonical UUID") from exc
        if str(selected) != operation_id:
            raise ValueError("operation_id must be a canonical lowercase UUID")
        if not isinstance(proposal_only, bool):
            raise ValueError("proposal_only must be boolean")
        run_id = "console-run-" + uuid.uuid5(
            uuid.NAMESPACE_URL,
            "kernel-research/console-run/v1/" + operation_id,
        ).hex
        if self.controller_db.exists():
            with ControllerStore(self.controller_db) as store:
                try:
                    existing = store.get_run(run_id)
                except ValueError:
                    existing = None
            if existing is not None:
                self._validate_run_snapshot(existing)
                snapshot = existing.get("workflow_snapshot")
                if (
                    not isinstance(snapshot, Mapping)
                    or snapshot.get("evidence_operation") != "console-run-v1"
                    or snapshot.get("console_operation_id") != operation_id
                    or bool(snapshot.get("proposal_only")) != proposal_only
                ):
                    raise ControllerDataIntegrityError(
                        "Console Run operation identity was reused"
                    )
                if existing["status"] in TERMINAL_RUN_STATUSES:
                    return existing
                return self.resume(run_id)
        return self._start_new_run(
            run_id=run_id,
            proposal_only=proposal_only,
            snapshot_extra={
                "evidence_operation": "console-run-v1",
                "console_operation_id": operation_id,
                "proposal_only": proposal_only,
            },
        )

    def requalify_candidate_for_adoption(
        self,
        *,
        candidate_path: str | os.PathLike[str],
        candidate_hash: str,
        namespace_id: str,
    ) -> dict[str, Any]:
        """Run the one-cycle LEGACY_UNKNOWN-to-resolved adoption bridge."""

        if namespace_id != LEGACY_RESEARCH_NAMESPACE.namespace_id:
            raise ValueError(
                "initial requalification is restricted to the exact LEGACY namespace"
            )
        return self._qualify_candidate_for_adoption(
            candidate_path=candidate_path,
            candidate_hash=candidate_hash,
            namespace=LEGACY_RESEARCH_NAMESPACE,
            operation="adoption-requalification",
        )

    def bootstrap_current_baseline(
        self,
        *,
        candidate_path: str | os.PathLike[str],
        candidate_hash: str,
    ) -> dict[str, Any]:
        """Repeat the pinned improvement chain under the CURRENT protocol.

        The immutable resolved LEGACY deployment proof owns both artifacts:
        its exact parent is first qualified as a new CURRENT baseline, then the
        currently deployed candidate traverses the ordinary CURRENT workflow.
        This method produces evidence only and never publishes deployment
        state.
        """

        return self._qualify_candidate_for_adoption(
            candidate_path=candidate_path,
            candidate_hash=candidate_hash,
            namespace=CURRENT_RESEARCH_NAMESPACE,
            operation="current-baseline-bootstrap",
        )

    def evaluate_manual_candidate(
        self,
        *,
        candidate: CandidateBundle | Mapping[str, Any],
        operation_id: str,
    ) -> dict[str, Any]:
        """Evaluate one operator-supplied CURRENT CandidateBundle exactly once.

        The Console is an evidence producer, not a baseline authority.  This
        entry point therefore reuses the ordinary POLICY/SMOKE/QUICK/FULL and
        CONFIRMATION state machine while freezing the exact deployment pin as
        parent.  A successful promotion decision remains eligible History
        evidence, but this method never writes Deployment or Campaign state.
        """

        try:
            operation_uuid = uuid.UUID(operation_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("operation_id must be a canonical UUID") from exc
        if str(operation_uuid) != operation_id:
            raise ValueError("operation_id must be a canonical lowercase UUID")
        bundle = (
            candidate
            if isinstance(candidate, CandidateBundle)
            else CandidateBundle.from_value(
                dict(candidate) if isinstance(candidate, Mapping) else candidate,
                limits=TRITON_PYTHON_BUNDLE_LIMITS,
            )
        )
        bundle.validate(TRITON_PYTHON_BUNDLE_LIMITS)
        target = _trusted_target(CURRENT_RESEARCH_NAMESPACE)
        if bundle.entrypoint != target.language.entrypoint or len(bundle.files) != 1:
            raise ValueError("manual evaluation requires the exact active entrypoint")
        source = bundle.files[0].content
        source_bytes = source.encode("utf-8")
        candidate_hash = hashlib.sha256(source_bytes).hexdigest()
        policy = target.validate_candidate(source)
        if not policy.valid or policy.sha256 != candidate_hash:
            raise ValueError("manual CandidateBundle fails the trusted target policy")
        if not isinstance(self.evaluator, DockerEvaluator):
            raise ControlledRuntimeError(
                "production manual evaluation requires DockerEvaluator"
            )

        run_uuid = uuid.uuid5(
            uuid.NAMESPACE_URL,
            "kernel-research/console-manual-evaluation/v1/"
            f"{operation_id}/{bundle.artifact_id}",
        )
        run_id = "console-manual-" + run_uuid.hex
        self._prepare_runtime_dirs()
        self._require_gpu_risk_acknowledgement()
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            self._assert_noise_resource_available()

            def manual_container_guard(
                guarded_run_id: str, action: str, timeout: float
            ) -> None:
                self._assert_noise_resource_available()
                self._before_docker_container(guarded_run_id, action, timeout)

            self.evaluator.before_container_start = manual_container_guard
            pin = self._deployment_pin()
            if (
                pin is None
                or pin.namespace_id != CURRENT_RESEARCH_NAMESPACE.namespace_id
                or not pin.execution_environment.is_resolved
            ):
                raise ControlledRuntimeError(
                    "manual evaluation requires a resolved CURRENT deployment pin"
                )
            baseline = self._best()
            if (
                baseline.id != pin.confirmation_experiment_id
                or baseline.experiment_uid != pin.confirmation_experiment_uid
                or baseline.candidate_hash != pin.candidate_hash
                or baseline.artifact_id != str(pin.baseline_ref.artifact_id)
            ):
                raise ControllerDataIntegrityError(
                    "manual evaluation deployment evidence does not match its pin"
                )
            runtime_environment = self._resolved_execution_environment(
                CURRENT_RESEARCH_NAMESPACE
            )
            try:
                pin.execution_environment.require_match(
                    runtime_environment, context="manual evaluation deployment"
                )
                pin.baseline_ref.require_environment(runtime_environment)
            except ValueError as exc:
                raise ControllerDataIntegrityError(str(exc)) from exc
            history_cutoff = self._history_cutoff()
            snapshot = self._workflow_snapshot(
                baseline=baseline,
                history_cutoff=history_cutoff,
                namespace=CURRENT_RESEARCH_NAMESPACE,
                baseline_ref=pin.baseline_ref,
            )
            material = dict(snapshot)
            material.pop("snapshot_digest", None)
            material.update(
                {
                    "evidence_operation": "console-manual-evaluation-v1",
                    "console_operation_id": operation_id,
                    "candidate_artifact_id": str(bundle.artifact_id),
                    "candidate_sha256": candidate_hash,
                }
            )
            snapshot = {**material, "snapshot_digest": canonical_sha256(material)}

            with ControllerStore(self.controller_db) as store:
                try:
                    existing = store.get_run(run_id)
                except ValueError:
                    existing = None
                if existing is not None:
                    self._validate_run_snapshot(existing)
                    persisted_snapshot = existing.get("workflow_snapshot")
                    if (
                        not isinstance(persisted_snapshot, Mapping)
                        or persisted_snapshot.get("evidence_operation")
                        != "console-manual-evaluation-v1"
                        or persisted_snapshot.get("console_operation_id")
                        != operation_id
                        or persisted_snapshot.get("candidate_artifact_id")
                        != str(bundle.artifact_id)
                        or persisted_snapshot.get("candidate_sha256")
                        != candidate_hash
                        or existing["baseline_ref"] != pin.baseline_ref.to_dict()
                    ):
                        raise ControllerDataIntegrityError(
                            "manual evaluation operation identity was reused"
                        )
                    # The operation owns its original History cutoff.  Replaying
                    # after the operation wrote evidence must not derive a new
                    # snapshot from the now-higher live cutoff.
                    snapshot = dict(persisted_snapshot)
                    history_cutoff = int(existing["history_cutoff"])
                    if existing["status"] in TERMINAL_RUN_STATUSES:
                        return {
                            "schema_version": 1,
                            "status": existing["status"],
                            "run_id": run_id,
                            "candidate_hash": candidate_hash,
                            "candidate_artifact_id": str(bundle.artifact_id),
                            "run": existing,
                        }

            preflight = self.doctor()
            if preflight["status"] != "SUCCESS":
                raise ControlledRuntimeError(
                    "manual evaluation preflight failed: "
                    + "; ".join(preflight["errors"])
                )
            run_dir = self.config.controller_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            run_dir.chmod(0o700)
            candidate_dir = run_dir / "candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            candidate_path = candidate_dir / f"{candidate_hash}.py"
            if candidate_path.exists():
                if candidate_path.is_symlink() or candidate_path.read_bytes() != source_bytes:
                    raise ControllerDataIntegrityError(
                        "persisted manual candidate differs from the operation"
                    )
            else:
                candidate_path.write_bytes(source_bytes)
                candidate_path.chmod(0o600)
            with HistoryStore(
                self.history_db, state_dir=self.config.state_dir
            ) as history:
                history.store_candidate_bundle(bundle)
                history.store_candidate_artifact(
                    source,
                    artifact_id=str(ArtifactId.source_sha256(candidate_hash)),
                    artifact_kind="source_text_v1",
                    manifest={
                        "format": "python_source_v1",
                        "entrypoint": target.language.entrypoint,
                        "media_type": "text/x-python",
                    },
                )
            with ControllerStore(self.controller_db) as store:
                try:
                    run = store.get_run(run_id)
                except ValueError:
                    with store.atomic_write():
                        run = store.create_run(
                            run_id=run_id,
                            deadline_epoch=(
                                self.clock() + self.config.max_hours * 3600
                            ),
                            config=self.config.redacted_dict(),
                            initial_best_hash=baseline.candidate_hash,
                            preflight=preflight,
                            namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                            resolved_config_digest=str(snapshot["snapshot_digest"]),
                            workflow_snapshot=snapshot,
                            baseline_ref=pin.baseline_ref.to_dict(),
                            history_cutoff=history_cutoff,
                        )
                        iteration = store.create_iteration(
                            run_id, 1, baseline.candidate_hash
                        )
                        store.accept_candidate(
                            int(iteration["id"]),
                            candidate_hash=candidate_hash,
                            hypothesis=(
                                "operator-supplied Console CandidateBundle"
                            ),
                            rationale=(
                                "one bounded CURRENT manual scientific "
                                "evaluation; no baseline authority"
                            ),
                            candidate_path=str(candidate_path),
                        )
                self._validate_run_snapshot(run)
                iteration = store.latest_iteration(run_id)
                if iteration is None or iteration["status"] != "RUNNING":
                    raise ControllerDataIntegrityError(
                        "manual evaluation has no live deterministic iteration"
                    )
                try:
                    action = self._process_iteration(
                        store=store,
                        run_id=run_id,
                        iteration=iteration,
                        run_dir=run_dir,
                    )
                except UnknownGPUOutcome as exc:
                    self._finish_iteration(
                        store,
                        int(iteration["id"]),
                        outcome="UNKNOWN_GPU_OUTCOME",
                        error=str(exc),
                    )
                    store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": "unknown manual evaluation GPU outcome"},
                        status=RunStatus.HARD_FAILED.value,
                        stop_reason=(
                            "GPU execution outcome is unknown; trusted doctor and "
                            "manual recovery are required"
                        ),
                    )
                except ActionBudgetExhausted as exc:
                    self._finish_iteration(
                        store,
                        int(iteration["id"]),
                        outcome="BUDGET_EXHAUSTED",
                        error=str(exc),
                    )
                    store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": "manual evaluation cannot fit deadline"},
                        status=RunStatus.BUDGET_EXHAUSTED.value,
                        stop_reason=str(exc),
                    )
                except ControllerDataIntegrityError as exc:
                    self._finish_iteration(
                        store,
                        int(iteration["id"]),
                        outcome="DATA_INTEGRITY_FAILURE",
                        error=str(exc),
                    )
                    store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": "manual evaluation data integrity failure"},
                        status=RunStatus.HARD_FAILED.value,
                        stop_reason=str(exc),
                    )
                except BaseException as exc:
                    current = store.get_iteration(int(iteration["id"]))
                    if current["status"] == "RUNNING":
                        attempts = store.list_evaluation_attempts(
                            run_id, iteration_id=int(iteration["id"])
                        )
                        unknown = any(
                            attempt["status"] in {"RUNNING", "UNKNOWN_OUTCOME"}
                            for attempt in attempts
                        )
                        self._finish_iteration(
                            store,
                            int(iteration["id"]),
                            outcome=(
                                "UNKNOWN_GPU_OUTCOME" if unknown else "CONTROLLER_ERROR"
                            ),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        store.update_run_with_event(
                            run_id,
                            "RUN_FINISHED",
                            {"reason": "manual evaluation interrupted"},
                            status=(
                                RunStatus.HARD_FAILED.value
                                if unknown
                                else RunStatus.FAILED.value
                            ),
                            stop_reason=(
                                "trusted doctor/manual recovery required"
                                if unknown
                                else "manual evaluation failed closed"
                            ),
                        )
                    raise
                else:
                    completed = store.get_run(run_id)
                    if completed["status"] == RunStatus.RUNNING.value:
                        finished_iteration = store.get_iteration(int(iteration["id"]))
                        if finished_iteration["status"] != "COMPLETED":
                            self._finish_iteration(
                                store,
                                int(iteration["id"]),
                                outcome="MANUAL_EVALUATION_STOPPED",
                                error="manual evaluation ended without a terminal stage",
                            )
                        completed = store.update_run_with_event(
                            run_id,
                            "RUN_FINISHED",
                            {"reason": "one manual candidate evaluation completed"},
                            status=RunStatus.STOPPED.value,
                            stop_reason="single manual candidate boundary reached",
                            final_best_hash=(
                                candidate_hash if action == "STOP" else baseline.candidate_hash
                            ),
                        )
                completed = store.get_run(run_id)
                return {
                    "schema_version": 1,
                    "status": completed["status"],
                    "run_id": run_id,
                    "namespace_id": CURRENT_RESEARCH_NAMESPACE.namespace_id,
                    "candidate_hash": candidate_hash,
                    "candidate_artifact_id": str(bundle.artifact_id),
                    "execution_environment": runtime_environment.to_dict(),
                    "run": completed,
                }

    def _qualify_candidate_for_adoption(
        self,
        *,
        candidate_path: str | os.PathLike[str],
        candidate_hash: str,
        namespace: ResearchNamespace,
        operation: str,
    ) -> dict[str, Any]:
        """Create fresh resolved evidence without changing deployment state.

        This one-cycle migration bridge first measures the exact accepted
        legacy baseline against itself.  Only that identical-artifact full
        result may seed a resolved BaselineRef.  The supplied candidate then
        traverses the normal POLICY/SMOKE/QUICK/FULL/CONFIRMATION workflow.
        Git and the deployment pin remain administrator-owned and untouched.
        """

        if operation not in {
            "adoption-requalification",
            "current-baseline-bootstrap",
        }:
            raise ValueError(
                "unknown trusted baseline qualification operation"
            )
        current_bootstrap = operation == "current-baseline-bootstrap"
        expected_namespace = (
            CURRENT_RESEARCH_NAMESPACE
            if current_bootstrap
            else LEGACY_RESEARCH_NAMESPACE
        )
        if namespace != expected_namespace:
            raise ValueError("baseline qualification namespace is not exact")
        if (
            not isinstance(candidate_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", candidate_hash) is None
        ):
            raise ValueError("candidate_hash must be 64 lowercase hex digits")
        if not isinstance(self.evaluator, DockerEvaluator):
            raise ControlledRuntimeError(
                "production requalification requires DockerEvaluator"
            )
        self.evaluator.before_container_start = self._before_docker_container
        supplied = Path(candidate_path)
        if supplied.is_symlink():
            raise ValueError("requalification candidate must not be a symlink")
        try:
            supplied = supplied.resolve(strict=True)
            candidate_bytes = supplied.read_bytes()
            candidate_source = candidate_bytes.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(
                f"requalification candidate cannot be read exactly: {exc}"
            ) from exc
        if not supplied.is_file():
            raise ValueError("requalification candidate must be a regular file")
        if hashlib.sha256(candidate_bytes).hexdigest() != candidate_hash:
            raise ValueError("candidate_path bytes do not match candidate_hash")

        target = _trusted_target(namespace)
        policy = target.validate_candidate(candidate_source)
        if not policy.valid or policy.sha256 != candidate_hash:
            raise ValueError("requalification candidate fails trusted policy")

        self._prepare_runtime_dirs()
        self._require_gpu_risk_acknowledgement()
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            self._assert_noise_resource_available()
            pin = self._deployment_pin()
            if pin is None:
                raise ControlledRuntimeError(
                    "baseline qualification requires an explicit deployment pin"
                )
            if current_bootstrap:
                if (
                    pin.namespace_id
                    != LEGACY_RESEARCH_NAMESPACE.namespace_id
                    or not pin.execution_environment.is_resolved
                    or pin.parent_baseline_ref is None
                    or not pin.parent_baseline_ref.execution_environment.is_resolved
                ):
                    raise ControlledRuntimeError(
                        "CURRENT bootstrap requires an exact resolved LEGACY "
                        "deployment proof with its parent baseline"
                    )
            elif pin.execution_environment.is_resolved:
                raise ControlledRuntimeError(
                    "deployment baseline is already resolved; requalification is unnecessary"
                )
            deployed = self._best()
            if current_bootstrap:
                if (
                    candidate_hash != pin.candidate_hash
                    or deployed.id != pin.confirmation_experiment_id
                    or deployed.experiment_uid
                    != pin.confirmation_experiment_uid
                    or deployed.artifact_id != str(pin.baseline_ref.artifact_id)
                ):
                    raise ControllerDataIntegrityError(
                        "CURRENT bootstrap candidate is not the exact deployed proof"
                    )
                repository_candidate = self.config.repository_dir / "kernel.py"
                try:
                    supplied_matches_repository = (
                        supplied == repository_candidate.resolve(strict=True)
                    )
                except OSError as exc:
                    raise ControllerDataIntegrityError(
                        "deployed kernel.py cannot be resolved"
                    ) from exc
                if not supplied_matches_repository:
                    raise ValueError(
                        "CURRENT bootstrap candidate_path must be the deployed kernel.py"
                    )
                promotion = deployed.result.get("promotion")
                source_id = (
                    promotion.get("baseline_experiment_id")
                    if isinstance(promotion, Mapping)
                    else None
                )
                if type(source_id) is not int:
                    raise ControllerDataIntegrityError(
                        "deployment proof has no exact parent baseline experiment"
                    )
                with HistoryStore(
                    self.history_db, state_dir=self.config.state_dir
                ) as history:
                    baseline = history.get_experiment(source_id)
                try:
                    deployed_identity = ExperimentIdentity.from_value(
                        dict(deployed.identity)
                    )
                    baseline_identity = (
                        None
                        if baseline is None
                        else ExperimentIdentity.from_value(dict(baseline.identity))
                    )
                except (TypeError, ValueError) as exc:
                    raise ControllerDataIntegrityError(
                        "deployment proof has invalid resolved identities"
                    ) from exc
                if (
                    baseline is None
                    or baseline_identity is None
                    or deployed_identity.namespace
                    != LEGACY_RESEARCH_NAMESPACE
                    or baseline_identity.namespace
                    != LEGACY_RESEARCH_NAMESPACE
                    or deployed_identity.baseline != pin.parent_baseline_ref
                    or baseline.artifact_id
                    != str(pin.parent_baseline_ref.artifact_id)
                    or baseline_identity.candidate_artifact_id
                    != pin.parent_baseline_ref.artifact_id
                    or baseline_identity.execution_environment
                    != pin.parent_baseline_ref.execution_environment
                    or deployed.baseline_experiment_uid
                    != baseline.experiment_uid
                    or baseline.status != "SUCCESS"
                    or not baseline.promotable
                ):
                    raise ControllerDataIntegrityError(
                        "deployment proof does not retain its exact resolved parent"
                    )
            else:
                baseline = deployed
            legacy_ref = pin.baseline_ref
            if not current_bootstrap and legacy_ref.execution_environment.is_resolved:
                raise ControllerDataIntegrityError(
                    "legacy requalification parent unexpectedly has resolved evidence"
                )
            runtime_environment = self._resolved_execution_environment(
                namespace
            )
            if current_bootstrap:
                try:
                    pin.execution_environment.require_match(
                        runtime_environment,
                        context="CURRENT bootstrap deployment",
                    )
                    pin.parent_baseline_ref.require_environment(
                        runtime_environment
                    )
                except ValueError as exc:
                    raise ControllerDataIntegrityError(str(exc)) from exc
            baseline_path = self._experiment_source_path(baseline)
            baseline_source = baseline_path.read_text(encoding="utf-8")
            preflight = self.doctor(run_id="preflight")
            if preflight["status"] != "SUCCESS":
                raise ControlledRuntimeError(
                    "baseline qualification preflight failed: "
                    + "; ".join(preflight["errors"])
                )
            qualification_uid = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"kernel-research/{operation}/baseline-qualification/v1/"
                    f"{baseline.experiment_uid}/{runtime_environment.digest}/"
                    f"{candidate_hash}",
                )
            )
            qualification_ref = BaselineRef.create(
                namespace=namespace,
                artifact_id=baseline.artifact_id,
                source="deployment",
                revision=f"qualification-{qualification_uid}",
                execution_environment=runtime_environment,
            )
            bundle = CandidateBundle.single_file(
                content=candidate_source,
                path=target.language.entrypoint,
                limits=target.language.bundle_limits,
            )
            operation_uid = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"kernel-research/{operation}/v1/"
                f"{qualification_uid}/{bundle.artifact_id}",
            )
            run_id = (
                "current-bootstrap-" if current_bootstrap else "requal-"
            ) + operation_uid.hex
            with ControllerStore(self.controller_db) as active_store:
                active_rows = active_store.connection.execute(
                    "SELECT id FROM runs WHERE status = 'RUNNING' AND id != ?",
                    (run_id,),
                ).fetchall()
                try:
                    existing_run = active_store.get_run(run_id)
                except ValueError:
                    existing_run = None
            if active_rows:
                raise ControlledRuntimeError(
                    "requalification is forbidden while another Run is active"
                )
            history_cutoff = (
                self._history_cutoff()
                if existing_run is None
                else existing_run["history_cutoff"]
            )
            snapshot = self._workflow_snapshot(
                baseline=baseline,
                history_cutoff=history_cutoff,
                namespace=namespace,
                baseline_ref=qualification_ref,
            )
            material = dict(snapshot)
            material.pop("snapshot_digest", None)
            material.update(
                {
                    "evidence_operation": operation,
                    "source_experiment_uid": baseline.experiment_uid,
                    "source_namespace_id": baseline.namespace_id,
                    "qualification_experiment_uid": qualification_uid,
                    "candidate_artifact_id": str(bundle.artifact_id),
                }
            )
            snapshot = {
                **material,
                "snapshot_digest": canonical_sha256(material),
            }
            run_dir = self.config.controller_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            run_dir.chmod(0o700)
            candidate_dir = run_dir / "candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_file = candidate_dir / f"{candidate_hash}.py"
            if candidate_file.exists():
                if (
                    candidate_file.is_symlink()
                    or candidate_file.read_bytes() != candidate_bytes
                ):
                    raise ControllerDataIntegrityError(
                        "persisted requalification candidate differs from the request"
                    )
            else:
                candidate_file.write_bytes(candidate_bytes)
                candidate_file.chmod(0o600)
            with HistoryStore(
                self.history_db, state_dir=self.config.state_dir
            ) as history:
                history.store_candidate_bundle(bundle)
                history.store_candidate_artifact(
                    candidate_source,
                    artifact_id=str(ArtifactId.source_sha256(candidate_hash)),
                    artifact_kind="source_text_v1",
                    manifest={
                        "format": "python_source_v1",
                        "entrypoint": target.language.entrypoint,
                        "media_type": "text/x-python",
                    },
                )

            qualification_identity = ExperimentIdentity.create(
                experiment_uid=qualification_uid,
                namespace=namespace,
                mode="DISCOVERY",
                candidate_artifact_id=baseline.artifact_id,
                parent_artifact_id=baseline.artifact_id,
                baseline=qualification_ref,
                execution_environment=runtime_environment,
                stage="baseline_qualification",
                suite="full",
                replicate_kind="qualification",
                replicate_index=0,
                proposer_profile=None,
                prompt_digest=None,
                feedback_digest=None,
                history_cutoff=history_cutoff,
                run_id=run_id,
                iteration=1,
            )
            with ControllerStore(self.controller_db) as store:
                try:
                    run = store.get_run(run_id)
                except ValueError:
                    run = store.create_run(
                        run_id=run_id,
                        deadline_epoch=self.clock()
                        + self.config.max_hours * 3600,
                        config=self.config.redacted_dict(),
                        initial_best_hash=baseline.candidate_hash,
                        preflight=preflight,
                        namespace_id=namespace.namespace_id,
                        resolved_config_digest=str(snapshot["snapshot_digest"]),
                        workflow_snapshot=snapshot,
                        baseline_ref=qualification_ref.to_dict(),
                        history_cutoff=history_cutoff,
                    )
                if run["workflow_snapshot"] != snapshot:
                    raise ControllerDataIntegrityError(
                        "requalification run identity conflicts with its deterministic ID"
                    )
                iterations = store.list_iterations(run_id)
                iteration = (
                    store.create_iteration(run_id, 1, baseline.candidate_hash)
                    if not iterations
                    else iterations[0]
                )
                if len(iterations) > 1:
                    raise ControllerDataIntegrityError(
                        "requalification run has multiple iterations"
                    )
                attempts = [
                    item
                    for item in store.list_evaluation_attempts(
                        run_id, iteration_id=int(iteration["id"])
                    )
                    if item["stage"] == "baseline_qualification"
                ]
                if len(attempts) > 1:
                    raise ControllerDataIntegrityError(
                        "requalification has multiple baseline qualification attempts"
                    )
                attempt = (
                    attempts[0]
                    if attempts
                    else store.create_evaluation_attempt(
                        experiment_uid=qualification_uid,
                        run_id=run_id,
                        iteration_id=int(iteration["id"]),
                        stage="baseline_qualification",
                        suite="full",
                        replicate_kind="qualification",
                        replicate_index=0,
                        candidate_artifact_id=baseline.artifact_id,
                        parent_artifact_id=baseline.artifact_id,
                        baseline_ref=qualification_ref.to_dict(),
                        condition_digest=qualification_identity.condition_digest,
                        request=qualification_identity.to_dict(),
                    )
                )
                with HistoryStore(
                    self.history_db, state_dir=self.config.state_dir
                ) as history:
                    qualification = history.get_experiment_by_uid(
                        qualification_uid
                    )
                raw_path = (
                    run_dir / "results" / "001" / "baseline_qualification.validated.json"
                )
                if qualification is None:
                    launched = False
                    if attempt["status"] == "RUNNING":
                        if not raw_path.is_file():
                            store.finish_evaluation_attempt(
                                qualification_uid,
                                status="UNKNOWN_OUTCOME",
                                result={},
                                error=(
                                    "controller restarted without complete baseline "
                                    "qualification evidence"
                                ),
                            )
                            raise UnknownGPUOutcome(
                                "baseline qualification has an unknown GPU outcome"
                            )
                        try:
                            raw = _strict_json_object_bytes(
                                raw_path.read_bytes(),
                                label="persisted qualification evidence",
                                max_bytes=EVALUATOR_OUTPUT_LIMIT_BYTES,
                            )
                        except (
                            OSError,
                            ValueError,
                            TypeError,
                            RecursionError,
                        ) as exc:
                            store.finish_evaluation_attempt(
                                qualification_uid,
                                status="UNKNOWN_OUTCOME",
                                result={},
                                error=f"persisted qualification evidence is invalid: {exc}",
                            )
                            raise UnknownGPUOutcome(
                                "baseline qualification has invalid persisted GPU evidence"
                            ) from exc
                    elif attempt["status"] == "PENDING":
                        if raw_path.exists():
                            raise ControllerDataIntegrityError(
                                "pending baseline qualification has unexpected "
                                "persisted evidence"
                            )
                        store.start_evaluation_attempt(qualification_uid)
                        launched = True
                        try:
                            raw = self.evaluator.evaluate(
                                candidate_path=baseline_path,
                                suite="full",
                                baseline_path=baseline_path,
                                run_id=run_id,
                                iteration_index=1,
                                stage="baseline_qualification",
                                request_identity=qualification_identity.to_dict(),
                            )
                        except BaseException as exc:
                            store.finish_evaluation_attempt(
                                qualification_uid,
                                status="UNKNOWN_OUTCOME",
                                result={},
                                error=(
                                    f"{type(exc).__name__}: qualification action "
                                    "interrupted"
                                ),
                            )
                            raise
                    elif attempt["status"] == "UNKNOWN_OUTCOME":
                        raise UnknownGPUOutcome(
                            "baseline qualification has an unknown GPU outcome "
                            "and cannot restart"
                        )
                    else:
                        raise ControllerDataIntegrityError(
                            "terminal qualification attempt has no History evidence"
                        )
                    try:
                        self._validate_evaluator_echo(
                            raw,
                            identity=qualification_identity,
                            candidate_hash=baseline.candidate_hash,
                            suite="full",
                            baseline_hash=baseline.candidate_hash,
                            backend=target.device.evaluator_backend(),
                        )
                    except Exception as exc:
                        store.finish_evaluation_attempt(
                            qualification_uid,
                            status="UNKNOWN_OUTCOME",
                            result=dict(raw),
                            error=f"qualification evaluator echo is invalid: {exc}",
                        )
                        raise UnknownGPUOutcome(
                            "baseline qualification returned untrusted GPU evidence"
                        ) from exc
                    if launched:
                        try:
                            encoded = (
                                json.dumps(
                                    raw,
                                    sort_keys=True,
                                    ensure_ascii=False,
                                    indent=2,
                                    allow_nan=False,
                                )
                                + "\n"
                            ).encode("utf-8")
                            _atomic_write_bytes(raw_path, encoded)
                        except (OSError, TypeError, ValueError) as exc:
                            store.finish_evaluation_attempt(
                                qualification_uid,
                                status="UNKNOWN_OUTCOME",
                                result={},
                                error=(
                                    "qualification evidence could not be persisted: "
                                    f"{exc}"
                                ),
                            )
                            raise UnknownGPUOutcome(
                                "baseline qualification evidence is not durable"
                            ) from exc
                    try:
                        recorder = (
                            record_current_baseline_bootstrap_result
                            if current_bootstrap
                            else record_baseline_qualification_result
                        )
                        recorder_arguments: dict[str, Any] = {
                            "candidate_source": baseline_source,
                            "result": raw,
                            "state_dir": self.config.state_dir,
                            "note": f"{operation}:{run_id}:baseline",
                            "identity": qualification_identity,
                            "git_revision": self.config.expected_git_commit[:12],
                        }
                        recorder_arguments[
                            "legacy_source_experiment_id"
                            if current_bootstrap
                            else "baseline_experiment_id"
                        ] = baseline.id
                        qualification = recorder(**recorder_arguments)
                    except Exception as exc:
                        store.finish_evaluation_attempt(
                            qualification_uid,
                            status="FAILED",
                            result=dict(raw),
                            error=f"qualification evidence could not be recorded: {exc}",
                        )
                        raise ControllerDataIntegrityError(
                            "trusted baseline qualification evidence could not be "
                            "recorded"
                        ) from exc
                if qualification is None:  # pragma: no cover
                    raise ControllerDataIntegrityError(
                        "baseline qualification record disappeared"
                    )
                self._validate_uid_reconciliation(
                    persisted=qualification,
                    identity=qualification_identity,
                    attempt=attempt,
                    candidate_hash=baseline.candidate_hash,
                    backend=target.device.evaluator_backend(),
                    suite="full",
                    stage="baseline_qualification",
                )
                if attempt["status"] == "PENDING":
                    store.start_evaluation_attempt(qualification_uid)
                current_attempt = store.get_evaluation_attempt_by_uid(
                    qualification_uid
                )
                if current_attempt is not None and current_attempt["status"] == "RUNNING":
                    store.finish_evaluation_attempt(
                        qualification_uid,
                        status="SUCCEEDED",
                        result=dict(qualification.result),
                        error=qualification.error_summary,
                    )
                store.link_history_experiment(
                    qualification_uid, qualification.id
                )
                if (
                    qualification.status != "SUCCESS"
                    or not qualification.promotable
                    or qualification.artifact_id != baseline.artifact_id
                    or qualification.result.get("promotion", {}).get("phase")
                    != "baseline"
                ):
                    store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": "baseline qualification failed"},
                        status="FAILED",
                        stop_reason="baseline qualification failed",
                    )
                    raise ControlledRuntimeError(
                        "baseline qualification did not produce eligible evidence"
                    )
                iteration = store.get_iteration(int(iteration["id"]))
                experiment_ids = dict(iteration.get("experiment_ids") or {})
                experiment_ids["baseline_qualification"] = qualification.id
                if not iteration.get("candidate_hash"):
                    store.update_iteration_with_event(
                        int(iteration["id"]),
                        "BASELINE_QUALIFIED",
                        {
                            "experiment_id": qualification.id,
                            "experiment_uid": qualification.experiment_uid,
                        },
                        experiment_ids=experiment_ids,
                    )
                    iteration = store.accept_candidate(
                        int(iteration["id"]),
                        candidate_hash=candidate_hash,
                        hypothesis=(
                            "operator-selected CURRENT baseline bootstrap"
                            if current_bootstrap
                            else "operator-selected candidate requalification"
                        ),
                        rationale=(
                            "repeat the immutable deployed improvement under the "
                            "CURRENT protocol"
                            if current_bootstrap
                            else "fresh resolved evidence required after legacy migration"
                        ),
                        candidate_path=str(candidate_file),
                    )
                outcome = self._process_iteration(
                    store=store,
                    run_id=run_id,
                    iteration=store.get_iteration(int(iteration["id"])),
                    run_dir=run_dir,
                )
                completed = store.get_run(run_id)
                if completed["status"] == "RUNNING" and outcome != "STOP":
                    completed = store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {"reason": "requalification candidate did not promote"},
                        status="FAILED",
                        stop_reason="requalification candidate did not promote",
                    )
                return {
                    "schema_version": 1,
                    "status": completed["status"],
                    "run_id": run_id,
                    "namespace_id": namespace.namespace_id,
                    "evaluation_protocol": namespace.evaluation_protocol.to_dict(),
                    "qualification_experiment_id": qualification.id,
                    "qualification_experiment_uid": qualification.experiment_uid,
                    "candidate_hash": candidate_hash,
                    "execution_environment": runtime_environment.to_dict(),
                    "run": completed,
                }

    def collect_noise(
        self,
        *,
        candidate_path: str | os.PathLike[str],
        namespace_id: str,
        baseline_experiment_uid: str,
        replicate_index: int,
    ) -> ExperimentRecord:
        """Collect one bounded, deterministic, non-promotable NOISE result.

        This deliberately uses the normal Docker evaluator and Controller
        attempt ledger.  The caller supplies no runner, image, protocol,
        backend, suite, timeout, BaselineRef, or promotion authority.
        """

        if namespace_id != CURRENT_RESEARCH_NAMESPACE.namespace_id:
            raise ValueError(
                "noise collection requires the exact built-in CURRENT namespace"
            )
        if (
            not isinstance(baseline_experiment_uid, str)
            or not baseline_experiment_uid
        ):
            raise ValueError("baseline_experiment_uid must be non-empty")
        if (
            type(replicate_index) is not int
            or replicate_index < 0
            or replicate_index > SQLITE_MAX_INT
        ):
            raise ValueError(
                "replicate_index must fit a non-negative SQLite INTEGER"
            )
        timeout = self.config.evaluator_timeout_sec
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
            or float(timeout) > MAX_EVALUATOR_TIMEOUT_SEC
        ):
            raise ValueError(
                "configured evaluator timeout is outside the fixed safety bound"
            )
        if not isinstance(self.evaluator, DockerEvaluator):
            raise ControlledRuntimeError(
                "production noise collection requires DockerEvaluator"
            )
        # An injected Docker adapter may be useful in tests, but it cannot
        # weaken the production launch-time Campaign/deadline guard.
        self.evaluator.before_container_start = self._before_docker_container

        supplied = Path(candidate_path)
        if supplied.is_symlink():
            raise ValueError("noise candidate must not be a symlink")
        try:
            supplied = supplied.resolve(strict=True)
            supplied_bytes = supplied.read_bytes()
            supplied_source = supplied_bytes.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"noise candidate cannot be read exactly: {exc}") from exc
        if not supplied.is_file():
            raise ValueError("noise candidate must be a regular file")

        deterministic_uid = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "kernel-research/noise/v1/"
                f"{namespace_id}/{baseline_experiment_uid}/{replicate_index}",
            )
        )
        run_id = "noise-" + uuid.UUID(deterministic_uid).hex
        self._prepare_runtime_dirs()
        self._require_gpu_risk_acknowledgement()
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            # First Campaign check is before any new durable noise intent.  The
            # Docker callback performs the same check again immediately before
            # runner.run while this canonical lock remains held.
            self._assert_noise_resource_available()
            with HistoryStore(
                self.history_db, state_dir=self.config.state_dir
            ) as history:
                baseline = history.get_experiment_by_uid(
                    baseline_experiment_uid
                )
                if baseline is None:
                    raise ValueError("frozen noise baseline does not exist")
                try:
                    baseline_identity, authoritative_bytes = (
                        trusted_noise_baseline_source(history, baseline)
                    )
                except (TypeError, ValueError) as exc:
                    raise ControllerDataIntegrityError(str(exc)) from exc
            if supplied_bytes != authoritative_bytes:
                raise ValueError(
                    "noise candidate bytes differ from the frozen baseline artifact"
                )
            target = _trusted_target(CURRENT_RESEARCH_NAMESPACE)
            policy = target.validate_candidate(supplied_source)
            if not policy.valid or policy.sha256 != baseline.candidate_hash:
                raise ValueError(
                    "noise candidate does not satisfy the frozen language/operator policy"
                )
            runtime_environment = self._resolved_execution_environment(
                CURRENT_RESEARCH_NAMESPACE
            )
            try:
                baseline_identity.execution_environment.require_match(
                    runtime_environment, context="noise runtime"
                )
            except ValueError as exc:
                raise ControllerDataIntegrityError(str(exc)) from exc
            baseline_ref = BaselineRef.create(
                namespace=CURRENT_RESEARCH_NAMESPACE,
                artifact_id=baseline_identity.candidate_artifact_id,
                source="deployment",
                revision=f"history-{baseline.id}",
                execution_environment=runtime_environment,
            )
            snapshot = self._workflow_snapshot(
                baseline=baseline,
                history_cutoff=baseline.id,
                namespace=CURRENT_RESEARCH_NAMESPACE,
                baseline_ref=baseline_ref,
            )
            snapshot_material = dict(snapshot)
            snapshot_material.pop("snapshot_digest", None)
            snapshot_material.update(
                {
                    "evidence_operation": "noise-collect",
                    "noise_baseline_experiment_uid": baseline.experiment_uid,
                    "noise_replicate_index": replicate_index,
                }
            )
            snapshot = {
                **snapshot_material,
                "snapshot_digest": canonical_sha256(snapshot_material),
            }
            identity = ExperimentIdentity.create(
                experiment_uid=deterministic_uid,
                namespace=CURRENT_RESEARCH_NAMESPACE,
                mode="DISCOVERY",
                candidate_artifact_id=baseline_identity.candidate_artifact_id,
                parent_artifact_id=baseline_identity.candidate_artifact_id,
                baseline=baseline_ref,
                execution_environment=runtime_environment,
                stage="NOISE",
                suite="full",
                replicate_kind="noise",
                replicate_index=replicate_index,
                proposer_profile=None,
                prompt_digest=None,
                feedback_digest=None,
                history_cutoff=baseline.id,
                run_id=run_id,
                iteration=replicate_index,
            )
            trusted_source_path = self._experiment_source_path(baseline)
            if trusted_source_path.read_bytes() != authoritative_bytes:
                raise ControllerDataIntegrityError(
                    "materialized baseline source changed before evaluation"
                )

            with ControllerStore(self.controller_db) as store:
                existing_run = store.connection.execute(
                    "SELECT 1 FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                self._assert_noise_attempt_ledger_clear(
                    store, current_experiment_uid=deterministic_uid
                )
                if existing_run is not None:
                    attempt = store.get_evaluation_attempt_by_uid(
                        deterministic_uid
                    )
                    if attempt is not None and attempt["status"] == "RUNNING":
                        store.finish_evaluation_attempt(
                            deterministic_uid,
                            status="UNKNOWN_OUTCOME",
                            result={},
                            error=(
                                "noise collector restarted after GPU execution began"
                            ),
                        )
                        prior_run = store.get_run(run_id)
                        if prior_run["status"] == RunStatus.RUNNING.value:
                            store.update_run_with_event(
                                run_id,
                                "RUN_FINISHED",
                                {"reason": "unknown noise GPU outcome"},
                                status=RunStatus.HARD_FAILED.value,
                                stop_reason="unknown noise GPU outcome",
                            )
                        raise self._noise_recovery_required(
                            "this deterministic noise attempt has UNKNOWN_GPU_OUTCOME"
                        )
                    if attempt is not None:
                        attempt_result = attempt.get("result")
                        attempt_result = (
                            attempt_result
                            if isinstance(attempt_result, Mapping)
                            else {}
                        )
                        attempt_error = str(attempt.get("error") or "").lower()
                        requires_recovery = (
                            attempt["status"]
                            in {"PENDING", "UNKNOWN_OUTCOME", "FAILED"}
                            or attempt_result.get("status") in HARD_STATUSES
                            or any(
                                marker in attempt_error
                                for marker in FATAL_GPU_MARKERS
                            )
                        )
                        if requires_recovery:
                            raise self._noise_recovery_required(
                                "this deterministic noise attempt is unresolved or hard"
                            )
                    raise ControlledRuntimeError(
                        "replicate_index already has a durable deterministic "
                        "noise intent; duplicate collection is forbidden"
                    )

                store.create_run(
                    run_id=run_id,
                    deadline_epoch=(
                        self.clock() + float(self.config.evaluator_timeout_sec) + 60.0
                    ),
                    config=self.config.redacted_dict(),
                    initial_best_hash=baseline.candidate_hash,
                    preflight={
                        "status": "NOT_REQUIRED",
                        "operation": "noise-collect",
                    },
                    namespace_id=namespace_id,
                    resolved_config_digest=str(snapshot["snapshot_digest"]),
                    workflow_snapshot=snapshot,
                    baseline_ref=baseline_ref.to_dict(),
                    history_cutoff=baseline.id,
                )
                iteration = store.create_iteration(
                    run_id, replicate_index, baseline.candidate_hash
                )
                iteration = store.update_iteration(
                    int(iteration["id"]),
                    candidate_hash=baseline.candidate_hash,
                    candidate_path=str(trusted_source_path),
                    hypothesis="same-artifact null remeasurement",
                    rationale="R1 independent c500/full noise evidence",
                )
                store.create_evaluation_attempt(
                    experiment_uid=deterministic_uid,
                    run_id=run_id,
                    iteration_id=int(iteration["id"]),
                    stage="NOISE",
                    suite="full",
                    replicate_kind="noise",
                    replicate_index=replicate_index,
                    candidate_artifact_id=str(identity.candidate_artifact_id),
                    parent_artifact_id=str(identity.parent_artifact_id),
                    baseline_ref=baseline_ref.to_dict(),
                    condition_digest=identity.condition_digest,
                    request=identity.to_dict(),
                )
                try:
                    record = self._evaluate_stage(
                        store=store,
                        run_id=run_id,
                        iteration=iteration,
                        source=authoritative_bytes.decode("utf-8"),
                        candidate_path=trusted_source_path,
                        suite="full",
                        stage="NOISE",
                    )
                except BaseException:
                    attempt = store.get_evaluation_attempt_by_uid(
                        deterministic_uid
                    )
                    run = store.get_run(run_id)
                    if run["status"] == RunStatus.RUNNING.value:
                        unknown = (
                            attempt is not None
                            and attempt["status"] == "UNKNOWN_OUTCOME"
                        )
                        self._finish_iteration(
                            store,
                            int(iteration["id"]),
                            outcome=(
                                "UNKNOWN_GPU_OUTCOME" if unknown else "FAILED"
                            ),
                            error=(
                                None if attempt is None else attempt.get("error")
                            ),
                        )
                        store.update_run_with_event(
                            run_id,
                            "RUN_FINISHED",
                            {
                                "reason": (
                                    "unknown noise GPU outcome"
                                    if unknown
                                    else "noise collection failed closed"
                                )
                            },
                            status=(
                                RunStatus.HARD_FAILED.value
                                if unknown
                                else RunStatus.FAILED.value
                            ),
                            stop_reason=(
                                "trusted doctor/manual recovery required"
                                if unknown
                                else "noise collection failed closed"
                            ),
                        )
                    raise
                hard = self._hard_failure(record)
                self._finish_iteration(
                    store,
                    int(iteration["id"]),
                    outcome="HARD_FAILURE" if hard else "NOISE_RECORDED",
                    result=record.result,
                    error=record.error_summary,
                )
                store.update_run_with_event(
                    run_id,
                    "RUN_FINISHED",
                    {
                        "reason": (
                            "hard noise GPU outcome"
                            if hard
                            else "bounded noise evidence recorded"
                        )
                    },
                    status=(
                        RunStatus.HARD_FAILED.value
                        if hard
                        else RunStatus.STOPPED.value
                    ),
                    stop_reason=(
                        "trusted doctor/manual recovery required"
                        if hard
                        else "one bounded noise replicate completed"
                    ),
                    final_best_hash=baseline.candidate_hash,
                )
                return record

    def resume(self, run_id: str) -> dict[str, Any]:
        with gpu_lock(self.config.controller_dir / "gpu1.lock"):
            with ControllerStore(self.controller_db) as store:
                run = store.get_run(run_id)
                self._validate_run_snapshot(run)
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
            try:
                if not isinstance(self.evaluator, DockerEvaluator):
                    self._authorize_external_action(
                        run_id,
                        action="doctor",
                        configured_timeout=None,
                    )
                preflight = self.doctor(
                    allowed_best_hashes=allowed_best_hashes,
                    run_id=run_id,
                )
            except ActionBudgetExhausted as exc:
                with ControllerStore(self.controller_db) as store:
                    return store.update_run_with_event(
                        run_id,
                        "RUN_FINISHED",
                        {
                            "reason": "resume preflight cannot fit deadline",
                            "detail": str(exc),
                        },
                        status=RunStatus.BUDGET_EXHAUSTED.value,
                        stop_reason=str(exc),
                    )
            if preflight["status"] != "SUCCESS":
                raise ControlledRuntimeError(
                    "controller resume preflight failed: "
                    + "; ".join(preflight["errors"])
                )
            workflow = run.get("workflow_snapshot")
            proposal_only = bool(
                isinstance(workflow, Mapping)
                and workflow.get("evidence_operation") == "console-run-v1"
                and workflow.get("proposal_only") is True
            )
            return self._run_loop(run_id, proposal_only=proposal_only)

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
            with ControllerStore(self.controller_db) as store:
                frozen_run = store.get_run(run_id)
            if frozen_run.get("config") != self.config.redacted_dict():
                raise ControlledRuntimeError(
                    "checkpoint config differs from the frozen Controller Run"
                )
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
                campaign_database = _campaign_database_for_checkpoint(
                    self.config, run
                )
                campaign_summary: dict[str, Any] | None = None
                if campaign_database is not None:
                    campaign_source, campaign_id = campaign_database
                    campaign_backup = (
                        temporary_destination / "campaign.sqlite3"
                    )
                    _sqlite_backup(campaign_source, campaign_backup)
                    campaign_connection = sqlite3.connect(campaign_backup)
                    campaign_connection.row_factory = sqlite3.Row
                    try:
                        campaign_row = campaign_connection.execute(
                            """
                            SELECT namespace_id, mode, snapshot_digest
                            FROM campaigns WHERE id = ?
                            """,
                            (campaign_id,),
                        ).fetchone()
                        baseline_ref = BaselineRef.from_value(
                            run["baseline_ref"]
                        )
                        child_snapshot = run["workflow_snapshot"].get(
                            "campaign_child"
                        )
                        child_row = campaign_connection.execute(
                            """
                            SELECT id, child_index, campaign_id,
                                   baseline_revision_id,
                                   proposer_profile_json,
                                   max_candidates, max_wall_seconds,
                                   max_consecutive_failures,
                                   stop_after_promotion, status
                            FROM child_runs
                            WHERE controller_run_id = ?
                            """,
                            (run_id,),
                        ).fetchone()
                        baseline_revision_id = (
                            child_snapshot.get("baseline_revision_id")
                            if isinstance(child_snapshot, Mapping)
                            else None
                        )
                        if baseline_revision_id is None and child_row is not None:
                            baseline_revision_id = child_row[
                                "baseline_revision_id"
                            ]
                        baseline_row = campaign_connection.execute(
                            """
                            SELECT namespace_id, artifact_id
                            FROM baseline_revisions
                            WHERE id = ? AND campaign_id = ?
                            """,
                            (baseline_revision_id, campaign_id),
                        ).fetchone()
                        child_matches = (
                            isinstance(child_snapshot, Mapping)
                            and child_row is not None
                            and int(child_row["id"])
                            == child_snapshot.get("child_id")
                            and int(child_row["child_index"])
                            == child_snapshot.get("child_index")
                            and str(child_row["campaign_id"]) == campaign_id
                            and str(child_row["baseline_revision_id"])
                            == baseline_revision_id
                            and json.loads(
                                str(child_row["proposer_profile_json"])
                            )
                            == run["workflow_snapshot"].get(
                                "proposer_profile"
                            )
                            and int(child_row["max_candidates"])
                            == int(run["config"]["max_candidates"])
                            and int(child_row["max_wall_seconds"])
                            == round(float(run["config"]["max_hours"]) * 3600)
                            and int(child_row["max_consecutive_failures"])
                            == int(
                                run["config"]["max_consecutive_failures"]
                            )
                            and bool(child_row["stop_after_promotion"])
                            and str(child_row["status"])
                            in {"RUNNING", "PROMOTED"}
                        )
                        if (
                            campaign_row is None
                            or baseline_row is None
                            or not child_matches
                            or str(campaign_row["namespace_id"])
                            != str(run["namespace_id"])
                            or str(campaign_row["mode"])
                            != str(run["workflow_snapshot"].get("mode"))
                            or str(campaign_row["snapshot_digest"])
                            != str(
                                run["workflow_snapshot"].get(
                                    "campaign_snapshot_digest"
                                )
                            )
                            or str(baseline_row["namespace_id"])
                            != str(run["namespace_id"])
                            or str(baseline_row["artifact_id"])
                            != str(baseline_ref.artifact_id)
                        ):
                            raise ControlledRuntimeError(
                                "Campaign checkpoint does not prove the frozen "
                                "campaign child and baseline revision"
                            )
                    finally:
                        campaign_connection.close()
                    campaign_summary = _sqlite_summary(
                        campaign_backup,
                        tables=(
                            "campaigns",
                            "baseline_revisions",
                            "child_runs",
                            "budget_actions",
                            "resource_leases",
                            "oj_nominations",
                            "outbox",
                            "soak_generations",
                            "soak_violations",
                        ),
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
                scientific_objects = self.config.state_dir / "objects"
                if scientific_objects.exists():
                    shutil.copytree(
                        scientific_objects,
                        temporary_destination / "state_objects",
                    )
                controller_objects = self.config.controller_dir / "objects"
                if controller_objects.exists():
                    shutil.copytree(
                        controller_objects,
                        temporary_destination / "controller_objects",
                    )
                (temporary_destination / "resolved-config.json").write_text(
                    json.dumps(
                        self.config.redacted_dict(),
                        sort_keys=True,
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (temporary_destination / "resolved-config.json").chmod(0o600)
                controller_summary = _sqlite_summary(
                    temporary_destination / "controller.sqlite3",
                    tables=(
                        "runs",
                        "iterations",
                        "events",
                        "proposal_attempts",
                        "evaluation_attempts",
                    ),
                )
                history_summary = _sqlite_summary(
                    temporary_destination / "history.sqlite3",
                    tables=(
                        "research_namespaces",
                        "candidate_artifacts",
                        "experiments",
                        "case_measurements",
                        "experiment_relations",
                    ),
                )
                database_summaries = {
                    "controller": controller_summary,
                    "history": history_summary,
                }
                if campaign_summary is not None:
                    database_summaries["campaign"] = campaign_summary
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
                scientific_object_count = (
                    len(
                        [
                            path
                            for path in (
                                temporary_destination / "state_objects"
                            ).rglob("*")
                            if path.is_file()
                        ]
                    )
                    if (temporary_destination / "state_objects").exists()
                    else 0
                )
                controller_object_count = (
                    len(
                        [
                            path
                            for path in (
                                temporary_destination / "controller_objects"
                            ).rglob("*")
                            if path.is_file()
                        ]
                    )
                    if (temporary_destination / "controller_objects").exists()
                    else 0
                )
                manifest = {
                    "schema_version": 2,
                    "run_id": run_id,
                    "run_status": run["status"],
                    "campaign_id": (
                        run.get("workflow_snapshot", {}).get("campaign_id")
                        if campaign_summary is not None
                        else None
                    ),
                    "config": self.config.redacted_dict(),
                    "databases": database_summaries,
                    "artifact_count": artifact_count,
                    "scientific_object_count": scientific_object_count,
                    "controller_object_count": controller_object_count,
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
