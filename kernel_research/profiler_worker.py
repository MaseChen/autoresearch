"""Trusted in-image worker for the bounded MetaX profiler.

The public host never forwards free-form profiler arguments.  This worker
accepts only the frozen contract, validates the exact image toolchain and
candidate bytes, and publishes crash-recoverable outcome evidence before and
after any GPU execution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

from .constants import CURRENT_C500_EVALUATION_PROTOCOL_ID, REQUIRED_MATCH_RATIO
from .contract import validate_candidate
from .platform.canonical import canonical_json_text, canonical_sha256
from .platform.profiles import CURRENT_RESEARCH_NAMESPACE
from .platform.proposal import CandidateBundle, TRITON_PYTHON_BUNDLE_LIMITS
from .profiler_contract import (
    MCTRACER_PATH,
    MCTRACER_SHA256,
    MCTRACER_VERSION,
    MCTOOLS_EXT_LITE_PATH,
    MCTOOLS_EXT_LITE_SHA256,
    MCTOOLS_EXT_PATH,
    MCTOOLS_EXT_SHA256,
    PROFILE_CANDIDATE_CONTAINER_PATH,
    PROFILE_CANDIDATE_LIMIT_BYTES,
    PROFILE_COLLECTION_SCHEMA_VERSION,
    PROFILE_HOME,
    PROFILE_OUTCOME_CONTAINER_PATH,
    PROFILE_OUTCOME_LIMIT_BYTES,
    PROFILE_RAW_TRACE_LIMIT_BYTES,
    PROFILE_RESULT_CONTAINER_PATH,
    PROFILE_RESULT_LIMIT_BYTES,
    PROFILE_TRACE_CONTAINER_PATH,
    PROFILE_TRACE_FILE_LIMIT,
    PROFILE_TRACE_MEMBER_LIMIT_BYTES,
    PROFILE_TRITON_CACHE_DIR,
    PROFILE_WORKER_OUTPUT_SCHEMA_VERSION,
    PROFILER_IMAGE,
    PROFILER_PROFILE_DIGEST,
    PROFILER_RECIPES,
    PROFILER_WORKER_REVISION,
)
from .research_policy import validate_research_candidate_bounded


_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FATAL_MARKERS = (
    "atu fault",
    "atu address translation",
    "xnack",
    "illegal memory access",
    "mcerrorillegaladdress",
    "xid",
    "device lost",
    "gpu reset",
    "hardware fault",
    "uncorrectable",
    "mc runtime error",
)
_EXECUTION_FAILURE_MARKERS = (
    "execvpe:",
    "no such file or directory",
    "error while loading shared libraries",
    "symbol lookup error",
    "segmentation fault",
)
_TRACE_DIRECTORY = Path("/output/metax-mctx")
_WARMUP_SENTINEL = Path("/output/warmup-sentinel.json")
_TARGET_SENTINEL = Path("/output/target-sentinel.json")
_TRACE_NAME = "bounded-profile"
_WARMUP_TIMEOUT_SECONDS = 240.0
_TARGET_TIMEOUT_SECONDS = 540.0


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    recipe_id: str
    case_id: str
    candidate_path: Path
    result_path: Path
    outcome_path: Path
    raw_trace_path: Path
    experiment_uid: str
    namespace_id: str
    condition_digest: str
    artifact_id: str
    candidate_sha256: str
    environment_digest: str
    soak_invariant_digest: str
    campaign_id: str
    run_id: str
    budget_action_key: str
    resource_id: str | None
    fencing_epoch: int | None

    @property
    def recipe(self) -> Mapping[str, Any]:
        return PROFILER_RECIPES[self.recipe_id]

    def echo(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": PROFILE_COLLECTION_SCHEMA_VERSION,
            "worker_revision": PROFILER_WORKER_REVISION,
            "profiler_profile_digest": PROFILER_PROFILE_DIGEST,
            "profiler_image": PROFILER_IMAGE,
            "recipe_id": self.recipe_id,
            "case_id": self.case_id,
            "experiment_uid": self.experiment_uid,
            "namespace_id": self.namespace_id,
            "condition_digest": self.condition_digest,
            "artifact_id": self.artifact_id,
            "candidate_sha256": self.candidate_sha256,
            "environment_digest": self.environment_digest,
            "soak_invariant_digest": self.soak_invariant_digest,
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "budget_action_key": self.budget_action_key,
        }
        if self.resource_id is not None:
            value.update(
                {
                    "resource_id": self.resource_id,
                    "fencing_epoch": self.fencing_epoch,
                }
            )
        return value


def _strict_token(value: object, field: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a bounded token")
    return value


def _strict_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or _HEX64.fullmatch(value.removeprefix("sha256:")) is None
    ):
        raise ValueError(f"{field} must be a sha256 digest")
    return value


def _regular_bytes(path: Path, *, exact_path: str, limit: int, field: str) -> bytes:
    if str(path) != exact_path or path != Path(exact_path):
        raise ValueError(f"{field} path must match the fixed container path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{field} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError(f"{field} must be a bounded regular file")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise ValueError(f"{field} exceeds its fixed size limit")
        if os.fstat(descriptor).st_size != metadata.st_size:
            raise ValueError(f"{field} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, content: bytes, *, limit: int, field: str) -> None:
    if len(content) > limit:
        raise ValueError(f"{field} exceeds its fixed size limit")
    if path.parent != Path("/output") or path.is_symlink():
        raise ValueError(f"{field} must use the fixed output directory")
    if path.exists() and not path.is_file():
        raise ValueError(f"{field} target must be a regular file")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        descriptor = -1
        os.replace(temporary_name, path)
        temporary_name = ""
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _canonical_write(path: Path, value: Mapping[str, Any], *, limit: int, field: str) -> None:
    _atomic_write(
        path,
        canonical_json_text(dict(value)).encode("utf-8"),
        limit=limit,
        field=field,
    )


def _strict_json_file(path: Path, *, limit: int, field: str) -> dict[str, Any]:
    data = _regular_bytes(path, exact_path=str(path), limit=limit, field=field)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict JSON") from exc
    if not isinstance(value, dict) or canonical_json_text(value).encode("utf-8") != data:
        raise ValueError(f"{field} is not a canonical JSON object")
    return value


def _sha256_file(path: Path, *, expected: str, field: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_absolute() or resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{field} is not a regular resolved file")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{field} sha256 does not match the frozen contract")
    return {"path": str(resolved), "sha256": digest}


def verify_toolchain() -> dict[str, Any]:
    """Validate exact image binaries, libraries, and mcTracer help semantics."""

    tools = {
        "mctracer": {
            **_sha256_file(
                Path(MCTRACER_PATH), expected=MCTRACER_SHA256, field="mcTracer"
            ),
            "version": MCTRACER_VERSION,
        },
        "libmcToolsExt_lite.so": _sha256_file(
            Path(MCTOOLS_EXT_LITE_PATH),
            expected=MCTOOLS_EXT_LITE_SHA256,
            field="libmcToolsExt_lite.so",
        ),
        "libmcToolsExt.so": _sha256_file(
            Path(MCTOOLS_EXT_PATH),
            expected=MCTOOLS_EXT_SHA256,
            field="libmcToolsExt.so",
        ),
    }
    try:
        probe = subprocess.run(
            (MCTRACER_PATH, "--help"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("mcTracer help probe is unavailable") from exc
    output = probe.stdout + b"\n" + probe.stderr
    if len(output) > 64 * 1024:
        raise ValueError("mcTracer help probe exceeded its output limit")
    try:
        text = output.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("mcTracer help probe is not UTF-8") from exc
    lowered = text.lower()
    required = (MCTRACER_VERSION, "Help Info", "Usage:")
    if (
        probe.returncode != 1
        or any(marker not in text for marker in required)
        or any(marker in lowered for marker in _EXECUTION_FAILURE_MARKERS)
        or any(marker in lowered for marker in _FATAL_MARKERS)
    ):
        raise ValueError("mcTracer help output does not match the frozen contract")
    descriptor = {
        "schema_version": 1,
        "worker_revision": PROFILER_WORKER_REVISION,
        "tools": tools,
        "help_contract": {
            "argv": [MCTRACER_PATH, "--help"],
            "stdin": "DEVNULL",
            "exit_code": 1,
            "version": MCTRACER_VERSION,
            "required_markers": ["Help Info", "Usage:"],
        },
    }
    descriptor["digest"] = canonical_sha256(descriptor)
    return descriptor


def _validate_request(args: argparse.Namespace) -> tuple[WorkerRequest, bytes]:
    try:
        recipe = PROFILER_RECIPES[args.recipe]
    except KeyError as exc:
        raise ValueError("recipe is not a frozen built-in") from exc
    if args.case_id != recipe["case_id"]:
        raise ValueError("case-id does not match the frozen recipe")
    if args.namespace_id != CURRENT_RESEARCH_NAMESPACE.namespace_id:
        raise ValueError("bounded profiler requires the exact CURRENT namespace")
    if args.candidate != PROFILE_CANDIDATE_CONTAINER_PATH:
        raise ValueError("candidate path does not match the fixed contract")
    fixed_outputs = {
        "result": PROFILE_RESULT_CONTAINER_PATH,
        "outcome": PROFILE_OUTCOME_CONTAINER_PATH,
        "raw_trace": PROFILE_TRACE_CONTAINER_PATH,
    }
    for name, expected in fixed_outputs.items():
        if getattr(args, name) != expected:
            raise ValueError(f"{name} path does not match the fixed contract")
    if args.max_result_bytes != PROFILE_RESULT_LIMIT_BYTES:
        raise ValueError("result byte limit differs from the fixed contract")
    if args.max_outcome_bytes != PROFILE_OUTCOME_LIMIT_BYTES:
        raise ValueError("outcome byte limit differs from the fixed contract")
    if args.max_raw_trace_bytes != PROFILE_RAW_TRACE_LIMIT_BYTES:
        raise ValueError("raw trace byte limit differs from the fixed contract")
    mounted = _regular_bytes(
        Path(args.candidate),
        exact_path=PROFILE_CANDIDATE_CONTAINER_PATH,
        limit=PROFILE_CANDIDATE_LIMIT_BYTES,
        field="candidate",
    )
    if str(args.artifact_id).startswith("bundle-sha256-v1:"):
        try:
            bundle_value = json.loads(mounted.decode("utf-8"))
            bundle = CandidateBundle.from_value(
                bundle_value, limits=TRITON_PYTHON_BUNDLE_LIMITS
            )
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("mounted candidate bundle is invalid") from exc
        if str(bundle.artifact_id) != args.artifact_id:
            raise ValueError("mounted candidate bundle artifact identity differs")
        entrypoint = next(
            item for item in bundle.files if item.path == bundle.entrypoint
        )
        candidate = entrypoint.content_bytes
    elif str(args.artifact_id).startswith("source-sha256-v1:"):
        candidate = mounted
    else:
        raise ValueError("candidate artifact ID is not a supported source bundle")
    actual_sha256 = hashlib.sha256(candidate).hexdigest()
    if (
        args.candidate_sha256 != actual_sha256
        or _HEX64.fullmatch(args.candidate_sha256) is None
    ):
        raise ValueError("candidate sha256 does not match entrypoint source bytes")
    try:
        source = candidate.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("candidate must be UTF-8 source") from exc
    public = validate_candidate(source=source)
    policy = validate_research_candidate_bounded(source)
    if not public.is_valid or not policy.valid:
        raise ValueError("candidate fails the exact built-in Triton/C500 policy")
    resource_id = args.resource_id
    fencing_epoch = args.fencing_epoch
    if recipe["requires_gpu"]:
        if resource_id != "gpu1" or not isinstance(fencing_epoch, int) or fencing_epoch <= 0:
            raise ValueError("hardware recipe requires the exact gpu1 fencing token")
    elif resource_id is not None or fencing_epoch is not None:
        raise ValueError("compile recipe must not receive a GPU fencing token")
    request = WorkerRequest(
        recipe_id=args.recipe,
        case_id=args.case_id,
        candidate_path=Path(args.candidate),
        result_path=Path(args.result),
        outcome_path=Path(args.outcome),
        raw_trace_path=Path(args.raw_trace),
        experiment_uid=_strict_token(args.experiment_uid, "experiment_uid"),
        namespace_id=args.namespace_id,
        condition_digest=_strict_digest(args.condition_digest, "condition_digest"),
        artifact_id=_strict_token(args.artifact_id, "artifact_id"),
        candidate_sha256=args.candidate_sha256,
        environment_digest=_strict_digest(args.environment_digest, "environment_digest"),
        soak_invariant_digest=_strict_digest(args.soak_invariant_digest, "soak_invariant_digest"),
        campaign_id=_strict_token(args.campaign_id, "campaign_id"),
        run_id=_strict_token(args.run_id, "run_id"),
        budget_action_key=_strict_token(args.budget_action_key, "budget_action_key"),
        resource_id=resource_id,
        fencing_epoch=fencing_epoch,
    )
    return request, candidate


def _outcome(
    request: WorkerRequest,
    *,
    status: str,
    gpu_state: str,
    completion_trusted: bool,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_WORKER_OUTPUT_SCHEMA_VERSION,
        **request.echo(),
        "status": status,
        "gpu_state": gpu_state,
        "completion_trusted": completion_trusted,
        "reason_code": reason_code,
    }


def _write_outcome(request: WorkerRequest, **values: Any) -> None:
    _canonical_write(
        request.outcome_path,
        _outcome(request, **values),
        limit=PROFILE_OUTCOME_LIMIT_BYTES,
        field="profiler outcome",
    )


def _compile_recipe(
    request: WorkerRequest, candidate: bytes, toolchain: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    manifest = {
        "schema_version": 1,
        "kind": "metax_compile_metadata_manifest_v1",
        "worker_revision": PROFILER_WORKER_REVISION,
        "recipe_id": request.recipe_id,
        "case_id": request.case_id,
        "namespace_id": request.namespace_id,
        "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
        "toolchain": dict(toolchain),
    }
    raw = canonical_json_text(manifest).encode("utf-8")
    descriptor = {
        "format": "canonical-json-manifest-v1",
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
        "file_count": 1,
    }
    metrics = {
        "compiler_version_digest": {
            "status": "AVAILABLE",
            "value": toolchain["digest"],
        },
        "register_count": {
            "status": "UNAVAILABLE",
            "reason_code": "COUNTER_NOT_EXPOSED",
        },
        "shared_memory_bytes": {
            "status": "UNAVAILABLE",
            "reason_code": "COUNTER_NOT_EXPOSED",
        },
        "spill_bytes": {
            "status": "UNAVAILABLE",
            "reason_code": "COUNTER_NOT_EXPOSED",
        },
    }
    return metrics, raw, descriptor


def _copy_candidate_to_tmp(candidate: bytes) -> Path:
    root = Path("/tmp/candidate")
    root.mkdir(mode=0o700, parents=False, exist_ok=True)
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("candidate tmp directory is not canonical")
    path = root / "kernel.py"
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, candidate)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def _run_fixed_case_target(
    candidate_path: Path, sentinel_path: Path, *, phase: str
) -> int:
    """Run the frozen quick case in either warmup or traced mode."""

    if phase not in {"warmup", "tracked"}:
        raise ValueError("profile target phase is not frozen")
    status = "FAILED"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "case_id": "quick_decode_gate_up",
        "status": status,
        "correctness_passed": False,
        "target_completed": True,
        "phase": phase,
        "warmup_launches": 0,
        "tracked_launches": 0,
    }
    try:
        from .backends import (
            _clone_readonly_inputs,
            _copy_dataset_to_device,
            _find_accelerator,
            _import_candidate,
            _load_c500_runtime,
            _matched_ratio,
            _readonly_inputs_unchanged,
            _synchronize,
            _torch_reference,
        )
        from .cases import generate_case, get_suite

        validation = validate_candidate(candidate_path)
        if not validation.valid:
            raise ValueError("temporary candidate failed validation")
        runtime, error = _load_c500_runtime()
        if runtime is None or error is not None:
            raise RuntimeError("C500 runtime is unavailable")
        torch, _triton = runtime
        accelerator, accelerator_error = _find_accelerator(torch)
        if accelerator_error is not None:
            raise RuntimeError(str(accelerator_error["reason"]))
        device = str(accelerator["device"])
        selected = [
            item
            for item in get_suite(
                "quick",
                evaluation_protocol_id=CURRENT_C500_EVALUATION_PROTOCOL_ID,
            )
            if item.name == "quick_decode_gate_up"
        ]
        if len(selected) != 1:
            raise RuntimeError("fixed profiling case is missing or duplicated")
        dataset = generate_case(selected[0])
        tensors = _copy_dataset_to_device(torch, dataset, device)
        del dataset
        snapshots = _clone_readonly_inputs(tensors)
        expected = _torch_reference(torch, tensors)
        module = _import_candidate(candidate_path, validation.sha256)
        run_kernel = getattr(module, "run_kernel")
        arguments = tensors["arguments"]
        if phase == "warmup":
            # One compile invocation is deliberately not counted as a warmup.
            run_kernel(*arguments)
            _synchronize(torch)
            if _matched_ratio(torch, tensors["out"], expected) < REQUIRED_MATCH_RATIO:
                raise ValueError("candidate correctness failed after compilation")
            for _ in range(10):
                run_kernel(*arguments)
                payload["warmup_launches"] += 1
            _synchronize(torch)
            success_sentinel = "METAX_PROFILE_WARMUP_SUCCESS_V1"
        else:
            # The persisted Triton cache was populated by the untraced warmup
            # process.  Exactly these ten candidate launches sit under mctx.
            for _ in range(10):
                run_kernel(*arguments)
                _synchronize(torch)
                payload["tracked_launches"] += 1
            success_sentinel = "METAX_PROFILE_TARGET_SUCCESS_V1"
        matched_ratio = _matched_ratio(torch, tensors["out"], expected)
        if matched_ratio < REQUIRED_MATCH_RATIO:
            raise ValueError(f"candidate correctness failed after {phase} launches")
        readonly, modified = _readonly_inputs_unchanged(torch, tensors, snapshots)
        if not readonly:
            raise ValueError(f"candidate modified read-only input {modified}")
        payload.update(
            {
                "status": "SUCCESS",
                "correctness_passed": True,
                "matched_ratio": matched_ratio,
                "target_success_sentinel": success_sentinel,
            }
        )
        result = 0
    except BaseException as exc:
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)[:2000]
        result = 2
    _canonical_write(
        sentinel_path,
        payload,
        limit=PROFILE_RESULT_LIMIT_BYTES,
        field="profile target sentinel",
    )
    return result


def _trace_files(root: Path) -> list[tuple[str, bytes]]:
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("mcTracer trace directory is unavailable")
    files: list[tuple[str, bytes]] = []
    total = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("mcTracer trace contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file() or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("mcTracer trace contains an invalid path")
        data = _regular_bytes(
            path,
            exact_path=str(path),
            limit=PROFILE_TRACE_MEMBER_LIMIT_BYTES,
            field="mcTracer trace member",
        )
        if not data:
            raise ValueError("mcTracer trace contains an empty file")
        files.append((relative, data))
        total += len(data)
        if len(files) > PROFILE_TRACE_FILE_LIMIT or total > PROFILE_RAW_TRACE_LIMIT_BYTES:
            raise ValueError("mcTracer trace exceeds its fixed aggregate limits")
    if not files:
        raise ValueError("mcTracer emitted no non-empty trace files")
    return files


def _deterministic_tar(files: Sequence[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, data in sorted(files):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    result = output.getvalue()
    if not result or len(result) > PROFILE_RAW_TRACE_LIMIT_BYTES:
        raise ValueError("canonical trace archive exceeds its fixed limit")
    return result


def _hardware_recipe(
    request: WorkerRequest, candidate: bytes, toolchain: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    candidate_path = _copy_candidate_to_tmp(candidate)
    _TRACE_DIRECTORY.mkdir(mode=0o700, parents=False, exist_ok=False)
    warmup_command = (
        sys.executable,
        "-m",
        "kernel_research.profiler_worker",
        "_hardware-target",
        "--phase",
        "warmup",
        "--candidate",
        str(candidate_path),
        "--sentinel",
        str(_WARMUP_SENTINEL),
    )
    command = (
        MCTRACER_PATH,
        "--mctx",
        "--odname",
        _TRACE_DIRECTORY.name,
        "--name",
        _TRACE_NAME,
        sys.executable,
        "-m",
        "kernel_research.profiler_worker",
        "_hardware-target",
        "--phase",
        "tracked",
        "--candidate",
        str(candidate_path),
        "--sentinel",
        str(_TARGET_SENTINEL),
    )
    _write_outcome(
        request,
        status="RUNNING",
        gpu_state="STARTED",
        completion_trusted=False,
        reason_code="GPU_EXECUTION_STARTED",
    )
    runtime_environment = {
        **os.environ,
        "HOME": PROFILE_HOME,
        "TMPDIR": "/tmp",
        "TRITON_CACHE_DIR": PROFILE_TRITON_CACHE_DIR,
    }
    try:
        warmup_result = subprocess.run(
            warmup_command,
            cwd="/output",
            env=runtime_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_WARMUP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("profile warmup timed out without trusted completion") from exc
    warmup_output = warmup_result.stdout + b"\n" + warmup_result.stderr
    if len(warmup_output) > 256 * 1024:
        raise ValueError("profile warmup output exceeded its fixed limit")
    warmup_text = warmup_output.decode("utf-8", errors="replace").lower()
    if any(marker in warmup_text for marker in _FATAL_MARKERS):
        _write_outcome(
            request,
            status="HARD_FAILURE",
            gpu_state="STARTED",
            completion_trusted=False,
            reason_code="FATAL_GPU_MARKER",
        )
        raise RuntimeError("profile warmup reported a fatal GPU marker")
    warmup_sentinel = _strict_json_file(
        _WARMUP_SENTINEL,
        limit=PROFILE_RESULT_LIMIT_BYTES,
        field="profile warmup sentinel",
    )
    warmup_success = (
        warmup_result.returncode == 0
        and warmup_sentinel.get("status") == "SUCCESS"
        and warmup_sentinel.get("phase") == "warmup"
        and warmup_sentinel.get("correctness_passed") is True
        and warmup_sentinel.get("target_completed") is True
        and warmup_sentinel.get("warmup_launches") == 10
        and warmup_sentinel.get("tracked_launches") == 0
        and warmup_sentinel.get("target_success_sentinel")
        == "METAX_PROFILE_WARMUP_SUCCESS_V1"
        and warmup_sentinel.get("case_id") == request.case_id
    )
    if not warmup_success:
        _write_outcome(
            request,
            status="FAILED",
            gpu_state="COMPLETED",
            completion_trusted=True,
            reason_code="WARMUP_FAILED",
        )
        raise ValueError("profile warmup did not produce trusted success evidence")
    try:
        command_result = subprocess.run(
            command,
            cwd="/output",
            env=runtime_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_TARGET_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("mcTracer target timed out without trusted completion") from exc
    combined = command_result.stdout + b"\n" + command_result.stderr
    if len(combined) > 256 * 1024:
        raise ValueError("mcTracer target output exceeded its fixed limit")
    text = combined.decode("utf-8", errors="replace").lower()
    hard = any(marker in text for marker in _FATAL_MARKERS)
    if hard:
        _write_outcome(
            request,
            status="HARD_FAILURE",
            gpu_state="STARTED",
            completion_trusted=False,
            reason_code="FATAL_GPU_MARKER",
        )
        raise RuntimeError("mcTracer reported a fatal GPU marker")
    sentinel = _strict_json_file(
        _TARGET_SENTINEL,
        limit=PROFILE_RESULT_LIMIT_BYTES,
        field="profile target sentinel",
    )
    target_success = (
        sentinel.get("status") == "SUCCESS"
        and sentinel.get("phase") == "tracked"
        and sentinel.get("correctness_passed") is True
        and sentinel.get("target_completed") is True
        and sentinel.get("warmup_launches") == 0
        and sentinel.get("tracked_launches") == 10
        and sentinel.get("target_success_sentinel")
        == "METAX_PROFILE_TARGET_SUCCESS_V1"
        and sentinel.get("case_id") == request.case_id
    )
    _write_outcome(
        request,
        status="RUNNING" if target_success else "FAILED",
        gpu_state="COMPLETED",
        completion_trusted=True,
        reason_code="TARGET_COMPLETED" if target_success else "TARGET_FAILED",
    )
    if not target_success:
        raise ValueError("profile target did not produce trusted success evidence")
    if command_result.returncode not in {0, 1}:
        raise ValueError("mcTracer returned an unaccepted exit status")
    if any(marker in text for marker in _EXECUTION_FAILURE_MARKERS):
        raise ValueError("mcTracer output contains an execution failure marker")
    files = _trace_files(_TRACE_DIRECTORY)
    raw = _deterministic_tar(files)
    descriptor = {
        "format": "deterministic-tar-v1",
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
        "file_count": len(files),
        "source_bytes": sum(len(data) for _, data in files),
        "mctracer_exit_code": command_result.returncode,
        "target_stdout_sha256": "sha256:" + hashlib.sha256(command_result.stdout).hexdigest(),
        "target_stderr_sha256": "sha256:" + hashlib.sha256(command_result.stderr).hexdigest(),
        "toolchain_digest": toolchain["digest"],
    }
    metrics = {
        metric_id: {
            "status": "UNAVAILABLE",
            "reason_code": "COUNTER_NOT_EXPOSED",
        }
        for metric_id, _kind, _unit in request.recipe["metrics"]
    }
    return metrics, raw, descriptor


def run_worker(args: argparse.Namespace) -> int:
    os.environ.update(
        {
            "HOME": PROFILE_HOME,
            "TMPDIR": "/tmp",
            "TRITON_CACHE_DIR": PROFILE_TRITON_CACHE_DIR,
        }
    )
    Path(PROFILE_HOME).mkdir(mode=0o700, parents=False, exist_ok=True)
    Path(PROFILE_TRITON_CACHE_DIR).mkdir(mode=0o700, parents=False, exist_ok=True)
    request, candidate = _validate_request(args)
    _write_outcome(
        request,
        status="RUNNING",
        gpu_state="NOT_STARTED",
        completion_trusted=True,
        reason_code="REQUEST_VALIDATED",
    )
    try:
        toolchain = verify_toolchain()
        if request.recipe["requires_gpu"]:
            metrics, raw_trace, descriptor = _hardware_recipe(
                request, candidate, toolchain
            )
            gpu_state = "COMPLETED"
        else:
            metrics, raw_trace, descriptor = _compile_recipe(
                request, candidate, toolchain
            )
            gpu_state = "NOT_STARTED"
        _atomic_write(
            request.raw_trace_path,
            raw_trace,
            limit=PROFILE_RAW_TRACE_LIMIT_BYTES,
            field="profiler raw trace",
        )
        result = {
            **request.echo(),
            "toolchain": toolchain,
            "trace_descriptor": descriptor,
            "metrics": metrics,
        }
        _canonical_write(
            request.result_path,
            result,
            limit=PROFILE_RESULT_LIMIT_BYTES,
            field="profiler result",
        )
        _write_outcome(
            request,
            status="SUCCESS",
            gpu_state=gpu_state,
            completion_trusted=True,
            reason_code="EVIDENCE_COMMITTED",
        )
        return 0
    except BaseException as exc:
        try:
            current = _strict_json_file(
                request.outcome_path,
                limit=PROFILE_OUTCOME_LIMIT_BYTES,
                field="profiler outcome",
            )
            if current.get("status") not in {"HARD_FAILURE"}:
                gpu_state = str(current.get("gpu_state", "NOT_STARTED"))
                completion = bool(current.get("completion_trusted", False))
                _write_outcome(
                    request,
                    status="FAILED",
                    gpu_state=gpu_state,
                    completion_trusted=completion,
                    reason_code=(
                        "WORKER_KNOWN_FAILURE"
                        if gpu_state == "NOT_STARTED" or completion
                        else "WORKER_UNKNOWN_OUTCOME"
                    ),
                )
        except BaseException:
            pass
        print(f"bounded-profiler: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bounded-profiler", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-toolchain", allow_abbrev=False)
    verify.set_defaults(
        handler=lambda _args: (
            print(canonical_json_text(verify_toolchain())),
            0,
        )[1]
    )
    run = subparsers.add_parser("run", allow_abbrev=False)
    run.add_argument("--recipe", required=True, choices=tuple(PROFILER_RECIPES))
    run.add_argument("--case-id", required=True)
    run.add_argument("--candidate", required=True)
    run.add_argument("--result", required=True)
    run.add_argument("--outcome", required=True)
    run.add_argument("--raw-trace", required=True)
    run.add_argument("--max-result-bytes", required=True, type=int)
    run.add_argument("--max-outcome-bytes", required=True, type=int)
    run.add_argument("--max-raw-trace-bytes", required=True, type=int)
    for field in (
        "experiment-uid",
        "namespace-id",
        "condition-digest",
        "artifact-id",
        "candidate-sha256",
        "environment-digest",
        "soak-invariant-digest",
        "campaign-id",
        "run-id",
        "budget-action-key",
    ):
        run.add_argument("--" + field, required=True)
    run.add_argument("--resource-id")
    run.add_argument("--fencing-epoch", type=int)
    run.set_defaults(handler=run_worker)
    target = subparsers.add_parser("_hardware-target", allow_abbrev=False)
    target.add_argument("--phase", required=True, choices=("warmup", "tracked"))
    target.add_argument("--candidate", required=True)
    target.add_argument("--sentinel", required=True)
    target.set_defaults(
        handler=lambda values: _run_fixed_case_target(
            Path(values.candidate), Path(values.sentinel), phase=values.phase
        )
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "_hardware-target":
        expected_sentinel = (
            _WARMUP_SENTINEL if args.phase == "warmup" else _TARGET_SENTINEL
        )
        if (
            args.candidate != "/tmp/candidate/kernel.py"
            or args.sentinel != str(expected_sentinel)
        ):
            raise ValueError("hardware target paths differ from the fixed contract")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
