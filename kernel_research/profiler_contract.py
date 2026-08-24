"""Dependency-light immutable contract for the bounded MetaX profiler.

This module intentionally imports neither NumPy nor the evaluator runtime.  It
is shared by the host launcher, the image worker, Dockerfile tests, and the
Campaign soak invariant so that profiler identity cannot drift independently
between those trust boundaries.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from .platform.canonical import canonical_sha256


PROFILER_BUILD_PROFILE_SCHEMA_VERSION = 1
PROFILER_ACTIVATION_PROFILE_SCHEMA_VERSION = 1
PROFILE_COLLECTION_SCHEMA_VERSION = 2
PROFILE_WORKER_OUTPUT_SCHEMA_VERSION = 2
PROFILE_PHASE_DIAGNOSTIC_SCHEMA_VERSION = 1

PROFILER_BASE_IMAGE = (
    "registry.cn-shanghai.aliyuncs.com/kcr-3rd/kesci_kernel_lab@sha256:"
    "5f1da890360acc5a81438d0e35a80079b34f30d1fa64fc954fef2e9a1ae45b64"
)
PROFILER_IMAGE_REPOSITORY = "ghcr.io/masechen/autoresearch-metax-profiler"
PROFILER_PLATFORM = "linux/amd64"
# The independently built A9 Linux/amd64 image passed exact RepoDigest pull,
# Image ID comparison, non-root/read-only runtime qualification and the private
# bounded shared-memory probe. Activation values remain excluded from the
# worker-visible build profile.
PROFILER_IMAGE = (
    PROFILER_IMAGE_REPOSITORY
    + "@sha256:f3f83880c9a0461156aa0c625fb5b0bb"
    + "4fbf1cba1438a61271264835514f1d27"
)
PROFILER_ACTIVE = True
PROFILER_WORKER_REVISION = "metax-bounded-profiler-worker-v3"
PROFILER_ENTRYPOINT = "/opt/kernel-research/bin/bounded-profiler"
PROFILER_IMAGE_UID = 1000
PROFILER_IMAGE_GID = 1000
PROFILER_LIBRARY_ROOT = "/opt/kernel-research/lib/kernel_research"
PROFILER_LIBRARY_NORMALIZER_CONTAINER_PATH = (
    "/opt/kernel-research/bin/normalize-library-tree"
)
PROFILER_LIBRARY_NORMALIZER_INTERPRETER = "python"
PROFILER_LIBRARY_NORMALIZER_SHA256 = (
    "a64c60dc263489144af2a288b14c03d393b2eb5eb082e73bf54deb82de3e366f"
)
PROFILER_LIBRARY_OWNER_UID = 0
PROFILER_LIBRARY_OWNER_GID = 0
PROFILER_LIBRARY_DIRECTORY_MODE = "0555"
PROFILER_LIBRARY_PYTHON_FILE_MODE = "0444"

MCTRACER_VERSION = "3.5.3.20-ef9e10e"
METAX_TOOLCHAIN_VERSION = "3.5.3"
MCTRACER_PATH = "/opt/maca-3.5.3/bin/mcTracer"
MCTRACER_SHA256 = (
    "92430f0c558e561e6d0b10c63b13c25606f3b147f50aef64cd2f04bfaddb12ff"
)
MCTOOLS_EXT_LITE_PATH = "/opt/maca-3.5.3/lib/libmcToolsExt_lite.so"
MCTOOLS_EXT_LITE_SHA256 = (
    "a36a51cbcf29ebbbf9646c01f3670c39ea17204cd461cc49f528ae3e52630f29"
)
MCTOOLS_EXT_PATH = "/opt/maca-3.5.3/lib/libmcToolsExt.so"
MCTOOLS_EXT_SHA256 = (
    "bb352432a8534a2c485b0598013fd71d0de2cf0b4dff799d0d82f00f416caa15"
)

PROFILE_RESULT_CONTAINER_PATH = "/output/result.json"
PROFILE_OUTCOME_CONTAINER_PATH = "/output/outcome.json"
PROFILE_TRACE_CONTAINER_PATH = "/output/raw.trace"
PROFILE_CANDIDATE_CONTAINER_PATH = "/input/candidate.cas"
PROFILE_OUTPUT_CONTAINER_DIRECTORY = "/output"
PROFILE_TRACE_DIRECTORY_CONTAINER_PATH = "/output/metax-mctx"
PROFILE_WARMUP_SENTINEL_CONTAINER_PATH = "/output/warmup-sentinel.json"
PROFILE_TARGET_SENTINEL_CONTAINER_PATH = "/output/target-sentinel.json"
PROFILE_WARMUP_PROCESS_CONTAINER_PATH = "/output/warmup-process.json"
PROFILE_TRACKED_PROCESS_CONTAINER_PATH = "/output/tracked-process.json"
PROFILE_TRACE_NAME = "bounded-profile"
PROFILE_HOME = "/tmp/profile-home"
PROFILE_TMPDIR = "/tmp"
PROFILE_TRITON_CACHE_DIR = "/tmp/triton-cache"
PROFILE_CANDIDATE_TMP_DIRECTORY = "/tmp/candidate"
PROFILE_CANDIDATE_TMP_PATH = "/tmp/candidate/kernel.py"

PROFILE_TIMEOUT_SECONDS = 900.0
PROFILE_WARMUP_TIMEOUT_SECONDS = 240.0
PROFILE_TARGET_TIMEOUT_SECONDS = 540.0
PROFILE_TOOLCHAIN_HELP_TIMEOUT_SECONDS = 10.0
PROFILE_CANARY_WALL_SECONDS = 1800.0
PROFILE_CANARY_GPU_SECONDS = 900.0
PROFILE_OUTPUT_LIMIT_BYTES = 256 * 1024
PROFILE_TOOLCHAIN_HELP_OUTPUT_LIMIT_BYTES = 64 * 1024
PROFILE_TARGET_OUTPUT_LIMIT_BYTES = 256 * 1024
PROFILE_PHASE_DIAGNOSTIC_STREAM_LIMIT_BYTES = 16 * 1024
PROFILE_PHASE_DIAGNOSTIC_LIMIT_BYTES = 64 * 1024
PROFILE_RESULT_LIMIT_BYTES = 64 * 1024
PROFILE_OUTCOME_LIMIT_BYTES = 64 * 1024
PROFILE_RAW_TRACE_LIMIT_BYTES = 64 * 1024 * 1024
PROFILE_CANARY_RAW_TRACE_TOTAL_LIMIT_BYTES = 64 * 1024 * 1024
PROFILE_CANDIDATE_LIMIT_BYTES = 2 * 1024 * 1024
PROFILE_TRACE_FILE_LIMIT = 256
PROFILE_TRACE_MEMBER_LIMIT_BYTES = 32 * 1024 * 1024
PROFILE_MEMORY_LIMIT = "24g"
PROFILE_HOST_MEMORY_SOURCE = "/proc/meminfo"
PROFILE_HOST_MIN_TOTAL_MEMORY_BYTES = 24 * 1024**3
PROFILE_HOST_MIN_AVAILABLE_MEMORY_BYTES = 24 * 1024**3
PROFILE_CPU_LIMIT = 4.0
PROFILE_PID_LIMIT = 128
PROFILE_IPC_MODE = "private"
PROFILE_SHARED_MEMORY_PATH = "/dev/shm"
PROFILE_SHARED_MEMORY_SIZE = "1g"
PROFILE_SHARED_MEMORY_SIZE_BYTES = 1024**3
PROFILE_SHARED_MEMORY_OWNER_UID = 0
PROFILE_SHARED_MEMORY_OWNER_GID = 0
PROFILE_SHARED_MEMORY_MODE = "1777"
PROFILE_HOST_IPC_ALLOWED = False
PROFILE_CONTAINER_IPC_SHARING_ALLOWED = False
PROFILE_RESOURCE_ID = "gpu1"
PROFILE_LEASE_TTL_SECONDS = PROFILE_TIMEOUT_SECONDS + 30.0
PROFILE_CANARY_LEASE_TTL_SECONDS = PROFILE_CANARY_WALL_SECONDS + 30.0
PROFILE_DOCKER_TMPFS_MODE = "0700"
PROFILE_DOCKER_TMPFS_SIZE = "64m"
PROFILE_DOCKER_TMPFS = (
    f"/tmp:rw,nosuid,nodev,noexec,size={PROFILE_DOCKER_TMPFS_SIZE},mode=700,"
    f"uid={PROFILER_IMAGE_UID},gid={PROFILER_IMAGE_GID}"
)
PROFILE_TRITON_CACHE_TMPFS_MODE = "0700"
PROFILE_TRITON_CACHE_TMPFS_SIZE = "1g"
PROFILE_TRITON_CACHE_TMPFS = (
    f"{PROFILE_TRITON_CACHE_DIR}:rw,nosuid,nodev,exec,"
    f"size={PROFILE_TRITON_CACHE_TMPFS_SIZE},mode=700,"
    f"uid={PROFILER_IMAGE_UID},gid={PROFILER_IMAGE_GID}"
)
PROFILE_GPU_DEVICE_PATHS = (
    "/dev/mxcd",
    "/dev/dri/card2",
    "/dev/dri/renderD129",
)

PROFILE_UNAVAILABLE_REASON_CODES = frozenset(
    {
        "COUNTER_NOT_EXPOSED",
        "NOT_SUPPORTED",
        "PERMISSION_DENIED",
        "TOOL_VERSION_UNSUPPORTED",
    }
)

_RECIPES: dict[str, Mapping[str, Any]] = {
    "metax-compile-metadata-v1": MappingProxyType(
        {
            "recipe_id": "metax-compile-metadata-v1",
            "case_id": "smoke_gate_up",
            "minimum_correctness_rank": 1,
            "requires_gpu": False,
            "warmup_iterations": 0,
            "tracked_iterations": 0,
            "metrics": (
                ("compiler_version_digest", "digest", "sha256"),
                ("register_count", "integer", "registers/thread"),
                ("shared_memory_bytes", "integer", "bytes/block"),
                ("spill_bytes", "integer", "bytes/thread"),
            ),
        }
    ),
    "metax-hardware-counters-v1": MappingProxyType(
        {
            "recipe_id": "metax-hardware-counters-v1",
            "case_id": "quick_decode_gate_up",
            "minimum_correctness_rank": 2,
            "requires_gpu": True,
            "warmup_iterations": 10,
            "tracked_iterations": 10,
            "metrics": (
                ("gpu_active_percent", "number", "percent"),
                ("achieved_occupancy_percent", "number", "percent"),
                ("dram_bandwidth_gbps", "number", "GB/s"),
                ("l2_hit_percent", "number", "percent"),
            ),
        }
    ),
}
PROFILER_RECIPES: Mapping[str, Mapping[str, Any]] = MappingProxyType(_RECIPES)
PROFILER_RECIPE_IDS = tuple(PROFILER_RECIPES)


def profiler_build_profile_snapshot() -> dict[str, Any]:
    """Return only identity that is knowable when the image is built."""

    return {
        "schema_version": PROFILER_BUILD_PROFILE_SCHEMA_VERSION,
        "platform": PROFILER_PLATFORM,
        "base_image": PROFILER_BASE_IMAGE,
        "worker_revision": PROFILER_WORKER_REVISION,
        "entrypoint": PROFILER_ENTRYPOINT,
        "image_user": {
            "uid": PROFILER_IMAGE_UID,
            "gid": PROFILER_IMAGE_GID,
            "runtime_binding": "exact",
            "host_process_binding": "exact",
        },
        "library_filesystem": {
            "root": PROFILER_LIBRARY_ROOT,
            "normalizer": {
                "container_path": PROFILER_LIBRARY_NORMALIZER_CONTAINER_PATH,
                "interpreter": PROFILER_LIBRARY_NORMALIZER_INTERPRETER,
                "sha256": PROFILER_LIBRARY_NORMALIZER_SHA256,
                "removed_after_use": True,
            },
            "owner": {
                "uid": PROFILER_LIBRARY_OWNER_UID,
                "gid": PROFILER_LIBRARY_OWNER_GID,
            },
            "directory_mode": PROFILER_LIBRARY_DIRECTORY_MODE,
            "python_file_mode": PROFILER_LIBRARY_PYTHON_FILE_MODE,
            "allowed_object_types": ["directory", "regular_file"],
            "allowed_file_suffixes": [".py"],
            "symlinks_allowed": False,
            "normalization_phase": "before-runtime-user",
            "non_root_import_probe": True,
        },
        "toolchain": {
            "mctracer": {
                "path": MCTRACER_PATH,
                "version": MCTRACER_VERSION,
                "sha256": MCTRACER_SHA256,
            },
            "libmcToolsExt_lite.so": {
                "path": MCTOOLS_EXT_LITE_PATH,
                "version": METAX_TOOLCHAIN_VERSION,
                "sha256": MCTOOLS_EXT_LITE_SHA256,
            },
            "libmcToolsExt.so": {
                "path": MCTOOLS_EXT_PATH,
                "version": METAX_TOOLCHAIN_VERSION,
                "sha256": MCTOOLS_EXT_SHA256,
            },
        },
        "recipes": [
            {
                **dict(PROFILER_RECIPES[recipe_id]),
                "metrics": [
                    list(item)
                    for item in PROFILER_RECIPES[recipe_id]["metrics"]
                ],
            }
            for recipe_id in PROFILER_RECIPE_IDS
        ],
        "paths": {
            "candidate": PROFILE_CANDIDATE_CONTAINER_PATH,
            "result": PROFILE_RESULT_CONTAINER_PATH,
            "outcome": PROFILE_OUTCOME_CONTAINER_PATH,
            "raw_trace": PROFILE_TRACE_CONTAINER_PATH,
            "output_directory": PROFILE_OUTPUT_CONTAINER_DIRECTORY,
            "trace_directory": PROFILE_TRACE_DIRECTORY_CONTAINER_PATH,
            "warmup_sentinel": PROFILE_WARMUP_SENTINEL_CONTAINER_PATH,
            "target_sentinel": PROFILE_TARGET_SENTINEL_CONTAINER_PATH,
            "warmup_process": PROFILE_WARMUP_PROCESS_CONTAINER_PATH,
            "tracked_process": PROFILE_TRACKED_PROCESS_CONTAINER_PATH,
            "trace_name": PROFILE_TRACE_NAME,
            "home": PROFILE_HOME,
            "tmpdir": PROFILE_TMPDIR,
            "triton_cache": PROFILE_TRITON_CACHE_DIR,
            "candidate_tmp_directory": PROFILE_CANDIDATE_TMP_DIRECTORY,
            "candidate_tmp_path": PROFILE_CANDIDATE_TMP_PATH,
        },
        "limits": {
            "action_timeout_seconds": PROFILE_TIMEOUT_SECONDS,
            "warmup_timeout_seconds": PROFILE_WARMUP_TIMEOUT_SECONDS,
            "target_timeout_seconds": PROFILE_TARGET_TIMEOUT_SECONDS,
            "toolchain_help_timeout_seconds": (
                PROFILE_TOOLCHAIN_HELP_TIMEOUT_SECONDS
            ),
            "canary_wall_seconds": PROFILE_CANARY_WALL_SECONDS,
            "canary_gpu_seconds": PROFILE_CANARY_GPU_SECONDS,
            "host_output_bytes": PROFILE_OUTPUT_LIMIT_BYTES,
            "toolchain_help_output_bytes": (
                PROFILE_TOOLCHAIN_HELP_OUTPUT_LIMIT_BYTES
            ),
            "target_output_bytes": PROFILE_TARGET_OUTPUT_LIMIT_BYTES,
            "phase_diagnostic_stream_bytes": (
                PROFILE_PHASE_DIAGNOSTIC_STREAM_LIMIT_BYTES
            ),
            "phase_diagnostic_bytes": PROFILE_PHASE_DIAGNOSTIC_LIMIT_BYTES,
            "candidate_bytes": PROFILE_CANDIDATE_LIMIT_BYTES,
            "result_bytes": PROFILE_RESULT_LIMIT_BYTES,
            "outcome_bytes": PROFILE_OUTCOME_LIMIT_BYTES,
            "raw_trace_bytes": PROFILE_RAW_TRACE_LIMIT_BYTES,
            "canary_raw_trace_total_bytes": (
                PROFILE_CANARY_RAW_TRACE_TOTAL_LIMIT_BYTES
            ),
            "trace_files": PROFILE_TRACE_FILE_LIMIT,
            "trace_member_bytes": PROFILE_TRACE_MEMBER_LIMIT_BYTES,
            "memory": PROFILE_MEMORY_LIMIT,
            "host_memory_source": PROFILE_HOST_MEMORY_SOURCE,
            "host_min_total_memory_bytes": (
                PROFILE_HOST_MIN_TOTAL_MEMORY_BYTES
            ),
            "host_min_available_memory_bytes": (
                PROFILE_HOST_MIN_AVAILABLE_MEMORY_BYTES
            ),
            "cpus": PROFILE_CPU_LIMIT,
            "pids": PROFILE_PID_LIMIT,
        },
        "isolation": {
            "read_only_root": True,
            "network": "none",
            "ipc": PROFILE_IPC_MODE,
            "shared_memory": {
                "path": PROFILE_SHARED_MEMORY_PATH,
                "size": PROFILE_SHARED_MEMORY_SIZE,
                "size_bytes": PROFILE_SHARED_MEMORY_SIZE_BYTES,
                "owner": {
                    "uid": PROFILE_SHARED_MEMORY_OWNER_UID,
                    "gid": PROFILE_SHARED_MEMORY_OWNER_GID,
                },
                "mode": PROFILE_SHARED_MEMORY_MODE,
                "runtime_user": {
                    "uid": PROFILER_IMAGE_UID,
                    "gid": PROFILER_IMAGE_GID,
                    "required_access": "rwx",
                },
                "host_ipc_allowed": PROFILE_HOST_IPC_ALLOWED,
                "container_ipc_sharing_allowed": (
                    PROFILE_CONTAINER_IPC_SHARING_ALLOWED
                ),
                "qualification_probe": "uid1000-create-read-delete-v1",
            },
            "cap_drop": "ALL",
            "no_new_privileges": True,
            "tmpfs": PROFILE_DOCKER_TMPFS,
            "triton_cache_tmpfs": PROFILE_TRITON_CACHE_TMPFS,
            "tmpfs_owner": {
                "uid": PROFILER_IMAGE_UID,
                "gid": PROFILER_IMAGE_GID,
                "mode": PROFILE_DOCKER_TMPFS_MODE,
                "size": PROFILE_DOCKER_TMPFS_SIZE,
            },
            "triton_cache_tmpfs_owner": {
                "uid": PROFILER_IMAGE_UID,
                "gid": PROFILER_IMAGE_GID,
                "mode": PROFILE_TRITON_CACHE_TMPFS_MODE,
                "size": PROFILE_TRITON_CACHE_TMPFS_SIZE,
            },
        },
        "resource": {
            "resource_id": PROFILE_RESOURCE_ID,
            "lease_ttl_seconds": PROFILE_LEASE_TTL_SECONDS,
            "canary_lease_ttl_seconds": PROFILE_CANARY_LEASE_TTL_SECONDS,
            "gpu_device_paths": list(PROFILE_GPU_DEVICE_PATHS),
        },
        "collection_schema_version": PROFILE_COLLECTION_SCHEMA_VERSION,
        "worker_output_schema_version": PROFILE_WORKER_OUTPUT_SCHEMA_VERSION,
        "phase_diagnostic_schema_version": (
            PROFILE_PHASE_DIAGNOSTIC_SCHEMA_VERSION
        ),
    }


PROFILER_BUILD_PROFILE_DIGEST = canonical_sha256(
    profiler_build_profile_snapshot()
)


def profiler_activation_profile_snapshot(
    *,
    active: bool | None = None,
    profiler_image: str | None = None,
) -> dict[str, Any]:
    """Bind one immutable image deployment to the build-time contract."""

    selected_active = PROFILER_ACTIVE if active is None else active
    selected_image = PROFILER_IMAGE if profiler_image is None else profiler_image
    if not isinstance(selected_active, bool):
        raise TypeError("profiler activation flag must be boolean")
    if not isinstance(selected_image, str):
        raise TypeError("profiler activation image must be a string")
    return {
        "schema_version": PROFILER_ACTIVATION_PROFILE_SCHEMA_VERSION,
        "build_profile_digest": PROFILER_BUILD_PROFILE_DIGEST,
        "active": selected_active,
        "profiler_image": selected_image,
    }


PROFILER_ACTIVATION_PROFILE_DIGEST = canonical_sha256(
    profiler_activation_profile_snapshot()
)


def require_active_profiler() -> None:
    """Fail before Docker unless an exact reviewed RepoDigest is active."""

    if not PROFILER_ACTIVE:
        raise ValueError(
            "bounded profiler image is inactive; build, publish, and pin Submit B"
        )
    prefix = PROFILER_IMAGE_REPOSITORY + "@sha256:"
    suffix = PROFILER_IMAGE.removeprefix(prefix)
    if not PROFILER_IMAGE.startswith(prefix) or len(suffix) != 64:
        raise ValueError("active profiler image is not the fixed GHCR RepoDigest")
    if suffix == "0" * 64 or any(
        character not in "0123456789abcdef" for character in suffix
    ):
        raise ValueError("active profiler image digest is invalid")


__all__ = [name for name in globals() if name.startswith("PROFILE") or name.startswith("MCT")]
__all__.extend(
    [
        "PROFILER_ACTIVE",
        "PROFILER_ACTIVATION_PROFILE_DIGEST",
        "PROFILER_BASE_IMAGE",
        "PROFILER_BUILD_PROFILE_DIGEST",
        "PROFILER_ENTRYPOINT",
        "PROFILER_IMAGE",
        "PROFILER_IMAGE_REPOSITORY",
        "PROFILER_RECIPES",
        "PROFILER_RECIPE_IDS",
        "PROFILER_WORKER_REVISION",
        "profiler_activation_profile_snapshot",
        "profiler_build_profile_snapshot",
        "require_active_profiler",
    ]
)
