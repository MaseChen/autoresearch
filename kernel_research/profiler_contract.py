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


PROFILER_PROFILE_SCHEMA_VERSION = 1
PROFILE_COLLECTION_SCHEMA_VERSION = 2
PROFILE_WORKER_OUTPUT_SCHEMA_VERSION = 2

PROFILER_BASE_IMAGE = (
    "registry.cn-shanghai.aliyuncs.com/kcr-3rd/kesci_kernel_lab@sha256:"
    "5f1da890360acc5a81438d0e35a80079b34f30d1fa64fc954fef2e9a1ae45b64"
)
PROFILER_IMAGE_REPOSITORY = "ghcr.io/masechen/autoresearch-metax-profiler"
# Submit A is deliberately non-runnable.  Submit B must replace this sentinel
# with the RepoDigest returned by the reviewed Linux/amd64 build and set the
# activation bit below.  Host launchers check the bit before invoking Docker.
PROFILER_IMAGE = PROFILER_IMAGE_REPOSITORY + "@sha256:" + ("0" * 64)
PROFILER_ACTIVE = False
PROFILER_WORKER_REVISION = "metax-bounded-profiler-worker-v1"
PROFILER_ENTRYPOINT = "/opt/kernel-research/bin/bounded-profiler"

MCTRACER_VERSION = "3.5.3.20-ef9e10e"
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
PROFILE_HOME = "/tmp/profile-home"
PROFILE_TMPDIR = "/tmp"
PROFILE_TRITON_CACHE_DIR = "/tmp/triton-cache"

PROFILE_TIMEOUT_SECONDS = 900.0
PROFILE_CANARY_WALL_SECONDS = 1800.0
PROFILE_CANARY_GPU_SECONDS = 900.0
PROFILE_OUTPUT_LIMIT_BYTES = 256 * 1024
PROFILE_RESULT_LIMIT_BYTES = 64 * 1024
PROFILE_OUTCOME_LIMIT_BYTES = 64 * 1024
PROFILE_RAW_TRACE_LIMIT_BYTES = 64 * 1024 * 1024
PROFILE_CANDIDATE_LIMIT_BYTES = 2 * 1024 * 1024
PROFILE_TRACE_FILE_LIMIT = 256
PROFILE_TRACE_MEMBER_LIMIT_BYTES = 32 * 1024 * 1024
PROFILE_MEMORY_LIMIT = "4g"
PROFILE_CPU_LIMIT = 4.0
PROFILE_PID_LIMIT = 128
PROFILE_RESOURCE_ID = "gpu1"
PROFILE_LEASE_TTL_SECONDS = PROFILE_TIMEOUT_SECONDS + 30.0

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


def profiler_profile_snapshot() -> dict[str, Any]:
    """Return the canonical, JSON-safe profiler identity frozen into soak."""

    return {
        "schema_version": PROFILER_PROFILE_SCHEMA_VERSION,
        "active": PROFILER_ACTIVE,
        "base_image": PROFILER_BASE_IMAGE,
        "profiler_image": PROFILER_IMAGE,
        "worker_revision": PROFILER_WORKER_REVISION,
        "entrypoint": PROFILER_ENTRYPOINT,
        "toolchain": {
            "mctracer": {
                "path": MCTRACER_PATH,
                "version": MCTRACER_VERSION,
                "sha256": MCTRACER_SHA256,
            },
            "libmcToolsExt_lite.so": {
                "path": MCTOOLS_EXT_LITE_PATH,
                "sha256": MCTOOLS_EXT_LITE_SHA256,
            },
            "libmcToolsExt.so": {
                "path": MCTOOLS_EXT_PATH,
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
        "limits": {
            "action_timeout_seconds": PROFILE_TIMEOUT_SECONDS,
            "canary_wall_seconds": PROFILE_CANARY_WALL_SECONDS,
            "canary_gpu_seconds": PROFILE_CANARY_GPU_SECONDS,
            "candidate_bytes": PROFILE_CANDIDATE_LIMIT_BYTES,
            "result_bytes": PROFILE_RESULT_LIMIT_BYTES,
            "outcome_bytes": PROFILE_OUTCOME_LIMIT_BYTES,
            "raw_trace_bytes": PROFILE_RAW_TRACE_LIMIT_BYTES,
            "trace_files": PROFILE_TRACE_FILE_LIMIT,
            "trace_member_bytes": PROFILE_TRACE_MEMBER_LIMIT_BYTES,
        },
        "collection_schema_version": PROFILE_COLLECTION_SCHEMA_VERSION,
        "worker_output_schema_version": PROFILE_WORKER_OUTPUT_SCHEMA_VERSION,
    }


PROFILER_PROFILE_DIGEST = canonical_sha256(profiler_profile_snapshot())


def require_active_profiler() -> None:
    """Fail before Docker while Submit A's profile is intentionally inactive."""

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
        "PROFILER_BASE_IMAGE",
        "PROFILER_ENTRYPOINT",
        "PROFILER_IMAGE",
        "PROFILER_IMAGE_REPOSITORY",
        "PROFILER_PROFILE_DIGEST",
        "PROFILER_RECIPES",
        "PROFILER_RECIPE_IDS",
        "PROFILER_WORKER_REVISION",
        "profiler_profile_snapshot",
        "require_active_profiler",
    ]
)
