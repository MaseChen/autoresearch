"""Canonical measurement roles for shadow candidate scoring."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .constants import REQUIRED_MATCH_RATIO
from .device_timing import device_event_protocol_snapshot
from .platform.canonical import canonical_sha256


SCORING_CANDIDATE_MEASUREMENT_SCHEMA_VERSION = 1
SCORING_CANDIDATE_WORKER_REVISION = "xpuoj-scoring-candidate-worker-v1"
QUALIFIED_SCORING_PROBE_ENVIRONMENT_SNAPSHOT_DIGEST = (
    "sha256:dc0a3e45d0fd35bd73b807b2c888f6f3bc34976b5d61703d1dc5fddabed030e6"
)

_QUALIFIED_SCORING_PROBE_ENVIRONMENT: dict[str, object] = {
    "accelerator_api": "cuda",
    "compile_probe_passed": None,
    "compile_probe_requested": False,
    "compile_probe_status": "NOT_RUN",
    "device": "cuda:0",
    "device_count": 1,
    "device_name": "MetaX C500",
    "driver_probe_error": None,
    "driver_version": "3.3.12",
    "driver_version_source": "mx-smi",
    "implementation": "CPython",
    "maca_path": "/opt/maca",
    "mctriton_path": "/opt/conda/lib/python3.10/site-packages/triton/__init__.py",
    "mctriton_version": "3.0.0",
    "mx_smi_driver_version": "3.3.12",
    "mx_smi_maca_version": "3.5.3.20",
    "mx_smi_path": "/opt/mxdriver/bin/mx-smi",
    "mx_smi_probe_error": None,
    "mx_smi_status_exit_code": 0,
    "mx_smi_version": "mx-smi  version: 2.2.12",
    "mx_smi_version_exit_code": 0,
    "platform": "Linux-5.15.0-119-generic-x86_64-with-glibc2.35",
    "python": "3.10.10",
    "sdk_probe_error": None,
    "sdk_system_maca_version_match": False,
    "sdk_version": "3.5.3.9",
    "sdk_version_source": "torch.version.maca",
    "system_maca_probe_error": None,
    "system_maca_version": "3.5.3.20",
    "system_maca_version_source": "mx-smi",
    "target_verified": True,
    "torch_path": "/opt/conda/lib/python3.10/site-packages/torch/__init__.py",
    "torch_version": "2.8.0+metax3.5.3.9",
    "triton_path": "/opt/conda/lib/python3.10/site-packages/triton/__init__.py",
    "triton_version": "3.0.0",
}


def qualified_scoring_probe_environment_snapshot() -> dict[str, object]:
    """Return the exact environment observed by all ten qualified probes."""

    value = dict(_QUALIFIED_SCORING_PROBE_ENVIRONMENT)
    if canonical_sha256(value) != QUALIFIED_SCORING_PROBE_ENVIRONMENT_SNAPSHOT_DIGEST:
        raise RuntimeError("qualified scoring probe environment constant is corrupt")
    return value


def scoring_candidate_worker_source_sha256() -> str:
    """Bind the worker implementation bytes into the measurement contract."""

    path = Path(__file__).with_name("scoring_candidate_worker.py")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("scoring candidate worker source is not a regular file")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def scoring_candidate_measurement_contract_snapshot() -> dict[str, object]:
    """Freeze candidate objective and incumbent-safety timing roles."""

    material: dict[str, object] = {
        "schema_version": SCORING_CANDIDATE_MEASUREMENT_SCHEMA_VERSION,
        "worker_revision": SCORING_CANDIDATE_WORKER_REVISION,
        "worker_source_sha256": scoring_candidate_worker_source_sha256(),
        "protocol_id": "xpuoj-th0-proxy-v1",
        "authority": "SHADOW_ONLY",
        "promotion_authority": False,
        "case_memory_policy": "regenerate-and-release-each-case",
        "phase_order": "all-candidate-anchors-before-paired-incumbent-safety",
        "candidate_anchor_pairing": "candidate-vs-candidate",
        "candidate_anchor_statistic": "median-of-12-self-paired-block-latencies",
        "paired_safety_pairing": "candidate-vs-deployment-incumbent",
        "paired_safety_statistic": "ratio-of-device-event-channel-medians",
        "minimum_correctness_ratio": REQUIRED_MATCH_RATIO,
        "qualified_probe_environment_snapshot_digest": (
            QUALIFIED_SCORING_PROBE_ENVIRONMENT_SNAPSHOT_DIGEST
        ),
        "timing_protocol": device_event_protocol_snapshot(),
    }
    return {**material, "digest": canonical_sha256(material)}


__all__ = [
    "SCORING_CANDIDATE_WORKER_REVISION",
    "QUALIFIED_SCORING_PROBE_ENVIRONMENT_SNAPSHOT_DIGEST",
    "qualified_scoring_probe_environment_snapshot",
    "scoring_candidate_worker_source_sha256",
    "scoring_candidate_measurement_contract_snapshot",
]
