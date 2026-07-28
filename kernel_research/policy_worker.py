"""Resource-limited subprocess entrypoint for autonomous candidate policy."""

from __future__ import annotations

import json
import resource
import sys

from .constants import (
    MAX_CANDIDATE_SOURCE_BYTES,
    POLICY_CPU_LIMIT_SEC,
    POLICY_MAX_ERRORS,
    POLICY_MEMORY_LIMIT_BYTES,
    POLICY_OUTPUT_LIMIT_BYTES,
)
from .contract import CandidateError
from .research_policy import ResearchPolicyResult, validate_research_candidate


def _set_limits() -> None:
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (POLICY_CPU_LIMIT_SEC, POLICY_CPU_LIMIT_SEC),
    )
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (POLICY_MEMORY_LIMIT_BYTES, POLICY_MEMORY_LIMIT_BYTES),
        )
    except (OSError, ValueError):
        # Darwin exposes RLIMIT_AS but rejects setting it. The deployment
        # target is Linux, where this limit is mandatory; local macOS still
        # retains the source, CPU, wall-clock and output bounds.
        if sys.platform != "darwin":
            raise
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (POLICY_OUTPUT_LIMIT_BYTES, POLICY_OUTPUT_LIMIT_BYTES),
    )


def _bounded_result(result: ResearchPolicyResult) -> ResearchPolicyResult:
    if len(result.errors) <= POLICY_MAX_ERRORS:
        return result
    retained = list(result.errors[:POLICY_MAX_ERRORS])
    retained.append(
        CandidateError(
            code="POLICY_ERROR_LIMIT",
            message=(
                f"policy returned more than {POLICY_MAX_ERRORS} errors; "
                "remaining errors were omitted"
            ),
        )
    )
    return ResearchPolicyResult(result.source, result.sha256, tuple(retained))


def main() -> int:
    _set_limits()
    source_bytes = sys.stdin.buffer.read(MAX_CANDIDATE_SOURCE_BYTES + 1)
    if len(source_bytes) > MAX_CANDIDATE_SOURCE_BYTES:
        print(
            json.dumps(
                {
                    "valid": False,
                    "sha256": "",
                    "errors": [
                        {
                            "code": "POLICY_RESOURCE_LIMIT",
                            "message": "candidate source exceeds 256 KiB",
                            "line": None,
                            "column": None,
                        }
                    ],
                }
            )
        )
        return 0
    try:
        source = source_bytes.decode("utf-8")
        result = _bounded_result(validate_research_candidate(source))
    except (UnicodeError, RecursionError, MemoryError) as exc:
        print(
            json.dumps(
                {
                    "valid": False,
                    "sha256": "",
                    "errors": [
                        {
                            "code": "POLICY_RESOURCE_LIMIT",
                            "message": type(exc).__name__,
                            "line": None,
                            "column": None,
                        }
                    ],
                }
            )
        )
        return 0
    encoded = json.dumps(
        result.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(encoded) > POLICY_OUTPUT_LIMIT_BYTES:
        raise RuntimeError("bounded policy result unexpectedly exceeded output limit")
    sys.stdout.buffer.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
