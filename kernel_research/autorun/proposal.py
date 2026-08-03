"""Proposal protocol, OpenCode event parsing and bounded prompt construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Mapping

from ..constants import MAX_FEEDBACK_CANDIDATES, OPENCODE_PROPOSER_STEPS
from .models import ProposalV1


FENCED_JSON_RE = re.compile(r"\A\s*```json\s*\n(.*?)\n```\s*\Z", re.DOTALL)
STEP_LIMIT_FALLBACK_RE = re.compile(
    r"\A(?:"
    r"maximum steps for this agent have been reached"
    r"|critical\s+[-–—]\s+maximum steps reached"
    r")[.!:]?\Z",
    re.IGNORECASE,
)


class ProposerStepLimitError(ValueError):
    """OpenCode emitted its forced summary after exhausting agent steps."""


class ProposerOutputTokenLimitError(ValueError):
    """OpenCode exhausted the model output budget before ProposalV1."""


class ProposalFormatError(ValueError):
    """The proposer response is not syntactically valid JSON/JSON fencing."""


class ProposerFormatRetryExhaustedError(ValueError):
    """Both strictly bounded Proposal JSON attempts failed syntax parsing."""


@dataclass(frozen=True)
class ProposalRequest:
    parent_candidate_hash: str
    accepted_kernel: str
    program_markdown: str
    environment: Mapping[str, Any]
    accepted_case_p50_us: Mapping[str, float]
    recent_experiments: tuple[Mapping[str, Any], ...]
    session_feedback: tuple[Mapping[str, Any], ...]


class Proposer(ABC):
    @abstractmethod
    def propose(self, request: ProposalRequest) -> ProposalV1:
        """Return one complete candidate proposal."""


def parse_proposal_text(text: str, *, expected_parent_hash: str) -> ProposalV1:
    """Parse either one JSON object or exactly one ``json`` fenced object."""

    fenced = FENCED_JSON_RE.fullmatch(text)
    payload = fenced.group(1) if fenced else text.strip()
    if "```" in payload:
        raise ProposalFormatError(
            "proposal contains malformed or additional fenced content"
        )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProposalFormatError(
            f"proposal is not valid JSON: {exc}"
        ) from exc
    return ProposalV1.from_value(value, expected_parent_hash=expected_parent_hash)


def _contains_forbidden_event(value: Any) -> str | None:
    if isinstance(value, dict):
        event_type = value.get("type")
        if isinstance(event_type, str):
            lowered = event_type.lower()
            if "tool" in lowered:
                return f"tool event {event_type!r}"
            if lowered in {"error", "failed", "failure"}:
                return f"error event {event_type!r}"
        for child in value.values():
            found = _contains_forbidden_event(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _contains_forbidden_event(child)
            if found:
                return found
    return None


@dataclass(frozen=True)
class _LengthFinish:
    reasoning_tokens: int | None
    output_tokens: int | None


def _length_finish(event: Mapping[str, Any]) -> _LengthFinish | None:
    event_type = str(event.get("type", "")).lower().replace("-", "_")
    part = event.get("part")
    if event_type != "step_finish" or not isinstance(part, Mapping):
        return None
    if str(part.get("reason", "")).lower() != "length":
        return None
    tokens = part.get("tokens")
    if not isinstance(tokens, Mapping):
        return _LengthFinish(None, None)

    def token(name: str) -> int | None:
        value = tokens.get(name)
        return value if type(value) is int and value >= 0 else None

    return _LengthFinish(token("reasoning"), token("output"))


def _output_limit_error(finish: _LengthFinish) -> str:
    reasoning = (
        "unknown"
        if finish.reasoning_tokens is None
        else str(finish.reasoning_tokens)
    )
    output = (
        "unknown" if finish.output_tokens is None else str(finish.output_tokens)
    )
    return (
        "PROPOSER_OUTPUT_TOKEN_LIMIT: OpenCode exhausted the model output "
        f"token budget before a complete ProposalV1 (reasoning={reasoning}, "
        f"output={output})"
    )


def parse_opencode_ndjson(
    raw: str, *, expected_parent_hash: str
) -> ProposalV1:
    """Extract final text from OpenCode JSON events and reject tool/error events."""

    text_parts: list[str] = []
    event_count = 0
    length_finish: _LengthFinish | None = None
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        event_count += 1
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProposalFormatError(
                f"OpenCode output line {line_number} is not JSON: {exc}"
            ) from exc
        forbidden = _contains_forbidden_event(event)
        if forbidden:
            raise ValueError(f"OpenCode emitted forbidden {forbidden}")
        if not isinstance(event, dict):
            raise ValueError("OpenCode event must be an object")
        observed_finish = _length_finish(event)
        if observed_finish is not None:
            length_finish = observed_finish
        event_type = str(event.get("type", "")).lower()
        part = event.get("part")
        if (
            event_type == "text"
            and isinstance(part, dict)
            and isinstance(part.get("text"), str)
        ):
            text_parts.append(part["text"])
        elif event_type == "text" and isinstance(event.get("text"), str):
            text_parts.append(event["text"])
        elif event_type in {"message", "assistant"}:
            candidate = event.get("content")
            if isinstance(candidate, str):
                text_parts.append(candidate)
    if event_count == 0:
        raise ValueError("OpenCode produced no JSON events")
    if not text_parts:
        if length_finish is not None:
            raise ProposerOutputTokenLimitError(
                _output_limit_error(length_finish)
            )
        raise ValueError("OpenCode produced no final text event")
    final_text = "".join(text_parts)
    try:
        return parse_proposal_text(
            final_text, expected_parent_hash=expected_parent_hash
        )
    except ProposalFormatError as exc:
        if length_finish is not None:
            raise ProposerOutputTokenLimitError(
                _output_limit_error(length_finish)
            ) from exc
        first_line = final_text.lstrip().splitlines()[0].strip()
        if STEP_LIMIT_FALLBACK_RE.fullmatch(first_line):
            raise ProposerStepLimitError(
                "PROPOSER_STEP_LIMIT: OpenCode exhausted its configured "
                f"{OPENCODE_PROPOSER_STEPS}-step proposal budget"
            ) from exc
        raise
    except ValueError as exc:
        if length_finish is not None:
            raise ProposerOutputTokenLimitError(
                _output_limit_error(length_finish)
            ) from exc
        raise


def build_prompt(request: ProposalRequest) -> str:
    """Build the complete one-shot context supplied to an untrusted proposer."""

    payload = {
        "accepted_candidate_hash": request.parent_candidate_hash,
        "environment": dict(request.environment),
        "accepted_case_p50_us": dict(request.accepted_case_p50_us),
        "recent_candidate_feedback": [
            dict(item)
            for item in request.recent_experiments[-MAX_FEEDBACK_CANDIDATES:]
        ],
        "session_feedback": [dict(item) for item in request.session_feedback],
    }
    sections: Iterable[str] = (
        "You are a Fused MoE Triton kernel proposal engine.",
        "Return exactly one ProposalV1 JSON object and no other text.",
        "The only fields are schema_version=1, parent_candidate_hash, "
        "hypothesis (1..1000 characters), rationale (1..8000 characters), "
        "and kernel_source (the complete non-empty file, at most 256 KiB).",
        "Make one falsifiable optimization hypothesis. Preserve correctness, "
        "the run_kernel interface, and the int64-safe B expert base.",
        "Do not request tools and do not describe edits: kernel_source must be "
        "the complete replacement kernel.py.",
        "Proposal metadata:\n"
        + json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2),
        "Research protocol:\n" + request.program_markdown,
        "Accepted kernel.py:\n```python\n" + request.accepted_kernel + "\n```",
    )
    return "\n\n".join(sections)
