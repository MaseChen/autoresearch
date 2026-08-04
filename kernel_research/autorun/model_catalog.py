"""Immutable allowlist for OpenCode models used by the proposer."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


@dataclass(frozen=True)
class OpenCodeModelSpec:
    """One audited OpenCode/provider model mapping."""

    qualified_id: str
    provider_id: str
    display_name: str
    context_tokens: int
    output_tokens: int
    request_output_token_cap: int
    reasoning_effort: Literal["high", "max"]


DEFAULT_OPENCODE_MODEL = "deepseek/deepseek-v4-pro"
OPENCODE_OUTPUT_TOKEN_MAX_ENV = (
    "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"
)

OPENCODE_MODEL_SPECS: Mapping[str, OpenCodeModelSpec] = MappingProxyType(
    {
        "deepseek/deepseek-v4-pro": OpenCodeModelSpec(
            qualified_id="deepseek/deepseek-v4-pro",
            provider_id="deepseek-v4-pro",
            display_name="DeepSeek V4 Pro",
            context_tokens=1_000_000,
            output_tokens=384_000,
            request_output_token_cap=384_000,
            reasoning_effort="max",
        ),
        "deepseek/deepseek-v4-flash": OpenCodeModelSpec(
            qualified_id="deepseek/deepseek-v4-flash",
            provider_id="deepseek-v4-flash",
            display_name="DeepSeek V4 Flash",
            context_tokens=1_000_000,
            output_tokens=384_000,
            request_output_token_cap=384_000,
            reasoning_effort="max",
        ),
    }
)


def resolve_opencode_model(value: object) -> OpenCodeModelSpec:
    """Resolve one exact allowlisted model without aliases or fallback."""

    allowed = ", ".join(sorted(OPENCODE_MODEL_SPECS))
    if not isinstance(value, str):
        raise ValueError(f"opencode_model must be one of: {allowed}")
    try:
        return OPENCODE_MODEL_SPECS[value]
    except KeyError as exc:
        raise ValueError(f"opencode_model must be one of: {allowed}") from exc


__all__ = [
    "DEFAULT_OPENCODE_MODEL",
    "OPENCODE_MODEL_SPECS",
    "OPENCODE_OUTPUT_TOKEN_MAX_ENV",
    "OpenCodeModelSpec",
    "resolve_opencode_model",
]
