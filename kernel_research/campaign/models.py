"""Dependency-light campaign value objects and state machines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping


MAX_CHILD_CANDIDATES = 5
MAX_CHILD_WALL_SECONDS = 6 * 60 * 60
MAX_CHILD_CONSECUTIVE_FAILURES = 3


class CampaignMode(str, Enum):
    BENCHMARK = "BENCHMARK"
    DISCOVERY = "DISCOVERY"


class CampaignStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED_OPERATOR = "PAUSED_OPERATOR"
    PAUSED_BUDGET = "PAUSED_BUDGET"
    PAUSED_UNKNOWN_OUTCOME = "PAUSED_UNKNOWN_OUTCOME"
    PAUSED_HARD_FAILURE = "PAUSED_HARD_FAILURE"
    PAUSED_DATA_INTEGRITY = "PAUSED_DATA_INTEGRITY"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class ChildRunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PROMOTED = "PROMOTED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    HARD_FAILED = "HARD_FAILED"
    UNKNOWN_GPU_OUTCOME = "UNKNOWN_GPU_OUTCOME"


TERMINAL_CAMPAIGN_STATUSES = frozenset(
    {CampaignStatus.COMPLETED.value, CampaignStatus.CANCELLED.value}
)
TERMINAL_CHILD_STATUSES = frozenset(
    status.value
    for status in ChildRunStatus
    if status not in {ChildRunStatus.PENDING, ChildRunStatus.RUNNING}
)


@dataclass(frozen=True, slots=True)
class BudgetAmount:
    candidates: int = 0
    wall_ms: int = 0
    gpu_ms: int = 0
    tokens: int = 0
    cost_microusd: int = 0

    def __post_init__(self) -> None:
        for name in (
            "candidates",
            "wall_ms",
            "gpu_ms",
            "tokens",
            "cost_microusd",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "candidates": self.candidates,
            "wall_ms": self.wall_ms,
            "gpu_ms": self.gpu_ms,
            "tokens": self.tokens,
            "cost_microusd": self.cost_microusd,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetAmount":
        expected = {
            "candidates",
            "wall_ms",
            "gpu_ms",
            "tokens",
            "cost_microusd",
        }
        unknown = set(value) - expected
        if unknown:
            raise ValueError("unknown budget fields: " + ", ".join(sorted(unknown)))
        return cls(**{name: int(value.get(name, 0)) for name in expected})

    def __add__(self, other: "BudgetAmount") -> "BudgetAmount":
        return BudgetAmount(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.to_dict()
            }
        )

    def fits_within(self, limit: "BudgetAmount") -> bool:
        return all(
            getattr(self, name) <= getattr(limit, name) for name in self.to_dict()
        )


@dataclass(frozen=True, slots=True)
class ResourceLease:
    resource_id: str
    campaign_id: str
    fencing_epoch: int
    expires_epoch: float
    status: str

    def __post_init__(self) -> None:
        if not self.resource_id or not self.campaign_id:
            raise ValueError("resource and campaign IDs must not be empty")
        if self.fencing_epoch <= 0:
            raise ValueError("fencing_epoch must be positive")
        if not math.isfinite(self.expires_epoch):
            raise ValueError("expires_epoch must be finite")


__all__ = [
    "BudgetAmount",
    "CampaignMode",
    "CampaignStatus",
    "ChildRunStatus",
    "MAX_CHILD_CANDIDATES",
    "MAX_CHILD_CONSECUTIVE_FAILURES",
    "MAX_CHILD_WALL_SECONDS",
    "ResourceLease",
    "TERMINAL_CAMPAIGN_STATUSES",
    "TERMINAL_CHILD_STATUSES",
]
