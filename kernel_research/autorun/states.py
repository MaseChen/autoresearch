"""Typed controller states and explicit legal transitions."""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    PROPOSE = "PROPOSE"
    POLICY = "POLICY"
    SMOKE = "SMOKE"
    QUICK = "QUICK"
    FULL_PRIMARY = "FULL_PRIMARY"
    CONFIRMATION = "CONFIRMATION"
    DONE = "DONE"


class RunStatus(str, Enum):
    RUNNING = "RUNNING"
    PROMOTED = "PROMOTED"
    PROPOSAL_READY = "PROPOSAL_READY"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    STOPPED = "STOPPED"
    HARD_FAILED = "HARD_FAILED"
    FAILED = "FAILED"


class IterationStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"


TERMINAL_RUN_STATUSES = frozenset(
    status.value for status in RunStatus if status is not RunStatus.RUNNING
)

STAGE_TRANSITIONS = {
    Stage.PROPOSE: frozenset({Stage.POLICY, Stage.DONE}),
    Stage.POLICY: frozenset({Stage.SMOKE, Stage.DONE}),
    Stage.SMOKE: frozenset({Stage.QUICK, Stage.DONE}),
    Stage.QUICK: frozenset({Stage.FULL_PRIMARY, Stage.DONE}),
    Stage.FULL_PRIMARY: frozenset({Stage.CONFIRMATION, Stage.DONE}),
    Stage.CONFIRMATION: frozenset({Stage.DONE}),
    Stage.DONE: frozenset(),
}


def validate_stage_transition(current: str, target: str) -> None:
    try:
        current_stage = Stage(current)
        target_stage = Stage(target)
    except ValueError as exc:
        raise ValueError(f"unknown iteration stage: {exc}") from exc
    if current_stage == target_stage:
        return
    if target_stage not in STAGE_TRANSITIONS[current_stage]:
        raise ValueError(
            f"illegal iteration stage transition: "
            f"{current_stage.value} -> {target_stage.value}"
        )
