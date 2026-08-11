"""Frozen, executable benchmark cohorts and derived aggregate reports.

The public CLI never accepts benchmark observations.  A cohort plan is bound
into the immutable Campaign snapshot, each assignment gets a deterministic
Controller run identity, and the final report is reconstructed from the
Campaign and Controller ledgers.  :func:`build_benchmark_report` remains a
small pure aggregation primitive for internal callers and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..autorun.summary import feedback_for_iteration
from ..autorun.states import TERMINAL_RUN_STATUSES
from ..autorun.store import ControllerStore
from ..constants import MAX_FEEDBACK_CANDIDATES
from ..history import HistoryStore
from ..platform.canonical import (
    canonical_json_text,
    canonical_sha256,
    require_sha256_digest,
)
from ..platform.identity import BaselineRef
from ..platform.profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    ProfileRef,
    ResearchNamespace,
)
from .models import (
    BudgetAmount,
    CampaignMode,
    CampaignStatus,
    MAX_CHILD_CANDIDATES,
    MAX_CHILD_CONSECUTIVE_FAILURES,
    MAX_CHILD_WALL_SECONDS,
    TERMINAL_CHILD_STATUSES,
)
from .store import CampaignStore
from .supervisor import CampaignSupervisor, ChildRunPlan, child_budget_idempotency_key


_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_ACTIVE_EXECUTABLE_PROPOSERS = frozenset(
    {
        "opencode-deepseek-v4-pro",
        "opencode-deepseek-v4-flash",
    }
)
CURRENT_PROMPT_PROTOCOL_DIGEST = canonical_sha256(
    {
        "id": "proposal-v1",
        "revision": "v1",
        "format": "single-json-object",
    }
)
_BENCHMARK_CAMPAIGN_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "campaign_profile",
        "deployment_profile",
        "namespace",
        "cohort_id",
        "history_cutoff",
        "prompt_protocol_digest",
        "feedback_snapshot_digest",
        "feedback_snapshot",
        "benchmark_plan_digest",
        "benchmark_plan",
    }
)


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ValueError(f"{name} must be a stable bounded identifier")
    return value


@dataclass(frozen=True, slots=True)
class BenchmarkArm:
    arm_id: str
    proposer_profile: ProfileRef

    def __post_init__(self) -> None:
        _token(self.arm_id, "arm_id")
        if (
            not isinstance(self.proposer_profile, ProfileRef)
            or self.proposer_profile.kind != "proposer"
        ):
            raise ValueError("benchmark arm requires an exact proposer profile")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "proposer_profile": self.proposer_profile.to_dict(),
        }

    @classmethod
    def from_value(cls, value: object) -> "BenchmarkArm":
        if not isinstance(value, dict) or set(value) != {
            "arm_id",
            "proposer_profile",
        }:
            raise ValueError("benchmark arm fields do not match the schema")
        return cls(
            arm_id=value["arm_id"],
            proposer_profile=ProfileRef.from_value(value["proposer_profile"]),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkAssignment:
    sequence_index: int
    round_index: int
    arm_id: str
    proposer_profile: ProfileRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "round_index": self.round_index,
            "arm_id": self.arm_id,
            "proposer_profile": self.proposer_profile.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkCohortPlan:
    cohort_id: str
    namespace: ResearchNamespace
    baseline: BaselineRef
    history_cutoff: int
    prompt_protocol_digest: str
    feedback_snapshot_digest: str
    arms: tuple[BenchmarkArm, ...]
    repetitions: int
    max_candidates: int = MAX_CHILD_CANDIDATES
    max_wall_seconds: int = MAX_CHILD_WALL_SECONDS
    max_consecutive_failures: int = MAX_CHILD_CONSECUTIVE_FAILURES

    def __post_init__(self) -> None:
        _token(self.cohort_id, "cohort_id")
        if not isinstance(self.namespace, ResearchNamespace):
            raise ValueError("namespace must be a ResearchNamespace")
        if not isinstance(self.baseline, BaselineRef):
            raise ValueError("baseline must be a BaselineRef")
        if self.baseline.namespace_id != self.namespace.namespace_id:
            raise ValueError("benchmark baseline belongs to another namespace")
        if type(self.history_cutoff) is not int or self.history_cutoff < 0:
            raise ValueError("history_cutoff must be a frozen non-negative integer")
        require_sha256_digest(
            self.prompt_protocol_digest, field="prompt_protocol_digest"
        )
        require_sha256_digest(
            self.feedback_snapshot_digest, field="feedback_snapshot_digest"
        )
        if not isinstance(self.arms, tuple) or any(
            not isinstance(arm, BenchmarkArm) for arm in self.arms
        ):
            raise ValueError("benchmark arms must be an immutable arm tuple")
        if len(self.arms) < 2:
            raise ValueError("benchmark cohort requires at least two arms")
        if len({arm.arm_id for arm in self.arms}) != len(self.arms):
            raise ValueError("benchmark arm IDs must be unique")
        if type(self.repetitions) is not int or not 1 <= self.repetitions <= 100:
            raise ValueError("benchmark repetitions must be between 1 and 100")
        for name, value, maximum in (
            ("max_candidates", self.max_candidates, MAX_CHILD_CANDIDATES),
            ("max_wall_seconds", self.max_wall_seconds, MAX_CHILD_WALL_SECONDS),
            (
                "max_consecutive_failures",
                self.max_consecutive_failures,
                MAX_CHILD_CONSECUTIVE_FAILURES,
            ),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} is outside the bounded child limit")

    @property
    def snapshot(self) -> dict[str, Any]:
        material = {
            "schema_version": 1,
            "mode": "BENCHMARK",
            "cohort_id": self.cohort_id,
            "namespace": self.namespace.to_dict(),
            "baseline": self.baseline.to_dict(),
            "history_cutoff": self.history_cutoff,
            "prompt_protocol_digest": self.prompt_protocol_digest,
            "feedback_snapshot_digest": self.feedback_snapshot_digest,
            "arms": [arm.to_dict() for arm in self.arms],
            "repetitions": self.repetitions,
            "child_limits": {
                "max_candidates": self.max_candidates,
                "max_wall_seconds": self.max_wall_seconds,
                "max_consecutive_failures": self.max_consecutive_failures,
                "stop_after_promotion": True,
            },
        }
        return {**material, "snapshot_digest": canonical_sha256(material)}

    def schedule(self) -> tuple[BenchmarkAssignment, ...]:
        """Return deterministic balanced interleaving across cohort arms."""

        seed = int.from_bytes(
            hashlib.sha256(self.cohort_id.encode("utf-8")).digest()[:8], "big"
        )
        base = list(self.arms)
        assignments: list[BenchmarkAssignment] = []
        sequence = 0
        for round_index in range(self.repetitions):
            offset = (seed + round_index) % len(base)
            ordered = base[offset:] + base[:offset]
            if round_index % 2:
                ordered = list(reversed(ordered))
            for arm in ordered:
                assignments.append(
                    BenchmarkAssignment(
                        sequence_index=sequence,
                        round_index=round_index,
                        arm_id=arm.arm_id,
                        proposer_profile=arm.proposer_profile,
                    )
                )
                sequence += 1
        return tuple(assignments)

    @classmethod
    def from_snapshot(cls, value: object) -> "BenchmarkCohortPlan":
        expected = {
            "schema_version",
            "mode",
            "cohort_id",
            "namespace",
            "baseline",
            "history_cutoff",
            "prompt_protocol_digest",
            "feedback_snapshot_digest",
            "arms",
            "repetitions",
            "child_limits",
            "snapshot_digest",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("benchmark snapshot fields do not match the schema")
        if value["schema_version"] != 1 or value["mode"] != "BENCHMARK":
            raise ValueError("benchmark snapshot version or mode is invalid")
        arms = value["arms"]
        if not isinstance(arms, list):
            raise ValueError("benchmark snapshot arms must be a list")
        limits = value["child_limits"]
        if not isinstance(limits, dict) or set(limits) != {
            "max_candidates",
            "max_wall_seconds",
            "max_consecutive_failures",
            "stop_after_promotion",
        }:
            raise ValueError("benchmark child limits do not match the schema")
        if limits["stop_after_promotion"] is not True:
            raise ValueError("benchmark children must stop after promotion")
        plan = cls(
            cohort_id=value["cohort_id"],
            namespace=ResearchNamespace.from_value(value["namespace"]),
            baseline=BaselineRef.from_value(value["baseline"]),
            history_cutoff=value["history_cutoff"],
            prompt_protocol_digest=value["prompt_protocol_digest"],
            feedback_snapshot_digest=value["feedback_snapshot_digest"],
            arms=tuple(BenchmarkArm.from_value(arm) for arm in arms),
            repetitions=value["repetitions"],
            max_candidates=limits["max_candidates"],
            max_wall_seconds=limits["max_wall_seconds"],
            max_consecutive_failures=limits["max_consecutive_failures"],
        )
        if value["snapshot_digest"] != plan.snapshot["snapshot_digest"]:
            raise ValueError("benchmark snapshot digest does not match its contents")
        return plan


def bind_benchmark_campaign_snapshot(
    base_snapshot: Mapping[str, Any],
    plan: BenchmarkCohortPlan,
    *,
    feedback_snapshot: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind one complete cohort plan and its exact prompt feedback snapshot."""

    if not isinstance(plan, BenchmarkCohortPlan):
        raise TypeError("plan must be a BenchmarkCohortPlan")
    expected_base = _BENCHMARK_CAMPAIGN_FIELDS - {
        "feedback_snapshot",
        "benchmark_plan_digest",
        "benchmark_plan",
    }
    if not isinstance(base_snapshot, Mapping) or set(base_snapshot) != expected_base:
        raise ValueError("benchmark Campaign base snapshot fields are invalid")
    base = json.loads(canonical_json_text(dict(base_snapshot)))
    expected = {
        "schema_version": 1,
        "mode": CampaignMode.BENCHMARK.value,
        "namespace": plan.namespace.to_dict(),
        "cohort_id": plan.cohort_id,
        "history_cutoff": plan.history_cutoff,
        "prompt_protocol_digest": plan.prompt_protocol_digest,
        "feedback_snapshot_digest": plan.feedback_snapshot_digest,
    }
    if any(base.get(name) != selected for name, selected in expected.items()):
        raise ValueError("benchmark plan differs from its Campaign base snapshot")
    feedback = _canonical_feedback_snapshot(feedback_snapshot)
    if canonical_sha256(feedback) != plan.feedback_snapshot_digest:
        raise ValueError(
            "benchmark feedback snapshot digest differs from the cohort plan"
        )
    plan_snapshot = plan.snapshot
    return {
        **base,
        "feedback_snapshot": feedback,
        "benchmark_plan_digest": plan_snapshot["snapshot_digest"],
        "benchmark_plan": plan_snapshot,
    }


def benchmark_plan_from_campaign_snapshot(
    value: Mapping[str, Any],
) -> BenchmarkCohortPlan:
    """Parse and cross-check the plan embedded in a BENCHMARK Campaign."""

    if not isinstance(value, Mapping) or set(value) != _BENCHMARK_CAMPAIGN_FIELDS:
        raise ValueError("benchmark Campaign snapshot fields do not match schema")
    snapshot = json.loads(canonical_json_text(dict(value)))
    plan = BenchmarkCohortPlan.from_snapshot(snapshot["benchmark_plan"])
    if snapshot["benchmark_plan_digest"] != plan.snapshot["snapshot_digest"]:
        raise ValueError("benchmark Campaign plan digest is inconsistent")
    expected = {
        "schema_version": 1,
        "mode": CampaignMode.BENCHMARK.value,
        "namespace": plan.namespace.to_dict(),
        "cohort_id": plan.cohort_id,
        "history_cutoff": plan.history_cutoff,
        "prompt_protocol_digest": plan.prompt_protocol_digest,
        "feedback_snapshot_digest": plan.feedback_snapshot_digest,
    }
    if any(snapshot.get(name) != selected for name, selected in expected.items()):
        raise ValueError("benchmark Campaign snapshot differs from its plan")
    feedback = _canonical_feedback_snapshot(snapshot["feedback_snapshot"])
    if canonical_sha256(feedback) != plan.feedback_snapshot_digest:
        raise ValueError("benchmark Campaign feedback snapshot digest is inconsistent")
    return plan


def _canonical_feedback_snapshot(
    value: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("benchmark feedback snapshot must be a JSON array")
    if len(value) > MAX_FEEDBACK_CANDIDATES:
        raise ValueError("benchmark feedback snapshot exceeds the trusted limit")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError("benchmark feedback snapshot entries must be objects")
    normalized = json.loads(
        canonical_json_text([dict(item) for item in value])
    )
    if not isinstance(normalized, list) or any(
        not isinstance(item, dict) for item in normalized
    ):
        raise ValueError("benchmark feedback snapshot is not strict JSON")
    return normalized


def derive_authoritative_benchmark_plan(
    draft: BenchmarkCohortPlan,
    controller: ControllerStore,
    history: HistoryStore,
) -> tuple[BenchmarkCohortPlan, tuple[dict[str, Any], ...]]:
    """Derive the only executable cutoff and feedback from trusted ledgers.

    ``benchmark create`` deliberately produces a planning draft.  This
    boundary ignores its caller-supplied cutoff and feedback digest, freezes
    the current History high-water mark, then projects the complete bounded
    Controller feedback visible under that cutoff.  Every Controller link is
    reconciled against History before the snapshot becomes executable.
    """

    if not isinstance(draft, BenchmarkCohortPlan):
        raise TypeError("draft must be a BenchmarkCohortPlan")
    records = history.list_all_experiments(limit=1, newest_first=True)
    history_cutoff = 0 if not records else int(records[0].id)
    protocol = BUILTIN_PROFILE_REGISTRY.resolve(
        draft.namespace.evaluation_protocol
    )
    roles = protocol.config.get("case_roles")
    if not isinstance(roles, Mapping):
        raise ValueError("benchmark evaluation protocol has no trusted case roles")
    hidden_case_ids = frozenset(
        str(case_id)
        for case_id, role in roles.items()
        if role == "holdout"
    )
    recent_items = controller.list_recent_scientific_iterations(
        exclude_run_id=None,
        limit=MAX_FEEDBACK_CANDIDATES,
        namespace_id=draft.namespace.namespace_id,
        history_cutoff=history_cutoff,
    )
    feedback: list[dict[str, Any]] = []
    for item in reversed(recent_items):
        run_id = item.get("run_id")
        iteration_id = item.get("id")
        if not isinstance(run_id, str) or type(iteration_id) is not int:
            raise ValueError("benchmark feedback has invalid Controller ownership")
        run = controller.get_run(run_id)
        if run.get("namespace_id") != draft.namespace.namespace_id:
            raise ValueError("benchmark feedback belongs to another namespace")
        attempts = controller.list_evaluation_attempts(
            run_id, iteration_id=iteration_id
        )
        succeeded = [
            attempt for attempt in attempts if attempt.get("status") == "SUCCEEDED"
        ]
        if not succeeded or any(
            type(attempt.get("history_experiment_id")) is not int
            for attempt in succeeded
        ):
            raise ValueError(
                "benchmark feedback has an unreconciled History experiment"
            )
        linked_ids = {
            int(attempt["history_experiment_id"]) for attempt in succeeded
        }
        declared = item.get("experiment_ids")
        declared_ids = (
            list(declared.values()) if isinstance(declared, Mapping) else []
        )
        if (
            not isinstance(declared, Mapping)
            or any(type(experiment_id) is not int for experiment_id in declared_ids)
            or len(declared_ids) != len(linked_ids)
            or set(declared_ids) != linked_ids
        ):
            raise ValueError(
                "benchmark feedback Controller experiment links are inconsistent"
            )
        for attempt in succeeded:
            experiment_id = int(attempt["history_experiment_id"])
            record = history.get_experiment(experiment_id)
            if (
                experiment_id > history_cutoff
                or record is None
                or record.experiment_uid != attempt.get("experiment_uid")
                or record.namespace_id != draft.namespace.namespace_id
            ):
                raise ValueError(
                    "benchmark feedback does not reconcile with frozen History"
                )
        feedback.append(
            feedback_for_iteration(item, hidden_case_ids=hidden_case_ids)
        )
    canonical_feedback = tuple(_canonical_feedback_snapshot(feedback))
    authoritative = replace(
        draft,
        history_cutoff=history_cutoff,
        feedback_snapshot_digest=canonical_sha256(list(canonical_feedback)),
    )
    return authoritative, canonical_feedback


def require_executable_benchmark_plan(plan: BenchmarkCohortPlan) -> None:
    """Reject planning-only Harness profiles at the trusted execution edge."""

    if not isinstance(plan, BenchmarkCohortPlan):
        raise TypeError("plan must be a BenchmarkCohortPlan")
    if plan.namespace not in {
        LEGACY_RESEARCH_NAMESPACE,
        CURRENT_RESEARCH_NAMESPACE,
    }:
        raise ValueError(
            "benchmark execution requires an exact active target namespace"
        )
    if plan.baseline.source != "campaign":
        raise ValueError("executable benchmark baseline must be a Campaign baseline")
    if (
        plan.namespace == CURRENT_RESEARCH_NAMESPACE
        and not plan.baseline.is_scientifically_comparable
    ):
        raise ValueError("CURRENT benchmark baseline must bind a resolved environment")
    if plan.prompt_protocol_digest != CURRENT_PROMPT_PROTOCOL_DIGEST:
        raise ValueError(
            "benchmark execution requires the exact active prompt protocol"
        )
    for arm in plan.arms:
        try:
            definition = BUILTIN_PROFILE_REGISTRY.resolve(arm.proposer_profile)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "benchmark proposer is not an exact trusted profile"
            ) from exc
        if (
            definition.ref.id not in _ACTIVE_EXECUTABLE_PROPOSERS
            or definition.implementation_id != definition.ref.id
            or definition.config.get("harness") != "opencode"
        ):
            raise ValueError(
                "benchmark execution currently permits only active OpenCode profiles"
            )


def benchmark_controller_run_id(
    campaign_id: str,
    plan: BenchmarkCohortPlan,
    assignment: BenchmarkAssignment,
) -> str:
    """Return the only Controller run ID authorized for an assignment."""

    _token(campaign_id, "campaign_id")
    if not isinstance(plan, BenchmarkCohortPlan):
        raise TypeError("plan must be a BenchmarkCohortPlan")
    if not isinstance(assignment, BenchmarkAssignment):
        raise TypeError("assignment must be a BenchmarkAssignment")
    material = (
        f"{campaign_id}\x00{plan.snapshot['snapshot_digest']}\x00"
        f"{assignment.sequence_index}\x00{assignment.arm_id}"
    )
    return "benchmark-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:48]


def _campaign_plan(
    store: CampaignStore, campaign_id: str
) -> tuple[dict[str, Any], BenchmarkCohortPlan, list[dict[str, Any]]]:
    campaign = store.get_campaign(campaign_id)
    if campaign.get("mode") != CampaignMode.BENCHMARK.value:
        raise ValueError("campaign is not a BENCHMARK campaign")
    snapshot = campaign.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("benchmark Campaign has no frozen snapshot")
    if canonical_sha256(snapshot) != campaign.get("snapshot_digest"):
        raise ValueError("benchmark Campaign snapshot digest is inconsistent")
    plan = benchmark_plan_from_campaign_snapshot(snapshot)
    require_executable_benchmark_plan(plan)
    if campaign.get("namespace_id") != plan.namespace.namespace_id:
        raise ValueError("benchmark Campaign belongs to another namespace")
    if campaign.get("allow_staged_lineage") is not False:
        raise ValueError("benchmark Campaign must permanently disable lineage")
    revisions = store.list_baseline_revisions(campaign_id)
    if len(revisions) != 1:
        raise ValueError("benchmark Campaign must have exactly one frozen baseline")
    revision = revisions[0]
    if (
        revision.get("id") != campaign.get("active_baseline_revision_id")
        or revision.get("revision_kind") != "DEPLOYMENT_SEED"
        or revision.get("baseline_ref") != plan.baseline.to_dict()
        or revision.get("artifact_id") != str(plan.baseline.artifact_id)
    ):
        raise ValueError("benchmark Campaign baseline differs from its plan")
    children = store.list_child_runs(campaign_id)
    _validate_child_prefix(campaign_id, plan, revision, children)
    return campaign, plan, children


def _validate_child_prefix(
    campaign_id: str,
    plan: BenchmarkCohortPlan,
    baseline_revision: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
) -> None:
    schedule = plan.schedule()
    if len(children) > len(schedule):
        raise ValueError("benchmark Campaign has children beyond its frozen schedule")
    for index, child in enumerate(children):
        assignment = schedule[index]
        expected = {
            "campaign_id": campaign_id,
            "child_index": index + 1,
            "controller_run_id": benchmark_controller_run_id(
                campaign_id, plan, assignment
            ),
            "baseline_revision_id": baseline_revision.get("id"),
            "proposer_profile": assignment.proposer_profile.to_dict(),
            "max_candidates": plan.max_candidates,
            "max_wall_seconds": plan.max_wall_seconds,
            "max_consecutive_failures": plan.max_consecutive_failures,
            "stop_after_promotion": True,
        }
        if any(child.get(name) != selected for name, selected in expected.items()):
            raise ValueError(
                f"benchmark child {index + 1} differs from its frozen assignment"
            )


def _execution_summary(
    campaign: Mapping[str, Any],
    plan: BenchmarkCohortPlan,
    children: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": campaign["id"],
        "campaign_status": campaign["status"],
        "cohort_id": plan.cohort_id,
        "benchmark_plan_digest": plan.snapshot["snapshot_digest"],
        "scheduled_children": len(plan.schedule()),
        "terminal_children": sum(
            child.get("status") in TERMINAL_CHILD_STATUSES for child in children
        ),
        "lineage_advanced": False,
    }


def execute_benchmark_campaign(
    store: CampaignStore,
    supervisor: CampaignSupervisor,
    *,
    campaign_id: str,
    reservation_builder: Callable[..., BudgetAmount],
) -> dict[str, Any]:
    """Execute/reconcile the frozen interleaved schedule to a terminal Campaign."""

    if not isinstance(store, CampaignStore):
        raise TypeError("store must be a CampaignStore")
    if not isinstance(supervisor, CampaignSupervisor):
        raise TypeError("supervisor must be a CampaignSupervisor")
    if supervisor.store is not store:
        raise ValueError(
            "benchmark Supervisor must own the authoritative CampaignStore"
        )
    if not callable(reservation_builder):
        raise TypeError("reservation_builder must be callable")
    campaign, plan, children = _campaign_plan(store, campaign_id)
    schedule = plan.schedule()
    if campaign["status"] == CampaignStatus.COMPLETED.value:
        if len(children) != len(schedule) or any(
            child["status"] not in TERMINAL_CHILD_STATUSES for child in children
        ):
            raise ValueError("completed benchmark Campaign has an incomplete schedule")
    elif campaign["status"] == CampaignStatus.CREATED.value:
        campaign = store.start_campaign(campaign_id)
    elif campaign["status"] not in {
        CampaignStatus.RUNNING.value,
        CampaignStatus.PAUSED_OPERATOR.value,
    }:
        return _execution_summary(campaign, plan, children)

    for index, assignment in enumerate(schedule):
        campaign, current_plan, children = _campaign_plan(store, campaign_id)
        if current_plan != plan:
            raise ValueError("benchmark plan changed during execution")
        existing = children[index] if index < len(children) else None
        if existing is not None and existing["status"] in TERMINAL_CHILD_STATUSES:
            run_id = benchmark_controller_run_id(campaign_id, plan, assignment)
            action = store.get_budget_action(
                campaign_id,
                idempotency_key=child_budget_idempotency_key(run_id),
            )
            reservation = BudgetAmount.from_mapping(action["reserved"])
            # All invariant reservation axes can be re-derived even after a
            # SETTLED action released its unused allowance.  Provider cost is
            # intentionally the full remaining cap observed at intent time,
            # so its persisted positive value is used as that frozen input.
            expected = reservation_builder(
                proposer_profile=assignment.proposer_profile.to_dict(),
                campaign_remaining=reservation,
                max_candidates=plan.max_candidates,
                max_wall_seconds=plan.max_wall_seconds,
                max_consecutive_failures=plan.max_consecutive_failures,
            )
            if reservation != expected:
                raise ValueError(
                    "terminal benchmark child reservation is not trustworthy"
                )
            supervisor.execute_child(
                campaign_id,
                ChildRunPlan(
                    controller_run_id=run_id,
                    proposer_profile=assignment.proposer_profile.to_dict(),
                    reservation=reservation,
                    max_candidates=plan.max_candidates,
                    max_wall_seconds=plan.max_wall_seconds,
                    max_consecutive_failures=plan.max_consecutive_failures,
                ),
            )
            continue
        if campaign["status"] == CampaignStatus.PAUSED_OPERATOR.value:
            previous = children[index - 1] if index else None
            if previous is None or previous.get("status") != "PROMOTED":
                raise ValueError("benchmark Campaign has an unexplained operator pause")
            campaign = store.resume_campaign(campaign_id)
        if campaign["status"] != CampaignStatus.RUNNING.value:
            return _execution_summary(campaign, plan, children)

        run_id = benchmark_controller_run_id(campaign_id, plan, assignment)
        remaining = BudgetAmount.from_mapping(
            store.budget_status(campaign_id)["remaining"]
        )
        if existing is None:
            reservation = reservation_builder(
                proposer_profile=assignment.proposer_profile.to_dict(),
                campaign_remaining=remaining,
                max_candidates=plan.max_candidates,
                max_wall_seconds=plan.max_wall_seconds,
                max_consecutive_failures=plan.max_consecutive_failures,
            )
        else:
            action = store.get_budget_action(
                campaign_id,
                idempotency_key=child_budget_idempotency_key(run_id),
            )
            if action.get("status") != "RESERVED":
                raise ValueError("nonterminal benchmark child has no reservation")
            reservation = BudgetAmount.from_mapping(action["reserved"])
            expected = reservation_builder(
                proposer_profile=assignment.proposer_profile.to_dict(),
                campaign_remaining=remaining + reservation,
                max_candidates=plan.max_candidates,
                max_wall_seconds=plan.max_wall_seconds,
                max_consecutive_failures=plan.max_consecutive_failures,
            )
            if reservation != expected:
                raise ValueError(
                    "benchmark child reservation differs from trusted derivation"
                )
        result = supervisor.execute_child(
            campaign_id,
            ChildRunPlan(
                controller_run_id=run_id,
                proposer_profile=assignment.proposer_profile.to_dict(),
                reservation=reservation,
                max_candidates=plan.max_candidates,
                max_wall_seconds=plan.max_wall_seconds,
                max_consecutive_failures=plan.max_consecutive_failures,
            ),
        )
        campaign = result["campaign"]
        if result.get("child") is None:
            return _execution_summary(
                campaign, plan, store.list_child_runs(campaign_id)
            )
        if campaign["status"] in {
            CampaignStatus.PAUSED_BUDGET.value,
            CampaignStatus.PAUSED_HARD_FAILURE.value,
            CampaignStatus.PAUSED_UNKNOWN_OUTCOME.value,
            CampaignStatus.PAUSED_DATA_INTEGRITY.value,
        }:
            return _execution_summary(
                campaign, plan, store.list_child_runs(campaign_id)
            )

    campaign, current_plan, children = _campaign_plan(store, campaign_id)
    if current_plan != plan or len(children) != len(schedule) or any(
        child["status"] not in TERMINAL_CHILD_STATUSES for child in children
    ):
        raise ValueError("benchmark schedule did not reach a complete terminal prefix")
    campaign = store.finish_campaign(
        campaign_id, reason="frozen benchmark schedule completed without lineage"
    )
    return _execution_summary(campaign, plan, children)


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    arm_id: str
    protocol_success: bool
    valid_candidates: int
    full_reached: int
    gpu_ms: int
    tokens: int
    cost_microusd: int
    promotions: int

    def __post_init__(self) -> None:
        _token(self.arm_id, "arm_id")
        if type(self.protocol_success) is not bool:
            raise ValueError("protocol_success must be boolean")
        for name in (
            "valid_candidates",
            "full_reached",
            "gpu_ms",
            "tokens",
            "cost_microusd",
            "promotions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.valid_candidates > 5:
            raise ValueError("one observation cannot exceed the child candidate cap")
        if self.full_reached > self.valid_candidates:
            raise ValueError("full_reached cannot exceed valid_candidates")
        if self.promotions > self.full_reached:
            raise ValueError("promotions cannot exceed full_reached")

    @classmethod
    def from_value(cls, value: object) -> "BenchmarkObservation":
        expected = {
            "arm_id",
            "protocol_success",
            "valid_candidates",
            "full_reached",
            "gpu_ms",
            "tokens",
            "cost_microusd",
            "promotions",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("benchmark observation fields do not match the schema")
        return cls(**{name: value[name] for name in expected})


def _duration_ms(start: object, finish: object) -> int | None:
    if not isinstance(start, str) or not isinstance(finish, str):
        return None
    try:
        start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
        finish_time = datetime.fromisoformat(finish.replace("Z", "+00:00"))
    except ValueError:
        return None
    seconds = (finish_time - start_time).total_seconds()
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return math.ceil(seconds * 1000)


def _derived_observation(
    store: CampaignStore,
    controller: ControllerStore,
    history: HistoryStore,
    *,
    campaign: Mapping[str, Any],
    plan: BenchmarkCohortPlan,
    assignment: BenchmarkAssignment,
    child: Mapping[str, Any],
) -> BenchmarkObservation:
    run_id = benchmark_controller_run_id(str(campaign["id"]), plan, assignment)
    if child.get("controller_run_id") != run_id:
        raise ValueError("benchmark child has the wrong Controller run identity")
    run = controller.get_run(run_id)
    if run.get("status") not in TERMINAL_RUN_STATUSES:
        raise ValueError("benchmark Controller run is not terminal")
    expected_run_status = {
        "PROMOTED": "PROMOTED",
        "BUDGET_EXHAUSTED": "BUDGET_EXHAUSTED",
        "STOPPED": "STOPPED",
        "FAILED": "FAILED",
    }.get(str(child.get("status")))
    if expected_run_status is None or run.get("status") != expected_run_status:
        raise ValueError("benchmark child and Controller terminal states differ")
    expected_run = {
        "id": run_id,
        "namespace_id": plan.namespace.namespace_id,
        "baseline_ref": plan.baseline.to_dict(),
        "history_cutoff": plan.history_cutoff,
    }
    if any(run.get(name) != selected for name, selected in expected_run.items()):
        raise ValueError("benchmark Controller run differs from its frozen assignment")
    workflow = run.get("workflow_snapshot")
    if not isinstance(workflow, Mapping):
        raise ValueError("benchmark Controller run has no workflow snapshot")
    expected_workflow = {
        "mode": CampaignMode.BENCHMARK.value,
        "namespace": plan.namespace.to_dict(),
        "baseline_ref": plan.baseline.to_dict(),
        "proposer_profile": assignment.proposer_profile.to_dict(),
        "history_cutoff": plan.history_cutoff,
        "cohort_id": plan.cohort_id,
        "prompt_protocol_digest": plan.prompt_protocol_digest,
        "feedback_snapshot_digest": plan.feedback_snapshot_digest,
        "campaign_id": campaign["id"],
        "campaign_snapshot_digest": campaign["snapshot_digest"],
        "campaign_snapshot": campaign["snapshot"],
        "budget": {
            "max_candidates": plan.max_candidates,
            "max_wall_seconds": plan.max_wall_seconds,
            "max_consecutive_failures": plan.max_consecutive_failures,
            "stop_after_promotion": True,
        },
        "campaign_child": {
            "child_id": child["id"],
            "child_index": assignment.sequence_index + 1,
            "controller_run_id": run_id,
            "baseline_revision_id": child["baseline_revision_id"],
        },
    }
    if any(
        workflow.get(name) != selected
        for name, selected in expected_workflow.items()
    ):
        raise ValueError("benchmark workflow snapshot differs from its assignment")
    if run.get("resolved_config_digest") != workflow.get("snapshot_digest"):
        raise ValueError("benchmark workflow digest column is inconsistent")
    material = dict(workflow)
    observed_digest = material.pop("snapshot_digest", None)
    if observed_digest != canonical_sha256(material):
        raise ValueError("benchmark workflow snapshot digest is invalid")

    proposals = controller.list_proposal_attempts(run_id)
    evaluations = controller.list_evaluation_attempts(run_id)
    history_records: dict[str, Any] = {}
    for proposal in proposals:
        if proposal.get("status") not in {"SUCCEEDED", "FAILED", "UNKNOWN_OUTCOME"}:
            raise ValueError("benchmark proposal attempt is not terminal")
        request = proposal.get("request")
        prompt_protocol = proposal.get("prompt_protocol")
        if (
            proposal.get("proposer_profile") != assignment.proposer_profile.to_dict()
            or not isinstance(request, Mapping)
            or request.get("namespace_id") != plan.namespace.namespace_id
            or request.get("baseline_ref") != plan.baseline.to_dict()
            or request.get("history_cutoff") != plan.history_cutoff
            or request.get("proposer_profile")
            != assignment.proposer_profile.to_dict()
            or request.get("recent_feedback_digest")
            != plan.feedback_snapshot_digest
            or not isinstance(prompt_protocol, Mapping)
            or prompt_protocol.get("digest") != plan.prompt_protocol_digest
            or request.get("prompt_protocol") != prompt_protocol
        ):
            raise ValueError("benchmark proposal provenance differs from the cohort")
    for evaluation in evaluations:
        if evaluation.get("status") not in {
            "SUCCEEDED",
            "FAILED",
            "UNKNOWN_OUTCOME",
        }:
            raise ValueError("benchmark evaluation attempt is not terminal")
        request = evaluation.get("request")
        if (
            evaluation.get("baseline_ref") != plan.baseline.to_dict()
            or not isinstance(request, Mapping)
            or request.get("namespace") != plan.namespace.to_dict()
            or request.get("baseline") != plan.baseline.to_dict()
            or request.get("mode") != CampaignMode.BENCHMARK.value
            or request.get("cohort_id") != plan.cohort_id
            or request.get("history_cutoff") != plan.history_cutoff
            or request.get("proposer_profile")
            != assignment.proposer_profile.to_dict()
            or request.get("feedback_digest") != plan.feedback_snapshot_digest
        ):
            raise ValueError("benchmark evaluation provenance differs from the cohort")
        prompt_digest = request.get("prompt_digest")
        if prompt_digest is not None:
            try:
                require_sha256_digest(prompt_digest, field="evaluation prompt_digest")
            except ValueError as exc:
                raise ValueError(
                    "benchmark evaluation prompt digest is invalid"
                ) from exc
        if (
            evaluation.get("status") == "SUCCEEDED"
            and type(evaluation.get("history_experiment_id")) is not int
        ):
            raise ValueError("succeeded benchmark evaluation is not linked to History")
        if evaluation.get("status") == "SUCCEEDED":
            history_record = history.get_experiment(
                int(evaluation["history_experiment_id"])
            )
            if (
                history_record is None
                or history_record.experiment_uid
                != evaluation.get("experiment_uid")
                or history_record.namespace_id != plan.namespace.namespace_id
                or history_record.artifact_id
                != evaluation.get("candidate_artifact_id")
                or history_record.condition_digest
                != evaluation.get("condition_digest")
                or history_record.status != "SUCCESS"
                or history_record.identity != request
            ):
                raise ValueError(
                    "benchmark Controller/History experiment link is inconsistent"
                )
            history_records[str(evaluation["experiment_uid"])] = history_record

    result = child.get("result")
    expected_result_fields = {
        "schema_version",
        "controller_run_id",
        "budget_idempotency_key",
        "outcome",
        "usage",
        "details",
        "resource_lease",
        "promotion",
    }
    if (
        not isinstance(result, Mapping)
        or set(result) != expected_result_fields
        or result.get("schema_version") != 1
        or result.get("controller_run_id") != run_id
        or result.get("outcome") != child.get("status")
        or result.get("budget_idempotency_key")
        != child_budget_idempotency_key(run_id)
    ):
        raise ValueError("benchmark child result is not trusted Supervisor evidence")
    usage = BudgetAmount.from_mapping(result["usage"])
    action = store.get_budget_action(
        str(campaign["id"]), idempotency_key=child_budget_idempotency_key(run_id)
    )
    if action.get("status") != "SETTLED" or action.get("actual") != usage.to_dict():
        raise ValueError("benchmark child usage differs from the budget ledger")
    valid_candidates = run.get("valid_candidates")
    if (
        type(valid_candidates) is not int
        or not 0 <= valid_candidates <= plan.max_candidates
        or usage.candidates != valid_candidates
    ):
        raise ValueError("benchmark valid-candidate evidence is inconsistent")
    details = result.get("details")
    unavailable = (
        details.get("usage_unavailable")
        if isinstance(details, Mapping)
        else None
    )
    if (
        not isinstance(details, Mapping)
        or details.get("proposal_attempts") != len(proposals)
        or details.get("evaluation_attempts") != len(evaluations)
        or not isinstance(unavailable, Mapping)
    ):
        raise ValueError("benchmark usage derivation details are inconsistent")

    if any(
        proposal.get("input_tokens") is None
        or proposal.get("output_tokens") is None
        for proposal in proposals
    ):
        if "tokens" not in unavailable:
            raise ValueError("unavailable benchmark token usage was not disclosed")
    else:
        token_total = sum(
            int(proposal.get("input_tokens") or 0)
            + int(proposal.get("output_tokens") or 0)
            for proposal in proposals
        )
        if usage.tokens != token_total or "tokens" in unavailable:
            raise ValueError("benchmark token usage differs from proposer evidence")
    if any(proposal.get("cost_usd") is None for proposal in proposals):
        if "cost_microusd" not in unavailable:
            raise ValueError("unavailable benchmark cost usage was not disclosed")
    else:
        costs = [float(proposal.get("cost_usd") or 0.0) for proposal in proposals]
        if any(not math.isfinite(cost) or cost < 0 for cost in costs):
            raise ValueError("benchmark proposer cost is invalid")
        cost_total = math.ceil(sum(costs) * 1_000_000)
        if usage.cost_microusd != cost_total or "cost_microusd" in unavailable:
            raise ValueError("benchmark cost usage differs from proposer evidence")
    evaluation_durations = [
        _duration_ms(
            evaluation.get("started_at"),
            evaluation.get("finished_at") or evaluation.get("updated_at"),
        )
        for evaluation in evaluations
    ]
    if any(duration is None for duration in evaluation_durations):
        if "gpu_ms" not in unavailable:
            raise ValueError("unavailable benchmark GPU usage was not disclosed")
    else:
        gpu_total = sum(int(duration) for duration in evaluation_durations)
        if usage.gpu_ms != gpu_total or "gpu_ms" in unavailable:
            raise ValueError("benchmark GPU usage differs from evaluation evidence")

    full_reached = len(
        {
            int(evaluation["iteration_id"])
            for evaluation in evaluations
            if evaluation.get("stage") == "full_primary"
        }
    )
    protocol_success = any(
        proposal.get("status") == "SUCCEEDED" for proposal in proposals
    )
    if valid_candidates and not protocol_success:
        raise ValueError("valid benchmark candidates have no successful proposal")
    promoted = child.get("status") == "PROMOTED"
    if promoted != (result.get("promotion") is not None):
        raise ValueError("benchmark promotion evidence is inconsistent")
    if promoted:
        promotion = result["promotion"]
        if not isinstance(promotion, Mapping):
            raise ValueError("benchmark promotion payload is invalid")
        primary_uid = promotion.get("primary_experiment_uid")
        confirmation_uid = promotion.get("confirmation_experiment_uid")
        attempts_by_uid = {
            evaluation.get("experiment_uid"): evaluation
            for evaluation in evaluations
        }
        primary_attempt = attempts_by_uid.get(primary_uid)
        confirmation_attempt = attempts_by_uid.get(confirmation_uid)
        primary_record = history_records.get(str(primary_uid))
        confirmation_record = history_records.get(str(confirmation_uid))
        if (
            not isinstance(primary_attempt, Mapping)
            or not isinstance(confirmation_attempt, Mapping)
            or primary_attempt.get("stage") != "full_primary"
            or confirmation_attempt.get("stage") != "confirmation"
            or primary_attempt.get("candidate_artifact_id")
            != promotion.get("artifact_id")
            or confirmation_attempt.get("candidate_artifact_id")
            != promotion.get("artifact_id")
            or primary_record is None
            or confirmation_record is None
            or primary_record.result.get("promotion", {}).get("phase")
            != "primary"
            or not confirmation_record.promotable
            or not bool(
                confirmation_record.result.get("promotion", {})
                .get("decision", {})
                .get("promoted")
            )
        ):
            raise ValueError("benchmark promotion is not proven by both experiments")
    return BenchmarkObservation(
        arm_id=assignment.arm_id,
        protocol_success=protocol_success,
        valid_candidates=valid_candidates,
        full_reached=full_reached,
        gpu_ms=usage.gpu_ms,
        tokens=usage.tokens,
        cost_microusd=usage.cost_microusd,
        promotions=int(promoted),
    )


def build_trusted_benchmark_report(
    store: CampaignStore,
    controller: ControllerStore,
    history: HistoryStore,
    *,
    campaign_id: str,
) -> dict[str, Any]:
    """Derive a complete report from durable trusted ledgers only."""

    if not isinstance(store, CampaignStore):
        raise TypeError("store must be a CampaignStore")
    if not isinstance(controller, ControllerStore):
        raise TypeError("controller must be a ControllerStore")
    if not isinstance(history, HistoryStore):
        raise TypeError("history must be a HistoryStore")
    quick_check = controller.connection.execute("PRAGMA quick_check").fetchone()
    if quick_check is None or quick_check[0] != "ok":
        raise ValueError("Controller database failed quick_check")
    campaign, plan, children = _campaign_plan(store, campaign_id)
    if campaign.get("status") != CampaignStatus.COMPLETED.value:
        raise ValueError("benchmark report requires a completed Campaign")
    schedule = plan.schedule()
    if len(children) != len(schedule) or any(
        child.get("status") not in TERMINAL_CHILD_STATUSES for child in children
    ):
        raise ValueError("benchmark Campaign did not complete its frozen schedule")
    observations = tuple(
        _derived_observation(
            store,
            controller,
            history,
            campaign=campaign,
            plan=plan,
            assignment=assignment,
            child=children[index],
        )
        for index, assignment in enumerate(schedule)
    )
    report = build_benchmark_report(plan, observations)
    return {
        **report,
        "campaign_id": campaign_id,
        "campaign_snapshot_digest": campaign["snapshot_digest"],
        "evidence_source": "CAMPAIGN_AND_CONTROLLER_LEDGERS",
        "assignment_count": len(observations),
    }


def build_benchmark_report(
    plan: BenchmarkCohortPlan,
    observations: Iterable[BenchmarkObservation],
) -> dict[str, Any]:
    """Aggregate model/Harness behavior without selecting a lucky candidate."""

    if not isinstance(plan, BenchmarkCohortPlan):
        raise TypeError("plan must be a BenchmarkCohortPlan")
    selected = tuple(observations)
    known = {arm.arm_id for arm in plan.arms}
    if any(observation.arm_id not in known for observation in selected):
        raise ValueError("benchmark observation belongs to an unknown arm")
    rows: list[dict[str, Any]] = []
    for arm in plan.arms:
        values = [item for item in selected if item.arm_id == arm.arm_id]
        runs = len(values)
        denominators = max(1, runs)
        valid = sum(item.valid_candidates for item in values)
        rows.append(
            {
                "arm_id": arm.arm_id,
                "proposer_profile": arm.proposer_profile.to_dict(),
                "child_run_count": runs,
                "protocol_success_rate": (
                    sum(item.protocol_success for item in values) / denominators
                ),
                "valid_candidate_count": valid,
                "valid_candidates_per_run": valid / denominators,
                "full_reached_count": sum(item.full_reached for item in values),
                "promotion_count": sum(item.promotions for item in values),
                "promotion_rate_per_valid_candidate": (
                    sum(item.promotions for item in values) / max(1, valid)
                ),
                "gpu_ms": sum(item.gpu_ms for item in values),
                "tokens": sum(item.tokens for item in values),
                "cost_microusd": sum(item.cost_microusd for item in values),
            }
        )
    return {
        "schema_version": 1,
        "command": "benchmark-report",
        "cohort_snapshot_digest": plan.snapshot["snapshot_digest"],
        "cohort_id": plan.cohort_id,
        "namespace_id": plan.namespace.namespace_id,
        "baseline": plan.baseline.to_dict(),
        "history_cutoff": plan.history_cutoff,
        "arm_metrics": rows,
        "ranking_basis": "aggregate_protocol_metrics",
        "single_best_candidate_ranking": False,
    }


def filter_feedback_at_cutoff(
    records: Sequence[Mapping[str, Any]], *, history_cutoff: int
) -> tuple[Mapping[str, Any], ...]:
    """Prevent a cohort from observing History appended after its cutoff."""

    if type(history_cutoff) is not int or history_cutoff < 0:
        raise ValueError("numeric feedback cutoff must be non-negative")
    result: list[Mapping[str, Any]] = []
    for record in records:
        experiment_id = record.get("experiment_id", record.get("id"))
        if type(experiment_id) is not int:
            raise ValueError("feedback record has no integer experiment ID")
        if experiment_id <= history_cutoff:
            result.append(record)
    return tuple(result)


__all__ = [
    "BenchmarkArm",
    "BenchmarkAssignment",
    "BenchmarkCohortPlan",
    "BenchmarkObservation",
    "CURRENT_PROMPT_PROTOCOL_DIGEST",
    "benchmark_controller_run_id",
    "benchmark_plan_from_campaign_snapshot",
    "bind_benchmark_campaign_snapshot",
    "build_benchmark_report",
    "build_trusted_benchmark_report",
    "derive_authoritative_benchmark_plan",
    "execute_benchmark_campaign",
    "filter_feedback_at_cutoff",
    "require_executable_benchmark_plan",
]
