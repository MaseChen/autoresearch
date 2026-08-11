"""Crash-safe orchestration across bounded child-run boundaries.

The supervisor deliberately does not know how a controller evaluates a
candidate.  A trusted caller injects a ``BoundedChildRunner`` whose request
contains the frozen child limits and a resource fencing token.  The only
automatic recovery operation is a lookup by the already-persisted controller
run ID; an in-flight action with no recoverable result is never executed again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Protocol

from ..platform.artifacts import ArtifactId
from ..platform.canonical import canonical_json_text, require_sha256_digest
from ..platform.identity import BaselineRef, ExecutionEnvironmentDigest
from .models import (
    BudgetAmount,
    CampaignMode,
    CampaignStatus,
    ChildRunStatus,
    MAX_CHILD_CANDIDATES,
    MAX_CHILD_CONSECUTIVE_FAILURES,
    MAX_CHILD_WALL_SECONDS,
    ResourceLease,
    TERMINAL_CHILD_STATUSES,
)
from .store import CampaignStore


SUPERVISOR_RESULT_SCHEMA_VERSION = 1


def _token(value: object, field_name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{field_name} must be a non-empty bounded string")
    if any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{field_name} contains control characters")
    return value


def _json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        # A canonical round-trip validates the JSON data model and severs all
        # references to caller-owned mutable containers.
        return json.loads(canonical_json_text(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain canonical JSON data") from exc


def child_budget_idempotency_key(controller_run_id: str) -> str:
    """Return the one stable budget-action identity for a child Controller run."""

    selected = _token(controller_run_id, "controller_run_id")
    digest = hashlib.sha256(selected.encode("utf-8")).hexdigest()
    return f"child-run:{digest}"


class RunnerOutcome(str, Enum):
    PROMOTED = ChildRunStatus.PROMOTED.value
    BUDGET_EXHAUSTED = ChildRunStatus.BUDGET_EXHAUSTED.value
    STOPPED = ChildRunStatus.STOPPED.value
    FAILED = ChildRunStatus.FAILED.value
    HARD_FAILED = ChildRunStatus.HARD_FAILED.value
    UNKNOWN_GPU_OUTCOME = ChildRunStatus.UNKNOWN_GPU_OUTCOME.value
    DATA_INTEGRITY = "DATA_INTEGRITY"


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    """Evidence required before a staged baseline CAS is even eligible."""

    artifact_id: str
    namespace_id: str
    parent_baseline_ref: Mapping[str, Any]
    execution_environment: Mapping[str, Any]
    primary_experiment_uid: str
    confirmation_experiment_uid: str
    checkpoint_digest: str
    policy_snapshot: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ArtifactId.parse(self.artifact_id)
        require_sha256_digest(self.namespace_id, field="namespace_id")
        try:
            parent = BaselineRef.from_value(dict(self.parent_baseline_ref))
            environment = ExecutionEnvironmentDigest.from_value(
                dict(self.execution_environment)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "promotion evidence has an invalid parent/environment identity"
            ) from exc
        if parent.namespace_id != self.namespace_id:
            raise ValueError("promotion parent belongs to another namespace")
        object.__setattr__(
            self,
            "parent_baseline_ref",
            _json_mapping(parent.to_dict(), "parent_baseline_ref"),
        )
        object.__setattr__(
            self,
            "execution_environment",
            _json_mapping(
                environment.to_dict(), "execution_environment"
            ),
        )
        _token(self.primary_experiment_uid, "primary_experiment_uid")
        _token(self.confirmation_experiment_uid, "confirmation_experiment_uid")
        if self.primary_experiment_uid == self.confirmation_experiment_uid:
            raise ValueError("primary and confirmation experiments must differ")
        require_sha256_digest(self.checkpoint_digest, field="checkpoint_digest")
        object.__setattr__(
            self,
            "policy_snapshot",
            _json_mapping(self.policy_snapshot, "policy_snapshot"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "namespace_id": self.namespace_id,
            "parent_baseline_ref": dict(self.parent_baseline_ref),
            "execution_environment": dict(self.execution_environment),
            "primary_experiment_uid": self.primary_experiment_uid,
            "confirmation_experiment_uid": self.confirmation_experiment_uid,
            "checkpoint_digest": self.checkpoint_digest,
            "policy_snapshot": dict(self.policy_snapshot),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PromotionEvidence":
        expected = {
            "artifact_id",
            "namespace_id",
            "parent_baseline_ref",
            "execution_environment",
            "primary_experiment_uid",
            "confirmation_experiment_uid",
            "checkpoint_digest",
            "policy_snapshot",
        }
        if set(value) != expected:
            raise ValueError("promotion evidence fields do not match schema")
        return cls(
            artifact_id=value["artifact_id"],
            namespace_id=value["namespace_id"],
            parent_baseline_ref=value["parent_baseline_ref"],
            execution_environment=value["execution_environment"],
            primary_experiment_uid=value["primary_experiment_uid"],
            confirmation_experiment_uid=value["confirmation_experiment_uid"],
            checkpoint_digest=value["checkpoint_digest"],
            policy_snapshot=value["policy_snapshot"],
        )


@dataclass(frozen=True, slots=True)
class ChildRunPlan:
    """One durable intent for one locally bounded controller run."""

    controller_run_id: str
    proposer_profile: Mapping[str, Any]
    reservation: BudgetAmount
    max_candidates: int = MAX_CHILD_CANDIDATES
    max_wall_seconds: int = MAX_CHILD_WALL_SECONDS
    max_consecutive_failures: int = MAX_CHILD_CONSECUTIVE_FAILURES

    def __post_init__(self) -> None:
        _token(self.controller_run_id, "controller_run_id")
        object.__setattr__(
            self,
            "proposer_profile",
            _json_mapping(self.proposer_profile, "proposer_profile"),
        )
        if not isinstance(self.reservation, BudgetAmount):
            raise TypeError("reservation must be a BudgetAmount")
        if not 1 <= self.max_candidates <= MAX_CHILD_CANDIDATES:
            raise ValueError("child run candidate limit must be between 1 and 5")
        if not 1 <= self.max_wall_seconds <= MAX_CHILD_WALL_SECONDS:
            raise ValueError("child run wall limit must be at most six hours")
        if not 1 <= self.max_consecutive_failures <= MAX_CHILD_CONSECUTIVE_FAILURES:
            raise ValueError("child run failure limit must be between 1 and 3")
        if self.reservation.candidates < self.max_candidates:
            raise ValueError("reservation must cover the child candidate limit")
        if self.reservation.wall_ms < self.max_wall_seconds * 1000:
            raise ValueError("reservation must cover the child wall-time limit")
        if self.reservation.gpu_ms <= 0:
            raise ValueError("reservation must include a positive GPU ceiling")
        if self.reservation.tokens <= 0:
            raise ValueError("reservation must include a positive token ceiling")
        if self.reservation.cost_microusd <= 0:
            raise ValueError("reservation must include a positive cost ceiling")

    @property
    def budget_idempotency_key(self) -> str:
        return child_budget_idempotency_key(self.controller_run_id)


@dataclass(frozen=True, slots=True)
class BoundedRunRequest:
    """Complete frozen boundary passed to the injected local runner."""

    campaign_id: str
    namespace_id: str
    mode: str
    campaign_snapshot_digest: str
    campaign_snapshot: Mapping[str, Any]
    child_id: int
    child_index: int
    controller_run_id: str
    baseline_revision_id: str
    baseline_artifact_id: str
    proposer_profile: Mapping[str, Any]
    budget_limit: BudgetAmount
    max_candidates: int
    max_wall_seconds: int
    max_consecutive_failures: int
    stop_after_promotion: bool
    resource_id: str
    fencing_epoch: int
    lease_expires_epoch: float
    recovery: bool
    # Added in the V2 trusted Controller bridge.  The default preserves
    # construction compatibility for injected/fake runners; the production
    # ResearchControllerRunner rejects absence and the Supervisor always sets
    # the frozen value.
    child_deadline_epoch: float | None = None
    # Exact scientific parent.  Scalar mirrors remain for durable Campaign
    # row lookup and V1 fake-runner compatibility only.
    baseline_ref: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BoundedRunResult:
    """Known terminal output of one bounded runner action."""

    outcome: RunnerOutcome | str
    usage: BudgetAmount
    details: Mapping[str, Any] = field(default_factory=dict)
    promotion: PromotionEvidence | None = None

    def __post_init__(self) -> None:
        selected = RunnerOutcome(self.outcome)
        object.__setattr__(self, "outcome", selected)
        if not isinstance(self.usage, BudgetAmount):
            raise TypeError("usage must be a BudgetAmount")
        object.__setattr__(self, "details", _json_mapping(self.details, "details"))
        if selected is RunnerOutcome.PROMOTED and self.promotion is None:
            raise ValueError("a promoted result requires promotion evidence")
        if selected is not RunnerOutcome.PROMOTED and self.promotion is not None:
            raise ValueError("only a promoted result may carry promotion evidence")
        if self.promotion is not None and not isinstance(
            self.promotion, PromotionEvidence
        ):
            raise TypeError("promotion must be PromotionEvidence")


class BoundedChildRunner(Protocol):
    """Injected adapter around the existing bounded RunEngine/Controller."""

    def run_child(self, request: BoundedRunRequest) -> BoundedRunResult:
        """Start the not-yet-executed child exactly once."""

    def recover_child(
        self, request: BoundedRunRequest
    ) -> BoundedRunResult | None:
        """Look up an existing child; return ``None`` when outcome is unknown."""

    def prove_staged_promotion(
        self,
        *,
        campaign_id: str,
        controller_run_id: str,
        evidence: PromotionEvidence,
    ) -> PromotionEvidence:
        """Re-read immutable scientific evidence before lineage advancement."""


class BoundedRunnerFailure(RuntimeError):
    """Base class for an explicitly classified runner failure."""

    def __init__(
        self,
        message: str,
        *,
        usage: BudgetAmount | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.details = dict(details or {})


class UnknownRunnerOutcome(BoundedRunnerFailure):
    """The external action may have run, but no trustworthy result exists."""


class HardRunnerFailure(BoundedRunnerFailure):
    """A classified hard device failure requiring resource quarantine."""


class DataIntegrityRunnerFailure(BoundedRunnerFailure):
    """An artifact, database, or evidence-integrity failure."""


class CampaignSupervisor:
    """Execute and reconcile one bounded child at a time on one local resource."""

    def __init__(
        self,
        store: CampaignStore,
        runner: BoundedChildRunner,
        *,
        resource_id: str,
        lease_ttl_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(store, CampaignStore):
            raise TypeError("store must be a CampaignStore")
        _token(resource_id, "resource_id")
        if (
            isinstance(lease_ttl_seconds, bool)
            or not isinstance(lease_ttl_seconds, (int, float))
            or not math.isfinite(lease_ttl_seconds)
            or lease_ttl_seconds <= 0
        ):
            raise ValueError("lease_ttl_seconds must be finite and positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.store = store
        self.runner = runner
        self.resource_id = resource_id
        self.lease_ttl_seconds = float(lease_ttl_seconds)
        self.clock = clock

    def execute_child(
        self, campaign_id: str, plan: ChildRunPlan
    ) -> dict[str, Any]:
        """Execute or safely reconcile one child-run boundary.

        A retry of a RUNNING child invokes only ``recover_child``.  If that
        lookup cannot prove a terminal result, the action is charged at its
        worst-case reservation, marked ``UNKNOWN_GPU_OUTCOME``, and the
        resource is quarantined.
        """

        _token(campaign_id, "campaign_id")
        if not isinstance(plan, ChildRunPlan):
            raise TypeError("plan must be a ChildRunPlan")
        if self.lease_ttl_seconds < plan.max_wall_seconds:
            raise ValueError("resource lease TTL must cover the child wall limit")

        campaign = self.store.get_campaign(campaign_id)
        child = self.store.get_child_run_by_controller_run_id(
            plan.controller_run_id
        )
        if child is not None:
            self._validate_child_intent(campaign_id, child, plan)
        elif campaign["status"] != CampaignStatus.RUNNING.value:
            raise ValueError("a new child requires a running campaign")
        else:
            active_children = [
                item
                for item in self.store.list_child_runs(campaign_id)
                if item["status"]
                in {ChildRunStatus.PENDING.value, ChildRunStatus.RUNNING.value}
            ]
            if active_children:
                raise ValueError(
                    "campaign has another active child; recover it before starting one"
                )

        try:
            self.store.reserve_budget(
                campaign_id,
                idempotency_key=plan.budget_idempotency_key,
                action_kind="CHILD_RUN",
                amount=plan.reservation,
            )
        except ValueError as exc:
            if child is None and "exceeds a frozen limit" in str(exc):
                paused = self.store.pause_campaign(
                    campaign_id,
                    status=CampaignStatus.PAUSED_BUDGET.value,
                    reason="remaining campaign budget cannot fund another bounded child",
                )
                return self._report(
                    disposition=CampaignStatus.PAUSED_BUDGET.value,
                    campaign=paused,
                    child=None,
                    budget_action=None,
                    recovered=False,
                    runner_invoked=False,
                )
            raise

        if child is not None and child["status"] in TERMINAL_CHILD_STATUSES:
            try:
                return self._reconcile_terminal(campaign_id, child, plan)
            except Exception:
                self._pause_data_integrity(
                    campaign_id,
                    "terminal child reconciliation found inconsistent durable state",
                )
                raise

        newly_created = child is None
        if child is None:
            try:
                child = self.store.create_child_run(
                    campaign_id,
                    controller_run_id=plan.controller_run_id,
                    proposer_profile=plan.proposer_profile,
                    max_candidates=plan.max_candidates,
                    max_wall_seconds=plan.max_wall_seconds,
                    max_consecutive_failures=plan.max_consecutive_failures,
                    stop_after_promotion=True,
                )
            except BaseException:
                # No external action exists yet.  A caught creation failure can
                # therefore release its reservation; a process kill instead
                # leaves the RESERVED intent reusable by the next invocation.
                self.store.cancel_budget(
                    campaign_id,
                    idempotency_key=plan.budget_idempotency_key,
                )
                raise

        lease = self.store.get_active_resource_lease(self.resource_id)
        now = float(self.clock())
        if not math.isfinite(now):
            raise ValueError("clock returned a non-finite timestamp")

        if child["status"] == ChildRunStatus.RUNNING.value:
            if (
                lease is None
                or lease.campaign_id != campaign_id
                or lease.expires_epoch <= now
            ):
                unknown = BoundedRunResult(
                    RunnerOutcome.UNKNOWN_GPU_OUTCOME,
                    plan.reservation,
                    {"reason": "running child has no valid owned fencing lease"},
                )
                return self._finalize(
                    campaign_id,
                    child,
                    plan,
                    unknown,
                    lease=lease if lease and lease.campaign_id == campaign_id else None,
                    recovered=True,
                    runner_invoked=False,
                )
            try:
                request = self._request(campaign, child, plan, lease, recovery=True)
                recovered_result = self.runner.recover_child(request)
            except BaseException as exc:
                return self._handle_runner_exception(
                    exc,
                    campaign_id=campaign_id,
                    child=child,
                    plan=plan,
                    lease=lease,
                    recovered=True,
                    runner_invoked=False,
                )
            if recovered_result is None:
                recovered_result = BoundedRunResult(
                    RunnerOutcome.UNKNOWN_GPU_OUTCOME,
                    plan.reservation,
                    {"reason": "runner could not prove the existing child outcome"},
                )
            return self._finalize(
                campaign_id,
                child,
                plan,
                recovered_result,
                lease=lease,
                recovered=True,
                runner_invoked=False,
            )

        if child["status"] != ChildRunStatus.PENDING.value:
            raise ValueError(f"unsupported child state {child['status']}")
        if campaign["status"] != CampaignStatus.RUNNING.value:
            raise ValueError("a pending child cannot start while campaign is paused")

        if lease is not None and newly_created:
            return self._fail_data_integrity_before_run(
                campaign_id,
                child,
                plan,
                lease,
                "resource has an orphan or foreign lease before child acquisition",
            )
        if lease is not None and lease.campaign_id != campaign_id:
            return self._fail_data_integrity_before_run(
                campaign_id,
                child,
                plan,
                None,
                "resource lease is owned by another campaign",
            )
        if lease is None or lease.expires_epoch <= now:
            try:
                lease = self.store.acquire_resource(
                    campaign_id,
                    resource_id=self.resource_id,
                    ttl_seconds=self.lease_ttl_seconds,
                    now_epoch=now,
                )
            except ValueError as exc:
                if "resource is quarantined" in str(exc):
                    return self._fail_hard_before_run(
                        campaign_id,
                        child,
                        plan,
                        "resource remains quarantined from an earlier hard outcome",
                    )
                raced_lease = self.store.get_active_resource_lease(self.resource_id)
                if raced_lease is not None:
                    return self._fail_data_integrity_before_run(
                        campaign_id,
                        child,
                        plan,
                        (
                            raced_lease
                            if raced_lease.campaign_id == campaign_id
                            else None
                        ),
                        "resource fencing changed during child acquisition",
                    )
                raise

        child = self.store.start_child_run(
            child["id"], controller_run_id=plan.controller_run_id
        )
        try:
            request = self._request(campaign, child, plan, lease, recovery=False)
            result = self.runner.run_child(request)
        except BaseException as exc:
            return self._handle_runner_exception(
                exc,
                campaign_id=campaign_id,
                child=child,
                plan=plan,
                lease=lease,
                recovered=False,
                runner_invoked=True,
            )
        return self._finalize(
            campaign_id,
            child,
            plan,
            result,
            lease=lease,
            recovered=False,
            runner_invoked=True,
        )

    def advance_staged_lineage(
        self,
        campaign_id: str,
        *,
        child_id: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """CAS a confirmed promotion after its child has paused the campaign."""

        selected_idempotency_key = _token(idempotency_key, "idempotency_key")
        campaign = self.store.get_campaign(campaign_id)
        if campaign["mode"] == CampaignMode.BENCHMARK.value:
            raise ValueError("benchmark campaigns cannot advance staged lineage")
        child = self.store.get_child_run(child_id)
        if child["campaign_id"] != campaign_id:
            raise ValueError("child does not belong to campaign")
        if child["status"] != ChildRunStatus.PROMOTED.value:
            raise ValueError("only a promoted child can advance staged lineage")
        result = self._parse_supervisor_result(child)
        promotion_value = result.get("promotion")
        if not isinstance(promotion_value, Mapping):
            raise ValueError("promoted child is missing immutable promotion evidence")
        promotion = PromotionEvidence.from_mapping(promotion_value)
        baseline = self._baseline(campaign_id, child["baseline_revision_id"])
        baseline_value = baseline.get("baseline_ref")
        if not isinstance(baseline_value, Mapping):
            raise ValueError(
                "Campaign baseline revision has no environment-bound reference"
            )
        try:
            parent_ref = BaselineRef.from_value(dict(baseline_value))
        except (TypeError, ValueError) as exc:
            raise ValueError("Campaign parent BaselineRef is invalid") from exc
        if promotion.parent_baseline_ref != parent_ref.to_dict():
            raise ValueError(
                "promotion evidence does not name the exact Campaign parent"
            )
        prover = getattr(self.runner, "prove_staged_promotion", None)
        if not callable(prover):
            raise ValueError(
                "runner cannot prove promotion identities from trusted History"
            )
        proven = prover(
            campaign_id=campaign_id,
            controller_run_id=str(child["controller_run_id"]),
            evidence=promotion,
        )
        if not isinstance(proven, PromotionEvidence) or proven.to_dict() != promotion.to_dict():
            raise ValueError("trusted promotion proof differs from child evidence")
        policy_snapshot = {
            "schema_version": SUPERVISOR_RESULT_SCHEMA_VERSION,
            "child_run_id": child_id,
            "controller_run_id": child["controller_run_id"],
            "checkpoint_digest": promotion.checkpoint_digest,
            "promotion_policy": dict(promotion.policy_snapshot),
        }
        existing = next(
            (
                revision
                for revision in self.store.list_baseline_revisions(campaign_id)
                if revision.get("idempotency_key") == selected_idempotency_key
            ),
            None,
        )
        if (
            existing is None
            and campaign["status"] != CampaignStatus.PAUSED_OPERATOR.value
        ):
            raise ValueError("lineage advancement requires a promotion pause")
        return self.store.advance_baseline(
            campaign_id,
            expected_parent_revision_id=child["baseline_revision_id"],
            parent_baseline_ref=promotion.parent_baseline_ref,
            artifact_id=promotion.artifact_id,
            primary_experiment_uid=promotion.primary_experiment_uid,
            confirmation_experiment_uid=promotion.confirmation_experiment_uid,
            evidence_namespace_id=promotion.namespace_id,
            evidence_execution_environment=promotion.execution_environment,
            policy_snapshot=policy_snapshot,
            idempotency_key=selected_idempotency_key,
        )

    def _validate_child_intent(
        self,
        campaign_id: str,
        child: Mapping[str, Any],
        plan: ChildRunPlan,
    ) -> None:
        expected = {
            "campaign_id": campaign_id,
            "controller_run_id": plan.controller_run_id,
            "proposer_profile": dict(plan.proposer_profile),
            "max_candidates": plan.max_candidates,
            "max_wall_seconds": plan.max_wall_seconds,
            "max_consecutive_failures": plan.max_consecutive_failures,
            "stop_after_promotion": True,
        }
        if any(child.get(key) != value for key, value in expected.items()):
            raise ValueError("controller_run_id belongs to a different child intent")

    def _baseline(self, campaign_id: str, revision_id: str) -> dict[str, Any]:
        for revision in self.store.list_baseline_revisions(campaign_id):
            if revision["id"] == revision_id:
                return revision
        raise DataIntegrityRunnerFailure(
            "child baseline revision is missing from the campaign lineage"
        )

    def _request(
        self,
        campaign: Mapping[str, Any],
        child: Mapping[str, Any],
        plan: ChildRunPlan,
        lease: ResourceLease,
        *,
        recovery: bool,
    ) -> BoundedRunRequest:
        baseline = self._baseline(campaign["id"], child["baseline_revision_id"])
        return BoundedRunRequest(
            campaign_id=campaign["id"],
            namespace_id=campaign["namespace_id"],
            mode=campaign["mode"],
            campaign_snapshot_digest=campaign["snapshot_digest"],
            campaign_snapshot=_json_mapping(
                campaign["snapshot"], "campaign snapshot"
            ),
            child_id=int(child["id"]),
            child_index=int(child["child_index"]),
            controller_run_id=plan.controller_run_id,
            baseline_revision_id=baseline["id"],
            baseline_artifact_id=baseline["artifact_id"],
            proposer_profile=dict(plan.proposer_profile),
            budget_limit=plan.reservation,
            max_candidates=plan.max_candidates,
            max_wall_seconds=plan.max_wall_seconds,
            max_consecutive_failures=plan.max_consecutive_failures,
            stop_after_promotion=True,
            resource_id=lease.resource_id,
            fencing_epoch=lease.fencing_epoch,
            lease_expires_epoch=lease.expires_epoch,
            # The resource acquisition epoch is recoverable from its frozen
            # expiry and this Supervisor's immutable TTL.  Basing the child
            # deadline on that epoch keeps it stable across process recovery
            # and includes Controller preflight time in the child wall limit.
            child_deadline_epoch=(
                lease.expires_epoch
                - self.lease_ttl_seconds
                + plan.max_wall_seconds
            ),
            baseline_ref=(
                None
                if baseline.get("baseline_ref") is None
                else _json_mapping(
                    baseline["baseline_ref"], "Campaign baseline reference"
                )
            ),
            recovery=recovery,
        )

    def _handle_runner_exception(
        self,
        exc: BaseException,
        *,
        campaign_id: str,
        child: Mapping[str, Any],
        plan: ChildRunPlan,
        lease: ResourceLease,
        recovered: bool,
        runner_invoked: bool,
    ) -> dict[str, Any]:
        details = {
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }
        if isinstance(exc, BoundedRunnerFailure):
            details.update(_json_mapping(exc.details, "failure details"))
            if isinstance(exc, HardRunnerFailure):
                outcome = RunnerOutcome.HARD_FAILED
                usage = exc.usage or plan.reservation
            elif isinstance(exc, DataIntegrityRunnerFailure):
                outcome = RunnerOutcome.DATA_INTEGRITY
                usage = plan.reservation
            else:
                outcome = RunnerOutcome.UNKNOWN_GPU_OUTCOME
                usage = plan.reservation
            result = BoundedRunResult(outcome, usage, details)
            return self._finalize(
                campaign_id,
                child,
                plan,
                result,
                lease=lease,
                recovered=recovered,
                runner_invoked=runner_invoked,
            )

        # An unclassified exception after RUNNING was persisted leaves the
        # external action unknowable.  Durably quarantine first, then preserve
        # the caller's exception semantics.
        result = BoundedRunResult(
            RunnerOutcome.UNKNOWN_GPU_OUTCOME,
            plan.reservation,
            details,
        )
        self._finalize(
            campaign_id,
            child,
            plan,
            result,
            lease=lease,
            recovered=recovered,
            runner_invoked=runner_invoked,
        )
        raise exc

    def _fail_data_integrity_before_run(
        self,
        campaign_id: str,
        child: Mapping[str, Any],
        plan: ChildRunPlan,
        lease: ResourceLease | None,
        reason: str,
    ) -> dict[str, Any]:
        child = self.store.start_child_run(
            child["id"], controller_run_id=plan.controller_run_id
        )
        result = BoundedRunResult(
            RunnerOutcome.DATA_INTEGRITY,
            BudgetAmount(),
            {"reason": reason},
        )
        return self._finalize(
            campaign_id,
            child,
            plan,
            result,
            lease=lease,
            recovered=False,
            runner_invoked=False,
        )

    def _fail_hard_before_run(
        self,
        campaign_id: str,
        child: Mapping[str, Any],
        plan: ChildRunPlan,
        reason: str,
    ) -> dict[str, Any]:
        child = self.store.start_child_run(
            child["id"], controller_run_id=plan.controller_run_id
        )
        return self._finalize(
            campaign_id,
            child,
            plan,
            BoundedRunResult(
                RunnerOutcome.HARD_FAILED,
                BudgetAmount(),
                {"reason": reason},
            ),
            lease=None,
            recovered=False,
            runner_invoked=False,
        )

    def _finalize(
        self,
        campaign_id: str,
        child: Mapping[str, Any],
        plan: ChildRunPlan,
        result: BoundedRunResult,
        *,
        lease: ResourceLease | None,
        recovered: bool,
        runner_invoked: bool,
    ) -> dict[str, Any]:
        if not isinstance(result, BoundedRunResult):
            result = BoundedRunResult(
                RunnerOutcome.DATA_INTEGRITY,
                plan.reservation,
                {"reason": "runner returned an object outside its result contract"},
            )
        campaign = self.store.get_campaign(campaign_id)
        if (
            result.promotion is not None
            and result.promotion.namespace_id != campaign["namespace_id"]
        ):
            result = BoundedRunResult(
                RunnerOutcome.DATA_INTEGRITY,
                plan.reservation,
                {"reason": "promotion evidence belongs to another namespace"},
            )
        if not result.usage.fits_within(plan.reservation):
            result = BoundedRunResult(
                RunnerOutcome.DATA_INTEGRITY,
                plan.reservation,
                {"reason": "runner usage exceeds its worst-case reservation"},
            )

        usage = (
            plan.reservation
            if result.outcome
            in {
                RunnerOutcome.UNKNOWN_GPU_OUTCOME,
                RunnerOutcome.DATA_INTEGRITY,
            }
            else result.usage
        )
        child_status = (
            ChildRunStatus.FAILED.value
            if result.outcome is RunnerOutcome.DATA_INTEGRITY
            else result.outcome.value
        )
        stored_result = {
            "schema_version": SUPERVISOR_RESULT_SCHEMA_VERSION,
            "controller_run_id": plan.controller_run_id,
            "budget_idempotency_key": plan.budget_idempotency_key,
            "outcome": result.outcome.value,
            "usage": usage.to_dict(),
            "details": dict(result.details),
            "resource_lease": (
                {
                    "resource_id": lease.resource_id,
                    "fencing_epoch": lease.fencing_epoch,
                }
                if lease is not None
                else None
            ),
            "promotion": (
                result.promotion.to_dict() if result.promotion is not None else None
            ),
        }
        try:
            finished = self.store.finish_child_run(
                child["id"], status=child_status, result=stored_result
            )
            budget_action = self.store.settle_budget(
                campaign_id,
                idempotency_key=plan.budget_idempotency_key,
                actual=usage,
            )
            self._release_for_outcome(campaign_id, lease, result.outcome)
        except Exception:
            self._pause_data_integrity(
                campaign_id,
                "child finalization could not reconcile result, budget, and lease",
            )
            raise
        if result.outcome is RunnerOutcome.DATA_INTEGRITY:
            self._pause_data_integrity(
                campaign_id, "child run reported a data-integrity failure"
            )
        return self._report(
            disposition=result.outcome.value,
            campaign=self.store.get_campaign(campaign_id),
            child=finished,
            budget_action=budget_action,
            recovered=recovered,
            runner_invoked=runner_invoked,
        )

    def _parse_supervisor_result(
        self, child: Mapping[str, Any]
    ) -> dict[str, Any]:
        value = child.get("result")
        if not isinstance(value, Mapping):
            raise ValueError("terminal child result is not a supervisor result")
        expected = {
            "schema_version",
            "controller_run_id",
            "budget_idempotency_key",
            "outcome",
            "usage",
            "details",
            "resource_lease",
            "promotion",
        }
        if set(value) != expected or value.get("schema_version") != 1:
            raise ValueError("terminal child result schema is invalid")
        if value.get("controller_run_id") != child.get("controller_run_id"):
            raise ValueError("terminal child result controller identity is invalid")
        return dict(value)

    def _reconcile_terminal(
        self,
        campaign_id: str,
        child: Mapping[str, Any],
        plan: ChildRunPlan,
    ) -> dict[str, Any]:
        result = self._parse_supervisor_result(child)
        if result["budget_idempotency_key"] != plan.budget_idempotency_key:
            raise ValueError("terminal child budget identity is invalid")
        usage = BudgetAmount.from_mapping(result["usage"])
        budget_action = self.store.settle_budget(
            campaign_id,
            idempotency_key=plan.budget_idempotency_key,
            actual=usage,
        )
        selected = RunnerOutcome(result["outcome"])
        lease_value = result["resource_lease"]
        active = self.store.get_active_resource_lease(self.resource_id)
        if (
            isinstance(lease_value, Mapping)
            and active is not None
            and lease_value.get("resource_id") == active.resource_id
            and lease_value.get("fencing_epoch") == active.fencing_epoch
            and active.campaign_id == campaign_id
        ):
            self._release_for_outcome(campaign_id, active, selected)
        if selected is RunnerOutcome.DATA_INTEGRITY:
            self._pause_data_integrity(
                campaign_id, "child run reported a data-integrity failure"
            )
        return self._report(
            disposition=selected.value,
            campaign=self.store.get_campaign(campaign_id),
            child=dict(child),
            budget_action=budget_action,
            recovered=True,
            runner_invoked=False,
        )

    def _pause_data_integrity(self, campaign_id: str, reason: str) -> None:
        self.store.pause_campaign(
            campaign_id,
            status=CampaignStatus.PAUSED_DATA_INTEGRITY.value,
            reason=reason,
        )

    def _release_for_outcome(
        self,
        campaign_id: str,
        lease: ResourceLease | None,
        outcome: RunnerOutcome,
    ) -> None:
        if lease is None or lease.campaign_id != campaign_id:
            return
        quarantine = outcome in {
            RunnerOutcome.HARD_FAILED,
            RunnerOutcome.UNKNOWN_GPU_OUTCOME,
            RunnerOutcome.DATA_INTEGRITY,
        }
        self.store.release_resource(
            lease,
            quarantine=quarantine,
            reason=(
                f"supervisor quarantined resource after {outcome.value}"
                if quarantine
                else ""
            ),
        )

    @staticmethod
    def _report(
        *,
        disposition: str,
        campaign: Mapping[str, Any],
        child: Mapping[str, Any] | None,
        budget_action: Mapping[str, Any] | None,
        recovered: bool,
        runner_invoked: bool,
    ) -> dict[str, Any]:
        return {
            "schema_version": SUPERVISOR_RESULT_SCHEMA_VERSION,
            "disposition": disposition,
            "campaign": dict(campaign),
            "child": dict(child) if child is not None else None,
            "budget_action": (
                dict(budget_action) if budget_action is not None else None
            ),
            "recovered": recovered,
            "runner_invoked": runner_invoked,
        }


__all__ = [
    "BoundedChildRunner",
    "BoundedRunRequest",
    "BoundedRunResult",
    "BoundedRunnerFailure",
    "CampaignSupervisor",
    "ChildRunPlan",
    "DataIntegrityRunnerFailure",
    "HardRunnerFailure",
    "PromotionEvidence",
    "RunnerOutcome",
    "SUPERVISOR_RESULT_SCHEMA_VERSION",
    "UnknownRunnerOutcome",
    "child_budget_idempotency_key",
]
