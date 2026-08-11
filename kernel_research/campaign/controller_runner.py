"""Trusted Campaign adapter for the current local ResearchController.

The generic :mod:`kernel_research.campaign.supervisor` owns orchestration and
budget reservations.  This module is the deliberately narrow bridge to the
already-bounded legacy Triton/C500/OpenCode controller.  It resolves only
reviewed built-in profiles, freezes a Campaign baseline into the Controller
run snapshot, and never adopts a result into Git or deployment configuration.

Recovery is lookup-first: ``run_child`` refuses every pre-existing Controller
intent, while ``recover_child`` never creates one and delegates only a proven
RUNNING intent to ``ResearchController.resume``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sqlite3
import time
from typing import Any, Callable, Mapping, Protocol

from ..autorun.controller import (
    ControllerDataIntegrityError,
    ResearchController,
    UnknownGPUOutcome,
    _target_binding_snapshot,
    _trusted_target,
)
from ..autorun.model_catalog import resolve_opencode_model
from ..autorun.errors import ControlledRuntimeError
from ..autorun.models import ControllerConfig
from ..autorun.states import RunStatus, TERMINAL_RUN_STATUSES
from ..autorun.store import ControllerStore
from ..history import ExperimentRecord, HistoryStore
from ..constants import MAX_PROPOSER_ATTEMPTS
from ..platform.artifacts import ArtifactId
from ..platform.canonical import canonical_sha256, require_sha256_digest
from ..platform.identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
)
from ..platform.profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    ProfileDefinition,
    ProfileRef,
    ResearchNamespace,
)
from .benchmark import (
    benchmark_controller_run_id,
    benchmark_plan_from_campaign_snapshot,
    require_executable_benchmark_plan,
)
from .models import (
    BudgetAmount,
    CampaignMode,
    CampaignStatus,
    MAX_CHILD_CANDIDATES,
    MAX_CHILD_CONSECUTIVE_FAILURES,
    MAX_CHILD_WALL_SECONDS,
    ResourceLease,
)
from .store import CampaignStore
from .supervisor import (
    BoundedRunRequest,
    BoundedRunResult,
    DataIntegrityRunnerFailure,
    HardRunnerFailure,
    PromotionEvidence,
    RunnerOutcome,
    UnknownRunnerOutcome,
    child_budget_idempotency_key,
)


CONTROLLER_RUNNER_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")

_CAMPAIGN_PROFILE = BUILTIN_PROFILE_REGISTRY.get(
    kind="campaign", profile_id="bounded-discovery", revision="v1"
)
_DEPLOYMENT_PROFILE = BUILTIN_PROFILE_REGISTRY.get(
    kind="deployment", profile_id="legacy-local-c500", revision="v1"
)
_ALLOWED_PROPOSER_IDS = frozenset(
    {
        "opencode-deepseek-v4-pro",
        "opencode-deepseek-v4-flash",
    }
)
_TRUSTED_TARGET_NAMESPACES = {
    LEGACY_RESEARCH_NAMESPACE.namespace_id: LEGACY_RESEARCH_NAMESPACE,
    CURRENT_RESEARCH_NAMESPACE.namespace_id: CURRENT_RESEARCH_NAMESPACE,
}


class ControllerFactory(Protocol):
    def __call__(self, config: ControllerConfig) -> ResearchController:
        ...


class FenceLookup(Protocol):
    def __call__(self, resource_id: str) -> ResourceLease | None:
        ...


def legacy_campaign_snapshot(
    *,
    mode: CampaignMode | str = CampaignMode.DISCOVERY,
    cohort_id: str | None = None,
    history_cutoff: int | None = None,
    prompt_protocol_digest: str | None = None,
    feedback_snapshot_digest: str | None = None,
    namespace: ResearchNamespace = LEGACY_RESEARCH_NAMESPACE,
) -> dict[str, Any]:
    """Build the minimum frozen Campaign snapshot accepted by this adapter.

    The returned object intentionally has no self-digest: ``CampaignStore``
    digests the complete snapshot when the Campaign is created.  Benchmark
    snapshots additionally require the cohort cutoff and prompt/feedback
    provenance; discovery snapshots acquire a fresh History cutoff at each
    child boundary.
    """

    selected_mode = CampaignMode(mode)
    selected_namespace = _trusted_target_namespace(namespace.to_dict())
    result: dict[str, Any] = {
        "schema_version": CONTROLLER_RUNNER_SCHEMA_VERSION,
        "mode": selected_mode.value,
        "campaign_profile": _CAMPAIGN_PROFILE.ref.to_dict(),
        "deployment_profile": _DEPLOYMENT_PROFILE.ref.to_dict(),
        "namespace": selected_namespace.to_dict(),
    }
    if selected_mode is CampaignMode.BENCHMARK:
        if not isinstance(cohort_id, str) or not _TOKEN.fullmatch(cohort_id):
            raise ValueError("benchmark cohort_id must be a stable identifier")
        if type(history_cutoff) is not int or history_cutoff < 0:
            raise ValueError("benchmark history_cutoff must be non-negative")
        result.update(
            {
                "cohort_id": cohort_id,
                "history_cutoff": history_cutoff,
                "prompt_protocol_digest": require_sha256_digest(
                    prompt_protocol_digest, field="prompt_protocol_digest"
                ),
                "feedback_snapshot_digest": require_sha256_digest(
                    feedback_snapshot_digest, field="feedback_snapshot_digest"
                ),
            }
        )
    elif any(
        value is not None
        for value in (
            cohort_id,
            history_cutoff,
            prompt_protocol_digest,
            feedback_snapshot_digest,
        )
    ):
        raise ValueError(
            "discovery Campaign snapshots do not freeze benchmark cohort fields"
        )
    return result


def _trusted_target_namespace(value: object) -> ResearchNamespace:
    try:
        selected = (
            value
            if isinstance(value, ResearchNamespace)
            else ResearchNamespace.from_value(value)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Campaign target namespace is invalid") from exc
    trusted = _TRUSTED_TARGET_NAMESPACES.get(selected.namespace_id)
    if trusted is None or selected != trusted:
        raise ValueError(
            "Campaign target must be the exact built-in LEGACY or CURRENT namespace"
        )
    return trusted


def trusted_resume_doctor(
    config: ControllerConfig,
    *,
    campaign_store: CampaignStore,
    campaign: Mapping[str, Any],
    resource_id: str,
    controller_factory: ControllerFactory = ResearchController,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Run the fixed trusted C500 doctor and return canonical resume evidence."""

    if not isinstance(config, ControllerConfig):
        raise TypeError("config must be ControllerConfig")
    if not isinstance(campaign_store, CampaignStore):
        raise TypeError("campaign_store must be CampaignStore")
    if not isinstance(campaign, Mapping):
        raise TypeError("campaign must be a mapping")
    campaign_id = campaign.get("id")
    if not isinstance(campaign_id, str) or not _TOKEN.fullmatch(campaign_id):
        raise ValueError("campaign has no stable identifier")
    if campaign.get("status") not in {
        CampaignStatus.PAUSED_HARD_FAILURE.value,
        CampaignStatus.PAUSED_UNKNOWN_OUTCOME.value,
    }:
        raise ValueError("trusted GPU doctor requires a hard/unknown paused Campaign")
    snapshot = campaign.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("Campaign has no frozen snapshot")
    namespace = _trusted_target_namespace(snapshot.get("namespace"))
    if campaign.get("namespace_id") != namespace.namespace_id:
        raise ValueError("Campaign namespace differs from its frozen snapshot")
    if not isinstance(resource_id, str) or not _TOKEN.fullmatch(resource_id):
        raise ValueError("resource_id must be a stable identifier")
    if resource_id not in tuple(
        _DEPLOYMENT_PROFILE.config.get("resource_pool", ())
    ):
        raise ValueError("resource_id is outside the trusted deployment profile")
    quarantine = campaign_store.get_latest_quarantined_resource(campaign_id)
    if quarantine is None or quarantine.resource_id != resource_id:
        raise ValueError(
            "resource_id does not match the latest Campaign quarantine"
        )
    controller = controller_factory(config)
    if getattr(controller, "config", None) != config:
        raise ValueError("trusted doctor factory changed ControllerConfig")
    doctor = getattr(controller, "campaign_resume_doctor", None)
    environment_resolver = getattr(
        controller, "_resolved_execution_environment", None
    )
    if not callable(doctor) or not callable(environment_resolver):
        raise ValueError("trusted doctor factory lacks required Controller methods")
    run_id = "campaign-resume-" + hashlib.sha256(
        f"{campaign_id}\x00{resource_id}".encode("utf-8")
    ).hexdigest()[:24]
    result = doctor(
        campaign_database=campaign_store.path,
        campaign_id=campaign_id,
        namespace_id=namespace.namespace_id,
        campaign_snapshot_digest=str(campaign["snapshot_digest"]),
        resource_id=resource_id,
        quarantine_fencing_epoch=quarantine.fencing_epoch,
        run_id=run_id,
    )
    probe = result.get("c500_probe") if isinstance(result, Mapping) else None
    probe_environment = (
        probe.get("environment") if isinstance(probe, Mapping) else None
    )
    if not (
        isinstance(result, Mapping)
        and result.get("status") == "SUCCESS"
        and isinstance(probe_environment, Mapping)
        and probe_environment.get("compile_probe_status") == "PASSED"
    ):
        raise ValueError("trusted C500 resume doctor did not pass")
    execution_environment = environment_resolver(namespace)
    if not isinstance(execution_environment, ExecutionEnvironmentDigest):
        raise ValueError("trusted doctor did not resolve an execution environment")
    observed_epoch = float(clock())
    if not math.isfinite(observed_epoch) or observed_epoch < 0:
        raise ValueError("doctor clock returned an invalid observed epoch")
    return {
        "schema_version": 1,
        "kind": "CAMPAIGN_RESUME_DOCTOR",
        "status": "SUCCESS",
        "campaign_id": campaign_id,
        "resource_id": resource_id,
        "observed_epoch": observed_epoch,
        "namespace_id": namespace.namespace_id,
        "config_digest": canonical_sha256(config.redacted_dict()),
        "execution_environment": execution_environment.to_dict(),
        "doctor_result": json.loads(
            json.dumps(
                dict(result),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
    }


def trusted_child_reservation(
    *,
    proposer_profile: Mapping[str, Any],
    campaign_remaining: BudgetAmount,
    max_candidates: int,
    max_wall_seconds: int,
    max_consecutive_failures: int,
) -> BudgetAmount:
    """Derive all five worst-case axes from trusted profiles and campaign state.

    Cost has no locally provable provider-side ceiling, so a child reserves the
    entire remaining Campaign cost allowance.  A known terminal result settles
    the measured amount and releases the unused reservation; an unknown result
    is charged at that frozen worst case.
    """

    if not isinstance(campaign_remaining, BudgetAmount):
        raise TypeError("campaign_remaining must be a BudgetAmount")
    for name, value, maximum in (
        ("max_candidates", max_candidates, MAX_CHILD_CANDIDATES),
        ("max_wall_seconds", max_wall_seconds, MAX_CHILD_WALL_SECONDS),
        (
            "max_consecutive_failures",
            max_consecutive_failures,
            MAX_CHILD_CONSECUTIVE_FAILURES,
        ),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} is outside the bounded child limit")
    try:
        reference = ProfileRef.from_value(proposer_profile)
        definition = BUILTIN_PROFILE_REGISTRY.resolve(reference)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "proposer_profile is not an exact trusted built-in profile"
        ) from exc
    if (
        definition.ref.kind != "proposer"
        or definition.ref.id not in _ALLOWED_PROPOSER_IDS
        or definition.config.get("harness") != "opencode"
    ):
        raise ValueError(
            "only the active trusted OpenCode proposer profiles may run"
        )
    model = resolve_opencode_model(definition.config.get("model"))
    # A failed proposal can consume a Controller failure slot without becoming
    # a valid candidate, and each proposal has at most two transport attempts.
    proposer_actions = max_candidates + max_consecutive_failures
    token_ceiling = (
        proposer_actions * MAX_PROPOSER_ATTEMPTS * model.context_tokens
    )
    wall_ms = max_wall_seconds * 1000
    derived = BudgetAmount(
        candidates=max_candidates,
        wall_ms=wall_ms,
        gpu_ms=wall_ms,
        tokens=token_ceiling,
        cost_microusd=campaign_remaining.cost_microusd,
    )
    missing = [
        name
        for name, value in derived.to_dict().items()
        if value <= 0
    ]
    if missing:
        raise ValueError(
            "campaign has no positive worst-case allowance for: "
            + ", ".join(missing)
        )
    if not derived.fits_within(campaign_remaining):
        raise ValueError(
            "remaining Campaign budget cannot fund the trusted child worst case"
        )
    return derived


class ResearchControllerRunner:
    """Concrete, fail-closed ``BoundedChildRunner`` for one local C500.

    ``fence_lookup`` should normally be
    ``CampaignStore.get_active_resource_lease`` from the same store used by
    the supervisor.  Making it explicit keeps the Controller unable to infer
    or mint a resource lease.
    """

    def __init__(
        self,
        config: ControllerConfig,
        *,
        campaign_store: CampaignStore,
        fence_lookup: FenceLookup,
        controller_factory: ControllerFactory = ResearchController,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, ControllerConfig):
            raise TypeError("config must be ControllerConfig")
        if not isinstance(campaign_store, CampaignStore):
            raise TypeError("campaign_store must be CampaignStore")
        if not callable(fence_lookup):
            raise TypeError("fence_lookup must be callable")
        if not callable(controller_factory):
            raise TypeError("controller_factory must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not config.stop_after_promotion:
            raise ValueError("Campaign child controllers must stop after promotion")
        self.config = config
        self.campaign_store = campaign_store
        self.fence_lookup = fence_lookup
        self.controller_factory = controller_factory
        self.clock = clock

    @property
    def controller_db(self) -> Path:
        return self.config.controller_dir / "controller.sqlite3"

    def run_child(self, request: BoundedRunRequest) -> BoundedRunResult:
        """Create and execute exactly one previously absent Controller run."""

        context = self._validate_request(request, recovery=False)
        if self._load_run(request.controller_run_id) is not None:
            raise DataIntegrityRunnerFailure(
                "run_child refuses to take over an existing Controller intent",
                details={"controller_run_id": request.controller_run_id},
            )
        self._require_fence(request, recovery=False)
        bounded_config = self._bounded_config(request, context["proposer"])
        history_cutoff = self._history_cutoff()
        if context["mode"] is CampaignMode.BENCHMARK:
            history_cutoff = int(context["snapshot"]["history_cutoff"])
        baseline_ref = self._campaign_baseline(request)
        snapshot = self._workflow_snapshot(
            request,
            bounded_config=bounded_config,
            proposer=context["proposer"],
            baseline_ref=baseline_ref,
            history_cutoff=history_cutoff,
        )
        controller = self._controller(bounded_config)
        try:
            run = controller.start_from_snapshot(
                run_id=request.controller_run_id,
                workflow_snapshot=snapshot,
                baseline_ref=baseline_ref,
                history_cutoff=history_cutoff,
            )
        except (ControllerDataIntegrityError, sqlite3.DatabaseError) as exc:
            raise DataIntegrityRunnerFailure(
                str(exc), details={"phase": "controller-start"}
            ) from exc
        except UnknownGPUOutcome as exc:
            raise UnknownRunnerOutcome(
                str(exc), details={"phase": "controller-start"}
            ) from exc
        except ControlledRuntimeError as exc:
            persisted = self._load_run(request.controller_run_id)
            if persisted is not None and persisted["status"] in TERMINAL_RUN_STATUSES:
                return self._terminal_result(controller, request, persisted)
            if persisted is not None:
                raise UnknownRunnerOutcome(
                    str(exc), details={"phase": "controller-start"}
                ) from exc
            raise HardRunnerFailure(
                str(exc),
                usage=BudgetAmount(),
                details={"phase": "bounded-preflight"},
            ) from exc
        except KeyboardInterrupt as exc:
            persisted = self._load_run(request.controller_run_id)
            if persisted is None:
                raise
            raise UnknownRunnerOutcome(
                "Controller was interrupted after persisting its run intent",
                details={"phase": "controller-start"},
            ) from exc
        except Exception as exc:
            persisted = self._load_run(request.controller_run_id)
            if persisted is None:
                raise DataIntegrityRunnerFailure(
                    f"Controller failed before persisting its run intent: {exc}",
                    details={"phase": "controller-start"},
                ) from exc
            if persisted["status"] in TERMINAL_RUN_STATUSES:
                return self._terminal_result(controller, request, persisted)
            raise UnknownRunnerOutcome(
                f"Controller outcome is unknown after an unclassified error: {exc}",
                details={"phase": "controller-start"},
            ) from exc
        return self._terminal_result(controller, request, run)

    def recover_child(
        self, request: BoundedRunRequest
    ) -> BoundedRunResult | None:
        """Reconcile only the already-persisted Controller intent.

        Absence returns ``None`` and never calls the Controller factory.  A
        RUNNING record may continue through ``resume``; the Controller's UID
        attempt protocol prevents unknown evaluator work from being repeated.
        """

        context = self._validate_request(request, recovery=True)
        run = self._load_run(request.controller_run_id)
        if run is None:
            return None
        bounded_config = self._bounded_config(request, context["proposer"])
        baseline_ref = self._campaign_baseline(request)
        self._validate_persisted_intent(
            request,
            run,
            bounded_config=bounded_config,
            proposer=context["proposer"],
            baseline_ref=baseline_ref,
        )
        self._require_fence(request, recovery=True)
        controller = self._controller(bounded_config)
        if run["status"] in TERMINAL_RUN_STATUSES:
            return self._terminal_result(controller, request, run)
        if run["status"] != RunStatus.RUNNING.value:
            raise DataIntegrityRunnerFailure(
                f"unsupported nonterminal Controller status {run['status']}"
            )
        try:
            recovered = controller.resume(request.controller_run_id)
        except (ControllerDataIntegrityError, sqlite3.DatabaseError) as exc:
            raise DataIntegrityRunnerFailure(
                str(exc), details={"phase": "controller-recovery"}
            ) from exc
        except UnknownGPUOutcome as exc:
            raise UnknownRunnerOutcome(
                str(exc), details={"phase": "controller-recovery"}
            ) from exc
        except ControlledRuntimeError as exc:
            persisted = self._load_run(request.controller_run_id)
            if persisted is not None and persisted["status"] in TERMINAL_RUN_STATUSES:
                return self._terminal_result(controller, request, persisted)
            raise UnknownRunnerOutcome(
                str(exc), details={"phase": "controller-recovery"}
            ) from exc
        except KeyboardInterrupt as exc:
            raise UnknownRunnerOutcome(
                "Controller recovery was interrupted with an existing run intent",
                details={"phase": "controller-recovery"},
            ) from exc
        except Exception as exc:
            persisted = self._load_run(request.controller_run_id)
            if persisted is not None and persisted["status"] in TERMINAL_RUN_STATUSES:
                return self._terminal_result(controller, request, persisted)
            raise UnknownRunnerOutcome(
                f"Controller recovery outcome is unknown: {exc}",
                details={"phase": "controller-recovery"},
            ) from exc
        return self._terminal_result(controller, request, recovered)

    def _validate_request(
        self, request: BoundedRunRequest, *, recovery: bool
    ) -> dict[str, Any]:
        if not isinstance(request, BoundedRunRequest):
            raise TypeError("request must be BoundedRunRequest")
        if request.recovery is not recovery:
            raise DataIntegrityRunnerFailure(
                "runner method does not match the frozen recovery flag"
            )
        if not _RUN_ID.fullmatch(request.controller_run_id):
            raise DataIntegrityRunnerFailure(
                "controller_run_id is not a safe local container identifier"
            )
        for name, value in (
            ("campaign_id", request.campaign_id),
            ("baseline_revision_id", request.baseline_revision_id),
        ):
            if not isinstance(value, str) or not _TOKEN.fullmatch(value):
                raise DataIntegrityRunnerFailure(
                    f"{name} is not a stable bounded identifier"
                )
        for name, value in (
            ("child_id", request.child_id),
            ("child_index", request.child_index),
        ):
            if type(value) is not int or value <= 0:
                raise DataIntegrityRunnerFailure(f"{name} must be positive")
        try:
            mode = CampaignMode(request.mode)
        except ValueError as exc:
            raise DataIntegrityRunnerFailure("unsupported Campaign mode") from exc
        if request.stop_after_promotion is not True:
            raise DataIntegrityRunnerFailure(
                "Campaign child must stop after its first promotion"
            )
        for name, value, maximum in (
            ("max_candidates", request.max_candidates, MAX_CHILD_CANDIDATES),
            (
                "max_wall_seconds",
                request.max_wall_seconds,
                MAX_CHILD_WALL_SECONDS,
            ),
            (
                "max_consecutive_failures",
                request.max_consecutive_failures,
                MAX_CHILD_CONSECUTIVE_FAILURES,
            ),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise DataIntegrityRunnerFailure(
                    f"{name} is outside the bounded child limit"
                )
        if not isinstance(request.budget_limit, BudgetAmount):
            raise DataIntegrityRunnerFailure("budget_limit must be BudgetAmount")
        try:
            ArtifactId.parse(request.baseline_artifact_id)
            baseline_ref = self._campaign_baseline(request)
        except (TypeError, ValueError) as exc:
            raise DataIntegrityRunnerFailure(
                "Campaign baseline has an invalid artifact identity"
            ) from exc
        snapshot = request.campaign_snapshot
        if not isinstance(snapshot, Mapping):
            raise DataIntegrityRunnerFailure("Campaign snapshot must be an object")
        snapshot = dict(snapshot)
        if canonical_sha256(snapshot) != request.campaign_snapshot_digest:
            raise DataIntegrityRunnerFailure(
                "Campaign snapshot digest does not match the frozen snapshot"
            )
        if snapshot.get("schema_version") != CONTROLLER_RUNNER_SCHEMA_VERSION:
            raise DataIntegrityRunnerFailure(
                "Campaign snapshot has an unsupported schema"
            )
        if snapshot.get("mode") != mode.value:
            raise DataIntegrityRunnerFailure(
                "Campaign mode differs from its frozen snapshot"
            )
        expected_snapshot_fields = {
            "schema_version",
            "mode",
            "campaign_profile",
            "deployment_profile",
            "namespace",
        }
        if mode is CampaignMode.BENCHMARK:
            expected_snapshot_fields.update(
                {
                    "cohort_id",
                    "history_cutoff",
                    "prompt_protocol_digest",
                    "feedback_snapshot_digest",
                    "feedback_snapshot",
                    "benchmark_plan_digest",
                    "benchmark_plan",
                }
            )
        if set(snapshot) != expected_snapshot_fields:
            raise DataIntegrityRunnerFailure(
                "Campaign snapshot fields do not match the trusted schema"
            )
        try:
            namespace = _trusted_target_namespace(snapshot.get("namespace"))
        except ValueError as exc:
            raise DataIntegrityRunnerFailure(
                "Campaign namespace snapshot is not an active built-in target"
            ) from exc
        if (
            request.namespace_id != namespace.namespace_id
            or baseline_ref.namespace_id != namespace.namespace_id
        ):
            raise DataIntegrityRunnerFailure(
                "Campaign request, snapshot and baseline namespaces differ"
            )
        if (
            namespace == CURRENT_RESEARCH_NAMESPACE
            and not baseline_ref.is_scientifically_comparable
        ):
            raise DataIntegrityRunnerFailure(
                "CURRENT Campaigns require a resolved V2 baseline"
            )
        self._exact_profile(
            snapshot.get("campaign_profile"), _CAMPAIGN_PROFILE, "campaign"
        )
        self._exact_profile(
            snapshot.get("deployment_profile"),
            _DEPLOYMENT_PROFILE,
            "deployment",
        )
        try:
            proposer = BUILTIN_PROFILE_REGISTRY.resolve(
                ProfileRef.from_value(request.proposer_profile)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataIntegrityRunnerFailure(
                "proposer profile is not a trusted built-in revision"
            ) from exc
        if (
            proposer.ref.kind != "proposer"
            or proposer.ref.id not in _ALLOWED_PROPOSER_IDS
            or proposer.implementation_id != proposer.ref.id
            or proposer.config.get("harness") != "opencode"
            or proposer.config.get("model")
            not in {
                "deepseek/deepseek-v4-pro",
                "deepseek/deepseek-v4-flash",
            }
        ):
            raise DataIntegrityRunnerFailure(
                "only the exact built-in OpenCode/DeepSeek proposer profiles are active"
            )
        resource_pool = tuple(_DEPLOYMENT_PROFILE.config.get("resource_pool", ()))
        if request.resource_id not in resource_pool:
            raise DataIntegrityRunnerFailure(
                "resource_id is outside the exact local C500 deployment profile"
            )
        if type(request.fencing_epoch) is not int or request.fencing_epoch <= 0:
            raise DataIntegrityRunnerFailure("resource fencing_epoch must be positive")
        if (
            isinstance(request.lease_expires_epoch, bool)
            or not isinstance(request.lease_expires_epoch, (int, float))
            or not math.isfinite(request.lease_expires_epoch)
        ):
            raise DataIntegrityRunnerFailure(
                "resource lease expiry must be a finite epoch"
            )
        if (
            isinstance(request.child_deadline_epoch, bool)
            or not isinstance(request.child_deadline_epoch, (int, float))
            or not math.isfinite(request.child_deadline_epoch)
            or float(request.child_deadline_epoch)
            > float(request.lease_expires_epoch)
        ):
            raise DataIntegrityRunnerFailure(
                "child deadline must be finite and no later than lease expiry"
            )
        if mode is CampaignMode.BENCHMARK:
            if (
                not isinstance(snapshot.get("cohort_id"), str)
                or not _TOKEN.fullmatch(str(snapshot["cohort_id"]))
                or type(snapshot.get("history_cutoff")) is not int
                or int(snapshot["history_cutoff"]) < 0
            ):
                raise DataIntegrityRunnerFailure(
                    "benchmark snapshot is missing cohort_id/history_cutoff"
                )
            for field in (
                "prompt_protocol_digest",
                "feedback_snapshot_digest",
            ):
                try:
                    require_sha256_digest(snapshot.get(field), field=field)
                except ValueError as exc:
                    raise DataIntegrityRunnerFailure(
                        f"benchmark snapshot has invalid {field}"
                    ) from exc
            try:
                benchmark_plan = benchmark_plan_from_campaign_snapshot(snapshot)
                require_executable_benchmark_plan(benchmark_plan)
            except (TypeError, ValueError) as exc:
                raise DataIntegrityRunnerFailure(
                    f"benchmark Campaign plan is invalid: {exc}"
                ) from exc
            schedule = benchmark_plan.schedule()
            assignment_index = request.child_index - 1
            if not 0 <= assignment_index < len(schedule):
                raise DataIntegrityRunnerFailure(
                    "benchmark child is outside the frozen assignment schedule"
                )
            assignment = schedule[assignment_index]
            if (
                request.controller_run_id
                != benchmark_controller_run_id(
                    request.campaign_id, benchmark_plan, assignment
                )
                or proposer.ref != assignment.proposer_profile
                or request.max_candidates != benchmark_plan.max_candidates
                or request.max_wall_seconds != benchmark_plan.max_wall_seconds
                or request.max_consecutive_failures
                != benchmark_plan.max_consecutive_failures
                or baseline_ref != benchmark_plan.baseline
            ):
                raise DataIntegrityRunnerFailure(
                    "benchmark child differs from its frozen assignment"
                )
        else:
            forbidden = {
                "cohort_id",
                "history_cutoff",
                "prompt_protocol_digest",
                "feedback_snapshot_digest",
            }
            if any(snapshot.get(field) is not None for field in forbidden):
                raise DataIntegrityRunnerFailure(
                    "discovery snapshot contains benchmark-only fields"
                )
        self._validate_trusted_budget(request, proposer)
        return {
            "mode": mode,
            "snapshot": snapshot,
            "proposer": proposer,
            "namespace": namespace,
            "baseline_ref": baseline_ref,
        }

    def _validate_trusted_budget(
        self,
        request: BoundedRunRequest,
        proposer: ProfileDefinition,
    ) -> None:
        """Prove the request reservation against the authoritative Campaign DB.

        The Supervisor persists the reservation before invoking this adapter.
        Adding that exact reservation back to the current remaining balance
        reconstructs the balance observed by the trusted CLI immediately before
        the intent.  Re-derivation therefore detects both a forged plan and a
        reservation belonging to another Controller action.
        """

        key = child_budget_idempotency_key(request.controller_run_id)
        try:
            campaign = self.campaign_store.get_campaign(request.campaign_id)
            action = self.campaign_store.get_budget_action(
                request.campaign_id, idempotency_key=key
            )
            status = self.campaign_store.budget_status(request.campaign_id)
            remaining = BudgetAmount.from_mapping(status["remaining"])
        except (KeyError, TypeError, ValueError, sqlite3.DatabaseError) as exc:
            raise DataIntegrityRunnerFailure(
                f"trusted Campaign budget intent cannot be read: {exc}"
            ) from exc
        campaign_identity = {
            "namespace_id": request.namespace_id,
            "mode": request.mode,
            "snapshot_digest": request.campaign_snapshot_digest,
            "status": CampaignStatus.RUNNING.value,
        }
        if any(
            campaign.get(name) != value
            for name, value in campaign_identity.items()
        ):
            raise DataIntegrityRunnerFailure(
                "Campaign budget store belongs to a different frozen campaign"
            )
        if (
            action.get("action_kind") != "CHILD_RUN"
            or action.get("status") != "RESERVED"
            or action.get("reserved") != request.budget_limit.to_dict()
        ):
            raise DataIntegrityRunnerFailure(
                "child budget action is absent, settled, or differs from the request"
            )
        revisions = self.campaign_store.list_baseline_revisions(
            request.campaign_id
        )
        revision = next(
            (
                item
                for item in revisions
                if item.get("id") == request.baseline_revision_id
            ),
            None,
        )
        if (
            revision is None
            or revision.get("artifact_id") != request.baseline_artifact_id
            or revision.get("namespace_id") != request.namespace_id
            or revision.get("baseline_ref") != request.baseline_ref
        ):
            raise DataIntegrityRunnerFailure(
                "child baseline request differs from authoritative Campaign lineage"
            )
        before_reservation = remaining + request.budget_limit
        try:
            expected = trusted_child_reservation(
                proposer_profile=proposer.ref.to_dict(),
                campaign_remaining=before_reservation,
                max_candidates=request.max_candidates,
                max_wall_seconds=request.max_wall_seconds,
                max_consecutive_failures=request.max_consecutive_failures,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataIntegrityRunnerFailure(
                f"trusted child worst-case budget cannot be derived: {exc}"
            ) from exc
        if request.budget_limit != expected:
            raise DataIntegrityRunnerFailure(
                "child budget reservation is not the exact trusted five-axis worst case"
            )

    @staticmethod
    def _exact_profile(
        value: object,
        expected: ProfileDefinition,
        name: str,
    ) -> None:
        try:
            selected = ProfileRef.from_value(value)
            resolved = BUILTIN_PROFILE_REGISTRY.resolve(selected)
        except (KeyError, TypeError, ValueError) as exc:
            raise DataIntegrityRunnerFailure(
                f"{name} profile is not a trusted built-in revision"
            ) from exc
        if resolved.ref != expected.ref:
            raise DataIntegrityRunnerFailure(
                f"{name} profile is not the exact active revision"
            )

    def _require_fence(
        self, request: BoundedRunRequest, *, recovery: bool
    ) -> ResourceLease:
        try:
            lease = self.fence_lookup(request.resource_id)
        except Exception as exc:
            failure = UnknownRunnerOutcome if recovery else DataIntegrityRunnerFailure
            raise failure(
                f"resource fencing lookup failed: {exc}",
                details={"resource_id": request.resource_id},
            ) from exc
        mismatch = (
            not isinstance(lease, ResourceLease)
            or lease.resource_id != request.resource_id
            or lease.campaign_id != request.campaign_id
            or lease.fencing_epoch != request.fencing_epoch
            or lease.expires_epoch != float(request.lease_expires_epoch)
            or lease.status != "ACTIVE"
        )
        now = float(self.clock())
        if not math.isfinite(now):
            raise DataIntegrityRunnerFailure("runner clock returned a non-finite epoch")
        if mismatch or lease is None or lease.expires_epoch <= now:
            failure = UnknownRunnerOutcome if recovery else DataIntegrityRunnerFailure
            raise failure(
                "resource fencing token is absent, stale, or no longer owned",
                details={
                    "resource_id": request.resource_id,
                    "fencing_epoch": request.fencing_epoch,
                },
            )
        return lease

    def _bounded_config(
        self, request: BoundedRunRequest, proposer: ProfileDefinition
    ) -> ControllerConfig:
        config = replace(
            self.config,
            opencode_model=str(proposer.config["model"]),
            max_candidates=request.max_candidates,
            max_hours=request.max_wall_seconds / 3600.0,
            max_consecutive_failures=request.max_consecutive_failures,
            stop_after_promotion=True,
        )
        if round(config.max_hours * 3600) != request.max_wall_seconds:
            raise DataIntegrityRunnerFailure(
                "ControllerConfig cannot represent the integral wall limit"
            )
        return config

    @staticmethod
    def _campaign_baseline(request: BoundedRunRequest) -> BaselineRef:
        try:
            if not isinstance(request.baseline_ref, Mapping):
                raise ValueError("Campaign baseline reference is absent")
            selected = BaselineRef.from_value(dict(request.baseline_ref))
            if (
                selected.source != "campaign"
                or selected.namespace_id != request.namespace_id
                or str(selected.artifact_id) != request.baseline_artifact_id
            ):
                raise ValueError("Campaign baseline scalar mirrors differ")
            return selected
        except (TypeError, ValueError) as exc:
            raise DataIntegrityRunnerFailure(
                "Campaign baseline cannot form a frozen BaselineRef"
            ) from exc

    def _history_cutoff(self) -> int:
        history_db = self.config.state_dir / "history.sqlite3"
        if not history_db.is_file():
            raise DataIntegrityRunnerFailure("History database does not exist")
        try:
            with HistoryStore(
                history_db, state_dir=self.config.state_dir
            ) as history:
                records = history.list_all_experiments(
                    limit=1, newest_first=True
                )
        except (OSError, sqlite3.DatabaseError) as exc:
            raise DataIntegrityRunnerFailure(
                f"History cutoff could not be frozen: {exc}"
            ) from exc
        return 0 if not records else int(records[0].id)

    def _workflow_snapshot(
        self,
        request: BoundedRunRequest,
        *,
        bounded_config: ControllerConfig,
        proposer: ProfileDefinition,
        baseline_ref: BaselineRef,
        history_cutoff: int,
    ) -> dict[str, Any]:
        namespace = _trusted_target_namespace(
            request.campaign_snapshot["namespace"]
        )
        try:
            target = _trusted_target(namespace)
        except ControllerDataIntegrityError as exc:
            raise DataIntegrityRunnerFailure(str(exc)) from exc
        runtime_environment = ResearchController(
            bounded_config
        )._resolved_execution_environment(namespace)
        if baseline_ref.execution_environment.is_resolved:
            try:
                baseline_ref.require_environment(runtime_environment)
            except ValueError as exc:
                raise DataIntegrityRunnerFailure(str(exc)) from exc
            comparable = True
            compatibility_mode = None
        else:
            if namespace != LEGACY_RESEARCH_NAMESPACE:
                raise DataIntegrityRunnerFailure(
                    "LEGACY_UNKNOWN evidence is restricted to the LEGACY namespace"
                )
            comparable = False
            compatibility_mode = "LEGACY_V1_EVIDENCE"
        references = {
            "deployment": _DEPLOYMENT_PROFILE.ref,
            "campaign": _CAMPAIGN_PROFILE.ref,
            "operator": namespace.operator,
            "language": namespace.language,
            "evaluator": namespace.evaluator,
            "evaluation_protocol": namespace.evaluation_protocol,
            "promotion_policy": namespace.promotion_policy,
            "proposer": proposer.ref,
        }
        campaign_snapshot = dict(request.campaign_snapshot)
        snapshot: dict[str, Any] = {
            "schema_version": 2,
            "mode": request.mode,
            "namespace": namespace.to_dict(),
            "deployment_profile": _DEPLOYMENT_PROFILE.ref.to_dict(),
            "campaign_profile": _CAMPAIGN_PROFILE.ref.to_dict(),
            "proposer_profile": proposer.ref.to_dict(),
            "resolved_profiles": {
                name: BUILTIN_PROFILE_REGISTRY.resolve(reference).to_dict()
                for name, reference in references.items()
            },
            "target_components": _target_binding_snapshot(target),
            "baseline_ref": baseline_ref.to_dict(),
            "execution_environment": (
                baseline_ref.execution_environment.to_dict()
            ),
            "runtime_execution_environment": runtime_environment.to_dict(),
            "scientifically_comparable": comparable,
            "compatibility_mode": compatibility_mode,
            "history_cutoff": history_cutoff,
            "workflow": [stage.stage_id for stage in target.protocol.stages],
            "budget": {
                "max_candidates": request.max_candidates,
                "max_wall_seconds": request.max_wall_seconds,
                "max_consecutive_failures": request.max_consecutive_failures,
                "stop_after_promotion": True,
            },
            "campaign_budget_reservation": {
                "reservation_derivation": "trusted-child-worst-case-v1",
                "reservation_idempotency_key": child_budget_idempotency_key(
                    request.controller_run_id
                ),
                "reservation": request.budget_limit.to_dict(),
            },
            "runtime_binding": bounded_config.redacted_dict(),
            "campaign_id": request.campaign_id,
            "campaign_snapshot_digest": request.campaign_snapshot_digest,
            "campaign_snapshot": campaign_snapshot,
            "campaign_child": {
                "child_id": request.child_id,
                "child_index": request.child_index,
                "controller_run_id": request.controller_run_id,
                "baseline_revision_id": request.baseline_revision_id,
            },
            "resource_lease": {
                "resource_id": request.resource_id,
                "fencing_epoch": request.fencing_epoch,
                "expires_epoch": float(request.lease_expires_epoch),
            },
            "child_deadline_epoch": float(request.child_deadline_epoch),
        }
        if request.mode == CampaignMode.BENCHMARK.value:
            snapshot.update(
                {
                    "cohort_id": campaign_snapshot["cohort_id"],
                    "prompt_protocol_digest": campaign_snapshot[
                        "prompt_protocol_digest"
                    ],
                    "feedback_snapshot_digest": campaign_snapshot[
                        "feedback_snapshot_digest"
                    ],
                    "benchmark_plan_digest": campaign_snapshot[
                        "benchmark_plan_digest"
                    ],
                }
            )
        return {**snapshot, "snapshot_digest": canonical_sha256(snapshot)}

    def _controller(self, config: ControllerConfig) -> ResearchController:
        try:
            controller = self.controller_factory(config)
        except Exception as exc:
            raise DataIntegrityRunnerFailure(
                f"trusted Controller factory failed: {exc}"
            ) from exc
        if getattr(controller, "config", None) != config:
            raise DataIntegrityRunnerFailure(
                "Controller factory did not preserve the bounded config"
            )
        for method in ("start_from_snapshot", "resume", "checkpoint"):
            if not callable(getattr(controller, method, None)):
                raise DataIntegrityRunnerFailure(
                    f"Controller factory result is missing {method}"
                )
        return controller

    def _load_run(self, run_id: str) -> dict[str, Any] | None:
        if not self.controller_db.is_file():
            return None
        try:
            with ControllerStore(self.controller_db) as store:
                row = store.connection.execute(
                    "SELECT 1 FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                return None if row is None else store.get_run(run_id)
        except (OSError, sqlite3.DatabaseError, ValueError) as exc:
            raise DataIntegrityRunnerFailure(
                f"Controller run intent cannot be read: {exc}"
            ) from exc

    def _validate_persisted_intent(
        self,
        request: BoundedRunRequest,
        run: Mapping[str, Any],
        *,
        bounded_config: ControllerConfig,
        proposer: ProfileDefinition,
        baseline_ref: BaselineRef,
    ) -> None:
        cutoff = run.get("history_cutoff")
        if type(cutoff) is not int or cutoff < 0:
            raise DataIntegrityRunnerFailure(
                "persisted Controller run has no numeric History cutoff"
            )
        expected_snapshot = self._workflow_snapshot(
            request,
            bounded_config=bounded_config,
            proposer=proposer,
            baseline_ref=baseline_ref,
            history_cutoff=cutoff,
        )
        expected = {
            "id": request.controller_run_id,
            "namespace_id": request.namespace_id,
            "resolved_config_digest": expected_snapshot["snapshot_digest"],
            "workflow_snapshot": expected_snapshot,
            "baseline_ref": baseline_ref.to_dict(),
            "history_cutoff": cutoff,
            "config": bounded_config.redacted_dict(),
        }
        conflicts = [
            key for key, value in expected.items() if run.get(key) != value
        ]
        if conflicts:
            raise DataIntegrityRunnerFailure(
                "persisted Controller intent differs from Campaign request: "
                + ", ".join(sorted(conflicts))
            )
        if request.mode == CampaignMode.BENCHMARK.value and (
            cutoff != int(request.campaign_snapshot["history_cutoff"])
        ):
            raise DataIntegrityRunnerFailure(
                "benchmark Controller cutoff differs from its Campaign cohort"
            )

    def _terminal_result(
        self,
        controller: ResearchController,
        request: BoundedRunRequest,
        run: Mapping[str, Any],
    ) -> BoundedRunResult:
        status = str(run.get("status"))
        try:
            usage, usage_details = self._usage(request, run)
        except DataIntegrityRunnerFailure:
            raise
        except (OverflowError, TypeError, ValueError) as exc:
            raise DataIntegrityRunnerFailure(
                f"Controller usage evidence is invalid: {exc}"
            ) from exc
        details: dict[str, Any] = {
            "controller_run_id": request.controller_run_id,
            "controller_status": status,
            "stop_reason": run.get("stop_reason"),
            **usage_details,
        }
        if status == RunStatus.PROMOTED.value:
            promotion = self._promotion_evidence(controller, request, run)
            return BoundedRunResult(
                RunnerOutcome.PROMOTED,
                usage,
                details,
                promotion=promotion,
            )
        direct = {
            RunStatus.BUDGET_EXHAUSTED.value: RunnerOutcome.BUDGET_EXHAUSTED,
            RunStatus.STOPPED.value: RunnerOutcome.STOPPED,
            RunStatus.FAILED.value: RunnerOutcome.FAILED,
        }
        if status in direct:
            return BoundedRunResult(direct[status], usage, details)
        if status == RunStatus.PROPOSAL_READY.value:
            raise DataIntegrityRunnerFailure(
                "proposal-only Controller result is invalid for a Campaign child",
                details=details,
            )
        if status == RunStatus.HARD_FAILED.value:
            outcome = self._latest_iteration_outcome(request.controller_run_id)
            details["iteration_outcome"] = outcome
            if outcome == "UNKNOWN_GPU_OUTCOME":
                raise UnknownRunnerOutcome(
                    "Controller recorded an unknown GPU outcome",
                    usage=usage,
                    details=details,
                )
            if outcome == "DATA_INTEGRITY_FAILURE":
                raise DataIntegrityRunnerFailure(
                    "Controller recorded a data-integrity failure",
                    usage=usage,
                    details=details,
                )
            if outcome == "HARD_FAILURE":
                raise HardRunnerFailure(
                    "Controller recorded a hard GPU failure",
                    usage=usage,
                    details=details,
                )
            raise DataIntegrityRunnerFailure(
                "HARD_FAILED Controller run has no classified terminal outcome",
                usage=usage,
                details=details,
            )
        if status == RunStatus.RUNNING.value:
            raise UnknownRunnerOutcome(
                "Controller returned without a durable terminal status",
                usage=usage,
                details=details,
            )
        raise DataIntegrityRunnerFailure(
            f"unknown Controller terminal status {status}",
            usage=usage,
            details=details,
        )

    def _latest_iteration_outcome(self, run_id: str) -> str | None:
        try:
            with ControllerStore(self.controller_db) as store:
                iteration = store.latest_iteration(run_id)
        except (OSError, sqlite3.DatabaseError, ValueError) as exc:
            raise DataIntegrityRunnerFailure(
                f"Controller iteration outcome cannot be read: {exc}"
            ) from exc
        return None if iteration is None else iteration.get("outcome")

    def _usage(
        self, request: BoundedRunRequest, run: Mapping[str, Any]
    ) -> tuple[BudgetAmount, dict[str, Any]]:
        try:
            with ControllerStore(self.controller_db) as store:
                proposals = store.list_proposal_attempts(
                    request.controller_run_id
                )
                evaluations = store.list_evaluation_attempts(
                    request.controller_run_id
                )
        except (OSError, sqlite3.DatabaseError, ValueError) as exc:
            raise DataIntegrityRunnerFailure(
                f"Controller usage evidence cannot be read: {exc}"
            ) from exc

        unavailable: dict[str, str] = {}
        wall_ms = self._duration_ms(
            run.get("created_at"), run.get("updated_at")
        )
        if wall_ms is None:
            wall_ms = request.budget_limit.wall_ms
            unavailable["wall_ms"] = "Controller timestamps are unavailable"

        gpu_ms = 0
        for attempt in evaluations:
            if attempt.get("started_at") is None:
                gpu_ms = request.budget_limit.gpu_ms
                unavailable["gpu_ms"] = (
                    "an evaluation attempt omitted its start timestamp"
                )
                break
            duration = self._duration_ms(
                attempt.get("started_at"),
                attempt.get("finished_at") or attempt.get("updated_at"),
            )
            if duration is None:
                gpu_ms = request.budget_limit.gpu_ms
                unavailable["gpu_ms"] = (
                    "an evaluation attempt has incomplete timestamps"
                )
                break
            gpu_ms += duration

        if proposals and any(
            attempt.get("input_tokens") is None
            or attempt.get("output_tokens") is None
            for attempt in proposals
        ):
            tokens = request.budget_limit.tokens
            unavailable["tokens"] = "a proposer attempt omitted token usage"
        else:
            tokens = sum(
                int(attempt.get("input_tokens") or 0)
                + int(attempt.get("output_tokens") or 0)
                for attempt in proposals
            )
        if proposals and any(
            attempt.get("cost_usd") is None for attempt in proposals
        ):
            cost_microusd = request.budget_limit.cost_microusd
            unavailable["cost_microusd"] = (
                "a proposer attempt omitted cost usage"
            )
        else:
            costs = [
                float(attempt.get("cost_usd") or 0.0)
                for attempt in proposals
            ]
            if any(not math.isfinite(cost) or cost < 0 for cost in costs):
                raise DataIntegrityRunnerFailure(
                    "a proposer attempt has invalid cost usage"
                )
            cost_microusd = math.ceil(sum(costs) * 1_000_000)
        candidate_value = run.get("valid_candidates")
        if (
            isinstance(candidate_value, bool)
            or not isinstance(candidate_value, int)
            or candidate_value < 0
        ):
            candidates = request.budget_limit.candidates
            unavailable["candidates"] = (
                "Controller valid-candidate usage is unavailable"
            )
        else:
            candidates = candidate_value
        return (
            BudgetAmount(
                candidates=candidates,
                wall_ms=wall_ms,
                gpu_ms=gpu_ms,
                tokens=tokens,
                cost_microusd=cost_microusd,
            ),
            {
                "proposal_attempts": len(proposals),
                "evaluation_attempts": len(evaluations),
                "usage_unavailable": unavailable,
            },
        )

    @staticmethod
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

    def _promotion_evidence(
        self,
        controller: ResearchController,
        request: BoundedRunRequest,
        run: Mapping[str, Any],
    ) -> PromotionEvidence:
        try:
            with ControllerStore(self.controller_db) as store:
                promoted = [
                    item
                    for item in store.list_iterations(request.controller_run_id)
                    if item.get("outcome") == "PROMOTED"
                ]
                if len(promoted) != 1:
                    raise ControllerDataIntegrityError(
                        "promoted run must contain exactly one promoted iteration"
                    )
                attempts = store.list_evaluation_attempts(
                    request.controller_run_id,
                    iteration_id=int(promoted[0]["id"]),
                )
            by_stage = {
                stage: [item for item in attempts if item.get("stage") == stage]
                for stage in ("full_primary", "confirmation")
            }
            if any(len(items) != 1 for items in by_stage.values()):
                raise ControllerDataIntegrityError(
                    "promotion requires one primary and one confirmation attempt"
                )
            primary_attempt = by_stage["full_primary"][0]
            confirmation_attempt = by_stage["confirmation"][0]
            proven = {
                "primary": self._proven_experiment(primary_attempt, request),
                "confirmation": self._proven_experiment(
                    confirmation_attempt, request
                ),
            }
            primary, primary_identity = proven["primary"]
            confirmation, confirmation_identity = proven["confirmation"]
            artifact_id = str(primary_attempt.get("candidate_artifact_id"))
            if (
                artifact_id != confirmation_attempt.get("candidate_artifact_id")
                or primary.artifact_id != artifact_id
                or confirmation.artifact_id != artifact_id
                or primary.candidate_hash != run.get("final_best_hash")
                or confirmation.candidate_hash != run.get("final_best_hash")
            ):
                raise ControllerDataIntegrityError(
                    "promotion attempts do not prove the final candidate artifact"
                )
            if primary.result.get("promotion", {}).get("phase") != "primary":
                raise ControllerDataIntegrityError(
                    "primary experiment has no primary promotion decision"
                )
            if not bool(
                confirmation.result.get("promotion", {})
                .get("decision", {})
                .get("promoted")
            ) or not confirmation.promotable:
                raise ControllerDataIntegrityError(
                    "confirmation experiment does not prove promotion"
                )
            if (
                primary_identity.candidate_artifact_id
                != confirmation_identity.candidate_artifact_id
                or primary_identity.baseline != confirmation_identity.baseline
                or primary_identity.execution_environment
                != confirmation_identity.execution_environment
            ):
                raise ControllerDataIntegrityError(
                    "primary and confirmation scientific identities differ"
                )
            checkpoint_digest = self._checkpoint_digest(
                controller, request.controller_run_id
            )
            promotion_profile = run["workflow_snapshot"]["resolved_profiles"][
                "promotion_policy"
            ]
            return PromotionEvidence(
                artifact_id=artifact_id,
                namespace_id=request.namespace_id,
                parent_baseline_ref=primary_identity.baseline.to_dict(),
                execution_environment=(
                    primary_identity.execution_environment.to_dict()
                ),
                primary_experiment_uid=primary.experiment_uid,
                confirmation_experiment_uid=confirmation.experiment_uid,
                checkpoint_digest=checkpoint_digest,
                policy_snapshot={
                    "profile": promotion_profile,
                    "primary": primary.result.get("promotion", {}),
                    "confirmation": confirmation.result.get("promotion", {}),
                },
            )
        except DataIntegrityRunnerFailure:
            raise
        except (KeyError, OSError, TypeError, ValueError, sqlite3.DatabaseError) as exc:
            raise DataIntegrityRunnerFailure(
                f"promotion evidence is incomplete or inconsistent: {exc}"
            ) from exc

    def _proven_experiment(
        self, attempt: Mapping[str, Any], request: BoundedRunRequest
    ) -> tuple[ExperimentRecord, ExperimentIdentity]:
        if (
            attempt.get("status") != "SUCCEEDED"
            or not isinstance(attempt.get("history_experiment_id"), int)
            or not isinstance(attempt.get("experiment_uid"), str)
        ):
            raise ControllerDataIntegrityError(
                "promotion evaluation attempt is not durably linked"
            )
        with HistoryStore(
            self.config.state_dir / "history.sqlite3",
            state_dir=self.config.state_dir,
        ) as history:
            record = history.get_experiment_by_uid(attempt["experiment_uid"])
        if (
            record is None
            or record.id != attempt["history_experiment_id"]
            or record.status != "SUCCESS"
            or record.namespace_id != request.namespace_id
        ):
            raise ControllerDataIntegrityError(
                "promotion attempt and History record do not match"
            )
        try:
            identity = ExperimentIdentity.from_value(record.identity)
        except (TypeError, ValueError) as exc:
            raise ControllerDataIntegrityError(
                "promotion History identity is invalid"
            ) from exc
        if (
            identity.experiment_uid != record.experiment_uid
            or identity.namespace.namespace_id != request.namespace_id
            or identity.run_id != request.controller_run_id
            or identity.campaign_id != request.campaign_id
            or identity.baseline.to_dict() != request.baseline_ref
            or str(identity.parent_artifact_id)
            != request.baseline_artifact_id
        ):
            raise ControllerDataIntegrityError(
                "promotion History identity differs from the Campaign child"
            )
        return record, identity

    def prove_staged_promotion(
        self,
        *,
        campaign_id: str,
        controller_run_id: str,
        evidence: PromotionEvidence,
    ) -> PromotionEvidence:
        """Re-read both History identities immediately before lineage CAS."""

        if not isinstance(evidence, PromotionEvidence):
            raise TypeError("evidence must be PromotionEvidence")
        try:
            parent = BaselineRef.from_value(dict(evidence.parent_baseline_ref))
            environment = ExecutionEnvironmentDigest.from_value(
                dict(evidence.execution_environment)
            )
        except (TypeError, ValueError) as exc:
            raise DataIntegrityRunnerFailure(
                "staged promotion has invalid parent/environment evidence"
            ) from exc
        if not parent.is_scientifically_comparable or not environment.is_resolved:
            raise DataIntegrityRunnerFailure(
                "LEGACY_UNKNOWN evidence cannot advance staged lineage"
            )
        try:
            parent.require_environment(environment)
        except ValueError as exc:
            raise DataIntegrityRunnerFailure(str(exc)) from exc
        identities: dict[str, tuple[ExperimentRecord, ExperimentIdentity]] = {}
        for role, uid, expected_stage in (
            ("primary", evidence.primary_experiment_uid, "full_primary"),
            (
                "confirmation",
                evidence.confirmation_experiment_uid,
                "confirmation",
            ),
        ):
            with HistoryStore(
                self.config.state_dir / "history.sqlite3",
                state_dir=self.config.state_dir,
            ) as history:
                record = history.get_experiment_by_uid(uid)
            if record is None or record.status != "SUCCESS":
                raise DataIntegrityRunnerFailure(
                    f"{role} History experiment is absent or unsuccessful"
                )
            try:
                identity = ExperimentIdentity.from_value(record.identity)
            except (TypeError, ValueError) as exc:
                raise DataIntegrityRunnerFailure(
                    f"{role} History experiment identity is invalid"
                ) from exc
            if (
                record.artifact_id != evidence.artifact_id
                or str(identity.candidate_artifact_id) != evidence.artifact_id
                or identity.experiment_uid != uid
                or identity.namespace.namespace_id != evidence.namespace_id
                or identity.campaign_id != campaign_id
                or identity.run_id != controller_run_id
                or identity.stage != expected_stage
                or identity.suite != "full"
                or identity.mode != CampaignMode.DISCOVERY.value
                or identity.replicate_kind != role
                or identity.baseline != parent
                or identity.execution_environment != environment
            ):
                raise DataIntegrityRunnerFailure(
                    f"{role} History identity differs from staged promotion"
                )
            identities[role] = (record, identity)
        primary, primary_identity = identities["primary"]
        confirmation, confirmation_identity = identities["confirmation"]
        if (
            primary_identity.condition_digest
            == confirmation_identity.condition_digest
        ):
            # Stage and replicate are part of the condition, so equality here
            # would signal malformed or replayed identity material.
            raise DataIntegrityRunnerFailure(
                "primary and confirmation identities are not distinct conditions"
            )
        if primary.result.get("promotion", {}).get("phase") != "primary":
            raise DataIntegrityRunnerFailure(
                "primary History result lacks a primary promotion decision"
            )
        confirmation_promotion = confirmation.result.get("promotion", {})
        if (
            not confirmation.promotable
            or not bool(
                confirmation_promotion.get("decision", {}).get("promoted")
            )
        ):
            raise DataIntegrityRunnerFailure(
                "confirmation History result does not prove promotion"
            )
        return evidence

    def _checkpoint_digest(
        self, controller: ResearchController, run_id: str
    ) -> str:
        checkpoint_root = self.config.checkpoint_dir
        valid = [
            path
            for path in sorted(checkpoint_root.glob(f"{run_id}-*"))
            if self._valid_checkpoint(path, run_id)
        ] if checkpoint_root.is_dir() else []
        if len(valid) > 1:
            raise DataIntegrityRunnerFailure(
                "multiple valid checkpoints exist for the promoted run"
            )
        if valid:
            selected = valid[0]
        else:
            try:
                result = controller.checkpoint(run_id)
            except Exception as exc:
                raise DataIntegrityRunnerFailure(
                    f"Campaign checkpoint could not be created: {exc}"
                ) from exc
            selected_value = result.get("path") if isinstance(result, Mapping) else None
            if not isinstance(selected_value, str):
                raise DataIntegrityRunnerFailure(
                    "Controller checkpoint did not return an immutable path"
                )
            selected = Path(selected_value)
            if not self._valid_checkpoint(selected, run_id):
                raise DataIntegrityRunnerFailure(
                    "Controller checkpoint failed post-write validation"
                )
        manifest = selected / "manifest.json"
        return "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()

    def _valid_checkpoint(self, path: Path, run_id: str) -> bool:
        try:
            root = self.config.checkpoint_dir.resolve(strict=True)
            if path.is_symlink() or not path.is_dir():
                return False
            selected = path.resolve(strict=True)
            if selected.parent != root or not selected.name.startswith(f"{run_id}-"):
                return False
            manifest_path = selected / "manifest.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                return False
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema_version") != 2
                or manifest.get("run_id") != run_id
                or manifest.get("run_status") != RunStatus.PROMOTED.value
                or not isinstance(manifest.get("files"), list)
            ):
                return False
            declared: set[str] = set()
            for item in manifest["files"]:
                if not isinstance(item, dict) or set(item) != {
                    "path",
                    "size_bytes",
                    "sha256",
                }:
                    return False
                relative = item["path"]
                if not isinstance(relative, str):
                    return False
                posix = PurePosixPath(relative)
                if (
                    posix.is_absolute()
                    or posix.as_posix() != relative
                    or not posix.parts
                    or any(part in {"", ".", ".."} for part in posix.parts)
                    or relative in declared
                ):
                    return False
                candidate = selected.joinpath(*posix.parts)
                if candidate.is_symlink() or not candidate.is_file():
                    return False
                if (
                    type(item["size_bytes"]) is not int
                    or item["size_bytes"] != candidate.stat().st_size
                    or not isinstance(item["sha256"], str)
                    or not _HEX.fullmatch(item["sha256"])
                    or hashlib.sha256(candidate.read_bytes()).hexdigest()
                    != item["sha256"]
                ):
                    return False
                declared.add(relative)
            descendants = list(selected.rglob("*"))
            if any(candidate.is_symlink() for candidate in descendants):
                return False
            actual = {
                candidate.relative_to(selected).as_posix()
                for candidate in descendants
                if candidate.is_file() and candidate != manifest_path
            }
            return actual == declared
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return False


__all__ = [
    "CONTROLLER_RUNNER_SCHEMA_VERSION",
    "FenceLookup",
    "ResearchControllerRunner",
    "legacy_campaign_snapshot",
    "trusted_child_reservation",
    "trusted_resume_doctor",
]
