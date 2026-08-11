"""Durable campaign orchestration primitives.

Campaigns sequence bounded controller runs.  They do not evaluate candidates,
promote History rows, write the repository, or adopt deployment baselines.
"""

from .models import (
    BudgetAmount,
    CampaignMode,
    CampaignStatus,
    ChildRunStatus,
    ResourceLease,
)
from .store import CampaignStore
from .supervisor import (
    BoundedChildRunner,
    BoundedRunRequest,
    BoundedRunResult,
    BoundedRunnerFailure,
    CampaignSupervisor,
    ChildRunPlan,
    DataIntegrityRunnerFailure,
    HardRunnerFailure,
    PromotionEvidence,
    RunnerOutcome,
    UnknownRunnerOutcome,
    child_budget_idempotency_key,
)
from .benchmark import (
    BenchmarkArm,
    BenchmarkAssignment,
    BenchmarkCohortPlan,
    BenchmarkObservation,
    CURRENT_PROMPT_PROTOCOL_DIGEST,
    benchmark_controller_run_id,
    benchmark_plan_from_campaign_snapshot,
    bind_benchmark_campaign_snapshot,
    build_benchmark_report,
    build_trusted_benchmark_report,
    derive_authoritative_benchmark_plan,
    execute_benchmark_campaign,
    filter_feedback_at_cutoff,
    require_executable_benchmark_plan,
)
from .soak import SOAK_STAGES, SoakGate, SoakStage
from .controller_runner import (
    ResearchControllerRunner,
    legacy_campaign_snapshot,
    trusted_child_reservation,
    trusted_resume_doctor,
)

__all__ = [
    "BudgetAmount",
    "BenchmarkArm",
    "BenchmarkAssignment",
    "BenchmarkCohortPlan",
    "BenchmarkObservation",
    "CURRENT_PROMPT_PROTOCOL_DIGEST",
    "BoundedChildRunner",
    "BoundedRunRequest",
    "BoundedRunResult",
    "BoundedRunnerFailure",
    "CampaignMode",
    "CampaignStatus",
    "CampaignStore",
    "CampaignSupervisor",
    "ChildRunPlan",
    "ChildRunStatus",
    "DataIntegrityRunnerFailure",
    "HardRunnerFailure",
    "PromotionEvidence",
    "ResourceLease",
    "ResearchControllerRunner",
    "RunnerOutcome",
    "SOAK_STAGES",
    "SoakGate",
    "SoakStage",
    "UnknownRunnerOutcome",
    "build_benchmark_report",
    "build_trusted_benchmark_report",
    "derive_authoritative_benchmark_plan",
    "benchmark_controller_run_id",
    "benchmark_plan_from_campaign_snapshot",
    "bind_benchmark_campaign_snapshot",
    "child_budget_idempotency_key",
    "filter_feedback_at_cutoff",
    "execute_benchmark_campaign",
    "legacy_campaign_snapshot",
    "require_executable_benchmark_plan",
    "trusted_child_reservation",
    "trusted_resume_doctor",
]
