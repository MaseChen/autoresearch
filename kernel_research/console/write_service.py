"""Fixed DTO-to-domain command facade for trusted Console operations."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping
import uuid

from ..autorun.admin import AdminManifest
from ..autorun.controller import ResearchController
from ..autorun.deployment import DEPLOYMENT_BASELINE_FILENAME, DeploymentBaselinePin
from ..autorun.models import ControllerConfig
from ..campaign import cli as campaign_cli
from ..campaign.benchmark import (
    BenchmarkArm,
    BenchmarkCohortPlan,
    CURRENT_PROMPT_PROTOCOL_DIGEST,
)
from ..campaign.controller_runner import legacy_campaign_snapshot
from ..campaign.models import BudgetAmount, CampaignMode
from ..platform.identity import BaselineRef
from ..platform.profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
)
from .operations import (
    MAX_CAMPAIGN_CANDIDATES,
    MAX_CAMPAIGN_COST_MICROUSD,
    MAX_CAMPAIGN_MILLISECONDS,
    MAX_CAMPAIGN_TOKENS,
)



def _config(manifest: AdminManifest, profile: object) -> ControllerConfig:
    if profile == "pro":
        path = manifest.pro_config
    elif profile == "flash":
        path = manifest.flash_config
    else:
        raise ValueError("profile must be pro or flash")
    config = ControllerConfig.load(path)
    if config.repository_dir != manifest.repository_dir:
        raise ValueError("Console config does not belong to the Admin repository")
    return config


def _campaign_database(manifest: AdminManifest) -> Path:
    return manifest.runtime_root / "campaign" / "campaign.sqlite3"


def _campaign_id(prefix: str, operation_id: str) -> str:
    return prefix + "-" + uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"kernel-research/console/{prefix}/v1/{operation_id}",
    ).hex


def _controller_run_id(operation_id: str) -> str:
    return "console-child-" + uuid.uuid5(
        uuid.NAMESPACE_URL,
        "kernel-research/console-child/v1/" + operation_id,
    ).hex


def _proposer_ref(profile: object) -> dict[str, Any]:
    profile_id = {
        "pro": "opencode-deepseek-v4-pro",
        "flash": "opencode-deepseek-v4-flash",
    }.get(profile)
    if profile_id is None:
        raise ValueError("profile must be pro or flash")
    return BUILTIN_PROFILE_REGISTRY.get(
        kind="proposer", profile_id=profile_id, revision="v1"
    ).ref.to_dict()


def _budget(value: object) -> BudgetAmount:
    if not isinstance(value, dict) or set(value) != {
        "candidates",
        "wall_ms",
        "gpu_ms",
        "tokens",
        "cost_microusd",
    }:
        raise ValueError("budget must contain the exact five bounded axes")
    if any(type(item) is not int for item in value.values()):
        raise ValueError("budget axes must be integers")
    budget = BudgetAmount.from_mapping(value)
    bounds = {
        "candidates": MAX_CAMPAIGN_CANDIDATES,
        "wall_ms": MAX_CAMPAIGN_MILLISECONDS,
        "gpu_ms": MAX_CAMPAIGN_MILLISECONDS,
        "tokens": MAX_CAMPAIGN_TOKENS,
        "cost_microusd": MAX_CAMPAIGN_COST_MICROUSD,
    }
    for name, maximum in bounds.items():
        selected = getattr(budget, name)
        if not 1 <= selected <= maximum:
            raise ValueError(f"budget {name} is outside the Console bound")
    return budget


def _campaign_baseline(
    manifest: AdminManifest, config: ControllerConfig
) -> BaselineRef:
    pin = DeploymentBaselinePin.load(
        manifest.runtime_root / DEPLOYMENT_BASELINE_FILENAME
    )
    if (
        pin.namespace_id != CURRENT_RESEARCH_NAMESPACE.namespace_id
        or not pin.execution_environment.is_resolved
    ):
        raise ValueError("Console Campaign creation requires a resolved CURRENT pin")
    record = ResearchController(config)._best()
    if (
        record.id != pin.confirmation_experiment_id
        or record.experiment_uid != pin.confirmation_experiment_uid
        or record.artifact_id != str(pin.baseline_ref.artifact_id)
        or record.candidate_hash != pin.candidate_hash
    ):
        raise ValueError("Console Campaign seed differs from trusted deployment proof")
    return BaselineRef.create(
        namespace=CURRENT_RESEARCH_NAMESPACE,
        artifact_id=pin.baseline_ref.artifact_id,
        source="campaign",
        revision="deployment-seed-" + hashlib.sha256(
            (
                pin.confirmation_experiment_uid
                + "\x00"
                + pin.baseline_ref.revision
            ).encode("utf-8")
        ).hexdigest(),
        execution_environment=pin.execution_environment,
    )


def _write_json(root: Path, name: str, value: Mapping[str, Any]) -> Path:
    path = root / name
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _call_campaign(argv: list[str]) -> Mapping[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        return_code = campaign_cli.main(argv)
    if return_code != 0:
        detail = stderr.getvalue().strip()
        raise ValueError("trusted Campaign command failed: " + detail[:4096])
    try:
        value = json.loads(stdout.getvalue())
    except json.JSONDecodeError as exc:
        raise ValueError("trusted Campaign command returned invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("status") != "SUCCESS"
        or "result" not in value
    ):
        raise ValueError("trusted Campaign command returned an invalid envelope")
    result = value["result"]
    return result if isinstance(result, Mapping) else {"value": result}


def _campaign_create(
    manifest: AdminManifest,
    *,
    operation_id: str,
    parameters: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if parameters["mode"] != "DISCOVERY":
        raise ValueError("CAMPAIGN_CREATE supports Discovery only")
    config = _config(manifest, parameters["profile"])
    baseline = _campaign_baseline(manifest, config)
    budget = _budget(parameters["budget"])
    campaign_id = _campaign_id("console-discovery", operation_id)
    snapshot = legacy_campaign_snapshot(
        mode=CampaignMode.DISCOVERY,
        namespace=CURRENT_RESEARCH_NAMESPACE,
    )
    policy = {
        "schema_version": 1,
        "source": "console-deployment-seed-v1",
        "deployment_confirmation_revision": baseline.revision,
        "namespace_id": baseline.namespace_id,
        "environment_digest": baseline.execution_environment.digest,
    }
    with tempfile.TemporaryDirectory(
        prefix="console-campaign-", dir=manifest.runtime_root
    ) as temporary:
        root = Path(temporary).resolve()
        initial = _write_json(root, "baseline.json", baseline.to_dict())
        snapshot_path = _write_json(root, "snapshot.json", snapshot)
        policy_path = _write_json(root, "policy.json", policy)
        result = _call_campaign(
            [
                "create",
                "--database",
                str(_campaign_database(manifest)),
                "--campaign-id",
                campaign_id,
                "--namespace-id",
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
                "--mode",
                CampaignMode.DISCOVERY.value,
                "--snapshot",
                str(snapshot_path),
                "--initial-baseline-ref",
                str(initial),
                "--initial-policy-snapshot",
                str(policy_path),
                "--max-candidates",
                str(budget.candidates),
                "--max-wall-ms",
                str(budget.wall_ms),
                "--max-gpu-ms",
                str(budget.gpu_ms),
                "--max-tokens",
                str(budget.tokens),
                "--max-cost-microusd",
                str(budget.cost_microusd),
                "--allow-staged-lineage",
            ]
        )
    return {"campaign_id": campaign_id}, result


def _benchmark_init(
    manifest: AdminManifest,
    *,
    operation_id: str,
    parameters: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    config = _config(manifest, parameters["profile"])
    budget = _budget(parameters["budget"])
    arms_value = parameters["arms"]
    if (
        not isinstance(arms_value, list)
        or not 2 <= len(arms_value) <= 4
        or any(not isinstance(item, str) for item in arms_value)
        or len(set(arms_value)) != len(arms_value)
        or any(item not in {"pro", "flash"} for item in arms_value)
    ):
        raise ValueError("benchmark arms must be 2-4 unique active profiles")
    repetitions = parameters["repetitions"]
    if type(repetitions) is not int or not 1 <= repetitions <= 100:
        raise ValueError("benchmark repetitions must be between 1 and 100")
    campaign_id = _campaign_id("console-benchmark", operation_id)
    cohort_id = _campaign_id("cohort", operation_id)
    baseline = _campaign_baseline(manifest, config)
    draft = BenchmarkCohortPlan(
        cohort_id=cohort_id,
        namespace=CURRENT_RESEARCH_NAMESPACE,
        baseline=baseline,
        history_cutoff=0,
        prompt_protocol_digest=CURRENT_PROMPT_PROTOCOL_DIGEST,
        feedback_snapshot_digest="sha256:" + "0" * 64,
        arms=tuple(
            BenchmarkArm(
                arm_id=f"{profile}-{index}",
                proposer_profile=BUILTIN_PROFILE_REGISTRY.resolve(
                    _proposer_ref(profile)
                ).ref,
            )
            for index, profile in enumerate(arms_value)
        ),
        repetitions=repetitions,
    )
    policy = {
        "schema_version": 1,
        "source": "console-benchmark-v1",
        "operation_id": operation_id,
    }
    with tempfile.TemporaryDirectory(
        prefix="console-benchmark-", dir=manifest.runtime_root
    ) as temporary:
        root = Path(temporary).resolve()
        plan_path = _write_json(root, "plan.json", draft.snapshot)
        policy_path = _write_json(root, "policy.json", policy)
        result = _call_campaign(
            [
                "benchmark",
                "init",
                "--database",
                str(_campaign_database(manifest)),
                "--config",
                str(
                    manifest.pro_config
                    if parameters["profile"] == "pro"
                    else manifest.flash_config
                ),
                "--campaign-id",
                campaign_id,
                "--plan",
                str(plan_path),
                "--initial-policy-snapshot",
                str(policy_path),
                "--max-candidates",
                str(budget.candidates),
                "--max-wall-ms",
                str(budget.wall_ms),
                "--max-gpu-ms",
                str(budget.gpu_ms),
                "--max-tokens",
                str(budget.tokens),
                "--max-cost-microusd",
                str(budget.cost_microusd),
            ]
        )
    return {"campaign_id": campaign_id, "cohort_id": cohort_id}, result


def execute_domain_operation(
    *,
    manifest: AdminManifest,
    kind: str,
    operation_id: str,
    parameters: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Execute one already-validated operation without arbitrary argv/path input."""

    if kind == "RUN_START":
        controller = ResearchController(_config(manifest, parameters["profile"]))
        result = controller.start_console_operation(
            operation_id=operation_id,
            proposal_only=parameters["proposal_only"],
        )
        return {"run_id": result["id"]}, result
    if kind in {"RUN_STOP", "RUN_RESUME"}:
        controller = ResearchController(_config(manifest, parameters["profile"]))
        result = (
            controller.stop(parameters["run_id"])
            if kind == "RUN_STOP"
            else controller.resume(parameters["run_id"])
        )
        return {"run_id": parameters["run_id"]}, result
    if kind == "CAMPAIGN_CREATE":
        return _campaign_create(
            manifest, operation_id=operation_id, parameters=parameters
        )
    if kind == "BENCHMARK_INIT":
        return _benchmark_init(
            manifest, operation_id=operation_id, parameters=parameters
        )

    campaign_id = parameters.get("campaign_id")
    if not isinstance(campaign_id, str):
        raise ValueError("Campaign operation is missing campaign_id")
    database = str(_campaign_database(manifest))
    if kind == "CAMPAIGN_START":
        result = _call_campaign(
            ["start", "--database", database, "--campaign-id", campaign_id]
        )
    elif kind == "CAMPAIGN_PAUSE":
        result = _call_campaign(
            [
                "pause",
                "--database",
                database,
                "--campaign-id",
                campaign_id,
                "--status",
                "PAUSED_OPERATOR",
                "--reason",
                parameters["reason"],
            ]
        )
    elif kind == "CAMPAIGN_RESUME":
        config_path = (
            manifest.pro_config
            if parameters["profile"] == "pro"
            else manifest.flash_config
        )
        result = _call_campaign(
            [
                "resume",
                "--database",
                database,
                "--campaign-id",
                campaign_id,
                "--config",
                str(config_path),
                "--resource-id",
                "gpu1",
            ]
        )
    elif kind == "CAMPAIGN_CHILD_EXECUTE":
        config_path = (
            manifest.pro_config
            if parameters["profile"] == "pro"
            else manifest.flash_config
        )
        with tempfile.TemporaryDirectory(
            prefix="console-child-", dir=manifest.runtime_root
        ) as temporary:
            profile_path = _write_json(
                Path(temporary).resolve(),
                "proposer.json",
                _proposer_ref(parameters["profile"]),
            )
            result = _call_campaign(
                [
                    "child",
                    "execute",
                    "--database",
                    database,
                    "--config",
                    str(config_path),
                    "--campaign-id",
                    campaign_id,
                    "--controller-run-id",
                    _controller_run_id(operation_id),
                    "--proposer-profile",
                    str(profile_path),
                    "--resource-id",
                    "gpu1",
                    "--max-candidates",
                    str(parameters["max_candidates"]),
                    "--max-wall-seconds",
                    str(parameters["max_wall_seconds"]),
                ]
            )
    elif kind == "CAMPAIGN_LINEAGE_ADVANCE":
        # The Supervisor derives every metric/evidence field from History and
        # Controller; the browser supplies only the child identity.
        config_path = manifest.pro_config
        result = _call_campaign(
            [
                "lineage",
                "advance",
                "--database",
                database,
                "--config",
                str(config_path),
                "--campaign-id",
                campaign_id,
                "--child-id",
                str(parameters["child_id"]),
                "--resource-id",
                "gpu1",
                "--idempotency-key",
                "console-lineage:" + hashlib.sha256(operation_id.encode()).hexdigest(),
            ]
        )
    elif kind == "BENCHMARK_EXECUTE":
        config_path = (
            manifest.pro_config
            if parameters["profile"] == "pro"
            else manifest.flash_config
        )
        result = _call_campaign(
            [
                "benchmark",
                "execute",
                "--database",
                database,
                "--config",
                str(config_path),
                "--campaign-id",
                campaign_id,
                "--resource-id",
                "gpu1",
            ]
        )
    else:
        raise ValueError("operation kind is not implemented by the trusted facade")
    return {"campaign_id": campaign_id}, result


__all__ = ["execute_domain_operation"]
