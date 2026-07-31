"""Bounded scientific summaries for prompts and public status output."""

from __future__ import annotations

from typing import Any, Mapping

from ..constants import MAX_FEEDBACK_ERROR_CHARS


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bounded_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= MAX_FEEDBACK_ERROR_CHARS:
        return text
    return text[: MAX_FEEDBACK_ERROR_CHARS - 3] + "..."


def summarize_result(
    result: Mapping[str, Any] | None,
    *,
    error: Any = None,
) -> dict[str, Any]:
    """Return the small, sample-free result projection used by the controller."""

    value = _mapping(result)
    promotion = _mapping(value.get("promotion"))
    decision = _mapping(promotion.get("decision"))
    global_score = promotion.get("global_score")
    if global_score is None:
        global_score = value.get("aggregate_score")
    promotion_reason = decision.get("reason")
    if promotion_reason is None:
        promotion_reason = promotion.get("reason")

    cases: list[dict[str, Any]] = []
    cases_value = value.get("cases")
    if isinstance(cases_value, list):
        for case_value in cases_value:
            case = _mapping(case_value)
            cases.append(
                {
                    "case_id": case.get("case_id"),
                    "status": case.get("status"),
                    "matched_ratio": case.get("matched_ratio"),
                    "p50_us": case.get("p50_us"),
                    "baseline_p50_us": case.get("baseline_p50_us"),
                    "error": _bounded_text(case.get("error")),
                }
            )

    summary_error = error
    if summary_error is None:
        summary_error = value.get("error")
    primary_speedup = decision.get("aggregate_speedup")
    confirmation_speedup = decision.get("confirmation_speedup")
    relative_speedup = (
        confirmation_speedup
        if confirmation_speedup is not None
        else primary_speedup
    )
    speedups = [
        float(value)
        for value in (primary_speedup, confirmation_speedup)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    conservative_speedup = min(speedups) if speedups else None
    primary_regression = decision.get("worst_case_regression")
    confirmation_regression = decision.get(
        "confirmation_worst_case_regression"
    )
    current_regression = (
        confirmation_regression
        if confirmation_regression is not None
        else primary_regression
    )
    return {
        "status": value.get("status"),
        "error": _bounded_text(summary_error),
        "promotion_phase": promotion.get("phase"),
        "promotion_reason": promotion_reason,
        "global_normalized_score": global_score,
        "relative_speedup_vs_accepted": relative_speedup,
        "primary_speedup_vs_accepted": primary_speedup,
        "confirmation_speedup_vs_accepted": confirmation_speedup,
        "conservative_speedup_vs_accepted": conservative_speedup,
        "worst_case_regression": current_regression,
        "primary_worst_case_regression": primary_regression,
        "confirmation_worst_case_regression": confirmation_regression,
        "per_case_speedups": decision.get("per_case_speedups"),
        "confirmation_case_speedups": decision.get(
            "confirmation_case_speedups"
        ),
        "cases": cases,
    }


def feedback_for_iteration(iteration: Mapping[str, Any]) -> dict[str, Any]:
    """Project one completed iteration into bounded proposer feedback."""

    return {
        "run_id": iteration.get("run_id"),
        "iteration_index": iteration.get("iteration_index"),
        "parent_hash": iteration.get("parent_hash"),
        "candidate_hash": iteration.get("candidate_hash"),
        "hypothesis": iteration.get("hypothesis"),
        "outcome": iteration.get("outcome"),
        "result_summary": summarize_result(
            _mapping(iteration.get("result")),
            error=iteration.get("error"),
        ),
    }


def compact_status_payload(
    payload: Mapping[str, Any],
    *,
    command: str = "status",
) -> dict[str, Any]:
    """Remove bulky immutable preflight/config data from a status response."""

    run_value = payload.get("run")
    run = _mapping(run_value) if run_value is not None else None
    compact_run: dict[str, Any] | None = None
    if run is not None:
        config = _mapping(run.get("config"))
        compact_run = {
            "id": run.get("id"),
            "status": run.get("status"),
            "valid_candidates": run.get("valid_candidates"),
            "consecutive_failures": run.get("consecutive_failures"),
            "max_candidates": config.get("max_candidates"),
            "max_hours": config.get("max_hours"),
            "deadline_epoch": run.get("deadline_epoch"),
            "stop_reason": run.get("stop_reason"),
            "stop_requested": run.get("stop_requested"),
            "initial_best_hash": run.get("initial_best_hash"),
            "final_best_hash": run.get("final_best_hash"),
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
        }

    compact_iterations = []
    iterations = payload.get("iterations")
    if isinstance(iterations, list):
        for iteration_value in iterations:
            iteration = _mapping(iteration_value)
            compact_iterations.append(
                {
                    "iteration_index": iteration.get("iteration_index"),
                    "stage": iteration.get("stage"),
                    "status": iteration.get("status"),
                    "outcome": iteration.get("outcome"),
                    "candidate_hash": iteration.get("candidate_hash"),
                    "hypothesis": iteration.get("hypothesis"),
                    "experiment_ids": iteration.get("experiment_ids"),
                    "active_container": iteration.get("active_container"),
                    "error": _bounded_text(iteration.get("error")),
                    "result_summary": iteration.get("result_summary"),
                }
            )
    return {
        "schema_version": 1,
        "command": command,
        "format": "compact",
        "status": payload.get("status"),
        "run": compact_run,
        "iterations": compact_iterations,
    }
