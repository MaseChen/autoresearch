"""Durable, evidence-driven qualification gates for Campaign soaks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Protocol

from .soak_collector import SoakObservation, SoakObservationCollector
from .store import (
    CampaignStore,
    SOAK_MAX_HEARTBEAT_GAP_SECONDS,
    SOAK_STAGE_ORDER,
    SOAK_STAGE_REQUIRED_SECONDS,
)


@dataclass(frozen=True, slots=True)
class SoakStage:
    """One fixed release-qualification phase."""

    stage_id: str
    required_seconds: int


SOAK_STAGES = tuple(
    SoakStage(stage_id, SOAK_STAGE_REQUIRED_SECONDS[stage_id])
    for stage_id in SOAK_STAGE_ORDER
)


class _Collector(Protocol):
    def collect(
        self, *, interval_start_epoch: float, interval_end_epoch: float
    ) -> SoakObservation: ...


class SoakGate:
    """Advance 24/72/168-hour gates only from trusted observations."""

    def __init__(
        self,
        store: CampaignStore,
        *,
        gate_id: str,
        collector: _Collector,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(gate_id, str) or not gate_id:
            raise ValueError("gate_id must be a non-empty string")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if not hasattr(collector, "collect") or not callable(collector.collect):
            raise TypeError("collector must provide trusted soak observations")
        self.store = store
        self.gate_id = gate_id
        self.collector = collector
        self._clock = clock

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("clock must return a finite non-negative number")
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError("clock must return a finite non-negative number")
        return result

    def start(self, *, stage: str | None = None) -> dict[str, object]:
        """Start the next phase; the first observation earns no retroactive time."""

        now = self._now()
        observation = self.collector.collect(
            interval_start_epoch=now,
            interval_end_epoch=now,
        )
        active = self.store.get_active_soak_generation(self.gate_id)
        if (
            active is not None
            and active["invariant_snapshot_digest"]
            != observation.invariant_snapshot_digest
        ):
            expected = SOAK_STAGE_ORDER[0]
        elif active is not None:
            expected = str(active["stage"])
        else:
            completed = self.store.completed_soak_stages(
                self.gate_id, observation.invariant_snapshot_digest
            )
            expected = next(
                (item for item in SOAK_STAGE_ORDER if item not in completed),
                None,
            )
        if expected is None:
            raise ValueError("all soak stages are already complete for this invariant")
        selected = expected if stage is None else stage
        if selected != expected:
            raise ValueError(f"soak stages must run in order; expected {expected}")
        return self.store.begin_soak_generation(
            gate_id=self.gate_id,
            stage=selected,
            observation=observation,
        )

    def heartbeat(self) -> dict[str, object]:
        """Collect and apply one bounded interval; callers cannot supply counts."""

        anchor = self.store.get_soak_heartbeat_anchor(self.gate_id)
        if anchor is None:
            raise ValueError("soak gate has no active generation")
        previous = float(anchor["last_heartbeat_epoch"])
        now = self._now()
        interval_start = previous if now >= previous else now
        observation = self.collector.collect(
            interval_start_epoch=interval_start,
            interval_end_epoch=now,
        )
        return self.store.record_soak_heartbeat(
            gate_id=self.gate_id,
            observation=observation,
        )

    def status(self) -> dict[str, object]:
        """Return status under the invariant observed now, without accruing time."""

        now = self._now()
        observation = self.collector.collect(
            interval_start_epoch=now,
            interval_end_epoch=now,
        )
        digest = observation.invariant_snapshot_digest
        completed = self.store.completed_soak_stages(self.gate_id, digest)
        active = self.store.get_active_soak_generation(self.gate_id)
        first_incomplete = next(
            (item for item in SOAK_STAGE_ORDER if item not in completed),
            None,
        )
        generations = self.store.list_soak_generations(self.gate_id)
        violations = self.store.list_soak_violations(self.gate_id)
        observations = self.store.list_soak_observations(self.gate_id)
        return {
            "schema_version": 2,
            "gate_id": self.gate_id,
            "collector_revision": observation.collector_revision,
            "current_invariant_snapshot_digest": digest,
            "current_observation_status": observation.status,
            "current_unavailable_reasons": list(
                observation.unavailable_reasons
            ),
            "completed_stages": list(completed),
            "active_generation": active,
            "active_invariant_matches": (
                active is None
                or active["invariant_snapshot_digest"] == digest
            ),
            "eligible_stage": first_incomplete if active is None else None,
            "pending_stage": first_incomplete,
            "profiling_allowed": (
                observation.status == "AVAILABLE"
                and not any(observation.counts.values())
                and self.store.soak_profiling_allowed(self.gate_id, digest)
            ),
            "maximum_heartbeat_gap_seconds": (
                SOAK_MAX_HEARTBEAT_GAP_SECONDS
            ),
            "generation_count": len(generations),
            "violation_count": len(violations),
            "observation_count": len(observations),
        }

    def profiling_allowed(self) -> bool:
        """Whether all three stages completed under the current invariant."""

        now = self._now()
        observation = self.collector.collect(
            interval_start_epoch=now,
            interval_end_epoch=now,
        )
        return (
            observation.status == "AVAILABLE"
            and not any(observation.counts.values())
            and self.store.soak_profiling_allowed(
                self.gate_id, observation.invariant_snapshot_digest
            )
        )


__all__ = [
    "SOAK_MAX_HEARTBEAT_GAP_SECONDS",
    "SOAK_STAGES",
    "SoakGate",
    "SoakObservationCollector",
    "SoakStage",
]
