"""Balanced device-event timing primitives for scoring protocol V2."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Callable, Sequence

from .platform.canonical import canonical_sha256


DEVICE_EVENT_WARMUP_ITERATIONS = 10
DEVICE_EVENT_MEASUREMENT_ROUNDS = 6
DEVICE_EVENT_LAUNCHES_PER_ROUND = 20
DEVICE_EVENT_TIMING_PROTOCOL_ID = "c500-device-event-paired-v1"


@dataclass(frozen=True, slots=True)
class DeviceEventRound:
    round_index: int
    order: str
    candidate_latency_ms: float
    incumbent_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "order": self.order,
            "candidate_latency_ms": self.candidate_latency_ms,
            "incumbent_latency_ms": self.incumbent_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class DeviceEventMeasurement:
    rounds: tuple[DeviceEventRound, ...]
    warmup_iterations: int = DEVICE_EVENT_WARMUP_ITERATIONS
    launches_per_round: int = DEVICE_EVENT_LAUNCHES_PER_ROUND
    protocol_id: str = DEVICE_EVENT_TIMING_PROTOCOL_ID

    def __post_init__(self) -> None:
        if len(self.rounds) != DEVICE_EVENT_MEASUREMENT_ROUNDS:
            raise ValueError("device-event measurement has the wrong round count")
        if tuple(round_.round_index for round_ in self.rounds) != tuple(
            range(DEVICE_EVENT_MEASUREMENT_ROUNDS)
        ):
            raise ValueError("device-event rounds must be contiguous and ordered")
        orders = tuple(round_.order for round_ in self.rounds)
        if orders != ("AB", "BA", "AB", "BA", "AB", "BA"):
            raise ValueError("device-event rounds must use the frozen AB/BA order")

    @property
    def candidate_median_ms(self) -> float:
        return statistics.median(
            round_.candidate_latency_ms for round_ in self.rounds
        )

    @property
    def incumbent_median_ms(self) -> float:
        return statistics.median(
            round_.incumbent_latency_ms for round_ in self.rounds
        )

    @property
    def candidate_mad_ms(self) -> float:
        center = self.candidate_median_ms
        return statistics.median(
            abs(round_.candidate_latency_ms - center) for round_ in self.rounds
        )

    @property
    def incumbent_mad_ms(self) -> float:
        center = self.incumbent_median_ms
        return statistics.median(
            abs(round_.incumbent_latency_ms - center) for round_ in self.rounds
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "warmup_iterations": self.warmup_iterations,
            "measurement_rounds": DEVICE_EVENT_MEASUREMENT_ROUNDS,
            "launches_per_round": self.launches_per_round,
            "candidate_median_ms": self.candidate_median_ms,
            "candidate_mad_ms": self.candidate_mad_ms,
            "incumbent_median_ms": self.incumbent_median_ms,
            "incumbent_mad_ms": self.incumbent_mad_ms,
            "rounds": [round_.to_dict() for round_ in self.rounds],
        }


def device_event_protocol_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "protocol_id": DEVICE_EVENT_TIMING_PROTOCOL_ID,
        "clock": "accelerator-device-event",
        "cache_policy": "warm-cache-steady-state",
        "l2_flush": False,
        "warmup_iterations": DEVICE_EVENT_WARMUP_ITERATIONS,
        "measurement_rounds": DEVICE_EVENT_MEASUREMENT_ROUNDS,
        "launches_per_round": DEVICE_EVENT_LAUNCHES_PER_ROUND,
        "round_order": ["AB", "BA", "AB", "BA", "AB", "BA"],
        "compile_time_included": False,
    }
    return {**snapshot, "digest": canonical_sha256(snapshot)}


def _accelerator_api(torch: Any) -> Any:
    for name in ("cuda", "maca"):
        api = getattr(torch, name, None)
        if api is None:
            continue
        try:
            if bool(api.is_available()):
                return api
        except Exception as exc:
            raise RuntimeError(f"torch.{name}.is_available failed") from exc
    raise RuntimeError("no available accelerator event API")


def _event_block_ms(
    api: Any,
    function: Callable[..., Any],
    arguments: Sequence[Any],
) -> float:
    try:
        start = api.Event(enable_timing=True)
        end = api.Event(enable_timing=True)
    except Exception as exc:
        raise RuntimeError("accelerator event construction failed") from exc
    start.record()
    for _ in range(DEVICE_EVENT_LAUNCHES_PER_ROUND):
        function(*arguments)
    end.record()
    api.synchronize()
    try:
        elapsed = float(start.elapsed_time(end))
    except Exception as exc:
        raise RuntimeError("accelerator event elapsed-time query failed") from exc
    normalized = elapsed / DEVICE_EVENT_LAUNCHES_PER_ROUND
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("device-event latency must be finite and positive")
    return normalized


def benchmark_device_event_interleaved(
    torch: Any,
    candidate: Callable[..., Any],
    candidate_arguments: Sequence[Any],
    incumbent: Callable[..., Any],
    incumbent_arguments: Sequence[Any],
) -> DeviceEventMeasurement:
    """Measure candidate and incumbent in the frozen balanced round order."""

    api = _accelerator_api(torch)
    for iteration in range(DEVICE_EVENT_WARMUP_ITERATIONS):
        if iteration % 2 == 0:
            incumbent(*incumbent_arguments)
            candidate(*candidate_arguments)
        else:
            candidate(*candidate_arguments)
            incumbent(*incumbent_arguments)
    api.synchronize()

    rounds: list[DeviceEventRound] = []
    for round_index in range(DEVICE_EVENT_MEASUREMENT_ROUNDS):
        if round_index % 2 == 0:
            candidate_ms = _event_block_ms(api, candidate, candidate_arguments)
            incumbent_ms = _event_block_ms(api, incumbent, incumbent_arguments)
            order = "AB"
        else:
            incumbent_ms = _event_block_ms(api, incumbent, incumbent_arguments)
            candidate_ms = _event_block_ms(api, candidate, candidate_arguments)
            order = "BA"
        rounds.append(
            DeviceEventRound(
                round_index=round_index,
                order=order,
                candidate_latency_ms=candidate_ms,
                incumbent_latency_ms=incumbent_ms,
            )
        )
    return DeviceEventMeasurement(rounds=tuple(rounds))


__all__ = [
    "DEVICE_EVENT_LAUNCHES_PER_ROUND",
    "DEVICE_EVENT_MEASUREMENT_ROUNDS",
    "DEVICE_EVENT_TIMING_PROTOCOL_ID",
    "DEVICE_EVENT_WARMUP_ITERATIONS",
    "DeviceEventMeasurement",
    "DeviceEventRound",
    "benchmark_device_event_interleaved",
    "device_event_protocol_snapshot",
]
