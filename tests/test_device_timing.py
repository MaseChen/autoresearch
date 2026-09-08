from __future__ import annotations

import unittest

from kernel_research.device_timing import (
    DEVICE_EVENT_LAUNCHES_PER_ROUND,
    DEVICE_EVENT_MEASUREMENT_ROUNDS,
    DEVICE_EVENT_WARMUP_ITERATIONS,
    benchmark_device_event_interleaved,
    device_event_protocol_snapshot,
)


class FakeEvent:
    def __init__(self, api, *, enable_timing: bool) -> None:
        if not enable_timing:
            raise AssertionError("timing must be enabled")
        self.api = api
        self.value = None

    def record(self) -> None:
        self.value = self.api.clock

    def elapsed_time(self, other) -> float:
        return float(other.value - self.value)


class FakeAccelerator:
    def __init__(self) -> None:
        self.clock = 0.0
        self.synchronize_calls = 0

    def is_available(self) -> bool:
        return True

    def Event(self, *, enable_timing: bool):
        return FakeEvent(self, enable_timing=enable_timing)

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeAccelerator()


class DeviceTimingTests(unittest.TestCase):
    def test_balanced_rounds_use_device_events_and_normalize_blocks(self) -> None:
        torch = FakeTorch()
        calls = {"candidate": 0, "incumbent": 0}

        def candidate() -> None:
            calls["candidate"] += 1
            torch.cuda.clock += 2.0

        def incumbent() -> None:
            calls["incumbent"] += 1
            torch.cuda.clock += 3.0

        result = benchmark_device_event_interleaved(
            torch, candidate, (), incumbent, ()
        )
        expected_calls = (
            DEVICE_EVENT_WARMUP_ITERATIONS
            + DEVICE_EVENT_MEASUREMENT_ROUNDS * DEVICE_EVENT_LAUNCHES_PER_ROUND
        )
        self.assertEqual(calls, {"candidate": expected_calls, "incumbent": expected_calls})
        self.assertEqual(
            [round_.order for round_ in result.rounds],
            ["AB", "BA", "AB", "BA", "AB", "BA"],
        )
        self.assertEqual(result.candidate_median_ms, 2.0)
        self.assertEqual(result.incumbent_median_ms, 3.0)
        self.assertEqual(result.candidate_mad_ms, 0.0)
        self.assertEqual(result.incumbent_mad_ms, 0.0)
        self.assertEqual(torch.cuda.synchronize_calls, 13)

    def test_protocol_snapshot_is_frozen_and_digest_bound(self) -> None:
        first = device_event_protocol_snapshot()
        second = device_event_protocol_snapshot()
        self.assertEqual(first, second)
        self.assertEqual(first["measurement_rounds"], 6)
        self.assertEqual(first["launches_per_round"], 20)
        self.assertFalse(first["compile_time_included"])

    def test_missing_events_and_nonpositive_time_fail_closed(self) -> None:
        class NoAccelerator:
            cuda = None

        with self.assertRaisesRegex(RuntimeError, "event API"):
            benchmark_device_event_interleaved(
                NoAccelerator(), lambda: None, (), lambda: None, ()
            )

        torch = FakeTorch()
        with self.assertRaisesRegex(ValueError, "positive"):
            benchmark_device_event_interleaved(
                torch, lambda: None, (), lambda: None, ()
            )


if __name__ == "__main__":
    unittest.main()
