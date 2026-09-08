from __future__ import annotations

import copy
import unittest

from kernel_research.device_timing import DeviceEventMeasurement, DeviceEventRound
from kernel_research.scoring_measurement import (
    scoring_baseline_measurement_contract_snapshot,
    validate_device_event_measurement,
)


def measurement(candidate: float = 1.0, incumbent: float = 3.0) -> dict:
    return DeviceEventMeasurement(
        rounds=tuple(
            DeviceEventRound(
                round_index=index,
                order="AB" if index % 2 == 0 else "BA",
                candidate_latency_ms=candidate,
                incumbent_latency_ms=incumbent,
            )
            for index in range(6)
        )
    ).to_dict()


class ScoringMeasurementTests(unittest.TestCase):
    def test_contract_is_canonical_and_role_separated(self) -> None:
        first = scoring_baseline_measurement_contract_snapshot()
        self.assertEqual(first, scoring_baseline_measurement_contract_snapshot())
        self.assertEqual(
            first["anchor_pairing"],
            "compiled-reference-vs-compiled-reference",
        )
        self.assertEqual(
            first["performance_pairing"],
            "compiled-reference-vs-eager-reference",
        )
        self.assertEqual(first["anchor_channel_count"], 2)
        self.assertEqual(first["anchor_blocks_per_channel"], 6)
        self.assertEqual(
            first["phase_order"],
            "all-case-anchors-before-performance-proof",
        )
        self.assertEqual(
            first["case_memory_policy"],
            "regenerate-and-release-each-case",
        )
        self.assertEqual(
            first["performance_role"],
            "qualification-only-non-regression-proof",
        )
        self.assertRegex(str(first["digest"]), r"^sha256:[0-9a-f]{64}$")

    def test_measurement_recomputes_zero_mad_and_combined_median(self) -> None:
        candidate, incumbent, combined = validate_device_event_measurement(
            measurement(), field="measurement"
        )
        self.assertEqual((candidate, incumbent, combined), (1.0, 3.0, 2.0))

    def test_measurement_rejects_protocol_round_and_statistic_drift(self) -> None:
        cases = [None, {}, {**measurement(), "unexpected": True}]
        changed = copy.deepcopy(measurement())
        changed["protocol_id"] = "other"
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["rounds"] = "not-an-array"
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["rounds"].pop()
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["rounds"][0] = None
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["rounds"][0]["unexpected"] = True
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["rounds"][0]["round_index"] = 1
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["candidate_median_ms"] = 2.0
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["rounds"][0]["order"] = "BA"
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["rounds"][0]["candidate_latency_ms"] = 0.0
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["rounds"][0]["candidate_latency_ms"] = True
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["rounds"][0]["incumbent_latency_ms"] = float("inf")
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["candidate_mad_ms"] = -1.0
        cases.append(changed)
        changed = copy.deepcopy(measurement())
        changed["incumbent_mad_ms"] = float("nan")
        cases.append(changed)
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    validate_device_event_measurement(value, field="measurement")


if __name__ == "__main__":
    unittest.main()
