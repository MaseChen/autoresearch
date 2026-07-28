from __future__ import annotations

import math
import unittest

import numpy as np

from kernel_research.cases import (
    FIXED_SEED,
    FULL_CASES,
    QUICK_CASES,
    SMOKE_CASES,
    CaseSpec,
    DataSet,
    generate_case,
    generate_expert_ids,
    get_suite,
)
from kernel_research.reference import (
    bfloat16_round,
    check_precision,
    fused_moe_reference,
    inputs_unchanged,
    reference_self_check,
    snapshot_inputs,
)
from kernel_research.scoring import (
    evaluate_promotion,
    geometric_mean_speedup,
    latency_percentiles,
    percentiles,
)


READ_ONLY_FIELDS = (
    "a",
    "b_col_major",
    "scale_a",
    "scale_b",
    "moe_weights",
    "token_ids",
    "expert_ids",
)


def make_dataset(
    *,
    em: int = 256,
    n: int = 2,
    k: int = 4,
    experts: int = 3,
) -> DataSet:
    spec = CaseSpec("manual", em, n, k, experts)
    return DataSet(
        spec=spec,
        a=np.zeros((em, k), dtype=np.int8),
        b_col_major=np.zeros((experts, n, k), dtype=np.int8),
        scale_a=np.ones(em, dtype=np.float32),
        scale_b=np.ones((experts, n), dtype=np.float32),
        moe_weights=np.ones(em, dtype=np.float32),
        token_ids=np.arange(em, dtype=np.int32),
        expert_ids=np.zeros(em // 128, dtype=np.int32),
        topk=8,
        out=np.zeros((em, n), dtype=np.float32),
    )


class CaseGenerationTests(unittest.TestCase):
    def test_suite_shapes_are_fixed(self) -> None:
        self.assertEqual(
            [(case.em, case.n, case.k, case.num_experts) for case in SMOKE_CASES],
            [(256, 128, 224, 4), (512, 224, 64, 8)],
        )
        self.assertEqual(len(QUICK_CASES), 4)
        self.assertEqual(
            [(case.em, case.n, case.k, case.num_experts) for case in FULL_CASES],
            [
                (4096, 4096, 7168, 256),
                (32768, 4096, 7168, 256),
                (4096, 7168, 2048, 256),
                (32768, 7168, 2048, 256),
            ],
        )
        self.assertIs(get_suite("smoke"), SMOKE_CASES)
        with self.assertRaises(ValueError):
            get_suite("unknown")

    def test_fixed_shape_address_ranges_only_require_int64_for_full_b(self) -> None:
        int32_max = np.iinfo(np.int32).max

        quick_b_offsets = [
            case.num_experts * case.n * case.k - 1 for case in QUICK_CASES
        ]
        full_b_offsets = [
            case.num_experts * case.n * case.k - 1 for case in FULL_CASES
        ]
        self.assertTrue(all(offset <= int32_max for offset in quick_b_offsets))
        self.assertTrue(all(offset > int32_max for offset in full_b_offsets))

        # Other fixed-shape linear offsets remain within int32. This protects
        # the deliberate minimal fix: only B's expert base needs promotion.
        for case in FULL_CASES:
            self.assertLessEqual(case.em * case.k - 1, int32_max)  # A
            self.assertLessEqual(case.num_experts * case.n - 1, int32_max)  # scale B
            self.assertLessEqual(case.em * case.n - 1, int32_max)  # out

    def test_generation_is_deterministic_and_inputs_are_pre_routed(self) -> None:
        first = generate_case(SMOKE_CASES[0])
        second = generate_case(SMOKE_CASES[0], seed=FIXED_SEED)
        for field in (*READ_ONLY_FIELDS, "out"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))

        # All flattened route ids with the same token quotient were gathered
        # from one raw token row before this DataSet was returned.
        seen: dict[int, tuple[np.ndarray, np.float32]] = {}
        for row, route_id in enumerate(first.token_ids):
            token = int(route_id) // first.topk
            if token in seen:
                expected_a, expected_scale = seen[token]
                np.testing.assert_array_equal(first.a[row], expected_a)
                self.assertEqual(first.scale_a[row], expected_scale)
            else:
                seen[token] = (first.a[row].copy(), first.scale_a[row])

    def test_uniform_and_zipf_tile_experts_are_valid_sparse_and_repeatable(self) -> None:
        uniform = generate_expert_ids(4096, 32, "uniform", seed=FIXED_SEED)
        zipf = generate_expert_ids(4096, 32, "zipf", seed=FIXED_SEED)
        zipf_again = generate_expert_ids(4096, 32, "skewed", seed=FIXED_SEED)

        self.assertEqual(uniform.dtype, np.int32)
        self.assertTrue(np.all((uniform >= 0) & (uniform < 32)))
        self.assertTrue(np.all((zipf >= 0) & (zipf < 32)))
        np.testing.assert_array_equal(zipf, zipf_again)
        self.assertEqual(zipf[0], zipf[1])
        self.assertGreater(np.count_nonzero(zipf == 0), np.count_nonzero(uniform == 0))

        sparse = generate_expert_ids(4, 8, "zipf", seed=FIXED_SEED)
        self.assertLess(len(np.unique(sparse)), 8)  # at least one zero-hit expert
        self.assertLess(len(np.unique(sparse)), len(sparse))  # repeated expert tile


class ReferenceTests(unittest.TestCase):
    def test_bfloat16_round_to_nearest_even_and_special_values(self) -> None:
        # Around 1.0, bf16 spacing is 2**-7.  These are exact halfway cases.
        values = np.array(
            [
                1.0 + 2.0**-8,
                1.0 + 3.0 * 2.0**-8,
                -0.0,
                np.inf,
                -np.inf,
                np.nan,
            ],
            dtype=np.float32,
        )
        rounded = bfloat16_round(values)
        self.assertEqual(rounded[0], np.float32(1.0))
        self.assertEqual(rounded[1], np.float32(1.0 + 2.0**-6))
        self.assertTrue(np.signbit(rounded[2]))
        self.assertTrue(np.isposinf(rounded[3]))
        self.assertTrue(np.isneginf(rounded[4]))
        self.assertTrue(np.isnan(rounded[5]))
        self.assertTrue(np.all((rounded[:-1].view(np.uint32) & 0xFFFF) == 0))

    def test_bfloat16_all_float32_subnormals_match_integer_rne(self) -> None:
        chunk_size = 1 << 16
        for sign in (np.uint32(0), np.uint32(0x80000000)):
            for start in range(1, 1 << 23, chunk_size):
                mantissa = np.arange(
                    start,
                    min(start + chunk_size, 1 << 23),
                    dtype=np.uint32,
                )
                bits = mantissa | sign
                actual = bfloat16_round(
                    bits.view(np.float32)
                ).view(np.uint32)
                upper = bits >> np.uint32(16)
                remainder = bits & np.uint32(0xFFFF)
                round_up = (remainder > np.uint32(0x8000)) | (
                    (remainder == np.uint32(0x8000))
                    & ((upper & np.uint32(1)) == np.uint32(1))
                )
                expected = (
                    upper + round_up.astype(np.uint32)
                ) << np.uint32(16)
                np.testing.assert_array_equal(actual, expected)

    def test_tile_mapping_formula_and_read_only_inputs(self) -> None:
        dataset = make_dataset(em=256, n=2, k=4, experts=3)
        dataset.a[:] = np.array([1, -2, 3, -4], dtype=np.int8)
        dataset.b_col_major[0] = np.array(
            [[2, 1, -1, 3], [-3, 2, 4, 1]], dtype=np.int8
        )
        dataset.b_col_major[2] = np.array(
            [[1, 1, 1, 1], [4, -2, 0, 1]], dtype=np.int8
        )
        dataset.scale_a[:] = np.float32(0.5)
        dataset.scale_b[0] = np.array([0.25, 0.5], dtype=np.float32)
        dataset.scale_b[2] = np.array([0.75, 0.125], dtype=np.float32)
        dataset.moe_weights[:] = np.float32(0.8)
        dataset.expert_ids[:] = np.array([0, 2], dtype=np.int32)

        before = snapshot_inputs(dataset)
        result = fused_moe_reference(dataset)
        self.assertTrue(inputs_unchanged(dataset, before))
        np.testing.assert_array_equal(dataset.out, np.zeros_like(dataset.out))

        for row in (0, 127, 128, 255):
            expert = 0 if row < 128 else 2
            accumulator = dataset.a[row].astype(np.int32) @ dataset.b_col_major[
                expert
            ].astype(np.int32).T
            expected = bfloat16_round(
                accumulator.astype(np.float32)
                * dataset.scale_a[row]
                * dataset.scale_b[expert]
                * dataset.moe_weights[row]
            )
            np.testing.assert_array_equal(result[row], expected)

        returned = fused_moe_reference(dataset, out=dataset.out)
        self.assertIs(returned, dataset.out)
        np.testing.assert_array_equal(returned, result)
        self.assertTrue(inputs_unchanged(dataset, before))

    def test_int8_extremes_accumulate_in_int32(self) -> None:
        dataset = make_dataset(em=128, n=2, k=256, experts=2)
        dataset.a[:, :] = np.int8(-128)
        dataset.b_col_major[0, 0, :] = np.int8(-128)
        dataset.b_col_major[0, 1, :] = np.int8(127)
        dataset.expert_ids[:] = 0

        result = fused_moe_reference(dataset)
        exact = np.array(
            [256 * 128 * 128, 256 * -128 * 127], dtype=np.int32
        ).astype(np.float32)
        expected = bfloat16_round(exact)
        np.testing.assert_array_equal(result[0], expected)
        np.testing.assert_array_equal(result[-1], expected)

    def test_all_zero_inputs_produce_zero_without_mutation(self) -> None:
        dataset = make_dataset(em=128, n=3, k=16, experts=2)
        dataset.scale_a[:] = 0.0
        dataset.scale_b[:] = 0.0
        dataset.moe_weights[:] = 0.0
        before = snapshot_inputs(dataset)

        result = fused_moe_reference(dataset)

        np.testing.assert_array_equal(result, np.zeros_like(result))
        self.assertTrue(inputs_unchanged(dataset, before))

    def test_invalid_shapes_and_expert_ids_fail_closed(self) -> None:
        invalid_expert = make_dataset(em=128, n=2, k=4, experts=2)
        invalid_expert.expert_ids[0] = 2
        with self.assertRaisesRegex(ValueError, "out-of-range expert"):
            fused_moe_reference(invalid_expert)

        invalid_shape = make_dataset(em=256, n=2, k=4, experts=2)
        invalid_shape.a = invalid_shape.a[:-1]
        with self.assertRaisesRegex(ValueError, "EM must be a multiple"):
            fused_moe_reference(invalid_shape)

    def test_precision_ratio_threshold(self) -> None:
        expected = np.ones(100, dtype=np.float32)
        actual = expected.copy()
        actual[0] = 100.0
        passed, ratio = check_precision(actual, expected)
        self.assertTrue(passed)
        self.assertEqual(ratio, 0.99)

        actual[1] = 100.0
        passed, ratio = check_precision(actual, expected)
        self.assertFalse(passed)
        self.assertEqual(ratio, 0.98)

    def test_reference_self_check_is_json_scalar_friendly(self) -> None:
        result = reference_self_check()
        self.assertTrue(result["ok"])
        self.assertEqual(result["seed"], FIXED_SEED)
        self.assertIs(type(result["ok"]), bool)


class ScoringTests(unittest.TestCase):
    def test_percentiles_use_linear_interpolation(self) -> None:
        self.assertEqual(percentiles([1, 2, 3, 4, 5]), {"p20": 1.8, "p50": 3.0, "p80": 4.2})
        self.assertEqual(
            latency_percentiles([1, 2, 3, 4, 5]),
            {"p20_us": 1.8, "p50_us": 3.0, "p80_us": 4.2},
        )

    def test_equal_weight_geometric_mean_speedup(self) -> None:
        speedup = geometric_mean_speedup(
            {"case-a": 100.0, "case-b": 200.0},
            {"case-a": 50.0, "case-b": 400.0},
        )
        self.assertTrue(math.isclose(speedup, 1.0, rel_tol=1.0e-12))

    def test_promotion_requires_one_percent_no_large_regression_and_confirmation(self) -> None:
        baseline = {name: 100.0 for name in ("a", "b", "c", "d")}
        candidate = {name: 99.0 for name in baseline}

        primary = evaluate_promotion(baseline, candidate)
        self.assertFalse(primary.promoted)
        self.assertTrue(primary.needs_confirmation)
        self.assertEqual(primary.reason, "confirmation_required")

        confirmed = evaluate_promotion(baseline, candidate, candidate)
        self.assertTrue(confirmed.promoted)
        self.assertEqual(confirmed.reason, "promoted")

        # A confirmation run uses its own interleaved baseline samples, so a
        # uniform clock-frequency shift does not invalidate the comparison.
        shifted_confirmation = evaluate_promotion(
            baseline,
            candidate,
            {name: 198.0 for name in baseline},
            confirmation_baseline={name: 200.0 for name in baseline},
        )
        self.assertTrue(shifted_confirmation.promoted)

        # Excellent aggregate performance cannot hide a >3% single-case loss.
        regression = evaluate_promotion(
            baseline,
            {"a": 50.0, "b": 50.0, "c": 50.0, "d": 104.0},
        )
        self.assertFalse(regression.needs_confirmation)
        self.assertEqual(regression.reason, "per_case_regression_exceeded")

        failed_confirmation = evaluate_promotion(
            baseline,
            candidate,
            {"a": 99.0, "b": 99.0, "c": 99.0, "d": 104.0},
        )
        self.assertFalse(failed_confirmation.promoted)
        self.assertEqual(failed_confirmation.reason, "confirmation_failed")

    def test_scoring_rejects_mismatched_or_invalid_samples(self) -> None:
        with self.assertRaises(ValueError):
            geometric_mean_speedup({"a": 1.0}, {"b": 1.0})
        with self.assertRaises(ValueError):
            percentiles([])
        with self.assertRaises(ValueError):
            percentiles([1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
