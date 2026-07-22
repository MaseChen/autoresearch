from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kernel_research.contract import KERNEL_PARAMETERS, validate_candidate
from kernel_research.history import CaseMeasurement, HistoryStore, SCHEMA_VERSION


VALID_SOURCE = """\
def run_kernel(a, b_col_major, scale_a, scale_b, moe_weights,
               token_ids, expert_ids, topk, out):
    return None
"""


class CandidateContractTests(unittest.TestCase):
    def test_valid_source_has_stable_identity_without_importing(self) -> None:
        validation = validate_candidate(source=VALID_SOURCE)

        self.assertTrue(validation.valid)
        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.source, VALID_SOURCE)
        self.assertEqual(
            validation.sha256,
            hashlib.sha256(VALID_SOURCE.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(validation.candidate_hash, validation.sha256)
        self.assertEqual(
            KERNEL_PARAMETERS,
            (
                "a",
                "b_col_major",
                "scale_a",
                "scale_b",
                "moe_weights",
                "token_ids",
                "expert_ids",
                "topk",
                "out",
            ),
        )
        json.dumps(validation.to_dict())

    def test_path_input_and_syntax_error_are_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            candidate = Path(temporary_directory) / "kernel.py"
            candidate.write_text("def run_kernel(:\n", encoding="utf-8")

            validation = validate_candidate(candidate)

        self.assertFalse(validation.valid)
        self.assertEqual(validation.errors[0].code, "SYNTAX_ERROR")
        self.assertIsNotNone(validation.errors[0].line)
        json.dumps(validation.to_dict())

    def test_duplicate_top_level_definition_is_rejected(self) -> None:
        validation = validate_candidate(source=VALID_SOURCE + "\n" + VALID_SOURCE)

        self.assertFalse(validation.valid)
        self.assertEqual(
            [error.code for error in validation.errors], ["RUN_KERNEL_DUPLICATE"]
        )

    def test_wrong_signature_async_variadics_and_defaults_are_rejected(self) -> None:
        candidates = {
            "wrong_order": (
                "def run_kernel(b_col_major, a, scale_a, scale_b, moe_weights, "
                "token_ids, expert_ids, topk, out): pass"
            ),
            "async": (
                "async def run_kernel(a, b_col_major, scale_a, scale_b, moe_weights, "
                "token_ids, expert_ids, topk, out): pass"
            ),
            "variadics": (
                "def run_kernel(a, b_col_major, scale_a, scale_b, moe_weights, "
                "token_ids, expert_ids, topk, out, *args, **kwargs): pass"
            ),
            "default": (
                "def run_kernel(a, b_col_major, scale_a, scale_b, moe_weights, "
                "token_ids, expert_ids, topk, out=None): pass"
            ),
        }
        expected_codes = {
            "wrong_order": {"SIGNATURE_MISMATCH"},
            "async": {"ASYNC_NOT_ALLOWED"},
            "variadics": {"VARARGS_NOT_ALLOWED", "KWARGS_NOT_ALLOWED"},
            "default": {"DEFAULTS_NOT_ALLOWED"},
        }

        for name, source in candidates.items():
            with self.subTest(name=name):
                errors = validate_candidate(source=source).errors
                self.assertTrue(expected_codes[name].issubset({error.code for error in errors}))


class HistoryStoreTests(unittest.TestCase):
    def test_restart_five_records_dedup_artifact_samples_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            database = state_dir / "history.sqlite3"
            sources = [
                VALID_SOURCE + f"\n# candidate {index}\n" for index in range(4)
            ]

            with HistoryStore(database, state_dir) as history:
                self.assertEqual(history.schema_version, SCHEMA_VERSION)
                self.assertEqual(history.journal_mode.lower(), "wal")
                records = []
                for index in range(4):
                    records.append(
                        history.record_experiment(
                            candidate_source=sources[index],
                            git_commit=f"commit-{index}",
                            backend="c500",
                            suite="full",
                            status="SUCCESS" if index in {1, 2} else "COMPILE_ERROR",
                            promotable=index in {1, 2},
                            aggregate_score=(None if index not in {1, 2} else 1.0 + index),
                            note=f"experiment {index}",
                            environment={"device": "C500", "ordinal": index},
                            error_summary=("compile failed" if index == 0 else None),
                            result={"version": 1, "status": "recorded"},
                            case_measurements=(
                                CaseMeasurement(
                                    name="full-0",
                                    matched_ratio=0.995,
                                    passed=True,
                                    raw_samples=(1.2 + index, 1.1 + index, 1.3 + index),
                                    baseline_samples=(2.0, 2.1, 1.9),
                                    metrics={"p50_ms": 1.2 + index},
                                ),
                            ),
                        )
                    )
                duplicate = history.record_experiment(
                    candidate_source=sources[0],
                    backend="mock",
                    suite="smoke",
                    status="MOCK_VALIDATED",
                    environment={"python": "test"},
                    case_measurements=(
                        {
                            "name": "smoke-0",
                            "matched_ratio": 1.0,
                            "passed": True,
                            "raw_samples": [],
                            "metrics": {"reference_checked": True},
                        },
                    ),
                )

                first = records[0]
                self.assertFalse(first.duplicate)
                self.assertTrue(duplicate.duplicate)
                self.assertEqual(duplicate.duplicate_of_id, first.id)
                self.assertEqual(history.candidate_seen_count(first.candidate_hash), 2)
                self.assertTrue(history.has_candidate(first.candidate_hash))
                self.assertEqual(len(history.find_by_candidate_hash(first.candidate_hash)), 2)
                artifact = state_dir / first.artifact_path
                self.assertEqual(artifact.read_text(encoding="utf-8"), sources[0])
                self.assertEqual(len(list((state_dir / "artifacts").glob("*.py"))), 4)

            with HistoryStore(database, state_dir) as reopened:
                all_records = reopened.list_experiments()
                self.assertEqual(len(all_records), 5)
                self.assertEqual(all_records[-1].duplicate_of_id, all_records[0].id)
                self.assertEqual(
                    all_records[1].case_measurements[0].raw_samples,
                    (2.2, 2.1, 2.3),
                )
                self.assertEqual(
                    all_records[1].case_measurements[0].baseline_samples,
                    (2.0, 2.1, 1.9),
                )
                self.assertEqual(all_records[1].environment["device"], "C500")
                best = reopened.get_best(backend="c500", suite="full")
                self.assertIsNotNone(best)
                assert best is not None
                self.assertEqual(best.aggregate_score, 3.0)
                self.assertEqual(len(json.loads(reopened.export_json())), 5)
                self.assertEqual(reopened.export_tsv().count("\n"), 6)
                self.assertIn("MOCK_VALIDATED", reopened.format_table())
                self.assertEqual(reopened.export("table"), reopened.format_table())

    def test_hash_mismatch_and_duplicate_case_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with HistoryStore(root / "history.sqlite3", root) as history:
                with self.assertRaises(ValueError):
                    history.record_experiment(
                        candidate_source=VALID_SOURCE,
                        candidate_hash="0" * 64,
                        backend="mock",
                        suite="smoke",
                        status="MOCK_VALIDATED",
                    )
                with self.assertRaises(ValueError):
                    history.record_experiment(
                        candidate_source=VALID_SOURCE,
                        backend="mock",
                        suite="smoke",
                        status="MOCK_VALIDATED",
                        case_measurements=(
                            {"name": "same"},
                            {"name": "same"},
                        ),
                    )
                self.assertEqual(history.list_experiments(), [])


if __name__ == "__main__":
    unittest.main()
