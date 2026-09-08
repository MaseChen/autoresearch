from __future__ import annotations

import unittest

from kernel_research.compiled_reference import (
    SCORING_COMPILER_CONFIG,
    compile_scoring_reference,
    scoring_reference_source_sha256,
)


class CompiledReferenceTests(unittest.TestCase):
    def test_compile_uses_the_exact_fullgraph_configuration(self) -> None:
        calls = []

        class FakeTorch:
            def compile(self, function, **kwargs):
                calls.append((function, kwargs))
                return "compiled"

        self.assertEqual(compile_scoring_reference(FakeTorch()), "compiled")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], SCORING_COMPILER_CONFIG)
        self.assertTrue(calls[0][1]["fullgraph"])
        self.assertFalse(calls[0][1]["dynamic"])
        self.assertEqual(calls[0][1]["backend"], "inductor")

    def test_missing_torch_compile_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            compile_scoring_reference(object())

    def test_source_identity_is_stable_and_tagged(self) -> None:
        first = scoring_reference_source_sha256()
        self.assertEqual(first, scoring_reference_source_sha256())
        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
