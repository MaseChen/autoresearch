from __future__ import annotations

import unittest

from kernel_research.compiled_reference import (
    SCORING_COMPILER_CONFIG,
    compile_scoring_reference,
    make_scoring_reference,
    scoring_reference_source_sha256,
)


class CompiledReferenceTests(unittest.TestCase):
    def test_reference_keeps_expert_selection_in_the_tensor_graph(self) -> None:
        index_select_calls = []

        class FakeTensor:
            def __init__(self, shape):
                self.shape = tuple(shape)

            def __getitem__(self, key):
                if isinstance(key, FakeTensor):
                    raise AssertionError("data-dependent scalar indexing is forbidden")
                if isinstance(key, slice):
                    start = 0 if key.start is None else key.start
                    stop = self.shape[0] if key.stop is None else key.stop
                    return FakeTensor((stop - start, *self.shape[1:]))
                if isinstance(key, int):
                    return FakeTensor(self.shape[1:])
                if isinstance(key, tuple):
                    return FakeTensor(self.shape)
                raise AssertionError(f"unsupported fake index: {key!r}")

            def to(self, *args, **kwargs):
                return FakeTensor(self.shape)

            def contiguous(self):
                return self

            def transpose(self, first, second):
                shape = list(self.shape)
                shape[first], shape[second] = shape[second], shape[first]
                return FakeTensor(shape)

            def __mul__(self, other):
                return FakeTensor(self.shape)

            __rmul__ = __mul__

        class FakeTorch:
            int64 = "int64"
            int32 = "int32"
            float32 = "float32"
            bfloat16 = "bfloat16"

            @staticmethod
            def index_select(tensor, dimension, index):
                if dimension != 0 or not isinstance(index, FakeTensor):
                    raise AssertionError("expert selection must use a tensor index")
                index_select_calls.append((tensor.shape, index.shape))
                return FakeTensor((index.shape[0], *tensor.shape[1:]))

            @staticmethod
            def _int_mm(left, right):
                return FakeTensor((left.shape[0], right.shape[1]))

            @staticmethod
            def cat(tensors, dim):
                if dim != 0:
                    raise AssertionError("reference must concatenate tile rows")
                tensors = tuple(tensors)
                return FakeTensor(
                    (sum(tensor.shape[0] for tensor in tensors), tensors[0].shape[1])
                )

        reference = make_scoring_reference(FakeTorch())
        output = reference(
            FakeTensor((256, 4)),
            FakeTensor((2, 3, 4)),
            FakeTensor((256,)),
            FakeTensor((2, 3)),
            FakeTensor((256,)),
            FakeTensor((2,)),
        )
        self.assertEqual(output.shape, (256, 3))
        self.assertEqual(
            index_select_calls,
            [((2, 3, 4), (1,)), ((2, 3), (1,))] * 2,
        )

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
