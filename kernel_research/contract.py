"""Static contract validation for editable kernel candidates.

Validation in this module is deliberately limited to Python syntax and the
public ``run_kernel`` signature.  Candidate code is never imported or
executed here.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any


KERNEL_PARAMETERS = (
    "a",
    "b_col_major",
    "scale_a",
    "scale_b",
    "moe_weights",
    "token_ids",
    "expert_ids",
    "topk",
    "out",
)

EVALUATION_STATUSES = frozenset(
    {
        "MOCK_VALIDATED",
        "UNSUPPORTED_ENV",
        "CONTRACT_ERROR",
        "COMPILE_ERROR",
        "PRECISION_FAILED",
        "CRASH",
        "TIMEOUT",
        "SUCCESS",
    }
)


@dataclass(frozen=True)
class CandidateError:
    """A machine-readable candidate validation error."""

    code: str
    message: str
    line: int | None = None
    column: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "column": self.column,
        }


@dataclass(frozen=True)
class CandidateValidation:
    """The source identity and static validation result for a candidate."""

    source: str
    sha256: str
    errors: tuple[CandidateError, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def is_valid(self) -> bool:
        """Compatibility alias for callers that prefer predicate naming."""

        return self.valid

    @property
    def candidate_hash(self) -> str:
        return self.sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sha256": self.sha256,
            "valid": self.valid,
            "errors": [error.to_dict() for error in self.errors],
        }


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _error(
    code: str,
    message: str,
    node: ast.AST | None = None,
) -> CandidateError:
    return CandidateError(
        code=code,
        message=message,
        line=getattr(node, "lineno", None),
        column=getattr(node, "col_offset", None),
    )


def validate_candidate(
    path: str | os.PathLike[str] | None = None,
    *,
    source: str | None = None,
) -> CandidateValidation:
    """Validate a candidate from exactly one of ``path`` or ``source``.

    File-read failures and all candidate-originated validation failures are
    returned as serializable errors.  Supplying neither or both inputs is a
    caller error and raises ``ValueError``.
    """

    if (path is None) == (source is None):
        raise ValueError("provide exactly one of path or source")

    if source is None:
        try:
            source = Path(path).read_text(encoding="utf-8")  # type: ignore[arg-type]
        except (OSError, UnicodeError) as exc:
            empty_source = ""
            return CandidateValidation(
                source=empty_source,
                sha256=_source_hash(empty_source),
                errors=(
                    CandidateError(
                        code="SOURCE_READ_ERROR",
                        message=f"could not read candidate source: {exc}",
                    ),
                ),
            )

    digest = _source_hash(source)
    try:
        tree = ast.parse(source, filename=str(path or "<candidate>"))
    except SyntaxError as exc:
        return CandidateValidation(
            source=source,
            sha256=digest,
            errors=(
                CandidateError(
                    code="SYNTAX_ERROR",
                    message=exc.msg,
                    line=exc.lineno,
                    column=exc.offset,
                ),
            ),
        )
    except (RecursionError, MemoryError) as exc:
        return CandidateValidation(
            source=source,
            sha256=digest,
            errors=(
                CandidateError(
                    code="AST_RESOURCE_LIMIT",
                    message=(
                        "candidate syntax tree exceeded parser resources: "
                        f"{type(exc).__name__}"
                    ),
                ),
            ),
        )

    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_kernel"
    ]
    if not definitions:
        return CandidateValidation(
            source=source,
            sha256=digest,
            errors=(
                CandidateError(
                    code="RUN_KERNEL_MISSING",
                    message="candidate must define one top-level run_kernel function",
                ),
            ),
        )
    if len(definitions) != 1:
        return CandidateValidation(
            source=source,
            sha256=digest,
            errors=(
                _error(
                    "RUN_KERNEL_DUPLICATE",
                    "candidate must define run_kernel exactly once at top level",
                    definitions[1],
                ),
            ),
        )

    function = definitions[0]
    errors: list[CandidateError] = []
    if isinstance(function, ast.AsyncFunctionDef):
        errors.append(
            _error(
                "ASYNC_NOT_ALLOWED",
                "run_kernel must be a synchronous function",
                function,
            )
        )

    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    actual_names = tuple(argument.arg for argument in positional)
    if actual_names != KERNEL_PARAMETERS:
        errors.append(
            _error(
                "SIGNATURE_MISMATCH",
                "run_kernel positional parameters must be exactly: "
                + ", ".join(KERNEL_PARAMETERS),
                function,
            )
        )
    if arguments.kwonlyargs:
        errors.append(
            _error(
                "KEYWORD_ONLY_NOT_ALLOWED",
                "run_kernel may not define keyword-only parameters",
                function,
            )
        )
    if arguments.vararg is not None:
        errors.append(
            _error(
                "VARARGS_NOT_ALLOWED",
                "run_kernel may not define *args",
                arguments.vararg,
            )
        )
    if arguments.kwarg is not None:
        errors.append(
            _error(
                "KWARGS_NOT_ALLOWED",
                "run_kernel may not define **kwargs",
                arguments.kwarg,
            )
        )
    if arguments.defaults or any(default is not None for default in arguments.kw_defaults):
        errors.append(
            _error(
                "DEFAULTS_NOT_ALLOWED",
                "run_kernel parameters may not have default values",
                function,
            )
        )

    return CandidateValidation(source=source, sha256=digest, errors=tuple(errors))


def validate_source(source: str) -> CandidateValidation:
    """Convenience wrapper for explicit source-string validation."""

    return validate_candidate(source=source)
