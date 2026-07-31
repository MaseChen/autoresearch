"""Conservative AST policy for autonomous research candidates.

This is a policy gate for accidental capability expansion.  It is not a
security sandbox and does not make an accelerator kernel safe for a shared
driver.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import subprocess
import sys
import tempfile
from typing import Any

from .constants import (
    C500_ALLOWED_NUM_WARPS,
    MAX_CANDIDATE_SOURCE_BYTES,
    POLICY_OUTPUT_LIMIT_BYTES,
    POLICY_WALL_TIMEOUT_SEC,
)
from .contract import CandidateError, validate_candidate


ALLOWED_IMPORTS = frozenset({"torch", "triton", "triton.language"})
DANGEROUS_CALLS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "delattr",
        "vars",
    }
)
DANGEROUS_ATTRIBUTES = frozenset(
    {
        "Popen",
        "bind",
        "check_call",
        "check_output",
        "chmod",
        "chown",
        "connect",
        "eval",
        "exec",
        "fork",
        "kill",
        "listen",
        "load",
        "load_library",
        "open",
        "popen",
        "remove",
        "rename",
        "replace",
        "request",
        "rmdir",
        "run",
        "save",
        "spawn",
        "system",
        "unlink",
        "urlopen",
    }
)
SIDE_EFFECT_FREE_TOP_LEVEL = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Import,
    ast.ImportFrom,
)


@dataclass(frozen=True)
class ResearchPolicyResult:
    source: str
    sha256: str
    errors: tuple[CandidateError, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "sha256": self.sha256,
            "errors": [error.to_dict() for error in self.errors],
            "boundary": (
                "research policy only; not a sandbox against malicious Python "
                "or accelerator kernels"
            ),
        }

    @classmethod
    def from_dict(
        cls, source: str, value: dict[str, Any]
    ) -> "ResearchPolicyResult":
        errors_value = value.get("errors")
        sha256 = value.get("sha256")
        expected_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if (
            not isinstance(sha256, str)
            or sha256 != expected_hash
            or not isinstance(errors_value, list)
        ):
            raise ValueError("policy worker returned an invalid result")
        errors: list[CandidateError] = []
        for item in errors_value:
            if not isinstance(item, dict):
                raise ValueError("policy worker returned an invalid error")
            errors.append(
                CandidateError(
                    code=str(item.get("code", "")),
                    message=str(item.get("message", "")),
                    line=(
                        None
                        if item.get("line") is None
                        else int(item["line"])
                    ),
                    column=(
                        None
                        if item.get("column") is None
                        else int(item["column"])
                    ),
                )
            )
        return cls(source=source, sha256=sha256, errors=tuple(errors))


def _error(code: str, message: str, node: ast.AST | None = None) -> CandidateError:
    return CandidateError(
        code=code,
        message=message,
        line=getattr(node, "lineno", None),
        column=getattr(node, "col_offset", None),
    )


def _literal_assignment(node: ast.Assign | ast.AnnAssign) -> bool:
    value = node.value
    if value is None:
        return False
    try:
        ast.literal_eval(value)
    except (ValueError, TypeError):
        return False
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return all(isinstance(target, ast.Name) for target in targets)


def _is_docstring(node: ast.AST, index: int) -> bool:
    return (
        index == 0
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _assigned_name(node: ast.AST) -> str | None:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return None
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return None
    return targets[0].id


def _assignment_value(node: ast.AST) -> ast.expr | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    return None


def _contains_name(node: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id in names
        for child in ast.walk(node)
    )


def _is_tl_int64(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "int64"
        and isinstance(node.value, ast.Name)
        and node.value.id == "tl"
    )


def _is_int64_cast(node: ast.AST, expert_aliases: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "to"
        and _contains_name(node.func.value, expert_aliases)
        and len(node.args) == 1
        and _is_tl_int64(node.args[0])
    ):
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "cast"
        and len(node.args) >= 2
        and _contains_name(node.args[0], expert_aliases)
        and _is_tl_int64(node.args[1])
    )


def _safe_expert_stride_product(
    node: ast.AST, expert_aliases: set[str]
) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return False
    for stride_side, cast_side in (
        (node.left, node.right),
        (node.right, node.left),
    ):
        if (
            isinstance(stride_side, ast.Name)
            and stride_side.id == "stride_be"
            and _is_int64_cast(cast_side, expert_aliases)
        ):
            return True
    return False


def _int64_stride_base_present(tree: ast.AST) -> bool:
    """Prove that an int64 expert stride product reaches a B tensor load."""

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _assigned_name(node) is not None
    ]
    expert_aliases: set[str] = set()
    for node in assignments:
        target = _assigned_name(node)
        value = _assignment_value(node)
        if target is None or value is None:
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "tl"
            and value.func.attr == "load"
            and value.args
            and _contains_name(value.args[0], {"expert_ids_ptr"})
        ):
            expert_aliases.add(target)
    changed = True
    while changed:
        changed = False
        for node in assignments:
            target = _assigned_name(node)
            value = _assignment_value(node)
            if (
                target is not None
                and value is not None
                and target not in expert_aliases
                and _contains_name(value, expert_aliases)
                and isinstance(value, ast.Name)
            ):
                expert_aliases.add(target)
                changed = True
    if not expert_aliases:
        return False

    safe_products: set[str] = set()
    b_pointers: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in assignments:
            target = _assigned_name(node)
            value = _assignment_value(node)
            if target is None or value is None:
                continue
            if target not in safe_products and (
                _safe_expert_stride_product(value, expert_aliases)
                or (
                    isinstance(value, ast.Name)
                    and value.id in safe_products
                )
            ):
                safe_products.add(target)
                changed = True
            safe_product_in_value = any(
                _safe_expert_stride_product(child, expert_aliases)
                for child in ast.walk(value)
            ) or _contains_name(value, safe_products)
            if (
                target not in b_pointers
                and safe_product_in_value
                and (
                    _contains_name(value, {"b_ptr"})
                    or _contains_name(value, b_pointers)
                )
            ):
                b_pointers.add(target)
                changed = True
            elif (
                target not in b_pointers
                and _contains_name(value, b_pointers)
            ):
                b_pointers.add(target)
                changed = True

    if not b_pointers:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr == "load"
            and node.args
            and _contains_name(node.args[0], b_pointers)
        ):
            return True
    return False


def _safe_decorator(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "triton"
        and target.attr in {"jit", "autotune", "heuristics"}
    )


def _is_triton_jit_decorator(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "triton"
        and target.attr == "jit"
    )


def _is_constexpr_annotation(node: ast.expr | None) -> bool:
    if not isinstance(node, ast.Attribute) or node.attr != "constexpr":
        return False
    if isinstance(node.value, ast.Name):
        return node.value.id == "tl"
    return (
        isinstance(node.value, ast.Attribute)
        and node.value.attr == "language"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "triton"
    )


def _plain_module_literal_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not _literal_assignment(node):
            continue
        if (
            isinstance(node, ast.AnnAssign)
            and _is_constexpr_annotation(node.annotation)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names.update(
            target.id for target in targets if isinstance(target, ast.Name)
        )
    return names


def _triton_non_constexpr_global_errors(
    tree: ast.Module,
) -> list[CandidateError]:
    plain_globals = _plain_module_literal_names(tree)
    if not plain_globals:
        return []
    errors: list[CandidateError] = []
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(
            _is_triton_jit_decorator(decorator)
            for decorator in function.decorator_list
        ):
            continue
        local_names = {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        local_names.update(
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        )
        if function.args.vararg is not None:
            local_names.add(function.args.vararg.arg)
        if function.args.kwarg is not None:
            local_names.add(function.args.kwarg.arg)
        global_declarations = {
            name
            for node in ast.walk(function)
            if isinstance(node, ast.Global)
            for name in node.names
        }
        local_names.difference_update(global_declarations)
        reported: set[str] = set()
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in plain_globals
                and node.id not in local_names
                and node.id not in reported
            ):
                errors.append(
                    _error(
                        "TRITON_NON_CONSTEXPR_GLOBAL",
                        f"@triton.jit function {function.name!r} reads "
                        f"ordinary module constant {node.id!r}; use a literal, "
                        "a tl.constexpr parameter, or an annotated constexpr",
                        node,
                    )
                )
                reported.add(node.id)
    return errors


def validate_research_candidate(source: str) -> ResearchPolicyResult:
    """Apply the public contract and autonomous-research policy to source."""

    contract = validate_candidate(source=source)
    errors = list(contract.errors)
    if not contract.is_valid:
        return ResearchPolicyResult(source, contract.sha256, tuple(errors))
    try:
        tree = ast.parse(source, filename="<candidate>")
    except (RecursionError, MemoryError) as exc:
        errors.append(
            _error(
                "POLICY_RESOURCE_LIMIT",
                "candidate syntax tree exceeded policy resources: "
                f"{type(exc).__name__}",
            )
        )
        return ResearchPolicyResult(source, contract.sha256, tuple(errors))

    errors.extend(_triton_non_constexpr_global_errors(tree))

    for index, node in enumerate(tree.body):
        if _is_docstring(node, index):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and _literal_assignment(node):
            continue
        if not isinstance(node, SIDE_EFFECT_FREE_TOP_LEVEL):
            errors.append(
                _error(
                    "MODULE_SIDE_EFFECT",
                    "top level may contain only imports, literal constants, "
                    "function definitions and a docstring",
                    node,
                )
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if not _safe_decorator(decorator):
                    errors.append(
                        _error(
                            "DECORATOR_NOT_ALLOWED",
                            "only triton.jit/autotune/heuristics decorators are allowed",
                            decorator,
                        )
                    )
            defaults = [
                *node.args.defaults,
                *(
                    default
                    for default in node.args.kw_defaults
                    if default is not None
                ),
            ]
            for default in defaults:
                try:
                    ast.literal_eval(default)
                except (ValueError, TypeError):
                    errors.append(
                        _error(
                            "EXECUTABLE_DEFAULT",
                            "function defaults must be literal values",
                            default,
                        )
                    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    errors.append(
                        _error(
                            "IMPORT_NOT_ALLOWED",
                            f"import {alias.name!r} is outside torch/triton",
                            node,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in ALLOWED_IMPORTS or node.level:
                errors.append(
                    _error(
                        "IMPORT_NOT_ALLOWED",
                        f"import from {module!r} is outside torch/triton",
                        node,
                    )
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in DANGEROUS_CALLS:
                errors.append(
                    _error(
                        "DANGEROUS_BUILTIN",
                        f"call to {node.func.id!r} is not allowed",
                        node,
                    )
                )
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr.endswith("__")
        ):
            errors.append(
                _error(
                    "DUNDER_ACCESS",
                    f"dunder attribute {node.attr!r} is not allowed",
                    node,
                )
            )
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in DANGEROUS_ATTRIBUTES
            and not (
                isinstance(node.value, ast.Name)
                and node.value.id == "tl"
            )
        ):
            errors.append(
                _error(
                    "DANGEROUS_ATTRIBUTE",
                    f"attribute {node.attr!r} is not allowed",
                    node,
                )
            )
        elif isinstance(node, ast.Name) and (
            node.id.startswith("__") or node.id.endswith("__")
        ):
            errors.append(
                _error(
                    "DUNDER_ACCESS",
                    f"dunder name {node.id!r} is not allowed",
                    node,
                )
            )
        elif isinstance(node, ast.keyword) and node.arg == "num_warps":
            value = (
                node.value.value
                if isinstance(node.value, ast.Constant)
                else None
            )
            if (
                type(value) is not int
                or value not in C500_ALLOWED_NUM_WARPS
            ):
                allowed = ", ".join(
                    str(item) for item in sorted(C500_ALLOWED_NUM_WARPS)
                )
                errors.append(
                    _error(
                        "C500_NUM_WARPS_INVALID",
                        "num_warps must be a literal supported by C500 "
                        f"mcTriton: one of {allowed}",
                        node.value,
                    )
                )

    if not _int64_stride_base_present(tree):
        errors.append(
            _error(
                "INT64_B_BASE_REQUIRED",
                "B expert base must multiply stride_be through a recognizable "
                ".to(tl.int64) conversion",
            )
        )
    return ResearchPolicyResult(source, contract.sha256, tuple(errors))


def _policy_resource_error(source: str, message: str) -> ResearchPolicyResult:
    return ResearchPolicyResult(
        source,
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
        (
            CandidateError(
                code="POLICY_RESOURCE_LIMIT",
                message=message,
            ),
        ),
    )


def validate_research_candidate_bounded(
    source: str,
    *,
    timeout_sec: float = POLICY_WALL_TIMEOUT_SEC,
) -> ResearchPolicyResult:
    """Validate policy in a resource-limited child interpreter."""

    source_bytes = source.encode("utf-8")
    if len(source_bytes) > MAX_CANDIDATE_SOURCE_BYTES:
        return _policy_resource_error(
            source, "candidate source exceeds 256 KiB policy limit"
        )
    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "kernel_research.policy_worker"],
                input=source_bytes,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=timeout_sec,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return _policy_resource_error(
                source, "candidate policy validation exceeded wall-clock limit"
            )
        except OSError as exc:
            return _policy_resource_error(
                source, f"candidate policy worker could not start: {exc}"
            )
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(POLICY_OUTPUT_LIMIT_BYTES + 1)
        stderr = stderr_file.read(POLICY_OUTPUT_LIMIT_BYTES + 1)
    if (
        len(stdout) > POLICY_OUTPUT_LIMIT_BYTES
        or len(stderr) > POLICY_OUTPUT_LIMIT_BYTES
    ):
        return _policy_resource_error(
            source, "candidate policy worker exceeded output limit"
        )
    if completed.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        return _policy_resource_error(
            source,
            "candidate policy worker failed"
            + (f": {detail[:1000]}" if detail else ""),
        )
    try:
        value = json.loads(stdout.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("result is not an object")
        return ResearchPolicyResult.from_dict(source, value)
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return _policy_resource_error(
            source, f"candidate policy worker returned invalid JSON: {exc}"
        )
