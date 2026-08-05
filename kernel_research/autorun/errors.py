"""Expected, user-facing failures raised by the trusted controller."""

from __future__ import annotations


class ControlledRuntimeError(RuntimeError):
    """An operational failure that the public CLI may serialize safely."""


class ProposalFieldLengthError(ValueError):
    """A decoded Proposal field exceeded its immutable hard limit."""

    def __init__(
        self,
        *,
        field: str,
        actual_length: int,
        hard_limit: int,
        retry_target: int,
    ) -> None:
        self.field = field
        self.actual_length = actual_length
        self.hard_limit = hard_limit
        self.retry_target = retry_target
        self.error_code = f"{field.upper()}_TOO_LONG"
        super().__init__(
            f"{field} exceeds {hard_limit} characters "
            f"(actual={actual_length}, retry_target={retry_target})"
        )
