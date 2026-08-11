"""OpenCode proposer adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..constants import (
    MAX_PROPOSER_ATTEMPTS,
    PROPOSAL_HYPOTHESIS_RETRY_TARGET,
    PROPOSAL_RATIONALE_RETRY_TARGET,
)
from .errors import ControlledRuntimeError, ProposalFieldLengthError
from .model_catalog import resolve_opencode_model
from .models import ControllerConfig, ProposalV1
from .proposal import (
    ProposalRequest,
    ProposerConstraintRetryExhaustedError,
    ProposalFormatError,
    Proposer,
    ProposerFormatRetryExhaustedError,
    build_prompt,
    parse_opencode_ndjson_result,
)
from .runtime import (
    CommandResult,
    CommandRunner,
    proposer_argv,
    write_opencode_config,
)


class OpenCodeProposer(Proposer):
    def __init__(
        self,
        config: ControllerConfig,
        *,
        run_id: str,
        iteration_index: int,
        run_dir: Path,
        runner: CommandRunner | None = None,
        before_container_start: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.iteration_index = iteration_index
        self.run_dir = run_dir
        self.runner = runner or CommandRunner()
        if before_container_start is not None and not callable(
            before_container_start
        ):
            raise TypeError("before_container_start must be callable")
        self.before_container_start = before_container_start
        self.container_name = (
            f"kar-proposer-{run_id}-{iteration_index:03d}"
        )
        self.raw_path = run_dir / "raw" / f"{iteration_index:03d}.ndjson"
        self.stderr_path = run_dir / "raw" / f"{iteration_index:03d}.stderr.txt"
        self.prompt_path = run_dir / "prompts" / f"{iteration_index:03d}.txt"
        self.last_result: CommandResult | None = None
        self.attempts: list[dict[str, object]] = []

    def _attempt_paths(self, attempt: int) -> tuple[Path, Path, Path]:
        suffix = "" if attempt == 1 else f".retry-{attempt - 1}"
        stem = f"{self.iteration_index:03d}{suffix}"
        return (
            self.run_dir / "prompts" / f"{stem}.txt",
            self.run_dir / "raw" / f"{stem}.ndjson",
            self.run_dir / "raw" / f"{stem}.stderr.txt",
        )

    def _write_prompt(self, prompt: str) -> None:
        self.prompt_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_path.write_text(prompt, encoding="utf-8")
        self.prompt_path.chmod(0o600)

    def _archive_result(self, *, stdout: str, stderr: str) -> None:
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_path.write_text(stdout, encoding="utf-8")
        self.stderr_path.write_text(stderr, encoding="utf-8")
        self.raw_path.chmod(0o600)
        self.stderr_path.chmod(0o600)

    def propose(self, request: ProposalRequest) -> ProposalV1:
        opencode_config = self.run_dir / "opencode.json"
        write_opencode_config(opencode_config, self.config)
        model = resolve_opencode_model(self.config.opencode_model)
        base_prompt = build_prompt(request)
        secret = self.config.deepseek_key_file.read_text(encoding="utf-8")
        deadline = time.monotonic() + self.config.proposer_timeout_sec
        remaining_output = self.config.proposer_max_output_bytes
        prior_format_error: str | None = None
        prior_constraint_error: ProposalFieldLengthError | None = None

        for attempt in range(1, MAX_PROPOSER_ATTEMPTS + 1):
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise ControlledRuntimeError(
                    "OpenCode proposal attempts exhausted the shared "
                    "configured timeout"
                )
            if remaining_output <= 0:
                raise ControlledRuntimeError(
                    "OpenCode proposal attempts exhausted the shared "
                    "output byte limit"
                )
            prompt = base_prompt
            retry_trigger: str | None = None
            if prior_format_error is not None:
                retry_trigger = "FORMAT_ERROR"
                prompt += (
                    "\n\nFORMAT RETRY: The previous response was rejected "
                    "because it was not syntactically valid ProposalV1 JSON "
                    f"({prior_format_error}). Generate it again from scratch. "
                    "Return one bare JSON object, correctly JSON-escape the "
                    "complete kernel_source string, and return no fence, "
                    "summary, or other text. The first byte must be '{'. "
                    "Close kernel_source and the outer object exactly once; "
                    "the final byte must be '}' with nothing after it."
                )
            elif prior_constraint_error is not None:
                retry_trigger = "FIELD_LENGTH_ERROR"
                prompt += (
                    "\n\nCOMPLIANCE RETRY: The previous ProposalV1 was "
                    "structurally valid, but "
                    f"{prior_constraint_error.field} contained "
                    f"{prior_constraint_error.actual_length} characters and "
                    f"exceeded its hard limit of "
                    f"{prior_constraint_error.hard_limit}. Regenerate the "
                    "complete ProposalV1 from scratch. Write hypothesis as "
                    "one sentence of at most "
                    f"{PROPOSAL_HYPOTHESIS_RETRY_TARGET} characters and keep "
                    "rationale at most "
                    f"{PROPOSAL_RATIONALE_RETRY_TARGET} characters. Move "
                    "evidence and "
                    "implementation detail into rationale. Do not truncate "
                    "or omit kernel_source, and do not return the previous "
                    "response."
                )
            (
                self.prompt_path,
                self.raw_path,
                self.stderr_path,
            ) = self._attempt_paths(attempt)
            attempt_record: dict[str, object] = {
                "attempt": attempt,
                "prompt_path": str(self.prompt_path),
                "raw_output_path": str(self.raw_path),
                "stderr_path": str(self.stderr_path),
                "outcome": "RUNNING",
            }
            if retry_trigger is not None:
                attempt_record["retry_trigger"] = retry_trigger
            self.attempts.append(attempt_record)
            self._write_prompt(prompt)
            argv = proposer_argv(
                self.config,
                name=self.container_name,
                run_id=self.run_id,
                opencode_config=opencode_config,
            )
            # The trusted host rechecks the frozen Run/Campaign window at the
            # last possible point before *each* Docker attempt.  This matters
            # for the bounded format/constraint retry: authorization for the
            # first container is never reused for the second one.
            if self.before_container_start is not None:
                self.before_container_start(remaining_time)
            result = self.runner.run(
                argv,
                input_text=prompt,
                timeout_sec=remaining_time,
                max_output_bytes=remaining_output,
                container_name=self.container_name,
                docker_binary=self.config.docker_binary,
            )
            self.last_result = result
            safe_stdout = result.stdout.replace(secret, "[REDACTED_SECRET]")
            safe_stderr = result.stderr.replace(secret, "[REDACTED_SECRET]")
            self._archive_result(
                stdout=safe_stdout,
                stderr=safe_stderr,
            )
            used_output = len(result.stdout.encode("utf-8")) + len(
                result.stderr.encode("utf-8")
            )
            remaining_output -= used_output
            if result.timed_out:
                attempt_record["outcome"] = "TIMEOUT"
                raise ControlledRuntimeError(
                    "OpenCode proposal exceeded its shared configured timeout"
                )
            if result.output_limited:
                attempt_record["outcome"] = "OUTPUT_LIMIT"
                raise ControlledRuntimeError(
                    "OpenCode output exceeded the shared configured byte limit"
                )
            if result.returncode != 0:
                attempt_record["outcome"] = "CONTAINER_ERROR"
                raise ControlledRuntimeError(
                    f"OpenCode container exited with code {result.returncode}"
                )
            try:
                parsed = parse_opencode_ndjson_result(
                    safe_stdout,
                    expected_parent_hash=request.parent_candidate_hash,
                    configured_output_token_cap=(
                        model.request_output_token_cap
                    ),
                )
            except ProposalFormatError as exc:
                attempt_record["outcome"] = "FORMAT_ERROR"
                attempt_record["error"] = str(exc)
                prior_format_error = str(exc)
                if attempt == MAX_PROPOSER_ATTEMPTS:
                    raise ProposerFormatRetryExhaustedError(
                        "PROPOSER_FORMAT_RETRY_EXHAUSTED: OpenCode produced "
                        "invalid ProposalV1 JSON after exhausting its two "
                        "bounded attempts"
                    ) from exc
                prior_constraint_error = None
                continue
            except ProposalFieldLengthError as exc:
                attempt_record.update(
                    {
                        "outcome": "CONSTRAINT_ERROR",
                        "error": str(exc),
                        "error_code": exc.error_code,
                        "field": exc.field,
                        "actual_length": exc.actual_length,
                        "hard_limit": exc.hard_limit,
                        "retry_target": exc.retry_target,
                    }
                )
                if attempt == MAX_PROPOSER_ATTEMPTS:
                    raise ProposerConstraintRetryExhaustedError(
                        "PROPOSER_CONSTRAINT_RETRY_EXHAUSTED: OpenCode "
                        "produced an overlong ProposalV1 field after "
                        f"exhausting its two bounded attempts "
                        f"(field={exc.field}, actual={exc.actual_length}, "
                        f"hard_limit={exc.hard_limit})"
                    ) from exc
                prior_constraint_error = exc
                prior_format_error = None
                continue
            except ValueError as exc:
                attempt_record["outcome"] = "PROPOSAL_ERROR"
                attempt_record["error"] = str(exc)
                raise
            attempt_record["outcome"] = "SUCCESS"
            if parsed.transport_recovery is not None:
                attempt_record["transport_recovery"] = (
                    parsed.transport_recovery
                )
            return parsed.proposal
        raise AssertionError("proposal attempt loop ended unexpectedly")
