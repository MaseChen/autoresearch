"""OpenCode proposer adapter."""

from __future__ import annotations

from pathlib import Path

from .errors import ControlledRuntimeError
from .models import ControllerConfig, ProposalV1
from .proposal import (
    ProposalRequest,
    Proposer,
    build_prompt,
    parse_opencode_ndjson,
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
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.iteration_index = iteration_index
        self.run_dir = run_dir
        self.runner = runner or CommandRunner()
        self.container_name = (
            f"kar-proposer-{run_id}-{iteration_index:03d}"
        )
        self.raw_path = run_dir / "raw" / f"{iteration_index:03d}.ndjson"
        self.stderr_path = run_dir / "raw" / f"{iteration_index:03d}.stderr.txt"
        self.prompt_path = run_dir / "prompts" / f"{iteration_index:03d}.txt"
        self.last_result: CommandResult | None = None

    def propose(self, request: ProposalRequest) -> ProposalV1:
        opencode_config = self.run_dir / "opencode.json"
        write_opencode_config(opencode_config, self.config)
        prompt = build_prompt(request)
        self.prompt_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_path.write_text(prompt, encoding="utf-8")
        self.prompt_path.chmod(0o600)
        argv = proposer_argv(
            self.config,
            name=self.container_name,
            run_id=self.run_id,
            opencode_config=opencode_config,
        )
        result = self.runner.run(
            argv,
            input_text=prompt,
            timeout_sec=self.config.proposer_timeout_sec,
            max_output_bytes=self.config.proposer_max_output_bytes,
            container_name=self.container_name,
            docker_binary=self.config.docker_binary,
        )
        self.last_result = result
        secret = self.config.deepseek_key_file.read_text(encoding="utf-8")
        safe_stdout = result.stdout.replace(secret, "[REDACTED_SECRET]")
        safe_stderr = result.stderr.replace(secret, "[REDACTED_SECRET]")
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_path.write_text(safe_stdout, encoding="utf-8")
        self.stderr_path.write_text(safe_stderr, encoding="utf-8")
        self.raw_path.chmod(0o600)
        self.stderr_path.chmod(0o600)
        if result.timed_out:
            raise ControlledRuntimeError(
                "OpenCode proposal exceeded its 20-minute timeout"
            )
        if result.output_limited:
            raise ControlledRuntimeError(
                "OpenCode output exceeded the configured byte limit"
            )
        if result.returncode != 0:
            raise ControlledRuntimeError(
                f"OpenCode container exited with code {result.returncode}"
            )
        return parse_opencode_ndjson(
            safe_stdout,
            expected_parent_hash=request.parent_candidate_hash,
        )
