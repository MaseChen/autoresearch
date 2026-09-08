from __future__ import annotations

import io
import json
import os
from pathlib import Path
import unittest
from unittest import mock
from contextlib import redirect_stderr

from kernel_research import cli
from kernel_research.autorun.models import ControllerConfig
from kernel_research.autorun.runtime import scoring_baseline_probe_argv


class ScoringCliTests(unittest.TestCase):
    def test_internal_probe_has_no_caller_controlled_arguments(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["score-baseline-probe"])
        self.assertEqual(args.command, "score-baseline-probe")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["score-baseline-probe", "--case", "other"])

    def test_public_qualifier_accepts_only_the_trusted_config(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            ["score", "baseline-qualify", "--config", "/trusted/config.json"]
        )
        self.assertEqual(args.score_command, "baseline-qualify")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "score",
                    "baseline-qualify",
                    "--config",
                    "/trusted/config.json",
                    "--device",
                    "gpu2",
                ]
            )

    def test_internal_probe_prints_qualified_result(self) -> None:
        payload = {"status": "QUALIFIED", "cases": []}
        with (
            mock.patch(
                "kernel_research.scoring_baseline_worker.run_scoring_baseline_probe",
                return_value=payload,
            ) as probe,
            mock.patch.dict(
                os.environ,
                {"KERNEL_RESEARCH_SCORING_FRAMEWORK_COMMIT": "d" * 40},
            ),
            mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            self.assertEqual(cli.main(["score-baseline-probe"]), 0)
        self.assertEqual(json.loads(stdout.getvalue()), payload)
        probe.assert_called_once_with(scoring_framework_git_commit="d" * 40)

    def test_runtime_argv_uses_fixed_image_entrypoint_and_command(self) -> None:
        config = mock.create_autospec(ControllerConfig, instance=True)
        config.docker_binary = "/usr/bin/docker"
        config.container_uid = 1000
        config.container_gid = 1000
        config.video_gid = 44
        config.evaluator_memory = "64g"
        config.evaluator_cpus = 8.0
        config.controller_dir = Path("/controller")
        config.resolved_framework_git_commit = "a" * 40
        config.expected_git_commit = "c" * 40
        config.gpu_devices = ()
        config.evaluator_image = "registry/image@sha256:" + "b" * 64
        argv = scoring_baseline_probe_argv(
            config,
            name="kar-score-baseline",
            run_id="score-baseline",
            cache_dir=Path("/cache"),
        )
        self.assertEqual(argv[-3:], ["-m", "kernel_research", "score-baseline-probe"])
        self.assertIn(
            "KERNEL_RESEARCH_SCORING_FRAMEWORK_COMMIT=" + "c" * 40,
            argv,
        )
        self.assertIn(
            "TORCHINDUCTOR_CACHE_DIR=/evaluator-cache/torchinductor",
            argv,
        )
        self.assertTrue(
            any("/framework/" + "c" * 40 in str(value) for value in argv)
        )
        self.assertIn("--read-only", argv)
        self.assertIn("--network", argv)
        self.assertEqual(argv[argv.index("--network") + 1], "none")

    def test_pre_gpu_finalizer_has_no_operation_or_evidence_parameters(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "score",
                "baseline-finalize-pre-gpu",
                "--config",
                "/trusted/config.json",
            ]
        )
        self.assertEqual(args.score_command, "baseline-finalize-pre-gpu")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "score",
                    "baseline-finalize-pre-gpu",
                    "--config",
                    "/trusted/config.json",
                    "--operation-id",
                    "untrusted",
                ]
            )

    def test_oom_abandonment_has_no_operation_or_evidence_parameters(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "score",
                "baseline-abandon-unknown-oom",
                "--config",
                "/trusted/config.json",
            ]
        )
        self.assertEqual(args.score_command, "baseline-abandon-unknown-oom")
        for untrusted in (
            ("--operation-id", "untrusted"),
            ("--evidence", "/untrusted/evidence.json"),
        ):
            with (
                self.subTest(untrusted=untrusted),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args(
                    [
                        "score",
                        "baseline-abandon-unknown-oom",
                        "--config",
                        "/trusted/config.json",
                        *untrusted,
                    ]
                )


if __name__ == "__main__":
    unittest.main()
