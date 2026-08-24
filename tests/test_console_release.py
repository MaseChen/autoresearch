from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kernel_research.console.release import build_inheritance_proof


ROOT = Path(__file__).resolve().parents[1]


class ConsoleReleaseTests(unittest.TestCase):
    def test_a9_inheritance_and_assets_are_exact(self) -> None:
        proof = build_inheritance_proof(
            ROOT,
            protected_manifest=ROOT / "docs/console/a9-protected-tree-v1.json",
            static_dir=ROOT / "kernel_research/console/static",
        )
        self.assertEqual(proof["status"], "A9_CANARY_INHERITANCE_ELIGIBLE")
        self.assertEqual(proof["source_commit"], "562d272aecf935c662abdc987df3aa5531328b73")
        self.assertTrue(proof["console_assets"]["files"])
        self.assertTrue(proof["proof_digest"].startswith("sha256:"))

    def test_protected_drift_and_forbidden_asset_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = json.loads(
                (ROOT / "docs/console/a9-protected-tree-v1.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest["files"]["kernel.py"] = "sha256:" + "0" * 64
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protected A9 file changed"):
                build_inheritance_proof(
                    ROOT,
                    protected_manifest=manifest_path,
                    static_dir=ROOT / "kernel_research/console/static",
                )
            static = root / "static"
            static.mkdir()
            (static / "index.html").write_text("ok", encoding="utf-8")
            (static / "secret.key").write_text("no", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                build_inheritance_proof(
                    ROOT,
                    protected_manifest=ROOT / "docs/console/a9-protected-tree-v1.json",
                    static_dir=static,
                )


if __name__ == "__main__":
    unittest.main()
