"""Generate a fail-closed A9 inheritance and Console asset identity proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Mapping

from ..platform.canonical import canonical_json_bytes, canonical_sha256
from ..profiler_contract import (
    PROFILER_ACTIVATION_PROFILE_DIGEST,
    PROFILER_BUILD_PROFILE_DIGEST,
    PROFILER_IMAGE,
    PROFILER_WORKER_REVISION,
)
from .protocol import AGENT_PROTOCOL_DIGEST, strict_json_loads


MAX_MANIFEST_BYTES = 256 * 1024


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if process.returncode != 0 or len(process.stdout) > 64 * 1024:
        raise ValueError("could not verify Console release Git identity")
    return process.stdout.decode("ascii").strip()


def _manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("A9 protected manifest is unavailable")
    value = strict_json_loads(path.read_bytes(), max_bytes=MAX_MANIFEST_BYTES)
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "source_commit",
        "a9",
        "files",
    }:
        raise ValueError("A9 protected manifest fields do not match")
    if value["schema_version"] != 1:
        raise ValueError("A9 protected manifest schema is unsupported")
    return value


def _asset_manifest(static_dir: Path) -> dict[str, Any]:
    if static_dir.is_symlink() or not static_dir.is_dir():
        raise ValueError("Console static asset directory is unavailable")
    files: list[dict[str, Any]] = []
    for path in sorted(static_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("Console static assets may not contain symlinks")
        if path.is_dir():
            continue
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Console static assets contain a non-regular object")
        relative = path.relative_to(static_dir).as_posix()
        if relative.endswith((".map", ".pem", ".key")):
            raise ValueError("Console static assets contain a forbidden file")
        files.append(
            {
                "path": relative,
                "byte_size": metadata.st_size,
                "sha256": _file_sha(path),
            }
        )
    if not files or "index.html" not in {item["path"] for item in files}:
        raise ValueError("Console static assets are incomplete")
    material = {"schema_version": 1, "files": files}
    return {**material, "asset_manifest_digest": canonical_sha256(material)}


def _require_clean_paths(repository: Path, paths: list[Path]) -> None:
    relative: list[str] = []
    for path in paths:
        selected = path if path.is_absolute() else repository / path
        try:
            value = selected.relative_to(repository).as_posix()
        except ValueError as exc:
            raise ValueError("release identity path escapes the repository") from exc
        if value not in relative:
            relative.append(value)
    process = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *relative,
        ],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if (
        process.returncode != 0
        or len(process.stdout) > 64 * 1024
        or process.stdout.strip()
    ):
        raise ValueError("Console release identity paths are not clean at HEAD")


def build_inheritance_proof(
    repository: Path,
    *,
    protected_manifest: Path,
    static_dir: Path,
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if not repository.is_dir() or repository.is_symlink():
        raise ValueError("repository must be a canonical directory")
    manifest = _manifest(protected_manifest)
    protected_paths = [repository / relative for relative in manifest["files"]]
    _require_clean_paths(
        repository,
        [
            Path("console"),
            Path("docs/adr/ADR-006-autoresearch-console-v1.md"),
            Path("docs/console"),
            Path("kernel_research/autorun/controller.py"),
            Path("kernel_research/console"),
            Path("pyproject.toml"),
            static_dir,
            *protected_paths,
        ],
    )
    source_commit = manifest["source_commit"]
    current_commit = _git(repository, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, current_commit],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("Console release is not a descendant of the A9 baseline")
    observed: dict[str, str] = {}
    for relative, expected in manifest["files"].items():
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"protected A9 file is unavailable: {relative}")
        digest = _file_sha(path)
        if digest != expected:
            raise ValueError(f"protected A9 file changed: {relative}")
        observed[relative] = digest
    a9 = manifest["a9"]
    current_profile = {
        "profiler_image": PROFILER_IMAGE,
        "build_profile_digest": PROFILER_BUILD_PROFILE_DIGEST,
        "activation_profile_digest": PROFILER_ACTIVATION_PROFILE_DIGEST,
        "worker_revision": PROFILER_WORKER_REVISION,
    }
    if any(current_profile.get(key) != a9.get(key) for key in current_profile):
        raise ValueError("active profiler identity differs from the A9 manifest")
    material = {
        "schema_version": 1,
        "status": "A9_CANARY_INHERITANCE_ELIGIBLE",
        "source_commit": source_commit,
        "console_commit": current_commit,
        "protected_files": observed,
        "profiler_identity": current_profile,
        "a9_canary_evidence_manifest_digest": a9[
            "canary_evidence_manifest_digest"
        ],
        "agent_protocol_digest": AGENT_PROTOCOL_DIGEST,
        "console_assets": _asset_manifest(static_dir),
    }
    return {**material, "proof_digest": canonical_sha256(material)}


def _atomic_output(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("release proof output must be an absolute non-symlink path")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = canonical_json_bytes(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".console-proof-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernel-autoresearch-console-release")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protected-manifest", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    proof = build_inheritance_proof(
        args.repository,
        protected_manifest=args.protected_manifest,
        static_dir=args.static_dir,
    )
    _atomic_output(args.output, proof)
    print(json.dumps(proof, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_inheritance_proof", "main"]
