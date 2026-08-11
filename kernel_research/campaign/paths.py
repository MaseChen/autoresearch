"""Trusted filesystem convention for production Campaign databases.

``CampaignStore`` intentionally remains path agnostic: tests, checkpoint
verification, and restore tooling need to open isolated copies.  Production
entry points use the helpers in this module so there is exactly one live
Campaign database location for a runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
from typing import Iterator


CAMPAIGN_DIRECTORY_NAME = "campaign"
CAMPAIGN_DATABASE_NAME = "campaign.sqlite3"
CAMPAIGN_MAINTENANCE_LOCK_NAME = "maintenance.lock"


def _canonical_absolute_path(
    value: str | os.PathLike[str], field: str
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{field} cannot be resolved canonically") from exc
    if path != resolved:
        raise ValueError(
            f"{field} must be canonical and must not traverse symlinks"
        )
    return path


def conventional_campaign_database(runtime_root: str | os.PathLike[str]) -> Path:
    """Return the sole production Campaign DB path for ``runtime_root``.

    The root itself must already be expressed as a canonical absolute path.
    This prevents an administrative caller from accidentally checking a
    different spelling (or symlink alias) of the live runtime.
    """

    root = _canonical_absolute_path(runtime_root, "runtime_root")
    if root == Path(root.anchor):
        raise ValueError("runtime_root must not be the filesystem root")
    return root / CAMPAIGN_DIRECTORY_NAME / CAMPAIGN_DATABASE_NAME


def conventional_campaign_maintenance_lock(
    runtime_root: str | os.PathLike[str],
) -> Path:
    """Return the single deployment/Campaign lifecycle serialization point."""

    root = _canonical_absolute_path(runtime_root, "runtime_root")
    if root == Path(root.anchor):
        raise ValueError("runtime_root must not be the filesystem root")
    return root / CAMPAIGN_DIRECTORY_NAME / CAMPAIGN_MAINTENANCE_LOCK_NAME


def _verified_lock_descriptor(path: Path, descriptor: int) -> os.stat_result:
    """Bind an open lock descriptor to the fixed non-aliased filesystem path."""

    opened = os.fstat(descriptor)
    try:
        selected = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError("Campaign maintenance lock path is unavailable") from exc
    if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(selected.st_mode):
        raise ValueError("Campaign maintenance lock must be a regular file")
    if (opened.st_dev, opened.st_ino) != (selected.st_dev, selected.st_ino):
        raise ValueError("Campaign maintenance lock descriptor is aliased")
    if opened.st_nlink != 1:
        raise ValueError("Campaign maintenance lock must not have filesystem aliases")
    if opened.st_uid != os.geteuid():
        raise ValueError("Campaign maintenance lock must be owned by the runtime user")
    return opened


def verify_inherited_campaign_maintenance_fence(
    runtime_root: str | os.PathLike[str], descriptor: int
) -> int:
    """Verify an update child's inherited descriptor still owns the fence.

    The child must not unlock this descriptor: it refers to the same open file
    description as the parent's lock.  A second open is used as a read-only
    ownership probe; it must be unable to acquire the exclusive lock.
    """

    if type(descriptor) is not int or descriptor < 3:
        raise ValueError("inherited Campaign maintenance descriptor is invalid")
    path = conventional_campaign_maintenance_lock(runtime_root)
    if path != _canonical_absolute_path(path, "Campaign maintenance lock"):
        raise ValueError("Campaign maintenance lock path is not canonical")
    _verified_lock_descriptor(path, descriptor)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    probe = os.open(path, flags)
    try:
        _verified_lock_descriptor(path, probe)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return descriptor
        fcntl.flock(probe, fcntl.LOCK_UN)
        raise ValueError("inherited Campaign maintenance descriptor is not locked")
    finally:
        os.close(probe)


@contextmanager
def campaign_maintenance_fence(
    runtime_root: str | os.PathLike[str],
) -> Iterator[int]:
    """Serialize deployment publication with Campaign lifecycle activation.

    This is a blocking host-local coordination fence.  Resource/GPU exclusion
    remains the responsibility of the separate ``gpu1.lock`` and Campaign
    lease mechanisms.
    """

    path = conventional_campaign_maintenance_lock(runtime_root)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Re-resolve after mkdir so an existing or concurrently introduced parent
    # symlink cannot provide an alternate spelling of the live runtime.
    if path != _canonical_absolute_path(path, "Campaign maintenance lock"):
        raise ValueError("Campaign maintenance lock path is not canonical")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags, 0o600)
    locked = False
    try:
        _verified_lock_descriptor(path, descriptor)
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        # Revalidate after waiting: the pathname must still identify the inode
        # whose lock is held before either side changes durable state.
        _verified_lock_descriptor(path, descriptor)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        yield descriptor
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def validate_production_campaign_database(
    database: str | os.PathLike[str],
    *,
    runtime_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Validate and return the unique live Campaign database path.

    When no root is supplied (as in the Campaign CLI), the conventional
    ``<runtime_root>/campaign/campaign.sqlite3`` shape identifies the root.
    Callers that already own a trusted runtime root (for example deployment
    administration) should supply it to bind the check to that exact runtime.

    Existence is deliberately not required because ``campaign create`` must
    be able to initialize the database.  Canonical comparison still resolves
    every existing parent, so database and directory symlink aliases fail
    closed, including broken database symlinks.
    """

    selected = _canonical_absolute_path(database, "campaign database")
    if (
        selected.name != CAMPAIGN_DATABASE_NAME
        or selected.parent.name != CAMPAIGN_DIRECTORY_NAME
    ):
        raise ValueError(
            "campaign database must use the conventional absolute path "
            "<runtime_root>/campaign/campaign.sqlite3"
        )
    root = selected.parent.parent if runtime_root is None else Path(runtime_root)
    expected = conventional_campaign_database(root)
    if selected != expected:
        raise ValueError(
            "campaign database does not belong to the supplied runtime_root"
        )
    return selected
