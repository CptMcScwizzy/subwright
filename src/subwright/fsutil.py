"""Filesystem operations.

The only module that writes, moves or deletes. Kept small and separate so the
dangerous operations are in one reviewable place.

The atomic write here is the fix for the worst bug in the original: it streamed
subtitles straight into the final path, so a crash mid-job left a truncated
.srt that looked like a finished one.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def atomic_write_text(target: Path, content: str, *, tmp: Path) -> None:
    """Write text so that `target` is either absent, the old content, or complete.

    Never a half-written file. `tmp` must be on the same filesystem as `target`
    (os.replace is only atomic within one) - layout.tmp_srt_for guarantees this
    by putting it in the same directory.
    """
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        # Includes KeyboardInterrupt/SystemExit deliberately: a signal arriving
        # mid-write must not leave the scratch file behind.
        #
        # The cleanup is itself guarded. If tmp is unremovable - a directory, a
        # permission problem - raising here would replace the real error with a
        # confusing one about the scratch file and hide the actual cause.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            log.warning("could not remove scratch file %s: %s", tmp, cleanup_exc)
        raise
    _fsync_dir(target.parent)


def _fsync_dir(path: Path) -> None:
    """Persist the rename itself, not just the file contents.

    Best-effort: not supported on Windows, and NFS servers vary.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def move(src: Path, dst: Path) -> None:
    """Move a file, creating the destination directory if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def set_owner(path: Path, uid: int, gid: int) -> None:
    """Best-effort chown so Plex and Stash can read what we produce.

    Normally a no-op: the container runs as 1000:1000 and the NFS export maps to
    the same, so files land correctly owned. Kept as belt-and-braces for anyone
    running it as root. A permission failure is logged, never fatal - refusing to
    finish a two-hour transcription over a chown would be absurd.
    """
    try:
        os.chown(path, uid, gid)
    except (PermissionError, AttributeError, OSError) as exc:
        log.debug("could not chown %s to %s:%s (%s)", path, uid, gid, exc)


def prune_backups(folder: Path, pattern: str, keep: int) -> list[Path]:
    """Delete all but the newest `keep` files matching `pattern`.

    The original never cleaned these up, so .srt.bak files accumulated forever.
    Returns what was removed, so the caller can log it.
    """
    if keep < 0:
        raise ValueError("keep must not be negative")
    matches = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for stale in matches[keep:]:
        try:
            stale.unlink()
            removed.append(stale)
        except OSError as exc:
            log.warning("could not remove old backup %s: %s", stale, exc)
    return removed


def write_marker(path: Path, content: str = "") -> None:
    """Write a small marker file (.translated, .processing, ...)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
