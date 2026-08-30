"""Finding work.

Pure-ish: takes a clock so the settle gate is testable without sleeping.
Returns paths; does no work and mutates nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import layout

log = logging.getLogger(__name__)

# A file is ignored until its mtime stops changing for this long, so a video
# still being copied in is not picked up half-written. Frozen from the original.
DEFAULT_SETTLE_SECONDS = 10


@dataclass(frozen=True)
class Candidate:
    path: Path
    mtime: float


def _settled(path: Path, now: float, settle_seconds: int) -> Candidate | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        # Vanished between glob and stat, or an NFS hiccup. Skip; it will be
        # picked up next pass if it really exists.
        return None
    if now - mtime <= settle_seconds:
        return None
    return Candidate(path=path, mtime=mtime)


def _videos_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and layout.is_video(p)]


def find_ingest(
    ingest: Path, *, now: float, settle_seconds: int = DEFAULT_SETTLE_SECONDS
) -> list[Path]:
    """Videos waiting in the drop folder, oldest first."""
    found = [
        c for p in _videos_in(ingest)
        if (c := _settled(p, now, settle_seconds)) is not None
    ]
    return [c.path for c in sorted(found, key=lambda c: c.mtime)]


def find_reprocess(
    folder: Path | None, *, now: float, settle_seconds: int = DEFAULT_SETTLE_SECONDS
) -> list[Path]:
    """Videos in the reprocess folder that have not already been done.

    The video is never moved out of that folder, so the marker is the only thing
    stopping the same file being transcribed again every poll.

    None means the rule has no reprocess folder, which is allowed.
    """
    if folder is None:
        return []
    out = []
    for path in _videos_in(folder):
        if layout.reprocessed_marker(folder, path).exists():
            continue
        if (c := _settled(path, now, settle_seconds)) is not None:
            out.append(c)
    return [c.path for c in sorted(out, key=lambda c: c.mtime)]


def find_resumable(output: Path, *, exclude: set[Path] | None = None) -> list[Path]:
    """Output folders holding a job that was interrupted.

    SAFETY: a folder is resumable ONLY if it contains a .processing claim that
    this application wrote. That is deliberate and load-bearing.

    The obvious alternative - "any folder with a video but no .translated" -
    would treat every hand-made folder, and anything predating the .translated
    marker, as unfinished work. On a tree that Plex and Stash already read, that
    means re-transcribing the existing library and overwriting its subtitles.

    Pinned by test_preexisting_library_folder_without_claim_is_untouched.
    """
    if not output.is_dir():
        return []
    # Compared as paths rather than by folder NAME. With configurable folders,
    # a rule's drop folder need not be called "ingest", and something unrelated
    # might be.
    skip = exclude or set()
    resumable = []
    for folder in sorted(output.iterdir()):
        if not folder.is_dir():
            continue
        if folder in skip:
            continue
        if not layout.claim_marker(folder).exists():
            continue
        if layout.translated_marker(folder).exists():
            # Finished, but the claim was not cleaned up - crash between the two
            # writes. Nothing to redo.
            continue
        videos = [p for p in folder.iterdir() if p.is_file() and layout.is_video(p)]
        if videos:
            resumable.append(videos[0])
    return resumable
