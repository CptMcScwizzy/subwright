"""Path and filename construction.

THIS MODULE OWNS THE CONTRACT WITH PLEX AND STASH.

Both media servers mount the watch tree and read the folders this module names:
    Plex  -> /mnt/data/translate:/mnt/translate
    Stash -> /mnt/data/translate:/mnt/data/translate

Every path the application creates is built here and nowhere else, so that the
externally-visible layout can be verified by reading one file and is pinned by
tests/test_layout_contract.py. Changing anything here changes what those media
servers see. Do not inline path construction elsewhere.

Layout produced for an ingested video 'Foo.mkv':

    <base>/Foo/
        Foo.mkv          the video, moved here from ingest/
        Foo.srt          generated subtitles
        .translated      success marker, written last
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Matched case-insensitively. A single set rather than the original's
# double-glob over lower and UPPER, which double-counted on case-insensitive
# mounts and still missed mixed case like '.Mp4'.
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts"}
)

INGEST_DIRNAME = "ingest"
REPROCESS_DIRNAME = "reprocess"

TRANSLATED_MARKER = ".translated"
ERROR_MARKER = ".translation_error"
CLAIM_MARKER = ".processing"
REPROCESSED_PREFIX = ".reprocessed_"

COLLISION_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def ingest_dir(base: Path) -> Path:
    return base / INGEST_DIRNAME


def reprocess_dir(base: Path) -> Path:
    return base / REPROCESS_DIRNAME


def output_dir(base: Path, video: Path, *, now: datetime, taken: bool = False) -> Path:
    """Folder a newly ingested video is moved into.

    Named after the video's stem. If that name is already taken, a timestamp is
    appended rather than merging into the existing folder - merging would
    overwrite the subtitles of whatever is already there.
    """
    candidate = base / video.stem
    if taken:
        return base / f"{video.stem}_{now.strftime(COLLISION_TIMESTAMP_FORMAT)}"
    return candidate


def srt_for(video: Path) -> Path:
    """Subtitle path sitting beside the video, e.g. Foo.mkv -> Foo.srt.

    Deliberately not with_suffix(): a stem containing dots ('Show.S01E02.mkv')
    would lose everything after the last dot.
    """
    return video.with_name(f"{video.stem}.srt")


def tmp_srt_for(video: Path) -> Path:
    """Scratch path for an in-progress subtitle.

    Same directory as the target - os.replace is only atomic within a
    filesystem, and on this deployment the tree is NFS.
    """
    return video.with_name(f"{video.stem}.srt.tmp")


def backup_srt_for(video: Path, *, now: datetime) -> Path:
    """Timestamped backup, so repeated reprocessing does not clobber history."""
    stamp = now.strftime(COLLISION_TIMESTAMP_FORMAT)
    return video.with_name(f"{video.stem}.srt.{stamp}.bak")


def translated_marker(folder: Path) -> Path:
    return folder / TRANSLATED_MARKER


def error_marker(folder: Path) -> Path:
    return folder / ERROR_MARKER


def claim_marker(folder: Path) -> Path:
    """Written when work starts, deleted when it finishes.

    Its presence is the ONLY thing that makes a folder eligible for resume.
    A folder without it is invisible to the resume scan no matter what it
    contains - which is what stops a restart from re-transcribing the existing
    media library. See scanner.find_resumable and
    test_preexisting_library_folder_without_claim_is_untouched.
    """
    return folder / CLAIM_MARKER


def reprocessed_marker(reprocess_folder: Path, video: Path) -> Path:
    """Guard preventing a reprocessed video being picked up again every poll.

    The video is never moved out of reprocess/, so without this the same file
    would be re-transcribed on a loop forever.
    """
    return reprocess_folder / f"{REPROCESSED_PREFIX}{video.stem}"
