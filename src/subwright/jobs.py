"""The three kinds of work: ingest, reprocess, resume.

Receives its Transcriber and clock as arguments and never constructs them. That
is what lets the whole of this module be tested without a GPU.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from . import fsutil, layout, srt
from .srt import Cue
from .transcriber import Transcriber

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]


class Progress(Protocol):
    """Somewhere to report a running job's progress.

    A protocol rather than an import of worker.Status, because worker imports
    this module. Status satisfies it structurally, and tests pass a recorder.

    Always optional: every job runs correctly with progress=None, which is what
    the self-test and the direct unit tests do.
    """

    def set_media_duration(self, duration: float) -> None: ...

    def observe_cue(self, cue: Cue) -> None: ...


@dataclass
class JobResult:
    video: Path
    srt_path: Path
    cue_count: int
    media_duration: float


def _write_cues(
    video: Path,
    transcriber: Transcriber,
    language: str | None,
    *,
    progress: Progress | None = None,
) -> JobResult:
    """Transcribe and write subtitles atomically beside the video."""
    target = layout.srt_for(video)
    tmp = layout.tmp_srt_for(video)

    cues_iter, info = transcriber.transcribe(video, language)
    if progress is not None:
        progress.set_media_duration(info.duration)

    # Accumulated before writing: a failure part-way through transcription must
    # not produce a partial file. Whisper output for a feature-length video is a
    # few hundred KB, so holding it is not a concern.
    #
    # The loop is written out rather than list() so each cue can be reported as
    # it arrives. faster-whisper's iterator is lazy - consuming it is what
    # actually drives transcription - so this is the only point at which the
    # progress of a running job is observable at all.
    cues = []
    for cue in cues_iter:
        cues.append(cue)
        if progress is not None:
            progress.observe_cue(cue)

    fsutil.atomic_write_text(target, srt.render(cues), tmp=tmp)
    return JobResult(video=video, srt_path=target, cue_count=len(cues),
                     media_duration=info.duration)


def run_ingest(
    video: Path,
    base: Path,
    transcriber: Transcriber,
    *,
    language: str | None,
    now: Clock,
    uid: int = 1000,
    gid: int = 1000,
    progress: Progress | None = None,
) -> JobResult:
    """Move a video from ingest/ into its own folder, then subtitle it.

    Order matters and is part of the contract: the video is moved FIRST, so it
    leaves ingest/ immediately and cannot be picked up twice. The .processing
    claim is written before the move, so an interrupted job is resumable.
    """
    stamp = now()
    folder = layout.output_dir(base, video, now=stamp)
    if folder.exists():
        folder = layout.output_dir(base, video, now=stamp, taken=True)
    folder.mkdir(parents=True, exist_ok=True)

    fsutil.write_marker(
        layout.claim_marker(folder),
        json.dumps({"source": video.name, "started": stamp.isoformat()}) + "\n",
    )

    moved = folder / video.name
    fsutil.move(video, moved)
    fsutil.set_owner(folder, uid, gid)
    fsutil.set_owner(moved, uid, gid)

    try:
        result = _write_cues(moved, transcriber, language, progress=progress)
    except Exception as exc:
        # Record the failure AND drop the claim.
        #
        # The claim distinguishes "the process died" from "this file cannot be
        # transcribed". Keeping it here would make a permanently broken file
        # retry on every poll forever, pinning the GPU. Dropping it means the
        # resume scan ignores this folder and retrying becomes an explicit
        # action. A kill gives this handler no chance to run, so the claim
        # survives - which is exactly the case resume exists for.
        fsutil.write_marker(
            layout.error_marker(folder),
            f"{stamp.isoformat()}\n{exc}\n",
        )
        layout.claim_marker(folder).unlink(missing_ok=True)
        raise

    fsutil.set_owner(result.srt_path, uid, gid)
    fsutil.write_marker(
        layout.translated_marker(folder),
        f"Translated on {now().isoformat()}\n",
    )
    fsutil.set_owner(layout.translated_marker(folder), uid, gid)
    layout.claim_marker(folder).unlink(missing_ok=True)
    layout.error_marker(folder).unlink(missing_ok=True)
    return result


def run_resume(
    video: Path,
    transcriber: Transcriber,
    *,
    language: str | None,
    now: Clock,
    uid: int = 1000,
    gid: int = 1000,
    progress: Progress | None = None,
) -> JobResult:
    """Finish a job whose folder still holds a .processing claim.

    The video has already been moved; only the subtitles are missing.
    """
    folder = video.parent
    log.info("resuming interrupted job in %s", folder)
    try:
        result = _write_cues(video, transcriber, language, progress=progress)
    except Exception as exc:
        fsutil.write_marker(layout.error_marker(folder), f"{now().isoformat()}\n{exc}\n")
        raise
    fsutil.set_owner(result.srt_path, uid, gid)
    fsutil.write_marker(
        layout.translated_marker(folder), f"Translated on {now().isoformat()}\n"
    )
    layout.claim_marker(folder).unlink(missing_ok=True)
    layout.error_marker(folder).unlink(missing_ok=True)
    return result


def run_reprocess(
    video: Path,
    base: Path,
    transcriber: Transcriber,
    *,
    language: str | None,
    now: Clock,
    keep_backups: int = 3,
    uid: int = 1000,
    gid: int = 1000,
    progress: Progress | None = None,
) -> JobResult:
    """Regenerate subtitles in place. The video is never moved."""
    folder = layout.reprocess_dir(base)
    existing = layout.srt_for(video)
    backup: Path | None = None

    if existing.exists():
        backup = layout.backup_srt_for(video, now=now())
        existing.replace(backup)

    try:
        result = _write_cues(video, transcriber, language, progress=progress)
    except Exception:
        if backup is not None and backup.exists():
            # Put the good subtitles back before giving up.
            backup.replace(existing)
        raise

    fsutil.set_owner(result.srt_path, uid, gid)
    fsutil.write_marker(layout.reprocessed_marker(folder, video), f"{now().isoformat()}\n")
    fsutil.prune_backups(video.parent, f"{video.stem}.srt.*.bak", keep=keep_backups)
    return result
