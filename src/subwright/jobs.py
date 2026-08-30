"""The three kinds of work: ingest, reprocess, resume.

Receives its Transcriber and clock as arguments and never constructs them. That
is what lets the whole of this module be tested without a GPU.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from . import existing, fsutil, layout, srt
from . import report as report_mod
from .mediaprobe import Prober
from .profiles import Diagnostics, Profile
from .srt import Cue
from .transcriber import MediaInfo, Transcriber

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]


class Progress(Protocol):
    """Somewhere to report a running job's progress.

    A protocol rather than an import of worker.Status, because worker imports
    this module. Status satisfies it structurally, and tests pass a recorder.

    Always optional: every job runs correctly with progress=None, which is what
    the self-test and the direct unit tests do.
    """

    def set_media_info(self, info: MediaInfo) -> None: ...

    def observe_cue(self, cue: Cue) -> None: ...


@dataclass
class JobResult:
    video: Path
    srt_path: Path
    cue_count: int
    media_duration: float
    # What Whisper thought the audio was, and how sure it was. Only meaningful
    # when no language was pinned, but recorded either way so the history says
    # what actually happened rather than what was configured at the time.
    detected_language: str | None = None
    language_probability: float | None = None
    # transcribed | sidecar | embedded. Worth recording: "this took six minutes
    # of GPU" and "this was copied from a file already there" are very different
    # events that would otherwise look identical in the history.
    source: str = "transcribed"
    source_detail: str | None = None
    diagnostics: Diagnostics | None = None
    cues: list[Cue] | None = None


def _write_cues(
    video: Path,
    transcriber: Transcriber,
    language: str | None,
    *,
    progress: Progress | None = None,
    profile: Profile | None = None,
) -> JobResult:
    """Transcribe and write subtitles atomically beside the video."""
    target = layout.srt_for(video)
    tmp = layout.tmp_srt_for(video)

    cues_iter, info = transcriber.transcribe(video, language, profile)
    if progress is not None:
        progress.set_media_info(info)

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
    diagnostics = Diagnostics(
        duration=info.duration,
        duration_after_vad=info.duration_after_vad or info.duration,
        cue_count=len(cues),
        logprobs=[c.avg_logprob for c in cues if c.avg_logprob is not None],
    )
    return JobResult(video=video, srt_path=target, cue_count=len(cues),
                     media_duration=info.duration,
                     detected_language=info.detected_language,
                     language_probability=info.language_probability,
                     diagnostics=diagnostics, cues=cues)


def _count_cues(text: str) -> int:
    return text.count(" --> ")


def _relocate_sidecars(
    chosen: Path | None, strays: list[Path], folder: Path, moved: Path,
) -> Path | None:
    """Move every sidecar out of the drop folder, following the video.

    Leaving them behind is what used to happen, and it stranded a perfectly
    good subtitle file in the drop folder while the video was re-transcribed.

    The one being used is renamed to sit beside the video as <stem>.srt, which
    is the name Plex and Stash look for. Others keep their names.
    """
    used = None
    for stray in strays:
        if chosen is not None and stray == chosen:
            target = layout.srt_for(moved)
            fsutil.move(stray, target)
            used = target
        else:
            fsutil.move(stray, folder / stray.name)
    return used


def _extract_embedded(moved: Path, stream_index: int, prober: Prober) -> int:
    """Pull a subtitle track out of the video. Returns the cue count.

    Written to the scratch path and moved into place, exactly as a transcribed
    file is, so a failure part-way cannot leave a truncated .srt that looks
    complete.
    """
    tmp = layout.tmp_srt_for(moved)
    target = layout.srt_for(moved)
    try:
        prober.extract(moved, stream_index, tmp)
        text = tmp.read_text(encoding="utf-8", errors="replace")
        if not existing.looks_like_srt(text):
            # ffmpeg reports success on an image-based track and writes nothing
            # usable. Checked rather than trusted.
            raise ValueError("extracted file does not look like subtitles")
        cues = _count_cues(text)
        tmp.replace(target)
        return cues
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _reuse_existing(
    found: existing.Reuse,
    moved: Path,
    strays: list[Path],
    folder: Path,
    prober: Prober | None,
) -> JobResult | None:
    """Use subtitles that already exist. None means "could not, transcribe instead".

    Every failure here is non-fatal by design. Reuse is an optimisation, and it
    must never turn a file that would have transcribed fine into a failed job.
    """
    try:
        if found.kind == "sidecar":
            used = _relocate_sidecars(found.sidecar, strays, folder, moved)
            if used is None:
                return None
            cues = _count_cues(used.read_text(encoding="utf-8", errors="replace"))
        else:
            _relocate_sidecars(None, strays, folder, moved)
            if prober is None or found.stream_index is None:
                return None
            cues = _extract_embedded(moved, found.stream_index, prober)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        log.warning(
            "could not reuse existing subtitles for %s (%s); transcribing instead",
            moved.name, exc,
        )
        return None

    log.info("reused existing subtitles for %s from %s (%d cues)",
             moved.name, found.detail, cues)
    return JobResult(
        video=moved, srt_path=layout.srt_for(moved), cue_count=cues,
        media_duration=0.0, source=found.kind, source_detail=found.detail,
    )


def write_report(
    result: JobResult,
    *,
    profile: Profile,
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
    rule_name: str | None,
    now: Clock,
    uid: int = 1000,
    gid: int = 1000,
) -> Path | None:
    """Write the diagnostic report beside the subtitles.

    Never raises: a report is an explanation of a finished job, and failing to
    write one must not fail the job it is explaining.
    """
    target = layout.report_for(result.video)
    try:
        text = report_mod.render(
            video=result.video,
            cues=result.cues or [],
            diagnostics=result.diagnostics or Diagnostics(
                duration=result.media_duration, cue_count=result.cue_count),
            profile=profile,
            model=model, device=device, compute_type=compute_type,
            language=language,
            detected_language=result.detected_language,
            language_probability=result.language_probability,
            rule_name=rule_name,
            source=result.source, source_detail=result.source_detail,
            now=now(),
        )
        fsutil.atomic_write_text(target, text, tmp=target.with_suffix(".txt.tmp"))
        fsutil.set_owner(target, uid, gid)
        return target
    except Exception as exc:  # noqa: BLE001 - see docstring
        log.warning("could not write the report for %s: %s", result.video.name, exc)
        return None


def run_ingest(
    video: Path,
    output: Path,
    transcriber: Transcriber,
    *,
    language: str | None,
    now: Clock,
    uid: int = 1000,
    gid: int = 1000,
    progress: Progress | None = None,
    profile: Profile | None = None,
    prober: Prober | None = None,
    reuse: bool = True,
) -> JobResult:
    """Move a video from ingest/ into its own folder, then subtitle it.

    Order matters and is part of the contract: the video is moved FIRST, so it
    leaves ingest/ immediately and cannot be picked up twice. The .processing
    claim is written before the move, so an interrupted job is resumable.
    """
    stamp = now()
    folder = layout.output_dir(output, video, now=stamp)
    if folder.exists():
        folder = layout.output_dir(output, video, now=stamp, taken=True)
    folder.mkdir(parents=True, exist_ok=True)

    fsutil.write_marker(
        layout.claim_marker(folder),
        json.dumps({"source": video.name, "started": stamp.isoformat()}) + "\n",
    )

    # Both of these must happen BEFORE the move: a sidecar lives beside the
    # video in the drop folder, and once the video has gone its siblings can no
    # longer be found from it.
    found = existing.find(video, prober) if reuse else None
    strays = existing.sidecars_for(video)

    moved = folder / video.name
    fsutil.move(video, moved)
    fsutil.set_owner(folder, uid, gid)
    fsutil.set_owner(moved, uid, gid)

    try:
        result = None
        if found is not None:
            result = _reuse_existing(found, moved, strays, folder, prober)
        elif strays:
            # Nothing usable, but they still follow the video rather than being
            # left behind in the drop folder.
            _relocate_sidecars(None, strays, folder, moved)
        if result is None:
            result = _write_cues(moved, transcriber, language, progress=progress,
                                 profile=profile)
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
    profile: Profile | None = None,
) -> JobResult:
    """Finish a job whose folder still holds a .processing claim.

    The video has already been moved; only the subtitles are missing.
    """
    folder = video.parent
    log.info("resuming interrupted job in %s", folder)
    try:
        result = _write_cues(video, transcriber, language, progress=progress,
                             profile=profile)
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
    marker_dir: Path,
    transcriber: Transcriber,
    *,
    language: str | None,
    now: Clock,
    keep_backups: int = 3,
    uid: int = 1000,
    gid: int = 1000,
    progress: Progress | None = None,
    profile: Profile | None = None,
) -> JobResult:
    """Regenerate subtitles in place. The video is never moved."""
    folder = marker_dir
    current = layout.srt_for(video)
    backup: Path | None = None

    if current.exists():
        # COPIED, not moved. Moving it first left the video with no subtitles
        # at all for the several minutes transcription takes - and if the
        # process was killed in that window, permanently, because the restore
        # lives in an exception handler a kill never reaches.
        #
        # Copying means the existing subtitles stay in place and readable the
        # whole time. _write_cues writes atomically, so they are replaced in
        # one step at the very end or not at all.
        backup = layout.backup_srt_for(video, now=now())
        shutil.copy2(current, backup)

    try:
        result = _write_cues(video, transcriber, language, progress=progress,
                             profile=profile)
    except Exception:
        # Nothing to restore - the original was never removed. The backup is
        # now a duplicate of a file that did not change, so drop it.
        if backup is not None:
            backup.unlink(missing_ok=True)
        raise

    fsutil.set_owner(result.srt_path, uid, gid)
    fsutil.write_marker(layout.reprocessed_marker(folder, video), f"{now().isoformat()}\n")
    fsutil.prune_backups(video.parent, f"{video.stem}.srt.*.bak", keep=keep_backups)
    return result
