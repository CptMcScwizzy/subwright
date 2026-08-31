"""Wiring the worker, the database and the web app into one process.

The worker runs in a background thread rather than an asyncio task because
transcription is blocking and GPU-bound - it would stall the event loop and the
UI would freeze for the length of a job.

The two halves communicate through:
  - SQLite, for anything that must survive a restart
  - a Status object guarded by a lock, for live progress
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from . import config
from .db import Database
from .jobs import JobResult
from .mediaprobe import FfmpegProber
from .transcriber import Transcriber
from .worker import Status, Worker

log = logging.getLogger(__name__)


class Runtime:
    """Owns the worker thread and the objects the web app talks to."""

    def __init__(
        self,
        settings: config.Settings,
        db: Database,
        transcriber: Transcriber,
    ) -> None:
        self.settings = settings
        self.db = db
        self.status = Status()
        # A single slot, not a map keyed by path: the worker is strictly serial,
        # and ingest MOVES the video before finishing, so the path a job ends
        # with is not the one it started with.
        self._current_job_id: int | None = None
        self._lock = threading.Lock()

        self.worker = Worker(
            settings.effective_rules,
            transcriber,
            poll_interval=settings.poll_interval,
            settle_seconds=settings.settle_seconds,
            keep_backups=settings.keep_backups,
            uid=settings.uid,
            gid=settings.gid,
            status=self.status,
            # Real ffprobe/ffmpeg here; the tests substitute a fake, and None
            # simply means embedded tracks are never looked for.
            prober=FfmpegProber(),
            reuse_subtitles=settings.reuse_subtitles,
            write_reports=settings.write_reports,
            subtitle_tag=settings.subtitle_language_tag,
            model=settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
            on_job_done=self._job_done,
            on_job_failed=self._job_failed,
        )
        # Patch in history recording at dispatch time.
        self.worker._dispatch = self._wrap_dispatch(self.worker._dispatch)
        self._thread: threading.Thread | None = None

    # --- history recording ---

    def _wrap_dispatch(self, inner):
        def dispatch(kind: str, video: Path, rule) -> None:
            with self._lock:
                self._current_job_id = self.db.start_job(kind, video)
            try:
                inner(kind, video, rule)
            finally:
                # inner() swallows job errors and reports them through
                # on_job_failed, but if anything slips past, do not leave a row
                # stuck at 'running' forever.
                with self._lock:
                    stranded, self._current_job_id = self._current_job_id, None
                if stranded is not None:
                    self.db.fail_job(stranded, "job ended without reporting a result")
        return dispatch

    def _take_job_id(self) -> int | None:
        with self._lock:
            job_id, self._current_job_id = self._current_job_id, None
            return job_id

    def _job_done(self, kind: str, result: JobResult) -> None:
        job_id = self._take_job_id()
        if job_id is not None:
            self.db.finish_job(
                job_id,
                output_path=result.srt_path,
                cue_count=result.cue_count,
                media_duration=result.media_duration,
                detected_language=result.detected_language,
                language_probability=result.language_probability,
                source=result.source,
                source_detail=result.source_detail,
            )

    def _job_failed(self, kind: str, video: Path, exc: Exception) -> None:
        job_id = self._take_job_id()
        if job_id is not None:
            self.db.fail_job(job_id, f"{type(exc).__name__}: {exc}")

    # --- lifecycle ---

    def start(self) -> None:
        # Anything still marked running belongs to a previous process.
        orphans = self.db.mark_orphans_interrupted()
        if orphans:
            log.info("marked %d job(s) from a previous run as interrupted", orphans)

        self._thread = threading.Thread(target=self.worker.run_forever,
                                        name="subwright-worker", daemon=True)
        self._thread.start()
        log.info("worker thread started")

    def stop(self, timeout: float = 60.0) -> None:
        self.worker.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # --- callbacks the web app uses ---

    def status_snapshot(self) -> dict:
        return self.status.snapshot()

    def apply_settings(self, new: config.Settings) -> None:
        """Push changed settings into the running worker.

        Anything the loop reads each pass takes effect immediately. Device and
        compute type are not among them: the model is loaded once at startup, so
        those need a restart. The settings page says so.
        """
        self.settings = new
        self.worker.rules = new.effective_rules
        self.worker.poll_interval = new.poll_interval
        self.worker.reuse_subtitles = new.reuse_subtitles
        self.worker.write_reports = new.write_reports
        self.worker.subtitle_tag = new.subtitle_language_tag
        self.worker.model = new.model
        self.worker.settle_seconds = new.settle_seconds
        self.worker.keep_backups = new.keep_backups
        log.info("settings updated; watch_dir=%s model=%s (model change needs a restart)",
                 new.watch_dir, new.model)

    def cancel_current(self) -> None:
        """Ask the worker to stop after the current job.

        Note this does not abort a transcription already in flight - faster-whisper
        gives no way to interrupt one. It stops further work being picked up.
        """
        log.info("cancel requested from the UI")
        self.worker.stop()

    def reprocess(self, job_row: dict) -> Path:
        """Run a finished file again, in place, with whatever settings apply now.

        Different from retry, and deliberately so. Retry puts a file back
        through ingest, which moves it into a NEW output folder and leaves the
        old one behind. Reprocess regenerates the subtitles exactly where they
        are, keeping the previous ones as a timestamped .bak - which is what
        you want when comparing two audio profiles on the same file.

        Nothing is copied or moved: the video can be several gigabytes, and it
        is already in the right place.
        """
        video = self._locate(job_row)
        for rule in self.settings.effective_rules:
            if video.parent == rule.ingest:
                # Still waiting to be picked up. Transcribing it where it sits
                # would write subtitles into the drop folder and then have them
                # ingested as a sidecar, which is a confusing way to arrive at
                # the right answer.
                raise ValueError(
                    f"{video.name} is still in the {rule.name} drop folder and "
                    f"will be picked up on the next scan"
                )
        self.worker.request_redo(video)
        log.info("queued %s to be transcribed again in place", video.name)
        return video

    def _locate(self, job_row: dict) -> Path:
        """Where the video for this history row actually is now."""
        filename = job_row.get("filename") or ""
        output_path = job_row.get("output_path")
        if output_path:
            # output_path is the .srt; the video sits beside it.
            candidate = Path(output_path).parent / filename
            if candidate.exists():
                return candidate

        source = Path(job_row.get("source_path") or "")
        if source.exists():
            return source

        stem = Path(filename).stem
        for rule in self.settings.effective_rules:
            candidate = rule.output / stem / filename
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"cannot find {filename} any more")

