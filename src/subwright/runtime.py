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
import shutil
import threading
from pathlib import Path

from . import config, layout
from .db import Database
from .jobs import JobResult
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
            settings.watch_dir,
            transcriber,
            language=settings.language_or_none,
            poll_interval=settings.poll_interval,
            settle_seconds=settings.settle_seconds,
            keep_backups=settings.keep_backups,
            uid=settings.uid,
            gid=settings.gid,
            status=self.status,
            on_job_done=self._job_done,
            on_job_failed=self._job_failed,
        )
        # Patch in history recording at dispatch time.
        self.worker._dispatch = self._wrap_dispatch(self.worker._dispatch)
        self._thread: threading.Thread | None = None

    # --- history recording ---

    def _wrap_dispatch(self, inner):
        def dispatch(kind: str, video: Path) -> None:
            with self._lock:
                self._current_job_id = self.db.start_job(kind, video)
            try:
                inner(kind, video)
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
        self.worker.base = new.watch_dir
        self.worker.language = new.language_or_none
        self.worker.poll_interval = new.poll_interval
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

    def requeue(self, job_row: dict) -> None:
        """Put a failed file back into ingest/ so it is picked up again.

        Copies rather than moves when the source still exists in an output
        folder, so a retry can never lose the video.
        """
        source = Path(job_row["source_path"])
        ingest = layout.ingest_dir(self.settings.watch_dir)
        ingest.mkdir(parents=True, exist_ok=True)

        if not source.exists():
            # The ingest path is gone because the file was moved into its output
            # folder before transcription. Look for it there.
            candidate = self.settings.watch_dir / Path(source.stem) / source.name
            if candidate.exists():
                source = candidate
            else:
                raise FileNotFoundError(f"cannot find {source.name} to retry")

        target = ingest / source.name
        if target.exists():
            raise FileExistsError(f"{source.name} is already waiting in ingest")
        shutil.copy2(source, target)
        log.info("requeued %s for another attempt", source.name)
