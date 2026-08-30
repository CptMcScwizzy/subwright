"""The poll loop.

Receives its transcriber, clock and sleep function, so the loop is tested
without a GPU and without real sleeping.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import jobs, layout, scanner
from .transcriber import Transcriber

log = logging.getLogger(__name__)


@dataclass
class Status:
    """Live state, read by the web UI. Guarded by its own lock."""

    current_file: str | None = None
    current_kind: str | None = None
    started_at: datetime | None = None
    media_duration: float = 0.0
    last_error: str | None = None
    processed: int = 0
    failed: int = 0
    running: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "current_file": self.current_file,
                "current_kind": self.current_kind,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "media_duration": self.media_duration,
                "last_error": self.last_error,
                "processed": self.processed,
                "failed": self.failed,
                "running": self.running,
            }

    def begin(self, path: Path, kind: str, when: datetime) -> None:
        with self._lock:
            self.current_file = path.name
            self.current_kind = kind
            self.started_at = when
            self.running = True

    def finish(self, ok: bool, error: str | None = None) -> None:
        with self._lock:
            self.current_file = None
            self.current_kind = None
            self.started_at = None
            self.running = False
            if ok:
                self.processed += 1
            else:
                self.failed += 1
                self.last_error = error


class Worker:
    def __init__(
        self,
        base: Path,
        transcriber: Transcriber,
        *,
        language: str | None,
        poll_interval: int = 30,
        settle_seconds: int = scanner.DEFAULT_SETTLE_SECONDS,
        keep_backups: int = 3,
        uid: int = 1000,
        gid: int = 1000,
        clock: Callable[[], datetime] = datetime.now,
        monotonic: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        status: Status | None = None,
        on_job_done: Callable[[str, jobs.JobResult], None] | None = None,
        on_job_failed: Callable[[str, Path, Exception], None] | None = None,
    ) -> None:
        self.base = base
        self.transcriber = transcriber
        self.language = language
        self.poll_interval = poll_interval
        self.settle_seconds = settle_seconds
        self.keep_backups = keep_backups
        self.uid = uid
        self.gid = gid
        self.clock = clock
        self.monotonic = monotonic
        self.sleep = sleep
        self.status = status or Status()
        self.on_job_done = on_job_done
        self.on_job_failed = on_job_failed
        self._stop = threading.Event()

    def stop(self) -> None:
        """Ask the loop to finish after the current job. Safe from a signal handler."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def run_once(self) -> int:
        """One pass: resume, then ingest, then reprocess. Returns jobs attempted."""
        layout.ingest_dir(self.base).mkdir(parents=True, exist_ok=True)
        layout.reprocess_dir(self.base).mkdir(parents=True, exist_ok=True)

        attempted = 0
        for video in scanner.find_resumable(self.base):
            if self.stopping:
                return attempted
            attempted += 1
            self._dispatch("resume", video)

        now = self.monotonic()
        for video in scanner.find_ingest(self.base, now=now, settle_seconds=self.settle_seconds):
            if self.stopping:
                return attempted
            attempted += 1
            self._dispatch("ingest", video)

        now = self.monotonic()
        for video in scanner.find_reprocess(self.base, now=now, settle_seconds=self.settle_seconds):
            if self.stopping:
                return attempted
            attempted += 1
            self._dispatch("reprocess", video)

        return attempted

    def _dispatch(self, kind: str, video: Path) -> None:
        self.status.begin(video, kind, self.clock())
        log.info("%s: %s", kind, video.name)
        try:
            if kind == "ingest":
                result = jobs.run_ingest(
                    video, self.base, self.transcriber, language=self.language,
                    now=self.clock, uid=self.uid, gid=self.gid,
                )
            elif kind == "resume":
                result = jobs.run_resume(
                    video, self.transcriber, language=self.language,
                    now=self.clock, uid=self.uid, gid=self.gid,
                )
            else:
                result = jobs.run_reprocess(
                    video, self.base, self.transcriber, language=self.language,
                    now=self.clock, keep_backups=self.keep_backups,
                    uid=self.uid, gid=self.gid,
                )
        except Exception as exc:
            # One bad file must never stop the loop.
            log.exception("%s failed for %s", kind, video.name)
            self.status.finish(ok=False, error=f"{video.name}: {exc}")
            if self.on_job_failed:
                self.on_job_failed(kind, video, exc)
            return
        log.info("%s done: %s (%d cues)", kind, video.name, result.cue_count)
        self.status.finish(ok=True)
        if self.on_job_done:
            self.on_job_done(kind, result)

    def run_forever(self) -> None:
        while not self.stopping:
            try:
                self.run_once()
            except Exception:
                # A scan-level failure - an NFS stale handle, say - must not kill
                # the process. Log it and try again next pass.
                log.exception("scan failed; continuing")
            if self.stopping:
                break
            self.sleep(self.poll_interval)
        log.info("worker stopped")
