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


# The preview line is displayed and never stored, so there is no reason to hold
# an unbounded string in memory or push a wall of text into an HTML fragment
# every few seconds.
PREVIEW_MAX_CHARS = 240


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
    # Live progress within the current job, updated as each cue is produced.
    position: float = 0.0
    cue_count: int = 0
    last_cue: str | None = None
    detected_language: str | None = None
    language_probability: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _progress(self) -> float | None:
        """How far into the media transcription has reached, 0-1, or None.

        Caller must hold the lock.

        This is position-in-media and deliberately never becomes an ETA. The
        original script had one and it swung between eight hours and forty
        minutes on the same file, because VAD discards most of the audio and how
        much it will discard is not knowable in advance.

        A consequence worth knowing: a file that ends in silence finishes at
        around 0.95, not 1.0, because the last speech genuinely is not at the
        end of the file. The bar jumping to done from there is correct.
        """
        if self.media_duration <= 0:
            return None
        return min(1.0, max(0.0, self.position / self.media_duration))

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
                "position": self.position,
                "cue_count": self.cue_count,
                "last_cue": self.last_cue,
                "progress": self._progress(),
                "detected_language": self.detected_language,
                "language_probability": self.language_probability,
            }

    def _reset_progress(self) -> None:
        """Caller must hold the lock."""
        self.media_duration = 0.0
        self.position = 0.0
        self.cue_count = 0
        self.last_cue = None
        self.detected_language = None
        self.language_probability = None

    def begin(self, path: Path, kind: str, when: datetime) -> None:
        with self._lock:
            self.current_file = path.name
            self.current_kind = kind
            self.started_at = when
            self.running = True
            # Without this the previous job's last line would sit under the new
            # filename until the new job produced its first cue.
            self._reset_progress()

    def set_media_info(self, info) -> None:
        """Called once the media is open: its length, and what language it is.

        Both arrive at the same moment because faster-whisper does the audio
        decode, voice detection and language detection before returning, and
        only then hands back the lazy cue iterator.
        """
        with self._lock:
            self.media_duration = max(0.0, info.duration)
            self.detected_language = info.detected_language
            self.language_probability = info.language_probability

    def observe_cue(self, cue) -> None:
        """Called for each cue as it is produced.

        Runs inside the transcription loop between GPU batches, so it stays
        cheap and never touches the disk.
        """
        # Whisper does emit cues containing newlines; this is rendered as a
        # single line, so collapse the whitespace here rather than in the
        # template.
        text = " ".join(cue.text.split())
        with self._lock:
            self.cue_count += 1
            # max(): cue ends are normally monotonic, but a bar that can jump
            # backwards looks broken, and nothing guarantees the ordering.
            self.position = max(self.position, cue.end)
            self.last_cue = text[:PREVIEW_MAX_CHARS] or None

    def finish(self, ok: bool, error: str | None = None) -> None:
        with self._lock:
            self.current_file = None
            self.current_kind = None
            self.started_at = None
            self.running = False
            self._reset_progress()
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
                    progress=self.status,
                )
            elif kind == "resume":
                result = jobs.run_resume(
                    video, self.transcriber, language=self.language,
                    now=self.clock, uid=self.uid, gid=self.gid,
                    progress=self.status,
                )
            else:
                result = jobs.run_reprocess(
                    video, self.base, self.transcriber, language=self.language,
                    now=self.clock, keep_backups=self.keep_backups,
                    uid=self.uid, gid=self.gid, progress=self.status,
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
