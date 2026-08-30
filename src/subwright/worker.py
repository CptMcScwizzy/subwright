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

from . import jobs, scanner
from .mediaprobe import Prober
from .rules import WatchRule
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
    current_rule: str | None = None
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
                "current_rule": self.current_rule,
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

    def begin(self, path: Path, kind: str, when: datetime,
              rule: str | None = None) -> None:
        with self._lock:
            self.current_file = path.name
            self.current_kind = kind
            self.current_rule = rule
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
            self.current_rule = None
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
        rules: list[WatchRule],
        transcriber: Transcriber,
        *,
        poll_interval: int = 30,
        settle_seconds: int = scanner.DEFAULT_SETTLE_SECONDS,
        keep_backups: int = 3,
        uid: int = 1000,
        gid: int = 1000,
        clock: Callable[[], datetime] = datetime.now,
        monotonic: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        status: Status | None = None,
        prober: Prober | None = None,
        reuse_subtitles: bool = True,
        write_reports: bool = False,
        model: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8",
        on_job_done: Callable[[str, jobs.JobResult], None] | None = None,
        on_job_failed: Callable[[str, Path, Exception], None] | None = None,
    ) -> None:
        self.rules = rules
        self.transcriber = transcriber
        self.poll_interval = poll_interval
        self.settle_seconds = settle_seconds
        self.keep_backups = keep_backups
        self.uid = uid
        self.gid = gid
        self.clock = clock
        self.monotonic = monotonic
        self.sleep = sleep
        self.status = status or Status()
        self.prober = prober
        self.reuse_subtitles = reuse_subtitles
        self.write_reports = write_reports
        # Only ever reported, never acted on - the model is loaded by the
        # transcriber. They are here so the report can say what produced it.
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.on_job_done = on_job_done
        self.on_job_failed = on_job_failed
        self._stop = threading.Event()
        # Files the UI has asked to run again, by absolute path. A list rather
        # than a folder scan because the point of a redo is to regenerate
        # subtitles where the video already IS - moving it, or symlinking it
        # somewhere, would write the new subtitles in the wrong place.
        self._redo: list[Path] = []
        self._redo_lock = threading.Lock()

    def stop(self) -> None:
        """Ask the loop to finish after the current job. Safe from a signal handler."""
        self._stop.set()

    def request_redo(self, video: Path) -> None:
        """Queue a finished file to be transcribed again, in place.

        Called from the web thread, so it only appends to a list under a lock;
        the work happens on the worker thread like everything else.
        """
        with self._redo_lock:
            if video not in self._redo:
                self._redo.append(video)

    @property
    def pending_redo(self) -> list[Path]:
        with self._redo_lock:
            return list(self._redo)

    def _take_redo(self) -> Path | None:
        with self._redo_lock:
            return self._redo.pop(0) if self._redo else None

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def run_once(self) -> int:
        """One pass over every enabled rule. Returns jobs attempted.

        Order within a rule is resume, then ingest, then reprocess: finishing
        interrupted work before starting new work stops a restart building up a
        backlog of half-done folders.

        Rules are taken in order and each is drained before the next, so a busy
        first folder can delay a later one. That is a consequence of being
        single-threaded, and is preferable to interleaving, which would make
        "what is it doing right now?" much harder to answer.
        """
        attempted = 0

        # Redos first. Someone is sitting there waiting for the result, which
        # a folder full of new arrivals is not.
        while not self.stopping:
            video = self._take_redo()
            if video is None:
                break
            attempted += 1
            self._dispatch("redo", video, self._rule_owning(video))

        for rule in self.rules:
            if not rule.enabled:
                continue
            attempted += self._run_rule(rule)
            if self.stopping:
                break
        return attempted

    def _rule_owning(self, video: Path) -> WatchRule:
        """Whose settings apply to this file, matched on where it lives."""
        for rule in self.rules:
            try:
                video.relative_to(rule.output)
            except ValueError:
                continue
            return rule
        return self.rules[0]

    def _run_rule(self, rule: WatchRule) -> int:
        rule.ingest.mkdir(parents=True, exist_ok=True)
        rule.output.mkdir(parents=True, exist_ok=True)
        if rule.reprocess is not None:
            rule.reprocess.mkdir(parents=True, exist_ok=True)

        attempted = 0
        for video in scanner.find_resumable(rule.output, exclude=rule.excluded_dirs):
            if self.stopping:
                return attempted
            attempted += 1
            self._dispatch("resume", video, rule)

        now = self.monotonic()
        for video in scanner.find_ingest(
            rule.ingest, now=now, settle_seconds=self.settle_seconds
        ):
            if self.stopping:
                return attempted
            attempted += 1
            self._dispatch("ingest", video, rule)

        now = self.monotonic()
        for video in scanner.find_reprocess(
            rule.reprocess, now=now, settle_seconds=self.settle_seconds
        ):
            if self.stopping:
                return attempted
            attempted += 1
            self._dispatch("reprocess", video, rule)

        return attempted

    def _dispatch(self, kind: str, video: Path, rule: WatchRule) -> None:
        self.status.begin(video, kind, self.clock(), rule=rule.name)
        log.info("%s [%s]: %s", kind, rule.name, video.name)
        language = rule.language_or_none
        profile = rule.audio_profile
        try:
            if kind == "ingest":
                result = jobs.run_ingest(
                    video, rule.output, self.transcriber, language=language,
                    now=self.clock, uid=self.uid, gid=self.gid,
                    progress=self.status, profile=profile,
                    prober=self.prober, reuse=self.reuse_subtitles,
                )
            elif kind == "redo":
                # Regenerated where the video already lives. run_reprocess
                # keeps the previous subtitles as a timestamped .bak, which is
                # what makes comparing two audio profiles safe.
                result = jobs.run_reprocess(
                    video, video.parent, self.transcriber, language=language,
                    now=self.clock, keep_backups=self.keep_backups,
                    uid=self.uid, gid=self.gid, progress=self.status,
                    profile=profile,
                )
            elif kind == "resume":
                result = jobs.run_resume(
                    video, self.transcriber, language=language,
                    now=self.clock, uid=self.uid, gid=self.gid,
                    progress=self.status, profile=profile,
                )
            else:
                result = jobs.run_reprocess(
                    video, rule.reprocess, self.transcriber, language=language,
                    now=self.clock, keep_backups=self.keep_backups,
                    uid=self.uid, gid=self.gid, progress=self.status,
                    profile=profile,
                )
        except Exception as exc:
            # One bad file must never stop the loop.
            log.exception("%s [%s] failed for %s", kind, rule.name, video.name)
            self.status.finish(ok=False, error=f"{video.name}: {exc}")
            if self.on_job_failed:
                self.on_job_failed(kind, video, exc)
            return
        if result.source == "transcribed":
            log.info("%s [%s] done: %s (%d cues)", kind, rule.name, video.name,
                     result.cue_count)
        else:
            log.info("%s [%s] done: %s (%d cues, reused from %s - no GPU time used)",
                     kind, rule.name, video.name, result.cue_count, result.source_detail)
        if self.write_reports:
            jobs.write_report(
                result, profile=profile, model=self.model, device=self.device,
                compute_type=self.compute_type, language=language,
                rule_name=rule.name, now=self.clock, uid=self.uid, gid=self.gid,
            )
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
