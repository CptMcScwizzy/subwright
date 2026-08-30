"""Self-test.

Runs the REAL pipeline against a temporary tree using a fake transcriber, then
checks the resulting folder layout is exactly what Plex and Stash expect, and
prints a plain PASS/FAIL table.

This exists because unit tests prove the code is right, but not that the
*image* is. It needs no GPU and no model, so it can be run anywhere, including
inside the shipped container:

    docker compose run --rm subwright --self-test

If this passes, the packaging, the Python environment and the whole job
pipeline are working. Only transcription itself is substituted.

Every check runs inside its own error guard: a broken pipeline must produce a
FAIL row, never a traceback. Someone reading this output should not have to
interpret a stack trace to learn that something is wrong.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from . import jobs, layout, scanner
from .srt import Cue
from .transcriber import MediaInfo

CUES = [
    Cue(0.0, 2.0, "First line."),
    Cue(2.0, 4.0, "Second line."),
    Cue(4.0, 30.0, "Third line, long enough that it must be capped."),
]


class _FakeTranscriber:
    """Mirrors tests/fakes.py. Duplicated deliberately: the test suite is not
    shipped inside the image, and this has to run there.

    `pace` and `count` exist only for --demo, so the progress bar and the live
    subtitle preview can be developed without a GPU. Both default to off, which
    keeps the self-test instant and its output byte-identical.
    """

    def __init__(self, *, pace: float = 0.0, count: int = 0) -> None:
        self.calls: list[tuple[Path, str | None]] = []
        self.pace = pace
        self.count = count

    def _cues(self):
        if not self.count:
            yield from CUES
            return
        # Spread evenly across the reported duration, so progress climbs steadily.
        step = 60.0 / self.count
        for i in range(self.count):
            if self.pace:
                time.sleep(self.pace)
            start = i * step
            yield Cue(start, start + min(step, 4.0),
                      f"Placeholder subtitle line {i + 1} of {self.count}.")

    def transcribe(self, path: Path, language, profile=None):
        self.calls.append((path, language))
        return self._cues(), MediaInfo(duration=60.0, detected_language="ja")


class _Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, label: str, predicate: Callable[[], bool]) -> None:
        """Record PASS/FAIL. An exception counts as FAIL, with its detail kept."""
        try:
            ok = bool(predicate())
            detail = ""
        except Exception as exc:  # noqa: BLE001 - any failure is a FAIL row
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        self.rows.append((label, ok, detail))

    def fail(self, label: str, detail: str) -> None:
        self.rows.append((label, False, detail))

    @property
    def passed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if ok)

    @property
    def failed(self) -> int:
        return len(self.rows) - self.passed

    def report(self) -> str:
        width = max((len(label) for label, _, _ in self.rows), default=10)
        lines = []
        for label, ok, detail in self.rows:
            lines.append(f"CHECK  {label.ljust(width)}  {'PASS' if ok else 'FAIL'}")
            if detail:
                lines.append(f"       {' ' * width}  {detail}")
        lines.append("")
        lines.append(f"RESULT {self.passed}/{len(self.rows)} PASS")
        return "\n".join(lines)


def _fixed_clock() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0)


def _inputs(base: Path) -> set[Path]:
    """Folders that hold work waiting to start, rather than results.

    Named literally rather than asked of layout.py, for the same reason as the
    checks below: a contract test that derives its expectations from the code it
    is checking cannot detect a change to that code.
    """
    return {base / "ingest", base / "reprocess"}


def _ingest_phase(base: Path, checks: _Checks) -> None:
    ingest = layout.ingest_dir(base)
    ingest.mkdir(parents=True, exist_ok=True)
    video = ingest / "sample.mkv"
    video.write_text("not really a video")
    folder = base / "sample"

    fake = _FakeTranscriber()
    try:
        jobs.run_ingest(video, base, fake, language="ja", now=_fixed_clock)
    except Exception as exc:  # noqa: BLE001
        checks.fail("ingest job completed", f"{type(exc).__name__}: {exc}")
        return

    def srt_text() -> str:
        return (folder / "sample.srt").read_text(encoding="utf-8")

    # Filenames below are written out LITERALLY, not derived from layout.py.
    # That is deliberate: if the checks asked layout.py what the marker is
    # called, a change there would move both the writer and the check together
    # and this would never notice. Stating the contract literally is what makes
    # it a contract test.
    checks.check("video moved out of ingest/", lambda: not video.exists())
    checks.check("output folder created", folder.is_dir)
    checks.check("video present in output folder", (folder / "sample.mkv").is_file)
    checks.check("subtitles written", (folder / "sample.srt").is_file)
    checks.check("success marker written", (folder / ".translated").is_file)
    checks.check("claim removed after success",
                 lambda: not (folder / ".processing").exists())
    checks.check("no scratch file left behind",
                 lambda: not (folder / "sample.srt.tmp").exists())
    checks.check("subtitles start at cue 1", lambda: srt_text().startswith("1\n"))
    checks.check("subtitle timestamps well formed",
                 lambda: "00:00:00,000 --> 00:00:02,000" in srt_text())
    checks.check("long cue capped at 5s",
                 lambda: "00:00:04,000 --> 00:00:09,000" in srt_text())
    checks.check("language passed to transcriber", lambda: fake.calls[0][1] == "ja")
    checks.check("completed folder not resumable",
                 lambda: scanner.find_resumable(base, exclude=_inputs(base)) == [])


def _stranger_phase(base: Path, checks: _Checks) -> None:
    """A folder this application did not create must never be adopted."""
    stranger = base / "SomethingIMadeMyself"
    stranger.mkdir(parents=True, exist_ok=True)
    (stranger / "video.mkv").write_text("x")
    (stranger / "video.srt").write_text("hand-edited subtitles")
    checks.check("pre-existing folder without a claim is ignored",
                 lambda: scanner.find_resumable(base, exclude=_inputs(base)) == [])


def _reuse_phase(base: Path, checks: _Checks) -> None:
    """Subtitles that already exist are used instead of the GPU.

    Runs against the real pipeline with a transcriber that records whether it
    was called, so this proves the saving actually happens rather than that the
    code merely exists.
    """
    ingest = base / "ingest"
    ingest.mkdir(parents=True, exist_ok=True)
    video = ingest / "hassubs.mkv"
    video.write_text("not really a video")
    sidecar = ingest / "hassubs.srt"
    sidecar.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nProvided, not transcribed.\n\n",
        encoding="utf-8",
    )
    folder = base / "hassubs"

    fake = _FakeTranscriber()
    try:
        result = jobs.run_ingest(video, base, fake, language="ja", now=_fixed_clock)
    except Exception as exc:  # noqa: BLE001
        checks.fail("reuse job completed", f"{type(exc).__name__}: {exc}")
        return

    checks.check("existing subtitles used instead of the GPU", lambda: fake.calls == [])
    checks.check("reused subtitles moved next to the video",
                 (folder / "hassubs.srt").is_file)
    checks.check("nothing left behind in ingest/", lambda: not sidecar.exists())
    checks.check("reused job still marked translated", (folder / ".translated").is_file)
    checks.check("reuse recorded as its own kind", lambda: result.source == "sidecar")


def _unusable_sidecar_phase(base: Path, checks: _Checks) -> None:
    """A file that is not a subtitle must not be mistaken for one."""
    ingest = base / "ingest"
    ingest.mkdir(parents=True, exist_ok=True)
    video = ingest / "badsubs.mkv"
    video.write_text("not really a video")
    # Comfortably over the minimum size, so this exercises the "does not look
    # like subtitles" check rather than the "too small" one. What an indexer
    # actually returns when it has nothing.
    (ingest / "badsubs.srt").write_text(
        "<!doctype html><html><head><title>404 Not Found</title></head>"
        "<body><h1>Not Found</h1><p>No subtitles for this release.</p></body></html>",
        encoding="utf-8",
    )

    fake = _FakeTranscriber()
    try:
        result = jobs.run_ingest(video, base, fake, language="ja", now=_fixed_clock)
    except Exception as exc:  # noqa: BLE001
        checks.fail("unusable sidecar handled", f"{type(exc).__name__}: {exc}")
        return

    checks.check("a file that is not subtitles is transcribed instead",
                 lambda: result.source == "transcribed" and len(fake.calls) == 1)


def _reprocess_phase(base: Path, checks: _Checks) -> None:
    reprocess = layout.reprocess_dir(base)
    reprocess.mkdir(parents=True, exist_ok=True)
    rvideo = reprocess / "existing.mkv"
    rvideo.write_text("x")
    (reprocess / "existing.srt").write_text("the previous subtitles")

    try:
        jobs.run_reprocess(rvideo, reprocess, _FakeTranscriber(), language=None, now=_fixed_clock)
    except Exception as exc:  # noqa: BLE001
        checks.fail("reprocess job completed", f"{type(exc).__name__}: {exc}")
        return

    # Literal names again - see the note in _ingest_phase.
    checks.check("reprocess left the video in place", rvideo.is_file)
    checks.check("reprocess rewrote the subtitles", (reprocess / "existing.srt").is_file)
    checks.check(
        "previous subtitles backed up",
        lambda: any(
            p.read_text() == "the previous subtitles"
            for p in reprocess.glob("existing.srt.*.bak")
        ),
    )
    checks.check("reprocess marker written so it does not loop",
                 (reprocess / ".reprocessed_existing").is_file)


def run(verbose: bool = True) -> int:
    """Returns 0 if every check passes, 1 otherwise."""
    checks = _Checks()
    try:
        with tempfile.TemporaryDirectory(prefix="subwright-selftest-") as tmp:
            base = Path(tmp)
            _ingest_phase(base, checks)
            _stranger_phase(base, checks)
            _reuse_phase(base, checks)
            _unusable_sidecar_phase(base, checks)
            _reprocess_phase(base, checks)
    except Exception:  # noqa: BLE001 - never let the self-test itself explode
        checks.fail("self-test ran to completion", traceback.format_exc(limit=3))

    if verbose:
        print(checks.report())
    return 0 if checks.failed == 0 else 1
