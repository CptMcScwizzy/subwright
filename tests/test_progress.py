"""Live progress reporting: the bar and the subtitle preview on the dashboard.

These cover the plumbing that lets the UI show what a running job is doing.
None of it may change what ends up on disk - the last test here is the one that
says so.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from subwright import jobs, layout
from subwright.srt import Cue
from subwright.transcriber import MediaInfo
from subwright.worker import PREVIEW_MAX_CHARS, Status

from .fakes import FakeTranscriber

NOW = datetime(2026, 1, 2, 3, 4, 5)


def _clock():
    return NOW


def _ingest(tmp_path: Path, name: str = "clip.mkv") -> tuple[Path, Path]:
    base = tmp_path / "translate"
    ingest = layout.ingest_dir(base)
    ingest.mkdir(parents=True)
    video = ingest / name
    video.write_bytes(b"data")
    return base, video


def test_progress_is_reported_while_a_job_runs_not_only_at_the_end(tmp_path):
    """The whole point: a caller sees cues as they are produced.

    If _write_cues went back to list(cues_iter), the recorder would receive
    everything in one burst after transcription finished and the UI would show
    nothing at all until the job was over.
    """
    seen: list[tuple[int, str]] = []

    class Recorder:
        def set_media_info(self, info): pass

        def observe_cue(self, cue):
            seen.append((len(seen), cue.text))

    base, video = _ingest(tmp_path)
    cues = [Cue(float(i), i + 1.0, f"line {i}") for i in range(5)]

    def watching():
        # Yielding one at a time and asserting in between proves the consumer is
        # lazy rather than draining the iterator up front.
        for i, cue in enumerate(cues):
            assert len(seen) == i, "cues are being drained before being observed"
            yield cue

    transcriber = FakeTranscriber(cues=cues)
    transcriber._cues = list(cues)
    original = transcriber.transcribe

    def transcribe(path, language, profile=None):
        _, info = original(path, language, profile)
        return watching(), info

    transcriber.transcribe = transcribe

    jobs.run_ingest(video, base, transcriber, language="ja", now=_clock,
                    progress=Recorder())

    assert [text for _, text in seen] == [f"line {i}" for i in range(5)]


def test_dashboard_shows_the_line_currently_being_transcribed(tmp_path):
    status = Status()
    status.begin(Path("clip.mkv"), "ingest", NOW)

    base, video = _ingest(tmp_path)
    transcriber = FakeTranscriber(cues=[
        Cue(0.0, 2.0, "The first thing said."),
        Cue(2.0, 4.0, "The last thing said."),
    ])
    jobs.run_ingest(video, base, transcriber, language="ja", now=_clock,
                    progress=status)

    # finish() has not been called, so the panel still reflects the job.
    assert status.snapshot()["last_cue"] == "The last thing said."


def test_progress_reaches_the_end_of_the_media(tmp_path):
    status = Status()
    status.begin(Path("clip.mkv"), "ingest", NOW)

    base, video = _ingest(tmp_path)
    transcriber = FakeTranscriber(
        cues=[Cue(0.0, 30.0, "first half"), Cue(30.0, 60.0, "second half")],
        duration=60.0,
    )
    jobs.run_ingest(video, base, transcriber, language="ja", now=_clock,
                    progress=status)

    snap = status.snapshot()
    assert snap["media_duration"] == 60.0
    assert snap["position"] == 60.0
    assert snap["progress"] == 1.0
    assert snap["cue_count"] == 2


def test_progress_stops_short_when_the_media_ends_in_silence(tmp_path):
    """Not a bug, and worth pinning so nobody 'fixes' it.

    Voice detection removes silence, so the final cue of a file that ends
    quietly is genuinely not at the end of the file. The bar stopping at 80% and
    then the job completing is the correct display.
    """
    status = Status()
    status.begin(Path("clip.mkv"), "ingest", NOW)

    base, video = _ingest(tmp_path)
    transcriber = FakeTranscriber(cues=[Cue(0.0, 48.0, "talking stops here")],
                                 duration=60.0)
    jobs.run_ingest(video, base, transcriber, language="ja", now=_clock,
                    progress=status)

    assert status.snapshot()["progress"] == 0.8


def test_progress_is_hidden_rather_than_wrong_when_duration_is_unknown():
    """A zero duration would divide by zero, or worse, render a full bar."""
    status = Status()
    status.begin(Path("clip.mkv"), "ingest", NOW)
    status.set_media_info(MediaInfo(duration=0.0))
    status.observe_cue(Cue(0.0, 5.0, "hello"))

    assert status.snapshot()["progress"] is None


def test_progress_never_exceeds_one_hundred_percent():
    status = Status()
    status.begin(Path("clip.mkv"), "ingest", NOW)
    status.set_media_info(MediaInfo(duration=10.0))
    # Whisper can emit a cue ending fractionally past the reported duration.
    status.observe_cue(Cue(0.0, 12.0, "over the end"))

    assert status.snapshot()["progress"] == 1.0


def test_the_bar_never_moves_backwards():
    status = Status()
    status.begin(Path("clip.mkv"), "ingest", NOW)
    status.set_media_info(MediaInfo(duration=100.0))
    status.observe_cue(Cue(0.0, 50.0, "halfway"))
    status.observe_cue(Cue(10.0, 20.0, "out of order"))

    assert status.snapshot()["position"] == 50.0


def test_preview_shows_a_multi_line_cue_on_one_line():
    status = Status()
    status.begin(Path("clip.mkv"), "ingest", NOW)
    status.observe_cue(Cue(0.0, 2.0, "  first part\n  second part  "))

    assert status.snapshot()["last_cue"] == "first part second part"


def test_preview_of_a_runaway_cue_is_truncated():
    status = Status()
    status.begin(Path("clip.mkv"), "ingest", NOW)
    status.observe_cue(Cue(0.0, 2.0, "x" * 5000))

    assert len(status.snapshot()["last_cue"]) == PREVIEW_MAX_CHARS


def test_a_new_job_does_not_show_the_previous_job_s_subtitle():
    status = Status()
    status.begin(Path("first.mkv"), "ingest", NOW)
    status.set_media_info(MediaInfo(duration=60.0))
    status.observe_cue(Cue(0.0, 30.0, "from the first video"))
    status.finish(ok=True)

    status.begin(Path("second.mkv"), "ingest", NOW)
    snap = status.snapshot()

    assert snap["last_cue"] is None
    assert snap["position"] == 0.0
    assert snap["cue_count"] == 0
    assert snap["progress"] is None


def test_an_idle_watcher_reports_no_progress():
    status = Status()
    status.begin(Path("clip.mkv"), "ingest", NOW)
    status.set_media_info(MediaInfo(duration=60.0))
    status.observe_cue(Cue(0.0, 60.0, "all done"))
    status.finish(ok=True)

    snap = status.snapshot()
    assert snap["running"] is False
    assert snap["last_cue"] is None
    assert snap["progress"] is None


def test_subtitles_are_identical_whether_or_not_progress_is_being_watched(tmp_path):
    """Progress reporting is observation only. If this ever fails, the feature
    has started changing the product, which is the one thing it must not do."""
    cues = [Cue(0.0, 2.0, "One."), Cue(2.0, 4.0, "Two."), Cue(4.0, 40.0, "Three.")]

    base_a, video_a = _ingest(tmp_path / "a")
    jobs.run_ingest(video_a, base_a, FakeTranscriber(cues=list(cues)),
                    language="ja", now=_clock, progress=None)

    base_b, video_b = _ingest(tmp_path / "b")
    jobs.run_ingest(video_b, base_b, FakeTranscriber(cues=list(cues)),
                    language="ja", now=_clock, progress=Status())

    written = [
        next(d for d in base.iterdir() if d.is_dir() and d.name == "clip")
        for base in (base_a, base_b)
    ]
    assert (written[0] / "clip.srt").read_text() == (written[1] / "clip.srt").read_text()
