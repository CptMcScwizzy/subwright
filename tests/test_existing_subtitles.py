"""Reusing subtitles that already exist instead of transcribing.

Extracting a track that is already in the file takes about a second;
transcribing the same film takes minutes of GPU. But the saving is worthless if
it ever produces subtitles that are wrong, so most of what follows is about
what must NOT be reused, and about falling back cleanly when reuse fails.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from subwright import existing, jobs, layout
from subwright.mediaprobe import SubtitleStream, choose_english_stream, parse_streams

from .fakes import FakeTranscriber

NOW = datetime(2026, 1, 2, 3, 4, 5)

SRT = """1
00:00:01,000 --> 00:00:03,000
Already subtitled.

2
00:00:04,000 --> 00:00:06,000
No GPU was harmed.
"""


def _clock():
    return NOW


def _drop(tmp_path: Path, name: str = "Foo.mkv") -> tuple[Path, Path]:
    ingest = tmp_path / "ingest"
    ingest.mkdir(parents=True, exist_ok=True)
    video = ingest / name
    video.write_bytes(b"data")
    return tmp_path, video


class FakeProber:
    """Canned ffprobe results. No ffmpeg, no media files."""

    def __init__(self, streams=None, *, extract_text: str | None = SRT,
                 fail_extract: Exception | None = None):
        self.streams = streams or []
        self.extract_text = extract_text
        self.fail_extract = fail_extract
        self.extracted: list[tuple[Path, int]] = []

    def subtitle_streams(self, path: Path):
        return list(self.streams)

    def extract(self, path: Path, index: int, target: Path) -> None:
        self.extracted.append((path, index))
        if self.fail_extract:
            raise self.fail_extract
        target.write_text(self.extract_text or "", encoding="utf-8")


# --- sidecar files ---

def test_a_subtitle_dropped_beside_the_video_is_used_instead_of_transcribing(tmp_path):
    base, video = _drop(tmp_path)
    video.with_suffix(".srt").write_text(SRT, encoding="utf-8")

    transcriber = FakeTranscriber()
    result = jobs.run_ingest(video, base, transcriber, language="ja", now=_clock)

    assert transcriber.calls == [], "the GPU was used despite subtitles being provided"
    assert result.source == "sidecar"
    assert (base / "Foo" / "Foo.srt").read_text(encoding="utf-8") == SRT
    assert (base / "Foo" / ".translated").is_file()


def test_the_sidecar_follows_the_video_instead_of_being_stranded(tmp_path):
    """This was a real bug: only the video was moved, so a perfectly good
    subtitle file was left behind in the drop folder and the video was
    transcribed from scratch anyway."""
    base, video = _drop(tmp_path)
    sidecar = video.with_suffix(".srt")
    sidecar.write_text(SRT, encoding="utf-8")

    jobs.run_ingest(video, base, FakeTranscriber(), language="ja", now=_clock)

    assert not sidecar.exists(), "the subtitle file was left in the drop folder"
    assert (base / "Foo" / "Foo.srt").is_file()


def test_a_plex_style_english_sidecar_is_recognised(tmp_path):
    base, video = _drop(tmp_path)
    (video.parent / "Foo.en.srt").write_text(SRT, encoding="utf-8")

    transcriber = FakeTranscriber()
    result = jobs.run_ingest(video, base, transcriber, language="ja", now=_clock)

    assert transcriber.calls == []
    assert result.source == "sidecar"
    # Renamed to the plain name, which is what Plex and Stash look for.
    assert (base / "Foo" / "Foo.srt").read_text(encoding="utf-8") == SRT


def test_a_sidecar_belonging_to_a_different_video_is_ignored(tmp_path):
    base, video = _drop(tmp_path)
    (video.parent / "SomethingElse.srt").write_text(SRT, encoding="utf-8")

    transcriber = FakeTranscriber()
    result = jobs.run_ingest(video, base, transcriber, language="ja", now=_clock)

    assert len(transcriber.calls) == 1, "an unrelated subtitle file was used"
    assert result.source == "transcribed"
    assert (video.parent / "SomethingElse.srt").is_file(), "it was not ours to move"


def test_an_empty_sidecar_is_ignored_and_the_video_is_transcribed(tmp_path):
    """Zero-byte and half-written subtitle files are common in download folders."""
    base, video = _drop(tmp_path)
    video.with_suffix(".srt").write_text("", encoding="utf-8")

    transcriber = FakeTranscriber()
    result = jobs.run_ingest(video, base, transcriber, language="ja", now=_clock)

    assert len(transcriber.calls) == 1
    assert result.source == "transcribed"


def test_a_sidecar_that_is_not_actually_a_subtitle_file_is_ignored(tmp_path):
    base, video = _drop(tmp_path)
    video.with_suffix(".srt").write_text(
        "<html><body>404 Not Found, your indexer is sad</body></html>", encoding="utf-8")

    transcriber = FakeTranscriber()
    result = jobs.run_ingest(video, base, transcriber, language="ja", now=_clock)

    assert result.source == "transcribed"
    assert len(transcriber.calls) == 1


def test_reuse_can_be_turned_off(tmp_path):
    base, video = _drop(tmp_path)
    video.with_suffix(".srt").write_text(SRT, encoding="utf-8")

    transcriber = FakeTranscriber()
    result = jobs.run_ingest(video, base, transcriber, language="ja", now=_clock,
                             reuse=False)

    assert len(transcriber.calls) == 1
    assert result.source == "transcribed"
    # Even with reuse off, the sidecar still follows the video rather than
    # being stranded - it is just overwritten by the new transcription.
    assert not video.with_suffix(".srt").exists()


# --- embedded tracks ---

def test_an_english_track_inside_the_video_is_extracted(tmp_path):
    base, video = _drop(tmp_path)
    prober = FakeProber([SubtitleStream(index=2, codec="subrip", language="eng")])

    transcriber = FakeTranscriber()
    result = jobs.run_ingest(video, base, transcriber, language="ja", now=_clock,
                             prober=prober)

    assert transcriber.calls == [], "the GPU was used despite an English track"
    assert result.source == "embedded"
    assert prober.extracted == [(base / "Foo" / "Foo.mkv", 2)]
    assert (base / "Foo" / "Foo.srt").read_text(encoding="utf-8") == SRT


def test_a_picture_based_track_is_not_extracted(tmp_path):
    """PGS and DVD subtitles are images. Converting them needs OCR, and
    extracting one anyway writes a file that looks fine and says nothing."""
    base, video = _drop(tmp_path)
    prober = FakeProber([
        SubtitleStream(index=2, codec="hdmv_pgs_subtitle", language="eng"),
    ])

    transcriber = FakeTranscriber()
    result = jobs.run_ingest(video, base, transcriber, language="ja", now=_clock,
                             prober=prober)

    assert prober.extracted == []
    assert result.source == "transcribed"
    assert len(transcriber.calls) == 1


def test_a_forced_track_is_not_used_as_the_whole_subtitle(tmp_path):
    """A forced track is the handful of lines where someone speaks a foreign
    language. Reusing it would mark the job done with almost the entire film
    unsubtitled - worse than not reusing at all."""
    base, video = _drop(tmp_path)
    prober = FakeProber([
        SubtitleStream(index=2, codec="subrip", language="eng", forced=True),
    ])

    result = jobs.run_ingest(video, base, FakeTranscriber(), language="ja",
                             now=_clock, prober=prober)

    assert prober.extracted == []
    assert result.source == "transcribed"


def test_a_non_english_track_is_not_used(tmp_path):
    """The output of this application is English. A Japanese track is the
    input, not the answer."""
    base, video = _drop(tmp_path)
    prober = FakeProber([SubtitleStream(index=2, codec="subrip", language="jpn")])

    result = jobs.run_ingest(video, base, FakeTranscriber(), language="ja",
                             now=_clock, prober=prober)

    assert prober.extracted == []
    assert result.source == "transcribed"


def test_a_failed_extraction_falls_back_to_transcribing(tmp_path):
    """Reuse is an optimisation. It must never turn a file that would have
    worked into a failed job."""
    base, video = _drop(tmp_path)
    prober = FakeProber([SubtitleStream(index=2, codec="subrip", language="eng")],
                        fail_extract=RuntimeError("ffmpeg fell over"))

    transcriber = FakeTranscriber()
    result = jobs.run_ingest(video, base, transcriber, language="ja", now=_clock,
                             prober=prober)

    assert result.source == "transcribed"
    assert len(transcriber.calls) == 1
    assert (base / "Foo" / "Foo.srt").is_file()
    assert (base / "Foo" / ".translated").is_file()


def test_an_extraction_that_produces_nonsense_falls_back_to_transcribing(tmp_path):
    """ffmpeg exits 0 on some tracks while writing nothing usable."""
    base, video = _drop(tmp_path)
    prober = FakeProber([SubtitleStream(index=2, codec="subrip", language="eng")],
                        extract_text="")

    transcriber = FakeTranscriber()
    result = jobs.run_ingest(video, base, transcriber, language="ja", now=_clock,
                             prober=prober)

    assert result.source == "transcribed"
    assert len(transcriber.calls) == 1
    assert not (base / "Foo" / "Foo.srt.tmp").exists(), "scratch file left behind"


def test_no_prober_means_embedded_tracks_are_simply_never_looked_for(tmp_path):
    base, video = _drop(tmp_path)
    result = jobs.run_ingest(video, base, FakeTranscriber(), language="ja",
                             now=_clock, prober=None)
    assert result.source == "transcribed"


def test_a_sidecar_wins_over_an_embedded_track(tmp_path):
    """Someone put the sidecar there deliberately; the embedded track is
    whatever the muxer happened to include."""
    base, video = _drop(tmp_path)
    video.with_suffix(".srt").write_text(SRT, encoding="utf-8")
    prober = FakeProber([SubtitleStream(index=2, codec="subrip", language="eng")])

    result = jobs.run_ingest(video, base, FakeTranscriber(), language="ja",
                             now=_clock, prober=prober)

    assert result.source == "sidecar"
    assert prober.extracted == []


# --- reprocess must never reuse ---

def test_reprocessing_regenerates_rather_than_reusing_what_is_there(tmp_path):
    """Putting a file in reprocess/ means "do this again". Reusing the existing
    subtitles would make it a no-op, which is the exact opposite."""
    folder = tmp_path / "reprocess"
    folder.mkdir(parents=True)
    video = folder / "Foo.mkv"
    video.write_bytes(b"data")
    layout.srt_for(video).write_text(SRT, encoding="utf-8")

    transcriber = FakeTranscriber()
    result = jobs.run_reprocess(video, folder, transcriber, language="ja", now=_clock)

    assert len(transcriber.calls) == 1, "reprocess reused the subtitles it was told to redo"
    assert result.source == "transcribed"
    assert layout.srt_for(video).read_text(encoding="utf-8") != SRT


# --- stream selection, in isolation ---

def test_a_plain_english_track_is_preferred_over_the_hearing_impaired_one():
    """SDH interleaves [door creaks] with the dialogue. Both are real, one is
    cleaner."""
    chosen = choose_english_stream([
        SubtitleStream(index=2, codec="subrip", language="eng", hearing_impaired=True),
        SubtitleStream(index=3, codec="subrip", language="eng"),
    ])
    assert chosen.index == 3


def test_the_hearing_impaired_track_is_used_when_it_is_the_only_one():
    chosen = choose_english_stream([
        SubtitleStream(index=2, codec="subrip", language="eng", hearing_impaired=True),
    ])
    assert chosen.index == 2


def test_a_track_labelled_english_only_in_its_title_is_still_found():
    """Some muxers leave the language tag empty."""
    chosen = choose_english_stream([
        SubtitleStream(index=2, codec="subrip", title="English (SDH)"),
    ])
    assert chosen is not None


def test_ffprobe_output_is_read_into_streams():
    payload = {"streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
        {"index": 3, "codec_type": "subtitle", "codec_name": "subrip",
         "tags": {"language": "eng", "title": "Full"},
         "disposition": {"default": 1, "forced": 0}},
    ]}
    streams = parse_streams(payload)
    assert len(streams) == 1
    assert streams[0].index == 3
    assert streams[0].language == "eng"
    assert streams[0].default is True
    assert streams[0].forced is False


def test_a_file_with_no_subtitle_streams_reads_as_empty():
    assert parse_streams({"streams": [{"index": 0, "codec_type": "video"}]}) == []
    assert parse_streams({}) == []


def test_something_that_is_not_a_subtitle_file_is_recognised():
    assert existing.looks_like_srt(SRT) is True
    assert existing.looks_like_srt("<html>404</html>") is False
    assert existing.looks_like_srt("") is False
