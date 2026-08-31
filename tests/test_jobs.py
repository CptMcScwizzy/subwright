"""Tests for the three job types.

These run the real jobs.py code with a fake transcriber - no GPU, no model, no
video files. Only the model itself is substituted.
"""

from datetime import datetime
from pathlib import Path

import pytest

from subwright import jobs, layout, scanner
from tests.fakes import FakeTranscriber

NOW = datetime(2026, 8, 29, 14, 30, 5)


def clock():
    return NOW


def make_ingest_video(base: Path, name: str = "Foo.mkv") -> Path:
    folder = layout.ingest_dir(base)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text("fake video")
    return p


# --- ingest ---

def test_ingest_produces_exactly_the_expected_layout(tmp_path: Path):
    video = make_ingest_video(tmp_path)
    jobs.run_ingest(video, tmp_path, FakeTranscriber(), language="ja", now=clock)

    folder = tmp_path / "Foo"
    assert sorted(p.name for p in folder.iterdir()) == [".translated", "Foo.en.srt", "Foo.mkv"]


def test_ingest_empties_the_ingest_directory(tmp_path: Path):
    video = make_ingest_video(tmp_path)
    jobs.run_ingest(video, tmp_path, FakeTranscriber(), language="ja", now=clock)
    assert list(layout.ingest_dir(tmp_path).iterdir()) == []


def test_video_is_moved_before_transcription_starts(tmp_path: Path):
    """Pins the ordering contract: move first, then transcribe.

    That is what stops the same file being picked up twice by the next poll
    while a long transcription is still running.
    """
    video = make_ingest_video(tmp_path)
    seen = {}

    def on_call(path: Path):
        seen["ingest_empty"] = list(layout.ingest_dir(tmp_path).iterdir()) == []
        seen["video_relocated"] = path == tmp_path / "Foo" / "Foo.mkv"
        seen["exists"] = path.exists()

    jobs.run_ingest(video, tmp_path, FakeTranscriber(on_call=on_call), language="ja", now=clock)
    assert seen == {"ingest_empty": True, "video_relocated": True, "exists": True}


def test_ingest_collision_appends_timestamp(tmp_path: Path):
    (tmp_path / "Foo").mkdir()
    video = make_ingest_video(tmp_path)
    jobs.run_ingest(video, tmp_path, FakeTranscriber(), language="ja", now=clock)
    assert (tmp_path / "Foo_20260829_143005" / "Foo.mkv").exists()


def test_ingest_passes_the_configured_language_through(tmp_path: Path):
    video = make_ingest_video(tmp_path)
    fake = FakeTranscriber()
    jobs.run_ingest(video, tmp_path, fake, language="ja", now=clock)
    assert fake.calls[0][1] == "ja"


def test_claim_is_removed_after_success(tmp_path: Path):
    video = make_ingest_video(tmp_path)
    jobs.run_ingest(video, tmp_path, FakeTranscriber(), language="ja", now=clock)
    assert not layout.claim_marker(tmp_path / "Foo").exists()


def test_failed_ingest_writes_an_error_marker(tmp_path: Path):
    video = make_ingest_video(tmp_path)
    fake = FakeTranscriber(raise_on_call=RuntimeError("no audio stream"))
    with pytest.raises(RuntimeError):
        jobs.run_ingest(video, tmp_path, fake, language="ja", now=clock)
    assert "no audio stream" in layout.error_marker(tmp_path / "Foo").read_text()


def test_failed_ingest_leaves_no_partial_srt(tmp_path: Path):
    """The bug that used to lose jobs silently."""
    video = make_ingest_video(tmp_path)
    fake = FakeTranscriber(raise_after_cues=1)
    with pytest.raises(RuntimeError):
        jobs.run_ingest(video, tmp_path, fake, language="ja", now=clock)
    folder = tmp_path / "Foo"
    assert not (folder / "Foo.en.srt").exists()
    assert not (folder / "Foo.srt.tmp").exists()


def test_failed_ingest_drops_the_claim_so_it_is_not_retried_forever(tmp_path: Path):
    """A broken file must not be retried on every poll.

    The claim means "the process died mid-job". A caught transcription failure
    is different - the file is bad, not the run - so the claim is dropped and
    retrying becomes an explicit action.
    """
    video = make_ingest_video(tmp_path)
    with pytest.raises(RuntimeError):
        jobs.run_ingest(video, tmp_path, FakeTranscriber(raise_after_cues=1),
                        language="ja", now=clock)
    assert not layout.claim_marker(tmp_path / "Foo").exists()
    assert scanner.find_resumable(tmp_path) == []


# --- resume ---

def test_interrupted_job_is_found_and_completed_on_restart(tmp_path: Path):
    """End-to-end for the worst original bug.

    Simulates the process being KILLED mid-transcription: the video has been
    moved, the claim is present, no subtitles were written, and no error
    handler got the chance to run.
    """
    folder = tmp_path / "Foo"
    folder.mkdir()
    (folder / "Foo.mkv").write_text("fake video")
    layout.claim_marker(folder).write_text('{"source": "Foo.mkv"}')

    # Restart: the scanner finds it because the claim is still there.
    resumable = scanner.find_resumable(tmp_path)
    assert resumable == [tmp_path / "Foo" / "Foo.mkv"]

    jobs.run_resume(resumable[0], FakeTranscriber(), language="ja", now=clock)
    folder = tmp_path / "Foo"
    assert (folder / "Foo.en.srt").exists()
    assert layout.translated_marker(folder).exists()
    assert not layout.claim_marker(folder).exists()


def test_resume_clears_any_previous_error_marker(tmp_path: Path):
    folder = tmp_path / "Foo"
    folder.mkdir()
    (folder / "Foo.mkv").write_text("fake video")
    layout.claim_marker(folder).write_text("{}")
    layout.error_marker(folder).write_text("an earlier failure")
    jobs.run_resume(folder / "Foo.mkv", FakeTranscriber(), language="ja", now=clock)
    assert not layout.error_marker(folder).exists()


# --- reprocess ---

def make_reprocess_video(base: Path, name: str = "Bar.mkv") -> Path:
    folder = layout.reprocess_dir(base)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text("fake video")
    return p


def test_reprocess_never_moves_the_video(tmp_path: Path):
    video = make_reprocess_video(tmp_path)
    jobs.run_reprocess(video, tmp_path, FakeTranscriber(), language="ja", now=clock)
    assert video.exists()


def test_reprocess_writes_subtitles_beside_the_video(tmp_path: Path):
    video = make_reprocess_video(tmp_path)
    jobs.run_reprocess(video, tmp_path, FakeTranscriber(), language="ja", now=clock)
    assert (layout.reprocess_dir(tmp_path) / "Bar.en.srt").exists()


def test_reprocess_backs_up_existing_subtitles(tmp_path: Path):
    video = make_reprocess_video(tmp_path)
    layout.srt_for(video).write_text("the old subtitles")
    jobs.run_reprocess(video, tmp_path, FakeTranscriber(), language="ja", now=clock)
    backups = list(layout.reprocess_dir(tmp_path).glob("Bar.en.srt.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text() == "the old subtitles"


def test_reprocess_writes_its_marker_so_it_does_not_loop(tmp_path: Path):
    video = make_reprocess_video(tmp_path)
    jobs.run_reprocess(video, tmp_path, FakeTranscriber(), language="ja", now=clock)
    assert scanner.find_reprocess(tmp_path, now=9e9) == []


def test_failed_reprocess_restores_the_original_subtitles(tmp_path: Path):
    video = make_reprocess_video(tmp_path)
    layout.srt_for(video).write_text("the good subtitles")
    with pytest.raises(RuntimeError):
        jobs.run_reprocess(video, tmp_path, FakeTranscriber(raise_after_cues=1),
                           language="ja", now=clock)
    assert layout.srt_for(video).read_text() == "the good subtitles"


def test_no_backup_is_made_when_there_were_no_subtitles(tmp_path: Path):
    video = make_reprocess_video(tmp_path)
    jobs.run_reprocess(video, tmp_path, FakeTranscriber(), language="ja", now=clock)
    assert list(layout.reprocess_dir(tmp_path).glob("*.bak")) == []
