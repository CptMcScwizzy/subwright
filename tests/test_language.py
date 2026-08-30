"""Source language: pinning it, auto-detecting it, and reporting what happened.

Whisper translates only INTO English, so there is no target language anywhere
in here. The last test says so, because it is the question everyone asks.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from subwright import config, jobs, languages, layout
from subwright.db import Database
from subwright.transcriber import TRANSCRIBE_OPTIONS, MediaInfo
from subwright.worker import Status

from .fakes import FakeTranscriber

NOW = datetime(2026, 1, 2, 3, 4, 5)


def _ingest(tmp_path: Path, name: str = "clip.mkv") -> tuple[Path, Path]:
    base = tmp_path / "translate"
    ingest = layout.ingest_dir(base)
    ingest.mkdir(parents=True)
    video = ingest / name
    video.write_bytes(b"data")
    return base, video


# --- choosing a language ---

def test_a_blank_language_means_auto_detect():
    assert config.Settings(language="").autodetect is True
    assert config.Settings(language="").language_or_none is None


def test_a_pinned_language_is_passed_to_whisper(tmp_path):
    base, video = _ingest(tmp_path)
    transcriber = FakeTranscriber()
    jobs.run_ingest(video, base, transcriber, language="ko", now=lambda: NOW)
    assert transcriber.calls[0][1] == "ko"


def test_auto_detect_passes_no_language_to_whisper(tmp_path):
    """None, not an empty string - faster-whisper treats "" as a language."""
    base, video = _ingest(tmp_path)
    transcriber = FakeTranscriber()
    settings = config.Settings(language="")
    jobs.run_ingest(video, base, transcriber,
                    language=settings.language_or_none, now=lambda: NOW)
    assert transcriber.calls[0][1] is None


def test_a_misspelt_language_is_rejected_before_any_gpu_time_is_spent():
    with pytest.raises(ValueError, match="unknown language"):
        config.Settings(language="japanese").validate()


def test_every_language_offered_in_the_ui_is_one_whisper_knows():
    offered = {code for _, opts in languages.choices() for code, _ in opts}
    assert offered == set(languages.LANGUAGES)


def test_the_common_languages_are_listed_before_the_rest():
    groups = languages.choices()
    assert groups[0][0] == "Common"
    assert [code for code, _ in groups[0][1]][:2] == ["ja", "ko"]


# --- reporting what was detected ---

def test_the_detected_language_is_recorded_on_the_job(tmp_path):
    base, video = _ingest(tmp_path)
    transcriber = FakeTranscriber()
    transcriber._info = MediaInfo(duration=60.0, detected_language="ja",
                                  language_probability=0.98)
    original = transcriber.transcribe

    def transcribe(path, language, profile=None):
        cues, _ = original(path, language, profile)
        return cues, transcriber._info

    transcriber.transcribe = transcribe

    result = jobs.run_ingest(video, base, transcriber, language=None, now=lambda: NOW)
    assert result.detected_language == "ja"
    assert result.language_probability == 0.98


def test_the_dashboard_shows_the_detected_language_while_the_job_runs():
    status = Status()
    status.begin(Path("clip.mkv"), "ingest", NOW)
    status.set_media_info(MediaInfo(duration=60.0, detected_language="ko",
                                    language_probability=0.91))
    snap = status.snapshot()
    assert snap["detected_language"] == "ko"
    assert snap["language_probability"] == 0.91


def test_a_language_code_is_shown_as_a_name():
    assert languages.name("ja") == "Japanese"
    assert languages.name("yue") == "Cantonese"
    assert languages.name(None) == "auto-detect"
    # An unknown code shows as itself rather than crashing the page.
    assert languages.name("zzz") == "zzz"


def test_a_weak_detection_is_flagged_rather_than_silently_trusted(tmp_path):
    """A wrong language does not fail - it produces fluent, confident, invented
    subtitles. Nothing else in the system would ever notice."""
    db = Database(tmp_path / "jobs.db")
    job_id = db.start_job("ingest", Path("mystery.mkv"))
    db.finish_job(job_id, cue_count=10, media_duration=60.0,
                  detected_language="cy", language_probability=0.31)
    row = db.recent_jobs()[0]
    assert row["language_probability"] < languages.LOW_CONFIDENCE


def test_a_database_written_before_language_was_recorded_still_opens(tmp_path):
    """Existing installs have a v1 jobs table. Opening one must migrate it, not
    fail and not lose the history already in it."""
    import sqlite3
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
            filename TEXT NOT NULL, source_path TEXT NOT NULL, output_path TEXT,
            status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
            cue_count INTEGER, media_duration REAL, error TEXT);
        INSERT INTO meta VALUES ('schema_version', '1');
        INSERT INTO jobs (kind, filename, source_path, status, started_at)
            VALUES ('ingest', 'older.mkv', '/x/older.mkv', 'done', '2026-01-01T00:00:00');
    """)
    conn.commit()
    conn.close()

    db = Database(path)
    rows = db.recent_jobs()
    assert len(rows) == 1, "existing history was lost by the migration"
    assert rows[0]["filename"] == "older.mkv"
    assert rows[0]["detected_language"] is None

    # And the migrated table accepts the new columns.
    job_id = db.start_job("ingest", Path("new.mkv"))
    db.finish_job(job_id, detected_language="ja", language_probability=0.99)
    assert db.recent_jobs()[0]["detected_language"] == "ja"


# --- the thing Whisper cannot do ---

def test_output_is_always_english_and_there_is_no_target_language_setting():
    """Whisper has two tasks: transcribe (same language) and translate (to
    English). There is no third. If this ever fails, someone has added a
    setting the model cannot honour."""
    assert TRANSCRIBE_OPTIONS["task"] == "translate"
    assert not any("target" in f.name for f in config.fields(config.Settings()))
