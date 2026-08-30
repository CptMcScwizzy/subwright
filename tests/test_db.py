"""Tests for settings storage and job history."""

from datetime import datetime
from pathlib import Path

from subwright.db import Database

WHEN = datetime(2026, 8, 29, 12, 0, 0)


def make_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "subwright.db")


def test_database_file_is_created(tmp_path: Path):
    db = make_db(tmp_path)
    assert db.path.exists()


def test_settings_round_trip(tmp_path: Path):
    db = make_db(tmp_path)
    db.save_settings({"model": "medium", "poll_interval": 15})
    assert db.load_settings() == {"model": "medium", "poll_interval": 15}


def test_settings_preserve_their_type(tmp_path: Path):
    """poll_interval must come back an int, not the string '15'."""
    db = make_db(tmp_path)
    db.save_settings({"poll_interval": 15})
    assert db.load_settings()["poll_interval"] == 15
    assert isinstance(db.load_settings()["poll_interval"], int)


def test_saving_a_setting_twice_updates_rather_than_duplicates(tmp_path: Path):
    db = make_db(tmp_path)
    db.save_settings({"model": "medium"})
    db.save_settings({"model": "small"})
    assert db.load_settings() == {"model": "small"}


def test_empty_settings_on_a_fresh_database(tmp_path: Path):
    assert make_db(tmp_path).load_settings() == {}


def test_job_lifecycle_from_running_to_done(tmp_path: Path):
    db = make_db(tmp_path)
    job_id = db.start_job("ingest", Path("/w/ingest/Foo.mkv"), when=WHEN)
    assert db.job(job_id)["status"] == "running"

    db.finish_job(job_id, output_path=Path("/w/Foo/Foo.srt"), cue_count=42,
                  media_duration=90.0, when=WHEN)
    row = db.job(job_id)
    assert row["status"] == "done"
    assert row["cue_count"] == 42
    assert row["error"] is None


def test_failed_job_records_the_error(tmp_path: Path):
    db = make_db(tmp_path)
    job_id = db.start_job("ingest", Path("/w/ingest/Bad.mkv"), when=WHEN)
    db.fail_job(job_id, "no audio stream", when=WHEN)
    row = db.job(job_id)
    assert row["status"] == "failed"
    assert "no audio stream" in row["error"]


def test_very_long_errors_are_truncated(tmp_path: Path):
    db = make_db(tmp_path)
    job_id = db.start_job("ingest", Path("/w/ingest/Bad.mkv"), when=WHEN)
    db.fail_job(job_id, "x" * 10_000, when=WHEN)
    assert len(db.job(job_id)["error"]) <= 2000


def test_jobs_still_running_at_startup_are_marked_interrupted(tmp_path: Path):
    """A killed job must not sit in the history claiming to be running."""
    db = make_db(tmp_path)
    db.start_job("ingest", Path("/w/ingest/Foo.mkv"), when=WHEN)
    assert db.mark_orphans_interrupted() == 1
    assert db.recent_jobs()[0]["status"] == "interrupted"


def test_marking_orphans_leaves_finished_jobs_alone(tmp_path: Path):
    db = make_db(tmp_path)
    job_id = db.start_job("ingest", Path("/w/ingest/Foo.mkv"), when=WHEN)
    db.finish_job(job_id, when=WHEN)
    assert db.mark_orphans_interrupted() == 0
    assert db.job(job_id)["status"] == "done"


def test_recent_jobs_are_newest_first(tmp_path: Path):
    db = make_db(tmp_path)
    for name in ("one.mkv", "two.mkv", "three.mkv"):
        db.start_job("ingest", Path("/w/ingest") / name, when=WHEN)
    assert [j["filename"] for j in db.recent_jobs()] == ["three.mkv", "two.mkv", "one.mkv"]


def test_recent_jobs_can_be_filtered_by_status(tmp_path: Path):
    db = make_db(tmp_path)
    ok = db.start_job("ingest", Path("/w/a.mkv"), when=WHEN)
    db.finish_job(ok, when=WHEN)
    bad = db.start_job("ingest", Path("/w/b.mkv"), when=WHEN)
    db.fail_job(bad, "boom", when=WHEN)

    assert [j["filename"] for j in db.recent_jobs(status="failed")] == ["b.mkv"]


def test_recent_jobs_respects_the_limit(tmp_path: Path):
    db = make_db(tmp_path)
    for i in range(10):
        db.start_job("ingest", Path(f"/w/{i}.mkv"), when=WHEN)
    assert len(db.recent_jobs(limit=3)) == 3


def test_counts_are_grouped_by_status(tmp_path: Path):
    db = make_db(tmp_path)
    a = db.start_job("ingest", Path("/w/a.mkv"), when=WHEN)
    db.finish_job(a, when=WHEN)
    b = db.start_job("ingest", Path("/w/b.mkv"), when=WHEN)
    db.fail_job(b, "boom", when=WHEN)
    assert db.counts() == {"done": 1, "failed": 1}


def test_reopening_an_existing_database_keeps_its_data(tmp_path: Path):
    path = tmp_path / "subwright.db"
    Database(path).save_settings({"model": "medium"})
    assert Database(path).load_settings() == {"model": "medium"}
