"""Tests for the wiring between the worker, the database and the web app."""

from pathlib import Path

import pytest

from subwright import config, layout
from subwright.db import Database
from subwright.runtime import Runtime
from tests.fakes import FakeTranscriber


def build(tmp_path: Path, transcriber=None):
    settings, _ = config.resolve(["-w", str(tmp_path), "-p", "1"], env={})
    db = Database(tmp_path / "cfg" / "subwright.db")
    return Runtime(settings, db, transcriber or FakeTranscriber()), db, settings


def put_ingest(base: Path, name: str) -> Path:
    folder = layout.ingest_dir(base)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text("fake video")
    return p


def test_successful_job_is_recorded_in_history(tmp_path: Path):
    rt, db, _ = build(tmp_path)
    put_ingest(tmp_path, "Foo.mkv")
    rt.worker.monotonic = lambda: 9e9   # past the settle gate
    rt.worker.run_once()

    jobs = db.recent_jobs()
    assert len(jobs) == 1
    assert jobs[0]["status"] == "done"
    assert jobs[0]["filename"] == "Foo.mkv"
    assert jobs[0]["cue_count"] == 3


def test_failed_job_is_recorded_with_its_error(tmp_path: Path):
    rt, db, _ = build(tmp_path, FakeTranscriber(raise_on_call=RuntimeError("no audio")))
    put_ingest(tmp_path, "Bad.mkv")
    rt.worker.monotonic = lambda: 9e9
    rt.worker.run_once()

    job = db.recent_jobs()[0]
    assert job["status"] == "failed"
    assert "no audio" in job["error"]


def test_settings_changes_reach_the_running_worker(tmp_path: Path):
    rt, _, settings = build(tmp_path)
    new = config.Settings(**{**settings.__dict__, "poll_interval": 99, "language": "fr"})
    rt.apply_settings(new)
    assert rt.worker.poll_interval == 99
    assert rt.worker.language == "fr"


def test_cancel_stops_the_worker_picking_up_more_work(tmp_path: Path):
    rt, _, _ = build(tmp_path)
    rt.cancel_current()
    assert rt.worker.stopping


def test_retry_copies_the_file_back_into_ingest(tmp_path: Path):
    rt, db, _ = build(tmp_path)
    put_ingest(tmp_path, "Foo.mkv")
    rt.worker.monotonic = lambda: 9e9
    rt.worker.run_once()

    job = db.recent_jobs()[0]
    rt.requeue(job)
    assert (layout.ingest_dir(tmp_path) / "Foo.mkv").exists()


def test_retry_does_not_remove_the_original(tmp_path: Path):
    """A retry must never be able to lose the video."""
    rt, db, _ = build(tmp_path)
    put_ingest(tmp_path, "Foo.mkv")
    rt.worker.monotonic = lambda: 9e9
    rt.worker.run_once()

    rt.requeue(db.recent_jobs()[0])
    assert (tmp_path / "Foo" / "Foo.mkv").exists()


def test_retry_refuses_when_the_file_is_already_queued(tmp_path: Path):
    rt, db, _ = build(tmp_path)
    put_ingest(tmp_path, "Foo.mkv")
    rt.worker.monotonic = lambda: 9e9
    rt.worker.run_once()

    job = db.recent_jobs()[0]
    rt.requeue(job)
    with pytest.raises(FileExistsError):
        rt.requeue(job)


def test_retry_reports_clearly_when_the_file_is_gone(tmp_path: Path):
    rt, db, _ = build(tmp_path)
    job_id = db.start_job("ingest", tmp_path / "ingest" / "Vanished.mkv")
    db.fail_job(job_id, "boom")
    with pytest.raises(FileNotFoundError):
        rt.requeue(db.job(job_id))


def test_jobs_left_running_by_a_previous_process_are_marked_interrupted(tmp_path: Path):
    rt, db, _ = build(tmp_path)
    db.start_job("ingest", tmp_path / "ingest" / "Killed.mkv")
    rt.start()
    try:
        assert db.recent_jobs()[0]["status"] == "interrupted"
    finally:
        rt.stop(timeout=5)
