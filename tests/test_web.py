"""Tests for the web UI.

Uses FastAPI's TestClient - no server, no browser, no network. Asserts on the
rendered HTML, so a broken template is a failing test rather than a page that
only breaks when someone opens it.
"""

from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from subwright import config
from subwright.db import Database
from subwright.web.app import create_app

WHEN = datetime(2026, 8, 29, 12, 0, 0)


@pytest.fixture
def env(tmp_path: Path):
    db = Database(tmp_path / "subwright.db")
    settings, _ = config.resolve([], env={})
    recorded: dict = {"saved": None, "cancelled": 0, "requeued": []}

    app = create_app(
        db,
        settings,
        status_provider=lambda: {
            "running": False, "current_file": None, "current_kind": None,
            "started_at": None, "media_duration": 0.0, "last_error": None,
            "processed": 3, "failed": 1,
        },
        on_settings_saved=lambda s: recorded.__setitem__("saved", s),
        cancel_current=lambda: recorded.__setitem__("cancelled", recorded["cancelled"] + 1),
        requeue=lambda row: recorded["requeued"].append(row["id"]),
        version="9.9.9",
    )
    return TestClient(app), db, recorded


# --- pages render ---

def test_dashboard_renders(env):
    client, _, _ = env
    r = client.get("/")
    assert r.status_code == 200
    assert "subwright" in r.text


def test_dashboard_shows_the_watch_folder_and_model(env):
    client, settings_db, _ = env
    r = client.get("/")
    # Compared without assuming a path separator: this is developed on Windows
    # and runs on Linux, and Path renders differently on each.
    assert "translate" in r.text
    assert "large-v3" in r.text


def test_dashboard_says_idle_when_nothing_is_running(env):
    client, _, _ = env
    assert "Idle" in client.get("/").text


def test_dashboard_lists_recent_jobs(env):
    client, db, _ = env
    job_id = db.start_job("ingest", Path("/w/ingest/Example.mkv"), when=WHEN)
    db.finish_job(job_id, cue_count=12, when=WHEN)
    assert "Example.mkv" in client.get("/").text


def test_status_fragment_renders_on_its_own(env):
    """HTMX polls this; it must be valid without the page around it."""
    client, _, _ = env
    r = client.get("/status")
    assert r.status_code == 200
    assert "Totals" in r.text


def test_history_page_renders(env):
    client, _, _ = env
    assert client.get("/history").status_code == 200


def test_history_can_filter_to_failures(env):
    client, db, _ = env
    ok = db.start_job("ingest", Path("/w/good.mkv"), when=WHEN)
    db.finish_job(ok, when=WHEN)
    bad = db.start_job("ingest", Path("/w/bad.mkv"), when=WHEN)
    db.fail_job(bad, "no audio stream", when=WHEN)

    text = client.get("/history?status=failed").text
    assert "bad.mkv" in text
    assert "good.mkv" not in text


def test_failed_job_shows_its_error_message(env):
    client, db, _ = env
    bad = db.start_job("ingest", Path("/w/bad.mkv"), when=WHEN)
    db.fail_job(bad, "no audio stream found", when=WHEN)
    assert "no audio stream found" in client.get("/history").text


def test_settings_page_renders_current_values(env):
    client, _, _ = env
    text = client.get("/settings").text
    assert "large-v3" in text
    assert "int8" in text


# --- saving settings ---

def _form(**overrides):
    base = {
        "watch_dir": "/mnt/data/translate",
        "model": "large-v3",
        "language": "ja",
        "poll_interval": 30,
        "device": "cuda",
        "compute_type": "int8",
        "settle_seconds": 10,
        "keep_backups": 3,
    }
    base.update(overrides)
    return base


def test_saving_settings_persists_them(env):
    client, db, _ = env
    client.post("/settings", data=_form(model="medium"), follow_redirects=False)
    assert db.load_settings()["model"] == "medium"


def test_saving_settings_notifies_the_worker(env):
    client, _, recorded = env
    client.post("/settings", data=_form(poll_interval=60), follow_redirects=False)
    assert recorded["saved"].poll_interval == 60


def test_saving_redirects_back_with_a_confirmation(env):
    client, _, _ = env
    r = client.post("/settings", data=_form(), follow_redirects=False)
    assert r.status_code == 303
    assert "saved=1" in r.headers["location"]


def test_invalid_settings_are_rejected_and_not_persisted(env):
    """A bad value must not be storable - it would break the next startup and
    could leave the UI unreachable."""
    client, db, _ = env
    r = client.post("/settings", data=_form(poll_interval=0), follow_redirects=False)
    assert r.status_code == 303
    assert "error=" in r.headers["location"]
    assert "poll_interval" not in db.load_settings()


def test_saved_settings_are_reflected_on_the_page(env):
    client, _, _ = env
    client.post("/settings", data=_form(model="small"), follow_redirects=False)
    assert "small" in client.get("/settings").text


# --- actions ---

def test_cancel_asks_the_worker_to_stop_the_current_job(env):
    client, _, recorded = env
    client.post("/cancel", follow_redirects=False)
    assert recorded["cancelled"] == 1


def test_retry_requeues_the_job(env):
    client, db, recorded = env
    bad = db.start_job("ingest", Path("/w/bad.mkv"), when=WHEN)
    db.fail_job(bad, "boom", when=WHEN)
    client.post(f"/jobs/{bad}/retry", follow_redirects=False)
    assert recorded["requeued"] == [bad]


def test_retrying_a_job_that_does_not_exist_is_handled(env):
    client, _, _ = env
    r = client.post("/jobs/9999/retry", follow_redirects=False)
    assert r.status_code == 303
    assert "error" in r.headers["location"]


# --- machine-readable ---

def test_healthz_is_cheap_and_returns_ok(env):
    client, _, _ = env
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_api_status_reports_counts_and_settings(env):
    client, db, _ = env
    job_id = db.start_job("ingest", Path("/w/a.mkv"), when=WHEN)
    db.finish_job(job_id, when=WHEN)
    body = client.get("/api/status").json()
    assert body["counts"] == {"done": 1}
    assert body["settings"]["model"] == "large-v3"
    assert body["version"] == "9.9.9"
