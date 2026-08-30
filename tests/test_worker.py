"""Tests for the poll loop. No GPU, no real sleeping."""

from datetime import datetime
from pathlib import Path

from subwright import layout
from subwright.worker import Worker
from tests.fakes import FakeTranscriber

NOW = datetime(2026, 8, 29, 14, 30, 5)


def build(base: Path, transcriber=None, **kw):
    sleeps: list[float] = []
    w = Worker(
        base,
        transcriber or FakeTranscriber(),
        language="ja",
        clock=lambda: NOW,
        monotonic=lambda: 9e9,      # everything is always past the settle gate
        sleep=sleeps.append,        # record instead of sleeping
        **kw,
    )
    return w, sleeps


def put_video(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text("fake video")
    return p


def test_creates_ingest_and_reprocess_directories_on_first_pass(tmp_path: Path):
    w, _ = build(tmp_path)
    w.run_once()
    assert layout.ingest_dir(tmp_path).is_dir()
    assert layout.reprocess_dir(tmp_path).is_dir()


def test_processes_an_ingest_video(tmp_path: Path):
    put_video(layout.ingest_dir(tmp_path), "Foo.mkv")
    w, _ = build(tmp_path)
    assert w.run_once() == 1
    assert (tmp_path / "Foo" / "Foo.srt").exists()


def test_processes_ingest_and_reprocess_in_the_same_pass(tmp_path: Path):
    put_video(layout.ingest_dir(tmp_path), "Foo.mkv")
    put_video(layout.reprocess_dir(tmp_path), "Bar.mkv")
    w, _ = build(tmp_path)
    assert w.run_once() == 2


def test_resumable_work_is_done_before_new_ingest(tmp_path: Path):
    # A claimed, unfinished folder plus a fresh ingest file.
    folder = tmp_path / "Old"
    put_video(folder, "Old.mkv")
    layout.claim_marker(folder).write_text("{}")
    put_video(layout.ingest_dir(tmp_path), "New.mkv")

    fake = FakeTranscriber()
    w, _ = build(tmp_path, fake)
    w.run_once()
    assert [c[0].name for c in fake.calls] == ["Old.mkv", "New.mkv"]


def test_one_failing_file_does_not_stop_the_others(tmp_path: Path):
    put_video(layout.ingest_dir(tmp_path), "Bad.mkv")
    w, _ = build(tmp_path, FakeTranscriber(raise_on_call=RuntimeError("boom")))
    w.run_once()
    assert w.status.snapshot()["failed"] == 1
    # Loop survived and can run again.
    assert w.run_once() == 0


def test_status_reports_counts(tmp_path: Path):
    put_video(layout.ingest_dir(tmp_path), "Foo.mkv")
    w, _ = build(tmp_path)
    w.run_once()
    snap = w.status.snapshot()
    assert snap["processed"] == 1 and snap["failed"] == 0
    assert snap["running"] is False and snap["current_file"] is None


def test_run_forever_sleeps_the_poll_interval_between_passes(tmp_path: Path):
    w, sleeps = build(tmp_path, poll_interval=30)

    calls = {"n": 0}
    original = w.run_once

    def counted():
        calls["n"] += 1
        if calls["n"] >= 3:
            w.stop()
        return original()

    w.run_once = counted
    w.run_forever()
    assert sleeps == [30, 30]      # slept after passes 1 and 2, not after the last


def test_stop_prevents_starting_further_work(tmp_path: Path):
    put_video(layout.ingest_dir(tmp_path), "Foo.mkv")
    put_video(layout.ingest_dir(tmp_path), "Bar.mkv")
    w, _ = build(tmp_path)
    w.stop()
    assert w.run_once() == 0


def test_scan_failure_does_not_kill_the_loop(tmp_path: Path):
    w, _ = build(tmp_path)
    calls = {"n": 0}

    def exploding():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("stale NFS file handle")
        w.stop()
        return 0

    w.run_once = exploding
    w.run_forever()          # must return, not raise
    assert calls["n"] == 2
