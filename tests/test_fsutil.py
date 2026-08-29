"""Tests for filesystem operations - especially the atomic write."""

from pathlib import Path

import pytest

from subwright import fsutil


def test_atomic_write_creates_the_file_with_exact_content(tmp_path: Path):
    target = tmp_path / "out.srt"
    fsutil.atomic_write_text(target, "hello\n", tmp=tmp_path / "out.srt.tmp")
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_atomic_write_leaves_no_temp_file_behind(tmp_path: Path):
    target = tmp_path / "out.srt"
    tmp = tmp_path / "out.srt.tmp"
    fsutil.atomic_write_text(target, "hello", tmp=tmp)
    assert not tmp.exists()


def test_atomic_write_replaces_existing_content(tmp_path: Path):
    target = tmp_path / "out.srt"
    target.write_text("old", encoding="utf-8")
    fsutil.atomic_write_text(target, "new", tmp=tmp_path / "out.srt.tmp")
    assert target.read_text(encoding="utf-8") == "new"


def test_failed_write_leaves_the_original_untouched(tmp_path: Path):
    """The core guarantee: a crash mid-write never corrupts a good file."""
    target = tmp_path / "out.srt"
    target.write_text("original", encoding="utf-8")

    # A directory where the scratch file should go makes the write fail.
    bad_tmp = tmp_path / "adir"
    bad_tmp.mkdir()

    with pytest.raises(OSError):
        fsutil.atomic_write_text(target, "new", tmp=bad_tmp)

    assert target.read_text(encoding="utf-8") == "original"


def test_failed_write_leaves_no_partial_target(tmp_path: Path):
    """If there was no file before, a failure must not create a truncated one."""
    target = tmp_path / "out.srt"
    bad_tmp = tmp_path / "adir"
    bad_tmp.mkdir()
    with pytest.raises(OSError):
        fsutil.atomic_write_text(target, "new", tmp=bad_tmp)
    assert not target.exists()


def test_move_creates_the_destination_directory(tmp_path: Path):
    src = tmp_path / "a.mkv"
    src.write_text("x")
    dst = tmp_path / "Foo" / "a.mkv"
    fsutil.move(src, dst)
    assert dst.exists() and not src.exists()


def test_set_owner_never_raises_when_not_permitted(tmp_path: Path):
    """A chown failure must not abort a finished transcription."""
    f = tmp_path / "a.txt"
    f.write_text("x")
    fsutil.set_owner(f, 1000, 1000)  # no assertion: must simply not raise


def test_prune_backups_keeps_only_the_newest_n(tmp_path: Path):
    import os, time
    for i in range(5):
        p = tmp_path / f"Foo.srt.2026010{i}_000000.bak"
        p.write_text("x")
        os.utime(p, (time.time() + i, time.time() + i))
    removed = fsutil.prune_backups(tmp_path, "Foo.srt.*.bak", keep=2)
    remaining = sorted(p.name for p in tmp_path.glob("Foo.srt.*.bak"))
    assert len(remaining) == 2
    assert len(removed) == 3


def test_prune_backups_does_nothing_when_under_the_limit(tmp_path: Path):
    (tmp_path / "Foo.srt.20260101_000000.bak").write_text("x")
    assert fsutil.prune_backups(tmp_path, "Foo.srt.*.bak", keep=3) == []


def test_prune_backups_ignores_unrelated_files(tmp_path: Path):
    (tmp_path / "Foo.mkv").write_text("x")
    (tmp_path / "Foo.srt").write_text("x")
    fsutil.prune_backups(tmp_path, "Foo.srt.*.bak", keep=0)
    assert (tmp_path / "Foo.mkv").exists()
    assert (tmp_path / "Foo.srt").exists()
