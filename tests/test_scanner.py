"""Tests for finding work: the settle gate, ordering, and the resume guard."""

import os
from pathlib import Path

from subwright import layout, scanner


def _inputs(base: Path) -> set[Path]:
    """The folders holding work waiting to start rather than results.

    find_resumable now takes these explicitly instead of matching on folder
    NAME, because a rule's drop folder need not be called "ingest".
    """
    return {layout.ingest_dir(base), layout.reprocess_dir(base)}

NOW = 1_000_000.0


def make_video(folder: Path, name: str, *, age_seconds: float) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text("fake video")
    mtime = NOW - age_seconds
    os.utime(p, (mtime, mtime))
    return p


# --- settle gate ---

def test_file_younger_than_the_settle_gate_is_skipped(tmp_path: Path):
    make_video(layout.ingest_dir(tmp_path), "a.mkv", age_seconds=9)
    assert scanner.find_ingest(layout.ingest_dir(tmp_path), now=NOW) == []


def test_file_older_than_the_settle_gate_is_picked_up(tmp_path: Path):
    v = make_video(layout.ingest_dir(tmp_path), "a.mkv", age_seconds=11)
    assert scanner.find_ingest(layout.ingest_dir(tmp_path), now=NOW) == [v]


def test_file_exactly_at_the_settle_boundary_is_skipped(tmp_path: Path):
    # Pins > rather than >=, matching the original.
    make_video(layout.ingest_dir(tmp_path), "a.mkv", age_seconds=10)
    assert scanner.find_ingest(layout.ingest_dir(tmp_path), now=NOW) == []


# --- ordering and filtering ---

def test_oldest_file_is_returned_first(tmp_path: Path):
    ingest = layout.ingest_dir(tmp_path)
    newer = make_video(ingest, "newer.mkv", age_seconds=20)
    older = make_video(ingest, "older.mkv", age_seconds=99)
    assert scanner.find_ingest(layout.ingest_dir(tmp_path), now=NOW) == [older, newer]


def test_non_video_files_are_ignored(tmp_path: Path):
    ingest = layout.ingest_dir(tmp_path)
    make_video(ingest, "notes.txt", age_seconds=99)
    make_video(ingest, "subs.srt", age_seconds=99)
    assert scanner.find_ingest(layout.ingest_dir(tmp_path), now=NOW) == []


def test_uppercase_extensions_are_picked_up(tmp_path: Path):
    v = make_video(layout.ingest_dir(tmp_path), "a.MKV", age_seconds=99)
    assert scanner.find_ingest(layout.ingest_dir(tmp_path), now=NOW) == [v]


def test_missing_ingest_directory_is_not_an_error(tmp_path: Path):
    assert scanner.find_ingest(layout.ingest_dir(tmp_path), now=NOW) == []


# --- reprocess ---

def test_reprocess_video_is_returned_when_unmarked(tmp_path: Path):
    v = make_video(layout.reprocess_dir(tmp_path), "a.mkv", age_seconds=99)
    assert scanner.find_reprocess(layout.reprocess_dir(tmp_path), now=NOW) == [v]


def test_reprocess_video_with_marker_is_skipped(tmp_path: Path):
    folder = layout.reprocess_dir(tmp_path)
    v = make_video(folder, "a.mkv", age_seconds=99)
    layout.reprocessed_marker(folder, v).write_text("done")
    assert scanner.find_reprocess(layout.reprocess_dir(tmp_path), now=NOW) == []


# --- resume: the guard that protects the existing library ---

def test_resume_finds_a_folder_with_a_claim_and_no_translated_marker(tmp_path: Path):
    folder = tmp_path / "Foo"
    v = make_video(folder, "Foo.mkv", age_seconds=99)
    layout.claim_marker(folder).write_text("{}")
    assert scanner.find_resumable(tmp_path, exclude=_inputs(tmp_path)) == [v]


def test_preexisting_library_folder_without_claim_is_untouched(tmp_path: Path):
    """THE safety test.

    A folder with a video and no .translated - exactly what a hand-made folder,
    or anything predating the marker, looks like. Without the claim requirement
    the watcher would re-transcribe it and overwrite its subtitles. Plex and
    Stash read this tree.
    """
    folder = tmp_path / "SomethingIMadeMyself"
    make_video(folder, "video.mkv", age_seconds=99)
    (folder / "video.srt").write_text("my carefully edited subtitles")
    assert scanner.find_resumable(tmp_path, exclude=_inputs(tmp_path)) == []


def test_completed_folder_is_not_resumed(tmp_path: Path):
    folder = tmp_path / "Foo"
    make_video(folder, "Foo.mkv", age_seconds=99)
    layout.translated_marker(folder).write_text("done")
    assert scanner.find_resumable(tmp_path, exclude=_inputs(tmp_path)) == []


def test_claim_left_behind_after_success_is_not_resumed(tmp_path: Path):
    """Crash between writing .translated and deleting the claim - nothing to redo."""
    folder = tmp_path / "Foo"
    make_video(folder, "Foo.mkv", age_seconds=99)
    layout.claim_marker(folder).write_text("{}")
    layout.translated_marker(folder).write_text("done")
    assert scanner.find_resumable(tmp_path, exclude=_inputs(tmp_path)) == []


def test_resume_never_looks_inside_ingest_or_reprocess(tmp_path: Path):
    for name in (layout.INGEST_DIRNAME, layout.REPROCESS_DIRNAME):
        folder = tmp_path / name
        make_video(folder, "a.mkv", age_seconds=99)
        layout.claim_marker(folder).write_text("{}")
    assert scanner.find_resumable(tmp_path, exclude=_inputs(tmp_path)) == []
