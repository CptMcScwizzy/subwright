"""The contract with Plex and Stash.

Both media servers mount /mnt/data/translate and read the folders this layout
produces. These tests assert the exact literal names. If one fails, the change
would be visible to those media servers - confirm that is intended before
updating the expectation.
"""

from datetime import datetime
from pathlib import Path

from subwright import layout

BASE = Path("/mnt/data/translate")
NOW = datetime(2026, 8, 29, 14, 30, 5)


def test_ingest_and_reprocess_dirs_are_named_exactly():
    assert layout.ingest_dir(BASE) == BASE / "ingest"
    assert layout.reprocess_dir(BASE) == BASE / "reprocess"


def test_output_folder_is_named_after_the_video_stem():
    video = layout.ingest_dir(BASE) / "Foo.mkv"
    assert layout.output_dir(BASE, video, now=NOW) == BASE / "Foo"


def test_output_folder_collision_appends_timestamp():
    video = layout.ingest_dir(BASE) / "Foo.mkv"
    got = layout.output_dir(BASE, video, now=NOW, taken=True)
    assert got == BASE / "Foo_20260829_143005"


def test_srt_sits_beside_the_video_with_the_same_stem():
    assert layout.srt_for(BASE / "Foo" / "Foo.mkv") == BASE / "Foo" / "Foo.srt"


def test_srt_naming_survives_dots_in_the_filename():
    # with_suffix() would produce 'Show.S01.srt' here - losing an episode number
    # is the kind of bug that silently mismatches subtitles to the wrong file.
    video = BASE / "X" / "Show.S01E02.mkv"
    assert layout.srt_for(video) == BASE / "X" / "Show.S01E02.srt"


def test_temp_srt_is_in_the_same_directory_as_the_target():
    # os.replace is only atomic within one filesystem; this tree is NFS.
    video = BASE / "Foo" / "Foo.mkv"
    assert layout.tmp_srt_for(video).parent == layout.srt_for(video).parent


def test_success_marker_is_dot_translated():
    assert layout.translated_marker(BASE / "Foo") == BASE / "Foo" / ".translated"


def test_claim_marker_is_dot_processing():
    assert layout.claim_marker(BASE / "Foo") == BASE / "Foo" / ".processing"


def test_reprocessed_marker_includes_the_video_stem():
    video = layout.reprocess_dir(BASE) / "Foo.mkv"
    assert layout.reprocessed_marker(layout.reprocess_dir(BASE), video) == (
        BASE / "reprocess" / ".reprocessed_Foo"
    )


def test_backup_srt_is_timestamped_so_repeats_do_not_clobber():
    video = layout.reprocess_dir(BASE) / "Foo.mkv"
    assert layout.backup_srt_for(video, now=NOW).name == "Foo.srt.20260829_143005.bak"


def test_supported_video_extensions_are_exactly_this_set():
    assert frozenset(
        {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts"}
    ) == layout.VIDEO_EXTENSIONS


def test_extension_matching_is_case_insensitive():
    assert layout.is_video(Path("a.MKV"))
    assert layout.is_video(Path("a.Mp4"))
    assert not layout.is_video(Path("a.txt"))
    assert not layout.is_video(Path("a.srt"))
