"""Watch rules: several folders, each with its own output and language.

The first group matters most. An installation that predates rules must behave
exactly as it did, because the alternative is a silent change to a folder layout
that Plex and Stash are already reading.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from subwright import config, layout, rules
from subwright.rules import RuleError, WatchRule
from subwright.worker import Worker

from .fakes import FakeTranscriber

NOW = datetime(2026, 1, 2, 3, 4, 5)


def _worker(rule_list, transcriber=None, **kw):
    return Worker(
        rule_list,
        transcriber or FakeTranscriber(),
        clock=lambda: NOW,
        monotonic=lambda: 9e9,
        sleep=lambda _: None,
        **kw,
    )


def _drop(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / name
    video.write_bytes(b"data")
    return video


# --- nothing changes for an existing installation ---

def test_an_installation_with_no_rules_gets_the_original_layout(tmp_path):
    settings = config.Settings(watch_dir=tmp_path, language="ja")
    effective = settings.effective_rules
    assert len(effective) == 1
    assert effective[0].ingest == tmp_path / "ingest"
    assert effective[0].reprocess == tmp_path / "reprocess"
    assert effective[0].output == tmp_path
    assert effective[0].language == "ja"


def test_the_default_rule_puts_output_exactly_where_it_always_went(tmp_path):
    rule = rules.default_for(tmp_path)
    video = _drop(rule.ingest, "Foo.mkv")
    _worker([rule]).run_once()
    assert (tmp_path / "Foo" / "Foo.mkv").is_file()
    assert (tmp_path / "Foo" / "Foo.en.srt").is_file()
    assert (tmp_path / "Foo" / ".translated").is_file()
    assert not video.exists()


# --- several folders ---

def test_each_folder_writes_into_its_own_output(tmp_path):
    a = WatchRule("anime", tmp_path / "a/in", tmp_path / "a/out")
    b = WatchRule("kdrama", tmp_path / "b/in", tmp_path / "b/out")
    _drop(a.ingest, "One.mkv")
    _drop(b.ingest, "Two.mkv")

    assert _worker([a, b]).run_once() == 2

    assert (tmp_path / "a/out/One/One.en.srt").is_file()
    assert (tmp_path / "b/out/Two/Two.en.srt").is_file()
    # Neither leaked into the other.
    assert not (tmp_path / "a/out/Two").exists()
    assert not (tmp_path / "b/out/One").exists()


def test_each_folder_uses_its_own_language(tmp_path):
    """The reason per-folder language exists: a folder you already know is
    Korean should not be guessed at just because another folder is Japanese."""
    a = WatchRule("jp", tmp_path / "a/in", tmp_path / "a/out", language="ja")
    b = WatchRule("kr", tmp_path / "b/in", tmp_path / "b/out", language="ko")
    _drop(a.ingest, "One.mkv")
    _drop(b.ingest, "Two.mkv")

    transcriber = FakeTranscriber()
    _worker([a, b], transcriber).run_once()

    asked = {path.name: language for path, language in transcriber.calls}
    assert asked == {"One.mkv": "ja", "Two.mkv": "ko"}


def test_a_folder_can_auto_detect_while_another_is_pinned(tmp_path):
    a = WatchRule("pinned", tmp_path / "a/in", tmp_path / "a/out", language="ja")
    b = WatchRule("auto", tmp_path / "b/in", tmp_path / "b/out", language="")
    _drop(a.ingest, "One.mkv")
    _drop(b.ingest, "Two.mkv")

    transcriber = FakeTranscriber()
    _worker([a, b], transcriber).run_once()

    asked = {path.name: language for path, language in transcriber.calls}
    assert asked == {"One.mkv": "ja", "Two.mkv": None}


def test_a_disabled_folder_is_not_watched(tmp_path):
    on = WatchRule("on", tmp_path / "a/in", tmp_path / "a/out")
    off = WatchRule("off", tmp_path / "b/in", tmp_path / "b/out", enabled=False)
    _drop(on.ingest, "One.mkv")
    ignored = _drop(off.ingest, "Two.mkv")

    assert _worker([on, off]).run_once() == 1
    assert ignored.exists(), "a disabled folder must be left completely alone"
    assert not (tmp_path / "b/out/Two").exists()


def test_a_folder_without_a_reprocess_directory_is_allowed(tmp_path):
    rule = WatchRule("simple", tmp_path / "in", tmp_path / "out", reprocess=None)
    _drop(rule.ingest, "One.mkv")
    assert _worker([rule]).run_once() == 1
    assert not (tmp_path / "out/reprocess").exists()


def test_resume_only_looks_inside_the_folder_that_owns_it(tmp_path):
    """An interrupted job in one folder must not be adopted by another rule."""
    a = WatchRule("a", tmp_path / "a/in", tmp_path / "a/out")
    b = WatchRule("b", tmp_path / "b/in", tmp_path / "b/out")
    stalled = tmp_path / "a/out" / "Halfway"
    stalled.mkdir(parents=True)
    (stalled / "Halfway.mkv").write_bytes(b"data")
    layout.claim_marker(stalled).write_text("{}")

    transcriber = FakeTranscriber()
    _worker([b], transcriber).run_once()
    assert transcriber.calls == [], "rule b adopted an interrupted job owned by rule a"

    _worker([a], transcriber).run_once()
    assert [p.name for p, _ in transcriber.calls] == ["Halfway.mkv"]


def test_a_drop_folder_that_is_not_called_ingest_still_works(tmp_path):
    """find_resumable used to skip folders by NAME. With configurable paths a
    drop folder can be called anything, and something unrelated might be called
    ingest."""
    rule = WatchRule("odd", tmp_path / "out/dropbox", tmp_path / "out")
    _drop(rule.ingest, "One.mkv")
    assert _worker([rule]).run_once() == 1
    assert (tmp_path / "out/One/One.en.srt").is_file()
    # The drop folder itself was never treated as an interrupted job.
    assert not (tmp_path / "out/dropbox/.translated").exists()


# --- rejected configurations ---

def test_two_folders_watching_the_same_directory_are_rejected(tmp_path):
    """Both would claim the same file, and the loser would fail on a video that
    had already been moved away."""
    shared = tmp_path / "in"
    with pytest.raises(RuleError, match="both watch"):
        rules.validate_all([
            WatchRule("a", shared, tmp_path / "a"),
            WatchRule("b", shared, tmp_path / "b"),
        ])


def test_two_folders_with_the_same_name_are_rejected(tmp_path):
    with pytest.raises(RuleError, match="both called"):
        rules.validate_all([
            WatchRule("media", tmp_path / "a/in", tmp_path / "a/out"),
            WatchRule("Media", tmp_path / "b/in", tmp_path / "b/out"),
        ])


def test_a_drop_folder_inside_another_folders_output_is_rejected(tmp_path):
    """Its dropped files would be seen as interrupted jobs by the other rule."""
    with pytest.raises(RuleError, match="confuse resume"):
        rules.validate_all([
            WatchRule("outer", tmp_path / "outer/in", tmp_path / "outer"),
            WatchRule("inner", tmp_path / "outer/dropbox", tmp_path / "elsewhere"),
        ])


def test_a_folder_that_is_its_own_output_is_rejected(tmp_path):
    with pytest.raises(RuleError, match="must be different"):
        WatchRule("same", tmp_path / "x", tmp_path / "x").validate()


def test_a_bad_language_on_one_folder_is_reported_with_that_folders_name(tmp_path):
    with pytest.raises(RuleError, match="kdrama"):
        rules.validate_all([
            WatchRule("kdrama", tmp_path / "in", tmp_path / "out", language="korean"),
        ])


def test_an_empty_rule_set_is_rejected():
    with pytest.raises(RuleError, match="at least one"):
        rules.validate_all([])


# --- storage round trip ---

def test_a_folder_survives_being_saved_and_loaded(tmp_path):
    original = WatchRule("anime", tmp_path / "in", tmp_path / "out",
                         reprocess=tmp_path / "again", language="ja", enabled=False)
    assert WatchRule.from_dict(original.to_dict()) == original


def test_a_folder_saved_without_a_reprocess_directory_loads_as_none(tmp_path):
    original = WatchRule("simple", tmp_path / "in", tmp_path / "out", reprocess=None)
    assert WatchRule.from_dict(original.to_dict()).reprocess is None
