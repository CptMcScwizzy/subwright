"""The Folders page.

Rows are read back as parallel lists, so the failure worth guarding against is
misalignment: one row's value landing on another row's rule. Several of these
exist purely to catch that.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from subwright import config
from subwright.db import Database
from subwright.web.app import create_app


@pytest.fixture
def env(tmp_path: Path):
    db = Database(tmp_path / "folders.db")
    settings, _ = config.resolve([], env={})
    settings.watch_dir = tmp_path / "translate"
    saved: dict = {"settings": None}
    app = create_app(db, settings,
                     on_settings_saved=lambda s: saved.__setitem__("settings", s),
                     version="9.9.9")
    return TestClient(app), db, settings, saved


def form(*rows: dict) -> dict:
    """Build the posted form for a table of folders.

    Most fields repeat once per row and are read positionally. Language is
    indexed by row instead, because its dropdown is disabled when auto-detect is
    chosen and a disabled control submits nothing.
    """
    data: dict = {"name": [], "ingest": [], "output": [], "reprocess": [], "enabled": []}
    for i, row in enumerate(rows):
        data["name"].append(row.get("name", ""))
        data["ingest"].append(str(row.get("ingest", "")))
        data["output"].append(str(row.get("output", "")))
        data["reprocess"].append(str(row.get("reprocess", "")))
        data["enabled"].append("1" if row.get("enabled", True) else "0")
        language = row.get("language", "")
        data[f"language_mode{i}"] = "fixed" if language else "auto"
        if language:
            data[f"language{i}"] = language
    return data


def row(name: str, base: Path, **kw) -> dict:
    return {"name": name, "ingest": base / "in", "output": base / "out", **kw}


def test_the_page_shows_the_existing_layout_before_anything_is_configured(env):
    client, _, _, _ = env
    body = client.get("/folders").text
    assert "ingest" in body
    assert "reprocess" in body


def test_saving_two_folders_stores_both(env, tmp_path):
    client, db, _, _ = env
    r = client.post("/folders", data=form(
        row("anime", tmp_path / "a", language="ja"),
        row("kdrama", tmp_path / "b", language="ko"),
    ), follow_redirects=False)
    assert r.status_code == 303

    stored = db.load_settings()["rules"]
    assert [s["name"] for s in stored] == ["anime", "kdrama"]
    assert [s["language"] for s in stored] == ["ja", "ko"]


def test_each_row_keeps_its_own_values(env, tmp_path):
    """The misalignment guard: three rows, all different, all distinguishable."""
    client, db, _, _ = env
    client.post("/folders", data=form(
        row("one", tmp_path / "1", language="ja"),
        row("two", tmp_path / "2"),
        row("three", tmp_path / "3", language="ko"),
    ))

    stored = db.load_settings()["rules"]
    assert [s["name"] for s in stored] == ["one", "two", "three"]
    # Row two auto-detects while its neighbours are pinned. Reading the indexed
    # language fields positionally would come back wrong here.
    assert [s["language"] for s in stored] == ["ja", "", "ko"]
    assert stored[1]["ingest"] == str(tmp_path / "2" / "in")


def test_turning_one_folder_off_does_not_move_the_others(env, tmp_path):
    """An On/Off select rather than a checkbox, precisely so this holds."""
    client, db, _, _ = env
    client.post("/folders", data=form(
        row("one", tmp_path / "1"),
        row("two", tmp_path / "2", enabled=False),
        row("three", tmp_path / "3"),
    ))

    stored = db.load_settings()["rules"]
    assert [s["enabled"] for s in stored] == [True, False, True]
    assert [s["name"] for s in stored] == ["one", "two", "three"]


def test_a_folder_watching_the_same_directory_as_another_is_refused(env, tmp_path):
    client, db, _, _ = env
    shared = tmp_path / "shared"
    r = client.post("/folders", data=form(
        {"name": "one", "ingest": shared, "output": tmp_path / "1"},
        {"name": "two", "ingest": shared, "output": tmp_path / "2"},
    ))

    assert r.status_code == 400
    assert "both watch" in r.text
    assert "rules" not in db.load_settings(), "an invalid layout must not be stored"


def test_a_rejected_save_keeps_what_was_typed(env, tmp_path):
    """Re-rendered rather than redirected, so nobody has to retype six paths."""
    client, _, _, _ = env
    shared = tmp_path / "shared"
    r = client.post("/folders", data=form(
        {"name": "memorable-name", "ingest": shared, "output": tmp_path / "1"},
        {"name": "two", "ingest": shared, "output": tmp_path / "2"},
    ))
    assert "memorable-name" in r.text


def test_adding_a_folder_does_not_save_a_blank_one(env, tmp_path):
    client, db, _, _ = env
    r = client.post("/folders/add", data=form(row("one", tmp_path / "1")))
    assert r.status_code == 200
    assert "one" in r.text
    # Nothing persisted: a blank row could never pass validation.
    assert "rules" not in db.load_settings()


def test_removing_a_folder_leaves_the_others_in_order(env, tmp_path):
    client, _, _, _ = env
    r = client.post("/folders/1/delete", data=form(
        row("keep-me", tmp_path / "1"),
        row("delete-me", tmp_path / "2"),
        row("keep-me-too", tmp_path / "3"),
    ))

    # Checked on the form rows, not the whole page: the removed name also
    # appears in the "Removed delete-me" confirmation, which is intended.
    assert 'value="delete-me"' not in r.text
    assert 'value="keep-me"' in r.text
    assert 'value="keep-me-too"' in r.text


def test_the_last_folder_cannot_be_removed(env, tmp_path):
    client, _, _, _ = env
    r = client.post("/folders/0/delete", data=form(row("only-one", tmp_path / "1")))
    assert "at least one" in r.text
    assert "only-one" in r.text


def test_saved_folders_reach_the_running_worker(env, tmp_path):
    client, _, _, saved = env
    client.post("/folders", data=form(row("live", tmp_path, language="ja")))
    assert saved["settings"] is not None
    assert [r.name for r in saved["settings"].effective_rules] == ["live"]


def test_stored_folders_are_loaded_on_the_next_start(env, tmp_path):
    """A restart has to come back with the same layout, or a container update
    would silently revert to watching a single folder."""
    client, db, _, _ = env
    client.post("/folders", data=form(
        row("anime", tmp_path / "a", language="ja"),
        row("kdrama", tmp_path / "b", language="ko"),
    ))

    reloaded, _ = config.resolve([], env={}, stored=db.load_settings())
    assert [r.name for r in reloaded.effective_rules] == ["anime", "kdrama"]
    assert [r.language for r in reloaded.effective_rules] == ["ja", "ko"]
