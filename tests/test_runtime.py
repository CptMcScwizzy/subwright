"""Tests for the wiring between the worker, the database and the web app."""

from pathlib import Path

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


