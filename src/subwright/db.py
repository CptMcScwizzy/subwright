"""SQLite storage for settings and job history.

Deliberately small: two tables, no ORM, no migration framework. The schema is
created if absent and versioned by a single integer, which is enough for
something this size.

MUST live on local disk, never on the NFS mount. SQLite over NFS is a
well-known source of corruption, and the watch tree here is NFSv4.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT    NOT NULL,
    filename       TEXT    NOT NULL,
    source_path    TEXT    NOT NULL,
    output_path    TEXT,
    status         TEXT    NOT NULL,
    started_at     TEXT    NOT NULL,
    finished_at    TEXT,
    cue_count      INTEGER,
    media_duration REAL,
    error          TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_started ON jobs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs (status);
"""

# Columns added after v1. Applied by ALTER for existing databases and for new
# ones alike, so both go through the same path and the migration is exercised
# every single startup rather than only on someone else's machine.
_ADDED_COLUMNS = {
    "jobs": {
        "detected_language":   "TEXT",
        "language_probability": "REAL",
        "source":              "TEXT",
        "source_detail":       "TEXT",
    },
}


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The worker thread and the web thread both write, so serialise access.
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._add_missing_columns(conn)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _add_missing_columns(conn: sqlite3.Connection) -> None:
        """Bring an older database up to date.

        SQLite has no ADD COLUMN IF NOT EXISTS, so the existing columns are read
        first. Only ever adds nullable columns: that is the one schema change
        that cannot lose data, and it keeps this honest about being a migration
        helper rather than a migration framework.
        """
        for table, columns in _ADDED_COLUMNS.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            # WAL lets the UI read while the worker writes.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    # --- settings ---

    def load_settings(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        out: dict[str, Any] = {}
        for row in rows:
            try:
                out[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                out[row["key"]] = row["value"]
        return out

    def save_settings(self, values: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(k, json.dumps(v)) for k, v in values.items()],
            )

    def clear_settings(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM settings")

    # --- jobs ---

    def start_job(self, kind: str, video: Path, *, when: datetime | None = None) -> int:
        when = when or datetime.now()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs (kind, filename, source_path, status, started_at) "
                "VALUES (?, ?, ?, 'running', ?)",
                (kind, video.name, str(video), when.isoformat(timespec="seconds")),
            )
            return int(cur.lastrowid)

    def finish_job(
        self,
        job_id: int,
        *,
        output_path: Path | None = None,
        cue_count: int | None = None,
        media_duration: float | None = None,
        detected_language: str | None = None,
        language_probability: float | None = None,
        source: str = "transcribed",
        source_detail: str | None = None,
        when: datetime | None = None,
    ) -> None:
        when = when or datetime.now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='done', finished_at=?, output_path=?, "
                "cue_count=?, media_duration=?, detected_language=?, "
                "language_probability=?, source=?, source_detail=?, "
                "error=NULL WHERE id=?",
                (
                    when.isoformat(timespec="seconds"),
                    str(output_path) if output_path else None,
                    cue_count,
                    media_duration,
                    detected_language,
                    language_probability,
                    source,
                    source_detail,
                    job_id,
                ),
            )

    def fail_job(self, job_id: int, error: str, *, when: datetime | None = None) -> None:
        when = when or datetime.now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
                (when.isoformat(timespec="seconds"), error[:2000], job_id),
            )

    def mark_orphans_interrupted(self) -> int:
        """Anything still 'running' at startup was cut short by a restart.

        Without this, a killed job would sit in the history claiming to be
        running forever, and the dashboard would show a job that is not there.
        """
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status='interrupted', "
                "error='interrupted by a restart' WHERE status='running'"
            )
            return cur.rowcount

    def recent_jobs(self, limit: int = 50, status: str | None = None) -> list[dict]:
        sql = "SELECT * FROM jobs"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def job(self, job_id: int) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def delete_job(self, job_id: int) -> bool:
        """Remove one row. Returns whether there was one to remove.

        History only. Nothing on disk is touched: the video, its subtitles and
        its markers all stay exactly where they are.
        """
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cur.rowcount > 0

    def clear_jobs(self, *, keep_running: bool = True) -> int:
        """Empty the history. Returns how many rows went.

        A job still marked running is kept by default - deleting the row for
        work that is happening right now would leave the dashboard describing
        a job it can no longer find.
        """
        sql = "DELETE FROM jobs"
        if keep_running:
            sql += " WHERE status != 'running'"
        with self._lock, self._connect() as conn:
            return conn.execute(sql).rowcount

    def counts(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}
