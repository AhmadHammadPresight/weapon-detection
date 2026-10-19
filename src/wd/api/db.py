"""SQLite storage for uploads and the alerts they raised."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- image | video
    filename TEXT NOT NULL,          -- the name the operator uploaded
    stored TEXT NOT NULL,            -- file name under data/uploads
    camera TEXT NOT NULL,
    start_time TEXT NOT NULL,        -- camera-local wall-clock time, ISO 8601 without zone
    status TEXT NOT NULL,            -- queued | processing | done | failed
    progress REAL NOT NULL DEFAULT 0,
    frames INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    detector TEXT,
    threshold REAL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    upload_id TEXT NOT NULL REFERENCES uploads(id),
    camera TEXT NOT NULL,
    label TEXT NOT NULL,             -- gun | knife
    confidence REAL NOT NULL,
    at_time TEXT NOT NULL,           -- camera-local time of the boxed frame
    offset_seconds REAL NOT NULL,
    frames INTEGER NOT NULL,
    image TEXT NOT NULL,             -- boxed frame, file name under data/alerts
    boxes TEXT NOT NULL,             -- JSON list of {label, confidence, box}
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_by_upload ON alerts(upload_id);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        conn.executescript(SCHEMA)


def insert_upload(conn: sqlite3.Connection, **row) -> None:
    row = {"created": now(), "status": "queued", **row}
    cols = ", ".join(row)
    conn.execute(f"INSERT INTO uploads ({cols}) VALUES ({', '.join('?' for _ in row)})", tuple(row.values()))


def update_upload(conn: sqlite3.Connection, upload_id: str, **fields) -> None:
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE uploads SET {sets} WHERE id = ?", (*fields.values(), upload_id))


def get_upload(conn: sqlite3.Connection, upload_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    return dict(row) if row else None


def list_uploads(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT u.*, (SELECT COUNT(*) FROM alerts a WHERE a.upload_id = u.id) AS alerts FROM uploads u ORDER BY created DESC, rowid DESC"
    )
    return [dict(r) for r in rows]


def unfinished_uploads(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM uploads WHERE status IN ('queued', 'processing') ORDER BY created, rowid")
    return [dict(r) for r in rows]


def upsert_alert(conn: sqlite3.Connection, **row) -> None:
    row = {"created": now(), **row, "boxes": json.dumps(row["boxes"])}
    cols = ", ".join(row)
    updates = ", ".join(f"{k} = excluded.{k}" for k in row if k not in ("id", "created"))
    conn.execute(
        f"INSERT INTO alerts ({cols}) VALUES ({', '.join('?' for _ in row)}) ON CONFLICT(id) DO UPDATE SET {updates}",
        tuple(row.values()),
    )


def list_alerts(conn: sqlite3.Connection, upload_id: str | None = None) -> list[dict]:
    query = "SELECT a.*, u.filename, u.kind FROM alerts a JOIN uploads u ON u.id = a.upload_id"
    args: tuple = ()
    if upload_id:
        query += " WHERE a.upload_id = ?"
        args = (upload_id,)
    rows = conn.execute(query + " ORDER BY a.created DESC, a.at_time DESC", args)
    return [{**dict(r), "boxes": json.loads(r["boxes"])} for r in rows]


def clear(conn: sqlite3.Connection) -> list[str]:
    """Delete every upload and alert; returns the stored file names so the caller can remove them."""
    files = [r["stored"] for r in conn.execute("SELECT stored FROM uploads")]
    files += [r["image"] for r in conn.execute("SELECT image FROM alerts")]
    conn.execute("DELETE FROM alerts")
    conn.execute("DELETE FROM uploads")
    return files
