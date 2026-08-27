"""SQLite-backed scan history. One row per scan; the full JSON payload lives in
the ``data`` column, with a few promoted columns for filtering/sorting."""

import json
import sqlite3
from contextlib import contextmanager

from app.core.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    verdict       TEXT NOT NULL,
    confidence    REAL NOT NULL,
    tree_id       TEXT,
    block         TEXT,
    data          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at DESC);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def insert_scan(scan: dict):
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scans (id, created_at, verdict, confidence, tree_id, block, data)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                scan["id"],
                scan["createdAt"],
                scan["verdict"],
                scan["confidence"],
                scan.get("treeId"),
                scan.get("block"),
                json.dumps(scan),
            ),
        )


def list_scans(verdict: str | None = None, limit: int = 100) -> list[dict]:
    query = "SELECT data FROM scans"
    params: list = []
    if verdict in ("HEALTHY", "INFECTED"):
        query += " WHERE verdict = ?"
        params.append(verdict)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [json.loads(r["data"]) for r in rows]


def get_scan(scan_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT data FROM scans WHERE id = ?", (scan_id,)).fetchone()
    return json.loads(row["data"]) if row else None


def delete_scan(scan_id: str) -> dict | None:
    """Delete one scan by id. Returns its stored data (so the caller can also
    remove the associated uploaded image), or None if it didn't exist."""
    with _conn() as conn:
        row = conn.execute("SELECT data FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    return json.loads(row["data"])


def delete_scans(verdict: str | None = None, before: str | None = None) -> list[dict]:
    """Bulk delete scans matching optional filters (verdict, created before an
    ISO timestamp). With no filters, deletes every scan. Returns the deleted
    rows' data so the caller can clean up their uploaded images."""
    conditions = []
    params: list = []
    if verdict in ("HEALTHY", "INFECTED"):
        conditions.append("verdict = ?")
        params.append(verdict)
    if before:
        conditions.append("created_at < ?")
        params.append(before)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    with _conn() as conn:
        rows = conn.execute(f"SELECT data FROM scans{where}", params).fetchall()
        conn.execute(f"DELETE FROM scans{where}", params)
    return [json.loads(r["data"]) for r in rows]


def stats() -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT "
            "COUNT(*) AS total, "
            "SUM(CASE WHEN verdict='HEALTHY' THEN 1 ELSE 0 END) AS healthy, "
            "SUM(CASE WHEN verdict='INFECTED' THEN 1 ELSE 0 END) AS infected "
            "FROM scans"
        ).fetchone()
    return {
        "totalScan": row["total"] or 0,
        "totalHealthy": row["healthy"] or 0,
        "totalInfected": row["infected"] or 0,
    }
