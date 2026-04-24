"""SQLite store. WAL mode. Small, local, boring."""

import json
import os
import random
import sqlite3
from pathlib import Path
from typing import Optional

PRIVATE_ROOT = Path.home() / ".reinforceclaw"
DB_PATH = PRIVATE_ROOT / "reinforceclaw.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    context TEXT,
    rollout_context TEXT,
    event_id TEXT,
    rating INTEGER NOT NULL,       -- +1 good, -1 bad.
    source TEXT DEFAULT 'cli',     -- cli | hook | discord | telegram | whatsapp
    trained INTEGER DEFAULT 0,
    adapter_version INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS adapters (
    version INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    parent_version INTEGER,
    status TEXT DEFAULT 'active',
    metrics TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS ema_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    reward_mean REAL DEFAULT 0.0,
    count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS training_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT
);
CREATE TABLE IF NOT EXISTS background_history (
    hour INTEGER PRIMARY KEY,
    pressure_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS migrations (
    name TEXT PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now'))
);
"""

_TABLES = {"feedback", "adapters", "ema_state", "training_state", "background_history", "migrations"}
_ALTER_TABLES = {"feedback", "adapters", "ema_state"}


def secure_private_dir(path: Path = PRIVATE_ROOT) -> Path:
    old = os.umask(0o077)
    try:
        path.mkdir(parents=True, exist_ok=True)
    finally:
        os.umask(old)
    try:
        path.chmod(0o700)
    except OSError as exc:
        raise PermissionError(f"could not secure private dir: {path}") from exc
    return path


def secure_private_file(path: Path) -> Path:
    try:
        if path.exists():
            path.chmod(0o600)
    except OSError as exc:
        raise PermissionError(f"could not secure private file: {path}") from exc
    return path


def _secure_sqlite_files(path: Path) -> None:
    for item in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        secure_private_file(item)


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path).expanduser() if db_path else DB_PATH
    secure_private_dir(path.parent)
    old = os.umask(0o077)
    try:
        conn = sqlite3.connect(str(path))
    finally:
        os.umask(old)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA secure_delete=ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.execute("INSERT OR IGNORE INTO ema_state (id, reward_mean, count) VALUES (1, 0.0, 0)")
    conn.execute("INSERT OR IGNORE INTO training_state (id, state) VALUES (1, NULL)")
    conn.commit()
    _secure_sqlite_files(path)
    return conn


def init(db_path: Optional[Path] = None) -> None:
    connect(db_path).close()


def _columns(conn, table):
    if table not in _TABLES:
        raise ValueError(f"invalid table: {table}")
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _add_missing_columns(conn, table, columns):
    if table not in _ALTER_TABLES:
        raise ValueError(f"invalid migration table: {table}")
    existing = _columns(conn, table)
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _migrate(conn):
    """Bring old local databases forward without dropping user ratings."""
    _add_missing_columns(conn, "feedback", {
        "context": "TEXT",
        "rollout_context": "TEXT",
        "event_id": "TEXT",
        "source": "TEXT DEFAULT 'cli'",
        "trained": "INTEGER DEFAULT 0",
        "adapter_version": "INTEGER",
    })
    _add_missing_columns(conn, "adapters", {
        "parent_version": "INTEGER",
        "status": "TEXT DEFAULT 'active'",
        "metrics": "TEXT",
        "created_at": "TEXT",
    })
    _add_missing_columns(conn, "ema_state", {
        "reward_mean": "REAL DEFAULT 0.0",
        "count": "INTEGER DEFAULT 0",
    })
    if not conn.execute("SELECT 1 FROM migrations WHERE name='feedback_v2'").fetchone():
        # Legacy unrated rows are intentionally discarded; ignored responses should not train.
        conn.execute("DELETE FROM feedback WHERE rating=0")
        conn.execute("DROP INDEX IF EXISTS idx_feedback_unique_rating")
        conn.execute("INSERT OR IGNORE INTO migrations(name) VALUES('feedback_v2')")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_event ON feedback(source, event_id) WHERE event_id IS NOT NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_untrained ON feedback(trained, rating, source, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_recent ON feedback(created_at)")
    conn.commit()


# -- feedback ops --

def _validate_rating(rating, allow_delete=False):
    allowed = (-1, 0, 1) if allow_delete else (-1, 1)
    if rating not in allowed:
        raise ValueError("rating must be -1, 0, or 1" if allow_delete else "rating must be -1 or 1")


def add_feedback(conn, model, prompt, response, rating, context=None, source="cli", event_id=None, rollout_context=None):
    _validate_rating(rating)
    with conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO feedback (model, prompt, response, context, rollout_context, event_id, rating, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (model, prompt, response, context, rollout_context, event_id, rating, source),
        )
    if cur.rowcount == 0:
        row = conn.execute("SELECT id FROM feedback WHERE source=? AND event_id=? LIMIT 1", (source, event_id)).fetchone() if event_id else None
        return row["id"] if row else None
    return cur.lastrowid


def get_feedback_by_ids(conn, ids):
    if not ids:
        return []
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT * FROM feedback WHERE id IN ({ph})", ids).fetchall()
    by_id = {row["id"]: dict(row) for row in rows}
    return [by_id[i] for i in ids if i in by_id]


def _feedback_query(base_where: str, order: str, limit: int, source):
    sql = f"SELECT * FROM feedback WHERE {base_where}"
    tail, params = _source_clause(source)
    sql += tail
    sql += f" ORDER BY {order}"
    if limit > 0:
        sql += " LIMIT ?"
        params = (*params, limit)
    return sql, params


def _source_clause(source):
    return (" AND source=?", (source,)) if source is not None else ("", ())


def get_untrained(conn, limit=0, source=None):
    sql, params = _feedback_query("trained=0 AND rating!=0", "created_at ASC", limit, source)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_replay(conn, limit=0, source=None):
    if limit > 0:
        sql = "SELECT COALESCE(MAX(id), 0) AS max_id FROM feedback WHERE trained=1 AND rating!=0"
        params = []
        if source is not None:
            sql += " AND source=?"
            params.append(source)
        max_id = conn.execute(sql, tuple(params)).fetchone()["max_id"]
        if max_id:
            start = random.randint(1, max_id)
            where = "trained=1 AND rating!=0 AND id>=?"
            params = [start]
            if source is not None:
                where += " AND source=?"
                params.append(source)
            rows = conn.execute(f"SELECT * FROM feedback WHERE {where} ORDER BY id LIMIT ?", params + [limit]).fetchall()
            if len(rows) >= limit:
                return [dict(r) for r in rows]
            where, params = "trained=1 AND rating!=0 AND id<?", [start]
            if source is not None:
                where += " AND source=?"
                params.append(source)
            more = conn.execute(f"SELECT * FROM feedback WHERE {where} ORDER BY id LIMIT ?", params + [limit - len(rows)]).fetchall()
            return [dict(r) for r in rows] + [dict(r) for r in more]
    sql, params = _feedback_query("trained=1 AND rating!=0", "id ASC", limit, source)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def mark_trained(conn, ids, adapter_version):
    if not ids:
        return
    ph = ",".join("?" for _ in ids)
    with conn:
        conn.execute(
            f"UPDATE feedback SET trained=1, adapter_version=? WHERE id IN ({ph})",
            [adapter_version] + ids,
        )


def revise_feedback_rating(conn, feedback_id, rating):
    _validate_rating(rating, allow_delete=True)
    if rating == 0:
        with conn:
            cur = conn.execute("DELETE FROM feedback WHERE id=?", (feedback_id,))
        return cur.rowcount
    with conn:
        cur = conn.execute(
            "UPDATE feedback SET rating=?, trained=0, adapter_version=NULL WHERE id=?",
            (rating, feedback_id),
        )
    return cur.rowcount


def latest_rated(conn, source=None, context=None):
    sql = "SELECT * FROM feedback WHERE rating!=0"
    params = []
    if source is not None:
        sql += " AND source=?"
        params.append(source)
    if context is not None:
        sql += " AND context=?"
        params.append(context)
    row = conn.execute(f"{sql} ORDER BY id DESC LIMIT 1", params).fetchone()
    return dict(row) if row else None


def remove_last(conn, source=None, context=None):
    sql = "SELECT * FROM feedback WHERE rating!=0"
    params = []
    if source is not None:
        sql += " AND source=?"
        params.append(source)
    if context is not None:
        sql += " AND context=?"
        params.append(context)
    row = conn.execute(f"{sql} ORDER BY id DESC LIMIT 1", params).fetchone()
    if not row:
        return None
    conn.execute("DELETE FROM feedback WHERE id = ?", (row["id"],))
    conn.commit()
    return dict(row)


def recent(conn, limit=20):
    """Last N ratings for history view."""
    rows = conn.execute(
        "SELECT id, prompt, rating, source, created_at FROM feedback "
        "WHERE rating!=0 ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def count(conn):
    row = conn.execute(
        "SELECT COUNT(*) as total, "
        "COALESCE(SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END),0) as good, "
        "COALESCE(SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END),0) as bad, "
        "COALESCE(SUM(CASE WHEN trained=0 THEN 1 ELSE 0 END),0) as untrained "
        "FROM feedback WHERE rating!=0"
    ).fetchone()
    return dict(row)


def count_trainable_untrained(conn, source=None):
    tail, params = _source_clause(source)
    row = conn.execute(f"SELECT COUNT(*) AS total FROM feedback WHERE trained=0 AND rating!=0{tail}", params).fetchone()
    return row["total"]


# -- ema ops --

def get_ema(conn):
    r = conn.execute("SELECT reward_mean, count FROM ema_state WHERE id=1").fetchone()
    return r["reward_mean"], r["count"]


def get_training_state(conn):
    row = conn.execute("SELECT state FROM training_state WHERE id=1").fetchone()
    if not row or not row["state"]:
        return None
    return json.loads(row["state"])


def save_training_state(conn, state):
    conn.execute("UPDATE training_state SET state=? WHERE id=1", (json.dumps(state),))
    conn.commit()


def clear_training_state(conn):
    conn.execute("UPDATE training_state SET state=NULL WHERE id=1")
    conn.commit()


def record_background_event(conn, kind, hour):
    field = "pressure_count" if kind == "pressure" else "success_count"
    with conn:
        conn.execute(
            f"INSERT INTO background_history (hour, pressure_count, success_count) VALUES (?, ?, ?) "
            f"ON CONFLICT(hour) DO UPDATE SET {field}={field}+1",
            (hour, 1 if field == "pressure_count" else 0, 1 if field == "success_count" else 0),
        )


def background_history(conn):
    rows = conn.execute("SELECT hour, pressure_count, success_count FROM background_history").fetchall()
    return {row["hour"]: dict(row) for row in rows}


def update_ema(conn, mean, n):
    conn.execute("UPDATE ema_state SET reward_mean=?, count=? WHERE id=1", (mean, n))
    conn.commit()


def record_training_round(conn, mean, count, version, path, parent=None, metrics=None, feedback_ids=None, clear_state=False):
    payload = dict(metrics or {})
    payload.setdefault("ema_mean", mean)
    payload.setdefault("ema_count", count)
    if feedback_ids is not None:
        payload.setdefault("feedback_ids", list(feedback_ids))
    with conn:
        conn.execute(
            "INSERT INTO adapters (version, path, parent_version, status, metrics) VALUES (?,?,?,?,?)",
            (version, path, parent, "candidate", json.dumps(payload) if payload else None),
        )
        if clear_state:
            conn.execute("UPDATE training_state SET state=NULL WHERE id=1")


# -- adapter ops --

def list_adapters(conn):
    rows = conn.execute(
        "SELECT version, status, created_at FROM adapters ORDER BY version DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def add_adapter(conn, version, path, parent=None, metrics=None):
    with conn:
        conn.execute("UPDATE adapters SET status='inactive' WHERE status='active'")
        conn.execute(
            "INSERT INTO adapters (version, path, parent_version, metrics) VALUES (?,?,?,?)",
            (version, path, parent, json.dumps(metrics) if metrics else None),
        )


def latest_adapter(conn):
    r = conn.execute(
        "SELECT * FROM adapters WHERE status='active' ORDER BY version DESC LIMIT 1"
    ).fetchone()
    return dict(r) if r else None


def latest_candidate(conn):
    r = conn.execute(
        "SELECT * FROM adapters WHERE status='candidate' ORDER BY version DESC LIMIT 1"
    ).fetchone()
    return dict(r) if r else None


def activate_adapter(conn, version):
    with conn:
        conn.execute("UPDATE adapters SET status='inactive' WHERE status='active'")
        conn.execute("UPDATE adapters SET status='active' WHERE version=?", (version,))
    return latest_adapter(conn)


def activate_training_round(conn, version, mean=None, count=None, feedback_ids=None):
    row = conn.execute("SELECT metrics FROM adapters WHERE version=?", (version,)).fetchone()
    metrics = json.loads(row["metrics"]) if row and row["metrics"] else {}
    if mean is None:
        mean = metrics.get("ema_mean")
    if count is None:
        count = metrics.get("ema_count")
    feedback_ids = list(feedback_ids or metrics.get("feedback_ids") or [])
    with conn:
        if mean is not None and count is not None:
            conn.execute("UPDATE ema_state SET reward_mean=?, count=? WHERE id=1", (mean, count))
        conn.execute("UPDATE adapters SET status='inactive' WHERE status='active'")
        conn.execute("UPDATE adapters SET status='active' WHERE version=?", (version,))
        if feedback_ids:
            ph = ",".join("?" for _ in feedback_ids)
            conn.execute(
                f"UPDATE feedback SET trained=1, adapter_version=? WHERE id IN ({ph})",
                [version] + feedback_ids,
            )


def reject_adapter(conn, version):
    conn.execute("UPDATE adapters SET status='rejected' WHERE version=?", (version,))
    conn.commit()


def rollback_to(conn, version):
    row = conn.execute("SELECT version FROM adapters WHERE version=?", (version,)).fetchone()
    if not row:
        return None
    with conn:
        conn.execute("UPDATE adapters SET status='inactive' WHERE status='active'")
        conn.execute("UPDATE adapters SET status='rolled_back' WHERE version > ?", (version,))
        conn.execute("UPDATE adapters SET status='active' WHERE version = ?", (version,))
    return latest_adapter(conn)


def rollback(conn):
    latest = latest_adapter(conn)
    if not latest:
        return None
    previous = conn.execute(
        "SELECT version FROM adapters WHERE status='inactive' ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if not previous:
        return None
    with conn:
        conn.execute("UPDATE adapters SET status='rolled_back' WHERE version=?", (latest["version"],))
        conn.execute("UPDATE adapters SET status='active' WHERE version = ?", (previous["version"],))
    return latest_adapter(conn)


def cleanup_adapters(conn, keep=20):
    rows = conn.execute(
        "SELECT version, path FROM adapters "
        "WHERE status IN ('inactive', 'rejected') "
        "ORDER BY version DESC LIMIT -1 OFFSET ?",
        (int(keep),),
    ).fetchall()
    paths = [r["path"] for r in rows]
    if paths:
        ph = ",".join("?" for _ in rows)
        with conn:
            conn.execute(f"DELETE FROM adapters WHERE version IN ({ph})", [r["version"] for r in rows])
    return paths


# -- nuclear option --

def reset_all(conn):
    conn.executescript(
        "DELETE FROM feedback; DELETE FROM adapters; "
        "DELETE FROM background_history; "
        "UPDATE ema_state SET reward_mean=0.0, count=0 WHERE id=1;"
        "UPDATE training_state SET state=NULL WHERE id=1;"
    )
    conn.commit()
