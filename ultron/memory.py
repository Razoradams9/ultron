"""Persistent memory for Ultron: conversation log + long-term facts.

Two tables:
  messages  — full chat transcript (user/assistant/tool events)
  facts     — durable statements about the world/user, FTS-searchable
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

from .config import DB_PATH, HISTORY_LIMIT

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        role TEXT NOT NULL,
        content TEXT,
        meta TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        source TEXT NOT NULL,
        text TEXT NOT NULL
    );
"""


def _connect(path) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite DB at `path` and apply the schema.

    Ensures the parent directory exists first \u2014 a bare connect() fails with
    'unable to open database file' when the folder is missing or the path is
    a dangling symlink (e.g. a Google Drive mount that isn't actually
    mounted)."""
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    # A broken symlink resolves to a missing target; drop it so we can recreate.
    if p.is_symlink() and not p.exists():
        p.unlink(missing_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with _lock:
        conn.executescript(_SCHEMA)
        conn.commit()
    return conn


def init() -> None:
    global _conn
    try:
        _conn = _connect(DB_PATH)
    except (sqlite3.OperationalError, OSError) as e:
        # The configured DB path is unusable (missing dir, dead Drive symlink,
        # read-only mount, ...). Don't take the whole server down for it \u2014
        # fall back to a writable temp DB so the app still boots. Memory just
        # won't persist across restarts.
        fallback = Path(tempfile.gettempdir()) / "ultron.db"
        print(
            f"[memory] could not open {DB_PATH} ({e}); "
            f"falling back to {fallback} (memory will not persist)",
            flush=True,
        )
        _conn = _connect(fallback)


def _ensure() -> sqlite3.Connection:
    if _conn is None:
        init()
    return _conn  # type: ignore[return-value]


def _ts() -> float:
    return time.time()


# ── conversation history ─────────────────────────────────────────
def log_message(role: str, content: Optional[str] = None, **meta: Any) -> None:
    """Store a transcript event. `content` may be None for pure tool events."""
    conn = _ensure()
    with _lock:
        conn.execute(
            "INSERT INTO messages (ts, role, content, meta) VALUES (?,?,?,?)",
            (_ts(), role, content, json.dumps(meta) if meta else None),
        )
        conn.commit()


def recent_messages(limit: int = HISTORY_LIMIT) -> List[dict]:
    """Return the most recent messages as [{'role', 'content'}, ...] oldest-first.

    Tool/meta events are skipped so the LLM sees a clean dialogue.
    """
    conn = _ensure()
    with _lock:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE content IS NOT NULL AND role IN ('user','assistant') "
            "ORDER BY ts DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"role": r["role"], "content": r["content"]}
        for r in reversed(rows)
    ]


def clear_all() -> None:
    conn = _ensure()
    with _lock:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM facts")
        conn.commit()


def stats() -> dict:
    conn = _ensure()
    with _lock:
        m = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        f = conn.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"]
    return {"messages": m, "facts": f}


# ── long-term facts ──────────────────────────────────────────────
def add_fact(text: str, source: str = "observation") -> int:
    conn = _ensure()
    with _lock:
        cur = conn.execute(
            "INSERT INTO facts (ts, source, text) VALUES (?,?,?)",
            (_ts(), source, text.strip()),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def search_facts(query: str, limit: int = 8) -> List[dict]:
    """Substring (LIKE) search over facts."""
    conn = _ensure()
    try:
        with _lock:
            rows = conn.execute(
                "SELECT id, ts, source, text FROM facts "
                "WHERE text LIKE ? ORDER BY ts DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def recall_context(query: str, limit: int = 8) -> str:
    """Human-readable fact digest for injection into the prompt."""
    hits = search_facts(query, limit)
    if not hits:
        return ""
    lines = [f"- {h['text']} (source: {h['source']})" for h in hits]
    return "Relevant long-term memory:\n" + "\n".join(lines)
