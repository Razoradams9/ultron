"""Persistent memory for Ultron: conversation log + long-term facts.

Two tables:
  messages  — full chat transcript (user/assistant/tool events)
  facts     — durable statements about the world/user, FTS-searchable
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, List, Optional

from .config import DB_PATH, HISTORY_LIMIT

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def init() -> None:
    global _conn
    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.executescript(
            """
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
        )
        _conn.commit()


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
