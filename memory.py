"""
Short-term conversational memory backed by SQLite.

Persists full chat turns (including source citations) per session so follow-up
questions survive reruns and browser refreshes when the session URL is kept.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).parent
MEMORY_DB_PATH = PROJECT_ROOT / "data" / "chat_memory.db"

# Max messages loaded from DB / sent to RAG memory (user + assistant pairs).
DEFAULT_MESSAGE_LIMIT = 40


def _connect() -> sqlite3.Connection:
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MEMORY_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                grounded INTEGER NOT NULL DEFAULT 0,
                sources_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            """
        )
        _migrate_messages_table(conn)


def _migrate_messages_table(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
    if "grounded" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN grounded INTEGER NOT NULL DEFAULT 0")
    if "sources_json" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN sources_json TEXT")
    conn.commit()


def new_session_id() -> str:
    return str(uuid.uuid4())


def ensure_session(session_id: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id) VALUES (?)
            ON CONFLICT(session_id) DO NOTHING
            """,
            (session_id,),
        )
        conn.commit()


def _touch_session(session_id: str, *, title: Optional[str] = None) -> None:
    ensure_session(session_id)
    with _connect() as conn:
        if title:
            conn.execute(
                """
                UPDATE sessions
                SET updated_at = datetime('now'),
                    title = COALESCE(title, ?)
                WHERE session_id = ?
                """,
                (title[:120], session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET updated_at = datetime('now') WHERE session_id = ?",
                (session_id,),
            )
        conn.commit()


def add_message(
    session_id: str,
    role: str,
    content: str,
    *,
    grounded: bool = False,
    sources: Optional[list[dict]] = None,
) -> None:
    init_db()
    sources_json = json.dumps(sources) if sources else None
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, grounded, sources_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, int(grounded), sources_json),
        )
        conn.commit()

    if role == "user":
        _touch_session(session_id, title=content.strip())
    else:
        _touch_session(session_id)


def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": row["role"], "content": row["content"]}
    if row["role"] == "assistant":
        msg["grounded"] = bool(row["grounded"])
        if row["sources_json"]:
            try:
                msg["sources"] = json.loads(row["sources_json"])
            except json.JSONDecodeError:
                msg["sources"] = []
        else:
            msg["sources"] = []
    return msg


def get_messages(session_id: str, *, limit: int = DEFAULT_MESSAGE_LIMIT) -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content, grounded, sources_json FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [_row_to_message(row) for row in reversed(rows)]


def get_history_for_rag(session_id: str, *, limit: int = DEFAULT_MESSAGE_LIMIT) -> list[dict]:
    """Plain role/content list for retrieval and LLM (excludes metadata)."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in get_messages(session_id, limit=limit)
    ]


def get_session_stats(session_id: str) -> dict[str, Any]:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT title, updated_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()["n"]
    return {
        "title": row["title"] if row else None,
        "updated_at": row["updated_at"] if row else None,
        "message_count": count,
    }


def get_history_summaries(session_id: str, *, limit: int = 8) -> list[dict]:
    """Compact pairs for the sidebar history panel."""
    messages = get_messages(session_id, limit=limit * 2)
    summaries: list[dict] = []
    pending_user: Optional[str] = None

    for msg in messages:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif msg["role"] == "assistant" and pending_user:
            summaries.append(
                {
                    "question": pending_user,
                    "answer_preview": msg["content"][:100] + ("…" if len(msg["content"]) > 100 else ""),
                    "grounded": msg.get("grounded", False),
                }
            )
            pending_user = None

    return summaries[-limit:]


def clear_session(session_id: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
