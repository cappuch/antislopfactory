import asyncio
import json
import os
import secrets
import sqlite3
import threading
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "threads.db")

_PRAGMAS = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA mmap_size=268435456;
PRAGMA cache_size=-64000;
PRAGMA busy_timeout=5000;
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id  TEXT PRIMARY KEY,
    risk       INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id                 INTEGER PRIMARY KEY,
    thread_id          TEXT NOT NULL REFERENCES threads(thread_id),
    seq                INTEGER NOT NULL,
    role               TEXT NOT NULL,
    reasoning_content  TEXT NOT NULL DEFAULT '',
    content            TEXT NOT NULL DEFAULT '',
    metadata           TEXT,
    embedding          BLOB,
    created_at         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_thread_seq ON messages(thread_id, seq);
CREATE INDEX IF NOT EXISTS idx_threads_risk ON threads(risk);
CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at DESC);
"""


def genid() -> str:
    """Generate a compact time-sortable ID: 8-char hex timestamp + 6-char random."""
    t = int(time.time() * 1000)
    return f"{t:011x}-{secrets.token_hex(3)}"


class ThreadStore:
    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            conn.executescript(_PRAGMAS)
            self._local.conn = conn
            with self._lock:
                self._connections.append(conn)
        return conn

    def _init_tables(self) -> None:
        self._conn().executescript(_SCHEMA)

    def _close_all(self) -> None:
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
        self._local.__dict__.pop("conn", None)

    async def initialize(self) -> None:
        await asyncio.to_thread(self._init_tables)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_all)

    def _create_thread(self, thread_id: str | None, risk: int = 0) -> str:
        tid = thread_id or genid()
        now = time.time()
        self._conn().execute(
            "INSERT OR IGNORE INTO threads (thread_id, risk, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (tid, risk, now, now),
        )
        self._conn().commit()
        return tid

    def _get_thread(self, thread_id: str) -> dict | None:
        row = self._conn().execute(
            "SELECT thread_id, risk, created_at, updated_at FROM threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return dict(row) if row else None

    def _update_risk(self, thread_id: str, risk: int) -> None:
        self._conn().execute(
            "UPDATE threads SET risk = ?, updated_at = ? WHERE thread_id = ?",
            (risk, time.time(), thread_id),
        )
        self._conn().commit()

    async def create_thread(self, thread_id: str | None = None, risk: int = 0) -> str:
        return await asyncio.to_thread(self._create_thread, thread_id, risk)

    async def get_thread(self, thread_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_thread, thread_id)

    async def update_risk(self, thread_id: str, risk: int) -> None:
        await asyncio.to_thread(self._update_risk, thread_id, risk)

    def _append_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        reasoning_content: str = "",
        metadata: dict | None = None,
        embedding=None,
    ) -> int:
        conn = self._conn()
        now = time.time()

        # upsert thread
        conn.execute(
            "INSERT INTO threads (thread_id, risk, created_at, updated_at) "
            "VALUES (?, 0, ?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at",
            (thread_id, now, now),
        )

        # compute next seq
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        seq = row[0]

        # serialize optional fields
        meta_json = json.dumps(metadata) if metadata is not None else None
        emb_blob = None
        if embedding is not None:
            emb_blob = embedding.astype("float32").tobytes()

        cur = conn.execute(
            "INSERT INTO messages "
            "(thread_id, seq, role, reasoning_content, content, metadata, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (thread_id, seq, role, reasoning_content, content, meta_json, emb_blob, now),
        )
        conn.commit()
        return cur.lastrowid

    def _get_messages(
        self, thread_id: str, limit: int | None = None, offset: int = 0
    ) -> list[dict]:
        sql = (
            "SELECT id, thread_id, seq, role, reasoning_content, content, "
            "metadata, embedding, created_at "
            "FROM messages WHERE thread_id = ? ORDER BY seq"
        )
        params: list = [thread_id]

        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)

        rows = self._conn().execute(sql, params).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            if d["metadata"] is not None:
                d["metadata"] = json.loads(d["metadata"])
            if d["embedding"] is not None:
                import numpy as np
                d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            out.append(d)
        return out

    async def append_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        reasoning_content: str = "",
        metadata: dict | None = None,
        embedding=None,
    ) -> int:
        return await asyncio.to_thread(
            self._append_message, thread_id, role, content,
            reasoning_content, metadata, embedding,
        )

    async def get_messages(
        self, thread_id: str, limit: int | None = None, offset: int = 0
    ) -> list[dict]:
        return await asyncio.to_thread(self._get_messages, thread_id, limit, offset)

    def _list_threads(self) -> list[dict]:
        rows = self._conn().execute(
            "SELECT t.thread_id, t.risk, t.created_at, t.updated_at, "
            "  m.content AS first_user_message "
            "FROM threads t "
            "LEFT JOIN messages m ON m.thread_id = t.thread_id "
            "  AND m.role = 'user' "
            "  AND m.seq = ("
            "    SELECT MIN(seq) FROM messages "
            "    WHERE thread_id = t.thread_id AND role = 'user'"
            "  ) "
            "ORDER BY t.updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    async def list_threads(self) -> list[dict]:
        return await asyncio.to_thread(self._list_threads)
