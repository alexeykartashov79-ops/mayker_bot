"""Single-process SQLite inbox/outbox with atomic state transitions.

Use one Store on one event-loop thread. A committed incoming update survives a
restart; an outgoing Telegram request remains pending until acknowledged. Telegram
does not provide sendMessage idempotency keys, so a crash after remote delivery
but before mark_sent can still cause a duplicate on retry.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any


CONTEXT_TIMEOUT = 7200


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _delay(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Delay must be a finite non-negative number")
    return result


def _empty_state() -> dict:
    return {"history": [], "order": None, "discount": None,
            "last_activity": 0, "handoff_until": 0}


class Store:
    def __init__(self, path: str | Path, context_timeout: float = CONTEXT_TIMEOUT):
        self.context_timeout = _delay(context_timeout)
        if self.context_timeout == 0:
            raise ValueError("context_timeout must be positive")
        self.path = str(path)
        if self.path != ":memory:":
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(target, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(descriptor)
            os.chmod(target, 0o600)
        self.conn = sqlite3.connect(self.path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS inbox (
                id INTEGER PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                next_at REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS inbox_chat_queue ON inbox(chat_id, status, id);
            CREATE TABLE IF NOT EXISTS outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_update INTEGER,
                chat_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                next_at REAL NOT NULL DEFAULT 0,
                after_id INTEGER,
                on_failure TEXT,
                fallback_for INTEGER UNIQUE,
                message_id INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS outbox_chat_queue ON outbox(chat_id, status, id);
            CREATE INDEX IF NOT EXISTS outbox_dependencies ON outbox(after_id);
            CREATE TABLE IF NOT EXISTS chat_state (
                chat_id INTEGER PRIMARY KEY,
                payload TEXT NOT NULL,
                last_activity REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    def close(self) -> None:
        self.conn.close()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_meta(self, key: str, value: Any) -> None:
        encoded = _json(value)
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", (key, encoded))

    def _ingest(self, update: dict, chat_id: int, now: float) -> bool:
        update_id = _integer(update.get("update_id"), "update_id")
        if update_id < 0:
            raise ValueError("update_id must not be negative")
        _integer(chat_id, "chat_id")
        cursor = self.conn.execute(
            "INSERT OR IGNORE INTO inbox(id,chat_id,payload,created_at,updated_at) VALUES(?,?,?,?,?)",
            (update_id, chat_id, _json(update), now, now),
        )
        return cursor.rowcount == 1

    def ingest(self, update: dict, chat_id: int) -> bool:
        with self.conn:
            return self._ingest(update, chat_id, time.time())

    def ingest_batch(self, entries: list[tuple[dict, int]], offset: int) -> None:
        _integer(offset, "offset")
        if offset < 0:
            raise ValueError("offset must not be negative")
        now = time.time()
        with self.conn:
            for update, chat_id in entries:
                self._ingest(update, chat_id, now)
            offset = max(offset, int(self.get_meta("polling_offset", 0)))
            self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                              ("polling_offset", _json(offset)))

    def pending_updates(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute("""
            SELECT i.* FROM inbox i
            WHERE i.status='pending' AND i.next_at<=?
              AND NOT EXISTS (
                  SELECT 1 FROM inbox earlier WHERE earlier.chat_id=i.chat_id
                    AND earlier.status='pending' AND earlier.id<i.id
              )
            ORDER BY i.id LIMIT ?
        """, (time.time(), limit)).fetchall()
        return [{"id": row["id"], "chat_id": row["chat_id"],
                 "update": json.loads(row["payload"]), "attempts": row["attempts"]}
                for row in rows]

    def fail_update(self, update_id: int, retry_delay: float = 5) -> None:
        now = time.time()
        delay = _delay(retry_delay)
        with self.conn:
            self.conn.execute("""
                UPDATE inbox SET attempts=attempts+1,next_at=?,updated_at=?
                WHERE id=? AND status='pending'
            """, (now + delay, now, update_id))

    def get_state(self, chat_id: int) -> dict:
        row = self.conn.execute("SELECT * FROM chat_state WHERE chat_id=?", (chat_id,)).fetchone()
        if row is None or time.time() - row["last_activity"] > self.context_timeout:
            return _empty_state()
        state = _empty_state()
        state.update(json.loads(row["payload"]))
        return state

    @staticmethod
    def _validate_action(action: dict, index: int | None = None) -> None:
        _integer(action.get("chat_id"), "chat_id")
        if action.get("method") not in {"sendMessage", "copyMessage"}:
            raise ValueError("Unsupported Telegram outbox method")
        if not isinstance(action.get("payload"), dict):
            raise ValueError("Outbox payload must be a dictionary")
        _json(action["payload"])
        after = action.get("after")
        if after is not None:
            _integer(after, "after")
            if index is None or not 0 <= after < index:
                raise ValueError("after must refer to an earlier action in the same commit")
        fallback = action.get("on_failure")
        if fallback is not None:
            if fallback.get("on_failure") is not None:
                raise ValueError("Nested failure actions are not allowed")
            Store._validate_action(fallback)

    def _insert_action(self, action: dict, now: float, source_update: int | None,
                       after_id: int | None = None, fallback_for: int | None = None) -> int:
        cursor = self.conn.execute("""
            INSERT INTO outbox(source_update,chat_id,method,payload,after_id,on_failure,
                               fallback_for,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?)
        """, (source_update, action["chat_id"], action["method"], _json(action["payload"]),
              after_id, _json(action["on_failure"]) if action.get("on_failure") else None,
              fallback_for, now, now))
        return cursor.lastrowid

    def commit(self, update_id: int, actions: list[dict],
               states: dict[int, dict] | None = None) -> None:
        now = time.time()
        with self.conn:
            update = self.conn.execute("SELECT status FROM inbox WHERE id=?", (update_id,)).fetchone()
            if update is None:
                raise ValueError("Cannot commit an update that has not been ingested")
            if update["status"] == "done":
                return
            for index, action in enumerate(actions):
                self._validate_action(action, index)
            ids = []
            for action in actions:
                after = action.get("after")
                ids.append(self._insert_action(action, now, update_id,
                                               ids[after] if after is not None else None))
            for chat_id, state in (states or {}).items():
                _integer(chat_id, "chat_id")
                activity = float(state.get("last_activity", now))
                if not math.isfinite(activity):
                    raise ValueError("last_activity must be finite")
                normalized = _empty_state()
                normalized.update(state)
                normalized["last_activity"] = activity
                self.conn.execute("INSERT OR REPLACE INTO chat_state(chat_id,payload,last_activity) VALUES(?,?,?)",
                                  (chat_id, _json(normalized), activity))
            self.conn.execute("UPDATE inbox SET status='done',updated_at=? WHERE id=?", (now, update_id))

    def pending_outbox(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute("""
            SELECT o.* FROM outbox o
            WHERE o.status='pending' AND o.next_at<=?
              AND (o.after_id IS NULL OR EXISTS (
                  SELECT 1 FROM outbox parent WHERE parent.id=o.after_id AND parent.status='sent'
              ))
              AND NOT EXISTS (
                  SELECT 1 FROM outbox earlier WHERE earlier.chat_id=o.chat_id
                    AND earlier.status='pending' AND earlier.id<o.id
              )
            ORDER BY o.id LIMIT ?
        """, (time.time(), limit)).fetchall()
        return [{"id": row["id"], "chat_id": row["chat_id"], "method": row["method"],
                 "payload": json.loads(row["payload"]), "attempts": row["attempts"]}
                for row in rows]

    def mark_sent(self, action_id: int, message_id: int | None = None) -> None:
        with self.conn:
            self.conn.execute("""
                UPDATE outbox SET status='sent',message_id=?,updated_at=?
                WHERE id=? AND status='pending'
            """, (message_id, time.time(), action_id))

    def fail_outbox(self, action_id: int, retry_delay: float = 5,
                    permanent: bool = False, max_attempts: int | None = 5) -> bool:
        """Record an unsuccessful send; return True if it is terminal.

        Calling again for an already failed/cancelled row is idempotent. A final
        failure cancels dependent success messages and enqueues its fallback once.
        max_attempts=None retries temporary failures indefinitely; a permanent
        failure always terminates immediately.
        """
        delay = _delay(retry_delay)
        if max_attempts is not None:
            _integer(max_attempts, "max_attempts")
            if max_attempts < 1:
                raise ValueError("max_attempts must be positive")
        now = time.time()
        with self.conn:
            row = self.conn.execute("SELECT * FROM outbox WHERE id=?", (action_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown outbox action")
            if row["status"] != "pending":
                return row["status"] in {"failed", "cancelled"}
            attempts = row["attempts"] + 1
            terminal = permanent or (max_attempts is not None and attempts >= max_attempts)
            self.conn.execute("""
                UPDATE outbox SET status=?,attempts=?,next_at=?,updated_at=? WHERE id=?
            """, ("failed" if terminal else "pending", attempts, now + delay, now, action_id))
            if terminal:
                self.conn.execute("""
                    WITH RECURSIVE children(id) AS (
                        SELECT id FROM outbox WHERE after_id=?
                        UNION ALL
                        SELECT o.id FROM outbox o JOIN children c ON o.after_id=c.id
                    )
                    UPDATE outbox SET status='cancelled',updated_at=?
                    WHERE status='pending' AND id IN (SELECT id FROM children)
                """, (action_id, now))
                if row["on_failure"]:
                    self._insert_action(json.loads(row["on_failure"]), now,
                                        row["source_update"], fallback_for=action_id)
            return terminal

    def cleanup(self, retention_days: int = 30) -> None:
        """Remove old completed data, retaining unfinished work and dependencies."""
        if isinstance(retention_days, bool) or retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        cutoff = time.time() - retention_days * 86400
        with self.conn:
            self.conn.execute("""
                DELETE FROM outbox WHERE status!='pending' AND updated_at<?
                  AND NOT EXISTS (
                      SELECT 1 FROM outbox child
                      WHERE child.after_id=outbox.id AND child.status='pending'
                  )
            """, (cutoff,))
            self.conn.execute("""
                DELETE FROM inbox WHERE status='done' AND updated_at<?
                  AND NOT EXISTS (
                      SELECT 1 FROM outbox o WHERE o.source_update=inbox.id AND o.status='pending'
                  )
            """, (cutoff,))
            self.conn.execute("""
                DELETE FROM chat_state WHERE last_activity<?
                  AND NOT EXISTS (SELECT 1 FROM inbox i WHERE i.chat_id=chat_state.chat_id AND i.status='pending')
                  AND NOT EXISTS (SELECT 1 FROM outbox o WHERE o.chat_id=chat_state.chat_id AND o.status='pending')
            """, (cutoff,))
