"""Append-only display archive. Source account databases are opened read-only.

This module never constructs an execution/strategy worker or a broker. Its own
SQLite file is separate from the financial ledger and is safe to rebuild from
saved evidence. Pagination fixes a watermark so live arrivals cannot move pages.
"""

import base64
import hashlib
import json
import sqlite3
import threading
import time
from collections import Counter
from contextlib import closing, contextmanager
from pathlib import Path


class HistoryUnavailable(Exception):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def event_key(source, payload, occurrence=0):
    return hashlib.sha256(canonical([source, payload, occurrence]).encode()).hexdigest()


def records(source, items):
    counts = Counter()
    for item in items:
        serial = canonical(item)
        occurrence = counts[serial]
        counts[serial] += 1
        timestamp = int(item.get("recorded_at", item["timestamp"]))
        category = (
            "fill"
            if source == "fill"
            else (
                "exception"
                if item.get("code")
                or any(
                    text in item["kind"] for text in ("unknown", "rejected", "review", "blocked")
                )
                else "event"
            )
        )
        yield (event_key(source, item, occurrence), timestamp, source, category, serial)


class LiveHistory:
    def __init__(self, source, archive, account_id, *, clock=None):
        self.source = Path(source)
        self.archive = Path(archive)
        self.account_id = account_id
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.lock = threading.RLock()
        self.archive.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connection() as db:
            db.executescript(
                "CREATE TABLE IF NOT EXISTS history_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, event_key TEXT NOT NULL UNIQUE, "
                "timestamp INTEGER NOT NULL, source TEXT NOT NULL, category TEXT NOT NULL, "
                "payload TEXT NOT NULL);"
                "CREATE INDEX IF NOT EXISTS history_time ON history_events(timestamp DESC,id DESC);"
                "CREATE INDEX IF NOT EXISTS history_category "
                "ON history_events(category,timestamp DESC,id DESC);"
                "CREATE TABLE IF NOT EXISTS history_meta "
                "(id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL);"
            )
        self.archive.chmod(0o600)

    @contextmanager
    def connection(self):
        # closing is explicit: sqlite's transaction context does not close handles.
        with closing(sqlite3.connect(self.archive, timeout=2)) as db, db:
            yield db

    def read_source(self, name):
        path = self.source / (name + ".sqlite")
        try:
            with closing(
                sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2)
            ) as db:
                db.execute("PRAGMA query_only=ON")
                row = db.execute("SELECT state FROM trx_state WHERE id=?", (name,)).fetchone()
            state = json.loads(row[0]) if row else None
            if not state or state.get("account_id") != self.account_id:
                raise HistoryUnavailable("history_account_mismatch")
            return state
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            raise HistoryUnavailable("history_source_unavailable") from exc

    def ingest(self, groups, *, metadata=None):
        with self.lock, self.connection() as db:
            prior = db.execute("SELECT payload FROM history_meta WHERE id=1").fetchone()
            meta = json.loads(prior[0]) if prior else {"account_id": self.account_id}
            if meta["account_id"] != self.account_id:
                raise HistoryUnavailable("history_account_mismatch")
            for source, items in groups.items():
                db.executemany(
                    "INSERT INTO history_events(event_key,timestamp,source,category,payload) "
                    "SELECT ?,?,?,?,? WHERE NOT EXISTS "
                    "(SELECT 1 FROM history_events WHERE event_key=?)",
                    [(*row, row[0]) for row in records(source, items)],
                )
            if metadata:
                meta.update(metadata)
            db.execute(
                "INSERT INTO history_meta VALUES(1,?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (canonical(meta),),
            )

    def sync(self):
        with self.lock:
            execution = self.read_source("execution")
            strategy = self.read_source("strategy")
            with self.connection() as db:
                row = db.execute("SELECT payload FROM history_meta WHERE id=1").fetchone()
                previous = json.loads(row[0]) if row else {}
            if previous.get("created_at", execution["created_at"]) != execution["created_at"]:
                raise HistoryUnavailable("history_ledger_mismatch")
            events = strategy.get("events", [])
            identities = {r[0] for r in records("strategy", events)}
            gap = previous.get("possible_strategy_gap", False)
            tail = previous.get("strategy_tail")
            if len(events) >= 500 and (not tail or tail not in identities):
                gap = True
            self.ingest(
                {
                    "strategy": events,
                    "execution": execution.get("events", []),
                    "fill": execution["book"].get("orders", []),
                },
                metadata={
                    "created_at": execution["created_at"],
                    "last_synced_at": self.clock(),
                    "possible_strategy_gap": gap,
                    "strategy_tail": list(records("strategy", events))[-1][0] if events else None,
                },
            )

            return strategy, execution

    def seed(self, evidence):
        """Import saved event evidence, never a financial state or reconstructed trade."""
        if evidence["account_id"] != self.account_id:
            raise HistoryUnavailable("history_account_mismatch")
        if (
            "created_at" in evidence
            and evidence["created_at"] != self.read_source("execution")["created_at"]
        ):
            raise HistoryUnavailable("history_ledger_mismatch")
        for snapshot in evidence["snapshots"]:
            self.ingest(snapshot)

    def query(self, *, start=None, end=None, kind="all", limit=30, cursor=None):
        if kind not in ("all", "fill", "exception") or not 1 <= limit <= 100:
            raise ValueError("invalid_history_query")
        if (
            (start is not None and start < 0)
            or (end is not None and end < 0)
            or (start is not None and end is not None and start >= end)
        ):
            raise ValueError("invalid_history_range")
        signature = hashlib.sha256(canonical([start, end, kind]).encode()).hexdigest()
        token = None
        if cursor:
            try:
                token = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
                if token["query"] != signature or any(
                    type(token[k]) is not int or token[k] < 0 for k in ("watermark", "at", "id")
                ):
                    raise ValueError()
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("invalid_history_cursor") from exc
        with self.lock, self.connection() as db:
            db.execute("BEGIN")
            watermark = (
                token["watermark"]
                if token
                else db.execute("SELECT COALESCE(MAX(id),0) FROM history_events").fetchone()[0]
            )
            where, args = ["id<=?"], [watermark]
            if start is not None:
                where.append("timestamp>=?")
                args.append(start)
            if end is not None:
                where.append("timestamp<?")
                args.append(end)
            if kind != "all":
                where.append("category=?")
                args.append(kind)
            clause = " AND ".join(where)
            total = db.execute(
                "SELECT COUNT(*) FROM history_events WHERE " + clause, args
            ).fetchone()[0]
            if token:
                where.append("(timestamp<? OR (timestamp=? AND id<?))")
                args.extend((token["at"], token["at"], token["id"]))
            rows = db.execute(
                "SELECT id,timestamp,source,category,payload FROM history_events WHERE "
                + " AND ".join(where)
                + " ORDER BY timestamp DESC,id DESC LIMIT ?",
                [*args, limit + 1],
            ).fetchall()
            available = db.execute(
                "SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM history_events"
            ).fetchone()
            meta_row = db.execute("SELECT payload FROM history_meta WHERE id=1").fetchone()
            meta = json.loads(meta_row[0]) if meta_row else {}
        page = rows[:limit]
        next_cursor = None
        if len(rows) > limit:
            last = page[-1]
            next_cursor = (
                base64.urlsafe_b64encode(
                    canonical(
                        {"watermark": watermark, "at": last[1], "id": last[0], "query": signature}
                    ).encode()
                )
                .decode()
                .rstrip("=")
            )
        return {
            "account_id": self.account_id,
            "rows": [
                {
                    "id": str(r[0]),
                    "timestamp": r[1],
                    "source": r[2],
                    "category": r[3],
                    "data": json.loads(r[4]),
                }
                for r in page
            ],
            "total": total,
            "next_cursor": next_cursor,
            "coverage": {
                k: meta.get(k) for k in ("created_at", "last_synced_at", "possible_strategy_gap")
            },
            "available": {"total": available[0], "first_at": available[1], "last_at": available[2]},
        }
