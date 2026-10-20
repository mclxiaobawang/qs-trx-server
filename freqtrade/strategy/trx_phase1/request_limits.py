"""Process-shared Bitget budgets. This database contains no credentials or responses.

Requests are paced (no token bursts); transactions end before any sleep/network I/O.
Financial writes wait at most half a second; longer waits defer to a later tick.
"""

import hashlib
import json
import random
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from copy import deepcopy
from email.utils import parsedate_to_datetime
from pathlib import Path
from uuid import uuid4

import ccxt


DEFAULT_PATH = Path(__file__).resolve().parents[3] / "user_data/trx_phase1/request-limits.sqlite"
IDENTITY_PATHS = {"v2/spot/account/info", "v3/account/info"}
RECONCILIATION = {
    "v2/spot/trade/orderInfo",
    "v2/spot/trade/fills",
    "v2/spot/account/assets",
    "v2/spot/account/bills",
    "v2/earn/savings/assets",
    "v2/earn/savings/records",
    "v2/earn/savings/subscribe-result",
    "v2/earn/savings/redeem-result",
}
_CACHE_LOCK = threading.Lock()
_CACHE = {}
_READ_LOCKS = {}


def coalesced(key, read, clock):
    """Only identity/immutable metadata, 30 seconds; never wallet, quota or candles."""
    with _CACHE_LOCK:
        if len(_CACHE) > 128:
            _CACHE.clear()
        lock = _READ_LOCKS.setdefault(key, threading.Lock())
    with lock:
        now = clock()
        value = _CACHE.get(key)
        if value and 0 <= now - value[0] < 30:
            return deepcopy(value[1])
        result = read()
        _CACHE[key] = (clock(), deepcopy(result))
        return result


class RateLimitWait(Exception):
    def __init__(self):
        super().__init__("request_rate_wait")


def credential_id(key, secret="", password=""):
    return hashlib.sha256(f"{key}\0{secret}\0{password}".encode()).hexdigest()


class RequestLimiter:
    def __init__(self, path=None, *, clock=time.time, sleep=time.sleep, jitter=None):
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.clock, self.sleep = clock, sleep
        self.jitter = jitter or (lambda: random.uniform(0, 0.25))  # noqa: S311 - retry jitter
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS buckets (key TEXT PRIMARY KEY, ready REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS queue (
                    id TEXT PRIMARY KEY, scope TEXT, priority INTEGER, created REAL, expires REAL
                );
                CREATE TABLE IF NOT EXISTS identities (
                    credential TEXT PRIMARY KEY, account TEXT, checked REAL
                );
                CREATE TABLE IF NOT EXISTS metrics (key TEXT PRIMARY KEY, value INTEGER);
                CREATE TABLE IF NOT EXISTS admissions (scope TEXT, stamp REAL);
            """)
        self.path.chmod(0o600)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            yield db
        finally:
            db.close()

    def scope(self, db, credential):
        if credential is None:
            return "public"
        row = db.execute(
            "SELECT account FROM identities WHERE credential=?", (credential,)
        ).fetchone()
        return "private:" + (row[0] if row else "unbound")

    def identity(self, credential):
        with self.db() as db:
            row = db.execute(
                "SELECT account, checked FROM identities WHERE credential=?", (credential,)
            ).fetchone()
        return row[0] if row and 0 <= self.clock() - row[1] < 30 else None

    def bind(self, credential, account):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            # Carry bootstrap pacing into the verified UID scope, including a
            # different API key for that same UID. Only SHA256 identities persist.
            for key, ready in db.execute(
                "SELECT key, ready FROM buckets WHERE key LIKE 'private:unbound%'"
            ).fetchall():
                self.set_ready(db, key.replace("private:unbound", "private:" + account, 1), ready)
            db.execute(
                "INSERT OR REPLACE INTO identities VALUES (?,?,?)",
                (credential, account, self.clock()),
            )
            db.commit()

    @staticmethod
    def set_ready(db, key, ready):
        db.execute(
            "INSERT INTO buckets VALUES (?,?) ON CONFLICT(key) DO UPDATE "
            "SET ready=MAX(ready,excluded.ready)",
            (key, ready),
        )

    @staticmethod
    def ready(db, key):
        row = db.execute("SELECT ready FROM buckets WHERE key=?", (key,)).fetchone()
        return row[0] if row else 0

    def acquire(self, path, credential=None, *, priority=1, max_wait=15, write=False):
        ticket, start = uuid4().hex, self.clock()
        try:
            while True:
                now = self.clock()
                with self.db() as db:
                    db.execute("BEGIN IMMEDIATE")
                    scope = self.scope(db, credential)
                    db.execute("DELETE FROM queue WHERE expires<?", (now,))
                    db.execute(
                        "INSERT OR REPLACE INTO queue VALUES (?,?,?,?,?)",
                        (ticket, scope, priority, start, now + 20),
                    )
                    first = db.execute(
                        "SELECT id FROM queue WHERE scope=? ORDER BY priority,created,id LIMIT 1",
                        (scope,),
                    ).fetchone()[0]
                    intervals = {scope: 0.125, scope + ":" + path: 0.125}
                    if path in IDENTITY_PATHS:
                        intervals[scope + ":identity"] = 1.2
                    if write and path == "v2/earn/savings/redeem":
                        intervals[scope + ":redemption"] = 61
                    ready = max(self.ready(db, k) for k in (*intervals, "cooldown:all"))
                    if first == ticket and ready <= now:
                        for key, interval in intervals.items():
                            self.set_ready(db, key, now + interval)
                        db.execute("DELETE FROM queue WHERE id=?", (ticket,))
                        db.execute("DELETE FROM admissions WHERE stamp<?", (now - 60,))
                        db.execute("INSERT INTO admissions VALUES (?,?)", (scope, now))
                        db.execute(
                            "INSERT INTO metrics VALUES ('requests',1) ON CONFLICT(key) "
                            "DO UPDATE SET value=value+1"
                        )
                        db.commit()
                        return
                    db.commit()
                if (write and (ready - now > 0.5 or now - start >= 0.5)) or now - start >= max_wait:
                    raise RateLimitWait()
                self.sleep(min(0.05, max_wait - (now - start)))
        finally:
            with self.db() as db:
                db.execute("DELETE FROM queue WHERE id=?", (ticket,))

    def limited(self, headers=None, attempt=0):
        now, delay = self.clock(), min(8, 2**attempt) + self.jitter()
        value = next((v for k, v in (headers or {}).items() if k.lower() == "retry-after"), None)
        if value is not None:
            try:
                delay = max(delay, float(value))
            except (TypeError, ValueError):
                try:
                    delay = max(delay, parsedate_to_datetime(value).timestamp() - now)
                except (TypeError, ValueError, OverflowError):
                    pass
        # Untrusted Retry-After may extend a cooldown, never cause a long sleep.
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            self.set_ready(db, "cooldown:all", now + min(86400, delay))
            db.execute(
                "INSERT INTO metrics VALUES ('responses_429',1) ON CONFLICT(key) "
                "DO UPDATE SET value=value+1"
            )
            db.commit()

    @classmethod
    def status(cls, path=None):
        path = Path(path) if path is not None else DEFAULT_PATH
        now = time.time()
        result = {
            "private_rps": 8,
            "public_rps": 8,
            "identity_interval_seconds": 1.2,
            "redemption_interval_seconds": 61,
            "requests": 0,
            "responses_429": 0,
            "queued": 0,
            "private_last_second": 0,
            "public_last_second": 0,
            "cooldown_until": None,
            "redemption_ready_at": None,
        }
        if not path.exists():
            return result
        try:
            return cls._read_status(path, now, result)
        except sqlite3.Error:
            # Metrics failure must not hide the rest of the account page.
            # Requests themselves still fail closed if the scheduler is unavailable.
            return {**result, "error": "request_limit_storage_unavailable"}

    @staticmethod
    def _read_status(path, now, result):
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)) as db, db:
            result.update(dict(db.execute("SELECT key,value FROM metrics")))
            for scope, count in db.execute(
                "SELECT CASE WHEN scope='public' THEN 'public' ELSE 'private' END, COUNT(*) "
                "FROM admissions WHERE stamp>? GROUP BY 1",
                (now - 1,),
            ):
                result[scope + "_last_second"] = count
            result["queued"] = db.execute(
                "SELECT COUNT(*) FROM queue WHERE expires>=?", (now,)
            ).fetchone()[0]
            for name, pattern in (
                ("cooldown_until", "cooldown:all"),
                ("redemption_ready_at", "%:redemption"),
            ):
                ready = db.execute(
                    "SELECT MAX(ready) FROM buckets WHERE key LIKE ?", (pattern,)
                ).fetchone()[0]
                result[name] = int(ready * 1000) if ready and ready > now else None
        return result


class LimitedReadMixin:
    """CCXT GET-only clients: gate before signing, retry only explicit 429s.

    The response hook catches non-JSON HTTP 429 before CCXT parses/raises it.
    Candle responses and wallet/product responses are deliberately never cached.
    """

    def request_limiter(self):
        if not hasattr(self, "_shared_limiter"):
            self._shared_limiter = RequestLimiter()
        return self._shared_limiter

    def request(
        self, path, api="public", method="GET", params=None, headers=None, body=None, config=None
    ):
        if method.upper() != "GET":
            raise PermissionError("Shared read transport is GET only")
        public = api == "public" or (isinstance(api, list) and api and api[0] == "public")
        credential = None if public else credential_id(self.apiKey, self.secret, self.password)
        limiter = self.request_limiter()

        def read():
            return self._limited_read(
                path, api, method, params, headers, body, config, credential, limiter
            )

        if path in IDENTITY_PATHS or path == "v2/spot/public/symbols":
            key = (
                "ccxt",
                str(limiter.path),
                credential,
                path,
                json.dumps(params or {}, sort_keys=True),
            )
            return coalesced(key, read, limiter.clock)
        return read()

    def _limited_read(self, path, api, method, params, headers, body, config, credential, limiter):
        for attempt in range(4):
            self._limit_attempt = attempt
            try:
                limiter.acquire(path, credential, priority=getattr(self, "request_priority", 2))
            except RateLimitWait:
                raise ccxt.RateLimitExceeded("request_rate_wait") from None
            # Disable CCXT's independent retry loop: each actual request must
            # obtain a new permit, and be freshly signed after any queue wait.
            self.options["maxRetriesOnFailure"] = 0
            self.enableRateLimit = False  # Shared admission happens before signing.
            self._response_limited = False
            try:
                response = super().request(
                    path,
                    api,
                    method,
                    params or {},
                    headers,
                    body,
                    {**(config or {}), "maxRetriesOnFailure": 0},
                )
            except ccxt.RateLimitExceeded:
                if not self._response_limited:
                    limiter.limited(attempt=attempt)
                if attempt < 3:
                    continue
                raise ccxt.RateLimitExceeded("request_rate_wait") from None
            if isinstance(response, dict) and str(response.get("code")) == "429":
                limiter.limited(attempt=attempt)
                if attempt < 3:
                    continue
                raise ccxt.RateLimitExceeded("request_rate_wait")
            if path in IDENTITY_PATHS and isinstance(response, dict):
                uid = response.get("data", {}).get("userId")
                if uid and credential:
                    limiter.bind(credential, hashlib.sha256(str(uid).encode()).hexdigest())
            return response

    def on_rest_response(
        self, code, reason, url, method, headers, body, request_headers, request_body
    ):
        try:
            payload = json.loads(body)
        except (ValueError, TypeError):
            payload = None
        if code == 429 or (isinstance(payload, dict) and str(payload.get("code")) == "429"):
            self.request_limiter().limited(headers, getattr(self, "_limit_attempt", 0))
            self._response_limited = True
            raise ccxt.RateLimitExceeded("request_rate_wait")
        return super().on_rest_response(
            code, reason, url, method, headers, body, request_headers, request_body
        )
