"""Independent C1 forward observation using public prices and virtual assets only."""

import fcntl
import hashlib
import json
import sqlite3
import threading
import time
from decimal import ROUND_DOWN
from pathlib import Path
from uuid import uuid4

import pandas as pd

from freqtrade.strategy.trx_phase1.candles import normalized_volatility
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import StateStore, money
from freqtrade.strategy.trx_phase1.price_position import DAY, OFFSET, build_price_reference
from freqtrade.strategy.trx_phase1.request_limits import RequestLimiter
from freqtrade.strategy.trx_phase1.research_earn import HistoricalEarn
from freqtrade.strategy.trx_phase1.shadow_execution import ShadowExecution, SimulatedBroker
from freqtrade.strategy.trx_phase1.shadow_market import PublicShadowMarket, seed_history


DEFAULT_EXPERIMENT = "a108ad7e-ed76-41e9-ac10-aa07b0e6b95e"
DEFAULT_RUN = "8c2b6b9d-dcd2-40c1-a9af-4d624b5990cf"
DEFAULT_LABEL = "C1 D | 首次 TRX 10% / USDGO 90% | 含理财冻结情景"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class ShadowStore(StateStore):
    def __init__(self, path):
        super().__init__(path)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS shadow_candles "
                "(timestamp INTEGER PRIMARY KEY, open REAL, high REAL, low REAL, "
                "close REAL, volume REAL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS shadow_samples "
                "(timestamp INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS shadow_candle_revisions "
                "(timestamp INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def save(self, key, version, state, sample=None):
        payload = json.dumps(state, allow_nan=False)
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            if version == 0:
                db.execute("INSERT INTO trx_state VALUES (?, 1, ?)", (key, payload))
            elif (
                db.execute(
                    "UPDATE trx_state SET version=version+1, state=? WHERE id=? AND version=?",
                    (payload, key, version),
                ).rowcount
                != 1
            ):
                raise RuntimeError("concurrent_shadow_change")
            if sample is not None:
                db.execute(
                    "INSERT INTO shadow_samples VALUES (?, ?) ON CONFLICT(timestamp) "
                    "DO UPDATE SET payload=excluded.payload",
                    (sample["timestamp"], json.dumps(sample, allow_nan=False)),
                )
        return version + 1

    def candles(self):
        with sqlite3.connect(self.path) as db:
            data = pd.read_sql_query("SELECT * FROM shadow_candles ORDER BY timestamp", db)
        if data.empty:
            return None
        data["date"] = pd.to_datetime(data.pop("timestamp"), unit="ms", utc=True)
        return data

    def add_candles(self, data):
        with sqlite3.connect(self.path) as db:
            last = db.execute("SELECT MAX(timestamp) FROM shadow_candles").fetchone()[0]
            stamps = pd.DatetimeIndex(data.date).as_unit("ms").asi8
            new = data.loc[stamps > (last if last is not None else 0)]
            rows = [
                (
                    int(pd.Timestamp(row.date).timestamp() * 1000),
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.volume,
                )
                for row in new.itertuples()
            ]
            db.executemany("INSERT INTO shadow_candles VALUES (?, ?, ?, ?, ?, ?)", rows)

    def curve(self):
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT payload FROM shadow_samples ORDER BY timestamp").fetchall()
        # Full minute observations remain in SQLite; the UI receives a bounded sample.
        stride = max(1, len(rows) // 2000)
        selected = rows[::stride]
        if rows and selected[-1] != rows[-1]:
            selected.append(rows[-1])
        return [json.loads(row[0]) for row in selected]

    def revisions(self):
        with sqlite3.connect(self.path) as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM shadow_candle_revisions ORDER BY timestamp"
                )
            ]


def create_shadow(root, source_path, seed_path, label=DEFAULT_LABEL):
    root, source_path, seed_path = Path(root), Path(source_path), Path(seed_path)
    saved = json.loads(source_path.read_text())
    cfg = ResearchConfig.from_saved(saved["config"])
    if cfg.model_dump() != saved["config"]:
        raise ValueError("complete_parameter_snapshot_required")
    provenance = saved.get("provenance", {})
    if provenance.get("exchange") != "bitget" or provenance.get("pair") != "TRX/USDT":
        raise ValueError("incompatible_source_market")
    if (
        cfg.startup not in ("allocated_position", "half_position")
        or cfg.sell_mode != "proportional"
    ):
        raise ValueError("unsupported_shadow_startup")
    if not seed_path.is_file():
        raise ValueError("history_seed_unavailable")
    shadow_id = str(uuid4())
    store = ShadowStore(root / shadow_id / "shadow.sqlite")
    state = {
        "schema": "trx-c1-shadow-v1",
        "shadow_id": shadow_id,
        "label": label,
        "created_at": int(time.time() * 1000),
        "activated_at": None,
        "source": {
            "run_id": saved["run_id"],
            "experiment_id": saved.get("experiment_id"),
            "snapshot_sha256": digest(saved["config"]),
            "parameters": saved["config"],
            "result_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "provenance": provenance,
            "historical_start": saved["start_utc"],
            "historical_end": saved["end_utc"],
        },
        "seed_path": str(seed_path.resolve()),
        "engine": Engine(cfg).state(),
        "execution": {},
        "phase": "created",
        "error": None,
        "warmup_until": None,
        "startup_complete": False,
        "startup_allocation": None,
        "quote": None,
        "market_rules": None,
        "peak": cfg.initial_cash,
        "max_drawdown": 0.0,
        "price_reference": None,
        "history": None,
    }
    store.save("shadow", 0, state)
    return shadow_id


class ShadowRuntime:
    def __init__(self, path, market=None):
        self.store = ShadowStore(Path(path))
        self.version, self.state = self.store.load("shadow")
        if self.state is None:
            raise FileNotFoundError("shadow_not_found")
        self.engine = Engine.restore(self.state["engine"])
        self.execution = ShadowExecution(
            SimulatedBroker(self.store.path), self.state["shadow_id"], self.state["execution"]
        )
        self.market = market or PublicShadowMarket()
        if market is None and self.store.path.parents[1].name == "shadows":
            self.market.client._shared_limiter = RequestLimiter(
                self.store.path.parents[2] / "request-limits.sqlite"
            )
        self.data = None
        self.reference = None
        self.reference_day = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.recovering = True
        self.worker_lock = None
        self.rules_checked = False

    def persist(self, engine=None, sample=None):
        if engine is not None:
            self.engine = engine
        self.state["engine"] = self.engine.state()
        self.state["execution"] = self.execution.state()
        self.version = self.store.save("shadow", self.version, self.state, sample)

    def set_reference(self, timestamp):
        cfg = self.engine.config
        day = (timestamp + OFFSET) // DAY
        if self.reference_day != day:
            self.reference = (
                build_price_reference(self.data, cfg) if cfg.price_position_enabled else None
            )
            self.reference_day = day
        value = self.reference.value_at(timestamp) if self.reference is not None else None
        if cfg.price_position_enabled and value is None:
            raise ValueError("complete_p75_history_required")
        self.engine.price_position_day = day
        self.engine.price_position_reference = value
        self.state["price_reference"] = {
            "day": day,
            "value": value,
            "history_end_exclusive": day * DAY - OFFSET,
            "lookback_days": cfg.price_position_lookback_days,
            "quantile": cfg.price_position_quantile,
        }

    def sample(self, timestamp, price):
        book = self.engine.book
        summary = book.summary(price)
        self.state["peak"] = max(self.state["peak"], summary["equity"])
        self.state["max_drawdown"] = max(
            self.state["max_drawdown"], 1 - summary["equity"] / self.state["peak"]
        )
        point = {"timestamp": timestamp, "price": price, **summary}
        self.persist(sample=point)

    def activate(self, now, quote):
        cfg, book = self.engine.config, self.engine.book
        self.engine.earn = earn = HistoricalEarn(book, cfg, now)
        self.engine.prepare_cash = earn.prepare_cash
        policy = earn.policy
        if not policy or not cfg.usdgo_product_available or not policy.quote_usable(now):
            raise ValueError("startup_fx_unavailable")
        budget = money(cfg.initial_cash) * (1 - money(cfg.initial_buy_fraction))
        mark, _, ask = policy.quote(now)
        quantity = (budget / (money(ask) * (1 + money(cfg.fx_fee)))).quantize(
            money(10) ** -cfg.fx_quantity_decimals, rounding=ROUND_DOWN
        )
        if cfg.usdgo_earn_cap and quantity > money(cfg.usdgo_earn_cap):
            raise ValueError("startup_quota_exceeded")
        paid = quantity * money(ask) * (1 + money(cfg.fx_fee))
        book.convert(f"shadow-usdgo-{now}", "to_usdgo", paid, quantity, mark)
        earn.subscribe_usdgo(quantity, now)
        earn.reconcile_usdgo()
        earn.conversions.append(
            {
                "timestamp": now,
                "side": "to_usdgo",
                "quantity": float(quantity),
                "paid": float(paid),
                "received": float(quantity),
                "rate": ask,
                "fee": float(paid - quantity * money(ask)),
                "simulated": True,
                "reason": "initial_allocated_position",
            }
        )
        self.engine.paused = False
        self.engine.last_price = quote["asks"][0][0]
        # Skip the partially observed activation minute. Historical seed is for
        # reference/volatility only; it is never traded or paid simulated interest.
        self.engine.last_close = (now // 60000 + 1) * 60000
        self.state["activated_at"] = now
        self.state["warmup_until"] = self.engine.last_close + cfg.volatility_window * 60000
        self.engine.make_plan(
            "buy",
            now,
            quote["asks"][0][0],
            0,
            now + 60000,
            startup=True,
            reference=quote["asks"][0][0],
        )
        self.state["startup_allocation"] = {
            "usdgo_budget_usdt": float(budget),
            "usdgo_quantity": float(quantity),
            "trx_budget_including_fee_usdt": cfg.initial_cash * cfg.initial_buy_fraction,
            "price_source": "observed_public_orderbook",
        }
        self.persist()
        self.execution.prepare(self.engine, quote, now, self.state["market_rules"])
        self.persist()
        self.engine = self.execution.resolve(self.engine, now, self.persist)
        self.complete_startup(now)

    def complete_startup(self, now):
        if self.state["startup_complete"]:
            return
        if self.engine.anchor and not self.execution.outbox and not self.engine.pending:
            self.state["startup_complete"] = True
            self.engine.initial_anchor_at = now
            self.engine.buy_at = self.engine.sell_at = 0
            self.state["startup_allocation"].update(
                trx_quantity=float(self.engine.book.principal), anchor=self.engine.anchor
            )
            self.persist()
        elif not self.execution.outbox and not self.engine.pending:
            raise ValueError("startup_not_filled")

    def poll(self):  # noqa: C901 - ordered market, accounting and execution checkpoints
        now, _ = self.market.snapshot()
        if self.data is None:
            self.data = self.store.candles()
            if self.data is None:
                self.data = seed_history(
                    self.state["seed_path"], now, self.engine.config.price_position_lookback_days
                )
        updated = self.market.update(self.data, now)
        self.store.add_candles(updated)
        self.data = updated
        now, quote = self.market.snapshot()  # Fetch again after any backfill, before execution.
        if self.stop_event.is_set():
            return
        close_at = int(pd.Timestamp(self.data.date.iloc[-1]).timestamp() * 1000) + 60000
        if now - close_at > 120000:
            raise ValueError("closed_market_data_stale")
        self.state["quote"] = quote
        self.state["history"] = {
            "start": str(self.data.date.iloc[0]),
            "end": str(self.data.date.iloc[-1]),
            "rows": len(self.data),
            "latest_close_at": close_at,
        }
        if not self.rules_checked or now - self.state["market_rules"]["observed_at"] > DAY:
            self.state["market_rules"] = self.market.rules()
            self.rules_checked = True
        if self.state["activated_at"] is None:
            self.set_reference(now)
            self.activate(now, quote)
            self.recovering = False

        # Ingest already accepted fills before accruing intervening hours. Open
        # orders stay reserved until the gap has accrued and cancellation is confirmed.
        if self.execution.outbox:
            order = self.execution.broker.query(self.execution.outbox["client_id"])
            if order:
                self.engine = self.execution.ingest(self.engine, order, now)
                self.persist()
        stamps = pd.DatetimeIndex(self.data.date).as_unit("ms").asi8 + 60000
        indexes = (stamps > self.engine.last_close).nonzero()[0]
        tail = self.data.iloc[
            max(
                0,
                (int(indexes[0]) if len(indexes) else len(self.data))
                - self.engine.config.volatility_window
                - 1,
            ) :
        ]
        sigmas = normalized_volatility(tail, self.engine.config.volatility_window)
        had_gap = self.recovering or (len(indexes) > 0 and now - int(stamps[indexes[0]]) > 120000)
        if had_gap and self.engine.pending and not self.execution.outbox:
            self.engine.finish(self.engine.last_close)
        for index in indexes:
            timestamp, price = int(stamps[index]), float(self.data.close.iloc[index])
            self.set_reference(timestamp)
            warming = timestamp < self.state["warmup_until"]
            self.engine.paused = had_gap or warming or not self.state["startup_complete"]
            self.engine.earn.advance(timestamp, price)
            self.engine.on_close(timestamp, price, float(sigmas.loc[index]))
            self.engine.earn.allocate_cash(timestamp, self.engine.pending)
            self.sample(timestamp, price)

        if (now + OFFSET) // DAY > (close_at + OFFSET) // DAY:
            self.engine.paused = True
            self.state.update(phase="confirming_daily_reference", error=None)
            self.persist()
            return
        self.set_reference(now)
        self.engine.paused = False
        self.engine = self.execution.resolve(self.engine, now, self.persist, cancel=had_gap)
        if had_gap:
            if self.engine.pending:
                self.engine.finish(now)
            if self.engine.anchor and now >= self.state["warmup_until"]:
                self.engine.schedule(now, self.engine.last_price, self.engine.last_sigma)
        self.complete_startup(now)
        warming = now < self.state["warmup_until"]
        self.engine.paused = warming
        if self.stop_event.is_set():
            return
        if not warming and not self.execution.outbox and self.engine.pending:
            price = (
                quote["asks"][0][0] if self.engine.pending.side == "buy" else quote["bids"][0][0]
            )
            if self.engine.executable(now, price):
                self.execution.prepare(self.engine, quote, now, self.state["market_rules"])
                self.persist()
                self.engine = self.execution.resolve(self.engine, now, self.persist)
        self.state.update(phase="warming_up" if warming else "observing", error=None)
        self.engine.reason = "warming_up" if warming else self.engine.reason
        self.recovering = False
        self.sample(now, (quote["asks"][0][0] + quote["bids"][0][0]) / 2)

    def _loop(self):
        try:
            self._run_loop()
        finally:
            if self.worker_lock:
                fcntl.flock(self.worker_lock, fcntl.LOCK_UN)
                self.worker_lock.close()
                self.worker_lock = None

    def _run_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                try:
                    self.poll()
                except Exception as exc:
                    # Restore the last atomic snapshot; never continue an uncommitted fill.
                    self.version, self.state = self.store.load("shadow")
                    self.engine = Engine.restore(self.state["engine"])
                    self.execution = ShadowExecution(
                        self.execution.broker, self.state["shadow_id"], self.state["execution"]
                    )
                    self.engine.paused = True
                    self.state.update(
                        phase="blocked", error=type(exc).__name__ + ":" + str(exc)[:160]
                    )
                    self.recovering = True
                    self.persist()
            self.stop_event.wait(15)
        with self.lock:
            self.engine = self.execution.resolve(
                self.engine, int(time.time() * 1000), self.persist, cancel=True
            )
            self.engine.paused = True
            self.state["phase"] = "paused"
            self.persist()

    def start(self):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return
            # A second worker cannot overwrite a newer account snapshot.
            version, _ = self.store.load("shadow")
            if version != self.version:
                raise RuntimeError("concurrent_shadow_change")
            self.worker_lock = self.store.path.with_suffix(".lock").open("a")
            try:
                fcntl.flock(self.worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.worker_lock.close()
                self.worker_lock = None
                raise RuntimeError("shadow_already_running") from None
            self.stop_event.clear()
            self.recovering = True
            self.state["phase"] = "connecting"
            self.persist()
            self.thread = threading.Thread(
                target=self._loop, daemon=True, name="trx-shadow-" + self.state["shadow_id"][:8]
            )
            self.thread.start()

    def stop(self):
        self.stop_event.set()

    def status(self, curve=True):
        # Read committed state without waiting on an in-flight network request.
        _, state = self.store.load("shadow")
        engine = Engine.restore(state["engine"])
        running = self.thread is not None and self.thread.is_alive()
        quote = state["quote"]
        price = (quote["asks"][0][0] + quote["bids"][0][0]) / 2 if quote else engine.last_price or 1
        return {
            **{
                key: value
                for key, value in state.items()
                if key not in ("engine", "execution", "seed_path")
            },
            "phase": state["phase"]
            if running
            else "paused"
            if state["activated_at"]
            else state["phase"],
            "running": running,
            "stopping": running and self.stop_event.is_set(),
            "stale": quote is None or int(time.time() * 1000) - quote["timestamp"] > 30000,
            "real_trading_enabled": False,
            "real_earn_enabled": False,
            "summary": {
                **engine.status(),
                **engine.book.summary(price),
                "trading_turnover": sum(
                    row["quantity"] * row["price"] for row in engine.book.orders
                ),
                "max_drawdown": state["max_drawdown"],
            },
            "orders": engine.book.reported_fills()[-100:],
            "outbox": state["execution"].get("outbox"),
            "execution_orders": state["execution"].get("orders", [])[-100:],
            "equity_curve": self.store.curve() if curve else [],
            "database_path": str(self.store.path.resolve()),
            "data_reconciliations": self.store.revisions(),
        }
