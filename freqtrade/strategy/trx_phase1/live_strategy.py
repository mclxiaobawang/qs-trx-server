"""Persistent C1 signals driving the real execution journal, without simulated Earn.

Construction and status reads never start a worker. Only an explicit caller can
start writes. Public market catch-up advances the clock without backfilling trades.
"""

import fcntl
import json
import math
import sqlite3
import threading
import time
from contextlib import closing
from copy import deepcopy

import ccxt
import pandas as pd

from freqtrade.strategy.trx_phase1.candles import normalized_volatility, validate_minutes
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import Ledger, StateStore, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest
from freqtrade.strategy.trx_phase1.live_funding import LiveFunding
from freqtrade.strategy.trx_phase1.live_observation import observed_close
from freqtrade.strategy.trx_phase1.price_position import DAY, OFFSET, build_price_reference
from freqtrade.strategy.trx_phase1.shadow_market import validate_quote


TERMINAL = {"completed", "expired", "failed", "needs_review"}
EVIDENCE_WAIT_CODES = {
    "wallet_reconciliation_required",
    "owned_balance_not_spendable",
    "interest_holding_evidence_incomplete",
    "interest_arrival_bill_unverified",
}
PUBLIC_RETRY_CODES = {
    "stale_execution_quote",
    "market_clock_mismatch",
    "public_market_response_failed",
    "public_history_incomplete",
}


class LiveStrategyStore(StateStore):
    def __init__(self, path):
        super().__init__(path)
        self.path.chmod(0o600)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS live_candles (timestamp INTEGER PRIMARY KEY, "
                "open REAL, high REAL, low REAL, close REAL, volume REAL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS live_samples "
                "(timestamp INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS live_observation_stats "
                "(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)"
            )

    def observe(self, sample, nav):
        """Save actual observations and drawdown atomically, never synthesize history."""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM live_observation_stats WHERE id=1").fetchone()
            stats = (
                json.loads(row[0])
                if row
                else {
                    "started_at": sample["timestamp"],
                    "latest_at": 0,
                    "samples": 0,
                    "peak_nav": nav,
                    "max_drawdown": 0,
                }
            )
            if sample["timestamp"] <= stats["latest_at"]:
                return
            stats["peak_nav"] = max(stats["peak_nav"], nav)
            stats["max_drawdown"] = max(stats["max_drawdown"], 1 - nav / stats["peak_nav"])
            stats["latest_at"] = sample["timestamp"]
            stats["samples"] += 1
            db.execute(
                "INSERT INTO live_samples VALUES (?, ?)",
                (sample["timestamp"], json.dumps(sample, allow_nan=False)),
            )
            db.execute(
                "INSERT INTO live_observation_stats VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (json.dumps(stats, allow_nan=False),),
            )

    def observations(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            row = db.execute("SELECT payload FROM live_observation_stats WHERE id=1").fetchone()
            stats = (
                json.loads(row[0])
                if row
                else {"started_at": None, "latest_at": None, "samples": 0, "max_drawdown": None}
            )
            stride = max(1, math.ceil(stats["samples"] / 1998))
            rows = db.execute(
                "SELECT payload FROM live_samples WHERE rowid % ? = 0 "
                "OR timestamp IN (?, ?) ORDER BY timestamp",
                (stride, stats["started_at"], stats["latest_at"]),
            ).fetchall()
        return {
            "history": {k: v for k, v in stats.items() if k != "peak_nav"},
            "equity_curve": [json.loads(row[0]) for row in rows],
        }

    def candles(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            data = pd.read_sql_query("SELECT * FROM live_candles ORDER BY timestamp", db)
        if data.empty:
            return None
        data["date"] = pd.to_datetime(data.pop("timestamp"), unit="ms", utc=True)
        return data

    def add_candles(self, data):
        with closing(sqlite3.connect(self.path)) as db, db:
            last = db.execute("SELECT MAX(timestamp) FROM live_candles").fetchone()[0] or 0
            stamps = pd.DatetimeIndex(data.date).as_unit("ms").asi8
            rows = [
                (
                    int(pd.Timestamp(r.date).timestamp() * 1000),
                    r.open,
                    r.high,
                    r.low,
                    r.close,
                    r.volume,
                )
                for r in data.loc[stamps > last].itertuples()
            ]
            db.executemany("INSERT INTO live_candles VALUES (?, ?, ?, ?, ?, ?)", rows)


class LiveStrategy:
    def __init__(self, execution, market, *, credential_guard=None):
        self.execution, self.market = execution, market
        self.store = LiveStrategyStore(execution.root / "strategy.sqlite")
        self.version, self.state = self.store.load("strategy")
        self.engine = Engine.restore(self.state["engine"]) if self.state else None
        self.data = None
        self.reference = None
        self.reference_day = None
        self.recovering = True
        self.credential_guard = credential_guard or (lambda: True)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.worker_lock = None

    @classmethod
    def create(cls, execution, market, snapshot, data, now):
        result = cls(execution, market)
        if result.state is not None:
            raise ExecutionError("live_strategy_already_configured")
        cfg = ResearchConfig.from_saved(snapshot)
        if (
            cfg.variant != "dynamic_period_only"
            or cfg.startup != "allocated_position"
            or not cfg.price_position_enabled
        ):
            raise ExecutionError("unsupported_live_candidate")
        cfg = cfg.model_copy(
            update={
                "initial_cash": float(money(execution.state["book"]["initial"])),
                "fee_rebate_ratio": 0,
                "slippage": 0,
                "trx_apr": 0,
                "usdgo_apr": 0,
                "usdt_apr": 0,
            }
        )
        result.engine = Engine(cfg)
        result.data = validate_minutes(data)
        result.store.add_candles(result.data)
        result.state = {
            "schema": "trx-c1-live-strategy-v1",
            "account_id": execution.account_id,
            "source_snapshot": snapshot,
            "source_sha256": digest(snapshot),
            "created_at": now,
            "desired": "paused",
            "phase": "prepared",
            "error": None,
            "engine": result.engine.state(),
            "activated_at": None,
            "warmup_until": None,
            "indicator_initialization": None,
            "startup_complete": False,
            "intent": None,
            "last_flow": None,
            "last_poll_at": None,
            "price_reference": None,
            "last_cash_switch": 0,
            "markets": {},
            "terms": {},
            "decisions": [],
            "events": [],
        }
        result.persist()
        return result

    def persist(self):
        self.state["engine"] = self.engine.state()
        self.state["events"] = self.state["events"][-500:]
        self.state["decisions"] = self.state["decisions"][-100:]
        self.state["evaluations"] = self.state.get("evaluations", [])[-200:]
        self.version = self.store.save("strategy", self.version, self.state)

    def event(self, kind, now, **values):
        self.state["events"].append({"kind": kind, "timestamp": now, **values})

    def sync_savings_observation(self):
        observation = getattr(self.execution.broker, "savings_observation", None)
        if isinstance(observation, dict):
            self.state["savings_observation"] = deepcopy(observation)

    def capture_observation(self, now):
        markets = self.state["markets"]
        if set(markets) != {"TRXUSDT", "USDGOUSDT"}:
            return
        times = [int(value["timestamp"]) for value in markets.values()]
        if any(not -30000 <= now - stamp <= 60000 for stamp in times):
            return
        execution = self.execution.status()
        book = Ledger.restore(execution["book"])
        price = money(markets["TRXUSDT"]["bid"])
        book.usdgo_mark = money(markets["USDGOUSDT"]["bid"])
        summary = book.summary(price)
        nav = float(book.equity(price) / book.fund_units) if book.fund_units else 0
        if not math.isfinite(nav) or nav <= 0:
            return
        self.store.observe({"timestamp": min(times), "price": float(price), **summary}, nav)

    def reference_at(self, timestamp):
        day = (timestamp + OFFSET) // DAY
        if day != self.reference_day:
            self.reference = build_price_reference(self.data, self.engine.config)
            self.reference_day = day
        reference = self.reference.value_at(timestamp)
        if reference is None:
            raise ExecutionError("complete_p75_history_required")
        self.engine.price_position_day, self.engine.price_position_reference = day, reference
        self.state["price_reference"] = {
            "day": day,
            "value": reference,
            "history_end_exclusive": day * DAY - OFFSET,
            "lookback_days": self.engine.config.price_position_lookback_days,
            "quantile": self.engine.config.price_position_quantile,
        }

    def initialize_indicators(self, now):
        """Seed indicators from confirmed history without replaying its decisions.

        The execution ledger/anchor is authoritative and is never reset here.
        A later confirmed bar must close strictly after this initialization to
        create a new signal. Existing execution intents still reconcile normally.
        """
        if self.data is None:
            self.data = self.store.candles()
        window = self.engine.config.volatility_window
        if self.data is None or len(self.data) < window + 1:
            raise ExecutionError("volatility_history_incomplete")
        closes = pd.DatetimeIndex(self.data.date).as_unit("ms").asi8 + 60000
        history = self.data.loc[closes <= now]
        if len(history) < window + 1:
            raise ExecutionError("volatility_history_incomplete")
        tail = validate_minutes(history.tail(window + 1))
        last = int(pd.Timestamp(tail.date.iloc[-1]).timestamp() * 1000) + 60000
        if not 0 <= now - last <= 120000:
            raise ExecutionError("finalized_market_data_stale")
        sigma = float(normalized_volatility(tail, window).iloc[-1])
        if not math.isfinite(sigma) or sigma < 0:
            raise ExecutionError("volatility_history_incomplete")
        self.reference_at(last)
        self.engine.last_close = last
        self.engine.last_price = float(tail.close.iloc[-1])
        self.engine.last_sigma = sigma
        self.engine.paused = True
        if not self.state["intent"]:
            self.engine.pending = None
            if self.engine.anchor:
                self.engine.schedule(now, self.engine.last_price, sigma)
            self.state.pop("reschedule_after_trade", None)
        self.state.update(
            warmup_until=None,
            indicator_initialization={
                "initialized_at": now,
                "history_close": last,
                "window_minutes": window,
                "rows_used": window + 1,
                "sigma": sigma,
                "signals_after": now,
            },
        )
        self.recovering = False
        self.event(
            "indicators_initialized_from_history",
            now,
            history_close=last,
            window_minutes=window,
            sigma=sigma,
        )

    def environment(self, now):
        markets = {
            symbol: self.execution.broker.market(symbol, now) for symbol in ("TRXUSDT", "USDGOUSDT")
        }
        terms = {}
        for rule in self.execution.state["products"]:
            if rule["redemption_minimum"] is None:
                raise ExecutionError("redemption_terms_unverified")
            value = self.execution.broker.product(rule["coin"], rule["product_id"])
            if value["status"] != "in_progress":
                raise ExecutionError("product_unavailable")
            terms[rule["coin"]] = value
        if set(terms) != {"TRX", "USDGO"}:
            raise ExecutionError("both_live_products_required")
        self.state.update(markets=markets, terms=terms)
        self.engine.config = self.engine.config.model_copy(
            update={
                "buy_fee": float(markets["TRXUSDT"]["fee"]),
                "sell_fee": float(markets["TRXUSDT"]["fee"]),
                "quantity_decimals": min(12, int(markets["TRXUSDT"]["quantity_decimals"])),
                "min_notional": max(
                    self.engine.config.min_notional, float(markets["TRXUSDT"]["minimum_notional"])
                ),
            }
        )
        return LiveFunding(self.execution.state, self.engine.config, markets, terms)

    def sync_book(self, funding):
        self.engine.book = funding.planning_book()
        self.engine.prepare_cash = lambda *_: float(funding.available_cash())
        # Anchor belongs to the actual TRX fill ledger, never FX or subscriptions.
        state = self.execution.state
        if state["anchor"] is not None:
            changed = self.engine.anchor_version != state["anchor_version"]
            self.engine.anchor = float(state["anchor"])
            self.engine.anchor_version = state["anchor_version"]
            if changed:
                self.engine.last_principal_fill_at = (
                    self.state["last_poll_at"] or self.state["created_at"]
                )

    def make_intent(self, steps, kind, now, expires_at):
        if kind == "trade" and self.engine.pending:
            self.engine.pending.submitted = True
        plan_id = self.engine.pending.plan_id if self.engine.pending else str(now)
        self.state["intent"] = {
            "id": "c1-" + digest([self.execution.account_id, kind, plan_id])[:32],
            "kind": kind,
            "created_at": now,
            "expires_at": expires_at,
            "steps": steps,
            "flow_id": None,
        }
        self.event("strategy_intent_saved", now, kind_name=kind)
        self.persist()  # Outbox precedes queue(), including a crash between the two.
        self.dispatch(now)

    def dispatch(self, now):
        intent = self.state["intent"]
        if intent is None:
            return
        state = self.execution.status()
        matches = [
            f for f in [*state["flows"], state["flow"]] if f and f.get("intent_id") == intent["id"]
        ]
        if matches:
            intent["flow_id"] = matches[0]["id"]
        elif now >= intent["expires_at"]:
            self.finish_intent({"status": "expired", "outputs": []}, now)
            return
        else:
            intent["flow_id"] = self.execution.queue(
                intent["steps"],
                now=now,
                expires_at=intent["expires_at"],
                label="C1 live · " + intent["kind"],
                strategy_snapshot=self.state["source_snapshot"],
                intent_id=intent["id"],
            )
        self.persist()

    def finish_intent(self, flow, now):
        intent = self.state["intent"]
        if flow["status"] in {"failed", "needs_review"}:
            code = (self.execution.state.get("operation") or {}).get("cancel_review_code")
            self.state.update(
                phase="blocked", error=code or "execution_requires_review", desired="paused"
            )
            return
        if intent["kind"] == "startup":
            if flow["status"] != "completed":
                self.state.update(
                    phase="blocked", error="startup_incomplete_requires_review", desired="paused"
                )
                return
            self.state["startup_complete"] = True
            self.state["startup_resume"] = None
        if any(s.get("symbol") == "USDGOUSDT" for s in intent["steps"]):
            self.state["last_cash_switch"] = now
        self.execution.reload()
        if self.execution.state["anchor"]:
            self.engine.anchor = float(self.execution.state["anchor"])
            self.engine.anchor_version = self.execution.state["anchor_version"]
        if intent["kind"] != "earn":
            self.engine.pending = None
            self.state["reschedule_after_trade"] = True
        self.state["last_flow"] = {
            "id": intent.get("flow_id"),
            "kind": intent["kind"],
            "status": flow["status"],
            "completed_at": now,
        }
        self.state["intent"] = None
        self.event("strategy_flow_finished", now, status=flow["status"])

    def resolve_execution(self, now):
        self.dispatch(now)
        if (
            self.state["intent"]
            and self.state["intent"]["flow_id"] is None
            and self.state["phase"] == "blocked"
        ):
            self.persist()
            return True
        state = self.execution.status()
        flow = state["flow"]
        if self.state["intent"]:
            if not flow or flow["id"] != self.state["intent"]["flow_id"]:
                raise ExecutionError("strategy_execution_binding_mismatch")
            if self.state["desired"] != "running":
                self.execution.request_stop(now)
            self.execution.tick(now)
            self.execution.reload()
            state, flow = self.execution.state, self.execution.state["flow"]
            if flow["status"] in TERMINAL:
                self.finish_intent(flow, now)
            elif (
                state["events"] and state["events"][-1]["kind"] == "read_or_reconciliation_blocked"
            ):
                code = state["events"][-1]["code"]
                if code in {
                    "signal_price_condition_changed",
                    "price_deviation_exceeded",
                    "order_below_minimum",
                }:
                    self.execution.request_stop(now)
                elif code != "redemption_cooldown":
                    raise ExecutionError(code)
            self.state["phase"] = (
                "executing" if self.state["desired"] == "running" else self.state["phase"]
            )
            self.persist()
            return True
        if flow and flow["status"] not in TERMINAL:
            raise ExecutionError("unowned_execution_flow")
        return False

    def market_minutes(self, now, funding):  # noqa: C901 - forward-only observation and scheduling
        if self.data is None:
            self.data = self.store.candles()
        self.data = self.market.update(self.data, now)
        self.store.add_candles(self.data)
        stamps = pd.DatetimeIndex(self.data.date).as_unit("ms").asi8 + 60000
        last = int(stamps[-1])
        if last > now or now - last > 120000:
            raise ExecutionError("finalized_market_data_stale")
        if (now + OFFSET) // DAY > (last + OFFSET) // DAY:
            self.state["phase"] = "confirming_daily_reference"
            return False
        self.reference_at(last)
        self.sync_book(funding)
        if (
            self.recovering
            or not self.state.get("indicator_initialization")
            or self.state["activated_at"] is None
        ):
            if self.state["activated_at"] is None:
                self.state["activated_at"] = now
            self.initialize_indicators(now)
            return True
        if (
            self.state.pop("reschedule_after_trade", False)
            and self.engine.anchor
            and math.isfinite(self.engine.last_sigma)
        ):
            self.engine.schedule(
                now, self.engine.last_price or self.engine.anchor, self.engine.last_sigma
            )
        indexes = [int(i) for i in (stamps > self.engine.last_close).nonzero()[0]]
        if not indexes:
            return True
        sigmas = normalized_volatility(
            self.data.iloc[max(0, indexes[0] - self.engine.config.volatility_window - 1) :],
            self.engine.config.volatility_window,
        )
        had_gap = self.recovering or now - int(stamps[indexes[0]]) > 120000
        for index in indexes:
            close, price = int(stamps[index]), float(self.data.close.iloc[index])
            historical = close <= self.state["indicator_initialization"]["signals_after"]
            self.engine.paused = had_gap or historical or not self.state["startup_complete"]
            if not self.engine.paused:
                self.reference_at(close)
            plan, evaluations = observed_close(
                self.engine,
                close,
                price,
                float(sigmas.loc[index]),
                now,
                "recovery_no_backfill"
                if had_gap or historical
                else "startup_not_complete"
                if not self.state["startup_complete"]
                else None,
            )
            known = {
                (r["timestamp"], r["side"], r["anchor_version"])
                for r in self.state.get("evaluations", [])
            }
            self.state.setdefault("evaluations", []).extend(
                r
                for r in evaluations
                if (r["timestamp"], r["side"], r["anchor_version"]) not in known
            )
            if plan is not None and not self.engine.paused and plan.created_at == close:
                self.state["decisions"].append(
                    {
                        "timestamp": close,
                        "side": plan.side,
                        "signal_price": price,
                        "anchor": plan.anchor,
                        "quantity": plan.quantity,
                        "budget": plan.budget,
                        "expires_at": plan.expires_at,
                        "buy_period": self.engine.buy_period,
                        "sell_period": self.engine.sell_period,
                    }
                )
        if had_gap:
            if not self.state["intent"]:
                self.engine.pending = None
            self.state["indicator_initialization"]["signals_after"] = now
            if self.engine.anchor and math.isfinite(self.engine.last_sigma):
                self.engine.schedule(now, self.engine.last_price, self.engine.last_sigma)
        self.recovering = False
        return True

    def tick(self, now):
        if not self.credential_guard():
            raise ExecutionError("credentials_changed_requires_preflight")
        self.state["last_poll_at"] = now
        busy = self.resolve_execution(now)
        if self.state["desired"] != "running":
            if self.state["phase"] != "blocked":
                self.state["phase"] = "paused"
            self.persist()
            return
        observed, quote = self.market.snapshot()
        if abs(observed - now) > 30000:
            raise ExecutionError("market_clock_mismatch")
        now = observed
        validate_quote(quote, now)
        if not self.execution.status()["operation"]:
            self.execution.import_interest(now)
        self.execution.reload()
        self.sync_savings_observation()
        funding = self.environment(now)
        if not self.market_minutes(now, funding):
            self.persist()
            return
        if busy:
            self.state["phase"] = "executing"
            self.capture_observation(now)
            self.persist()
            return
        self.decide(funding, now)
        self.state["error"] = None
        self.state["evidence_wait_started_at"] = None
        self.capture_observation(now)
        self.persist()

    def poll_error(self, exc, now):
        code = (
            str(exc)
            if isinstance(exc, ExecutionError)
            or (isinstance(exc, ValueError) and str(exc) in PUBLIC_RETRY_CODES)
            else "market_network_unavailable"
            if isinstance(exc, ccxt.NetworkError)
            else "unverified_strategy_evidence"
        )
        retryable = code.startswith("exchange_read_") or code in PUBLIC_RETRY_CODES | {
            "exchange_response_unavailable",
            "quote_stale",
            "quote_expired_during_preflight",
            "finalized_market_data_stale",
            "market_network_unavailable",
            "request_rate_wait",
        }
        if code in EVIDENCE_WAIT_CODES:
            since = self.state.get("evidence_wait_started_at")
            if since is None:
                since = now
                self.state["evidence_wait_started_at"] = now
            retryable = now - since < 5 * 60000
            if not retryable:
                code = "account_evidence_wait_expired"
        self.state.update(phase="recovering" if retryable else "blocked", error=code)
        self.recovering = True
        if not retryable:
            self.state["desired"] = "paused"
            self.execution.request_stop(now)
        self.event("strategy_poll_blocked", now, code=code)
        self.persist()

    def decide(self, funding, now):
        if not self.state.get("indicator_initialization"):
            self.state["phase"] = "warming_up"
        elif not self.state["startup_complete"]:
            self.make_intent(
                self.state.get("startup_resume") or funding.startup(),
                "startup",
                now,
                now + 10 * 60000,
            )
            self.state["phase"] = "executing"
        elif self.engine.pending:
            if now >= self.engine.pending.expires_at:
                self.engine.finish(now)
            else:
                steps = funding.for_plan(self.engine.pending)
                self.make_intent(steps, "trade", now, self.engine.pending.expires_at)
                self.state["phase"] = "executing"
        else:
            idle = funding.idle(now, self.state["last_cash_switch"])
            if idle:
                self.make_intent(idle, "earn", now, now + 10 * 60000)
                self.state["phase"] = "executing"
            else:
                self.state["phase"] = "observing"

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ExecutionError("strategy_already_running")
            if not self.execution.broker.writes_enabled:
                raise ExecutionError("live_transport_not_enabled")
            self.worker_lock = (self.execution.root / "strategy-worker.lock").open("a+")
            (self.execution.root / "strategy-worker.lock").chmod(0o600)
            try:
                fcntl.flock(self.worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                version, state = self.store.load("strategy")
                if version != self.version:
                    raise ExecutionError("strategy_state_changed")
                if state["phase"] == "blocked":
                    raise ExecutionError("strategy_requires_operator_review")
            except Exception:
                self.worker_lock.close()
                self.worker_lock = None
                raise ExecutionError("strategy_start_unavailable") from None
            self.state.update(desired="running", phase="connecting", error=None)
            self.recovering = True
            self.stop_event.clear()
            self.persist()
            self.thread = threading.Thread(target=self.run, daemon=True, name="trx-c1-live")
            self.thread.start()

    def pause(self):
        # The worker remains responsible for accepted orders and delayed arrivals.
        with self.lock:
            self.state.update(desired="paused", phase="pausing")
            self.persist()

    def run(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    try:
                        self.tick(int(time.time() * 1000))
                    except Exception as exc:
                        self.poll_error(exc, int(time.time() * 1000))
                    if self.state["phase"] == "blocked":
                        break
                    if (
                        self.state["desired"] != "running"
                        and not self.execution.status()["operation"]
                    ):
                        if self.state["phase"] != "blocked":
                            self.state["phase"] = "paused"
                            self.persist()
                        break
                self.stop_event.wait(15)
        finally:
            if self.worker_lock:
                fcntl.flock(self.worker_lock, fcntl.LOCK_UN)
                self.worker_lock.close()
                self.worker_lock = None
            if hasattr(self.execution.broker, "transport"):
                self.execution.broker.transport.session.close()
            self.market.close()

    def status(self):
        _, state = self.store.load("strategy")
        result = deepcopy(state)
        running = bool(self.thread and self.thread.is_alive())
        result["worker_running"] = running
        if not running and result["desired"] == "running":
            result["phase"] = "restart_requires_preflight"
        result["execution"] = self.execution.status()
        return result
