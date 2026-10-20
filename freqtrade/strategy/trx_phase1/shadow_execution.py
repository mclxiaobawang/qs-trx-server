"""Durable, virtual order transport and an idempotent execution outbox.

This module has no exchange client, credentials or HTTP transport. Broker orders
commit independently of the local portfolio, so acceptance/acknowledgement and
restart faults can be tested without ever sending a real exchange order.
"""

import hashlib
import json
import sqlite3
from copy import deepcopy
from decimal import ROUND_DOWN

from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.shadow_market import validate_quote


TERMINAL = {"filled", "cancelled", "rejected"}


class SimulatedBroker:
    def __init__(self, path, fault="none"):
        self.path, self.fault = path, fault
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS shadow_broker "
                "(client_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def query(self, client_id):
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT payload FROM shadow_broker WHERE client_id=?", (client_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, order):
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO shadow_broker VALUES (?, ?) ON CONFLICT(client_id) "
                "DO UPDATE SET payload=excluded.payload",
                (order["client_id"], json.dumps(order, allow_nan=False)),
            )

    def submit(self, request):
        existing = self.query(request["client_id"])
        if existing:
            if existing["request"] != request:
                raise ValueError("client_id_conflict")
            return existing
        if self.fault == "timeout_before_accept":
            raise TimeoutError("simulated_timeout_before_accept")
        fills = deepcopy(request["fills"])
        status = "filled"
        if self.fault == "reject":
            fills, status = [], "rejected"
        elif self.fault in ("partial", "cancel_fill_race"):
            fills = [dict(fills[0])]
            fills[0]["quantity"] /= 2
            fills[0]["fee_quote"] /= 2
            fills[0]["fee_base"] /= 2
            status = "open"
        order = {
            "client_id": request["client_id"],
            "request": request,
            "status": status,
            "fills": fills,
        }
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM shadow_broker WHERE client_id=?", (request["client_id"],)
            ).fetchone()
            if row:
                existing = json.loads(row[0])
                if existing["request"] != request:
                    raise ValueError("client_id_conflict")
                return existing
            db.execute(
                "INSERT INTO shadow_broker VALUES (?, ?)",
                (request["client_id"], json.dumps(order, allow_nan=False)),
            )
        if self.fault == "timeout_after_accept":
            raise TimeoutError("simulated_timeout_after_accept")
        return order

    def cancel(self, client_id, now):
        order = self.query(client_id)
        if order is None:
            return None
        if order["status"] not in TERMINAL:
            if self.fault == "cancel_fill_race":
                rest = deepcopy(order["fills"][0])
                rest["event_id"] += ":rest"
                order["fills"] += [rest] + deepcopy(order["request"]["fills"][1:])
                order["status"] = "filled"
            else:
                order["status"] = "cancelled"
            order["resolved_at"] = now
            self._put(order)
        return order


class ShadowExecution:
    def __init__(self, broker, namespace, state=None):
        self.broker, self.namespace = broker, namespace
        self.outbox = deepcopy((state or {}).get("outbox"))
        self.receipts = deepcopy((state or {}).get("receipts", {}))
        self.orders = deepcopy((state or {}).get("orders", []))

    def state(self):
        return {"outbox": self.outbox, "receipts": self.receipts, "orders": self.orders}

    def prepare(self, engine, quote, now, market_rules):
        validate_quote(quote, now)
        plan = engine.pending
        if self.outbox:
            raise ValueError("order_resolution_required")
        if not plan or not engine.executable(
            now, quote["asks"][0][0] if plan.side == "buy" else quote["bids"][0][0]
        ):
            raise ValueError("execution_preflight_failed")
        if now - quote["timestamp"] > 10000 or quote["timestamp"] > now + 2000:
            raise ValueError("stale_execution_quote")
        cfg = engine.config
        decimals = min(cfg.quantity_decimals, market_rules["quantity_decimals"])
        quantum = money(10) ** -decimals
        quantity = money(
            engine.execution_quantity(
                quote["asks"][0][0] if plan.side == "buy" else quote["bids"][0][0]
            )
        )
        budget = money(max(0, min(float(engine.book.free_cash), plan.budget - plan.filled_cash)))
        rate = cfg.buy_fee if plan.side == "buy" else cfg.sell_fee
        client_id = (
            "shadow-" + hashlib.sha256(f"{self.namespace}:{plan.plan_id}".encode()).hexdigest()[:32]
        )
        fills, consumed = [], money(0)
        levels = quote["asks"] if plan.side == "buy" else quote["bids"]
        for level, (raw_price, available) in enumerate(levels):
            price = money(raw_price) * money(
                1 + cfg.slippage if plan.side == "buy" else 1 - cfg.slippage
            )
            remaining = quantity - consumed
            if remaining <= 0:
                break
            amount = min(remaining, money(available))
            if plan.side == "buy":
                amount = min(amount, budget / (price * (1 + money(rate))))
            amount = amount.quantize(quantum, rounding=ROUND_DOWN)
            if amount <= 0:
                continue
            if not engine.executable(now, float(price)):
                break
            notional = amount * price
            fee = notional * money(rate)
            fills.append(
                {
                    "event_id": f"{client_id}:{level}",
                    "order_id": client_id,
                    "timestamp": now,
                    "quantity": float(amount),
                    "price": float(price),
                    "fee_quote": float(fee),
                    "fee_base": 0.0,
                }
            )
            consumed += amount
            budget -= notional + fee
        notional = sum((money(row["quantity"]) * money(row["price"]) for row in fills), money(0))
        if notional < money(max(cfg.min_notional, market_rules["min_notional"])) or (
            consumed < money(max(cfg.min_quantity, market_rules["min_quantity"]))
        ):
            raise ValueError("insufficient_executable_depth")
        self.outbox = {
            "client_id": client_id,
            "plan_id": plan.plan_id,
            "side": plan.side,
            "created_at": now,
            "expires_at": plan.expires_at,
            "status": "prepared",
            "quantity": float(consumed),
            "fills": fills,
            "quote_timestamp": quote["timestamp"],
            "market_rules": market_rules,
        }
        plan.submitted = True

    def ingest(self, engine, order, now):
        if not self.outbox or order["client_id"] != self.outbox["client_id"]:
            raise ValueError("unexpected_order_receipt")
        if order["status"] not in TERMINAL | {"open"}:
            raise ValueError("unknown_order_status")
        candidate = Engine.restore(engine.state())
        candidate.price_position_day = engine.price_position_day
        candidate.price_position_reference = engine.price_position_reference
        receipts = dict(self.receipts)
        total = sum((money(row["quantity"]) for row in order["fills"]), money(0))
        if total > money(self.outbox["quantity"]) + money("1e-10"):
            raise ValueError("fill_exceeds_submitted_quantity")
        for row in order["fills"]:
            digest = hashlib.sha256(
                json.dumps(row, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            event_id = row["event_id"]
            if event_id in receipts:
                if receipts[event_id] != digest:
                    raise ValueError("conflicting_fill_receipt")
                continue
            if not event_id.startswith(self.outbox["client_id"] + ":") or (
                row["order_id"] != self.outbox["client_id"]
            ):
                raise ValueError("unexpected_fill_receipt")
            if (
                money(row["quantity"]) <= 0
                or money(row["price"]) <= 0
                or any(money(row[key]) < 0 for key in ("fee_quote", "fee_base"))
                or not self.outbox["created_at"] <= row["timestamp"] <= now
            ):
                raise ValueError("invalid_fill_receipt")
            # Broker receipts are facts, not another signal: apply fees and even
            # unexpected losses truthfully. The candidate isolates invalid batches.
            candidate.fill(
                event_id,
                row["order_id"],
                row["timestamp"],
                row["quantity"],
                row["price"],
                row["fee_quote"],
                row["fee_base"],
                final=False,
            )
            receipts[event_id] = digest
        if order["status"] in TERMINAL:
            candidate.finish(
                order.get(
                    "resolved_at", max((row["timestamp"] for row in order["fills"]), default=now)
                )
            )
            self.orders.append(
                {
                    "client_id": order["client_id"],
                    "side": self.outbox["side"],
                    "status": order["status"],
                    "created_at": self.outbox["created_at"],
                    "resolved_at": now,
                    "fills": deepcopy(order["fills"]),
                    "quote_timestamp": self.outbox["quote_timestamp"],
                }
            )
            self.outbox = None
        else:
            self.outbox["status"] = "open"
        self.receipts = receipts
        return candidate

    def resolve(self, engine, now, persist, cancel=False):
        """Every prepared or ambiguous order is queried before any submission."""
        if not self.outbox:
            return engine
        order = self.broker.query(self.outbox["client_id"])
        if order is None:
            if (
                self.outbox["status"] == "prepared"
                and not cancel
                and now < self.outbox["expires_at"]
            ):
                self.outbox["status"] = "submitting"
                persist(engine)  # Durable intent precedes the independent broker commit.
                try:
                    order = self.broker.submit(deepcopy(self.outbox))
                except TimeoutError:
                    self.outbox["status"] = "unknown"
                    persist(engine)
                    return engine
            else:
                # Confirmed missing after an ambiguous attempt: abandon this
                # signal; never blindly resend it, including after a restart.
                engine.finish(now)
                self.orders.append(
                    {
                        "client_id": self.outbox["client_id"],
                        "side": self.outbox["side"],
                        "status": "not_accepted",
                        "created_at": self.outbox["created_at"],
                        "resolved_at": now,
                        "fills": [],
                    }
                )
                self.outbox = None
                persist(engine)
                return engine
        if (
            order
            and (cancel or now >= self.outbox["expires_at"])
            and order["status"] not in TERMINAL
        ):
            self.broker.cancel(self.outbox["client_id"], now)
            order = self.broker.query(self.outbox["client_id"])
        if order:
            engine = self.ingest(engine, order, now)
            persist(engine)
        return engine


def check_execution_faults(directory, config):
    """Bounded, deterministic transport checks; never touches an observation account."""
    import tempfile
    import time
    from pathlib import Path

    report = {"observed_at": int(time.time() * 1000), "simulated": True, "checks": []}
    now = 1700000040000
    quote = {"timestamp": now, "asks": [[0.3, 100000]], "bids": [[0.2999, 100000]]}
    rules = {"quantity_decimals": 4, "min_quantity": 3, "min_notional": 1}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        for fault in (
            "none",
            "timeout_before_accept",
            "timeout_after_accept",
            "reject",
            "partial",
            "cancel_fill_race",
            "restart_before_ack",
        ):
            engine = Engine(config)
            engine.last_price = 0.3
            engine.make_plan("buy", now, 0.3, 0, now + 60000, startup=True, reference=0.3)
            broker = SimulatedBroker(
                Path(temporary) / f"{fault}.sqlite",
                fault if fault != "restart_before_ack" else "none",
            )
            driver = ShadowExecution(broker, fault)
            saved = {}
            context = {"saved": saved, "driver": driver}

            def persist(candidate, context=context):
                context["saved"].update(
                    engine=deepcopy(candidate.state()),
                    execution=deepcopy(context["driver"].state()),
                )

            driver.prepare(engine, quote, now, rules)
            submitted = driver.outbox["quantity"]
            persist(engine)
            if fault == "restart_before_ack":
                driver.outbox["status"] = "submitting"
                persist(engine)
                broker.submit(deepcopy(driver.outbox))
            else:
                engine = driver.resolve(engine, now, persist)
            # Every check reloads the committed portfolio/outbox, then queries
            # broker truth. Repeated receipts must not create another trade.
            engine = Engine.restore(saved["engine"])
            driver = ShadowExecution(broker, fault, saved["execution"])
            context["driver"] = driver
            engine = driver.resolve(engine, now + 1, persist)
            if fault in ("partial", "cancel_fill_race"):
                before = float(engine.book.principal)
                engine = driver.resolve(engine, now + 2, persist)
                if float(engine.book.principal) != before:
                    raise ValueError("duplicate_receipt_changed_balance")
                engine = driver.resolve(engine, now + 3, persist, cancel=True)
            expected = (
                0
                if fault in ("timeout_before_accept", "reject")
                else (submitted / 2 if fault == "partial" else submitted)
            )
            actual = float(engine.book.principal)
            passed = abs(actual - expected) < 1e-8 and driver.outbox is None
            report["checks"].append(
                {
                    "case": fault,
                    "passed": passed,
                    "expected_trx": expected,
                    "actual_trx": actual,
                    "fills": len(engine.book.orders),
                }
            )
    report["passed"] = all(row["passed"] for row in report["checks"])
    target = directory / "fault-check.json"
    temporary_path = target.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary_path.replace(target)
    return report
