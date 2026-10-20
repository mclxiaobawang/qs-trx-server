"""All provider writes below affect a separate synthetic exchange SQLite database."""

import hashlib
import json
import sqlite3
from copy import deepcopy
from decimal import ROUND_DOWN, Decimal

import pytest

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import (
    DAY,
    ExecutionError,
    WriteNotSent,
    WriteRejected,
    WriteUnknown,
)
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution


NOW = 1792166400000
ACCOUNT = hashlib.sha256(b"synthetic-account").hexdigest()
RULES = [
    {
        "coin": coin,
        "product_id": coin + "-flex",
        "redemption_minimum": "0.1",
        "redemption_decimals": 8,
        "redemption_evidence": "Synthetic product terms",
    }
    for coin in ("USDGO", "TRX")
]


class ExchangeFixture:
    """Provider acceptance commits independently of the local execution journal."""

    writes_enabled = True

    def __init__(self, path):
        self.path = path
        self.fault = None
        self.lagged_balances = None
        self.hide_bills = False
        self.hide_records = False
        self.read_failure = False
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS fixture (id INTEGER PRIMARY KEY, state TEXT)")
            if not db.execute("SELECT 1 FROM fixture").fetchone():
                state = {
                    "wallet": {"USDT": "1000.4", "USDGO": "0", "TRX": "0"},
                    "orders": {},
                    "holdings": {},
                    "savings": {},
                    "records": [],
                    "bills": [],
                    "writes": [],
                }
                db.execute("INSERT INTO fixture VALUES (1, ?)", (json.dumps(state),))

    def state(self):
        with sqlite3.connect(self.path) as db:
            return json.loads(db.execute("SELECT state FROM fixture WHERE id=1").fetchone()[0])

    def put(self, state):
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE fixture SET state=? WHERE id=1", (json.dumps(state),))

    def account_id(self):
        return ACCOUNT

    def balances(self):
        if self.read_failure:
            raise ExecutionError("exchange_response_unavailable")
        wallet = self.lagged_balances or self.state()["wallet"]
        return {
            c: {"available": q, "frozen": "0", "locked": "0", "restricted": "0"}
            for c, q in wallet.items()
        }

    def market(self, symbol, now):
        price = self.state().get("trx_price", "0.3") if symbol == "TRXUSDT" else "1.001"
        return {
            "bid": price,
            "ask": price,
            "fee": "0.001" if symbol == "TRXUSDT" else "0",
            "quantity_decimals": 6 if symbol == "TRXUSDT" else 4,
            "price_decimals": 4,
            "minimum_notional": "1",
            "bids": [[price, "1000000"]],
            "asks": [[price, "1000000"]],
            "timestamp": now,
        }

    def product(self, coin, product_id):
        assert product_id == coin + "-flex"
        return {
            "status": "in_progress",
            "subscribePrecision": "8",
            "singleMinAmount": "0.1",
            "singleMaxAmount": "1000000",
            "remainingAmount": "1000000",
        }

    def holdings(self, start, end):
        return deepcopy(self.state()["holdings"])

    @staticmethod
    def history_window(start, end):
        if not 0 <= end - start < 90 * DAY:
            raise ExecutionError("history_window_exceeded")

    def savings_records(self, coin, start, end, *, order_type=None):
        return (
            []
            if self.hide_records
            else [
                r
                for r in self.state()["records"]
                if r["coinName"] == coin
                and start <= int(r["ts"]) <= end
                and (order_type is None or r["orderType"] == order_type)
            ]
        )

    def bills(self, coin, start, end):
        return (
            []
            if self.hide_bills
            else [
                b
                for b in self.state()["bills"]
                if b["coin"] == coin and start <= int(b["cTime"]) <= end
            ]
        )

    def order(self, client_id):
        return deepcopy(self.state()["orders"].get(client_id))

    def fills(self, order_id, symbol, start, end):
        orders = [r for r in self.state()["orders"].values() if r["orderId"] == order_id]
        return deepcopy(orders[0]["fills"]) if orders else []

    def result(self, kind, order_id):
        return (
            {"result": "success"} if order_id in self.state()["savings"] else {"result": "pending"}
        )

    @staticmethod
    def apply_spot(state, order, quote_budget, stamp):
        base = "TRX" if order["symbol"] == "TRXUSDT" else "USDGO"
        price = money(state.get("trx_price", "0.3") if base == "TRX" else "1.001")
        side = order["side"]
        q = quote_budget / price if side == "buy" else quote_budget
        q = q.quantize(Decimal("0.000001" if base == "TRX" else "0.0001"), rounding=ROUND_DOWN)
        quote = q * price
        fee = (
            q * money("0.001")
            if base == "TRX" and side == "buy"
            else quote * money("0.001")
            if base == "TRX"
            else money(0)
        )
        fee_coin = base if side == "buy" else "USDT"
        fill = {
            "id": order["orderId"] + "-" + str(len(order["fills"])),
            "order_id": order["orderId"],
            "side": side,
            "quantity": str(q),
            "quote": str(quote),
            "timestamp": stamp,
            "fees": [{"coin": fee_coin, "cost": str(fee)}],
        }
        order["fills"].append(fill)
        order["baseVolume"] = str(money(order["baseVolume"]) + q)
        order["quoteVolume"] = str(money(order["quoteVolume"]) + quote)
        sign = 1 if side == "buy" else -1
        state["wallet"][base] = str(
            money(state["wallet"][base]) + sign * q - (fee if fee_coin == base else 0)
        )
        state["wallet"]["USDT"] = str(
            money(state["wallet"]["USDT"]) - sign * quote - (fee if fee_coin == "USDT" else 0)
        )

    def submit(self, op):
        if not self.writes_enabled:
            raise AssertionError("fixture_write_disabled")
        if self.fault == "reject":
            raise WriteRejected("exchange_write_40014")
        if self.fault == "timeout_before_accept":
            raise WriteUnknown("exchange_response_unavailable")
        state = self.state()
        state["writes"].append({"id": op["id"], "kind": op["kind"]})
        oid = "exchange-" + str(len(state["writes"]))
        request = op["request"]
        if op["kind"] == "trade":
            order = {
                "clientOid": op["id"],
                "orderId": oid,
                "symbol": request["symbol"],
                "side": request["side"],
                "orderType": "market",
                "baseVolume": "0",
                "quoteVolume": "0",
                "status": "filled",
                "fills": [],
                "requested": request["size"],
            }
            amount = money(request["size"])
            if self.fault in {"partial", "cancel_fill_race"}:
                amount /= 2
                order["status"] = "partially_filled"
            self.apply_spot(state, order, amount, op["created_at"])
            state["orders"][op["id"]] = order
        else:
            coin = request["productId"].split("-")[0]
            amount = money(request["amount"])
            sign = -1 if op["kind"] == "subscribe" else 1
            if op["kind"] == "subscribe":
                asset_id = "holding-" + oid
                state["holdings"][asset_id] = {
                    "order_id": asset_id,
                    "product_id": request["productId"],
                    "coin": coin,
                    "amount": str(amount),
                }
            else:
                asset_id = request["orderId"]
                state["holdings"][asset_id]["amount"] = str(
                    money(state["holdings"][asset_id]["amount"]) - amount
                )
            state["wallet"][coin] = str(money(state["wallet"][coin]) + sign * amount)
            state["savings"][oid] = {"kind": op["kind"], "holding_id": asset_id}
            state["records"].append(
                {
                    "orderId": oid,
                    "coinName": coin,
                    "settleCoinName": coin,
                    "productType": "flexible",
                    "orderType": op["kind"],
                    "amount": str(amount),
                    "ts": str(op["created_at"]),
                }
            )
            state["bills"].append(
                {
                    "billId": "bill-" + oid,
                    "bizOrderId": oid,
                    "coin": coin,
                    "groupType": "financial",
                    "size": str(sign * amount),
                    "fees": "0",
                    "cTime": str(op["created_at"]),
                }
            )
        self.put(state)
        if self.fault == "timeout_after_accept":
            raise WriteUnknown("exchange_response_unavailable")
        if self.fault == "crash_after_accept":
            raise KeyboardInterrupt
        return {"orderId": oid, "clientOid": op["id"] if op["kind"] == "trade" else None}

    def cancel(self, client_id, symbol):
        if self.fault == "cancel_not_sent":
            raise WriteNotSent("request_rate_wait")
        if self.fault == "cancel_timeout_before_accept":
            raise WriteUnknown("exchange_response_unavailable")
        if self.fault == "cancel_rejected":
            raise WriteRejected("exchange_write_40014")
        state = self.state()
        order = state["orders"][client_id]
        if self.fault == "cancel_fill_race":
            self.apply_spot(state, order, money(order["requested"]) / 2, NOW + 20000)
            order["status"] = "filled"
        else:
            order["status"] = "cancelled"
        self.put(state)
        if self.fault == "cancel_timeout_after_accept":
            raise WriteUnknown("exchange_response_unavailable")
        if self.fault == "cancel_crash_after_accept":
            raise KeyboardInterrupt
        return {"orderId": order["orderId"]}


@pytest.fixture
def runtime(tmp_path):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    return LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)


def trade(symbol="USDGOUSDT", side="buy", amount="100", minimum="1"):
    return {
        "kind": "trade",
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "reference_price": "0.3" if symbol == "TRXUSDT" else "1.001",
        "minimum_receive": minimum,
    }


def subscribe(coin, amount="previous"):
    return {"kind": "subscribe", "coin": coin, "product_id": coin + "-flex", "amount": amount}


def run_flow(runtime, steps, now=NOW, ttl=600000):
    runtime.queue(steps, now=now, expires_at=now + ttl, label="Synthetic controlled workflow")
    for i in range(30):
        runtime.tick(now + i * 100)
        state = runtime.status()
        if state["flow"]["status"] in {"completed", "failed", "expired", "needs_review"}:
            return state
    return runtime.status()


def test_purchase_subscribe_redeem_fx_buy_trx_subscribe_complete_chain(runtime, tmp_path):
    first = run_flow(runtime, [trade(), subscribe("USDGO")])
    assert first["flow"]["status"] == "completed"
    assert money(first["book"]["locked"]["USDGO"]) > 99
    holding = first["flow"]["outputs"][-1]["holding_id"]
    quantity = first["flow"]["outputs"][-1]["amount"]
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    recovered = LiveExecution(tmp_path, broker, ACCOUNT)
    steps = [
        {
            "kind": "redeem",
            "coin": "USDGO",
            "product_id": "USDGO-flex",
            "holding_id": holding,
            "amount": quantity,
        },
        trade(side="sell", amount="previous"),
        trade("TRXUSDT", amount="previous"),
        subscribe("TRX"),
    ]
    final = run_flow(recovered, steps, NOW + 70000)
    assert final["flow"]["status"] == "completed"
    book = Ledger.restore(final["book"])
    assert book.principal > 0
    assert sum(lot.locked for lot in book.lots) > 0
    assert book.interest_trx == 0 and book.fee_rebates == 0
    assert len(broker.state()["writes"]) == 6
    assert final["anchor_version"] == 1
    recovered.check_wallet(broker.balances(), book)


@pytest.mark.parametrize("fault", ["timeout_after_accept", "crash_after_accept"])
def test_spot_acceptance_before_disconnect_is_not_submitted_twice(runtime, tmp_path, fault):
    runtime.queue([trade()], now=NOW, expires_at=NOW + 60000, label="Recovery")
    runtime.tick(NOW)
    runtime.broker.fault = fault
    if fault == "crash_after_accept":
        with pytest.raises(KeyboardInterrupt):
            runtime.tick(NOW + 100)
    else:
        runtime.tick(NOW + 100)
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    restored = LiveExecution(tmp_path, broker, ACCOUNT)
    restored.tick(NOW + 200)
    assert restored.status()["flow"]["status"] == "completed"
    assert len(broker.state()["writes"]) == 1
    assert len(restored.status()["book"]["events"]) == 1


def test_spot_timeout_before_accept_is_not_blindly_retried(runtime):
    runtime.queue([trade()], now=NOW, expires_at=NOW + 1000, label="Uncertain")
    runtime.tick(NOW)
    runtime.broker.fault = "timeout_before_accept"
    runtime.tick(NOW + 100)
    runtime.broker.fault = None
    runtime.tick(NOW + 500)
    runtime.tick(NOW + 2000)
    assert runtime.status()["flow"]["status"] == "needs_review"
    assert runtime.broker.state()["writes"] == []
    assert money(runtime.status()["book"]["cash"]) == money("1000.4")


def test_unknown_earn_acceptance_requires_operator_identity_then_evidence(runtime, tmp_path):
    assert run_flow(runtime, [trade()])["flow"]["status"] == "completed"
    runtime.queue(
        [subscribe("USDGO", "10")], now=NOW + 10000, expires_at=NOW + 60000, label="Earn uncertain"
    )
    runtime.tick(NOW + 10000)
    runtime.broker.fault = "timeout_after_accept"
    runtime.tick(NOW + 10100)
    restored = LiveExecution(tmp_path, ExchangeFixture(tmp_path / "exchange.sqlite"), ACCOUNT)
    restored.tick(NOW + 10200)
    state = restored.status()
    assert state["flow"]["status"] == "needs_review"
    assert money(state["book"]["locked"]["USDGO"]) == 0
    assert len(restored.broker.state()["writes"]) == 2
    oid = list(restored.broker.state()["savings"])[-1]
    restored.attach_exchange_order(state["operation"]["id"], oid, now=NOW + 10300)
    restored.tick(NOW + 10400)
    assert restored.status()["flow"]["status"] == "completed"
    assert money(restored.status()["book"]["locked"]["USDGO"]) == 10
    assert len(restored.broker.state()["writes"]) == 2


def test_redemption_success_is_not_spendable_until_bill_and_wallet_arrive(runtime):
    first = run_flow(runtime, [trade(), subscribe("USDGO")])
    holding = first["flow"]["outputs"][-1]["holding_id"]
    runtime.queue(
        [
            {
                "kind": "redeem",
                "coin": "USDGO",
                "product_id": "USDGO-flex",
                "holding_id": holding,
                "amount": "10",
            },
            trade(side="sell", amount="previous"),
        ],
        now=NOW + 70000,
        expires_at=NOW + 170000,
        label="Wait for arrival",
    )
    runtime.tick(NOW + 70000)
    before_wallet = runtime.broker.state()["wallet"]
    runtime.tick(NOW + 70100)
    runtime.broker.hide_bills = True
    runtime.tick(NOW + 70200)
    assert runtime.status()["flow"]["index"] == 0
    runtime.broker.hide_bills = False
    runtime.broker.lagged_balances = before_wallet
    runtime.tick(NOW + 70300)
    assert runtime.status()["flow"]["index"] == 0
    runtime.broker.lagged_balances = None
    runtime.tick(NOW + 70400)
    assert runtime.status()["flow"]["index"] == 1
    assert len(runtime.broker.state()["writes"]) == 3


@pytest.mark.parametrize("history", ["different_ids", "missing"])
def test_earn_history_is_auxiliary_and_principal_is_only_relocated_once(runtime, tmp_path, history):
    initial = run_flow(runtime, [trade()])
    before = initial["book"]

    def change_history():
        state = runtime.broker.state()
        if history == "missing":
            state["records"] = []
        else:
            for row in state["records"]:
                row["orderId"] = "history-" + row["orderId"]
        runtime.broker.put(state)

    runtime.queue(
        [subscribe("USDGO", "10")],
        now=NOW + 10000,
        expires_at=NOW + 60000,
        label="Principal location",
    )
    runtime.tick(NOW + 10000)
    runtime.tick(NOW + 10100)
    change_history()
    runtime.tick(NOW + 10200)
    first = runtime.status()
    assert first["flow"]["status"] == "completed"
    assert money(first["book"]["locked"]["USDGO"]) == 10
    holding = first["flow"]["outputs"][-1]["holding_id"]
    runtime.queue(
        [
            {
                "kind": "redeem",
                "coin": "USDGO",
                "product_id": "USDGO-flex",
                "holding_id": holding,
                "amount": "10",
            }
        ],
        now=NOW + 70000,
        expires_at=NOW + 170000,
        label="Return principal",
    )
    runtime.tick(NOW + 70000)
    runtime.tick(NOW + 70100)
    change_history()
    runtime.tick(NOW + 70200)
    settled = runtime.status()
    assert settled["flow"]["status"] == "completed"
    # Decimal zero may retain the product's scale (0E-8); compare its value.
    assert {
        **settled["book"],
        "locked": {k: money(v) for k, v in settled["book"]["locked"].items()},
    } == {**before, "locked": {k: money(v) for k, v in before["locked"].items()}}
    assert len(runtime.broker.state()["writes"]) == 3
    restored = LiveExecution(tmp_path, ExchangeFixture(tmp_path / "exchange.sqlite"), ACCOUNT)
    restored.broker.writes_enabled = False
    restored.tick(NOW + 70300)
    restored.tick(NOW + 70400)
    final = restored.status()
    assert final["book"] == settled["book"]
    assert len(final["flow"]["outputs"]) == 1
    assert len([e for e in final["events"] if e["kind"] == "operation_settled"]) == 3
    assert len(restored.broker.state()["writes"]) == 3


def test_redemption_waits_for_exact_holding_snapshot_even_if_order_bill_and_wallet_arrive(runtime):
    first = run_flow(runtime, [trade(), subscribe("USDGO")])
    holding = first["flow"]["outputs"][-1]["holding_id"]
    before_holdings = runtime.broker.state()["holdings"]
    runtime.queue(
        [
            {
                "kind": "redeem",
                "coin": "USDGO",
                "product_id": "USDGO-flex",
                "holding_id": holding,
                "amount": "10",
            }
        ],
        now=NOW + 70000,
        expires_at=NOW + 170000,
        label="Holding snapshot delay",
    )
    runtime.tick(NOW + 70000)
    runtime.tick(NOW + 70100)
    state = runtime.broker.state()
    arrived_holdings = state["holdings"]
    state["holdings"] = before_holdings
    runtime.broker.put(state)
    runtime.tick(NOW + 70200)
    assert runtime.status()["flow"]["index"] == 0
    state["holdings"] = arrived_holdings
    runtime.broker.put(state)
    runtime.tick(NOW + 70300)
    assert runtime.status()["flow"]["status"] == "completed"


@pytest.mark.parametrize("commit_before_crash", [False, True])
def test_redemption_restart_before_or_after_journal_commit_never_credits_twice(
    runtime, tmp_path, mocker, commit_before_crash
):
    first = run_flow(runtime, [trade(), subscribe("USDGO")])
    holding = first["flow"]["outputs"][-1]["holding_id"]
    runtime.queue(
        [
            {
                "kind": "redeem",
                "coin": "USDGO",
                "product_id": "USDGO-flex",
                "holding_id": holding,
                "amount": "10",
            }
        ],
        now=NOW + 70000,
        expires_at=NOW + 170000,
        label="Single settlement after recovery",
    )
    runtime.tick(NOW + 70000)
    runtime.tick(NOW + 70100)
    saved = runtime.save

    def crash():
        if commit_before_crash:
            saved()
        raise KeyboardInterrupt

    mocker.patch.object(runtime, "save", side_effect=crash)
    with pytest.raises(KeyboardInterrupt):
        runtime.tick(NOW + 70200)
    restored = LiveExecution(tmp_path, ExchangeFixture(tmp_path / "exchange.sqlite"), ACCOUNT)
    restored.broker.writes_enabled = False
    restored.tick(NOW + 70300)
    restored.tick(NOW + 70400)
    final = restored.status()
    assert final["flow"]["status"] == "completed"
    assert money(final["book"]["locked"]["USDGO"]) == money(first["book"]["locked"]["USDGO"]) - 10
    assert final["book"]["usdgo"] == first["book"]["usdgo"]
    assert final["book"]["stable_interest"] == first["book"]["stable_interest"]
    assert len(final["flow"]["outputs"]) == 1
    assert len(restored.broker.state()["writes"]) == 3


def test_unrelated_deposit_cannot_be_used_as_redemption_arrival(runtime):
    first = run_flow(runtime, [trade(), subscribe("USDGO")])
    holding = first["flow"]["outputs"][-1]["holding_id"]
    runtime.queue(
        [
            {
                "kind": "redeem",
                "coin": "USDGO",
                "product_id": "USDGO-flex",
                "holding_id": holding,
                "amount": "10",
            }
        ],
        now=NOW + 70000,
        expires_at=NOW + 170000,
        label="Attribution",
    )
    runtime.tick(NOW + 70000)
    runtime.tick(NOW + 70100)
    state = runtime.broker.state()
    state["bills"][-1]["bizOrderId"] = "manual-deposit"
    runtime.broker.put(state)
    runtime.tick(NOW + 70200)
    assert runtime.status()["flow"]["status"] == "running"
    assert money(runtime.status()["book"]["locked"]["USDGO"]) == money(
        first["book"]["locked"]["USDGO"]
    )


def test_cancel_fill_race_and_restart_apply_each_real_fill_once(runtime, tmp_path):
    runtime.queue(
        [trade("TRXUSDT", minimum="1")], now=NOW, expires_at=NOW + 10000, label="Cancel race"
    )
    runtime.tick(NOW)
    runtime.broker.fault = "cancel_fill_race"
    runtime.tick(NOW + 100)
    runtime.tick(NOW + 200)
    partial = Ledger.restore(runtime.status()["book"]).principal
    runtime.tick(NOW + 11000)
    runtime.tick(NOW + 21000)
    state = runtime.status()
    assert state["flow"]["status"] == "completed"
    assert Ledger.restore(state["book"]).principal == partial * 2
    assert state["anchor_version"] == 1
    restarted = LiveExecution(tmp_path, ExchangeFixture(tmp_path / "exchange.sqlite"), ACCOUNT)
    restarted.tick(NOW + 22000)
    assert restarted.status()["book"] == state["book"]
    assert len(restarted.broker.state()["writes"]) == 1


def test_partial_cancel_stops_and_does_not_repeat_anchor(runtime):
    runtime.queue(
        [trade("TRXUSDT"), subscribe("TRX")], now=NOW, expires_at=NOW + 10000, label="Partial"
    )
    runtime.tick(NOW)
    runtime.broker.fault = "partial"
    runtime.tick(NOW + 100)
    runtime.tick(NOW + 200)
    runtime.tick(NOW + 11000)
    runtime.tick(NOW + 12000)
    state = runtime.status()
    assert state["flow"]["status"] == "needs_review"
    runtime.tick(NOW + 13000)
    assert runtime.status()["anchor_version"] == state["anchor_version"] == 1
    assert len(runtime.broker.state()["writes"]) == 1


def open_partial_trade(runtime):
    runtime.queue(
        [trade("TRXUSDT", minimum="1"), subscribe("TRX")],
        now=NOW,
        expires_at=NOW + 10000,
        label="Cancellation recovery",
    )
    runtime.tick(NOW)
    runtime.broker.fault = "partial"
    runtime.tick(NOW + 100)
    runtime.tick(NOW + 200)
    return runtime.status()["operation"]["id"]


@pytest.mark.parametrize("before_journal", [True, False])
def test_cancel_not_sent_rechecks_order_then_recovers_across_restart(
    runtime, tmp_path, mocker, before_journal
):
    client_id = open_partial_trade(runtime)
    broker = runtime.broker
    cancels = mocker.spy(broker, "cancel")
    gate = mocker.patch.object(broker, "reserve_cancellation", create=True)
    if before_journal:
        gate.side_effect = WriteNotSent("request_rate_wait")
    else:
        broker.fault = "cancel_not_sent"
    runtime.tick(NOW + 11000)
    state = runtime.status()
    assert not state["operation"]["cancel_attempted"]
    assert state["operation"]["cancel_outcome"] == "deferred"
    assert cancels.call_count == (0 if before_journal else 1)
    started = state["operation"]["cancel_started_at"]
    gate.side_effect = None
    broker.fault = None
    observations = []
    query = broker.order
    cancel = broker.cancel

    def observe(oid):
        observations.append("query")
        return query(oid)

    def send(oid, symbol):
        observations.append("cancel")
        return cancel(oid, symbol)

    mocker.patch.object(broker, "order", side_effect=observe)
    mocker.patch.object(broker, "cancel", side_effect=send)
    restored = LiveExecution(tmp_path, broker, ACCOUNT)
    restored.tick(NOW + 12000)
    assert observations == ["query", "cancel"]
    assert broker.state()["orders"][client_id]["status"] == "cancelled"
    assert restored.status()["operation"]["cancel_started_at"] == started
    restored.tick(NOW + 13000)
    final = restored.status()
    assert final["flow"]["status"] == "needs_review"  # actual partial fill retained
    assert final["anchor_version"] == 1
    assert len(final["book"]["orders"]) == 1
    assert len(broker.state()["writes"]) == 1  # never execute the next subscription


def test_cancel_not_sent_observes_a_fill_before_trying_again(runtime, tmp_path, mocker):
    client_id = open_partial_trade(runtime)
    broker = runtime.broker
    broker.fault = "cancel_not_sent"
    cancels = mocker.spy(broker, "cancel")
    runtime.tick(NOW + 11000)
    remote = broker.state()
    order = remote["orders"][client_id]
    broker.apply_spot(remote, order, money(order["requested"]) / 2, NOW + 12000)
    order["status"] = "filled"
    broker.put(remote)
    restored = LiveExecution(tmp_path, broker, ACCOUNT)
    restored.tick(NOW + 13000)
    assert cancels.call_count == 1
    assert restored.status()["operation"] is None
    assert restored.status()["anchor_version"] == 1
    restored.tick(NOW + 14000)
    assert restored.status()["flow"]["status"] == "expired"  # no retrospective next step
    assert len(broker.state()["writes"]) == 1


@pytest.mark.parametrize("accepted", [False, True])
def test_cancel_timeout_queries_only_after_restart_and_escalates_if_still_open(
    runtime, tmp_path, mocker, accepted
):
    client_id = open_partial_trade(runtime)
    broker = runtime.broker
    broker.fault = "cancel_timeout_after_accept" if accepted else "cancel_timeout_before_accept"
    cancels = mocker.spy(broker, "cancel")
    runtime.tick(NOW + 11000)
    assert runtime.status()["operation"]["cancel_outcome"] == "unknown"
    restored = LiveExecution(tmp_path, broker, ACCOUNT)
    # Even beyond the deadline, an actual terminal status takes precedence.
    restored.tick(NOW + (90000 if accepted else 12000))
    if not accepted:
        assert restored.status()["operation"]["status"] == "open"
        # Repeated reads and restart must not reset the persisted deadline.
        restored.tick(NOW + 70000)
        restored = LiveExecution(tmp_path, broker, ACCOUNT)
        restored.tick(NOW + 71000)
    state = restored.status()
    assert cancels.call_count == 1
    assert state["flow"]["status"] == "needs_review"
    code = state["events"][-1]["code"]
    assert code == (
        "partial_or_minimum_receipt_not_satisfied" if accepted else "cancel_confirmation_timeout"
    )
    assert broker.state()["orders"][client_id]["status"] == (
        "cancelled" if accepted else "partially_filled"
    )
    assert len(broker.state()["writes"]) == 1


def test_acknowledged_cancel_that_remains_open_escalates_in_readonly_recovery(
    runtime, tmp_path, mocker
):
    client_id = open_partial_trade(runtime)
    broker = runtime.broker
    order_id = broker.state()["orders"][client_id]["orderId"]
    cancel = mocker.patch.object(broker, "cancel", return_value={"orderId": order_id})
    runtime.tick(NOW + 11000)
    assert runtime.status()["operation"]["cancel_outcome"] == "accepted"
    broker.writes_enabled = False
    restored = LiveExecution(tmp_path, broker, ACCOUNT)
    restored.tick(NOW + 71000)
    assert restored.status()["flow"]["status"] == "needs_review"
    assert restored.status()["events"][-1]["code"] == "cancel_confirmation_timeout"
    cancel.assert_called_once()


@pytest.mark.parametrize("fills_before_query", [False, True])
def test_cancel_explicit_rejection_requeries_before_operator_review_or_settlement(
    runtime, tmp_path, mocker, fills_before_query
):
    client_id = open_partial_trade(runtime)
    broker = runtime.broker
    broker.fault = "cancel_rejected"
    cancels = mocker.spy(broker, "cancel")
    runtime.tick(NOW + 11000)
    assert runtime.status()["operation"]["cancel_outcome"] == "rejected"
    if fills_before_query:
        remote = broker.state()
        order = remote["orders"][client_id]
        broker.apply_spot(remote, order, money(order["requested"]) / 2, NOW + 12000)
        order["status"] = "filled"
        broker.put(remote)
    restored = LiveExecution(tmp_path, broker, ACCOUNT)
    restored.tick(NOW + 13000)
    state = restored.status()
    if fills_before_query:
        assert state["operation"] is None
        assert state["anchor_version"] == 1
    else:
        assert state["flow"]["status"] == "needs_review"
        assert state["events"][-1]["code"] == "cancel_rejected_requires_review"
        assert state["operation"]["cancel_review_code"] == "cancel_rejected_requires_review"
        restored.tick(NOW + 90000)
    assert cancels.call_count == 1
    assert len(broker.state()["writes"]) == 1


def test_continuous_cancel_admission_failure_has_a_persisted_deadline(runtime, tmp_path, mocker):
    open_partial_trade(runtime)
    broker = runtime.broker
    gate = mocker.patch.object(
        broker, "reserve_cancellation", create=True, side_effect=WriteNotSent("request_rate_wait")
    )
    cancel = mocker.spy(broker, "cancel")
    runtime.tick(NOW + 11000)
    restored = LiveExecution(tmp_path, broker, ACCOUNT)
    restored.tick(NOW + 71000)
    state = restored.status()
    assert state["flow"]["status"] == "needs_review"
    assert state["events"][-1]["code"] == "cancel_admission_timeout"
    assert gate.call_count == 1 and cancel.call_count == 0
    assert len(broker.state()["writes"]) == 1


def test_old_cancel_attempt_snapshot_recovers_without_resubmitting(runtime, tmp_path, mocker):
    open_partial_trade(runtime)
    op = runtime.state["operation"]
    op["cancel_attempted"] = True
    runtime.event("cancel_attempt", NOW + 11000, operation_id=op["id"])
    runtime.save()
    restored = LiveExecution(tmp_path, runtime.broker, ACCOUNT)
    cancel = mocker.spy(runtime.broker, "cancel")
    restored.tick(NOW + 71000)
    assert restored.status()["flow"]["status"] == "needs_review"
    assert restored.status()["events"][-1]["code"] == "cancel_confirmation_timeout"
    cancel.assert_not_called()


def test_crash_after_cancel_acceptance_only_queries_on_restart(runtime, tmp_path, mocker):
    open_partial_trade(runtime)
    broker = runtime.broker
    broker.fault = "cancel_crash_after_accept"
    cancels = mocker.spy(broker, "cancel")
    with pytest.raises(KeyboardInterrupt):
        runtime.tick(NOW + 11000)
    assert runtime.status()["operation"]["cancel_outcome"] == "sending"
    restored = LiveExecution(tmp_path, broker, ACCOUNT)
    restored.tick(NOW + 12000)
    assert restored.status()["flow"]["status"] == "needs_review"  # partial fill, no duplicate debit
    assert restored.status()["anchor_version"] == 1
    cancels.assert_called_once()


def test_definitive_rejection_keeps_balances_and_can_queue_new_flow(runtime):
    runtime.queue([trade()], now=NOW, expires_at=NOW + 60000, label="Rejected")
    runtime.tick(NOW)
    runtime.broker.fault = "reject"
    runtime.tick(NOW + 100)
    assert runtime.status()["flow"]["status"] == "failed"
    assert runtime.status()["operation"] is None
    assert money(runtime.status()["book"]["cash"]) == money("1000.4")
    runtime.broker.fault = None
    assert run_flow(runtime, [trade()], NOW + 1000)["flow"]["status"] == "completed"


def test_expired_flow_after_redemption_never_places_retrospective_trade(runtime):
    first = run_flow(runtime, [trade(), subscribe("USDGO")])
    holding = first["flow"]["outputs"][-1]["holding_id"]
    runtime.queue(
        [
            {
                "kind": "redeem",
                "coin": "USDGO",
                "product_id": "USDGO-flex",
                "holding_id": holding,
                "amount": "10",
            },
            trade(side="sell", amount="previous"),
        ],
        now=NOW + 70000,
        expires_at=NOW + 71000,
        label="Signal expiry",
    )
    runtime.tick(NOW + 70000)
    runtime.tick(NOW + 70100)
    runtime.tick(NOW + 72000)
    runtime.tick(NOW + 72100)
    assert runtime.status()["flow"]["status"] == "expired"
    assert len(runtime.broker.state()["writes"]) == 3


def test_read_failure_preserves_durable_intent_and_read_only_poll_cannot_submit(runtime):
    runtime.queue([trade()], now=NOW, expires_at=NOW + 60000, label="Read only")
    runtime.tick(NOW)
    runtime.broker.writes_enabled = False
    runtime.tick(NOW + 100)
    assert runtime.status()["operation"]["status"] == "prepared"
    runtime.broker.writes_enabled = True
    runtime.broker.read_failure = True
    runtime.tick(NOW + 200)
    assert runtime.broker.state()["writes"] == []
    runtime.broker.read_failure = False
    runtime.tick(NOW + 300)
    runtime.tick(NOW + 400)
    assert runtime.status()["flow"]["status"] == "completed"


def test_account_identity_change_blocks_all_new_writes(runtime, mocker):
    runtime.queue([trade()], now=NOW, expires_at=NOW + 60000, label="Account identity")
    mocker.patch.object(runtime.broker, "account_id", return_value="another-account")
    with pytest.raises(ExecutionError, match="account_binding_mismatch"):
        runtime.tick(NOW)
    assert runtime.broker.state()["writes"] == []


def test_conflicting_fill_batch_is_atomic(runtime):
    runtime.queue([trade("TRXUSDT")], now=NOW, expires_at=NOW + 60000, label="Conflicting receipts")
    runtime.tick(NOW)
    runtime.broker.fault = "partial"
    runtime.tick(NOW + 100)
    runtime.tick(NOW + 200)
    book = runtime.status()["book"]
    state = runtime.broker.state()
    order = next(iter(state["orders"].values()))
    order["fills"][0]["fees"][0]["cost"] = "0.123"
    runtime.broker.put(state)
    runtime.tick(NOW + 300)
    assert runtime.status()["book"] == book
    assert runtime.status()["events"][-1]["code"] == "conflicting_fill_receipt"


def test_missing_redemption_minimum_does_not_use_subscription_minimum(runtime):
    first = run_flow(runtime, [trade(), subscribe("USDGO")])
    holding = first["flow"]["outputs"][-1]["holding_id"]
    runtime.reload()
    runtime.state["products"][0]["redemption_minimum"] = None
    runtime.save()
    state = run_flow(
        runtime,
        [
            {
                "kind": "redeem",
                "coin": "USDGO",
                "product_id": "USDGO-flex",
                "holding_id": holding,
                "amount": "10",
            }
        ],
        NOW + 70000,
    )
    assert state["events"][-1]["code"] == "redemption_terms_unverified"
    assert len(runtime.broker.state()["writes"]) == 2


def test_trx_redemption_profitable_sale_usdgo_purchase_and_subscription(runtime):
    first = run_flow(runtime, [trade("TRXUSDT"), subscribe("TRX")])
    assert first["flow"]["status"] == "completed"
    holding = first["flow"]["outputs"][-1]["holding_id"]
    amount = first["flow"]["outputs"][-1]["amount"]
    state = runtime.broker.state()
    state["trx_price"] = "0.31"
    runtime.broker.put(state)
    sell = {**trade("TRXUSDT", "sell", "previous"), "reference_price": "0.31"}
    final = run_flow(
        runtime,
        [
            {
                "kind": "redeem",
                "coin": "TRX",
                "product_id": "TRX-flex",
                "holding_id": holding,
                "amount": amount,
            },
            sell,
            trade(amount="previous"),
            subscribe("USDGO"),
        ],
        NOW + 70000,
    )
    assert final["flow"]["status"] == "completed"
    book = Ledger.restore(final["book"])
    assert book.realized > 0
    assert book.usdgo > 100
    assert book.locked["USDGO"] > 100
    assert book.interest_trx == 0
    assert final["anchor_version"] == 2
    runtime.check_wallet(runtime.broker.balances(), book)


def credit_provider_interest(runtime, coin, amount, destination, now):
    positions = runtime.status()["holdings"]
    holding = next(key for key, p in positions.items() if p["coin"] == coin)
    remote = runtime.broker.state()
    remote["records"].append(
        {
            "orderId": holding,
            "coinName": coin,
            "settleCoinName": coin,
            "productType": "flexible",
            "orderType": "pay_interest",
            "amount": amount,
            "ts": str(now),
        }
    )
    if destination == "earn":
        remote["holdings"][holding]["amount"] = str(
            money(remote["holdings"][holding]["amount"]) + money(amount)
        )
    else:
        remote["wallet"][coin] = str(money(remote["wallet"][coin]) + money(amount))
        remote["bills"].append(
            {
                "billId": "interest-" + str(now),
                "bizOrderId": holding,
                "coin": coin,
                "groupType": "financial",
                "size": amount,
                "fees": "0",
                "cTime": str(now),
            }
        )
    runtime.broker.put(remote)


@pytest.mark.parametrize("coin", ["TRX", "USDGO"])
@pytest.mark.parametrize("destination", ["spot", "earn"])
def test_actual_native_interest_is_attributed_once_and_survives_restart(
    runtime, tmp_path, coin, destination
):
    symbol = "TRXUSDT" if coin == "TRX" else "USDGOUSDT"
    assert run_flow(runtime, [trade(symbol), subscribe(coin)])["flow"]["status"] == "completed"
    credit_provider_interest(runtime, coin, "0.01", destination, NOW + 10000)
    assert runtime.import_interest(NOW + 11000) == 1
    once = runtime.status()
    restored = LiveExecution(tmp_path, ExchangeFixture(tmp_path / "exchange.sqlite"), ACCOUNT)
    assert restored.import_interest(NOW + 12000) == 0
    assert restored.status()["book"] == once["book"]
    book = Ledger.restore(once["book"])
    if coin == "TRX":
        assert book.interest_trx == money("0.01")
        assert book.locked["TRX_INTEREST"] == (money("0.01") if destination == "earn" else 0)
    else:
        assert book.stable_interest == money("0.01001")
    assert book.fee_rebates == 0


def test_interest_record_without_arrival_evidence_is_not_credited(runtime):
    run_flow(runtime, [trade("TRXUSDT"), subscribe("TRX")])
    credit_provider_interest(runtime, "TRX", "0.01", "spot", NOW + 10000)
    runtime.broker.hide_bills = True
    before = runtime.status()["book"]
    with pytest.raises(ExecutionError, match="interest_arrival_bill_unverified"):
        runtime.import_interest(NOW + 11000)
    assert runtime.status()["book"] == before


def test_retained_trx_interest_cannot_be_sold_as_principal(runtime):
    run_flow(runtime, [trade("TRXUSDT"), subscribe("TRX")])
    credit_provider_interest(runtime, "TRX", "0.01", "spot", NOW + 10000)
    runtime.import_interest(NOW + 11000)
    state = run_flow(runtime, [trade("TRXUSDT", "sell", "0.01", minimum="0.001")], NOW + 20000)
    assert state["events"][-1]["code"] in {
        "order_below_minimum",
        "profitable_principal_unavailable",
    }
    assert len(runtime.broker.state()["writes"]) == 2
    assert money(state["book"]["interest_trx"]) == money("0.01")


def test_second_writer_cannot_advance_the_same_account(runtime, tmp_path):
    second = LiveExecution(tmp_path, runtime.broker, ACCOUNT)
    with runtime.exclusive():
        with pytest.raises(ExecutionError, match="execution_busy"):
            second.queue([trade()], now=NOW, expires_at=NOW + 10000, label="Concurrent")
    assert runtime.broker.state()["writes"] == []


def test_signal_can_be_stopped_without_discarding_an_accepted_redemption(runtime):
    first = run_flow(runtime, [trade(), subscribe("USDGO")])
    holding = first["flow"]["outputs"][-1]["holding_id"]
    runtime.queue(
        [
            {
                "kind": "redeem",
                "coin": "USDGO",
                "product_id": "USDGO-flex",
                "holding_id": holding,
                "amount": "10",
            },
            trade(side="sell", amount="previous"),
        ],
        now=NOW + 70000,
        expires_at=NOW + 170000,
        label="Stop",
    )
    runtime.tick(NOW + 70000)
    runtime.tick(NOW + 70100)
    runtime.request_stop(NOW + 70200)
    assert runtime.status()["operation"] is not None
    runtime.tick(NOW + 70300)
    runtime.tick(NOW + 70400)
    assert runtime.status()["flow"]["status"] == "expired"
    assert len(runtime.broker.state()["writes"]) == 3


def test_web_execution_status_is_local_only_and_shows_durable_stage(runtime, tmp_path):
    from freqtrade.rpc.api_server import api_trx

    runtime.queue([trade()], now=NOW, expires_at=NOW + 60000, label="Inspectable")
    result = api_trx.live_execution_status({"user_data_dir": tmp_path.parent})
    # This fixture's root differs from the webserver convention, so also inspect
    # the service directly. Neither path constructs or calls an exchange client.
    assert result["page_can_submit_orders"] is False
    from freqtrade.strategy.trx_phase1.live_execution import local_execution_status

    result = local_execution_status(tmp_path)
    assert result["accounts"][0]["flow"]["label"] == "Inspectable"
    assert runtime.broker.state()["writes"] == []


@pytest.mark.parametrize("destination", ["spot", "earn"])
def test_native_interest_arriving_during_redemption_is_reconciled_atomically(runtime, destination):
    first = run_flow(runtime, [trade(), subscribe("USDGO")])
    holding = first["flow"]["outputs"][-1]["holding_id"]
    runtime.queue(
        [
            {
                "kind": "redeem",
                "coin": "USDGO",
                "product_id": "USDGO-flex",
                "holding_id": holding,
                "amount": "10",
            }
        ],
        now=NOW + 70000,
        expires_at=NOW + 170000,
        label="Concurrent credited interest",
    )
    runtime.tick(NOW + 70000)
    runtime.tick(NOW + 70100)
    credit_provider_interest(runtime, "USDGO", "0.01", destination, NOW + 70200)
    runtime.tick(NOW + 70300)
    state = runtime.status()
    assert state["flow"]["status"] == "completed"
    assert money(state["book"]["stable_interest"]) == money("0.01001")
    assert len(runtime.broker.state()["writes"]) == 3
    assert runtime.import_interest(NOW + 70400) == 0


def test_reported_turnover_preserves_exact_cost_when_unit_price_repeats():
    book = Ledger("10")
    price = money(1) / 3
    book.buy("buy", "order", NOW, 3, price, quote_amount="1")
    assert book.cash == 9
    assert book.cost == 1
    book.sell(
        "sell", "sale", NOW + 1000, 3, price, [{"lot_id": "order", "quantity": 3}], quote_amount="1"
    )
    assert book.cash == 10 and book.cost == 0 and book.realized == 0


def test_signal_expiring_during_preflight_never_reaches_submission(runtime, mocker):
    runtime.queue([trade()], now=NOW, expires_at=NOW + 1000, label="Slow preflight")
    runtime.tick(NOW)
    mocker.patch("freqtrade.strategy.trx_phase1.live_execution.time.monotonic", side_effect=[0, 2])
    runtime.tick(NOW + 100)
    assert runtime.status()["flow"]["status"] == "expired"
    assert runtime.broker.state()["writes"] == []


def test_rate_deferral_before_journal_never_becomes_an_unknown_order(runtime, mocker):
    from freqtrade.strategy.trx_phase1.live_broker import WriteNotSent

    runtime.queue([trade()], now=NOW, expires_at=NOW + 10000, label="Rate limited")
    runtime.tick(NOW)
    gate = mocker.patch.object(
        runtime.broker,
        "reserve_submission",
        create=True,
        side_effect=WriteNotSent("request_rate_wait"),
    )
    runtime.tick(NOW + 100)
    assert runtime.state["operation"]["status"] == "prepared"
    assert runtime.broker.state()["writes"] == []
    assert not any(e["kind"] == "write_attempt" for e in runtime.status()["events"])
    gate.side_effect = None
    runtime.tick(NOW + 200)
    assert len(runtime.broker.state()["writes"]) == 1


def test_signal_expiring_in_rate_queue_is_discarded(runtime, mocker):
    runtime.queue([trade()], now=NOW, expires_at=NOW + 1000, label="Queue expiry")
    runtime.tick(NOW)
    mocker.patch.object(runtime.broker, "reserve_submission", create=True)
    mocker.patch("freqtrade.strategy.trx_phase1.live_execution.time.monotonic", side_effect=[0, 2])
    runtime.tick(NOW + 100)
    assert runtime.status()["flow"]["status"] == "expired"
    assert runtime.broker.state()["writes"] == []


def test_redeeming_one_earn_order_unlocks_its_original_cost_lots_only(runtime):
    run_flow(runtime, [trade("TRXUSDT"), subscribe("TRX")])
    old_book = Ledger.restore(runtime.status()["book"])
    original_lot = old_book.lots[0].lot_id
    remote = runtime.broker.state()
    remote["trx_price"] = "0.2"
    runtime.broker.put(remote)
    second = run_flow(
        runtime, [{**trade("TRXUSDT"), "reference_price": "0.2"}, subscribe("TRX")], NOW + 10000
    )
    assert second["flow"]["status"] == "completed"
    holding = second["flow"]["outputs"][-1]["holding_id"]
    new_lot = Ledger.restore(second["book"]).lots[-1].lot_id
    assert set(second["holdings"][holding]["lots"]) == {new_lot}
    remote = runtime.broker.state()
    remote["trx_price"] = "0.25"
    runtime.broker.put(remote)
    final = run_flow(
        runtime,
        [
            {
                "kind": "redeem",
                "coin": "TRX",
                "product_id": "TRX-flex",
                "holding_id": holding,
                "amount": "10",
            },
            {**trade("TRXUSDT", "sell", "10"), "reference_price": "0.25"},
        ],
        NOW + 80000,
    )
    assert final["flow"]["status"] == "completed"
    book = Ledger.restore(final["book"])
    assert book.realized > 0
    assert book.lots[0].lot_id == original_lot
    assert book.lots[0].locked == old_book.lots[0].locked
    assert all(a["lot_id"] == new_lot for a in book.orders[-1]["allocations"])
