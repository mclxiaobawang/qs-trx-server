"""The actual HTTP admission path over a separate synthetic provider journal."""

import json
import sqlite3
from copy import deepcopy
from types import SimpleNamespace

import pytest

from freqtrade.strategy.trx_phase1.account import AccountSettings
from freqtrade.strategy.trx_phase1.ledger import Ledger
from freqtrade.strategy.trx_phase1.live_broker import ClassicBroker, ClassicTransport, WriteNotSent
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from freqtrade.strategy.trx_phase1.request_limits import RequestLimiter
from tests.strategy.trx_phase1.test_live_execution import (
    NOW,
    RULES,
    ExchangeFixture,
    subscribe,
    trade,
)


class Clock:
    def __init__(self):
        self.now = NOW / 1000

    def __call__(self):
        return self.now

    def sleep(self, n):
        self.now += n


class GatedBroker:
    writes_enabled = True

    def __init__(self, path, clock):
        self.backend = ExchangeFixture(path / "provider.sqlite")
        self.clock_value = clock
        self.calls = []
        self.operation = None
        limiter = RequestLimiter(
            path / "limits.sqlite", clock=clock, sleep=clock.sleep, jitter=lambda: 0
        )
        self.transport = ClassicTransport(
            AccountSettings(api_key="offline", secret_key="offline", passphrase="offline"),
            writes_enabled=True,
            session=SimpleNamespace(request=self.http),
            limiter=limiter,
        )

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def clock(self):
        return int(self.clock_value() * 1000)

    def reserve_submission(self, op, expires):
        return ClassicBroker.reserve_submission(self, op, expires)

    def submit(self, op):
        self.operation = deepcopy(op)
        return ClassicBroker.submit(self, op)

    def http(self, method, url, **kwargs):
        assert method == "POST" and json.loads(kwargs["data"]) == self.operation["request"]
        self.calls.append(
            {"time": self.clock_value(), "kind": self.operation["kind"], "id": self.operation["id"]}
        )
        result = self.backend.submit(self.operation)
        return SimpleNamespace(
            status_code=200, headers={}, json=lambda: {"code": "00000", "data": result}
        )


@pytest.fixture
def gated(tmp_path):
    clock = Clock()
    broker = GatedBroker(tmp_path, clock)
    execution = LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)
    original = execution.save

    def slow_journal():
        original()
        clock.sleep(0.12)

    execution.save = slow_journal
    return execution, broker, clock


def advance(execution, clock, steps=30):
    for _ in range(steps):
        clock.sleep(1)
        execution.tick(int(clock() * 1000))
        if execution.status()["flow"]["status"] in {
            "completed",
            "failed",
            "expired",
            "needs_review",
        }:
            break
    return execution.status()


def queue(execution, clock, steps, ttl=600000):
    now = int(clock() * 1000)
    execution.queue(steps, now=now, expires_at=now + ttl, label="Offline new-valid-plan only")
    return advance(execution, clock)


def test_slow_journal_full_subscribe_redeem_arrival_sell_chain(gated):
    execution, broker, clock = gated
    first = queue(execution, clock, [trade("TRXUSDT"), subscribe("TRX")])
    assert first["flow"]["status"] == "completed"
    holding = first["flow"]["outputs"][-1]["holding_id"]
    old = Ledger.restore(first["book"])
    state = broker.backend.state()
    state["trx_price"] = ".4"
    broker.backend.put(state)
    steps = [
        {
            "kind": "redeem",
            "coin": "TRX",
            "product_id": "TRX-flex",
            "holding_id": holding,
            "amount": "20",
        },
        {**trade("TRXUSDT", "sell", "20"), "reference_price": ".4"},
    ]
    now = int(clock() * 1000)
    execution.queue(steps, now=now, expires_at=now + 600000, label="Offline fresh sell")
    clock.sleep(1)
    execution.tick(int(clock() * 1000))
    clock.sleep(1)
    execution.tick(int(clock() * 1000))
    assert execution.status()["operation"]["status"] == "accepted"
    # Native redemption result, wallet and bill must agree before the sell step.
    broker.backend.hide_bills = True
    clock.sleep(1)
    execution.tick(int(clock() * 1000))
    assert len(broker.calls) == 3 and execution.status()["flow"]["index"] == 0
    broker.backend.hide_bills = False
    broker.backend.lagged_balances = execution.status()["operation"]["before_balances"]
    clock.sleep(1)
    execution.tick(int(clock() * 1000))
    assert len(broker.calls) == 3 and execution.status()["flow"]["index"] == 0
    broker.backend.lagged_balances = None
    final = advance(execution, clock)
    assert final["flow"]["status"] == "completed"
    assert [v["kind"] for v in broker.calls] == ["trade", "subscribe", "redeem", "trade"]
    assert len({v["id"] for v in broker.calls}) == 4
    book = Ledger.restore(final["book"])
    assert book.realized > 0 and book.cost < old.cost
    assert final["book"]["initial"] == first["book"]["initial"]
    execution.check_wallet(broker.balances(), book)
    with sqlite3.connect(broker.transport.limiter.path) as db:
        # Exactly one budget admission per first HTTP submission, never twice per redeem.
        assert db.execute("SELECT value FROM metrics WHERE key='requests'").fetchone()[0] == 4


@pytest.mark.parametrize("kind", ["subscribe", "redeem"])
@pytest.mark.parametrize("journal_delay", [0.1, 1.0, 5.0])
def test_admit_once_after_slow_journal_and_preserve_redemption_cooldown(
    tmp_path, kind, journal_delay
):
    clock = Clock()
    calls = []
    limiter = RequestLimiter(
        tmp_path / "limits.sqlite", clock=clock, sleep=clock.sleep, jitter=lambda: 0
    )
    session = SimpleNamespace(
        request=lambda *a, **k: (
            calls.append(clock())
            or SimpleNamespace(
                status_code=200,
                headers={},
                json=lambda: {"code": "00000", "data": {"orderId": "offline"}},
            )
        )
    )
    transport = ClassicTransport(
        AccountSettings(api_key="offline", secret_key="offline", passphrase="offline"),
        writes_enabled=True,
        session=session,
        limiter=limiter,
    )
    path = "v2/earn/savings/" + kind
    body = {"productId": "p", "periodType": "flexible", "amount": "1"}
    if kind == "redeem":
        body["orderId"] = "holding"
    transport.reserve_write(path, 10)
    clock.sleep(journal_delay)
    transport.call("POST", path, body)
    assert len(calls) == 1
    with sqlite3.connect(limiter.path) as db:
        assert db.execute("SELECT value FROM metrics WHERE key='requests'").fetchone()[0] == 1
    if kind == "redeem":
        transport.reserve_write(path, 10)
        with pytest.raises(WriteNotSent):
            transport.call("POST", path, body)
        assert len(calls) == 1
        clock.sleep(61)
        transport.reserve_write(path, 10)
        transport.call("POST", path, body)
        assert len(calls) == 2 and calls[1] - calls[0] >= 61


def test_deadline_expiring_during_admission_never_sends(tmp_path):
    clock = Clock()
    limiter = RequestLimiter(
        tmp_path / "limits.sqlite", clock=clock, sleep=clock.sleep, jitter=lambda: 0
    )
    calls = []
    transport = ClassicTransport(
        AccountSettings(api_key="offline", secret_key="offline", passphrase="offline"),
        writes_enabled=True,
        session=SimpleNamespace(request=lambda *a, **k: calls.append(1)),
        limiter=limiter,
    )
    with limiter.db() as db:
        limiter.set_ready(db, "private:unbound", clock() + 0.4)
    transport.reserve_write("v2/earn/savings/redeem", 0.2)
    with pytest.raises(WriteNotSent):
        transport.call(
            "POST",
            "v2/earn/savings/redeem",
            {"productId": "p", "orderId": "holding", "periodType": "flexible", "amount": "1"},
        )
    assert calls == []


def test_provider_acceptance_then_timeout_is_not_redeemed_twice(gated):
    execution, broker, clock = gated
    first = queue(execution, clock, [trade("TRXUSDT"), subscribe("TRX")])
    holding = first["flow"]["outputs"][-1]["holding_id"]
    broker.backend.fault = "timeout_after_accept"
    execution.queue(
        [
            {
                "kind": "redeem",
                "coin": "TRX",
                "product_id": "TRX-flex",
                "holding_id": holding,
                "amount": "20",
            }
        ],
        now=int(clock() * 1000),
        expires_at=int(clock() * 1000) + 600000,
        label="Offline unknown result",
    )
    advance(execution, clock, 5)
    count = len(broker.calls)
    assert count == 3
    broker.backend.fault = None
    clock.sleep(61)
    advance(execution, clock, 5)
    assert len(broker.calls) == count and execution.status()["operation"]["status"] in {
        "unknown",
        "needs_review",
    }
