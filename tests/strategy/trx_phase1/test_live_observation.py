from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_observation import dashboard, observed_close
from tests.strategy.trx_phase1.test_live_strategy import (
    advance,
    candidate,
    live,  # noqa: F401 - shared live exchange fixture
    next_buy_close,
    startup,
)


@pytest.mark.parametrize("price", [0.3, 0.28, 0.32])
@pytest.mark.parametrize("paused", [True, False])
def test_observing_a_close_cannot_change_engine_decisions_or_ledger(price, paused):
    engine = Engine(candidate())
    now = 1792256400000
    engine.book.buy("fill", "lot", now - 3600000, 100, 0.3, 0.03)
    engine.anchor, engine.anchor_version = 0.3, 1
    engine.buy_at = engine.sell_at = now
    engine.price_position_day = (now + 480 * 60000) // 86400000
    engine.price_position_reference = 0.31
    engine.paused = paused
    control = Engine.restore(engine.state())
    control.price_position_day = engine.price_position_day
    control.price_position_reference = engine.price_position_reference
    expected = control.on_close(now, price, 0.001)
    result, rows = observed_close(engine, now, price, 0.001, now + 30000)
    assert engine.state() == control.state()
    assert result == expected
    assert engine.audit is None
    assert len(rows) == 2
    assert all(r["processed_at"] == now + 30000 for r in rows)
    if paused:
        assert {r["result"] for r in rows} == {"skipped"}
    elif price == 0.3:
        assert {r["result"] for r in rows} == {"not_triggered"}
    else:
        assert any(r["result"] == "plan_created" for r in rows)


def test_actual_negative_confirmation_survives_restart_without_backfilled_rows(live):  # noqa: F811
    runtime, broker, market, root = live
    startup(live)
    assert runtime.state.get("evaluations", []) == []
    before = deepcopy(runtime.execution.status())
    next_buy_close(live)
    rows = runtime.state["evaluations"]
    assert len(rows) == 2
    assert all(r["result"] == "not_triggered" for r in rows)
    from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
    from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategy

    restored = LiveStrategy(LiveExecution(root, broker, runtime.execution.account_id), market)
    assert restored.state["evaluations"] == rows
    advance(live)
    assert runtime.state["evaluations"] == rows
    for field in ("book", "holdings", "receipts", "anchor", "anchor_version"):
        assert runtime.execution.status()[field] == before[field]


def test_dashboard_is_read_only_accounts_for_native_interest_and_excludes_external():
    book = Ledger(1000)
    book.buy("fill", "lot", 1, 100, 0.3, 0.03)
    book.convert("fx", "to_usdgo", 800, 799, 1.001)
    book.credit_interest("trx-income", "TRX", ".00333986")
    book.credit_interest("usdgo-income", "USDGO", ".01023801", fx="1.0011")
    book.lots[0].locked = money(100)
    book.locked["USDGO"] = money(799)
    engine = Engine(candidate(), book)
    engine.anchor = 0.33597
    engine.buy_at = engine.sell_at = 1792260000000
    state = {
        "markets": {
            "TRXUSDT": {"bid": ".33463", "ask": ".33464", "timestamp": 1792256400000},
            "USDGOUSDT": {"bid": "1.0011", "ask": "1.0012", "fee": "0", "timestamp": 1792256400000},
        },
        "price_reference": {"value": 0.332},
    }
    execution = {
        "holdings": {
            "trx": {
                "coin": "TRX",
                "principal": "100",
                "interest": "0",
                "lots": {"lot": "100"},
                "unassigned_amount": "4",
            },
            "usdgo": {
                "coin": "USDGO",
                "principal": "799",
                "interest": "0",
                "unassigned_amount": ".9",
            },
        },
        "unassigned_offsets": {"TRX": ".0001", "USDGO": ".1"},
        "interest_checked_at": 1792256400000,
    }
    original = deepcopy((state, execution, book.state(), engine.state()))
    result = dashboard(state, engine.state(), execution, book, 1792256430000)
    assert (state, execution, book.state(), engine.state()) == original
    attribution = result["attribution"]
    assert attribution["reconciled"]
    assert abs(money(attribution["residual"])) < money("1e-8")
    rows = {r["coin"]: r for r in result["locations"]}
    assert money(rows["TRX"]["spot_available"]) == money(".00333986")
    assert money(rows["TRX"]["outside_strategy"]) == money("4.0001")
    assert all(r["consistent"] for r in rows.values())
    signals = result["signal"]["sides"]
    assert signals["buy"]["target"] == "0.33126642"
    assert signals["sell"]["target"] == "0.34000164"
    assert money(signals["buy"]["target_fraction"]) == money(".12")
    assert money(signals["sell"]["target_fraction"]) == money(".21")
    assert signals["buy"]["distance_ratio"] == pytest.approx((0.33464 - 0.33126642) / 0.33464)


def test_signal_without_quote_stays_unknown_and_location_mismatch_does_not_unlock():
    book = Ledger(1000)
    engine = Engine(candidate(), book)
    result = dashboard({}, engine.state(), {"holdings": {}}, book, 1)
    assert result["signal"]["sides"]["buy"]["distance_ratio"] is None
    assert result["attribution"] is None
    book.buy("fill", "lot", 1, 100, 0.3)
    book.lots[0].locked = money(100)
    result = dashboard(
        {"markets": {"TRXUSDT": {"bid": ".4", "ask": ".41"}}},
        engine.state(),
        {"holdings": {}},
        book,
        1,
    )
    assert not result["signal"]["principal_location_verified"]
    assert "current_eligible_trx" not in result["signal"]["sides"]["sell"]
    assert book.lots[0].locked == money(100)
