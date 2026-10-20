from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from freqtrade.strategy.trx_phase1.candles import aggregate_minutes, append_closed_minutes
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import Ledger, StateStore
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.runtime import PaperRuntime
from freqtrade.strategy.trx_phase1.treasury import (
    BitgetSavings,
    ProductTerms,
    Treasury,
    cash_switch_quote,
)


def candles(count=600):
    close = 1 + np.sin(np.arange(count) / 15) * 0.1
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=count, freq="1min", tz="UTC"),
            "open": opening,
            "high": np.maximum(opening, close),
            "low": np.minimum(opening, close),
            "close": close,
            "volume": 10.0,
        }
    )


def research(**kwargs):
    return ResearchConfig(
        **{
            "periods": [1, 5],
            "volatility_window": 20,
            "volatility_multiplier": 0,
            "buy_occupation_multiplier": 0,
            "sell_occupation_multiplier": 0,
            "min_notional": 1,
            **kwargs,
        }
    )


def test_prd_worked_example_and_interest():
    book = Ledger(2000)
    for index, price in enumerate([0.3, 0.29, 0.28]):
        book.buy(str(index), str(index), index, 1000, price, price)
    allocations = book.qualifying(0.292, 1500, 0.001)
    assert allocations == [{"lot_id": "2", "quantity": 1000}, {"lot_id": "1", "quantity": 500}]
    sale = book.sell("sale", "sale", 5, 1500, 0.292, allocations, 0.438)
    assert sale["profit"] == pytest.approx(12.137)
    assert [part["profit"] for part in sale["allocations"]] == pytest.approx([11.428, 0.709])
    assert book.cost == Decimal("445.445")
    assert book.equity(0.292) == Decimal("2004.692")
    book.credit_interest("interest", "TRX", 3)
    assert book.equity(0.292) == Decimal("2005.568")
    assert book.deployable == Decimal("2012.137")


def test_buy_fee_in_base_and_actual_loss_are_truthful():
    book = Ledger(100)
    book.buy("b", "b", 1, 10, 1, fee_base=1)
    assert book.principal == 9
    assert book.cost == 10
    sale = book.sell("s", "s", 2, 9, 0.9, [{"lot_id": "b", "quantity": 9}], 0.01)
    assert sale["profit"] == pytest.approx(-1.91)
    assert book.summary(0.9)["total_profit"] == pytest.approx(-1.91)


def test_skip_unprofitable_latest_lot_and_never_sell_interest():
    book = Ledger(100)
    book.buy("a", "a", 1, 10, 1)
    book.buy("b", "b", 2, 10, 2)
    book.credit_interest("i", "TRX", 20)
    assert book.qualifying(1.5, 100, 0.001) == [{"lot_id": "a", "quantity": 10}]
    before = book.state()
    with pytest.raises(ValueError):
        book.sell("s", "s", 3, 11, 1.5, [{"lot_id": "a", "quantity": 10}])
    assert book.state() == before


def test_fill_idempotency_and_base_sell_fee():
    book = Ledger(100)
    book.buy("a", "a", 1, 10, 1)
    assert book.buy("a", "a", 1, 10, 1) is None
    sale = book.sell("s", "s", 2, 9, 2, [{"lot_id": "a", "quantity": 10}], fee_base=1)
    assert book.principal == 0
    assert sale["profit"] == 8
    assert book.sell("s", "s", 2, 9, 2, []) is None


def test_partial_decision_vwap_cancel_and_restart(tmp_path):
    engine = Engine(research(startup="small_buy", initial_buy_fraction=0.1))
    engine.on_close(60000, 1, 0)
    anchor = engine.anchor
    plan = engine.pending
    engine.fill("a", "order", 60000, 20, 0.99, final=False)
    assert engine.anchor == anchor
    store = StateStore(tmp_path / "state.sqlite")
    version = store.save("session", 0, engine.state())
    loaded_version, state = store.load("session")
    restored = Engine.restore(state)
    assert restored.pending.plan_id == plan.plan_id
    restored.fill("b", "order", 60001, 30, 1.01, final=False)
    restored.finish(60002)
    assert restored.anchor == pytest.approx((20 * 0.99 + 30 * 1.01) / 50)
    assert restored.book.principal == 50
    assert restored.book.lots[0].unit_cost == pytest.approx(Decimal("1.002"))
    assert store.save("session", loaded_version, restored.state()) == 2
    with pytest.raises(RuntimeError, match="Concurrent"):
        store.save("session", version, engine.state())


def test_zero_fill_does_not_update_anchor_and_old_signal_expires():
    engine = Engine(research(startup="small_buy"))
    engine.on_close(60000, 1, 0)
    engine.on_close(120000, 0.99, 0)
    assert engine.anchor == 1 and engine.anchor_version == 0
    assert engine.pending is None
    assert engine.reason == "signal_expired"


def test_duplicate_close_and_missing_minutes():
    engine = Engine(research(startup="small_buy"))
    plan = engine.on_close(60000, 1, 0)
    assert engine.on_close(60000, 0.5, 0) is plan
    assert engine.anchor == 1
    with pytest.raises(ValueError, match="gap"):
        engine.on_close(180000, 0.5, 0)
    assert engine.paused


def test_execution_rechecks_price_and_cash():
    engine = Engine(research(startup="small_buy"))
    engine.on_close(60000, 1, 0)
    assert engine.executable(60000, 1)
    assert not engine.executable(120000, 1)
    assert not engine.executable(60000, 1.5)
    assert engine.status()["reserved_cash"] > 0


def test_repriced_partial_buy_keeps_original_budget():
    engine = Engine(research(startup="small_buy", initial_buy_fraction=0.1))
    engine.on_close(60000, 1, 0)
    budget = engine.pending.budget
    quantity = engine.execution_quantity(1.009)
    assert quantity * 1.009 * 1.001 <= budget
    assert quantity < engine.pending.quantity
    engine.fill("a", "a", 60000, 20, 1.009, 20 * 1.009 * 0.001, final=False)
    remaining = budget - engine.pending.filled_cash
    assert engine.execution_quantity(1.009) * 1.009 * 1.001 <= remaining
    assert engine.status()["reserved_cash"] == pytest.approx(remaining)


def test_cash_flows_do_not_create_investment_returns():
    book = Ledger(100)
    book.buy("b", "b", 1, 10, 1)
    assert book.summary(2)["return_ratio"] == pytest.approx(0.1)
    book.external_flow("deposit", 110, 2)
    assert book.summary(2)["total_profit"] == 10
    assert book.summary(2)["return_ratio"] == pytest.approx(0.1)
    book.external_flow("withdraw", -55, 2)
    restored = Ledger.restore(book.state())
    assert restored.summary(2)["return_ratio"] == pytest.approx(0.1)
    assert restored.summary(3)["total_profit"] == 20
    assert restored.summary(3)["return_ratio"] == pytest.approx(175 / 150 - 1)
    assert Ledger(100).summary(1)["return_ratio"] == 0
    with pytest.raises(ValueError, match="positive equity"):
        Ledger(100).external_flow("all", -100, 1)


def test_partial_fills_are_one_order_with_complete_profit():
    book = Ledger(100)
    book.buy("b1", "b", 1, 5, 1, 0.005)
    book.buy("b2", "b", 2, 5, 1.2, 0.006)
    book.sell("s1", "s", 3, 4, 2, [{"lot_id": "b", "quantity": 4}], 0.008)
    book.sell("s2", "s", 4, 6, 2.1, [{"lot_id": "b", "quantity": 6}], 0.0126)
    orders = book.order_summary()
    assert len(orders) == 2
    assert orders[0]["fill_count"] == 2
    assert orders[0]["price"] == pytest.approx(1.1)
    assert orders[1]["profit"] == pytest.approx(float(book.realized))
    assert orders[1]["allocations"][0]["quantity"] == 10
    assert book.cost == 0 and book.principal == 0


def test_paper_api_restores_saved_portfolio_without_starting(tmp_path, mocker):
    from freqtrade.rpc.api_server import api_trx

    engine = Engine(research())
    engine.on_close(60000, 1, 0)
    engine.book.buy("b", "b", 60000, 10, 1, 0.01)
    StateStore(tmp_path / "trx_phase1/paper.sqlite").save("paper", 0, engine.state())
    mocker.patch.object(api_trx, "_runtime", None)
    status = api_trx.paper_status({"user_data_dir": tmp_path})
    assert status["running"] is False
    assert status["summary"]["principal_trx"] == 10
    assert status["summary"]["cash"] == pytest.approx(989.99)
    assert status["config"] == engine.config.model_dump()


def test_paper_restart_skips_offline_fills_and_preserves_portfolio(tmp_path):
    data = candles(120)
    data["date"] = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1),
        periods=120,
        freq="1min",
    )
    path = tmp_path / "data.feather"
    data.to_feather(path)
    exchange = SimpleNamespace(update=lambda existing, now: existing)
    config = research(startup="small_buy")
    runtime = PaperRuntime(tmp_path, path, exchange, config)
    old_close = int(data.date.iloc[-6].timestamp() * 1000) + 60000
    runtime.engine.on_close(old_close, 1, 0)
    runtime.save()
    restored = PaperRuntime(tmp_path, path, exchange, config)
    restored.poll()
    assert restored.engine.pending is None
    assert restored.engine.book.principal == 0
    assert restored.engine.book.cash == 1000
    assert restored.engine.anchor == 1
    assert restored.engine.reason == "recovered_data_waiting_new_signal"
    restored.stop()
    resumed = PaperRuntime(tmp_path, path, exchange, config)
    assert resumed.engine.book.state() == restored.engine.book.state()


def test_aggregate_close_boundaries_and_no_future_bar():
    data = candles(12)
    result = aggregate_minutes(data, 5, pd.Timestamp("2026-01-01T00:09:00Z"))
    assert len(result) == 1
    assert result.close_time.iloc[0] == pd.Timestamp("2026-01-01T00:05:00Z")
    assert result.close.iloc[0] == data.close.iloc[4]
    assert result.volume.iloc[0] == 50
    with pytest.raises(ValueError, match="gap"):
        aggregate_minutes(data.drop(index=3), 5)


def test_append_retains_only_finalized_and_rejects_revisions():
    data = candles(10)
    combined = append_closed_minutes(data.iloc[:5], data.iloc[4:], data.date.iloc[8])
    assert len(combined) == 8
    revised = data.iloc[4:].copy()
    revised.loc[4, "close"] += 0.01
    with pytest.raises(ValueError, match="revised"):
        append_closed_minutes(data.iloc[:5], revised, data.date.iloc[9])


@pytest.mark.parametrize("variant", ["fixed", "dynamic_size", "dynamic_period"])
@pytest.mark.parametrize("rebate_ratio", [0, 0.5])
def test_replay_future_prefix_and_conservation(variant, rebate_ratio):
    config = research(
        variant=variant, startup="small_buy", execution_margin=0, fee_rebate_ratio=rebate_ratio
    )
    data = candles(1000)
    full = replay(data, config)
    prefix = replay(data.iloc[:600], config)
    end = int((data.date.iloc[599] + pd.Timedelta("1min")).timestamp() * 1000)
    # Later dated refunds can revise reported net costs, never past executions.
    assert [fill for fill in full["fills"] if fill["timestamp"] < end] == prefix["fills"]
    if not rebate_ratio:
        assert [order for order in full["orders"] if order["timestamp"] < end] == prefix["orders"]
    assert full["orders"]
    for sale in (row for row in full["orders"] if row["side"] == "sell"):
        assert all(allocation["profit"] > 0 for allocation in sale["allocations"])
    summary = full["summary"]
    assert summary["equity"] == pytest.approx(
        summary["cash"] + summary["principal_trx"] * data.close.iloc[-1]
    )
    assert summary["total_profit"] == pytest.approx(
        summary["realized_profit"] + summary["unrealized_profit"]
    )
    assert all(order["timestamp"] < end for order in prefix["orders"])


def test_daily_refund_is_not_spendable_before_credit_and_does_not_change_anchor():
    config = research(startup="small_buy", buy_fee=0.01)
    engine = Engine(config)
    engine.on_close(60000, 1, 0)
    engine.fill("b", "b", 60000, 10, 1, 0.1)
    due = config.rebate_due_at(60000)
    assert due == 16 * 3600000  # Next midnight in UTC+8.
    assert config.rebate_due_at(due) == due + 86400000
    before = engine.book.summary(1)
    assert before["pending_fee_rebates"] == pytest.approx(0.05)
    assert before["fees"] == before["gross_fees"] == 0.1
    assert engine.book.settle_fee_rebates(due - 1) == 0
    assert engine.book.cash == Decimal("989.9")
    engine.book.settle_fee_rebates(due)
    assert engine.book.cash == Decimal("989.95")
    assert engine.book.cost == Decimal("10.05")
    assert engine.book.summary(1)["fees"] == 0.05
    assert engine.anchor == 1 and engine.anchor_version == 1
    assert engine.book.settle_fee_rebates(due) == 0


def test_buy_refund_allocates_between_sold_cost_and_held_cost():
    book = Ledger(200)
    book.buy("b", "b", 1, 10, 10, 1)
    book.sell("s", "s", 2, 5, 11, [{"lot_id": "b", "quantity": 5}], 0.55)
    book.queue_fee_rebate("b", 0.5, 100)
    book.queue_fee_rebate("s", 0.275, 100)
    raw = deepcopy(book.orders)
    book.settle_fee_rebates(100)
    assert book.orders == raw
    assert book.cash == Decimal("154.225")
    assert book.cost == Decimal("50.25")
    assert book.realized == Decimal("4.475")
    orders = book.order_summary()
    sale = orders[1]
    assert sale["cost"] == pytest.approx(50.25)
    assert sale["allocations"][0]["net_proceeds"] == pytest.approx(54.725)
    assert sale["profit"] == pytest.approx(4.475)
    assert book.summary(11)["total_profit"] == pytest.approx(9.225)
    assert book.summary(11)["fees"] == pytest.approx(0.775)


def test_partial_fill_refunds_survive_restart_and_are_not_counted_twice():
    book = Ledger(100)
    book.buy("b1", "b", 1, 5, 1, 0.05)
    book.queue_fee_rebate("b1", 0.025, 10)
    book.settle_fee_rebates(10)
    book.buy("b2", "b", 11, 5, 2, 0.1)
    book.queue_fee_rebate("b2", 0.05, 20)
    restored = Ledger.restore(book.state())
    restored.settle_fee_rebates(20)
    restored.settle_fee_rebates(20)
    assert restored.cash == Decimal("84.925")
    assert restored.cost == Decimal("15.075")
    assert restored.fee_rebates == Decimal("0.075")
    order = restored.order_summary()[0]
    assert order["rebate_pending"] == pytest.approx(0)
    assert order["net_fee"] == pytest.approx(0.075)
    assert order["fill_count"] == 2


def test_fee_paid_in_base_refunds_usdt_without_inventing_trx():
    book = Ledger(1000)
    book.buy("b", "b", 1, 100, 1, fee_base=1)
    book.credit_interest("i", "TRX", 3)
    book.queue_fee_rebate("b", 0.5, 2)
    book.settle_fee_rebates(2)
    assert book.principal == 99 and book.interest_trx == 3
    assert book.cash == Decimal("900.5")
    assert book.cost == Decimal("99.5")
    assert book.summary(1)["fees"] == 0.5


def test_unpaid_refund_cannot_qualify_an_unprofitable_sale():
    book = Ledger(100)
    book.buy("b", "b", 1, 1, 1, 0.01)
    book.queue_fee_rebate("b", 0.005, 100)
    assert book.qualifying(1.02, 1, 0.01) == []
    book.settle_fee_rebates(100)
    assert book.qualifying(1.015, 1, 0.01) == []
    assert book.qualifying(1.02, 1, 0.01) == [{"lot_id": "b", "quantity": 1}]


def test_legacy_state_does_not_invent_rebates():
    engine = Engine(research())
    state = engine.state()
    state["config"]["version"] = "trx-phase1-v0.1"
    for name in ("fee_rebate_ratio", "rebate_settlement_hour", "rebate_utc_offset_minutes"):
        state["config"].pop(name)
    for name in ("fee_rebates", "pending_rebates", "rebate_receipts"):
        state["book"].pop(name)
    old = Engine.restore(state)
    assert old.config.fee_rebate_ratio == 0
    assert old.book.fee_rebates == 0
    assert ResearchConfig().fee_rebate_ratio == 0.5


def test_final_day_refunds_are_not_credited_early():
    result = replay(candles(22), research(startup="small_buy"))
    assert result["summary"]["pending_fee_rebates"] > 0
    assert result["summary"]["fee_rebates"] == 0
    assert result["summary"]["fees"] == result["summary"]["gross_fees"]
    assert result["rebate_receipts"] == []


def test_end_positions_are_not_force_liquidated():
    result = replay(candles(22), research(startup="small_buy"))
    assert result["summary"]["principal_trx"] > 0
    assert result["summary"]["sell_orders"] == 0


@pytest.mark.parametrize(
    "parameters",
    [
        {"initial_cash": float("nan")},
        {"periods": [5, 5]},
        {"periods": [7]},
        {"minimum_threshold": 0.5},
    ],
)
def test_invalid_research_configuration(parameters):
    with pytest.raises(ValidationError):
        ResearchConfig(**parameters)


def terms(coin="USDT", rate=0.05):
    return ProductTerms("verified-fixture", coin, True, 1, 10000, 10000, 0, 0, ((10000, rate),))


def test_earn_locks_do_not_double_count_and_arrival_is_required():
    engine = Engine(research())
    treasury = Treasury(engine)
    treasury.lock_subscription("sub", terms(), 500)
    assert engine.book.equity(1) == 1000
    assert engine.book.free_cash == 500
    treasury.request_redemption("r", "USDT", 100)
    with pytest.raises(ValueError, match="arrival"):
        treasury.confirm_arrival("r", "arr", 100)
    assert engine.book.free_cash == 500
    treasury.confirm_arrival("r", "arr", 100, wallet_verified=True)
    treasury.confirm_arrival("r", "arr", 100, wallet_verified=True)
    assert engine.book.free_cash == 600
    assert engine.book.equity(1) == 1000


def test_principal_opportunity_interrupts_treasury_and_interest_is_credited_once():
    engine = Engine(research(startup="small_buy"))
    treasury = Treasury(engine)
    engine.on_close(60000, 1, 0)
    with pytest.raises(ValueError, match="priority"):
        treasury.lock_subscription("sub", terms(), 500)
    rows = [
        {
            "orderType": "pay_interest",
            "orderId": "unique",
            "settleCoinName": "TRX",
            "amount": "3",
            "ts": "60000",
        }
    ]
    treasury.import_credited_interest(rows)
    treasury.import_credited_interest(rows)
    assert engine.book.interest_trx == 3
    assert engine.book.deployable == 1000


def test_cash_switch_uses_tiers_spread_and_opportunity_cost():
    proposal = cash_switch_quote(
        1000, 24, 0.999, 1.001, 0.001, terms(rate=0.01), terms("USDGO", 0.05)
    )
    assert proposal["expected_net_benefit"] < 0 and not proposal["eligible"]
    high = cash_switch_quote(
        1000, 24 * 30, 0.999, 1.001, 0.001, terms(rate=0.01), terms("USDGO", 0.1)
    )
    assert high["eligible"]
    assert not cash_switch_quote(1000, 24, 0.95, 0.96, 0.001, terms(), terms("USDGO"))["eligible"]
    assert not cash_switch_quote(
        1000, 24, 1, 1, 0, terms(), terms("USDGO"), principal_pending=True
    )["eligible"]


def test_real_earn_adapter_cannot_write_without_activation():
    adapter = BitgetSavings(None)
    with pytest.raises(PermissionError):
        adapter.subscribe("product", 100)
    with pytest.raises(PermissionError):
        adapter.redeem("product", "asset", 100)
