from copy import deepcopy

import pytest

from freqtrade.rpc.api_server.api_trx import assert_reproduced
from freqtrade.strategy.trx_phase1.funding_preparation import (
    FundingFactory,
    FundingTerms,
    PreparedEngine,
)
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.research_earn import HOUR, HistoricalEarn
from tests.strategy.trx_phase1.test_diagnostics import bars, config


def setup(**changes):
    cfg = config(
        variant="fixed",
        periods=[5],
        execution_preflight=True,
        usdt_redemption_delay_minutes=60,
        redemption_delay_minutes=1,
        trx_redemption_delay_minutes=60,
        usdt_apr=0.05,
        usdgo_apr=0,
        slippage=0,
        tail_completion=False,
    )
    engine, earn = FundingFactory({"enabled": True, **changes})(cfg, 0)
    engine.anchor = 1
    engine.last_price = 0.8
    engine.last_sigma = 0.001
    earn.sweep_cash(0)
    earn.advance(2 * HOUR, 0.8)
    return engine, earn


def test_buy_intent_survives_signal_expires_and_needs_fresh_confirmation():
    engine, earn = setup()
    now = 2 * HOUR
    assert engine.make_plan("buy", now, 0.8, 0.012, now + 5 * 60000) is None
    intent = deepcopy(engine.preparation.intents["buy"])
    budget = intent["budget_cap"]
    assert budget == pytest.approx(float(engine.book.deployable) * engine.config.buy_fraction)
    assert engine.pending is None and engine.book.free_cash < engine.book.cash
    clone = PreparedEngine.restore(engine.state())
    for instance in (engine, clone):
        instance.earn.advance(now + 59 * 60000, 0.8)
        assert instance.preparation.ready_cash == 0 and instance.pending is None
        instance.earn.advance(now + HOUR, 0.8)
        assert instance.preparation.ready_cash == pytest.approx(budget)
        assert instance.pending is None and not instance.book.orders
        assert instance.earn.policy.requests[intent["id"]]["status"] == "arrived"
    assert engine.state() == clone.state()
    fresh_at = now + HOUR + 5 * 60000
    plan = engine.make_plan("buy", fresh_at, 0.8, 0.012, fresh_at + 5 * 60000)
    assert plan and plan.plan_id != intent["source_signal_id"]
    assert plan.created_at > intent["source_signal_expires_at"]
    assert engine.executable(fresh_at, 0.8)
    assert not engine.executable(fresh_at, 0.999)
    engine.fill(
        plan.plan_id, plan.plan_id, fresh_at, plan.quantity, 0.8, plan.quantity * 0.8 * 0.001
    )
    assert engine.book.principal > 0 and engine.book.interest_trx == 0
    assert all(row["status"] != "canceled_before_arrival" for row in earn.policy.requests.values())


def test_two_sided_arrival_cannot_sell_a_lot_that_lost_profit_or_use_interest():
    engine, earn = setup()
    # Use uninvested capital to establish a principal lot, without inventing an arrival.
    engine.book.locked["USDT"] = 0
    engine.book.buy("principal", "principal", 0, 100, 0.8)
    earn.sync_usdt(2 * HOUR)
    engine.book.credit_interest("retained", "TRX", 50)
    engine.anchor = 0.5
    now = 2 * HOUR
    assert engine.make_plan("sell", now, 1, 0.012, now + 5 * 60000) is None
    assert engine.book.lots[0].locked == 30
    state = engine.state()
    clone = PreparedEngine.restore(state)
    assert clone.book.interest_trx == 50 and clone.book.lots[0].locked == 30
    earn.advance(now + HOUR - 1, 1)
    assert engine.preparation.ready_trx == 0
    earn.advance(now + HOUR, 1)
    assert engine.preparation.ready_trx == 30 and engine.book.lots[0].locked == 0
    assert engine.make_plan("sell", now + HOUR, 0.7, 0.012, now + HOUR + 5 * 60000) is None
    assert not [row for row in engine.book.orders if row["side"] == "sell"]
    engine.last_price = 1
    plan = engine.make_plan("sell", now + HOUR + 5 * 60000, 1, 0.012, now + HOUR + 10 * 60000)
    assert plan and plan.quantity <= 30
    retained = engine.book.interest_trx
    engine.fill(
        plan.plan_id, plan.plan_id, plan.created_at, plan.quantity, 1, plan.quantity * 0.001
    )
    assert engine.book.interest_trx == retained
    assert engine.book.orders[-1]["profit"] > 0
    assert all(part["lot_id"] == "principal" for part in engine.book.orders[-1]["allocations"])


def test_idle_funding_returns_to_earn_and_cooldown_prevents_churn():
    engine, earn = setup(ready_hold_minutes=5)
    now = 2 * HOUR
    engine.make_plan("buy", now, 0.8, 0.012, now + 5 * 60000)
    budget = engine.preparation.intents["buy"]["budget_cap"]
    earn.advance(now + HOUR, 0.8)
    invested = float(earn.usdt_eligible)
    assert engine.preparation.ready_cash == pytest.approx(budget)
    earn.advance(now + HOUR + 5 * 60000, 0.8)
    assert engine.preparation.intents["buy"] is None
    assert float(earn.usdt_eligible) == invested
    assert earn.usdt_waiting
    assert (
        engine.make_plan("buy", now + HOUR + 10 * 60000, 0.8, 0.012, now + HOUR + 15 * 60000)
        is None
    )
    assert len(earn.policy.requests) == 1


def test_disabled_and_fast_preparation_reproduce_unchanged_economics(tmp_path):
    frame = bars(800)
    cfg = config(periods=[5, 15, 60], usdt_redemption_delay_minutes=1, redemption_delay_minutes=1)
    original = replay(frame, cfg)
    for enabled in (False, True):
        actual = replay(frame, cfg, engine_factory=FundingFactory({"enabled": enabled}))
        assert_reproduced(original, actual)
    with pytest.raises(ValueError, match="cannot change trading"):
        FundingTerms(environment_changes={"buy_fraction": 0.5})


def test_environment_transition_preserves_all_existing_assets_and_requests():
    engine, earn = setup(switch_at=3 * HOUR, environment_changes={"trx_earn_cap": 50})
    engine.book.buy("held", "held", 0, 10, 1)
    engine.make_plan("buy", 2 * HOUR, 0.8, 0.012, 2 * HOUR + 5 * 60000)
    before, anchor = engine.book.state(), engine.anchor
    engine.before_minute(3 * HOUR)
    assert engine.book.state() == before and engine.anchor == anchor
    assert engine.transition["principal_trx"] == 10 and engine.config.trx_earn_cap == 50
    assert engine.preparation.ready_cash == 0
    assert engine.transition["earn_state"]["policy"]["requests"]
    earn.advance(3 * HOUR, 0.8)
    assert engine.preparation.ready_cash > 0


def test_usdgo_funding_stays_locked_when_arrival_quote_is_unavailable():
    engine, earn = setup()
    engine.book.convert("idle-usdgo", "to_usdgo", 1000, 1000, 1)
    earn.subscribe_usdgo(1000, 2 * HOUR)
    earn.sync_usdt(2 * HOUR)
    engine.config = engine.config.model_copy(update={"redemption_delay_minutes": 60})
    earn.config = engine.config
    engine.make_plan("buy", 2 * HOUR, 0.8, 0.012, 2 * HOUR + 5 * 60000)
    assert engine.book.locked["USDGO"] > 0
    engine.config = engine.config.model_copy(update={"fx_quote_available": False})
    earn.config = engine.config
    clone = PreparedEngine.restore(engine.state())
    for instance in (engine, clone):
        instance.earn.advance(3 * HOUR, 0.8)
        assert instance.preparation.ready_cash == 0
        assert instance.book.locked["USDGO"] > 0
        instance.config = instance.config.model_copy(update={"fx_quote_available": True})
        instance.earn.config = instance.config
        instance.earn.advance(3 * HOUR + 60000, 0.8)
        assert instance.book.locked["USDGO"] == 0
        assert instance.preparation.ready_cash > 100
        assert not instance.book.orders
    assert engine.state() == clone.state()


def test_pruning_completed_cohorts_preserves_all_economics(monkeypatch):
    frame = bars(2400)
    cfg = config(variant="dynamic_quantity", periods=[5], redemption_delay_minutes=60)
    pruned = replay(frame, cfg)
    monkeypatch.setattr(HistoricalEarn, "prune_completed_redemptions", lambda *_: None)
    original = replay(frame, cfg)
    assert_reproduced(original, pruned)
    assert original["interest_receipts"] == pruned["interest_receipts"]
