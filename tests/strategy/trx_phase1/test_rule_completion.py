from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from freqtrade.strategy.trx_phase1.config import MECHANISMS, ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.research_earn import HOUR, HistoricalEarn
from freqtrade.strategy.trx_phase1.runtime import PaperRuntime
from freqtrade.strategy.trx_phase1.studies import marginal_effects, run_study
from tests.strategy.trx_phase1.test_phase1 import candles, research


def session(**options):
    config = research(earn_model="net_benefit_hourly", cash_min_hold_minutes=0, **options)
    engine = Engine(config)
    engine.earn = HistoricalEarn(engine.book, config, 0)
    engine.prepare_cash = engine.earn.prepare_cash
    engine.earn.sweep_cash(0)
    return engine


def test_all_eight_mechanisms_are_independent_and_complete():
    assert set(MECHANISMS.values()) == {
        (a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)
    }
    for variant, mask in MECHANISMS.items():
        engine = Engine(
            research(variant=variant, volatility_multiplier=0.5, buy_occupation_multiplier=0.02)
        )
        engine.book.buy("b", "b", 0, 600, 1)
        engine.schedule(60000, 1, 0.03)
        assert engine.buy_period > 1 if mask[2] else engine.buy_period == 1
        assert (
            engine.threshold("buy", 0.03, 0.6, 1) == engine.config.buy_threshold
            if not mask[0]
            else engine.threshold("buy", 0.03, 0.6, 1) >= engine.config.minimum_threshold
        )


def test_rebound_strength_is_bounded_and_only_affects_dynamic_quantities():
    def target(variant, price):
        engine = Engine(research(variant=variant, sell_fraction=0.1, sell_rebound_multiplier=0.2))
        engine.book.buy("b", "b", 0, 100, 0.5)
        engine.anchor = 1
        return engine.make_plan("sell", 60000, price, 0.02, 120000).quantity

    assert target("dynamic_quantity", 1.04) > target("dynamic_quantity", 1.02)
    assert target("dynamic_quantity", 1.5) <= 100
    assert target("fixed", 1.04) == target("fixed", 1.02) == 10
    assert target("dynamic_threshold", 1.04) == 10


def test_cash_choice_accounts_for_costs_alternative_income_and_eligibility():
    good = session()
    assert good.book.usdgo == 1000 and good.book.cash == 0
    for options, reason in [
        ({"fx_fee": 0.01}, "cash_cost_exceeds_benefit"),
        ({"usdt_apr": 0.13}, "cash_cost_exceeds_benefit"),
        ({"fx_quote_available": False}, "usdgo_quote_missing"),
        ({"fx_rate": 0.97}, "usdgo_price_deviation"),
        ({"usdgo_product_available": False}, "usdgo_product_unavailable"),
    ]:
        engine = session(**options)
        assert engine.book.cash == 1000 and engine.book.usdgo == 0
        assert engine.earn.policy.reason == reason
    capped = session(usdgo_earn_cap=200)
    assert capped.book.usdgo == 200 and capped.book.cash == 800


def test_delayed_funding_is_not_spendable_and_expiry_releases_it_without_conversion():
    engine = session(redemption_delay_minutes=10)
    engine.anchor = 1
    plan = engine.make_plan("buy", 60000, 0.98, 0.01, 5 * 60000)
    assert plan and plan.ready_at == 11 * 60000
    assert engine.book.free_cash == 0 and engine.book.locked["USDGO"] > 0
    assert not engine.executable(120000, 0.98)
    engine.earn.advance(5 * 60000, 0.98)
    engine.finish(5 * 60000)
    assert engine.book.locked["USDGO"] == 0 and engine.book.cash == 0
    assert all(row["side"] != "to_usdt" for row in engine.earn.conversions)
    assert engine.book.equity(0.98) == 1000


def test_arrived_funding_can_execute_and_cash_hold_never_blocks_a_valid_purchase():
    engine = session(redemption_delay_minutes=2)
    engine.anchor = 1
    plan = engine.make_plan("buy", 60000, 0.98, 0.01, 10 * 60000)
    assert plan
    engine.earn.advance(3 * 60000, 0.98)
    assert engine.book.cash > 0 and engine.book.locked["USDGO"] == 0
    assert engine.executable(3 * 60000, 0.98)
    engine.fill("fill", plan.plan_id, 3 * 60000, engine.execution_quantity(0.98), 0.98)
    assert engine.book.principal > 0


def test_deviation_marks_existing_assets_and_does_not_force_a_loss_exchange():
    engine = session(fx_shock_after_minutes=120, fx_shock_rate=0.97)
    engine.earn.advance(2 * HOUR, 1)
    assert engine.book.usdgo_mark == Decimal("0.97")
    assert engine.book.equity(1) < 1000
    before = engine.book.usdgo
    assert engine.earn.prepare_cash("request", 2 * HOUR, 100) == 0
    assert engine.book.usdgo == before


def test_usdt_earns_when_usdgo_has_no_net_advantage_and_trx_unavailable_does_not_block_buys():
    engine = session(usdt_apr=0.13, trx_product_available=False)
    engine.earn.advance(HOUR, 1)
    engine.earn.advance(2 * HOUR, 1)
    assert engine.book.stable_interest > 0 and engine.book.cash > 1000
    engine.anchor = 1
    assert engine.make_plan("buy", 2 * HOUR, 0.98, 0.01, 2 * HOUR + 60000)


def test_earn_and_pending_arrivals_restore_exactly_without_duplicate_credits():
    engine = session(redemption_delay_minutes=2)
    engine.anchor = 1
    engine.make_plan("buy", 60000, 0.98, 0.01, 10 * 60000)
    restored = Engine.restore(deepcopy(engine.state()))
    for timestamp in (3 * 60000, HOUR, 2 * HOUR, 3 * HOUR):
        engine.earn.advance(timestamp, 1)
        restored.earn.advance(timestamp, 1)
    assert restored.state() == engine.state()
    before = deepcopy(restored.state())
    restored.earn.advance(3 * HOUR, 1)
    assert restored.state() == before


def test_closed_cash_requests_stay_auditable_and_restart_does_not_repeat_arrivals():
    engine = session(redemption_delay_minutes=2)
    policy = engine.earn.policy
    policy.prepare_cash("expired", 0, 100, expires=60000)
    policy.advance(60000)
    policy.prepare_cash("active", 60000, 100, expires=10 * 60000)
    restored = Engine.restore(deepcopy(engine.state()))
    recovered = restored.earn.policy
    assert recovered.requests["expired"]["status"] == "canceled_before_arrival"
    recovered.advance(3 * 60000)
    assert recovered.requests["active"]["status"] == "arrived"
    before = deepcopy(restored.state())
    recovered.advance(3 * 60000)
    assert restored.state() == before
    assert recovered.requests["expired"]["status"] == "canceled_before_arrival"
    assert restored.book.locked["USDGO"] == 0


def test_stopping_additions_keeps_profitable_sells_and_manual_exits_are_separate():
    engine = Engine(research(variant="fixed", sell_fee=0))
    engine.book.buy("b", "b", 0, 100, 1)
    engine.book.credit_interest("i", "TRX", 5)
    engine.anchor = 1
    engine.accept_new_buys = False
    engine.on_close(60000, 1, 0)
    assert engine.on_close(120000, 0.9, 0) is None
    with pytest.raises(ValueError, match="explicit loss"):
        engine.manual_exit("x", 180000, "principal", 20, 0.9)
    engine.manual_exit("x", 180000, "principal", 20, 0.9, True)
    assert engine.book.realized == -2 and engine.book.interest_trx == 5
    engine.manual_exit("y", 180000, "interest", 2, 0.9, True)
    assert engine.book.interest_trx == 3 and engine.book.interest_sale_proceeds == Decimal("1.8")
    assert engine.book.realized == -2 and engine.anchor == 1
    assert [row["kind"] for row in engine.book.order_summary()[1:]] == [
        "manual_principal_exit",
        "manual_interest_exit",
    ]


def test_interest_exit_refund_does_not_create_principal_profit():
    engine = Engine(research(sell_fee=0.01))
    engine.book.credit_interest("i", "TRX", 10)
    engine.manual_exit("sale", 60000, "interest", 5, 2, True)
    engine.book.settle_fee_rebates(engine.config.rebate_due_at(60000))
    assert engine.book.realized == 0
    assert engine.book.interest_sale_proceeds == Decimal("9.95")
    assert engine.book.order_summary()[0]["profit"] is None


@pytest.mark.parametrize("source", ["principal", "interest"])
def test_manual_preparation_preserves_assets_waits_for_arrival_and_can_be_canceled(source):
    engine = session(redemption_delay_minutes=60)
    engine.earn.prepare_cash("initial", 0, 100)
    engine.earn.advance(HOUR, 1)
    engine.book.buy("b", "b", HOUR, 50, 1)
    engine.book.credit_interest("i", "TRX", 10)
    engine.earn.retained_waiting.append((2 * HOUR, Decimal(10)))
    before = engine.book.equity(1)
    engine.prepare_manual_exit("manual", 2 * HOUR, source, 5)
    assert engine.book.equity(1) == before
    restored = Engine.restore(deepcopy(engine.state()))
    with pytest.raises(ValueError, match="not arrived"):
        restored.manual_exit("manual", 2 * HOUR + 60000, source, 5, 0.9, True)
    restored.cancel_manual_exit(2 * HOUR + 60000)
    assert restored.book.equity(1) == before
    assert restored.book.locked["TRX_INTEREST"] == 0
    assert all(lot.locked == 0 for lot in restored.book.lots)
    engine.manual_exit("manual", 3 * HOUR, source, 5, 0.9, True)
    assert engine.manual_preparation is None
    if source == "principal":
        assert engine.book.realized < 0 and engine.book.interest_trx == 10
    else:
        assert engine.book.principal == 50 and engine.book.interest_trx == 5
    committed = deepcopy(engine.state())
    engine.prepare_manual_exit("manual", 4 * HOUR, source, 5)
    engine.manual_exit("manual", 4 * HOUR, source, 5, 0.9, True)
    assert engine.state() == committed


@pytest.mark.parametrize("variant", list(MECHANISMS))
def test_new_cash_policy_replay_is_causal_and_conserves_nav(variant):
    config = research(
        variant=variant,
        earn_model="net_benefit_hourly",
        startup="small_buy",
        fx_fee=0.00002,
        fx_spread=0.00002,
    )
    data = candles(1000)
    full, prefix = replay(data, config), replay(data.iloc[:600], config)
    end = int((data.date.iloc[599] + pd.Timedelta("1min")).timestamp() * 1000)
    assert [row for row in full["fills"] if row["timestamp"] < end] == prefix["fills"]
    assert [row for row in full["interest_receipts"] if row["timestamp"] <= end] == prefix[
        "interest_receipts"
    ]
    s = full["summary"]
    assert s["total_profit"] == pytest.approx(
        s["realized_profit"]
        + s["unrealized_profit"]
        + s["stable_interest"]
        + s["fx_profit"]
        + s["usdgo_unrealized_profit"]
        + s["trx_interest_market_value"]
    )
    assert 0 <= s["mean_trx_exposure"] <= 1


def test_paper_earn_restart_and_configuration_preserve_assets(tmp_path):
    data = candles(120)
    data["date"] = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1),
        periods=120,
        freq="1min",
    )
    path = tmp_path / "data.feather"
    data.to_feather(path)
    exchange = SimpleNamespace(update=lambda existing, now: existing)
    config = research(earn_model="net_benefit_hourly")
    runtime = PaperRuntime(tmp_path, path, exchange, config)
    runtime.poll()
    runtime.save()
    saved = runtime.engine.state()
    restored = PaperRuntime(tmp_path, path, exchange, config)
    assert restored.engine.state() == saved
    restored.poll()
    assert restored.engine.book.equity(restored.engine.last_price) == runtime.engine.book.equity(
        runtime.engine.last_price
    )
    with pytest.raises(ValueError, match="initial capital"):
        restored.configure(config.model_copy(update={"initial_cash": 2000}))


def test_factorial_effects_compare_one_toggle_at_a_time_and_study_freezes_parameters(tmp_path):
    entries = [
        {
            "variant": variant,
            "mechanisms": ResearchConfig(variant=variant).mechanisms,
            "summary": {
                "total_profit": 10 * mask[0] + 20 * mask[1] + 30 * mask[2],
                "return_ratio": (10 * mask[0] + 20 * mask[1] + 30 * mask[2]) / 1000,
                "max_drawdown": 0.1,
            },
        }
        for variant, mask in MECHANISMS.items()
    ]
    effects = marginal_effects(entries)
    assert [row["mean_profit_change"] for row in effects] == [10, 20, 30]
    assert all(len(row["pairs"]) == 4 for row in effects)
    with pytest.raises(ValueError, match="all eight"):
        marginal_effects(entries[:-1])
    result = run_study(candles(180), research(earn_model="net_benefit_hourly"), tmp_path)
    assert len(result["variants"]) == 8 and len(result["stress"]) == 6
    assert result["windows"] == []
    assert (
        result["protocol"]["coefficient_search"] is False
        and result["protocol"]["known_history"] is True
    )
    assert (tmp_path / result["study_id"] / "manifest.json").exists()
