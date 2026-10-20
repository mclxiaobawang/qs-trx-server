import numpy as np
import pandas as pd
import pytest

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import Ledger
from freqtrade.strategy.trx_phase1.price_position import DAY, OFFSET, build_price_reference
from freqtrade.strategy.trx_phase1.replay import replay
from tests.strategy.trx_phase1.test_half_position_startup import scenario


def minutes(values, start="2025-12-31 16:00:00+00:00"):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=len(values), freq="min"),
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": 10.0,
        }
    )


def config(**updates):
    return ResearchConfig(
        **{
            "version": "trx-phase1-v0.7",
            "variant": "dynamic_period_only",
            "buy_fraction": 0.12,
            "sell_fraction": 0.12,
            "initial_cash": 1000,
            "price_position_enabled": True,
            "price_position_lookback_days": 1,
            "min_notional": 0.01,
            "tail_completion": False,
            **updates,
        }
    )


def test_reference_uses_complete_previous_bj_day_and_updates_at_midnight():
    data = minutes([1.0] * 1440 + [2.0] * 1440 + [100.0] * 50)
    cfg = config()
    refs = build_price_reference(data, cfg)
    start = int(data.date.iloc[0].timestamp() * 1000)
    assert refs.value_at(start + DAY - 60000) is None
    assert refs.value_at(start + DAY) == 1
    assert refs.value_at(start + 2 * DAY - 60000) == 1
    assert refs.value_at(start + 2 * DAY) == 2
    # The first prior day may not be completed by silently carrying a partial day.
    partial = build_price_reference(data.iloc[1:], cfg)
    assert partial.value_at(start + DAY) is None


def test_future_prices_cannot_change_earlier_daily_reference():
    data = minutes(np.r_[np.linspace(1, 2, 1440), np.linspace(2, 3, 1440), [4.0] * 60])
    start = int(data.date.iloc[0].timestamp() * 1000)
    a = build_price_reference(data, config())
    b = build_price_reference(data.iloc[:1440], config())
    assert a.value_at(start + DAY) == b.value_at(start + DAY) == pytest.approx(1.75)
    changed = data.copy()
    changed.loc[1440:, ["open", "high", "low", "close"]] = 999
    assert build_price_reference(changed, config()).value_at(start + DAY) == a.value_at(start + DAY)


def engine(cfg, reference):
    book = Ledger(cfg.initial_cash)
    book.buy("seed", "seed", 0, 100, 0.5)
    result = Engine(cfg, book)
    result.anchor = 1
    result.price_position_day = (60000 + OFFSET) // DAY
    result.price_position_reference = reference
    return result


@pytest.mark.parametrize(
    "reference,buy_budget,sell_quantity", [(1, 60, 18), (2, 120, 12), (None, 120, 12)]
)
def test_high_only_adjustment_and_strict_equality(reference, buy_budget, sell_quantity):
    cfg = config()
    buyer = engine(cfg, reference)
    seller = engine(cfg, reference)
    assert buyer.make_plan("buy", 60000, 2, 0.012, 120000).budget == buy_budget
    assert seller.make_plan("sell", 60000, 2, 0.012, 120000).quantity == sell_quantity


def test_multiplier_is_after_dynamic_sizing_and_does_not_sell_losing_lots():
    cfg = config(variant="dynamic_quantity_period")
    off = cfg.model_copy(update={"price_position_enabled": False})
    normal_buy = engine(off, 1).make_plan("buy", 60000, 2, 0.012, 120000)
    normal_sell = engine(off, 1).make_plan("sell", 60000, 2, 0.012, 120000)
    assert engine(cfg, 1).make_plan("buy", 60000, 2, 0.012, 120000).budget == pytest.approx(
        normal_buy.budget * 0.5
    )
    assert engine(cfg, 1).make_plan("sell", 60000, 2, 0.012, 120000).quantity == pytest.approx(
        normal_sell.quantity * 1.5, abs=2e-6
    )
    assert (
        engine(cfg.model_copy(update={"sell_fraction": 1}), 1)
        .make_plan("sell", 60000, 2, 0.012, 120000)
        .quantity
        == 100
    )
    assert engine(cfg, 0.1).make_plan("sell", 60000, 0.4, 0.012, 120000) is None


def test_startup_is_excluded_and_insufficient_history_replays_original_path():
    data, original = scenario(1 + np.sin(np.arange(180) / 15) * 0.05)
    cfg = original.model_copy(update={"price_position_enabled": True})
    a, b = replay(data, original), replay(data, cfg)
    for key in [
        "summary",
        "fills",
        "orders",
        "interest_receipts",
        "equity_curve",
        "lots",
        "startup_allocation",
    ]:
        assert a[key] == b[key], key
    assert b["price_position"]["reference_available_minutes"] == 0
    assert all(e["multiplier"] == 1 for e in b["price_position"]["sizing_events"])


def test_enabled_engine_requires_reference_context_and_old_configs_remain_off():
    e = Engine(config())
    e.anchor = 1
    with pytest.raises(ValueError, match="causal reference"):
        e.make_plan("buy", 60000, 1, 0.012, 120000)
    assert not ResearchConfig.from_saved({"variant": "dynamic_period_only"}).price_position_enabled
