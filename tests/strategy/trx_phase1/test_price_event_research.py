"""Meaningful execution invariants for the isolated, non-live research runner."""

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.strategy.trx_phase1.config import ResearchConfig


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "user_data/trx_phase1/control_matrices/price-event-range-2026-10-16/source/event_model.py"
)
if not SOURCE.is_file():
    pytest.skip(
        "Optional archived price-event research source is unavailable", allow_module_level=True
    )

spec = importlib.util.spec_from_file_location("isolated_event_model", SOURCE)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def frame(high=0.3, low=0.3, volume=1000000):
    n = 1100
    f = pd.DataFrame(
        {
            "date": pd.date_range("2025-10-12", periods=n, freq="1min", tz="UTC"),
            "open": 0.3,
            "close": 0.3,
            "high": 0.3,
            "low": 0.3,
            "volume": 1000000.0,
        }
    )
    f.loc[123, ["high", "low", "volume"]] = [high, low, volume]
    return f


def cfg():
    return ResearchConfig(
        initial_cash=10000,
        startup="half_position",
        initial_buy_fraction=0.5,
        variant="fixed",
        periods=[60, 240, 1440],
        sell_fraction=0.12,
        slippage=0,
        earn_model="net_benefit_hourly",
        trx_apr=0,
        usdgo_apr=0,
        usdt_apr=0,
        redemption_delay_minutes=1,
        usdt_redemption_delay_minutes=1,
    )


def run(tmp_path, f, execution="resting_limit", sizing="proportional", path="high_first", bps=0):
    return model.replay_events(
        f,
        cfg(),
        model.Terms(execution, sizing, path, bps),
        tmp_path / f"{execution}-{sizing}-{path}-{bps}.sqlite",
        model.EventRecorder,
    )


def test_two_old_quotes_may_fill_but_new_quotes_never_fill_inside_the_creation_bar(tmp_path):
    a = run(tmp_path, frame(0.31, 0.29), path="high_first")
    b = run(tmp_path, frame(0.31, 0.29), path="low_first")
    for result in [a, b]:
        spike_bar = int(frame().date.iloc[123].timestamp() * 1000) // 60000
        fills = [f for f in result["fills"] if f["timestamp"] // 60000 == spike_bar]
        # One prepared buy + one prepared sell, despite a 6% intraminute range.
        assert len(fills) == 2
        assert {f["side"] for f in fills} == {"buy", "sell"}
        assert all(f["timestamp"] // 60000 > f["signal_timestamp"] // 60000 for f in fills)
        assert sum(f["stale_opposite_quote_fill"] for f in fills) == 1
        assert all(x["profit"] > 0 for f in fills if f["side"] == "sell" for x in f["allocations"])
    assert [f["side"] for f in a["fills"]][1:3] == ["sell", "buy"]
    assert [f["side"] for f in b["fills"]][1:3] == ["buy", "sell"]
    assert a["fills"][2]["price"] != b["fills"][2]["price"]


def test_positive_volume_and_penetration_are_real_execution_constraints(tmp_path):
    touched = run(tmp_path, frame(0.30360001, 0.3))
    penetrated = run(tmp_path, frame(0.30360001, 0.3), bps=1)
    empty = run(tmp_path, frame(0.31, 0.29, 0))
    assert touched["summary"]["sell_orders"] == 1
    assert penetrated["summary"]["sell_orders"] == 0
    assert len(empty["fills"]) == 1  # Only the common initial half allocation.


@pytest.mark.parametrize("sizing", ["linear", "quadratic"])
def test_target_is_fixed_center_and_never_bypasses_profitable_lot_restriction(tmp_path, sizing):
    f = frame()
    p = np.linspace(0.3, 0.20, len(f) - 121)
    f.loc[121:, "open"] = np.r_[0.3, p[:-1]]
    f.loc[121:, "close"] = p
    f.loc[121:, "high"] = np.maximum(f.loc[121:, "open"], p)
    f.loc[121:, "low"] = np.minimum(f.loc[121:, "open"], p)
    r = run(tmp_path, f, sizing=sizing)
    assert r["execution_study"]["fixed_center"] == 0.3
    assert r["summary"]["buy_orders"] > 1
    assert r["summary"]["sell_orders"] == 0
    assert r["summary"]["cash"] >= 0
    assert r["summary"]["interest_trx"] == r["summary"]["stable_interest"] == 0
    assert r["execution_study"]["counts"]["lower_boundary_minutes"] > 0
    assert r["summary"]["equity"] - 10000 == pytest.approx(
        r["summary"]["realized_profit"] + r["summary"]["unrealized_profit"], abs=1e-7
    )


def test_market_touch_is_not_filled_at_the_unobserved_intraminute_threshold(tmp_path):
    r = run(tmp_path, frame(0.31, 0.29), execution="market_next_open")
    # Touch is observed after the bar; next-open price .3 no longer meets either gate.
    assert len(r["fills"]) == 1
    assert r["execution_study"]["counts"]["market_rejections"] >= 1


@pytest.mark.parametrize("sizing", ["linear", "quadratic"])
def test_upper_target_releases_profitable_principal_without_resetting_the_center(tmp_path, sizing):
    f = frame()
    prices = np.linspace(0.3, 0.4, len(f) - 121)
    f.loc[121:, "open"] = np.r_[0.3, prices[:-1]]
    f.loc[121:, "close"] = prices
    f.loc[121:, "high"] = np.maximum(f.loc[121:, "open"], prices)
    f.loc[121:, "low"] = np.minimum(f.loc[121:, "open"], prices)
    result = run(tmp_path, f, sizing=sizing)
    assert result["execution_study"]["fixed_center"] == 0.3
    assert result["summary"]["sell_orders"] > 1
    assert result["summary"]["principal_trx"] * 0.4 < cfg().min_notional
    assert all(
        a["profit"] > 0
        for fill in result["fills"]
        if fill["side"] == "sell"
        for a in fill["allocations"]
    )


def test_full_minute_ledger_distinguishes_order_reserve_from_redemption(tmp_path):
    r = run(tmp_path, frame(0.31, 0.29))
    path = tmp_path / "resting_limit-proportional-high_first-0.sqlite"
    with sqlite3.connect(path) as db:
        count, negative, asset, profit = db.execute(
            "SELECT count(*), min(free_cash), "
            "max(abs(equity-cash-usdgo_value-principal_trx*price)),"
            "max(abs(equity-10000-realized_profit-principal_unrealized-fx_profit-fx_unrealized)) "
            "FROM samples"
        ).fetchone()
        assert count == 1100 and negative >= -1e-8 and max(asset, profit) < 1e-7
        resting = db.execute(
            "SELECT count(*) FROM samples WHERE cash-free_cash>1000 "
            "AND redemption_usdt<1e-6 AND locked_usdgo=0"
        ).fetchone()[0]
        assert resting > 0
    assert r["summary"]["pending_fee_rebates"] >= 0
