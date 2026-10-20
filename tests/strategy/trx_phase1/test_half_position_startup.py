import numpy as np
import pandas as pd
import pytest

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.replay import replay


def scenario(prices, **updates):
    prices = np.asarray(prices, dtype=float)
    opening = np.r_[1.0, prices[:-1]]
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(prices), freq="1min", tz="UTC"),
            "open": opening,
            "close": prices,
            "high": np.maximum(opening, prices),
            "low": np.minimum(opening, prices),
            "volume": 10,
        }
    )
    cfg = ResearchConfig(
        version="trx-phase1-v0.6",
        initial_cash=10000,
        startup="half_position",
        initial_buy_fraction=0.5,
        variant="fixed",
        periods=[5, 15],
        volatility_window=20,
        earn_model="net_benefit_hourly",
        buy_fraction=0.12,
        sell_fraction=0.12,
        slippage=0,
        fee_rebate_ratio=0.5,
        trx_apr=0.1033,
        usdgo_apr=0.10,
        redemption_delay_minutes=1,
        **updates,
    )
    return data, cfg


def test_first_open_allocation_includes_fees_and_uses_no_future_close(tmp_path):
    data, cfg = scenario([1.0] * 120)
    run = replay(data, cfg, diagnostic_path=tmp_path / "half.sqlite")
    allocation = run["startup_allocation"]
    fill = run["fills"][0]
    assert fill["timestamp"] == int(data.date.iloc[0].timestamp() * 1000)
    assert fill["startup"] and fill["confirmation_period"] == 0
    assert allocation["usdgo_quantity"] == 5000
    assert allocation["trx_budget_including_fee_usdt"] == 5000
    assert 0 <= 5000 - fill["cost"] < 0.000002
    assert fill["fee"] == pytest.approx(fill["quantity"] * fill["price"] * 0.001)
    assert run["equity_curve"][0]["equity"] == pytest.approx(10000 - fill["fee"])
    assert run["equity_curve"][0]["anchor"] == 1.0
    assert run["summary"]["buy_orders"] == 1 and run["summary"]["sell_orders"] == 0
    changed = data.copy()
    changed.loc[:, "close"] = 2.0
    changed.loc[:, "high"] = 2.0
    assert replay(changed, cfg)["fills"][0] == fill
    assert ResearchConfig.from_saved(run["config"]) == cfg


@pytest.mark.parametrize("direction,side", [(1, "sell"), (-1, "buy")])
def test_first_threshold_after_warmup_can_sell_rise_or_buy_fall(direction, side):
    data, cfg = scenario([1.0] * 30 + [1.0 + direction * 0.02] * 80)
    run = replay(data, cfg)
    assert len(run["fills"]) >= 2
    assert run["fills"][1]["side"] == side
    assert run["fills"][1]["signal_anchor"] == 1.0
    assert run["fills"][1]["signal_threshold"] == 0.012
    assert run["fills"][1]["timestamp"] >= int(data.date.iloc[20].timestamp() * 1000)
    assert all(o["profit"] > 0 for o in run["orders"] if o["side"] == "sell")
    assert run["summary"]["interest_sale_proceeds"] == 0


def test_full_and_prefix_replays_keep_identical_past_startup_and_fills():
    data, cfg = scenario(1 + np.sin(np.arange(300) / 18) * 0.08)
    full, prefix = replay(data, cfg), replay(data.iloc[:150], cfg)
    cutoff = int((data.date.iloc[149] + pd.Timedelta("1min")).timestamp() * 1000)
    assert full["startup_allocation"] == prefix["startup_allocation"]
    assert [f for f in full["fills"] if f["timestamp"] < cutoff] == prefix["fills"]
    assert [r for r in full["interest_receipts"] if r["timestamp"] <= cutoff] == prefix[
        "interest_receipts"
    ]


def test_half_position_cannot_silently_use_different_allocation_or_exceed_quota():
    data, cfg = scenario([1.0] * 30)
    with pytest.raises(ValueError, match="initial_buy_fraction"):
        ResearchConfig.from_saved({**cfg.model_dump(), "initial_buy_fraction": 0.02})
    with pytest.raises(ValueError, match="quota"):
        replay(data, cfg.model_copy(update={"usdgo_earn_cap": 4999}))


def test_wait_control_keeps_zero_initial_trx():
    data, cfg = scenario([1.0] * 120)
    control = replay(data, cfg.model_copy(update={"startup": "wait"}))
    assert "startup_allocation" not in control
    assert control["fills"] == []
    assert control["equity_curve"][0]["principal_trx"] == 0
    assert control["equity_curve"][0]["usdgo"] == 10000


@pytest.mark.parametrize("fraction", [0.1, 0.9])
def test_allocated_first_open_preserves_capital_and_ignores_future_prices(fraction):
    data, half = scenario([1.0] * 120)
    cfg = ResearchConfig.from_saved(
        {**half.model_dump(), "startup": "allocated_position", "initial_buy_fraction": fraction}
    )
    run = replay(data, cfg)
    first, allocation = run["fills"][0], run["startup_allocation"]
    assert first["timestamp"] == int(data.date.iloc[0].timestamp() * 1000)
    assert first["confirmation_period"] == 0
    assert first["cost"] == pytest.approx(10000 * fraction, abs=0.000002)
    assert allocation["usdgo_quantity"] == pytest.approx(10000 * (1 - fraction))
    assert run["equity_curve"][0]["equity"] + first["fee"] == pytest.approx(10000)
    changed = data.copy()
    changed.loc[:, "close"] = 2.0
    changed.loc[:, "high"] = 2.0
    assert replay(changed, cfg)["fills"][0] == first
    assert ResearchConfig.from_saved(run["config"]) == cfg


def test_allocated_fifty_percent_preserves_entire_legacy_half_account_path():
    data, half = scenario(1 + np.sin(np.arange(400) / 18) * 0.08)
    allocated = ResearchConfig.from_saved({**half.model_dump(), "startup": "allocated_position"})
    old, new = replay(data, half), replay(data, allocated)
    for key in ["summary", "fills", "equity_curve", "interest_receipts", "lots", "decisions"]:
        assert new[key] == old[key], key


@pytest.mark.parametrize("direction,side", [(1, "sell"), (-1, "buy")])
def test_allocated_start_preserves_first_fill_anchor_and_signal_warmup(direction, side):
    data, half = scenario([1.0] * 30 + [1.0 + direction * 0.02] * 80)
    cfg = ResearchConfig.from_saved(
        {**half.model_dump(), "startup": "allocated_position", "initial_buy_fraction": 0.1}
    )
    run = replay(data, cfg)
    fill = run["fills"][1]
    assert fill["side"] == side and fill["signal_anchor"] == 1.0
    assert fill["timestamp"] >= int(data.date.iloc[20].timestamp() * 1000)
    assert all(o["profit"] > 0 for o in run["orders"] if o["side"] == "sell")


def test_allocated_start_cannot_spend_all_capital_or_exceed_usdgo_quota():
    data, half = scenario([1.0] * 30)
    saved = {**half.model_dump(), "startup": "allocated_position"}
    with pytest.raises(ValueError, match="positive remaining"):
        ResearchConfig.from_saved({**saved, "initial_buy_fraction": 1})
    with pytest.raises(ValueError, match="quota"):
        replay(
            data,
            ResearchConfig.from_saved(
                {**saved, "initial_buy_fraction": 0.1, "usdgo_earn_cap": 8999}
            ),
        )
