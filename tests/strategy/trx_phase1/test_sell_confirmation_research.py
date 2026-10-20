"""Replay-only sell caps preserve buy rules and expose causal execution changes."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.research_earn import HistoricalEarn


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "user_data/trx_phase1/control_matrices/c1-sell-sizing-controls-2026-10-16"
    / "source/research_engine.py"
)
if not SOURCE.is_file():
    pytest.skip(
        "Optional archived sell-sizing research source is unavailable", allow_module_level=True
    )

spec = importlib.util.spec_from_file_location("isolated_sell_confirmation", SOURCE)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def config():
    return ResearchConfig(
        initial_cash=10000,
        variant="dynamic_period_only",
        periods=[60, 240, 1440],
        sell_fraction=0.12,
        buy_threshold=0.012,
        sell_threshold=0.012,
        slippage=0,
    )


def seeded(cap=None):
    cfg = config()
    engine = model.SellPeriodCapEngine(cfg, {}, cap) if cap else model.VolumeCheckedEngine(cfg, {})
    engine.book.buy("seed", "seed", 0, 8000, 1)
    return engine


def test_cap_preserves_buy_tier_and_only_publishes_the_effective_sell_schedule():
    normal, capped = seeded(), seeded(60)
    events = []

    class Observer:
        def event(self, kind, engine, timestamp, values):
            events.append((kind, engine.buy_period, engine.sell_period, values))

    capped.audit = Observer()
    for engine in [normal, capped]:
        engine.schedule(121 * 60000, 1, 0)
    assert normal.buy_period == capped.buy_period == 1440
    assert normal.buy_at == capped.buy_at
    assert normal.buy_threshold == capped.buy_threshold
    assert normal.tier == capped.tier
    assert normal.sell_period == 240 and capped.sell_period == 60
    assert normal.sell_at == 240 * 60000 and capped.sell_at == 180 * 60000
    assert [(kind, buy, sell) for kind, buy, sell, _ in events] == [("schedule", 1440, 60)]
    old_sell = capped.sell_at
    capped.schedule(181 * 60000, 1, 0, buy=True, sell=False)
    assert capped.sell_at == old_sell


def test_loose_cap_preserves_original_schedule_over_multiple_tier_updates():
    normal, capped = seeded(), seeded(1440)
    for i, price in enumerate([1, 1.05, 0.9, 1, 1.1]):
        for engine in [normal, capped]:
            engine.schedule((121 + i * 60) * 60000, price, 0)
        for key in [
            "tier",
            "relax_count",
            "buy_at",
            "sell_at",
            "buy_period",
            "sell_period",
            "buy_threshold",
            "sell_threshold",
        ]:
            assert getattr(normal, key) == getattr(capped, key)


def test_brief_profitable_rebound_can_fill_after_hourly_confirmation_but_not_before_it():
    n = 500
    closes = np.ones(n)
    closes[179:182] = 1.03
    opens = np.r_[1.0, closes[:-1]]
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-10-12", periods=n, freq="min", tz="UTC"),
            "open": opens,
            "close": closes,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "volume": 1e6,
        }
    )
    volumes = dict(zip(pd.DatetimeIndex(frame.date).as_unit("ms").asi8, frame.volume, strict=True))

    def factory(cap):
        def account(cfg, timestamp):
            engine = seeded(cap)
            engine.volumes = volumes
            return engine, HistoricalEarn(engine.book, cfg, timestamp)

        return account

    normal = replay(frame, config(), engine_factory=factory(None))
    capped = replay(frame, config(), engine_factory=factory(60))
    assert normal["summary"]["sell_orders"] == 0
    assert capped["summary"]["sell_orders"] == 1
    sell = next(f for f in capped["fills"] if f["side"] == "sell")
    assert sell["confirmation_period"] == 60
    assert sell["timestamp"] >= int(frame.date.iloc[180].timestamp() * 1000)
    assert all(a["profit"] > 0 for a in sell["allocations"])
