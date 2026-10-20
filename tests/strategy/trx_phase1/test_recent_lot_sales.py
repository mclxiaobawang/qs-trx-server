import pytest

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.runtime import PaperRuntime
from tests.strategy.trx_phase1.test_half_position_startup import scenario


def config(**updates):
    return ResearchConfig(
        **{
            "version": "trx-phase1-v0.8",
            "initial_cash": 10000,
            "startup": "half_position",
            "initial_buy_fraction": 0.5,
            "variant": "dynamic_period_only",
            "periods": [60, 240, 1440],
            "earn_model": "net_benefit_hourly",
            "sell_fraction": 0.12,
            "slippage": 0,
            "sell_mode": "recent_lot_then_core",
            **updates,
        }
    )


def engine(*, added=True, **updates):
    book = Ledger(10000)
    book.buy("core", "core", 0, 1000, 0.5)
    if added:
        book.buy("added", "added", 1, 100, 0.8)
    book.interest_trx = money(17)
    result = Engine(config(**updates), book)
    result.core_lot_ids = ["core"]
    result.anchor = 0.8
    result.last_sigma = 0
    result.price_position_day = 0
    result.price_position_reference = 0.7
    return result


def sell(e, price=0.9):
    return e.make_plan("sell", 60000, price, 0.012, 3600000)


def test_entire_added_quantity_overrides_high_p75_without_spilling_into_core_or_interest():
    e = engine(price_position_enabled=True)
    plan = sell(e)
    assert plan.quantity == 100
    assert plan.allocations == [{"lot_id": "added", "quantity": 100}]
    assert plan.inventory_source == "added_lot"
    assert e.price_position_events[-1]["multiplier"] == 1
    assert e.price_position_events[-1]["application"] == "added_lot_quantity_unscaled"
    e.fill("s1", plan.plan_id, 120000, 40, 0.9, 0.036, final=False)
    assert e.execution_quantity(0.9) == 60
    restored = Engine.restore(e.state())
    restored.price_position_day = 0
    restored.price_position_reference = 0.7
    assert restored.core_lot_ids == ["core"]
    assert restored.pending.inventory_source == "added_lot"
    restored.fill("s2", plan.plan_id, 180000, 60, 0.9, 0.054)
    assert restored.book.lots[0].quantity == 1000
    assert restored.book.lots[1].quantity == 0
    assert restored.book.interest_trx == 17
    # A separate later signal, after the added lot is gone, can sell the core.
    next_plan = sell(restored, 1.0)
    assert next_plan.inventory_source == "core_fraction" and next_plan.quantity == 180


def test_newest_profitable_added_lot_only_and_losing_added_lot_blocks_core():
    e = engine()
    e.book.buy("older", "older", 2, 70, 0.7)
    e.book.buy("losing", "losing", 3, 50, 1.1)
    plan = sell(e)
    assert plan.quantity == 70 and plan.allocations[0]["lot_id"] == "older"
    e.fill("s1", plan.plan_id, 120000, 70, 0.9, 0.063)
    plan = sell(e)
    assert plan.quantity == 100 and plan.allocations[0]["lot_id"] == "added"
    e.fill("s2", plan.plan_id, 180000, 100, 0.9, 0.09)
    assert sell(e) is None
    assert e.sell_policy_events[-1]["remaining_added_lots"] == 1
    assert e.book.lots[0].quantity == 1000  # Profitable core is still held.


def test_minimum_order_and_final_execution_profit_gate_still_apply():
    e = engine()
    assert sell(e) is not None
    assert not e.executable(120000, 0.8)
    assert e.book.realized == 0
    tiny = engine(added=False)
    tiny.book.buy("dust", "dust", 1, 1, 0.8)
    assert sell(tiny) is None
    assert tiny.book.lots[0].quantity == 1000
    assert tiny.book.lots[1].quantity == 1


@pytest.mark.parametrize("high,quantity", [(True, 180), (False, 120)])
def test_core_without_added_lots_retains_original_p75_fraction(high, quantity):
    e = engine(added=False, price_position_enabled=True)
    e.price_position_reference = 0.7 if high else 2
    assert sell(e).quantity == quantity


def test_startup_is_core_and_buy_budget_keeps_original_p75_half_multiplier():
    data, cfg = scenario([1.0] * 120)
    run = replay(data, cfg.model_copy(update={"sell_mode": "recent_lot_then_core"}))
    assert run["sell_policy"]["core_lot_ids"] == [run["fills"][0]["order_id"]]
    assert run["fills"][0]["startup"]
    e = engine(price_position_enabled=True)
    expected = float(e.book.deployable) * 0.12 * 0.5
    assert e.make_plan("buy", 60000, 0.9, 0.012, 3600000).budget == expected


def test_default_mode_preserves_original_path_and_new_mode_is_replay_only(tmp_path):
    data, cfg = scenario([1.0] * 30 + [0.98] * 40 + [1.02] * 50)
    original = replay(data, cfg)
    explicit = replay(data, cfg.model_copy(update={"sell_mode": "proportional"}))
    for key in ["summary", "fills", "orders", "equity_curve", "lots"]:
        assert original[key] == explicit[key]
    assert ResearchConfig.from_saved({"variant": "dynamic_period_only"}).sell_mode == "proportional"
    with pytest.raises(ValueError, match="historical replay"):
        PaperRuntime(tmp_path, tmp_path / "unused", None, config())
    runtime = object.__new__(PaperRuntime)
    with pytest.raises(ValueError, match="historical replay"):
        runtime.configure(config())
    assert not list(tmp_path.iterdir())
