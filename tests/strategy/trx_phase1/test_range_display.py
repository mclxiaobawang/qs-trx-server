from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_observation import dashboard
from freqtrade.strategy.trx_phase1.range_display import display_config
from tests.strategy.trx_phase1.test_live_range_policy import completed as completed_fixture
from tests.strategy.trx_phase1.test_live_range_policy import live as live_fixture


@pytest.fixture
def live(tmp_path):
    return live_fixture.__wrapped__(tmp_path)


@pytest.fixture
def completed(live):
    return completed_fixture.__wrapped__(live)


@pytest.mark.parametrize(
    "inside,high", [(True, False), (False, False), (True, True), (False, True)]
)
def test_active_dashboard_budget_uses_current_regime_and_p75_without_writes(
    completed, inside, high
):
    runtime, broker, market, binding, _ = completed
    state = deepcopy(runtime.state)
    execution = deepcopy(runtime.execution.state)
    engine = deepcopy(runtime.engine.state())
    state["range_policy"] = binding
    day = ((market.now + 28800000) // 86400000) * 86400000 - 28800000
    state["active_price_band"] = {
        "effective_at": day,
        "lower": 0.29 if inside else 0.31,
        "upper": 0.3016 if inside else 0.3224,
    }
    state["price_reference"] = {
        "value": 0.2 if high else 0.4,
        "day": (market.now + 28800000) // 86400000,
    }
    engine["last_price"] = 0.3
    engine["buy_threshold"] = 0.00725 if inside else 0.014
    engine["sell_threshold"] = 0.01075 if inside else 0.012
    book = Ledger.restore(execution["book"])
    book.usdgo_mark = money(state["markets"]["USDGOUSDT"]["bid"])
    before = deepcopy((state, engine, execution, book.state(), broker.state()))
    result = dashboard(state, engine, execution, book, market.now)
    buy = result["signal"]["sides"]["buy"]
    fraction = money(".12" if inside else ".09") * money(".25" if high else "1")
    available = money(result["signal"]["redeemable_cash_estimate"])
    assert money(buy["target_fraction"]) == fraction
    assert money(buy["target_budget_estimate"]) == min(available, book.deployable * fraction)
    assert result["signal"]["range_policy"]["inside"] == inside
    assert buy["target_high_price"] == high
    config = display_config(state, engine, market.now)
    assert config["buy_fraction"] == (0.12 if inside else 0.09)
    assert config["buy_threshold"] == engine["buy_threshold"]
    assert (state, engine, execution, book.state(), broker.state()) == before


def test_inactive_display_retains_the_exact_original_config(completed):
    runtime, _, market, _, _ = completed
    state, engine = deepcopy(runtime.state), deepcopy(runtime.engine.state())
    assert display_config(state, engine, market.now) == engine["config"]


@pytest.mark.parametrize(
    "high,above,expected", [(True, True, ".03"), (False, True, ".09"), (True, False, ".03")]
)
def test_candidate_dashboard_applies_intersection_to_new_buy_budget(
    completed, high, above, expected
):
    runtime, _, market, binding, _ = completed
    state, engine, execution = deepcopy(
        (runtime.state, runtime.engine.state(), runtime.execution.state)
    )
    binding["policy"]["pause_upper_p75_buys"] = True
    state["range_policy"] = binding
    day = ((market.now + 28800000) // 86400000) * 86400000 - 28800000
    state["active_price_band"] = {
        "effective_at": day,
        "lower": 0.28 if above else 0.29,
        "upper": 0.2912 if above else 0.3016,
    }
    state["price_reference"] = {
        "value": 0.2 if high else 0.4,
        "day": (market.now + 28800000) // 86400000,
    }
    engine["last_price"] = 0.3
    book = Ledger.restore(execution["book"])
    before = deepcopy((state, engine, execution, book.state()))
    result = dashboard(state, engine, execution, book, market.now)
    buy = result["signal"]["sides"]["buy"]
    assert money(buy["target_fraction"]) == money(expected)
    assert result["signal"]["range_policy"]["new_buy_paused"] == (high and above)
    if high and above:
        assert buy["preview_current_new_buy_paused"]
        assert buy["preview_target_inside"]
        assert money(buy["target"]) == money(state["active_price_band"]["upper"])
    assert (state, engine, execution, book.state()) == before


def test_candidate_display_never_estimates_unfiltered_budget_with_stale_p75(completed):
    runtime, _, market, binding, _ = completed
    state, engine, execution = deepcopy(
        (runtime.state, runtime.engine.state(), runtime.execution.state)
    )
    binding["policy"]["pause_upper_p75_buys"] = True
    state["range_policy"] = binding
    state["active_price_band"] = {
        "effective_at": ((market.now + 28800000) // 86400000) * 86400000 - 28800000,
        "lower": 0.28,
        "upper": 0.2912,
    }
    state["price_reference"] = {"value": 0.2, "day": -1}
    engine["last_price"] = 0.3
    result = dashboard(state, engine, execution, Ledger.restore(execution["book"]), market.now)
    assert result["signal"]["range_policy"]["ready"] is False
    assert result["signal"]["sides"]["buy"]["target_fraction"] is None
    assert result["signal"]["sides"]["buy"]["target_budget_estimate"] is None
