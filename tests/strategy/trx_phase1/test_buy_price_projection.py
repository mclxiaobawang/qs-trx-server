"""Buy forecasts choose an eligible future price, then price that same scenario."""

from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.range_display import apply_range_signal


NOW = 1793096220000
DAY = (NOW + 28800000) // 86400000


def snapshot(anchor=".3278193703910563", ask=".32764", p75=".33223"):
    policy = {
        "half_life_days": 90,
        "width_cap_pct": 4,
        "inside_buy_threshold": 0.00725,
        "outside_buy_threshold": 0.014,
        "inside_buy_fraction": 0.12,
        "outside_buy_fraction": 0.09,
        "pause_upper_p75_buys": True,
    }
    state = {
        "range_policy": {"policy": policy},
        "active_price_band": {
            "effective_at": DAY * 86400000 - 28800000,
            "lower": 0.326,
            "upper": 0.33904,
        },
        "price_reference": {"day": DAY, "value": p75},
    }
    engine = {
        "last_price": ask,
        "last_close": NOW,
        "config": {
            "price_position_enabled": True,
            "price_position_buy_multiplier": 0.25,
        },
    }
    signal = {
        "quote": {"ask": ask},
        "sides": {
            "buy": {"reference": anchor, "threshold": ".00725", "target": "old"},
            "sell": {"target": ".331343428622760155225", "target_fraction": ".12"},
        },
    }
    return signal, state, engine


@pytest.mark.parametrize(
    "anchor,ask,p75,target,threshold,fraction,kind",
    [
        (
            ".3278193703910563",
            ".32764",
            ".33223",
            ".3232298992055815118",
            ".014",
            ".09",
            "threshold",
        ),
        (".33", ".329", ".34", ".3276075", ".00725", ".12", "threshold"),
        # A price gap below the band must not inherit an already-passed inside target.
        (".329", ".325", ".34", ".324394", ".014", ".09", "threshold"),
        # Current price is above P75, but the next eligible price is below it.
        (".337", ".336", ".335", ".33455675", ".00725", ".12", "threshold"),
        # Exit the intersection exactly at P75; strict > means equality is allowed.
        (".38", ".36", ".35", ".35", ".014", ".09", "p75_boundary"),
        # The upper band endpoint is inside, so the intersection ends there.
        (".36", ".35", ".33223", ".33904", ".00725", ".03", "range_upper"),
        (".3278193703910563", ".32", ".33223", ".32", ".014", ".09", "current_price"),
        (".36", ".33904", ".33223", ".33904", ".00725", ".03", "current_price"),
        (".34", ".326", ".33223", ".326", ".00725", ".12", "current_price"),
    ],
)
def test_projection_matches_rule_at_its_own_price_without_mutating_state(
    anchor, ask, p75, target, threshold, fraction, kind
):
    signal, state, engine = snapshot(anchor, ask, p75)
    before = deepcopy((state, engine, signal["sides"]["sell"]))
    apply_range_signal(signal, state, engine, NOW)
    buy = signal["sides"]["buy"]
    assert money(buy["target"]) == money(target)
    assert money(buy["threshold"]) == money(threshold)
    assert money(buy["target_fraction"]) == money(fraction)
    assert buy["preview_target_kind"] == kind
    price = money(buy["target"])
    assert price <= money(ask) and price <= money(anchor) * (1 - money(threshold))
    inside = money(".326") <= price <= money(".33904")
    assert inside == buy["preview_target_inside"]
    assert not (price > money(".33904") and price > money(p75))
    assert (state, engine, signal["sides"]["sell"]) == before


@pytest.mark.parametrize("stale", ["band", "p75"])
def test_missing_current_day_reference_never_emits_a_target_or_quantity(stale):
    signal, state, engine = snapshot()
    if stale == "band":
        state["active_price_band"]["effective_at"] -= 86400000
    else:
        state["price_reference"]["day"] -= 1
    apply_range_signal(signal, state, engine, NOW)
    assert signal["sides"]["buy"]["target"] is None
    assert signal["sides"]["buy"]["target_fraction"] is None
    assert signal["sides"]["buy"]["price_condition_met"] is None
