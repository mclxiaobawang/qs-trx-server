"""Read-only range projections shared with the independent, credential-free observer."""

from copy import deepcopy

from .ledger import money


def range_view(state, engine, now):
    binding = state.get("range_policy")
    if not binding:
        return None
    policy = binding["policy"]
    band = state.get("active_price_band") or engine.get("band")
    result = {**policy, "band": deepcopy(band), "ready": False}
    day = ((int(now) + 28800000) // 86400000) * 86400000 - 28800000
    if not band or band.get("effective_at") != day or not engine.get("last_price"):
        return result
    price = money(engine["last_price"])
    inside = money(band["lower"]) <= price <= money(band["upper"])
    base = money(policy["inside_buy_fraction"] if inside else policy["outside_buy_fraction"])
    p75 = (state.get("price_reference") or {}).get("value")
    high = price > money(p75) if p75 is not None else None
    multiplier = money(engine["config"]["price_position_buy_multiplier"]) if high else money(1)
    intersection = False
    if policy.get("pause_upper_p75_buys", False):
        if (state.get("price_reference") or {}).get("day") != (
            int(now) + 28800000
        ) // 86400000 or high is None:
            return result
        # The credential-free observer has no trading-engine dependencies.
        intersection = price > money(band["upper"]) and price > money(p75)
        result.update(
            upper_p75_intersection=intersection,
            intersection_buy_multiplier="0" if intersection else "1",
            new_buy_paused=intersection,
        )
    result.update(
        ready=True,
        latest_close=str(price),
        latest_close_at=engine["last_close"],
        inside=inside,
        base_buy_fraction=str(base),
        sizing_high_price=high,
        sizing_multiplier=str(multiplier) if high is not None else None,
        estimated_fraction=(
            str(money(0) if intersection else base * multiplier) if high is not None else None
        ),
        budget_basis=(
            "latest_confirmed_close_regime_p75_and_filter_estimate"
            if policy.get("pause_upper_p75_buys", False)
            else "latest_confirmed_close_regime_and_p75_estimate"
        ),
        activated_at=state.get("range_policy_activated_at"),
    )
    return result


def buy_price_projection(signal, state, engine, view, now):
    """Find the next eligible price at/below the quote, without a trading engine."""
    cfg, policy = engine["config"], state["range_policy"]["policy"]
    reference = state.get("price_reference") or {}
    enabled = cfg.get("price_position_enabled", False)
    if not view["ready"] or (
        enabled
        and (
            reference.get("day") != (int(now) + 28800000) // 86400000
            or reference.get("value") is None
        )
    ):
        return None
    anchor = money(signal["sides"]["buy"]["reference"])
    ceiling = money(signal["quote"].get("ask") or engine["last_price"])
    lower, upper = (money(view["band"][k]) for k in ("lower", "upper"))
    p75 = money(reference["value"]) if enabled else None
    if min(anchor, ceiling, lower) <= 0 or upper < lower or (enabled and p75 <= 0):
        return None
    inside_threshold = money(policy["inside_buy_threshold"])
    outside_threshold = money(policy["outside_buy_threshold"])
    inside_target = anchor * (1 - inside_threshold)
    outside_target = anchor * (1 - outside_threshold)
    pause = policy.get("pause_upper_p75_buys", False)

    def blocked(price):
        return pause and p75 is not None and price > upper and price > p75

    # The below-band zone excludes lower; the inside zone includes both edges.
    # With the bound policy's lower inside threshold, an excluded lower endpoint
    # is always covered by an eligible inside candidate instead.
    candidates = []
    below = min(ceiling, outside_target)
    if 0 < below < lower:
        candidates.append((below, outside_threshold, False))
    inside = min(ceiling, inside_target, upper)
    if lower <= inside <= upper:
        candidates.append((inside, inside_threshold, True))
    above = (
        min(ceiling, outside_target, p75)
        if pause and p75 is not None
        else min(ceiling, outside_target)
    )
    if above > upper and not blocked(above):
        candidates.append((above, outside_threshold, False))
    if not candidates:
        return None
    target, threshold, inside = max(candidates, key=lambda row: row[0])
    high = p75 is not None and target > p75
    fraction = money(policy["inside_buy_fraction"] if inside else policy["outside_buy_fraction"])
    multiplier = money(cfg["price_position_buy_multiplier"]) if high else money(1)
    current_inside = lower <= ceiling <= upper
    current_threshold = inside_threshold if current_inside else outside_threshold
    met = ceiling <= anchor * (1 - current_threshold) and not blocked(ceiling)
    kind = (
        "current_price"
        if met
        else "range_upper"
        if target == upper
        else "p75_boundary"
        if pause and p75 is not None and target == p75 and target > upper
        else "threshold"
    )
    return {
        "threshold": str(threshold),
        "target": str(target),
        "distance_ratio": float((ceiling - target) / ceiling),
        "price_condition_met": met,
        "base_fraction": str(fraction),
        "target_fraction": str(fraction * multiplier),
        "target_high_price": high,
        "sizing_high_price": high,
        "fraction_basis": "next_eligible_buy_price_regime_p75_and_filter",
        "preview_basis": "next_eligible_buy_price_regime_p75_and_filter",
        "preview_target_inside": inside,
        "preview_target_kind": kind,
        "preview_current_new_buy_paused": blocked(ceiling),
    }


def apply_range_signal(signal, state, engine, now):
    view = range_view(state, engine, now)
    if view is None:
        return
    signal["range_policy"] = view
    buy = signal["sides"]["buy"]
    projected = buy_price_projection(signal, state, engine, view, now)
    if projected is None:
        buy.update(
            target=None,
            distance_ratio=None,
            price_condition_met=None,
            target_fraction=None,
            target_high_price=None,
            preview_basis="range_reference_pending",
            fraction_basis="range_reference_pending",
        )
    else:
        buy.update(projected)


def display_config(state, engine, now):
    config = deepcopy(engine["config"])
    view = range_view(state, engine, now)
    if view is not None:
        config["active_range_policy"] = view
        if view["ready"]:
            config["buy_fraction"] = float(view["base_buy_fraction"])
            config["buy_threshold"] = engine["buy_threshold"]
            config["sell_threshold"] = engine["sell_threshold"]
    return config
