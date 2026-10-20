"""Read-only live dashboards and forward-only records of actual signal evaluations.

No exchange calls, execution writes or alternative strategy decisions belong here.
"""

from copy import deepcopy

from .ledger import money


class CloseAudit:
    def __init__(self):
        self.events = []

    def event(self, kind, engine, timestamp, values):
        if kind in {"evaluation", "blocked", "sizing"}:
            self.events.append((kind, deepcopy(values)))


def observed_close(engine, timestamp, price, sigma, processed_at, skip_reason=None):
    """Observe the one authoritative on_close call; never evaluate a second path."""
    audit, previous = CloseAudit(), engine.audit
    anchor_version = engine.anchor_version
    due = {side: getattr(engine, side + "_at") for side in ("buy", "sell")}
    engine.audit = audit
    try:
        plan = engine.on_close(timestamp, price, sigma)
    finally:
        engine.audit = previous
    rows = []
    blocked = next((v.get("reason") for k, v in reversed(audit.events) if k == "blocked"), None)
    for side in ("buy", "sell"):
        evaluation = next(
            (v for k, v in audit.events if k == "evaluation" and v["side"] == side), None
        )
        if evaluation is None and (not due[side] or timestamp < due[side]):
            continue
        result = (
            "skipped"
            if evaluation is None
            else "not_triggered"
            if not evaluation["met"]
            else "plan_created"
            if plan is not None and plan.side == side and plan.created_at == timestamp
            else "blocked"
        )
        reason = (
            skip_reason or ("existing_plan_pending" if engine.pending else "evaluation_unavailable")
            if result == "skipped"
            else "threshold_not_met"
            if result == "not_triggered"
            else "signal_plan_created"
            if result == "plan_created"
            else blocked or engine.reason
        )
        rows.append(
            {
                "timestamp": timestamp,
                "processed_at": processed_at,
                "side": side,
                "scheduled_at": due[side],
                "anchor_version": anchor_version,
                "price": price,
                "result": result,
                "reason": reason,
                **(evaluation or {}),
                "plan_id": plan.plan_id if result == "plan_created" else None,
            }
        )
    return plan, rows


def dashboard(state, engine, execution, book, now):
    """Derive display values from a single saved snapshot, preserving its amounts."""
    quotes = state.get("markets", {})
    trx, fx = quotes.get("TRXUSDT", {}), quotes.get("USDGOUSDT", {})
    cfg = engine["config"]
    quote = {
        "observed_at": trx.get("timestamp"),
        "bid": trx.get("bid"),
        "ask": trx.get("ask"),
    }
    signal = {"quote": quote, "last_price": engine["last_price"], "sides": {}}
    for side in ("buy", "sell"):
        reference = money(engine["anchor"])
        if (
            side == "buy"
            and engine.get("recovery_reference")
            and (
                engine["last_close"] > engine.get("recovery_started_at", 0)
                and reference > 0
                and 1 - money(engine["last_price"]) / reference < money(engine["buy_threshold"])
            )
        ):
            reference = money(engine["recovery_reference"])
        threshold = money(engine[side + "_threshold"])
        target = reference * (1 - threshold if side == "buy" else 1 + threshold)
        current = money(trx["ask" if side == "buy" else "bid"]) if trx else None
        distance = (
            (current - target if side == "buy" else target - current) / current
            if current and target > 0
            else None
        )
        p75 = (state.get("price_reference") or {}).get("value")
        high = bool(p75 is not None and target > money(p75))
        multiplier = money(cfg["price_position_" + side + "_multiplier"]) if high else money(1)
        fraction = min(money(1), money(cfg[side + "_fraction"]) * multiplier)
        signal["sides"][side] = {
            "reference": str(reference),
            "threshold": str(threshold),
            "target": str(target) if target > 0 else None,
            "distance_ratio": float(distance) if distance is not None else None,
            "price_condition_met": distance <= 0 if distance is not None else None,
            "period": engine[side + "_period"],
            "next_at": engine[side + "_at"],
            "target_high_price": high if p75 is not None else None,
            "target_fraction": str(fraction) if p75 is not None else None,
        }

    # Locked Earn principal is available to the planner only after a mapped redemption.
    planning = deepcopy(book)
    mapped = {}
    for holding in execution["holdings"].values():
        for lot_id, quantity in holding.get("lots", {}).items():
            mapped[lot_id] = mapped.get(lot_id, money(0)) + money(quantity)
    location_verified = all(mapped.get(lot.lot_id, money(0)) == lot.locked for lot in planning.lots)
    for lot in planning.lots:
        lot.locked = money(0)
    if trx and location_verified:
        for name, price in (
            ("current", money(trx["bid"])),
            ("target", money(signal["sides"]["sell"]["target"] or 0)),
        ):
            eligible = planning.qualifying(
                price, planning.principal, cfg["sell_fee"], cfg["execution_margin"]
            )
            signal["sides"]["sell"][name + "_eligible_trx"] = str(
                sum((money(row["quantity"]) for row in eligible), money(0))
            )
    signal["principal_location_verified"] = location_verified
    if fx:
        fee = money(fx["fee"])
        available = book.free_cash + book.usdgo * money(fx["bid"]) * (1 - fee) / (1 + fee)
        signal["redeemable_cash_estimate"] = str(available)
        buy_fraction = signal["sides"]["buy"]["target_fraction"]
        signal["sides"]["buy"]["target_budget_estimate"] = (
            str(max(money(0), min(available, book.deployable * money(buy_fraction))))
            if buy_fraction is not None
            else None
        )

    locations = []
    for coin, total, principal, interest, locked in (
        ("USDT", book.cash, book.cash, money(0), book.locked["USDT"]),
        ("USDGO", book.usdgo, None, None, book.locked["USDGO"]),
        (
            "TRX",
            book.principal + book.interest_trx,
            book.principal,
            book.interest_trx,
            sum((lot.locked for lot in book.lots), money(0)) + book.locked["TRX_INTEREST"],
        ),
    ):
        holdings = [h for h in execution["holdings"].values() if h["coin"] == coin]
        earn = sum((money(h["principal"]) + money(h["interest"]) for h in holdings), money(0))
        outside = money(execution.get("unassigned_offsets", {}).get(coin, 0)) + sum(
            (money(h.get("unassigned_amount", 0)) for h in holdings), money(0)
        )
        locations.append(
            {
                "coin": coin,
                "total": str(total),
                "principal": str(principal) if principal is not None else None,
                "interest": str(interest) if interest is not None else None,
                "spot_available": str(total - locked),
                "earn": str(earn),
                "other_locked": str(locked - earn),
                "outside_strategy": str(outside),
                "consistent": 0 <= earn <= locked <= total,
            }
        )

    components = None
    if trx and (not book.usdgo or fx):
        price = money(trx["bid"])
        components = {
            "principal_realized": str(book.realized),
            "principal_unrealized": str(book.principal * price - book.cost),
            "trx_interest_value": str(book.interest_trx * price),
            "interest_sale_proceeds": str(book.interest_sale_proceeds),
            "stable_interest": str(book.stable_interest),
            "fx_realized": str(book.fx_profit),
            "usdgo_unrealized": str(book.usdgo * money(fx["bid"]) - book.usdgo_cost) if fx else "0",
        }
        total = book.equity(price) - book.initial - book.external_flows
        residual = total - sum((money(v) for v in components.values()), money(0))
        components.update(
            total=str(total),
            residual=str(residual),
            reconciled=abs(residual) <= money("0.00000001"),
        )
    return {
        "observed_at": now,
        "signal": signal,
        "locations": locations,
        "attribution": components,
        "reconciliation_at": execution.get("interest_checked_at"),
    }
