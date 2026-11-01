"""Pure, non-confirmable preview of an explicit USDGO Earn capital contribution."""

import hashlib
import json
import re
from copy import deepcopy
from decimal import localcontext


if __package__:
    from .ledger import Ledger, money
else:
    from ledger import Ledger, money


HOUR = 3600000
DAY = 86400000
OFFSET = 480 * 60000


class ContributionPreviewError(ValueError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(value, code):
    if not value:
        raise ContributionPreviewError(code)


def ready(execution, strategy, now):
    require(
        execution["account_id"] == strategy["account_id"]
        and strategy.get("activated_at")
        and strategy.get("source_sha256") == fingerprint(strategy["source_snapshot"]),
        "contribution_existing_strategy_binding_required",
    )
    require(
        strategy["startup_complete"]
        and strategy["phase"] in {"observing", "paused", "prepared"}
        and not strategy.get("error")
        and not execution.get("error")
        and not strategy.get("intent")
        and not strategy["engine"].get("pending")
        and not execution.get("operation")
        and (
            not execution.get("flow")
            or execution["flow"]["status"] in {"completed", "expired", "failed"}
        ),
        "contribution_requires_idle_strategy",
    )
    require(
        execution.get("income_accounting", {}).get("status") == "active"
        and not execution.get("pending_income"),
        "contribution_income_unresolved",
    )
    stamps = [execution.get("interest_checked_at", 0)]
    stamps += [
        strategy.get("markets", {}).get(symbol, {}).get("timestamp", 0)
        for symbol in ("TRXUSDT", "USDGOUSDT")
    ]
    require(
        all(stamp and -5000 <= now - stamp <= 60000 for stamp in stamps),
        "contribution_snapshot_stale",
    )
    return min(stamps)


def buy_budget(book, strategy, now):
    """Same fixed-quantity target convention as the running dashboard, with cash cap."""
    engine, markets = strategy["engine"], strategy["markets"]
    cfg = engine["config"]
    dynamic = cfg.get("buy_quantity_dynamic")
    require(
        dynamic is not True
        and not (
            dynamic is None
            and cfg["variant"]
            in {"dynamic_quantity", "dynamic_size", "dynamic_quantity_period", "dynamic_period"}
        ),
        "contribution_dynamic_sizing_unsupported",
    )
    reference = money(engine["anchor"])
    recovery = money(engine.get("recovery_reference") or 0)
    recovering = bool(
        recovery
        and engine["last_close"] > engine.get("recovery_started_at", 0)
        and reference > 0
        and 1 - money(engine["last_price"]) / reference < money(engine["buy_threshold"])
    )
    if recovering:
        reference = recovery
    target = reference * (1 - money(engine["buy_threshold"]))
    multiplier = money(1)
    p75 = strategy.get("price_reference") or {}
    if cfg["price_position_enabled"]:
        require(
            p75.get("value") is not None and p75.get("day") == (now + OFFSET) // DAY,
            "contribution_current_price_reference_required",
        )
        if target > money(p75["value"]):
            multiplier = money(cfg["price_position_buy_multiplier"])
    fraction = min(money(1), money(cfg["buy_fraction"]) * multiplier)
    if recovering:
        fraction = min(fraction, money(cfg["recovery_buy_fraction"]))
    fx = markets["USDGOUSDT"]
    fee = money(fx["fee"])
    require(
        0 <= fee < 1 and 0 <= fraction <= 1 and target > 0,
        "contribution_budget_unverified",
    )
    available = book.free_cash + book.usdgo * money(fx["bid"]) * (1 - fee) / (1 + fee)
    return {
        "capital_base_usdt": str(book.deployable),
        "base_fraction": str(money(cfg["buy_fraction"])),
        "p75_multiplier": str(multiplier),
        "effective_fraction": str(fraction),
        "target_price_usdt": str(target),
        "available_usdt_estimate": str(available),
        "target_budget_usdt": str(max(money(0), min(available, book.deployable * fraction))),
        "requires_original_signal": True,
    }


def preview(execution, strategy, coin, amount, now, *, holding_id=None):
    with localcontext() as context:
        context.prec = 60
        return project(execution, strategy, coin, amount, now, holding_id=holding_id)


def project(execution, strategy, coin, amount, now, *, holding_id=None):
    observed = ready(execution, strategy, now)
    require(coin == "USDGO", "contribution_selected_coin_unsupported")
    require(
        isinstance(amount, str) and re.fullmatch(r"\d{1,15}(\.\d{1,8})?", amount),
        "contribution_invalid_amount",
    )
    quantity = money(amount)
    require(quantity > 0, "contribution_invalid_amount")
    positions = execution["holdings"]
    eligible = [
        (key, row)
        for key, row in positions.items()
        if row["coin"] == coin
        and money(row["unassigned_amount"]) - money(row.get("reserved_amount", 0)) > 0
        and (holding_id is None or key == holding_id)
    ]
    require(len(eligible) == 1, "contribution_selected_holding_missing_or_ambiguous")
    hid, holding = eligible[0]
    rules = [p for p in execution["products"] if p["coin"] == coin]
    require(
        len(rules) == 1 and rules[0]["product_id"] == holding["product_id"],
        "contribution_selected_product_mismatch",
    )
    outside = money(holding["unassigned_amount"])
    reserved = money(holding.get("reserved_amount", 0))
    require(
        0 <= reserved <= outside and quantity <= outside - reserved,
        "contribution_exceeds_selected_external_amount",
    )
    before = Ledger.restore(execution["book"])
    mapped = sum(
        (
            money(p["principal"]) + money(p["interest"])
            for p in positions.values()
            if p["coin"] == coin
        ),
        money(0),
    )
    require(
        0 <= before.locked[coin] <= before.usdgo and mapped == before.locked[coin],
        "contribution_earn_location_mismatch",
    )
    trx = money(strategy["markets"]["TRXUSDT"]["bid"])
    fx = money(strategy["markets"]["USDGOUSDT"]["bid"])
    require(min(trx, fx) > 0, "contribution_quote_invalid")
    before.usdgo_mark = fx
    equity = before.equity(trx)
    require(equity > 0 and before.fund_units > 0, "contribution_requires_positive_nav")
    value, nav = quantity * fx, equity / before.fund_units
    after = Ledger.restore(before.state())
    after.usdgo += quantity
    after.usdgo_cost += value
    after.locked[coin] += quantity
    after.external_flows += value
    after.fund_units += value / nav
    hypothetical = deepcopy(positions)
    hypothetical[hid]["principal"] = str(money(holding["principal"]) + quantity)
    hypothetical[hid]["unassigned_amount"] = str(outside - quantity)
    ownership = {
        key: {
            "coin": p["coin"],
            "product_id": p["product_id"],
            "strategy": str(money(p["principal"]) + money(p["interest"])),
            "external": str(money(p["unassigned_amount"])),
        }
        for key, p in hypothetical.items()
    }
    prior = execution["income_accounting"]["checkpoints"]
    require(prior and prior[-1]["timestamp"] <= now, "contribution_checkpoint_conflict")
    checkpoint = {
        "timestamp": now,
        "positions": ownership,
        "capital_transfers": {hid: str(quantity)},
    }
    profit = equity - before.initial - before.external_flows
    after_equity = after.equity(trx)
    after_profit = after_equity - after.initial - after.external_flows
    require(
        abs(after_profit - profit) < money("1e-40")
        and abs(after_equity / after.fund_units - nav) < money("1e-40")
        and before.lots == after.lots
        and before.orders == after.orders,
        "contribution_projection_not_conserved",
    )

    def summary(book):
        current_equity = book.equity(trx)
        return {
            "initial_capital_usdt": str(book.initial),
            "net_capital_usdt": str(book.initial + book.external_flows),
            "equity_usdt": str(current_equity),
            "profit_usdt": str(current_equity - book.initial - book.external_flows),
            "usdgo_quantity": str(book.usdgo),
            "usdgo_cost_usdt": str(book.usdgo_cost),
            "locked_usdgo": str(book.locked[coin]),
            "trx_cost_usdt": str(book.cost),
            "fund_units": str(book.fund_units),
            "nav": str(current_equity / book.fund_units),
            "buy": buy_budget(book, strategy, now),
        }

    first_period = (now + HOUR - 1) // HOUR * HOUR
    result = {
        "schema": "trx-selected-usdgo-contribution-preview-v1",
        "preview_only": True,
        "execution_enabled": False,
        "account_id": execution["account_id"],
        "created_at": now,
        "observed_at": observed,
        "expires_at": observed + 60000,
        "performance_started_at": strategy["activated_at"],
        "selection": {
            "coin": coin,
            "amount": str(quantity),
            "location": "earn",
            "holding_id": hid,
            "product_id": holding["product_id"],
            "mark_usdt": str(fx),
            "value_usdt": str(value),
            "external_earn_before": str(outside),
            "external_earn_after": str(outside - quantity),
        },
        "before": summary(before),
        "after": summary(after),
        "excluded": {
            "all_external_spot": deepcopy(execution["unassigned_offsets"]),
            "remaining_external_earn": {
                key: {"coin": p["coin"], "amount": p["unassigned_amount"]}
                for key, p in hypothetical.items()
            },
            "rewards": "remain_outside_strategy",
            "previously_credited_interest": "unchanged_not_reclassified_as_strategy_income",
        },
        "interest": {
            "basis": "hourly-effective-principal-v1",
            "hypothetical_checkpoint": checkpoint,
            "first_complete_period_start": first_period,
            "first_complete_period_end": first_period + HOUR,
            "before_and_transition_periods": "retain_prior_ownership_even_if_paid_later",
            "after_complete_period": "allocate_using_new_ownership_from_original_native_bills",
            "historical_allocations_unchanged": True,
        },
        "unchanged": {
            "initial_capital": True,
            "trx_lots_and_cost": True,
            "performance_start": True,
            "historical_income": True,
            "cumulative_profit": True,
            "nav": True,
            "usdt_and_trx_outside_strategy": True,
            "strategy_parameters": True,
        },
        "conditions": [
            "read_only_saved_reconciled_account_snapshot",
            "no_first_purchase",
            "no_redemption_or_subscription",
            "no_confirmation_or_execution_endpoint",
            "refresh_prices_and_ownership_before_any_future_contribution",
        ],
    }
    result["preview_id"] = fingerprint(result)
    return result
