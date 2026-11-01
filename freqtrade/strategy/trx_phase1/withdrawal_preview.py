"""Read-only USDGO capital withdrawal projection; no transfer or journal writer."""

import hashlib
import json
from decimal import ROUND_DOWN, ROUND_UP, localcontext


if __package__:
    from .ledger import Ledger, money
else:
    from ledger import Ledger, money


class WithdrawalPreviewError(ValueError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def ready(execution, strategy, now):
    basis = execution.get("performance_basis")
    if (
        not basis
        or basis != strategy.get("performance_basis")
        or basis.get("sha256") != fingerprint({k: v for k, v in basis.items() if k != "sha256"})
        or execution["account_id"] != strategy["account_id"]
        or basis["started_at"] != strategy.get("activated_at")
        or execution["created_at"] != basis["started_at"]
        or basis["started_at"] > now
    ):
        raise WithdrawalPreviewError("withdrawal_start_basis_required")
    if (
        not strategy["startup_complete"]
        or strategy["phase"] not in {"observing", "paused", "prepared"}
        or strategy.get("error")
        or execution.get("error")
        or strategy.get("intent")
        or strategy["engine"].get("pending")
        or execution.get("operation")
        or (execution.get("flow") and execution["flow"]["status"] not in {"completed", "expired"})
    ):
        raise WithdrawalPreviewError("withdrawal_requires_idle_strategy")
    if execution.get("income_accounting", {}).get("status") != "active":
        raise WithdrawalPreviewError("withdrawal_income_unresolved")
    stamps = [execution.get("interest_checked_at", 0)]
    stamps += [
        strategy.get("markets", {}).get(symbol, {}).get("timestamp", 0)
        for symbol in ("TRXUSDT", "USDGOUSDT")
    ]
    if any(not stamp or not -5000 <= now - stamp <= 60000 for stamp in stamps):
        raise WithdrawalPreviewError("withdrawal_snapshot_stale")
    return min(stamps)


def sources(execution, amount):
    book = Ledger.restore(execution["book"])
    free = book.usdgo - book.locked["USDGO"]
    positions = [(key, row) for key, row in execution["holdings"].items() if row["coin"] == "USDGO"]
    if (
        not 0 <= free <= book.usdgo
        or sum((money(row["principal"]) for _, row in positions), money(0)) != book.locked["USDGO"]
    ):
        raise WithdrawalPreviewError("withdrawal_location_mismatch")
    rules = [p for p in execution["products"] if p["coin"] == "USDGO"]
    if len(rules) != 1:
        raise WithdrawalPreviewError("withdrawal_redemption_rules_required")
    rule = rules[0]
    decimals = rule.get("redemption_decimals")
    if (
        type(decimals) is not int
        or not 0 <= decimals <= 8
        or not rule.get("redemption_evidence")
        or money(rule.get("redemption_minimum") or 0) <= 0
    ):
        raise WithdrawalPreviewError("withdrawal_redemption_rules_required")
    step = money(10) ** -decimals
    spot = min(amount, free)
    needed = amount - spot
    plan = []
    for key, row in positions:
        if needed <= 0:
            break
        if row["product_id"] != rule["product_id"]:
            raise WithdrawalPreviewError("withdrawal_location_mismatch")
        available = money(row["principal"]).quantize(step, rounding=ROUND_DOWN)
        redeem = max(money(rule["redemption_minimum"]), needed.quantize(step, rounding=ROUND_UP))
        redeem = min(available, redeem)
        if redeem < money(rule["redemption_minimum"]):
            continue
        plan.append(
            {
                "kind": "redeem",
                "coin": "USDGO",
                "holding_id": key,
                "product_id": rule["product_id"],
                "amount": str(redeem),
            }
        )
        needed -= redeem
    if needed > 0:
        raise WithdrawalPreviewError("withdrawal_redeemable_usdgo_insufficient")
    redeemed = sum((money(row["amount"]) for row in plan), money(0))
    return {
        "strategy_spot_used": str(spot),
        "strategy_earn_to_redeem": str(redeemed),
        "strategy_spot_remaining": str(free + redeemed - amount),
        "redemptions": plan,
        "redemption_rule": rule,
    }


def budget(book, strategy):
    engine, markets = strategy["engine"], strategy["markets"]
    cfg = engine["config"]
    dynamic = cfg.get("buy_quantity_dynamic")
    if dynamic is True or (
        dynamic is None
        and cfg["variant"]
        in {"dynamic_quantity", "dynamic_size", "dynamic_quantity_period", "dynamic_period"}
    ):
        raise WithdrawalPreviewError("withdrawal_dynamic_sizing_unsupported")
    reference = money(engine["anchor"])
    recovery = money(engine.get("recovery_reference", 0))
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
    p75 = (strategy.get("price_reference") or {}).get("value")
    if cfg["price_position_enabled"]:
        if p75 is None:
            raise WithdrawalPreviewError("withdrawal_price_reference_required")
        if target > money(p75):
            multiplier = money(cfg["price_position_buy_multiplier"])
    fraction = min(money(1), money(cfg["buy_fraction"]) * multiplier)
    if recovering:
        fraction = min(fraction, money(cfg["recovery_buy_fraction"]))
    fx = markets["USDGOUSDT"]
    fee = money(fx["fee"])
    if not 0 <= fee < 1 or not 0 < fraction <= 1 or target <= 0:
        raise WithdrawalPreviewError("withdrawal_budget_unverified")
    available = book.free_cash + book.usdgo * money(fx["bid"]) * (1 - fee) / (1 + fee)
    return {
        "capital_base_usdt": str(book.deployable),
        "effective_fraction": str(fraction),
        "target_price_usdt": str(target),
        "available_usdt_estimate": str(available),
        "target_budget_usdt": str(min(available, book.deployable * fraction)),
    }


def preview(execution, strategy, amount, now):
    with localcontext() as context:
        context.prec = 50
        return project(execution, strategy, amount, now)


def project(execution, strategy, amount, now):
    observed = ready(execution, strategy, now)
    try:
        quantity = money(amount)
        if quantity <= 0 or quantity != quantity.quantize(money("0.00000001")):
            raise ValueError
    except (ValueError, ArithmeticError):
        raise WithdrawalPreviewError("withdrawal_invalid_amount") from None
    before = Ledger.restore(execution["book"])
    if quantity > before.usdgo:
        raise WithdrawalPreviewError("withdrawal_exceeds_strategy_usdgo")
    trx = money(strategy["markets"]["TRXUSDT"]["bid"])
    fx = money(strategy["markets"]["USDGOUSDT"]["bid"])
    if min(trx, fx) <= 0:
        raise WithdrawalPreviewError("withdrawal_quote_invalid")
    plan = sources(execution, quantity)
    value = quantity * fx
    before.usdgo_mark = fx
    equity = before.equity(trx)
    if before.fund_units <= 0 or equity <= value or before.deployable <= value:
        raise WithdrawalPreviewError("withdrawal_remaining_capital_nonpositive")
    cost = before.usdgo_cost * quantity / before.usdgo
    after = Ledger.restore(before.state())
    after.usdgo -= quantity
    after.usdgo_cost -= cost
    after.external_flows -= value
    after.fund_units *= (equity - value) / equity
    # This is a hypothetical position after redemption and withdrawal, never saved.
    after.locked["USDGO"] -= money(plan["strategy_earn_to_redeem"])
    before_budget, after_budget = budget(before, strategy), budget(after, strategy)
    profit = equity - before.initial - before.external_flows
    after_equity = after.equity(trx)
    after_profit = after_equity - after.initial - after.external_flows
    if abs(after_profit - profit) > money("1e-35"):
        raise WithdrawalPreviewError("withdrawal_projection_not_conserved")
    nav = equity / before.fund_units
    result = {
        "schema": "trx-usdgo-withdrawal-preview-v1",
        "preview_only": True,
        "execution_enabled": False,
        "account_id": execution["account_id"],
        "created_at": now,
        "observed_at": observed,
        "expires_at": observed + 60000,
        "performance_started_at": execution["performance_basis"]["started_at"],
        "performance_basis_sha256": execution["performance_basis"]["sha256"],
        "withdrawal": {
            "coin": "USDGO",
            "quantity": str(quantity),
            "mark_usdt": str(fx),
            "value_usdt": str(value),
            "removed_usdgo_cost_usdt": str(cost),
        },
        "sources": plan,
        "before": {
            "initial_capital_usdt": str(before.initial),
            "net_capital_usdt": str(before.initial + before.external_flows),
            "equity_usdt": str(equity),
            "profit_usdt": str(profit),
            "usdgo_quantity": str(before.usdgo),
            "usdgo_cost_usdt": str(before.usdgo_cost),
            "fund_units": str(before.fund_units),
            "nav": str(nav),
            "usdt_cash": str(before.cash),
            "buy": before_budget,
        },
        "after": {
            "initial_capital_usdt": str(after.initial),
            "net_capital_usdt": str(after.initial + after.external_flows),
            "equity_usdt": str(after_equity),
            "profit_usdt": str(after_profit),
            "usdgo_quantity": str(after.usdgo),
            "usdgo_cost_usdt": str(after.usdgo_cost),
            "fund_units": str(after.fund_units),
            "nav": str(after_equity / after.fund_units),
            "usdt_cash": str(after.cash),
            "buy": after_budget,
        },
        "retained_withdrawn_usdgo_valuation_profit_usdt": str(value - cost),
        "buy_budget_reduction_usdt": str(
            money(before_budget["target_budget_usdt"]) - money(after_budget["target_budget_usdt"])
        ),
        "unchanged": {
            "trx_lots_and_cost": True,
            "initial_capital": True,
            "performance_start": True,
            "cumulative_profit": True,
            "nav": True,
            "usdt_outside_strategy": True,
            "historical_interest_attribution": True,
        },
        "conditions": [
            "read_only_saved_exchange_reconciliation_and_quotes",
            "actual_execution_requires_fresh_exchange_evidence",
            "redeem_excess_remains_strategy_spot",
            "withdrawal_is_capital_outflow_not_loss",
            "no_first_purchase",
            "no_confirmation_or_execution_endpoint",
        ],
        "pending_income_excluded_count": len(execution.get("pending_income", {})),
        "new_fees_not_assumed": True,
    }
    result["preview_id"] = fingerprint(result)
    return result
