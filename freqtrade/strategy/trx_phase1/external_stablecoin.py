"""External stablecoin flows use the native order journal and never debit the strategy book."""

from copy import deepcopy

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError, identifier
from freqtrade.strategy.trx_phase1.live_execution import digest


ZERO = money(0)
COINS = ("USDT", "USDC", "USDGO")
SOURCE = "external_stablecoin"


def enabled(state, policy):
    return (
        (policy or {}).get("external_rebalance") is True
        and not state.get("performance_basis")
        and not state.get("usdgo_reserve")
    )


def authorize(execution):
    state = execution.state
    if not enabled(state, execution.stablecoin_policy):
        raise ExecutionError("external_stablecoin_policy_not_enabled")
    if state.get("pending_income") or state.get("income_accounting", {}).get("status") != "active":
        raise ExecutionError("external_stablecoin_income_unresolved")


def holding_amount(position):
    return money(position["unassigned_amount"]) - money(position.get("reserved_amount") or 0)


def spendable(execution, balances, book, coin):
    outside = money(execution.state["unassigned_offsets"][coin])
    available = money(balances[coin]["available"]) - execution.owned_spot(book)[coin]
    return max(ZERO, min(outside, available))


def interest_sources(execution, amount, now):
    # The legacy USDGO interest reinvestment can never spend a receipt twice.
    from freqtrade.strategy.trx_phase1.external_income_reinvestment import (
        available_sources,
        read_policy,
    )

    if amount <= 0 or (policy := read_policy(execution, now)) is None:
        return {}
    sources = {}
    for key, available in available_sources(execution.state, policy, now):
        consumed = min(amount, available)
        if consumed:
            sources[key] = str(consumed)
            amount -= consumed
        if not amount:
            break
    return sources


def stage(execution, op, deltas, now, bill_ids=()):
    # A previously submitted operation must still reconcile after the
    # rebalance switch is turned off.
    if not execution.stablecoin_policy:
        raise ExecutionError("stablecoin_policy_not_enabled")
    state = deepcopy(execution.state)
    offsets = state["unassigned_offsets"]
    for coin, delta in deltas.items():
        if coin not in COINS:
            raise ExecutionError("external_stablecoin_currency_scope")
        new = money(offsets[coin]) + delta
        if new < 0:
            raise ExecutionError("external_stablecoin_strategy_funds_protected")
        offsets[coin] = str(new)
    business_id = op["exchange_id"]
    key = "external-stablecoin:" + business_id
    record = {
        "classification": "verified_external_movement",
        "source": SOURCE,
        "operation_kind": op["kind"],
        "business_id": business_id,
        "timestamp": now,
        "bill_ids": list(bill_ids),
        "deltas": {coin: str(delta) for coin, delta in deltas.items()},
        "external_interest_sources": interest_sources(execution, -deltas.get("USDGO", ZERO), now),
    }
    if key in state.get("verified_external_movements", {}):
        raise ExecutionError("external_stablecoin_receipt_already_settled")
    state.setdefault("verified_external_movements", {})[key] = record
    state["receipts"]["verified-external-movement:" + key] = digest(record)
    state["external_usdc"]["spot"] = offsets["USDC"]
    return state, record


def check_and_commit(execution, state, now, holding=None):
    original = execution.state
    execution.state = state
    try:
        book = Ledger.restore(state["book"])
        book, positions, receipts, records, accounting = execution.income_candidate(
            book, state["holdings"], state["receipts"], now
        )
        if holding is not None:
            key, observed = holding
            position = positions[key]
            expected = sum(
                (money(position[k]) for k in ("principal", "interest", "unassigned_amount")),
                ZERO,
            )
            if expected != money(observed["amount"] if observed else 0):
                execution.state = original
                return False
        execution.check_wallet(execution.broker.balances(), book, records)
        execution.commit_candidate(book, positions, receipts, records, now, accounting)
        execution.state["external_usdc"]["savings"] = str(
            sum(
                (money(p["unassigned_amount"]) for p in positions.values() if p["coin"] == "USDC"),
                ZERO,
            )
        )
        return True
    except BaseException:
        execution.state = original
        raise


def settle_trade(execution, step, op, order_id, status, fills, receipts, deltas, base, quote, now):
    op.update(exchange_id=order_id, fills={f["id"]: f for f in fills})
    if op["status"] == "needs_review" and "external-stablecoin:" + order_id in execution.state.get(
        "verified_external_movements", {}
    ):
        return  # The terminal fill was already booked; retain the review.
    if status not in {"filled", "cancelled"}:
        # Open and partial orders retain the reservation and do not fund another leg.
        op["status"] = "open"
        execution.state["operation"] = op
        if now >= execution.state["flow"]["expires_at"]:
            execution.cancel_open_order(op, step, order_id, now)
        execution.save()
        return
    if not base:
        execution.review("partial_or_minimum_receipt_not_satisfied", now)
        return
    relevant = {c: deltas[c] for c in (step.input_coin, step.output_coin)}
    original = execution.state
    state, record = stage(execution, op, relevant, now)
    state["operation"] = op
    state["receipts"].update(receipts)
    try:
        check_and_commit(execution, state, now)
        received = deltas[step.output_coin]
        execution.event("external_stablecoin_settled", now, **record)
        if abs((quote / base) / step.reference_price - 1) > step.maximum_deviation:
            execution.review("actual_fill_deviation_exceeded", now)
        elif received < step.minimum_receive or (
            status == "cancelled" and not step.continue_partial
        ):
            execution.review("partial_or_minimum_receipt_not_satisfied", now)
        else:
            execution.complete(
                {"coin": step.output_coin, "amount": str(received), "attribution": SOURCE}, now
            )
            execution.save()
    except BaseException:
        execution.state = original
        raise


def settle_earn(execution, step, op, bills, holdings, key, now):
    before = execution.state
    amount = money(op["amount"])
    delta = -amount if step.kind == "subscribe" else amount
    bill_ids = [identifier(b["billId"]) for b in bills]
    state, record = stage(execution, op, {step.coin: delta}, now, bill_ids)
    original = op["before_holdings"].get(key)
    observed = holdings.get(key)
    if observed and (observed["coin"] != step.coin or observed["product_id"] != step.product_id):
        raise ExecutionError("savings_holding_identity_mismatch")
    positions = state["holdings"]
    if key not in positions:
        if original and money(original["amount"]):
            raise ExecutionError("external_stablecoin_holding_unmapped")
        positions[key] = {
            "coin": step.coin,
            "product_id": step.product_id,
            "principal": "0",
            "interest": "0",
            "unassigned_amount": "0",
            "owned_since": op["created_at"],
            "subscription_ids": [],
            "lots": {},
        }
    position = positions[key]
    if position["coin"] != step.coin or position["product_id"] != step.product_id:
        raise ExecutionError("savings_holding_identity_mismatch")
    if delta > holding_amount(position):
        raise ExecutionError("external_stablecoin_strategy_funds_protected")
    position["unassigned_amount"] = str(money(position["unassigned_amount"]) - delta)
    if step.kind == "subscribe":
        position["subscription_ids"].append(op["exchange_id"])
    used = state.setdefault("account_movement_bill_receipts", {})
    for bill, bill_id in zip(bills, bill_ids, strict=True):
        if bill_id in used:
            raise ExecutionError("external_stablecoin_receipt_already_settled")
        used[bill_id] = digest(execution.interest_bill(bill))
    try:
        if not check_and_commit(execution, state, now, (key, observed)):
            return
        execution.state["operation"]["bill_ids"] = bill_ids
        execution.event("external_stablecoin_settled", now, **record)
        execution.complete(
            {"coin": step.coin, "amount": str(amount), "holding_id": key, "attribution": SOURCE},
            now,
        )
        execution.save()
    except BaseException:
        execution.state = before
        raise
