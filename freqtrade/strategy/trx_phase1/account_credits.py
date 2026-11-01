"""Quarantine proven account credits outside strategy funds and profit."""

from freqtrade.strategy.trx_phase1.account_movement_receipts import (
    external_usdt_income_allowed,
    known_movement_bill,
)
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_broker import COINS, DAY, ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest
from freqtrade.strategy.trx_phase1.reconciliation_wait import wallet_error


UNIT = money("0.00000001")


def credit_kinds(state, coin="USDT"):
    kinds = {"ACTIVITY_ASSET_REWARD_USER_IN": ("other", "outside_strategy_activity_reward")}
    if coin != "USDT":
        return kinds
    kinds["REBATE_REWARDS"] = ("other", "strategy_order_link_unverified")
    kinds["CASH_COUPON_ACTIVITY_USER_IN"] = ("other", "outside_strategy_cash_coupon_reward")
    kinds["REBATE_COUPON_ACTIVITY_USER_IN"] = ("other", "outside_strategy_rebate_coupon_reward")
    basis = state.get("performance_basis")
    if (
        basis
        and basis.get("sha256") == digest({k: v for k, v in basis.items() if k != "sha256"})
        and basis.get("started_at") == state["created_at"]
        and basis.get("income_policy", {}).get("usdt_is_outside_strategy") is True
        and any(
            row.get("coin") == "USDT"
            for row in basis.get("manifest", {}).get("excluded_assets", [])
        )
    ):
        kinds["BATCH_INTEREST_USER_IN"] = ("financial", "outside_strategy_usdt_interest")
    if external_usdt_income_allowed(state):
        kinds["BATCH_INTEREST_USER_IN"] = ("financial", "outside_strategy_usdt_interest")
    return kinds


def quarantine(execution, book, receipts, records, now, *, bills_by_coin=None):
    """Stage verified external credits; the caller commits only after full wallet checks."""
    start = max(
        execution.state["created_at"], execution.state.get("interest_checked_at", now) - DAY
    )
    available = dict(bills_by_coin or {})
    for coin in COINS:
        if coin not in available:
            available[coin] = execution.broker.bills(coin, start, now)
    for coin in COINS:
        receipts, records = quarantine_coin(
            execution, book, receipts, records, now, coin, available[coin]
        )
    return receipts, records


def quarantine_coin(execution, book, receipts, records, now, coin, bills):
    kinds = credit_kinds(execution.state, coin)
    new = []
    for raw in bills:
        if known_movement_bill(execution, raw) or known_account_credit(execution, receipts, raw):
            continue
        kind = raw.get("businessType")
        if kind not in kinds:
            continue
        row = execution.interest_bill(raw)
        if (
            row["coin"] != coin
            or row["group"] != kinds[kind][0]
            or money(row["amount"]) <= 0
            or money(row["fees"]) != 0
            or not execution.state["created_at"] <= row["timestamp"] <= now
            or money(row["amount"]) % UNIT
        ):
            raise ExecutionError("account_credit_evidence_required")
        rebate_coupon_link(row, receipts)
        key = "account-credit:" + row["bill_id"]
        fingerprint = digest(row)
        baseline = execution.state.get("account_credit_baseline_bills", {})
        previous = receipts.get(key, baseline.get(row["bill_id"]))
        if previous is not None:
            if previous != fingerprint:
                raise ExecutionError("conflicting_account_credit")
            continue
        new.append((key, row, fingerprint))
    if not new:
        return receipts, records
    validate_movements(execution, receipts, records, coin, bills, kinds)
    balances = execution.broker.balances()
    offsets, _ = execution.income_adjustments(records)
    actual = sum((money(value) for value in balances[coin].values()), money(0))
    nominal = sum((money(row["amount"]) for _, row, _ in new), money(0))
    expected = execution.owned_spot(book)[coin] + money(offsets[coin]) + nominal
    residual = actual - expected
    # Bills have eight decimals while the wallet may have more. Keep this bounded
    # residual in external funds, explicitly labelled; never round the ledger.
    # Preserve the existing bounded USDT wallet precision rule. Activity rewards
    # in the other native coins require an exact match, without absorbing gaps.
    if (coin == "USDT" and abs(residual) >= UNIT * len(new)) or (coin != "USDT" and residual):
        raise wallet_error("account_credit_wallet_evidence_required", coin, actual, expected)

    receipts, records = dict(receipts), list(records)
    for index, (key, row, fingerprint) in enumerate(new):
        tail = residual if index == len(new) - 1 else money(0)
        record = {
            **row,
            "reported_amount": row["amount"],
            "precision_residual": str(tail),
            "offset_delta": str(money(row["amount"]) + tail),
            "attribution": "external_account_credit",
            "destination": "spot",
            "recorded_at": now,
            "reason": kinds[row["type"]][1],
            **rebate_coupon_link(row, receipts),
        }
        receipts[key] = fingerprint
        records.append((key, record))
    return receipts, records


def known_account_credit(execution, receipts, raw):
    """A verified prior credit stays settled after its Earn position is closed."""
    key = "account-credit:" + str(raw.get("billId"))
    saved = execution.state.get("quarantined_account_credits", {}).get(key)
    if saved is None:
        return False
    row = execution.interest_bill(raw)
    if (
        receipts.get(key) != digest(row)
        or any(saved.get(field) != value for field, value in row.items())
        or saved.get("attribution") != "external_account_credit"
        or saved.get("destination") != "spot"
        or money(row["amount"]) <= 0
        or money(row["fees"]) != 0
    ):
        raise ExecutionError("conflicting_account_credit")
    return True


def rebate_coupon_link(row, receipts):
    """A coupon stays external; a committed fill proves its transaction link."""
    if row["type"] != "REBATE_COUPON_ACTIVITY_USER_IN":
        return {}
    matches = [
        key.split(":")
        for key in receipts
        if key.startswith("spot:") and key.rsplit(":", 1)[-1] == row["order_id"]
    ]
    if len(matches) != 1 or len(matches[0]) != 3:
        raise ExecutionError("rebate_coupon_trade_evidence_required")
    return {"linked_order_id": matches[0][1], "linked_fill_id": matches[0][2]}


def validate_movements(execution, receipts, records, coin, bills, kinds):
    known_fills = {key.rsplit(":", 1)[-1] for key in receipts if key.startswith("spot:")}
    settled_bills = {
        str(bill)
        for event in execution.state.get("events", [])
        if event.get("kind") == "operation_settled"
        for bill in event.get("bill_ids", [])
    }
    credited_bills = {str(credit["bill_id"]) for _, credit in records if credit.get("bill_id")}
    operation = execution.state.get("operation") or {}
    for raw in bills:
        if known_movement_bill(execution, raw) or known_account_credit(execution, receipts, raw):
            continue
        if raw.get("businessType") in kinds:
            continue
        if raw.get("groupType") == "transaction" and str(raw.get("bizOrderId")) in known_fills:
            continue
        if coin != "USDT":
            bill_id = str(raw.get("billId"))
            if raw.get("groupType") == "financial" and (
                (
                    raw.get("businessType") == "BATCH_INTEREST_USER_IN"
                    and (
                        "income-bill:" + bill_id in receipts
                        or bill_id in execution.state.get("income_baseline_bills", {})
                        or bill_id in credited_bills
                    )
                )
                or bill_id in settled_bills
                or (
                    operation.get("kind") in {"subscribe", "redeem"}
                    and operation.get("step", {}).get("coin") == coin
                    and operation.get("exchange_id") == str(raw.get("bizOrderId"))
                )
            ):
                continue
        raise ExecutionError(
            "unrecognized_usdt_movement" if coin == "USDT" else "unrecognized_account_movement"
        )
