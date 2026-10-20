"""Quarantine native account rebates outside strategy funds and profit."""

from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_broker import DAY, ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest


UNIT = money("0.00000001")


def quarantine(execution, book, receipts, records, now):
    start = max(
        execution.state["created_at"], execution.state.get("interest_checked_at", now) - DAY
    )
    bills = execution.broker.bills("USDT", start, now)
    new = []
    for raw in bills:
        if raw.get("businessType") != "REBATE_REWARDS":
            continue
        row = execution.interest_bill(raw)
        if (
            row["coin"] != "USDT"
            or row["group"] != "other"
            or money(row["amount"]) <= 0
            or money(row["fees"]) != 0
            or not execution.state["created_at"] <= row["timestamp"] <= now
            or money(row["amount"]) % UNIT
        ):
            raise ExecutionError("account_credit_evidence_required")
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
    known_fills = {key.rsplit(":", 1)[-1] for key in receipts if key.startswith("spot:")}
    for raw in bills:
        if raw.get("businessType") == "REBATE_REWARDS":
            continue
        if raw.get("groupType") != "transaction" or str(raw.get("bizOrderId")) not in known_fills:
            raise ExecutionError("unrecognized_usdt_movement")
    balances = execution.broker.balances()
    offsets, _ = execution.income_adjustments(records)
    actual = sum((money(value) for value in balances["USDT"].values()), money(0))
    nominal = sum((money(row["amount"]) for _, row, _ in new), money(0))
    expected = execution.owned_spot(book)["USDT"] + money(offsets["USDT"]) + nominal
    residual = actual - expected
    # Bills have eight decimals while the wallet may have more. Keep this bounded
    # residual in external funds, explicitly labelled; never round the ledger.
    if abs(residual) >= UNIT * len(new):
        raise ExecutionError("account_credit_wallet_evidence_required")
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
            "reason": "strategy_order_link_unverified",
        }
        receipts[key] = fingerprint
        records.append((key, record))
    return receipts, records
