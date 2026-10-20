"""Operator-approved account credits become capital, never attributed fee income.

The caller must serialize with the live execution journal and supply a fresh,
independently observed USDT wallet. No exchange writes or strategy-state writes.
"""

from copy import deepcopy

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest


def candidate(state, policy, markets, wallet_usdt, now):
    """Return one auditable candidate; leave the input untouched on every path."""
    fingerprint = digest(policy)
    previous = state.get("asset_contributions", {}).get(fingerprint)
    if previous:
        return deepcopy(state), deepcopy(previous)
    if (
        policy.get("version") != "account-credit-capital-v1"
        or policy.get("account_id") != state["account_id"]
        or policy.get("ledger_created_at") != state["created_at"]
        or not policy.get("approved_at")
        or not policy["approved_at"] <= now < policy.get("expires_at", 0)
        or not policy.get("credits")
        or len({r["receipt_id"] for r in policy["credits"]}) != len(policy["credits"])
    ):
        raise ExecutionError("account_credit_capital_policy_required")
    if (
        state.get("error")
        or state.get("operation")
        or state.get("pending_income")
        or (state.get("flow") and state["flow"]["status"] not in {"completed", "expired", "failed"})
    ):
        raise ExecutionError("account_credit_capital_requires_idle_execution")
    used = {
        row["receipt_id"]
        for record in state.get("asset_contributions", {}).values()
        for row in record.get("source_credits", [])
    }
    sources = []
    for approved in policy["credits"]:
        key = approved["receipt_id"]
        row = state.get("quarantined_account_credits", {}).get(key)
        receipt = (
            {
                k: row.get(k)
                for k in (
                    "bill_id",
                    "order_id",
                    "coin",
                    "group",
                    "type",
                    "amount",
                    "fees",
                    "timestamp",
                )
            }
            if row
            else None
        )
        if (
            key in used
            or not row
            or key not in state["receipts"]
            or state["receipts"][key] != digest(receipt)
            or row["bill_id"] != approved["bill_id"]
            or row.get("coin") != "USDT"
            or row.get("type") != "REBATE_REWARDS"
            or row.get("group") != "other"
            or money(row["fees"]) != 0
            or row.get("attribution") != "external_account_credit"
            or money(row["offset_delta"]) != money(approved["amount"])
            or money(row["offset_delta"]) <= 0
        ):
            raise ExecutionError("account_credit_capital_evidence_mismatch")
        sources.append(
            {"receipt_id": key, "bill_id": row["bill_id"], "amount": row["offset_delta"]}
        )
    for symbol in ("TRXUSDT", "USDGOUSDT"):
        quote = markets[symbol]
        if not -5000 <= now - quote["timestamp"] <= 60000 or money(quote["bid"]) <= 0:
            raise ExecutionError("account_credit_capital_quote_stale")
    work = deepcopy(state)
    book = Ledger.restore(work["book"])
    outside = money(work["unassigned_offsets"]["USDT"])
    amount = sum((money(r["amount"]) for r in sources), money(0))
    if outside < amount or book.free_cash + outside != money(wallet_usdt):
        raise ExecutionError("account_credit_capital_wallet_mismatch")
    units = book.fund_units
    book.usdgo_mark = money(markets["USDGOUSDT"]["bid"])
    book.external_flow("capital-contribution:" + fingerprint, amount, markets["TRXUSDT"]["bid"])
    work["unassigned_offsets"]["USDT"] = str(outside - amount)
    work["book"] = book.state()
    record = {
        "review_hash": fingerprint,
        "timestamp": now,
        "basis": "contribution_market_value",
        "assets": [
            {
                "coin": "USDT",
                "amount": str(amount),
                "mark": "1",
                "value": str(amount),
                "locations": [{"location": "spot", "amount": str(amount)}],
            }
        ],
        "value": str(amount),
        "fund_units_before": str(units),
        "fund_units_after": str(book.fund_units),
        "source_credits": sources,
        "classification": "operator_capital_unlinked_rebate",
    }
    if policy.get("authorization_policy_sha256"):
        record.update(
            classification="automatic_capital_unlinked_rebate",
            authorization_policy_sha256=policy["authorization_policy_sha256"],
            authorization_approved_at=policy["approved_at"],
        )
    work.setdefault("asset_contributions", {})[fingerprint] = record
    work["events"].append(
        {"kind": "capital_contribution_recorded", "timestamp": now, "contribution": record}
    )
    return work, deepcopy(record)
