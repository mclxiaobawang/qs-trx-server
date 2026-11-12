"""Reinvest verified external USDGO interest without admitting strategy capital.

Planning is pure. Only the existing serialized execution workflow may submit an
explicit external-interest subscription. Acknowledged results, native bills,
wallets and the holding location must all reconcile before committing it.
"""

import json
from copy import deepcopy

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError, rounded
from freqtrade.strategy.trx_phase1.live_execution import digest
from freqtrade.strategy.trx_phase1.reconciliation_wait import verified_income_candidate


VERSION = "external-usdgo-interest-reinvestment-v1"
POLICY_FILE = "external-usdgo-interest-reinvestment.json"
SOURCE = "external_interest"


class ReinvestmentDeferred(ExecutionError):
    pass


def ownership_baseline(state):
    """New manual ownership decisions require a fresh unspent-source review."""
    contributions = {}
    for key, record in state.get("asset_contributions", {}).items():
        assets = record.get("assets", [])
        if (
            record.get("classification") == "automatic_capital_unlinked_rebate"
            and assets
            and all(row.get("coin") == "USDT" for row in assets)
        ):
            continue
        contributions[key] = record
    return digest(
        {
            "contributions": contributions,
            "selected": state.get("selected_contribution_batches", {}),
            "manual_movements": state.get("account_movement_reconciliations", {}),
            "repairs": state.get("account_repairs", {}),
        }
    )


def validate_policy(state, policy, now):
    try:
        position = state["holdings"][policy["holding_id"]]
        valid = (
            policy["version"] == VERSION
            and policy["enabled"] is True
            and policy["account_id"] == state["account_id"]
            and policy["ledger_created_at"] == state["created_at"]
            and 0 < policy["approved_at"] <= now
            and policy["scope"] == "verified_external_USDGO_interest_only"
            and position["coin"] == "USDGO"
            and position["product_id"] == policy["product_id"]
            and money(position["unassigned_amount"]) >= 0
            and isinstance(policy["baseline_sources"], dict)
            and any(
                p["coin"] == "USDGO" and p["product_id"] == policy["product_id"]
                for p in state["products"]
            )
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ReinvestmentDeferred("external_income_policy_invalid")
    if policy.get("ownership_baseline_sha256") != ownership_baseline(state):
        raise ReinvestmentDeferred("external_income_ownership_review_required")
    return digest(policy)


def read_policy(execution, now):
    path = execution.root.parents[1] / POLICY_FILE
    try:
        if not path.exists():
            return None
        if path.stat().st_size > 1048576:
            raise ReinvestmentDeferred("external_income_policy_invalid")
        policy = json.loads(path.read_text())
    except (OSError, ValueError):
        raise ReinvestmentDeferred("external_income_policy_invalid") from None
    if not isinstance(policy, dict):
        raise ReinvestmentDeferred("external_income_policy_invalid")
    if policy.get("enabled") is False:
        return None
    validate_policy(execution.state, policy, now)
    return policy


def available_sources(state, policy, now):  # noqa: C901 - one atomic receipt/consumption scope
    """Unspent, native-receipt-backed interest; other external cash is excluded."""
    validate_policy(state, policy, now)
    accounting = state.get("income_accounting") or {}
    if accounting.get("status") != "active" or state.get("pending_income"):
        raise ReinvestmentDeferred("external_income_attribution_pending")
    spent = {}
    for exchange_id, record in state.get("external_income_reinvestments", {}).items():
        key = "external-income-subscription:" + exchange_id
        if record.get("classification") != "external_interest_reinvestment" or state[
            "receipts"
        ].get(key) != digest(record):
            raise ReinvestmentDeferred("external_income_journal_conflict")
        for source, amount in record["sources"].items():
            if money(amount) <= 0:
                raise ReinvestmentDeferred("external_income_journal_conflict")
            spent[source] = spent.get(source, money(0)) + money(amount)
    eligible = []
    for order_id, record in state.get("verified_savings_location_movements", {}).items():
        if (
            record.get("classification") != "verified_savings_location_only"
            or record.get("coin") not in {"USDGO", "USDC"}
            or state["receipts"].get("observed-savings-subscription:" + order_id) != digest(record)
            or sum(
                (money(v) for v in record.get("external_interest_sources", {}).values()), money(0)
            )
            > money(record["external_amount"])
        ):
            raise ReinvestmentDeferred("external_income_location_receipt_conflict")
        if record["coin"] == "USDC":
            if record.get("external_interest_sources"):
                raise ReinvestmentDeferred("external_income_location_receipt_conflict")
            continue  # A verified USDC location never consumes USDGO interest.
        for source, amount in record.get("external_interest_sources", {}).items():
            if money(amount) <= 0:
                raise ReinvestmentDeferred("external_income_location_receipt_conflict")
            spent[source] = spent.get(source, money(0)) + money(amount)
    from .verified_external_movements import consumed_interest

    external_used, _ = consumed_interest(state)
    for source, amount in external_used.items():
        spent[source] = spent.get(source, money(0)) + amount
    allocations = accounting.get("allocations", {})
    for key, expected in policy["baseline_sources"].items():
        if key not in allocations or digest(allocations[key]) != expected:
            raise ReinvestmentDeferred("external_income_baseline_conflict")
    for key, row in sorted(allocations.items(), key=lambda item: (item[1]["timestamp"], item[0])):
        if row.get("coin") != "USDGO":
            continue
        if key not in policy["baseline_sources"] and row["timestamp"] <= policy["approved_at"]:
            continue
        amount = money(row["external_amount"])
        receipt_key = "income-allocation:" + row["bill_id"]
        if (
            state["receipts"].get(receipt_key) != digest(row)
            or row.get("attribution") != "strategy_allocation"
            or row.get("destination") != "spot"
            or row.get("policy_sha256") != accounting.get("policy_sha256")
            or amount < 0
            or row["timestamp"] > now
            or money(row["account_amount"]) != money(row["amount"]) + amount
            or spent.get(key, money(0)) > amount
        ):
            raise ReinvestmentDeferred("external_income_receipt_conflict")
        remainder = amount - spent.pop(key, money(0))
        if remainder:
            eligible.append((key, remainder))
    if spent:
        raise ReinvestmentDeferred("external_income_journal_conflict")
    if sum((value for _, value in eligible), money(0)) > money(
        state["unassigned_offsets"]["USDGO"]
    ):
        raise ReinvestmentDeferred("external_income_location_review_required")
    return eligible


def plan_steps(execution, terms, now):
    policy = read_policy(execution, now)
    if policy is None:
        return []
    state = execution.state
    if state.get("error") or state.get("operation"):
        return []
    sources = available_sources(state, policy, now)
    terms = terms["USDGO"]
    if terms["status"] != "in_progress":
        return []
    precision = int(terms["subscribePrecision"])
    if not 0 <= precision <= 18:
        raise ReinvestmentDeferred("external_income_precision_unverified")
    amount = rounded(
        min(
            sum((value for _, value in sources), money(0)),
            money(terms["singleMaxAmount"]),
            money(terms["remainingAmount"]),
        ),
        precision,
    )
    if amount <= 0 or amount < money(terms["singleMinAmount"]):
        return []
    selected, remaining = {}, amount
    for key, value in sources:
        part = min(remaining, value)
        if part:
            selected[key] = str(part)
            remaining -= part
    return [
        {
            "kind": "subscribe",
            "coin": "USDGO",
            "product_id": policy["product_id"],
            "amount": str(amount),
            "source": SOURCE,
            "external_income_sources": selected,
            "external_income_policy_sha256": digest(policy),
        }
    ]


def authorize(execution, step, balances, book, now):
    policy = read_policy(execution, now)
    if policy is None or digest(policy) != step.external_income_policy_sha256:
        raise ReinvestmentDeferred("external_income_policy_changed")
    verify_sources(execution.state, step, policy, now)
    available = money(balances["USDGO"]["available"]) - execution.owned_spot(book)["USDGO"]
    if money(step.amount) > available:
        raise ReinvestmentDeferred("external_income_not_spendable")
    return policy


def verify_sources(state, step, policy, now):
    if (
        step.source != SOURCE
        or step.kind != "subscribe"
        or step.coin != "USDGO"
        or step.product_id != policy["product_id"]
        or digest(policy) != step.external_income_policy_sha256
        or sum(step.external_income_sources.values(), money(0)) != money(step.amount)
    ):
        raise ReinvestmentDeferred("external_income_step_invalid")
    remaining = dict(available_sources(state, policy, now))
    if any(
        q <= 0 or q > remaining.get(key, money(0))
        for key, q in step.external_income_sources.items()
    ):
        raise ReinvestmentDeferred("external_income_already_used")


def location_candidate(execution, step, op, bills, holding_id, now):
    state, policy = execution.state, op["external_income_policy"]
    validate_policy(state, policy, now)
    amount = money(op["amount"])
    if amount != money(step.amount) or holding_id != policy["holding_id"]:
        raise ReinvestmentDeferred("external_income_holding_changed")
    record = {
        "classification": "external_interest_reinvestment",
        "coin": "USDGO",
        "amount": str(amount),
        "sources": {k: str(v) for k, v in step.external_income_sources.items()},
        "exchange_id": op["exchange_id"],
        "operation_id": op["id"],
        "holding_id": holding_id,
        "policy_sha256": digest(policy),
        "bill_ids": sorted(str(b["billId"]) for b in bills),
        "recorded_at": now,
        "capital_changed": False,
        "strategy_income_changed": False,
    }
    if op["exchange_id"] in state.get("external_income_reinvestments", {}):
        raise ReinvestmentDeferred("external_income_subscription_already_settled")
    verify_sources(state, step, policy, now)
    if (
        not bills
        or any(
            b["coin"] != "USDGO"
            or b["groupType"] != "financial"
            or b["businessType"] != "SINGLE_USER_OUT"
            or str(b["bizOrderId"]) != op["exchange_id"]
            or money(b["fees"]) != 0
            or not op["created_at"] - 1000 <= int(b["cTime"]) <= now
            for b in bills
        )
        or sum((money(b["size"]) for b in bills), money(0)) != -amount
    ):
        raise ReinvestmentDeferred("external_income_bill_mismatch")
    work = deepcopy(state)
    position = work["holdings"][holding_id]
    if op["exchange_id"] in position["subscription_ids"]:
        raise ReinvestmentDeferred("external_income_subscription_already_settled")
    position["unassigned_amount"] = str(money(position["unassigned_amount"]) + amount)
    position["subscription_ids"].append(op["exchange_id"])
    work["unassigned_offsets"]["USDGO"] = str(money(work["unassigned_offsets"]["USDGO"]) - amount)
    for bill in bills:
        bid = str(bill["billId"])
        if bid in work.setdefault("account_movement_bill_receipts", {}):
            raise ReinvestmentDeferred("external_income_bill_already_used")
        work["account_movement_bill_receipts"][bid] = digest(execution.interest_bill(bill))
    work.setdefault("external_income_reinvestments", {})[op["exchange_id"]] = record
    work["receipts"]["external-income-subscription:" + op["exchange_id"]] = digest(record)
    return work, record


def settle(execution, step, op, bills, holdings, holding_id, now):
    """One existing execution lock and one native SQLite commit, no new POST."""
    original = execution.state
    work, record = location_candidate(execution, step, op, bills, holding_id, now)
    try:
        execution.state = work
        book, positions, receipts, records, accounting, _, checked_at = verified_income_candidate(
            execution, Ledger.restore(work["book"]), work["holdings"], work["receipts"], now
        )
        observed = holdings.get(holding_id)
        position = positions[holding_id]
        expected = sum(
            (money(position[k]) for k in ["principal", "interest", "unassigned_amount"]), money(0)
        )
        if not observed or money(observed["amount"]) != expected:
            execution.state = original
            return False
        if observed["coin"] != "USDGO" or observed["product_id"] != step.product_id:
            raise ReinvestmentDeferred("savings_holding_identity_mismatch")
        for key in [
            "initial",
            "external_flows",
            "fund_units",
            "lots",
            "orders",
            "realized",
            "fees",
            "fee_rebates",
        ]:
            if book.state()[key] != original["book"][key]:
                raise ReinvestmentDeferred("external_income_strategy_basis_changed")
        execution.commit_candidate(book, positions, receipts, records, checked_at, accounting)
        execution.state["operation"]["bill_ids"] = record["bill_ids"]
        execution.event("external_interest_reinvested", checked_at, **record)
        execution.complete(
            {
                "coin": "USDGO",
                "amount": record["amount"],
                "holding_id": holding_id,
                "attribution": "external_interest",
            },
            checked_at,
        )
        execution.save()
    except BaseException:
        execution.state = original
        raise
