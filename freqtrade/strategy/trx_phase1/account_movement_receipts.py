"""Narrow recognition of already confirmed account movements and external USDT Earn."""

from .ledger import money
from .live_broker import ExecutionError
from .live_execution import digest


def known_movement_bill(execution, raw):
    previous = execution.state.get("account_movement_bill_receipts", {}).get(str(raw.get("billId")))
    if previous is None:
        return False
    if previous != digest(execution.interest_bill(raw)):
        raise ExecutionError("conflicting_confirmed_account_movement")
    return True


def external_usdt_income_allowed(state):
    policy = state.get("outside_usdt_income_policy")
    if not isinstance(policy, dict):
        return False
    body = {k: v for k, v in policy.items() if k != "sha256"}
    positions = {hid: p for hid, p in state.get("holdings", {}).items() if p.get("coin") == "USDT"}
    try:
        mixed = policy["version"] == "verified-usdt-earn-location-v2"
        if mixed:
            if policy.get("interest_is_outside_strategy") is not True:
                return False
            owned = sum((money(p["principal"]) for p in positions.values()), money(0))
            if money(state["book"]["locked"]["USDT"]) != owned or owned > money(
                state["book"]["cash"]
            ):
                return False
        return bool(positions) and (
            policy["version"]
            in {"verified-external-usdt-earn-v1", "verified-usdt-earn-location-v2"}
            and policy["sha256"] == digest(body)
            and policy["account_id"] == state["account_id"]
            and policy["ledger_created_at"] == state["created_at"]
            and bool(policy["approved_at"])
            and set(policy["holdings"]) == set(positions)
            and all(
                (
                    money(p["principal"]) >= 0
                    and money(p["interest"]) == 0
                    and money(p["principal"])
                    == money(policy["holdings"][hid]["strategy_principal"])
                    if mixed
                    else money(p["principal"]) + money(p["interest"]) == 0
                )
                and money(p["unassigned_amount"]) > 0
                and p["product_id"] == policy["holdings"][hid]["product_id"]
                and money(p["unassigned_amount"]) == money(policy["holdings"][hid]["amount"])
                for hid, p in positions.items()
            )
        )
    except (KeyError, TypeError, ValueError):
        return False
