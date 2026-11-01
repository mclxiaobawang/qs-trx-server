"""Closing USDT Earn cannot invalidate a previously reconciled interest receipt."""

from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.account_movement_receipts import external_usdt_income_allowed
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest
from tests.strategy.trx_phase1.test_live_contribution import execution as execution  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_execution import NOW
from tests.strategy.trx_phase1.test_rebate_coupon_rewards import coupon
from tests.strategy.trx_phase1.test_reconciliation_wait import allow_outside_usdt, usdt_interest


def close_after_credit(execution):
    allow_outside_usdt(execution)
    usdt_interest(execution.broker, bill="old-interest")
    execution.import_interest(NOW + 4 * 3600000)
    # Simulate an independently approved full redemption already reconciled to spot.
    remote = execution.broker.state()
    remote["holdings"].pop("external-usdt")
    remote["wallet"]["USDT"] = str(money(remote["wallet"]["USDT"]) + 10)
    execution.broker.put(remote)
    execution.state["holdings"]["external-usdt"]["unassigned_amount"] = "0"
    execution.state["unassigned_offsets"]["USDT"] = str(
        money(execution.state["unassigned_offsets"]["USDT"]) + 10
    )
    policy = execution.state["outside_usdt_income_policy"]
    policy["holdings"]["external-usdt"]["amount"] = "0"
    policy["sha256"] = digest({k: v for k, v in policy.items() if k != "sha256"})
    execution.save()
    assert not external_usdt_income_allowed(execution.state)


def test_old_interest_stays_settled_when_later_coupon_arrives_after_full_redemption(execution):
    close_after_credit(execution)
    before = deepcopy(execution.status())
    coupon(execution, amount=".02749569", bill="new-coupon")
    writes = deepcopy(execution.broker.state()["writes"])
    execution.import_interest(NOW + 4 * 3600000 + 1000)
    after = execution.status()
    assert after["book"] == before["book"]
    assert after["holdings"] == before["holdings"]
    assert (
        after["quarantined_account_credits"]["account-credit:old-interest"]
        == before["quarantined_account_credits"]["account-credit:old-interest"]
    )
    assert money(after["unassigned_offsets"]["USDT"]) - money(
        before["unassigned_offsets"]["USDT"]
    ) == money(".02749569")
    assert execution.import_interest(NOW + 4 * 3600000 + 2000) == 0
    assert execution.broker.state()["writes"] == writes


@pytest.mark.parametrize(
    "field,value", [("size", ".12550483"), ("bizOrderId", "changed"), ("businessType", "DEPOSIT")]
)
def test_modified_prior_credit_still_blocks_after_position_closure(execution, field, value):
    close_after_credit(execution)
    coupon(execution, bill="new-coupon")
    remote = execution.broker.state()
    next(r for r in remote["bills"] if r["billId"] == "old-interest")[field] = value
    execution.broker.put(remote)
    before = deepcopy(execution.status())
    with pytest.raises(ExecutionError, match="conflicting_account_credit"):
        execution.import_interest(NOW + 4 * 3600000 + 1000)
    assert execution.status() == before


def test_closed_position_fix_does_not_accept_new_unreviewed_deposit(execution):
    close_after_credit(execution)
    coupon(execution, bill="new-coupon")
    remote = execution.broker.state()
    remote["bills"].append(
        {
            "billId": "unreviewed",
            "bizOrderId": "deposit",
            "coin": "USDT",
            "groupType": "deposit",
            "businessType": "DEPOSIT",
            "size": "1",
            "fees": "0",
            "cTime": str(NOW + 3 * 3600000),
        }
    )
    remote["wallet"]["USDT"] = str(money(remote["wallet"]["USDT"]) + 1)
    execution.broker.put(remote)
    before = deepcopy(execution.status())
    with pytest.raises(ExecutionError, match="unrecognized_usdt_movement"):
        execution.import_interest(NOW + 4 * 3600000 + 1000)
    assert execution.status() == before
