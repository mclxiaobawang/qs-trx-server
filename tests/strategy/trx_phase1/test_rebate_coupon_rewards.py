"""Trade-linked coupon rewards reconcile once without becoming strategy funds."""

from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from tests.strategy.trx_phase1.test_activity_rewards import reward
from tests.strategy.trx_phase1.test_live_contribution import execution as execution  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_execution import NOW
from tests.strategy.trx_phase1.test_reconciliation_wait import allow_outside_usdt, usdt_interest


AMOUNTS = [
    ".00160039",
    ".00972154",
    ".03873227",
    ".03858844",
    ".03962546",
    ".00896547",
    ".05283395",
    ".00210156",
]


def coupon(execution, amount=".00160039", bill="coupon", fill=None):
    key = next(k for k in execution.state["receipts"] if k.startswith("spot:"))
    _, order, trade = key.split(":")
    reward(execution.broker, "USDT", amount, NOW + 3 * 3600000, bill)
    state = execution.broker.state()
    state["bills"][-1].update(
        businessType="REBATE_COUPON_ACTIVITY_USER_IN", bizOrderId=fill or trade
    )
    execution.broker.put(state)
    return order, trade


def test_eight_coupons_and_later_usdt_interest_reconcile_once_after_restart(execution):
    allow_outside_usdt(execution)
    before = deepcopy(execution.status())
    writes = deepcopy(execution.broker.state()["writes"])
    for i, amount in enumerate(AMOUNTS):
        order, trade = coupon(execution, amount, "coupon-" + str(i))
    usdt_interest(execution.broker)
    execution.import_interest(NOW + 4 * 3600000)
    after = execution.status()
    assert after["book"] == before["book"]
    assert after["holdings"] == before["holdings"]
    for key in ["created_at", "anchor", "anchor_version", "flows", "flow", "operation"]:
        assert after[key] == before[key]
    assert money(after["unassigned_offsets"]["USDT"]) - money(
        before["unassigned_offsets"]["USDT"]
    ) == money(".19216908") + money(".12550482")
    for i in range(8):
        row = after["quarantined_account_credits"]["account-credit:coupon-" + str(i)]
        assert row["linked_order_id"] == order and row["linked_fill_id"] == trade
        assert row["reason"] == "outside_strategy_rebate_coupon_reward"
    execution.check_wallet(execution.broker.balances(), Ledger.restore(after["book"]))
    recovered = LiveExecution(execution.root.parents[1], execution.broker, execution.account_id)
    assert recovered.import_interest(NOW + 4 * 3600000 + 1000) == 0
    assert recovered.status()["book"] == before["book"]
    assert recovered.status()["unassigned_offsets"] == after["unassigned_offsets"]
    assert recovered.status()["quarantined_account_credits"] == after["quarantined_account_credits"]
    assert execution.broker.state()["writes"] == writes


@pytest.mark.parametrize("fault", ["unlinked", "ambiguous", "changed", "deposit", "gap"])
def test_invalid_coupon_or_other_unexplained_movement_still_blocks(execution, fault):
    _, trade = coupon(execution, fill="unproven" if fault == "unlinked" else None)
    if fault == "ambiguous":
        execution.state["receipts"]["spot:other-order:" + trade] = "unrelated"
        execution.save()
    if fault == "changed":
        execution.import_interest(NOW + 4 * 3600000)
    remote = execution.broker.state()
    if fault == "changed":
        remote["bills"][-1]["size"] = ".00160040"
    elif fault == "deposit":
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
    elif fault == "gap":
        remote["wallet"]["USDT"] = str(money(remote["wallet"]["USDT"]) + 1)
    execution.broker.put(remote)
    before = deepcopy(execution.status())
    writes = deepcopy(remote["writes"])
    with pytest.raises(ExecutionError):
        execution.import_interest(NOW + 4 * 3600000 + 1000)
    assert execution.status() == before
    assert execution.broker.state()["writes"] == writes
