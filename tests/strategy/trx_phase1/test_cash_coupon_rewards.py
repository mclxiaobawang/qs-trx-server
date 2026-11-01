"""Cash-coupon credits are distinct from earned interest and linked rebates."""

from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from tests.strategy.trx_phase1.test_activity_rewards import reward
from tests.strategy.trx_phase1.test_live_contribution import execution as execution  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_execution import NOW


def coupon(broker, amount, key):
    reward(broker, "USDT", amount, NOW + 3 * 3600000, key)
    state = broker.state()
    state["bills"][-1]["businessType"] = "CASH_COUPON_ACTIVITY_USER_IN"
    broker.put(state)


def test_four_cash_coupons_quarantined_once_without_changing_strategy_capital(execution):
    before = deepcopy(execution.status())
    writes = deepcopy(execution.broker.state()["writes"])
    amounts = [".03255427", ".00512202", ".07805579", ".00595255"]
    for i, amount in enumerate(amounts):
        coupon(execution.broker, amount, "coupon-" + str(i))
    execution.import_interest(NOW + 4 * 3600000)
    after = execution.status()
    assert after["book"] == before["book"]
    assert after["holdings"] == before["holdings"]
    assert after["created_at"] == before["created_at"]
    assert money(after["unassigned_offsets"]["USDT"]) - money(
        before["unassigned_offsets"]["USDT"]
    ) == money(".12168463")
    assert Ledger.restore(after["book"]).deployable == Ledger.restore(before["book"]).deployable
    records = after["quarantined_account_credits"]
    assert len(records) == 4
    assert all(v["reason"] == "outside_strategy_cash_coupon_reward" for v in records.values())
    assert execution.import_interest(NOW + 4 * 3600000 + 1000) == 0
    assert execution.status()["quarantined_account_credits"] == records
    assert execution.broker.state()["writes"] == writes


def test_cash_coupon_does_not_whitelist_an_unrelated_deposit(execution):
    coupon(execution.broker, ".03255427", "coupon")
    remote = execution.broker.state()
    remote["wallet"]["USDT"] = str(money(remote["wallet"]["USDT"]) + money(29000))
    remote["bills"].append(
        {
            "billId": "unreviewed-deposit",
            "bizOrderId": "deposit",
            "coin": "USDT",
            "groupType": "deposit",
            "businessType": "DEPOSIT",
            "size": "29000",
            "fees": "0",
            "cTime": str(NOW + 3 * 3600000),
        }
    )
    execution.broker.put(remote)
    before = deepcopy(execution.status())
    with pytest.raises(ExecutionError, match="unrecognized_usdt_movement"):
        execution.import_interest(NOW + 4 * 3600000)
    assert execution.status() == before


def test_changed_cash_coupon_fingerprint_never_recredits(execution):
    coupon(execution.broker, ".03255427", "coupon")
    execution.import_interest(NOW + 4 * 3600000)
    before = deepcopy(execution.status())
    remote = execution.broker.state()
    remote["bills"][-1]["size"] = ".03255428"
    execution.broker.put(remote)
    with pytest.raises(ExecutionError, match="conflicting_account_credit"):
        execution.import_interest(NOW + 4 * 3600000 + 1000)
    assert execution.status() == before
