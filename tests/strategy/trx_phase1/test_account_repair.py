"""Native credits and verified wallet sweeps conserve ownership across retries."""

from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.account_repair import repair
from freqtrade.strategy.trx_phase1.income_allocation import HOUR, effective_units
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from tests.strategy.trx_phase1.test_live_contribution import execution as execution  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_execution import NOW, RULES, ExchangeFixture
from tests.strategy.trx_phase1.test_live_income import batch_credit


def rebate(broker, stamp, residual="0.000000008466"):
    remote = broker.state()
    remote["wallet"]["USDT"] = str(
        money(remote["wallet"]["USDT"]) + money(".04635835") + money(residual)
    )
    remote["bills"].append(
        {
            "billId": "cash-rebate",
            "bizOrderId": "unlinked-account-credit",
            "coin": "USDT",
            "groupType": "other",
            "businessType": "REBATE_REWARDS",
            "size": ".04635835",
            "fees": "0",
            "cTime": str(stamp),
        }
    )
    broker.put(remote)


def sweep(execution):
    stamp = NOW + 3 * HOUR + 60000
    batch_credit(execution.broker, "USDGO", ".01", stamp - 1000)
    remote = execution.broker.state()
    amount = money(remote["wallet"]["USDGO"])
    remote["wallet"]["USDGO"] = "0"
    remote["holdings"]["USDGO"]["amount"] = str(
        money(remote["holdings"]["USDGO"]["amount"]) + amount
    )
    remote["savings"]["manual-subscribe-business"] = {}
    remote["bills"].append(
        {
            "billId": "manual-subscribe-bill",
            "bizOrderId": "manual-subscribe-business",
            "coin": "USDGO",
            "groupType": "financial",
            "businessType": "SINGLE_USER_OUT",
            "size": str(-amount),
            "fees": "0",
            "balance": "0",
            "cTime": str(stamp),
        }
    )
    execution.broker.put(remote)
    batch_credit(execution.broker, "USDGO", ".01", NOW + 5 * HOUR + 1000)
    batch_credit(execution.broker, "TRX", ".02", NOW + 5 * HOUR + 1001)
    rebate(execution.broker, NOW + 5 * HOUR + 2000)
    return {
        "version": "trx-account-repair-v1",
        "account_id": execution.account_id,
        "ledger_created_at": execution.state["created_at"],
        "products": execution.state["products"],
        "approved_at": NOW + 6 * HOUR,
        "subscription": {
            "coin": "USDGO",
            "bill_id": "manual-subscribe-bill",
            "business_order_id": "manual-subscribe-business",
            "holding_id": "USDGO",
            "product_id": "USDGO-flex",
            "timestamp": stamp,
            "amount": str(amount),
        },
    }


def test_rebate_stays_outside_profit_capital_and_spendable_cash(execution):
    before = execution.status()
    writes = execution.broker.state()["writes"]
    rebate(execution.broker, NOW + 3 * HOUR)
    execution.import_interest(NOW + 4 * HOUR)
    after = execution.status()
    assert after["book"] == before["book"]
    assert money(after["unassigned_offsets"]["USDT"]) == money(".046358358466")
    credit = next(iter(after["quarantined_account_credits"].values()))
    assert money(credit["precision_residual"]) == money(".000000008466")
    assert not after["pending_income"]
    recovered = LiveExecution(execution.root.parents[1], execution.broker, execution.account_id)
    assert recovered.import_interest(NOW + 4 * HOUR + 1000) == 0
    assert recovered.status()["book"] == before["book"]
    assert recovered.status()["quarantined_account_credits"] == after["quarantined_account_credits"]
    assert execution.broker.state()["writes"] == writes


def test_rebate_already_in_allocation_balance_is_not_counted_again(tmp_path):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    rebate(broker, NOW)
    runtime = LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)
    before = runtime.status()
    assert runtime.import_interest(NOW + 1000) == 0
    assert runtime.status()["book"] == before["book"]
    assert runtime.status()["unassigned_offsets"] == before["unassigned_offsets"]
    assert not runtime.status().get("quarantined_account_credits")


@pytest.mark.parametrize("failure", ["wallet_lag", "large_tail", "unknown_movement", "mutation"])
def test_rebate_without_complete_evidence_never_commits(execution, failure):
    original_wallet = execution.broker.state()["wallet"]
    rebate(execution.broker, NOW + 3 * HOUR)
    if failure == "mutation":
        execution.import_interest(NOW + 4 * HOUR)
    before = execution.status()
    remote = execution.broker.state()
    if failure == "wallet_lag":
        execution.broker.lagged_balances = original_wallet
    elif failure == "large_tail":
        remote["wallet"]["USDT"] = str(money(remote["wallet"]["USDT"]) + money(".00000002"))
    elif failure == "unknown_movement":
        remote["bills"].append(
            {"coin": "USDT", "groupType": "transfer", "cTime": str(NOW + 3 * HOUR)}
        )
    else:
        remote["bills"][-1]["size"] = ".04"
    execution.broker.put(remote)
    with pytest.raises(ExecutionError):
        execution.import_interest(NOW + 4 * HOUR + 1000)
    assert execution.status() == before


def test_verified_sweep_replays_income_by_accrual_hour_and_is_idempotent(execution):
    before = execution.status()
    policy = sweep(execution)
    writes = execution.broker.state()["writes"]
    result = repair(execution, policy, NOW + 6 * HOUR)
    after = execution.status()
    old, book = Ledger.restore(before["book"]), Ledger.restore(after["book"])
    assert result["native_interest_receipts"] == 3 and result["exchange_writes"] == 0
    movement = result["movement"]
    assert money(movement["strategy_amount"]) + money(movement["external_amount"]) == money(
        policy["subscription"]["amount"]
    )
    assert money(movement["strategy_amount"]) > 0 and money(movement["external_amount"]) > money(
        ".1"
    )
    assert book.locked["USDGO"] == old.locked["USDGO"] + money(movement["strategy_amount"])
    assert book.stable_interest > 0 and book.interest_trx > 0
    for key in (
        "initial",
        "external_flows",
        "fund_units",
        "cash",
        "lots",
        "orders",
        "fees",
        "fee_rebates",
        "realized",
    ):
        assert after["book"][key] == before["book"][key]
    for key in ("anchor", "anchor_version", "flows", "operation"):
        assert after[key] == before[key]
    assert not after.get("asset_contributions") and not after["pending_income"]
    history = after["income_accounting"]["checkpoints"]
    old_position = before["income_accounting"]["checkpoints"][-1]["positions"]["USDGO"]
    during, _ = effective_units(history, "USDGO", NOW + 3 * HOUR, NOW + 4 * HOUR)
    following, _ = effective_units(history, "USDGO", NOW + 4 * HOUR, NOW + 5 * HOUR)
    assert during["strategy"] == money(old_position["strategy"])
    assert following["strategy"] == during["strategy"] + money(movement["strategy_amount"])
    execution.check_wallet(execution.broker.balances(), book)
    recovered = LiveExecution(execution.root.parents[1], execution.broker, execution.account_id)
    assert repair(recovered, policy, NOW + 6 * HOUR + 1000) == result
    assert recovered.import_interest(NOW + 6 * HOUR + 1000) == 0
    assert recovered.status()["book"] == after["book"]
    assert execution.broker.state()["writes"] == writes


@pytest.mark.parametrize(
    "failure",
    [
        "policy",
        "result",
        "bill",
        "holding",
        "incomplete_sweep",
        "ambiguous_time",
        "unexplained_cash",
    ],
)
def test_repair_failure_leaves_original_state_untouched(execution, failure):
    policy = sweep(execution)
    before = execution.status()
    remote = execution.broker.state()
    subscription = next(b for b in remote["bills"] if b["billId"] == "manual-subscribe-bill")
    prior_income = next(
        b for b in remote["bills"] if b["billId"] == "batch-" + str(NOW + 3 * HOUR + 59000)
    )
    if failure == "policy":
        policy["account_id"] = "wrong-account"
    elif failure == "result":
        remote["savings"].pop("manual-subscribe-business")
    elif failure == "bill":
        subscription["balance"] = "1"
    elif failure == "holding":
        remote["holdings"]["USDGO"]["amount"] = "1"
    elif failure == "incomplete_sweep":
        prior_income["size"] = ".005"
    elif failure == "ambiguous_time":
        prior_income["cTime"] = str(policy["subscription"]["timestamp"])
    else:
        remote["wallet"]["USDT"] = str(money(remote["wallet"]["USDT"]) + money(1))
    execution.broker.put(remote)
    with pytest.raises(ExecutionError):
        repair(execution, policy, NOW + 6 * HOUR)
    assert execution.status() == before


@pytest.mark.parametrize("when", ["before", "after"])
def test_crash_at_single_commit_retries_without_duplicate_income_or_sweep(execution, mocker, when):
    policy = sweep(execution)
    save = execution.store.save

    def crash(*args):
        if when == "after":
            save(*args)
        raise KeyboardInterrupt

    mocker.patch.object(execution.store, "save", side_effect=crash)
    with pytest.raises(KeyboardInterrupt):
        repair(execution, policy, NOW + 6 * HOUR)
    recovered = LiveExecution(execution.root.parents[1], execution.broker, execution.account_id)
    repair(recovered, policy, NOW + 6 * HOUR + 1000)
    after = deepcopy(recovered.status())
    assert len(after["account_repairs"]) == 1
    assert len(after["income_accounting"]["allocations"]) == 3
    assert len(after["quarantined_account_credits"]) == 1
    assert (
        len([e for e in after["events"] if e["kind"] == "external_savings_location_recorded"]) == 1
    )
    assert recovered.import_interest(NOW + 6 * HOUR + 2000) == 0
    assert recovered.status()["book"] == after["book"]
