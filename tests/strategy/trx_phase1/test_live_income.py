"""Native income evidence, asynchronous arrival and ownership isolation."""

import pytest

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from tests.strategy.trx_phase1.test_live_execution import (
    ACCOUNT,
    NOW,
    RULES,
    ExchangeFixture,
    credit_provider_interest,
    run_flow,
    subscribe,
    trade,
)


@pytest.fixture
def runtime(tmp_path):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    return LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)


def batch_credit(broker, coin="TRX", amount="0.00004995", stamp=NOW + 1000):
    state = broker.state()
    state["wallet"][coin] = str(money(state["wallet"][coin]) + money(amount))
    state["bills"].append(
        {
            "billId": "batch-" + str(stamp),
            "bizOrderId": "interest-business-" + str(stamp),
            "coin": coin,
            "groupType": "financial",
            "businessType": "BATCH_INTEREST_USER_IN",
            "size": amount,
            "fees": "0",
            "cTime": str(stamp),
        }
    )
    broker.put(state)


def test_real_bill_shape_without_history_is_quarantined_once_after_restart(runtime, tmp_path):
    before = runtime.status()["book"]
    batch_credit(runtime.broker)
    assert runtime.import_interest(NOW + 2000) == 1
    once = runtime.status()
    assert once["book"] == before
    assert money(once["unassigned_offsets"]["TRX"]) == money("0.00004995")
    assert len(once["pending_income"]) == 1
    recovered = LiveExecution(tmp_path, runtime.broker, ACCOUNT)
    assert recovered.import_interest(NOW + 3000) == 0
    assert recovered.status()["book"] == before
    assert recovered.status()["pending_income"] == once["pending_income"]
    assert runtime.broker.state()["writes"] == []


@pytest.mark.parametrize("coin", ["TRX", "USDGO"])
def test_late_exact_history_reassigns_one_credit_without_adding_assets_twice(runtime, coin):
    symbol = "TRXUSDT" if coin == "TRX" else "USDGOUSDT"
    run_flow(runtime, [trade(symbol), subscribe(coin)])
    credit_provider_interest(runtime, coin, "0.01", "spot", NOW + 10000)
    state = runtime.broker.state()
    state["bills"][-1]["businessType"] = "BATCH_INTEREST_USER_IN"
    runtime.broker.put(state)
    runtime.broker.hide_records = True
    assert runtime.import_interest(NOW + 11000) == 1
    assert money(runtime.status()["unassigned_offsets"][coin]) == money("0.01")
    pending = runtime.status()["pending_income"]
    assert len(pending) == 1
    runtime.broker.hide_records = False
    assert runtime.import_interest(NOW + 12000) == 1
    state = runtime.status()
    book = Ledger.restore(state["book"])
    assert state["pending_income"] == {}
    assert money(state["unassigned_offsets"][coin]) == 0
    assert (book.interest_trx if coin == "TRX" else book.stable_interest) > 0
    assert runtime.import_interest(NOW + 13000) == 0
    assert runtime.status()["book"] == state["book"]


def test_bill_before_wallet_never_partially_commits_offsets_or_income(runtime):
    before = runtime.status()
    batch_credit(runtime.broker)
    runtime.broker.lagged_balances = {"USDT": "1000.4", "USDGO": "0", "TRX": "0"}
    with pytest.raises(ExecutionError, match="wallet_reconciliation_required"):
        runtime.import_interest(NOW + 2000)
    assert runtime.status() == before
    runtime.broker.lagged_balances = None
    assert runtime.import_interest(NOW + 3000) == 1


def test_unrelated_deposit_is_never_reclassified_as_interest(runtime):
    state = runtime.broker.state()
    state["wallet"]["TRX"] = "0.00004995"
    runtime.broker.put(state)
    with pytest.raises(ExecutionError, match="wallet_reconciliation_required"):
        runtime.import_interest(NOW + 2000)
    assert runtime.status()["pending_income"] == {}


def test_income_already_in_allocation_balance_is_not_imported_again(tmp_path):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    batch_credit(broker, stamp=NOW)
    execution = LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)
    assert execution.import_interest(NOW + 1000) == 0
    assert execution.status()["pending_income"] == {}
    assert money(execution.status()["unassigned_offsets"]["TRX"]) == money("0.00004995")


@pytest.mark.parametrize("destination", ["spot", "earn"])
def test_pooled_manual_and_strategy_holding_does_not_guess_interest_ownership(runtime, destination):
    state = runtime.broker.state()
    state["holdings"]["manual"] = {
        "order_id": "manual",
        "product_id": "USDGO-flex",
        "coin": "USDGO",
        "amount": "0.9",
    }
    runtime.broker.put(state)
    submit = runtime.broker.submit

    def pooled(op):
        result = submit(op)
        if op["kind"] == "subscribe":
            state = runtime.broker.state()
            new = state["holdings"].pop("holding-" + result["orderId"])
            state["holdings"]["manual"]["amount"] = str(money("0.9") + money(new["amount"]))
            state["savings"][result["orderId"]]["holding_id"] = "manual"
            runtime.broker.put(state)
        return result

    runtime.broker.submit = pooled
    assert run_flow(runtime, [trade(), subscribe("USDGO")])["flow"]["status"] == "completed"
    before = runtime.status()["book"]
    credit_provider_interest(runtime, "USDGO", "0.01", destination, NOW + 10000)
    assert runtime.import_interest(NOW + 11000) == 1
    assert runtime.status()["book"] == before
    assert len(runtime.status()["pending_income"]) == 1
    assert runtime.import_interest(NOW + 12000) == 0


@pytest.mark.parametrize("when", ["before", "after"])
def test_crash_around_quarantine_commit_is_atomic(runtime, tmp_path, mocker, when):
    batch_credit(runtime.broker)
    save = runtime.store.save

    def crash(*args):
        if when == "after":
            save(*args)
        raise KeyboardInterrupt

    mocker.patch.object(runtime.store, "save", side_effect=crash)
    with pytest.raises(KeyboardInterrupt):
        runtime.import_interest(NOW + 2000)
    recovered = LiveExecution(tmp_path, runtime.broker, ACCOUNT)
    assert recovered.import_interest(NOW + 3000) == (1 if when == "before" else 0)
    assert len(recovered.status()["pending_income"]) == 1
    assert money(recovered.status()["unassigned_offsets"]["TRX"]) == money("0.00004995")


def test_mutated_receipt_is_not_accepted_as_another_income(runtime):
    batch_credit(runtime.broker)
    runtime.import_interest(NOW + 2000)
    before = runtime.status()
    state = runtime.broker.state()
    state["bills"][-1]["size"] = "0.1"
    runtime.broker.put(state)
    with pytest.raises(ExecutionError, match="conflicting_interest_bill"):
        runtime.import_interest(NOW + 3000)
    assert runtime.status() == before


def test_unmapped_history_id_does_not_block_a_proven_account_credit(runtime):
    run_flow(runtime, [trade("TRXUSDT"), subscribe("TRX")])
    credit_provider_interest(runtime, "TRX", "0.01", "spot", NOW + 10000)
    state = runtime.broker.state()
    state["bills"][-1].update(
        businessType="BATCH_INTEREST_USER_IN", bizOrderId="independent-business-id"
    )
    runtime.broker.put(state)
    assert runtime.import_interest(NOW + 11000) == 1
    assert Ledger.restore(runtime.status()["book"]).interest_trx == 0
    assert len(runtime.status()["pending_income"]) == 1
    assert runtime.import_interest(NOW + 12000) == 0
