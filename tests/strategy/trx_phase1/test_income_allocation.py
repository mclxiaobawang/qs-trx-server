"""Approved receipt allocation, conservation, late evidence and crash recovery."""

import json
from copy import deepcopy
from decimal import ROUND_DOWN

import pytest

from freqtrade.strategy.trx_phase1.income_allocation import (
    HOUR,
    VERSION,
    effective_units,
    ownership,
)
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from tests.strategy.trx_phase1.test_live_execution import (
    NOW,
    RULES,
    ExchangeFixture,
    credit_provider_interest,
    run_flow,
    subscribe,
    trade,
)
from tests.strategy.trx_phase1.test_live_income import batch_credit


def pooled(tmp_path, coin):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    runtime = LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)
    state = broker.state()
    state["holdings"]["manual"] = {
        "order_id": "manual",
        "coin": coin,
        "product_id": coin + "-flex",
        "amount": "0.9",
    }
    broker.put(state)
    submit = broker.submit

    def merge(op):
        result = submit(op)
        if op["kind"] == "subscribe":
            state = broker.state()
            new = state["holdings"].pop("holding-" + result["orderId"])
            state["holdings"]["manual"]["amount"] = str(money("0.9") + money(new["amount"]))
            state["savings"][result["orderId"]]["holding_id"] = "manual"
            broker.put(state)
        return result

    broker.submit = merge
    run_flow(runtime, [trade("TRXUSDT" if coin == "TRX" else "USDGOUSDT"), subscribe(coin)])
    return runtime


def approve(runtime):
    current = ownership(runtime.status()["holdings"])
    initial = deepcopy(current)
    for row in initial.values():
        row["strategy"] = "0"
    policy = {
        "version": VERSION,
        "account_id": runtime.account_id,
        "approved_at": NOW + 1000,
        "products": runtime.status()["products"],
        "checkpoints": [
            {"timestamp": NOW, "positions": initial},
            {"timestamp": NOW + 1000, "positions": current},
        ],
    }
    path = runtime.root.parents[1] / "income-allocation-policy.json"
    path.write_text(json.dumps(policy))
    return path


def credited(book, coin):
    return book.interest_trx if coin == "TRX" else book.stable_interest / money("1.001")


@pytest.mark.parametrize("coin", ["TRX", "USDGO"])
def test_quarantined_native_receipt_is_split_once_and_future_hours_continue(tmp_path, coin):
    runtime = pooled(tmp_path, coin)
    batch_credit(runtime.broker, coin, "0.01", NOW + 2 * HOUR + 1000)
    runtime.import_interest(NOW + 2 * HOUR + 2000)
    before = runtime.status()
    approve(runtime)
    assert runtime.import_interest(NOW + 2 * HOUR + 3000) == 1
    after = runtime.status()
    row = next(iter(after["income_accounting"]["allocations"].values()))
    p = before["holdings"]["manual"]
    expected = (
        money(".01") * money(p["principal"]) / (money(p["principal"]) + money(".9"))
    ).quantize(money(".00000001"), rounding=ROUND_DOWN)
    assert money(row["amount"]) == expected
    assert money(row["amount"]) + money(row["external_amount"]) == money(".01")
    assert credited(Ledger.restore(after["book"]), coin) == expected
    assert after["pending_income"] == {}
    assert after["book"]["orders"] == before["book"]["orders"]
    assert after["book"]["lots"] == before["book"]["lots"]
    assert after["anchor"] == before["anchor"]
    assert after["holdings"] == before["holdings"]
    recovered = LiveExecution(tmp_path, runtime.broker, runtime.account_id)
    assert recovered.import_interest(NOW + 2 * HOUR + 4000) == 0
    assert recovered.status()["book"] == after["book"]
    writes = runtime.broker.state()["writes"]
    batch_credit(runtime.broker, coin, "0.01", NOW + 3 * HOUR + 1000)
    assert recovered.import_interest(NOW + 3 * HOUR + 2000) == 2
    assert credited(Ledger.restore(recovered.status()["book"]), coin) == 2 * expected
    assert runtime.broker.state()["writes"] == writes


def test_receipt_before_first_complete_strategy_accrual_hour_is_external(tmp_path):
    runtime = pooled(tmp_path, "TRX")
    approve(runtime)
    batch_credit(runtime.broker, "TRX", "0.01", NOW + HOUR + 1000)
    before = runtime.status()["book"]
    runtime.import_interest(NOW + HOUR + 2000)
    state = runtime.status()
    assert state["book"] == before
    assert not state["pending_income"]
    row = next(iter(state["income_accounting"]["allocations"].values()))
    assert money(row["external_amount"]) == money(".01") and money(row["amount"]) == 0


@pytest.mark.parametrize("when", ["before", "after"])
def test_crash_at_allocation_commit_never_duplicates_income(tmp_path, mocker, when):
    runtime = pooled(tmp_path, "TRX")
    approve(runtime)
    batch_credit(runtime.broker, "TRX", "0.01", NOW + 2 * HOUR + 1000)
    save = runtime.store.save

    def crash(*args):
        if when == "after":
            save(*args)
        raise KeyboardInterrupt

    mocker.patch.object(runtime.store, "save", side_effect=crash)
    with pytest.raises(KeyboardInterrupt):
        runtime.import_interest(NOW + 2 * HOUR + 2000)
    recovered = LiveExecution(tmp_path, runtime.broker, runtime.account_id)
    recovered.import_interest(NOW + 2 * HOUR + 3000)
    state = recovered.status()
    assert len(state["income_accounting"]["allocations"]) == 1
    expected = money(next(iter(state["income_accounting"]["allocations"].values()))["amount"])
    assert Ledger.restore(state["book"]).interest_trx == expected
    assert not state["pending_income"]


def test_wallet_lag_does_not_partially_commit_allocated_income(tmp_path):
    runtime = pooled(tmp_path, "TRX")
    approve(runtime)
    before = runtime.status()
    runtime.broker.lagged_balances = runtime.broker.state()["wallet"]
    batch_credit(runtime.broker, "TRX", "0.01", NOW + 2 * HOUR + 1000)
    with pytest.raises(ExecutionError, match="wallet_reconciliation_required"):
        runtime.import_interest(NOW + 2 * HOUR + 2000)
    assert runtime.status() == before


def test_late_exact_history_only_corroborates_an_already_allocated_bill(tmp_path):
    runtime = pooled(tmp_path, "TRX")
    credit_provider_interest(runtime, "TRX", "0.01", "spot", NOW + 2 * HOUR + 1000)
    remote = runtime.broker.state()
    remote["bills"][-1]["businessType"] = "BATCH_INTEREST_USER_IN"
    runtime.broker.put(remote)
    runtime.broker.hide_records = True
    approve(runtime)
    runtime.import_interest(NOW + 2 * HOUR + 2000)
    before = runtime.status()
    runtime.broker.hide_records = False
    assert runtime.import_interest(NOW + 2 * HOUR + 3000) == 0
    after = runtime.status()
    assert after["book"] == before["book"]
    assert after["income_accounting"]["allocations"] == before["income_accounting"]["allocations"]


def test_changed_manual_holding_and_policy_do_not_silently_reassign_money(tmp_path):
    runtime = pooled(tmp_path, "TRX")
    path = approve(runtime)
    batch_credit(runtime.broker, "TRX", "0.01", NOW + 2 * HOUR + 1000)
    state = runtime.broker.state()
    state["holdings"]["manual"]["amount"] = str(money(state["holdings"]["manual"]["amount"]) + 1)
    runtime.broker.put(state)
    before = runtime.status()["book"]
    runtime.import_interest(NOW + 2 * HOUR + 2000)
    assert runtime.status()["income_accounting"]["status"] == "holding_evidence_required"
    assert runtime.status()["book"] == before and runtime.status()["pending_income"]
    policy = json.loads(path.read_text())
    policy["approved_at"] += 1
    path.write_text(json.dumps(policy))
    runtime.import_interest(NOW + 2 * HOUR + 3000)
    assert runtime.status()["income_accounting"]["status"] == "policy_validation_required"
    assert runtime.status()["book"] == before


def test_effective_principal_excludes_mid_hour_additions_and_redeemed_units():
    def point(t, owned):
        return {
            "timestamp": t,
            "positions": {
                "x": {
                    "coin": "TRX",
                    "product_id": "TRX-flex",
                    "strategy": str(owned),
                    "external": "10",
                }
            },
        }

    history = [point(0, 100), point(HOUR // 2, 150), point(HOUR + HOUR // 2, 50)]
    assert effective_units(history, "TRX", 0, HOUR)[0] == {
        "strategy": money(100),
        "external": money(10),
    }
    assert effective_units(history, "TRX", HOUR, 2 * HOUR)[0] == {
        "strategy": money(50),
        "external": money(10),
    }
