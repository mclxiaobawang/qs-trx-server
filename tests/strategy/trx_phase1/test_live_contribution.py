"""In-kind capital, preserved NAV/receipts, crash recovery and hourly ownership."""

from copy import deepcopy
from decimal import ROUND_DOWN

import pytest

from freqtrade.strategy.trx_phase1.income_allocation import HOUR, effective_units
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_contribution import apply, preview
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from tests.strategy.trx_phase1.test_income_allocation import approve
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


@pytest.fixture
def execution(tmp_path):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    state = broker.state()
    for coin, spot, earn in (("TRX", ".00380112", "4.46553"), ("USDGO", ".100153", ".9")):
        state["wallet"][coin] = spot
        state["holdings"][coin] = {
            "order_id": coin,
            "product_id": coin + "-flex",
            "coin": coin,
            "amount": earn,
        }
    broker.put(state)
    result = LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)
    submit = broker.submit

    def merge(op):
        response = submit(op)
        if op["kind"] == "subscribe":
            state = broker.state()
            row = state["holdings"].pop("holding-" + response["orderId"])
            state["holdings"][row["coin"]]["amount"] = str(
                money(state["holdings"][row["coin"]]["amount"]) + money(row["amount"])
            )
            state["savings"][response["orderId"]]["holding_id"] = row["coin"]
            broker.put(state)
        return response

    broker.submit = merge
    for coin in ("TRX", "USDGO"):
        run_flow(result, [trade(coin + "USDT"), subscribe(coin)])
    approve(result)
    result.import_interest(NOW + 2 * HOUR)
    return result


def test_preview_never_moves_assets_or_sends_orders(execution):
    before = execution.status()
    provider = execution.broker.state()
    review = preview(execution, NOW + 2 * HOUR + 2000)
    assert {a["coin"]: money(a["amount"]) for a in review["assets"]} == {
        "USDGO": money("1.000153"),
        "TRX": money("4.46933112"),
    }
    after = execution.status()
    for key in ("book", "holdings", "unassigned_offsets", "anchor", "anchor_version"):
        assert before[key] == after[key]
    assert execution.broker.state() == provider


def test_review_clock_uses_completed_read_time(execution):
    now = NOW + 2 * HOUR + 2000
    execution.broker.clock = lambda: now + 20000
    review = preview(execution, now)
    assert review["created_at"] == now + 20000
    assert review["expires_at"] == now + 20000 + 5 * 60000
    record = apply(execution, review["review_hash"], now + 21000)
    assert record["timestamp"] == now + 21000


def test_capital_transfer_conserves_nav_profit_locations_and_old_history(execution):
    before = execution.status()
    old = Ledger.restore(before["book"])
    old.usdgo_mark = money("1.001")
    review = preview(execution, NOW + 2 * HOUR + 2000)
    provider = execution.broker.state()
    record = apply(execution, review["review_hash"], NOW + 2 * HOUR + 3000)
    after = execution.status()
    new = Ledger.restore(after["book"])
    value = money(record["value"])
    assert new.equity(".3") == old.equity(".3") + value
    assert abs(new.equity(".3") / new.fund_units - old.equity(".3") / old.fund_units) < money(
        "1e-25"
    )
    assert abs(
        money(new.summary(".3")["total_profit"]) - money(old.summary(".3")["total_profit"])
    ) < money("1e-12")
    assert new.initial == old.initial and new.external_flows == old.external_flows + value
    assert new.principal == old.principal + money("4.46933112")
    assert new.usdgo == old.usdgo + money("1.000153")
    assert new.interest_trx == old.interest_trx and new.stable_interest == old.stable_interest
    assert new.orders == old.orders and new.fees == old.fees
    for key in ("anchor", "anchor_version", "receipts", "flows", "operation"):
        assert before[key] == after[key]
    assert before["income_accounting"]["allocations"] == after["income_accounting"]["allocations"]
    assert all(money(after["unassigned_offsets"][c]) == 0 for c in ("TRX", "USDGO"))
    assert all(money(h["unassigned_amount"]) == 0 for h in after["holdings"].values())
    assert sum((lot.locked for lot in new.lots), money(0)) == money(
        after["holdings"]["TRX"]["principal"]
    )
    assert new.locked["USDGO"] == money(after["holdings"]["USDGO"]["principal"])
    assert execution.broker.state() == provider
    assert apply(execution, review["review_hash"], NOW + 3 * HOUR) == record
    assert execution.status() == after


@pytest.mark.parametrize("when", ["before", "after"])
def test_crash_at_commit_is_atomic_and_retry_is_idempotent(execution, mocker, when):
    review = preview(execution, NOW + 2 * HOUR + 2000)
    save = execution.store.save

    def crash(*args):
        if when == "after":
            save(*args)
        raise KeyboardInterrupt

    mocker.patch.object(execution.store, "save", side_effect=crash)
    with pytest.raises(KeyboardInterrupt):
        apply(execution, review["review_hash"], NOW + 2 * HOUR + 3000)
    recovered = LiveExecution(execution.root.parents[1], execution.broker, execution.account_id)
    apply(recovered, review["review_hash"], NOW + 2 * HOUR + 4000)
    assert len(recovered.status()["asset_contributions"]) == 1
    assert (
        len(
            [
                e
                for e in recovered.status()["events"]
                if e["kind"] == "capital_contribution_recorded"
            ]
        )
        == 1
    )
    recovered.check_wallet(execution.broker.balances(), Ledger.restore(recovered.status()["book"]))


@pytest.mark.parametrize("failure", ["expired", "hash", "price", "wallet", "holding", "operation"])
def test_unreviewed_changed_or_unreconciled_assets_never_commit(execution, failure):
    review = preview(execution, NOW + 2 * HOUR + 2000)
    now = NOW + 2 * HOUR + 3000
    review_hash = review["review_hash"]
    state = execution.broker.state()
    if failure == "expired":
        now = review["expires_at"]
    elif failure == "hash":
        review_hash = "0" * 64
    elif failure == "price":
        state["trx_price"] = ".4"
    elif failure == "wallet":
        state["wallet"]["TRX"] = "0"
    elif failure == "holding":
        state["holdings"]["TRX"]["amount"] = "1"
    else:
        execution.reload()
        execution.state["operation"] = {"status": "unknown"}
        execution.save()
    execution.broker.put(state)
    before = execution.status()
    with pytest.raises(ExecutionError):
        apply(execution, review_hash, now)
    assert execution.status() == before


def test_prior_and_transition_hours_keep_old_ownership_then_full_new_hour(execution):
    now = NOW + 2 * HOUR + 2000
    review = preview(execution, now)
    apply(execution, review["review_hash"], now + 1000)
    history = execution.status()["income_accounting"]["checkpoints"]
    original = history[-2]["positions"]["TRX"]
    for start in (NOW + HOUR, NOW + 2 * HOUR):
        weights, _ = effective_units(history, "TRX", start, start + HOUR)
        assert weights == {k: money(original[k]) for k in ("strategy", "external")}
    weights, _ = effective_units(history, "TRX", NOW + 3 * HOUR, NOW + 4 * HOUR)
    assert weights["external"] == 0
    assert weights["strategy"] == money(original["strategy"]) + money(original["external"])
    prior_allocations = deepcopy(execution.status()["income_accounting"]["allocations"])
    batch_credit(execution.broker, "TRX", ".01", NOW + 4 * HOUR + 1000)
    execution.import_interest(NOW + 4 * HOUR + 2000)
    assert Ledger.restore(execution.status()["book"]).interest_trx == money(".01")
    for k, value in prior_allocations.items():
        assert execution.status()["income_accounting"]["allocations"][k] == value
    after = execution.status()["book"]
    execution.import_interest(NOW + 4 * HOUR + 3000)
    assert execution.status()["book"] == after


@pytest.mark.parametrize("coin", ["TRX", "USDGO"])
def test_late_exact_receipt_keeps_historical_ownership_after_full_adoption(execution, coin):
    before = execution.status()
    now = NOW + 2 * HOUR + 2000
    review = preview(execution, now)
    apply(execution, review["review_hash"], now + 1000)
    credit_provider_interest(execution, coin, ".01", "spot", NOW + 2 * HOUR + 1000)
    remote = execution.broker.state()
    remote["bills"][-1]["businessType"] = "BATCH_INTEREST_USER_IN"
    execution.broker.put(remote)
    execution.import_interest(now + 2000)
    allocations = execution.status()["income_accounting"]["allocations"]
    assert len(allocations) == 1
    credit = next(iter(allocations.values()))
    old = before["holdings"][coin]
    expected = (
        money(".01")
        * money(old["principal"])
        / (money(old["principal"]) + money(old["unassigned_amount"]))
    ).quantize(money(".00000001"), rounding=ROUND_DOWN)
    assert money(credit["amount"]) == expected
    assert money(credit["external_amount"]) > 0
    assert not execution.status()["pending_income"]
