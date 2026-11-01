import json
from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.external_income_reinvestment import (
    POLICY_FILE,
    VERSION,
    ReinvestmentDeferred,
    authorize,
    available_sources,
    ownership_baseline,
    plan_steps,
)
from freqtrade.strategy.trx_phase1.income_allocation import HOUR
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_execution import ExecutionStep, LiveExecution, digest
from tests.strategy.trx_phase1.test_income_allocation import approve, pooled
from tests.strategy.trx_phase1.test_live_execution import NOW, ExchangeFixture, run_flow
from tests.strategy.trx_phase1.test_live_income import batch_credit


@pytest.fixture
def example(tmp_path):
    runtime = pooled(tmp_path, "USDGO")
    provider = runtime.broker.state()
    runtime.state["holdings"]["manual"]["unassigned_amount"] = "20000"
    provider["holdings"]["manual"]["amount"] = str(
        money(runtime.state["holdings"]["manual"]["principal"]) + money(20000)
    )
    # Five external USDGO of unrelated origin must remain unused.
    provider["wallet"]["USDGO"] = "5"
    runtime.state["unassigned_offsets"]["USDGO"] = "5"
    runtime.broker.put(provider)
    runtime.save()
    approve(runtime)
    batch_credit(runtime.broker, "USDGO", ".6", NOW + 2 * HOUR + 1000)
    runtime.import_interest(NOW + 2 * HOUR + 2000)
    now = NOW + 2 * HOUR + 3000
    state = runtime.state
    policy = {
        "version": VERSION,
        "enabled": True,
        "account_id": runtime.account_id,
        "ledger_created_at": state["created_at"],
        "approved_at": now,
        "scope": "verified_external_USDGO_interest_only",
        "holding_id": "manual",
        "product_id": "USDGO-flex",
        "baseline_sources": {
            k: digest(v) for k, v in state["income_accounting"]["allocations"].items()
        },
        "ownership_baseline_sha256": ownership_baseline(state),
    }
    path = runtime.root.parents[1] / POLICY_FILE
    path.write_text(json.dumps(policy))

    def merged_submit(op):
        try:
            return ExchangeFixture.submit(runtime.broker, op)
        finally:
            provider = runtime.broker.state()
            for hid in [key for key in provider["holdings"] if key.startswith("holding-exchange-")]:
                position = provider["holdings"].pop(hid)
                provider["holdings"]["manual"]["amount"] = str(
                    money(provider["holdings"]["manual"]["amount"]) + money(position["amount"])
                )
                oid = hid.removeprefix("holding-")
                provider["savings"][oid]["holding_id"] = "manual"
            for bill in provider["bills"]:
                if bill.get("bizOrderId") in provider["savings"]:
                    bill["businessType"] = "SINGLE_USER_OUT"
            runtime.broker.put(provider)

    runtime.broker.submit = merged_submit
    terms = {"USDGO": runtime.broker.product("USDGO", "USDGO-flex")}
    return runtime, policy, path, terms, now


def test_only_verified_external_interest_is_selected_not_other_external_or_strategy_cash(example):
    runtime, _, _, terms, now = example
    before = deepcopy(runtime.state)
    steps = plan_steps(runtime, terms, now)
    assert len(steps) == 1 and steps[0]["source"] == "external_interest"
    row = next(iter(runtime.state["income_accounting"]["allocations"].values()))
    assert money(steps[0]["amount"]) == money(row["external_amount"])
    assert money(steps[0]["amount"]) < money(runtime.state["unassigned_offsets"]["USDGO"]) - money(
        "4.9"
    )
    assert runtime.state == before
    assert ExecutionStep.model_validate(steps[0]).coin == "USDGO"


@pytest.mark.parametrize(
    "problem",
    ["account", "ledger", "scope", "baseline", "receipt", "already_spent", "wallet", "pending"],
)
def test_invalid_or_unavailable_funds_defer_without_changing_any_state(example, problem):
    runtime, policy, _, _, now = example
    state = deepcopy(runtime.state)
    if problem == "account":
        policy["account_id"] = "other"
    elif problem == "ledger":
        policy["ledger_created_at"] += 1
    elif problem == "scope":
        policy["scope"] = "all_external_money"
    elif problem == "baseline":
        policy["baseline_sources"]["missing"] = "unknown"
    elif problem == "receipt":
        state["receipts"][
            next(k for k in state["receipts"] if k.startswith("income-allocation:"))
        ] = "wrong"
    elif problem == "already_spent":
        state["external_income_reinvestments"] = {"fake": {"classification": "bad"}}
    elif problem == "wallet":
        state["unassigned_offsets"]["USDGO"] = "0"
    else:
        state["pending_income"] = {"pending": {}}
    original = deepcopy(state)
    with pytest.raises(ReinvestmentDeferred):
        available_sources(state, policy, now)
    assert state == original


@pytest.mark.parametrize(
    "problem", ["below_minimum", "no_capacity", "product_unavailable", "disabled"]
)
def test_product_minimum_and_capacity_and_disable_never_create_write(example, problem):
    runtime, policy, path, terms, now = example
    if problem == "below_minimum":
        terms["USDGO"]["singleMinAmount"] = "1"
    elif problem == "no_capacity":
        terms["USDGO"]["remainingAmount"] = "0"
    elif problem == "product_unavailable":
        terms["USDGO"]["status"] = "closed"
    else:
        policy["enabled"] = False
        path.write_text(json.dumps(policy))
    assert plan_steps(runtime, terms, now) == []


def test_nonspendable_external_balance_cannot_consume_strategy_cash(example):
    runtime, _, _, terms, now = example
    step = ExecutionStep.model_validate(plan_steps(runtime, terms, now)[0])
    balances = runtime.broker.balances()
    balances["USDGO"]["available"] = str(
        runtime.owned_spot(Ledger.restore(runtime.state["book"]))["USDGO"]
    )
    with pytest.raises(ReinvestmentDeferred, match="not_spendable"):
        authorize(runtime, step, balances, Ledger.restore(runtime.state["book"]), now)


def test_real_native_workflow_changes_only_external_location_and_deduplicates_after_restart(
    example,
):
    runtime, policy, _, terms, now = example
    before = deepcopy(runtime.state)
    steps = plan_steps(runtime, terms, now)
    amount = money(steps[0]["amount"])
    writes = len(runtime.broker.state()["writes"])
    state = run_flow(runtime, steps, now + 1000)
    assert state["flow"]["status"] == "completed", state["events"][-3:]
    assert state["book"] == before["book"]
    assert (
        state["anchor"] == before["anchor"] and state["anchor_version"] == before["anchor_version"]
    )
    assert state["holdings"]["manual"]["principal"] == before["holdings"]["manual"]["principal"]
    assert money(state["holdings"]["manual"]["unassigned_amount"]) == money(20000) + amount
    assert money(state["unassigned_offsets"]["USDGO"]) == money(5)
    assert len(runtime.broker.state()["writes"]) == writes + 1
    assert len(state["external_income_reinvestments"]) == 1
    recovered = LiveExecution(runtime.root.parents[1], runtime.broker, runtime.account_id)
    assert available_sources(recovered.state, policy, now + 10000) == []
    assert plan_steps(recovered, terms, now + 10000) == []
    assert recovered.import_interest(now + 10000) == 0
    assert recovered.state["book"] == before["book"]


def test_future_whole_period_income_stays_split_and_external_interest_can_compound(example):
    runtime, _, _, terms, now = example
    before = deepcopy(runtime.state)
    steps = plan_steps(runtime, terms, now)
    amount = money(steps[0]["amount"])
    assert run_flow(runtime, steps, now + 1000)["flow"]["status"] == "completed"
    batch_credit(runtime.broker, "USDGO", ".6", NOW + 4 * HOUR + 1000)
    runtime.import_interest(NOW + 4 * HOUR + 2000)
    row = max(
        runtime.state["income_accounting"]["allocations"].values(), key=lambda v: v["timestamp"]
    )
    assert money(row["external_units"]) == money(20000) + amount
    assert money(row["amount"]) < money(".01")
    assert money(row["account_amount"]) == money(row["amount"]) + money(row["external_amount"])
    for key in ["initial", "external_flows", "fund_units", "lots", "orders", "realized"]:
        assert runtime.state["book"][key] == before["book"][key]
    next_steps = plan_steps(runtime, terms, NOW + 4 * HOUR + 3000)
    assert money(next_steps[0]["amount"]) == money(row["external_amount"])


def test_product_capacity_reserves_only_a_partial_receipt_and_retains_remainder(example):
    runtime, policy, _, terms, now = example
    terms["USDGO"]["remainingAmount"] = ".2"
    steps = plan_steps(runtime, terms, now)
    assert money(steps[0]["amount"]) == money(".2")
    total = sum((v for _, v in available_sources(runtime.state, policy, now)), money(0))
    assert run_flow(runtime, steps, now + 1000)["flow"]["status"] == "completed"
    assert sum(
        (v for _, v in available_sources(runtime.state, policy, now + 5000)), money(0)
    ) == total - money(".2")


def test_disabled_policy_after_acknowledgement_still_reconciles_once(example):
    runtime, policy, path, terms, now = example
    steps = plan_steps(runtime, terms, now)
    runtime.queue(steps, now=now, expires_at=now + 600000, label="Synthetic external interest")
    runtime.tick(now + 100)
    runtime.tick(now + 200)
    assert runtime.status()["operation"]["status"] == "accepted"
    policy["enabled"] = False
    path.write_text(json.dumps(policy))
    runtime.tick(now + 300)
    assert runtime.status()["flow"]["status"] == "completed"
    assert plan_steps(runtime, terms, now + 400) == []


def test_unknown_earn_write_is_not_automatically_submitted_again(example):
    runtime, _, _, terms, now = example
    steps = plan_steps(runtime, terms, now)
    before = deepcopy(runtime.state["book"])
    writes = len(runtime.broker.state()["writes"])
    runtime.broker.fault = "timeout_after_accept"
    state = run_flow(runtime, steps, now + 1000)
    assert state["flow"]["status"] == "needs_review"
    assert len(runtime.broker.state()["writes"]) == writes + 1
    assert state["book"] == before


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "redeem", "holding_id": "manual"},
        {"coin": "TRX"},
        {"amount": "previous"},
        {"external_income_sources": {}},
        {"source": "principal"},
    ],
)
def test_external_source_cannot_be_used_as_generic_trade_or_redemption_scope(example, change):
    runtime, _, _, terms, now = example
    step = {**plan_steps(runtime, terms, now)[0], **change}
    with pytest.raises(ValueError):
        ExecutionStep.model_validate(step)


def test_new_manual_ownership_reconciliation_freezes_external_reinvestment_for_review(example):
    runtime, policy, _, _, now = example
    runtime.state.setdefault("account_movement_reconciliations", {})["manual-new"] = {
        "outside_USDGO_spent": "0.2"
    }
    before = deepcopy(runtime.state)
    with pytest.raises(ReinvestmentDeferred, match="ownership_review"):
        available_sources(runtime.state, policy, now)
    assert runtime.state == before


def test_unrelated_auto_USDT_rebate_capital_does_not_block_external_USDGO_interest(example):
    runtime, policy, _, _, now = example
    before = available_sources(runtime.state, policy, now)
    runtime.state.setdefault("asset_contributions", {})["rebate-new"] = {
        "classification": "automatic_capital_unlinked_rebate",
        "assets": [{"coin": "USDT", "amount": "0.02"}],
        "source_credits": [{"coin": "USDT", "amount": "0.02"}],
    }
    assert available_sources(runtime.state, policy, now) == before
