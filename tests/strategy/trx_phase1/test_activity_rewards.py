"""Verified activity rewards never become strategy income or spendable capital."""

from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from tests.strategy.trx_phase1.test_live_contribution import execution as execution  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_execution import NOW, RULES, ExchangeFixture
from tests.strategy.trx_phase1.test_live_income import batch_credit
from tests.strategy.trx_phase1.test_live_ops import observed as observed  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_service import service as service  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_service import setup
from tests.strategy.trx_phase1.test_live_strategy import live as live  # noqa: PLC0414


def reward(broker, coin="USDGO", amount="0.00017776", stamp=NOW + 3 * 3600000, bill="reward-1"):
    state = broker.state()
    state["wallet"][coin] = str(money(state["wallet"][coin]) + money(amount))
    state["bills"].append(
        {
            "billId": bill,
            "bizOrderId": "activity-" + bill,
            "coin": coin,
            "groupType": "other",
            "businessType": "ACTIVITY_ASSET_REWARD_USER_IN",
            "size": amount,
            "fees": "0",
            "cTime": str(stamp),
        }
    )
    broker.put(state)


@pytest.mark.parametrize("coin", ["USDT", "USDGO", "TRX"])
def test_reward_is_external_exactly_once_across_restart(execution, coin):
    before = execution.status()
    writes = execution.broker.state()["writes"]
    reward(execution.broker, coin)
    execution.import_interest(NOW + 4 * 3600000)
    once = execution.status()
    assert once["book"] == before["book"]
    assert once["holdings"] == before["holdings"]
    for key in ["created_at", "anchor", "anchor_version", "flows", "operation"]:
        assert once[key] == before[key]
    assert money(once["unassigned_offsets"][coin]) == money(
        before["unassigned_offsets"][coin]
    ) + money(".00017776")
    record = once["quarantined_account_credits"]["account-credit:reward-1"]
    assert record["reason"] == "outside_strategy_activity_reward"
    assert record["precision_residual"] == "0" or money(record["precision_residual"]) == 0
    assert once["pending_income"] == before["pending_income"]
    recovered = LiveExecution(execution.root.parents[1], execution.broker, execution.account_id)
    assert recovered.import_interest(NOW + 4 * 3600000 + 1000) == 0
    assert recovered.status()["quarantined_account_credits"] == once["quarantined_account_credits"]
    assert recovered.status()["book"] == before["book"]
    assert execution.broker.state()["writes"] == writes


def test_two_real_reward_amounts_and_mixed_interest_conserve_wallet(execution):
    before = execution.status()
    stamp = NOW + 3 * 3600000
    reward(execution.broker, stamp=stamp)
    reward(execution.broker, amount=".00052460", stamp=stamp + 1000, bill="reward-2")
    batch_credit(execution.broker, "USDGO", ".71263290", stamp + 2000)
    batch_credit(execution.broker, "TRX", ".21728655", stamp + 3000)
    execution.import_interest(stamp + 4000)
    after = execution.status()
    rewards = after["quarantined_account_credits"]
    assert sum((money(v["offset_delta"]) for v in rewards.values()), money(0)) == money(".00070236")
    assert after["book"]["lots"] == before["book"]["lots"]
    assert after["holdings"] == before["holdings"]
    execution.check_wallet(execution.broker.balances(), Ledger.restore(after["book"]))


@pytest.mark.parametrize(
    "fault", ["mutation", "hidden", "fee", "negative", "unknown", "large_gap", "tiny_gap"]
)
def test_incomplete_reward_evidence_never_commits(execution, fault):
    reward(execution.broker)
    if fault == "mutation":
        execution.import_interest(NOW + 4 * 3600000)
    before = execution.status()
    remote = execution.broker.state()
    if fault == "mutation":
        remote["bills"][-1]["size"] = ".00017777"
    elif fault == "hidden":
        remote["bills"].pop()
    elif fault == "fee":
        remote["bills"][-1]["fees"] = ".00000001"
    elif fault == "negative":
        remote["bills"][-1]["size"] = "-.00017776"
    elif fault == "unknown":
        remote["bills"].append(
            {"coin": "USDGO", "groupType": "transfer", "cTime": str(NOW + 3 * 3600000)}
        )
    else:
        remote["wallet"]["USDGO"] = str(
            money(remote["wallet"]["USDGO"]) + money(".1" if fault == "large_gap" else ".000000001")
        )
    execution.broker.put(remote)
    with pytest.raises(ExecutionError):
        execution.import_interest(NOW + 4 * 3600000 + 1000)
    assert execution.status() == before


def test_lost_commit_response_replay_does_not_duplicate_reward(execution, monkeypatch):
    reward(execution.broker)
    original = execution.save

    def lost_response():
        original()
        raise RuntimeError("commit_response_lost")

    monkeypatch.setattr(execution, "save", lost_response)
    with pytest.raises(RuntimeError, match="commit_response_lost"):
        execution.import_interest(NOW + 4 * 3600000)
    recovered = LiveExecution(execution.root.parents[1], execution.broker, execution.account_id)
    before = deepcopy(recovered.status())
    assert recovered.import_interest(NOW + 4 * 3600000 + 1000) == 0
    assert (
        recovered.status()["quarantined_account_credits"] == before["quarantined_account_credits"]
    )
    assert recovered.status()["unassigned_offsets"] == before["unassigned_offsets"]


@pytest.mark.parametrize("coin", ["USDT", "USDGO", "TRX"])
def test_reward_in_initial_balance_is_baselined(tmp_path, coin):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    reward(broker, coin, stamp=NOW)
    runtime = LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)
    before = runtime.status()
    assert runtime.import_interest(NOW + 1000) == 0
    assert runtime.status()["unassigned_offsets"] == before["unassigned_offsets"]
    assert not runtime.status().get("quarantined_account_credits")


def test_paused_reconciliation_clears_resolved_wait_without_starting(service):
    svc, broker, clock, _, _ = service
    svc.prepare(setup())
    runtime = svc.runtime
    runtime.state.update(
        startup_complete=True,
        desired="paused",
        phase="blocked",
        error="account_evidence_wait_expired",
        evidence_wait_started_at=NOW,
    )
    runtime.persist()
    before = runtime.execution.status()["book"]
    clock[0] += 3600000
    reward(broker, stamp=clock[0])
    result = svc.reconcile()["runtime"]
    assert result["phase"] == "paused" and result["error"] is None
    assert result["evidence_wait_started_at"] is None
    assert result["desired"] == "paused" and not result["worker_running"]
    assert runtime.execution.status()["book"] == before
    assert not broker.state()["writes"]


def test_reward_wallet_delay_retries_then_remains_protected(live):
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    runtime.poll_error(ExecutionError("account_credit_wallet_evidence_required"), market.now)
    assert runtime.state["desired"] == "running" and runtime.state["phase"] == "recovering"
    runtime.poll_error(
        ExecutionError("account_credit_wallet_evidence_required"), market.now + 300000
    )
    assert runtime.state["desired"] == "paused" and runtime.state["phase"] == "blocked"
    assert not broker.state()["writes"]


def test_reward_totals_include_receipts_outside_recent_thirty(observed):
    runtime, _, view = observed
    reward_records = {
        str(i): {
            "coin": "USDGO",
            "type": "ACTIVITY_ASSET_REWARD_USER_IN",
            "offset_delta": ".00017776",
        }
        for i in range(40)
    }
    with runtime.execution.exclusive():
        runtime.execution.reload()
        runtime.execution.state["quarantined_account_credits"] = reward_records
        runtime.execution.save()
    shown = view.view()["runtime"]["execution"]
    assert len(shown["quarantined_account_credits"]) == 30
    assert money(shown["activity_reward_totals"]["USDGO"]) == money(".00711040")
