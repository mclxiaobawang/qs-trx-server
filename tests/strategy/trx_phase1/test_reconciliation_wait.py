"""Payout races cannot bypass a wallet check, extend a wait, or rewrite strategy capital."""

# ruff: noqa: F811 - imported pytest fixtures
from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1 import reconciliation_wait as wait
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_strategy import EVIDENCE_WAIT_CODES
from tests.strategy.trx_phase1.test_activity_rewards import reward
from tests.strategy.trx_phase1.test_live_contribution import execution as execution  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_execution import NOW
from tests.strategy.trx_phase1.test_live_ops import monitor, observed  # noqa: F401
from tests.strategy.trx_phase1.test_live_strategy import live, startup  # noqa: F401


def state(**fields):
    return {
        "desired": "running",
        "phase": "recovering",
        "error": "wallet_reconciliation_required",
        "evidence_wait_started_at": NOW,
        **fields,
    }


def test_grace_is_bounded_and_only_covers_existing_evidence_codes():
    assert wait.WAIT_CODES == EVIDENCE_WAIT_CODES
    assert wait.pending_wait(state(), NOW + 299999)
    for value in [
        state(desired="paused"),
        state(phase="blocked"),
        state(error="unrecognized_usdt_movement"),
        state(error="exchange_read_40014"),
        state(evidence_wait_started_at=None),
        state(evidence_wait_started_at=NOW + 1),
    ]:
        assert wait.pending_wait(value, NOW) is None
    assert wait.pending_wait(state(), NOW + 300000) is None
    assert wait.pending_wait(state(), NOW + 1, {"error": "ambiguous_write"}) is None
    assert wait.pending_wait(state(phase="executing"), NOW + 1)


def test_only_a_later_exact_verification_clears_an_active_flow_error():
    current = state(
        phase="executing",
        evidence_wait_last_seen_at=NOW + 1000,
        events=[
            {
                "kind": "strategy_poll_blocked",
                "timestamp": NOW + 2000,
                "code": "wallet_reconciliation_required",
            }
        ],
    )
    for stamp in [NOW, NOW + 1000, NOW + 2000, NOW + 9999999]:
        before = deepcopy(current)
        assert wait.clear_verified_wait(current, {"interest_checked_at": stamp}, NOW + 3000) is None
        assert current == before
    assert wait.clear_verified_wait(current, {"interest_checked_at": NOW + 2500}, NOW + 3000)
    assert current["error"] is None and current["evidence_wait_started_at"] is None
    blocked = state(desired="paused", phase="blocked", error="account_evidence_wait_expired")
    assert (
        wait.clear_verified_wait(blocked, {"interest_checked_at": NOW + 2500}, NOW + 3000) is None
    )


def usdt_interest(broker, amount=".12550482", bill="payout"):
    reward(broker, "USDT", amount, NOW + 3 * 3600000, bill)
    remote = broker.state()
    remote["bills"][-1].update(groupType="financial", businessType="BATCH_INTEREST_USER_IN")
    broker.put(remote)


def allow_outside_usdt(execution):
    from freqtrade.strategy.trx_phase1.live_execution import digest

    remote = execution.broker.state()
    remote["holdings"]["external-usdt"] = {
        "order_id": "external-usdt",
        "product_id": "USDT-flex",
        "coin": "USDT",
        "amount": "10",
    }
    execution.broker.put(remote)
    execution.state["holdings"]["external-usdt"] = {
        "coin": "USDT",
        "product_id": "USDT-flex",
        "principal": "0",
        "interest": "0",
        "unassigned_amount": "10",
        "subscription_ids": [],
        "lots": {},
    }
    policy = {
        "version": "verified-external-usdt-earn-v1",
        "account_id": execution.account_id,
        "ledger_created_at": execution.state["created_at"],
        "approved_at": NOW,
        "holdings": {"external-usdt": {"product_id": "USDT-flex", "amount": "10"}},
    }
    execution.state["outside_usdt_income_policy"] = {**policy, "sha256": digest(policy)}
    execution.save()


def test_late_payout_bill_refreshes_once_and_stays_outside_strategy(execution, monkeypatch):
    allow_outside_usdt(execution)
    before = deepcopy(execution.status())
    writes = deepcopy(execution.broker.state()["writes"])
    usdt_interest(execution.broker)
    original = execution.broker.bills
    calls = []

    def lag(coin, start, end):
        if coin == "USDT":
            calls.append(end)
            if len(calls) == 1:
                return []
        return original(coin, start, end)

    monkeypatch.setattr(execution.broker, "bills", lag)
    ticks = iter([0, 0, 0.02, 0.03, 0.03])
    monkeypatch.setattr(wait, "monotonic", lambda: next(ticks))
    execution.import_interest(NOW + 4 * 3600000)
    after = execution.status()
    assert len(calls) == 2 and calls[1] == calls[0] + 20
    assert after["interest_checked_at"] == NOW + 4 * 3600000
    assert after["book"] == before["book"] and after["holdings"] == before["holdings"]
    assert after["created_at"] == before["created_at"]
    assert money(after["unassigned_offsets"]["USDT"]) - money(
        before["unassigned_offsets"]["USDT"]
    ) == money(".12550482")
    assert (
        after["quarantined_account_credits"]["account-credit:payout"]["reason"]
        == "outside_strategy_usdt_interest"
    )
    assert execution.broker.state()["writes"] == writes
    execution.import_interest(NOW + 4 * 3600000 + 1)
    assert len(execution.status()["quarantined_account_credits"]) == 1


def test_unproven_gap_never_commits_or_writes_and_diagnostics_include_exact_delta(
    execution, monkeypatch, caplog
):
    before = deepcopy(execution.status())
    writes = deepcopy(execution.broker.state()["writes"])
    remote = execution.broker.state()
    remote["wallet"]["USDT"] = str(money(remote["wallet"]["USDT"]) + money(".12550482"))
    execution.broker.put(remote)
    calls = []
    original = execution.broker.balances

    def balances():
        calls.append(1)
        return original()

    monkeypatch.setattr(execution.broker, "balances", balances)
    with pytest.raises(ExecutionError, match="wallet_reconciliation_required") as result:
        execution.import_interest(NOW + 4 * 3600000)
    assert len(calls) == 2 and execution.status() == before
    assert result.value.wallet_mismatch["coin"] == "USDT"
    assert money(result.value.wallet_mismatch["difference"]) == money(".12550482")
    assert "coin=USDT" in caplog.text and "actual=" in caplog.text and "expected=" in caplog.text
    assert execution.broker.state()["writes"] == writes


def test_valid_payout_does_not_hide_an_unrelated_deposit(execution):
    allow_outside_usdt(execution)
    usdt_interest(execution.broker)
    remote = execution.broker.state()
    remote["wallet"]["USDT"] = str(money(remote["wallet"]["USDT"]) + 10)
    remote["bills"].append(
        {
            "billId": "foreign",
            "bizOrderId": "deposit",
            "coin": "USDT",
            "groupType": "deposit",
            "businessType": "DEPOSIT",
            "size": "10",
            "fees": "0",
            "cTime": str(NOW + 3 * 3600000),
        }
    )
    execution.broker.put(remote)
    before = deepcopy(execution.status())
    with pytest.raises(ExecutionError, match="unrecognized_usdt_movement"):
        execution.import_interest(NOW + 4 * 3600000)
    assert execution.status() == before


def test_verified_wait_clears_before_busy_branch_without_restarting_indicators(live, monkeypatch):
    runtime, _broker, market, _ = live
    startup(live)
    marker = deepcopy(runtime.state["indicator_initialization"])
    before = deepcopy(runtime.execution.status()["book"])
    assert runtime.recovering is False
    runtime.poll_error(ExecutionError("wallet_reconciliation_required"), market.now)
    assert runtime.recovering is False
    market.now += 15000
    monkeypatch.setattr(runtime, "resolve_execution", lambda now: True)
    monkeypatch.setattr(
        runtime, "decide", lambda *a: pytest.fail("busy flow must keep the original intent")
    )
    runtime.tick(market.now)
    assert runtime.state["phase"] == "executing" and runtime.state["error"] is None
    assert runtime.state["evidence_wait_started_at"] is None
    assert runtime.state["indicator_initialization"] == marker
    assert runtime.execution.status()["book"] == before
    assert any(e["kind"] == "account_evidence_wait_resolved" for e in runtime.state["events"])


def test_long_evidence_wait_still_blocks_without_writes(live):
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    for elapsed in [0, 200000, 299999]:
        runtime.poll_error(ExecutionError("wallet_reconciliation_required"), market.now + elapsed)
        assert runtime.state["desired"] == "running"
    runtime.poll_error(ExecutionError("wallet_reconciliation_required"), market.now + 300000)
    assert (
        runtime.state["desired"] == "paused"
        and runtime.state["error"] == "account_evidence_wait_expired"
    )
    assert broker.state()["writes"] == []


def test_monitor_does_not_alarm_during_grace_but_keeps_timeout_and_stale_checks(
    observed, tmp_path, monkeypatch
):
    runtime, broker, view, ops = monitor(observed, tmp_path, monkeypatch)
    runtime.state.update(
        desired="running",
        phase="executing",
        error="wallet_reconciliation_required",
        evidence_wait_started_at=broker.now,
        last_poll_at=broker.now,
    )
    runtime.persist()
    for _ in range(2):
        assert not any(a["code"] == "strategy_error" and a["active"] for a in ops.check()["alerts"])
    rendered = view.view()["runtime"]
    assert rendered["reconciliation_wait"] and rendered["error"] is None
    assert rendered["persisted_error"] == "wallet_reconciliation_required"
    broker.now += 300000
    for _ in range(2):
        ops.check()
    assert any(a["code"] == "strategy_error" and a["active"] for a in ops.status()["alerts"])
    assert any(a["code"] == "worker_stale" and a["active"] for a in ops.status()["alerts"])
