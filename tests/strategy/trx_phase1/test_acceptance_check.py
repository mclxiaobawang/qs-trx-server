import json

import pytest

from freqtrade.strategy.trx_phase1 import acceptance_check as cli
from freqtrade.strategy.trx_phase1.acceptance_check import (
    PHASES,
    MinimumChainCheck,
    MinimumChainExecution,
)
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from tests.strategy.trx_phase1.test_live_execution import NOW, RULES, ExchangeFixture


@pytest.fixture
def check(tmp_path):
    (tmp_path / "live-product-rules.local.json").write_text(json.dumps({"products": RULES}))
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    broker.writes_enabled = False
    clock = [NOW]
    return MinimumChainCheck(tmp_path, broker, clock=lambda: clock[0]), broker, clock


def execute_once(check, broker, clock=None, *, accept_test_closeout=False):
    broker.writes_enabled = True
    try:
        result = check.execute(execute=True, accept_test_closeout=accept_test_closeout)
        if clock is not None:
            clock[0] += 1000
        return result
    finally:
        broker.writes_enabled = False


def test_prepare_is_readonly_minimal_bounded_and_does_not_start_strategy(check):
    check, broker, _ = check
    result = check.prepare()
    assert 1 < float(result["capital_limit_usdt"]) < 2
    assert result["completed_count"] == 0 and not result["complete"]
    assert result["execution"]["operation"]["status"] == "prepared"
    assert broker.state()["writes"] == []
    assert not (check.root / "live-strategy-control.json").exists()
    assert not (check.root / "live_execution").exists()
    assert check.control.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ExecutionError, match="operator_execute_flag_required"):
        check.execute()


def test_full_chain_is_independent_and_accepts_small_test_loss_without_rising_prices(check):
    check, broker, clock = check
    for index in range(len(PHASES)):
        clock[0] += 65000
        result = check.prepare()
        if index == 7:
            assert not result["test_closeout"]["profit_required"]
            assert money(result["test_closeout"]["expected_realized_pnl_usdt"]) < 0
            assert money(result["test_closeout"]["conservative_loss_at_price_floor_usdt"]) < money(
                "0.01"
            )
        assert result["execution"]["operation"]["status"] == "prepared"
        execute_once(check, broker, clock, accept_test_closeout=index == 7)
        assert len(broker.state()["writes"]) == index + 1
        with pytest.raises(ExecutionError, match="request_already_attempted_or_not_prepared"):
            execute_once(check, broker)
        result = check.verify()
        assert result["completed_count"] == index + 1
        assert check.verify()["completed_count"] == index + 1
    assert result["complete"] and check.load()["released_at"]
    assert money(result["execution"]["book"]["realized"]) < 0
    assert all(float(p["principal"]) == 0 for p in result["execution"]["holdings"].values())
    assert check.prepare()["complete"]
    assert len(broker.state()["writes"]) == len(PHASES)


def prepare_closeout(check, broker, clock):
    for _ in range(7):
        clock[0] += 65000
        check.prepare()
        execute_once(check, broker, clock)
        check.verify()
    clock[0] += 65000
    return check.prepare()


def test_final_closeout_needs_explicit_operator_acknowledgement_and_recovers_timeout(check):
    check, broker, clock = check
    prepare_closeout(check, broker, clock)
    with pytest.raises(ExecutionError, match="test_closeout_confirmation_required"):
        execute_once(check, broker)
    assert len(broker.state()["writes"]) == 7
    broker.fault = "timeout_after_accept"
    execute_once(check, broker, clock, accept_test_closeout=True)
    assert len(broker.state()["writes"]) == 8
    broker.fault = None
    restored = MinimumChainCheck(check.root, broker, clock=lambda: clock[0])
    with pytest.raises(ExecutionError, match="request_already_attempted_or_not_prepared"):
        execute_once(restored, broker, accept_test_closeout=True)
    assert restored.verify()["complete"]
    assert restored.verify()["complete"]
    assert len(broker.state()["writes"]) == 8


def test_production_executor_still_rejects_same_loss_making_sale(check):
    check, broker, clock = check
    prepared = prepare_closeout(check, broker, clock)
    runtime = LiveExecution(check.folder, broker, check.load()["account_id"])
    assert prepared["execution"]["operation"]["status"] == "prepared"
    broker.writes_enabled = True
    runtime.tick(clock[0])
    assert runtime.status()["events"][-1]["code"] == "profitable_principal_unavailable"
    assert len(broker.state()["writes"]) == 7


def test_final_closeout_does_not_consume_manual_trx_or_retained_interest(check):
    check, broker, clock = check
    remote = broker.state()
    remote["wallet"]["TRX"] = "11"
    broker.put(remote)
    prepare_closeout(check, broker, clock)
    runtime = check.runtime(check.load())
    with runtime.exclusive():
        runtime.reload()
        book = Ledger.restore(runtime.state["book"])
        book.credit_interest("synthetic-retained-interest", "TRX", "0.2")
        runtime.state["book"] = book.state()
        runtime.save()
    remote = broker.state()
    remote["wallet"]["TRX"] = str(money(remote["wallet"]["TRX"]) + money("0.2"))
    broker.put(remote)
    prepared = check.prepare()
    execute_once(check, broker, clock, accept_test_closeout=True)
    settled = check.verify()
    assert settled["complete"]
    assert money(settled["execution"]["book"]["interest_trx"]) == money("0.2")
    remaining = Ledger.restore(settled["execution"]["book"]).principal
    assert money(broker.state()["wallet"]["TRX"]) == money("11.2") + remaining
    assert money(prepared["test_closeout"]["sell_quantity_trx"]) > 0


def test_final_closeout_loss_limit_and_price_refresh_block_before_submission(check):
    check, broker, clock = check
    prepare_closeout(check, broker, clock)
    remote = broker.state()
    remote["trx_price"] = "0.299"
    broker.put(remote)
    blocked = execute_once(check, broker, accept_test_closeout=True)
    assert blocked["execution"]["events"][-1]["code"] == "price_deviation_exceeded"
    assert len(broker.state()["writes"]) == 7
    remote["trx_price"] = "0.2975"
    broker.put(remote)
    blocked = check.prepare()
    assert blocked["execution"]["operation"] is None
    assert blocked["execution"]["events"][-1]["code"] == "acceptance_closeout_loss_limit_exceeded"
    assert len(broker.state()["writes"]) == 7


def test_closeout_policy_cannot_be_reused_outside_test_ledger(check, tmp_path):
    check, broker, clock = check
    prepared = prepare_closeout(check, broker, clock)
    runtime = MinimumChainExecution(tmp_path / "production", broker, check.load())
    runtime.state = prepared["execution"]
    runtime.save()
    broker.writes_enabled = True
    runtime.tick(clock[0])
    assert runtime.status()["events"][-1]["code"] == "acceptance_closeout_scope_invalid"
    assert len(broker.state()["writes"]) == 7


def test_unexpected_fill_below_preview_is_booked_truthfully_and_not_marked_passed(check, mocker):
    check, broker, clock = check
    prepare_closeout(check, broker, clock)
    original_submit = broker.submit

    def adverse_fill(operation):
        remote = broker.state()
        remote["trx_price"] = "0.29"
        broker.put(remote)
        return original_submit(operation)

    mocker.patch.object(broker, "submit", side_effect=adverse_fill)
    execute_once(check, broker, clock, accept_test_closeout=True)
    settled = check.verify()
    assert settled["completed_count"] == 7 and not settled["complete"]
    assert settled["execution"]["flow"]["status"] == "needs_review"
    assert money(settled["execution"]["book"]["realized"]) < -money("0.01")
    before = settled["execution"]["book"]
    assert check.verify()["execution"]["book"] == before
    assert len(broker.state()["writes"]) == 8


def test_cli_final_closeout_acknowledgement_is_required(check, mocker, capsys):
    check, broker, clock = check
    prepare_closeout(check, broker, clock)

    def factory(writes=False):
        broker.writes_enabled = writes
        return broker

    mocker.patch.object(cli.LiveService, "broker", side_effect=factory)
    mocker.patch.object(cli.time, "time", return_value=clock[0] / 1000)
    mocker.patch.object(cli.time, "monotonic", return_value=0)
    args = ["--root", str(check.root), "execute", "--execute"]
    assert cli.main(args) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "test_closeout_confirmation_required"
    assert len(broker.state()["writes"]) == 7
    assert cli.main(args + ["--accept-test-closeout"]) == 0
    capsys.readouterr()
    clock[0] += 1000
    broker.writes_enabled = False
    assert check.verify()["complete"]


def test_rejected_final_closeout_cannot_be_replaced_with_another_write_attempt(check):
    check, broker, clock = check
    prepare_closeout(check, broker, clock)
    broker.fault = "reject"
    result = execute_once(check, broker, clock, accept_test_closeout=True)
    assert result["execution"]["flow"]["status"] == "failed"
    broker.fault = None
    result = check.prepare()
    assert result["execution"]["operation"] is None
    assert result["execution"]["events"][-1]["code"] == "acceptance_closeout_already_attempted"
    assert len(broker.state()["writes"]) == 7


def test_final_closeout_requires_the_exact_phase_five_buy_receipt(check):
    check, broker, clock = check
    prepare_closeout(check, broker, clock)
    control = check.load()
    control["completed"][4]["flow_id"] = "unrelated-flow"
    check.save(control)
    result = check.prepare()
    assert result["execution"]["operation"] is None
    assert result["execution"]["events"][-1]["code"] == "acceptance_closeout_buy_receipt_mismatch"
    assert len(broker.state()["writes"]) == 7


def test_ambiguous_earn_request_cannot_be_repeated_and_can_be_attached_exactly(check):
    check, broker, clock = check
    check.prepare()
    execute_once(check, broker, clock)
    check.verify()
    clock[0] += 65000
    check.prepare()
    broker.fault = "timeout_after_accept"
    result = execute_once(check, broker, clock)
    assert result["execution"]["operation"]["status"] == "unknown"
    broker.fault = None
    check.verify()
    with pytest.raises(ExecutionError, match="request_already_attempted_or_not_prepared"):
        execute_once(check, broker)
    exact_id = next(iter(broker.state()["savings"]))
    assert check.verify(exact_id)["completed_count"] == 2
    assert len(broker.state()["writes"]) == 2


def test_unsubmitted_preparation_can_be_released_but_started_chain_cannot(check):
    check, broker, _ = check
    check.prepare()
    assert check.cancel()["cancelled"]
    assert broker.state()["writes"] == []
    with pytest.raises(ExecutionError, match="test_request_not_prepared"):
        execute_once(check, broker)


def test_started_test_cannot_be_cancelled_or_replaced_with_strategy(check):
    check, broker, _ = check
    check.prepare()
    execute_once(check, broker)
    with pytest.raises(ExecutionError, match="finish_started_chain_before_release"):
        check.cancel()


def test_write_transport_is_not_accepted_for_prepare_or_verify(check):
    check, broker, _ = check
    check.prepare()
    broker.writes_enabled = True
    with pytest.raises(ExecutionError, match="requires_readonly_transport"):
        check.prepare()
    with pytest.raises(ExecutionError, match="requires_readonly_transport"):
        check.verify()
    assert broker.state()["writes"] == []


def test_operator_command_is_inert_without_execute_flag(check, mocker, capsys):
    check, broker, _ = check

    def factory(writes=False):
        broker.writes_enabled = writes
        return broker

    mocker.patch.object(cli.LiveService, "broker", side_effect=factory)
    mocker.patch.object(cli.time, "time", return_value=NOW / 1000)
    mocker.patch.object(cli.time, "monotonic", return_value=0)
    args = ["--root", str(check.root)]
    assert cli.main(args + ["prepare"]) == 0
    capsys.readouterr()
    assert cli.main(args + ["execute"]) == 1
    assert broker.state()["writes"] == []
    assert cli.main(args + ["execute", "--execute"]) == 0
    capsys.readouterr()
    assert len(broker.state()["writes"]) == 1
    assert cli.main(args + ["verify"]) == 0
    assert json.loads(capsys.readouterr().out)["completed_count"] == 1
