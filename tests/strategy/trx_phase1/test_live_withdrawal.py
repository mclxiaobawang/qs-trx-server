"""Operator-reviewed withdrawal, receipt ownership, crash recovery and no duplicate debit."""

import json
from copy import deepcopy
from decimal import localcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from freqtrade.rpc.api_server import api_trx
from freqtrade.rpc.api_server.app import create_app
from freqtrade.strategy.trx_phase1 import live_withdrawal
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution, digest
from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategy
from tests.strategy.trx_phase1.test_income_allocation import approve
from tests.strategy.trx_phase1.test_live_strategy import advance, live, startup  # noqa: F401


@pytest.fixture
def account(live):  # noqa: F811 - shared pytest fixture
    startup(live)
    runtime, broker, _market, _ = live
    for _ in range(20):
        advance(live)
        if runtime.state["phase"] == "observing":
            break
    assert runtime.state["phase"] == "observing"
    execution = runtime.execution
    policy = json.loads(approve(execution).read_text())
    execution.reload()
    policy["approved_at"] = execution.state["created_at"]
    execution.state["startup_income_policy"] = policy
    execution.state.pop("income_accounting", None)
    basis = {"started_at": execution.state["created_at"], "income_policy_sha256": digest(policy)}
    basis["sha256"] = digest(basis)
    execution.state["performance_basis"] = basis
    execution.state["unassigned_offsets"]["USDT"] = "100"
    provider = broker.state()
    provider["wallet"]["USDT"] = str(money(provider["wallet"]["USDT"]) + 100)
    broker.put(provider)
    execution.save()
    runtime.state.update(performance_basis=basis, activated_at=basis["started_at"])
    runtime.persist()
    return live


def run_withdrawal(account, limit=30):
    for _ in range(limit):
        advance(account)
        records = account[0].execution.status().get("asset_withdrawals", {})
        if records and all(row["status"] != "confirmed" for row in records.values()):
            return next(reversed(records.values()))
    raise AssertionError(account[0].execution.status())


def test_review_and_confirmation_do_not_move_funds_until_current_worker_executes(account):
    runtime, broker, market, _ = account
    provider = broker.state()
    old = runtime.execution.status()["book"]
    review = live_withdrawal.review(runtime, "100", market.now)
    assert review["execution_enabled"] and review["destination"] == "same_account_spot"
    assert broker.state() == provider
    assert runtime.execution.status()["book"] == old
    record = live_withdrawal.confirm(runtime, review["review_hash"], market.now + 1000)
    assert record["status"] == "confirmed"
    assert broker.state() == provider
    assert runtime.execution.status()["book"] == old
    assert live_withdrawal.confirm(runtime, review["review_hash"], market.now + 5000) == record


def test_completed_withdrawal_preserves_profit_nav_trx_cost_start_and_usdt(account):
    runtime, broker, market, _ = account
    before = runtime.execution.status()
    writes_before = len(broker.state()["writes"])
    old = Ledger.restore(before["book"])
    review = live_withdrawal.review(runtime, "100", market.now)
    live_withdrawal.confirm(runtime, review["review_hash"], market.now)
    record = run_withdrawal(account)
    assert record["status"] == "completed"
    after = runtime.execution.status()
    book = Ledger.restore(after["book"])
    fx, price = money(record["mark_usdt"]), money(".3")
    with localcontext() as context:
        context.prec = 50
        assert old.usdgo - book.usdgo == 100
        assert old.deployable - book.deployable == 100 * fx
        assert abs(
            old.equity(price, fx) / old.fund_units - book.equity(price, fx) / book.fund_units
        ) < money("1e-40")
        assert abs(
            (old.equity(price, fx) - old.initial - old.external_flows)
            - (book.equity(price, fx) - book.initial - book.external_flows)
        ) < money("1e-35")
    assert book.cash == old.cash
    assert book.lots == old.lots
    assert after["unassigned_offsets"]["USDT"] == "100"
    assert (
        money(after["unassigned_offsets"]["USDGO"])
        == money(before["unassigned_offsets"]["USDGO"]) + 100
    )
    for key in ("created_at", "performance_basis", "anchor", "anchor_version"):
        assert after[key] == before[key]
    new_writes = broker.state()["writes"][writes_before:]
    assert len(new_writes) == 1 and new_writes[0]["kind"] == "redeem"
    provider = broker.state()
    assert live_withdrawal.confirm(runtime, review["review_hash"], market.now) == record
    assert broker.state() == provider
    runtime.execution.check_wallet(broker.balances(), book)
    assert runtime.state["desired"] == "running" and runtime.state["error"] is None


def test_confirmed_outbox_survives_crash_before_strategy_intent_is_saved(account):
    runtime, broker, market, root = account
    review = live_withdrawal.review(runtime, "100", market.now)
    live_withdrawal.confirm(runtime, review["review_hash"], market.now)
    restored = LiveStrategy(LiveExecution(root, broker, runtime.execution.account_id), market)
    account = (restored, broker, market, root)
    result = run_withdrawal(account)
    assert result["status"] == "completed"
    assert (
        len(
            [
                e
                for e in restored.execution.status()["events"]
                if e["kind"] == "capital_withdrawal_recorded"
            ]
        )
        == 1
    )


@pytest.mark.parametrize("failure", ["expired", "hash", "wallet", "basis", "intent", "paused"])
def test_unreviewed_or_changed_conditions_never_queue_a_withdrawal(account, failure):
    runtime, broker, market, _ = account
    review = live_withdrawal.review(runtime, "100", market.now)
    key, now = review["review_hash"], market.now
    if failure == "expired":
        now = review["expires_at"]
    elif failure == "hash":
        key = "0" * 64
    elif failure == "wallet":
        row = broker.state()
        row["wallet"]["USDGO"] = "1"
        broker.put(row)
    elif failure == "basis":
        runtime.execution.reload()
        runtime.execution.state["performance_basis"]["sha256"] = "0" * 64
        runtime.execution.save()
    elif failure == "intent":
        runtime.state["intent"] = {"kind": "trade"}
    else:
        runtime.state["desired"] = "paused"
    provider = broker.state()
    with pytest.raises(ExecutionError):
        live_withdrawal.confirm(runtime, key, now)
    assert broker.state() == provider
    assert not runtime.execution.status().get("asset_withdrawals")


def test_missing_receipt_never_debits_capital_and_second_request_is_rejected(account):
    runtime, broker, market, _ = account
    old = runtime.execution.status()["book"]
    review = live_withdrawal.review(runtime, "100", market.now)
    live_withdrawal.confirm(runtime, review["review_hash"], market.now)
    broker.hide_bills = True
    for _ in range(6):
        advance(account)
    assert runtime.execution.status()["book"]["external_flows"] == old["external_flows"]
    assert (
        runtime.execution.status()["asset_withdrawals"][review["review_hash"]]["status"]
        == "confirmed"
    )
    with pytest.raises(ExecutionError):
        live_withdrawal.review(runtime, "100", market.now)
    broker.hide_bills = False
    assert run_withdrawal(account)["status"] == "completed"


def test_crash_after_capital_commit_never_withdraws_twice(account, mocker):
    runtime, broker, market, _ = account
    review = live_withdrawal.review(runtime, "100", market.now)
    live_withdrawal.confirm(runtime, review["review_hash"], market.now)
    original = runtime.execution.store.save

    def save_then_crash(key, version, state):
        result = original(key, version, state)
        if (
            state.get("asset_withdrawals", {}).get(review["review_hash"], {}).get("status")
            == "completed"
        ):
            raise KeyboardInterrupt
        return result

    mocker.patch.object(runtime.execution.store, "save", side_effect=save_then_crash)
    with pytest.raises(KeyboardInterrupt):
        run_withdrawal(account)
    runtime.execution.store.save = original
    stored = runtime.execution.status()
    provider = broker.state()
    record = live_withdrawal.confirm(runtime, review["review_hash"], market.now)
    assert record["status"] == "completed"
    assert runtime.execution.status() == stored and broker.state() == provider
    live_withdrawal.finish(runtime, review["review_hash"], {"status": "completed"}, market.now)
    assert runtime.execution.status() == stored


def test_unknown_redemption_acknowledgement_is_never_resubmitted(account):
    runtime, broker, market, _ = account
    review = live_withdrawal.review(runtime, "100", market.now)
    live_withdrawal.confirm(runtime, review["review_hash"], market.now)
    before = len(broker.state()["writes"])
    broker.fault = "timeout_after_accept"
    for _ in range(4):
        advance(account)
    assert len(broker.state()["writes"]) == before + 1
    assert (
        runtime.execution.status()["asset_withdrawals"][review["review_hash"]]["status"]
        == "confirmed"
    )
    assert money(runtime.execution.status()["book"]["external_flows"]) == 0


def test_signal_during_redemption_uses_reduced_budget_only_after_withdrawal(account):
    runtime, broker, market, _ = account
    review = live_withdrawal.review(runtime, "100", market.now)
    live_withdrawal.confirm(runtime, review["review_hash"], market.now)
    runtime.engine.buy_at = market.now + 30000
    provider = broker.state()
    provider["trx_price"] = ".29"
    broker.put(provider)
    broker.hide_bills = True
    for _ in range(7):
        advance(account)
        assert runtime.engine.pending is None
    assert any(
        row["reason"] == "capital_withdrawal_in_progress" for row in runtime.state["evaluations"]
    )
    broker.hide_bills = False
    run_withdrawal(account)
    # The next eligible plan is sized from the synchronized post-withdrawal book.
    plan = runtime.engine.make_plan("buy", market.now + 1, 0.29, 0.014, market.now + 60000)
    assert plan is not None and plan.side == "buy"
    capital = Ledger.restore(runtime.execution.status()["book"]).deployable
    assert plan.budget == pytest.approx(float(capital) * 0.12)


def test_formal_http_routes_require_authentication_and_fixed_destination(tmp_path):
    service = SimpleNamespace(withdrawal=Mock(return_value={"ok": True}))
    app = create_app(
        {
            "user_data_dir": str(tmp_path),
            "api_server": {
                "username": "operator",
                "password": "password",
                "jwt_secret_key": "x" * 40,
            },
        }
    )
    app.dependency_overrides[api_trx.get_live_service] = lambda: service
    client = TestClient(app)
    path = "/api/v1/trx/live/withdrawal/"
    assert (
        client.post(
            path + "confirm", json={"review_hash": "a" * 64, "destination": "same_account_spot"}
        ).status_code
        == 401
    )
    for suffix, body in (
        ("review", {"amount": "1e3", "destination": "same_account_spot"}),
        ("review", {"amount": "1000", "destination": "external_address"}),
        ("confirm", {"review_hash": "a" * 64, "destination": "external_address"}),
        (
            "confirm",
            {"review_hash": "a" * 64, "destination": "same_account_spot", "amount": "99999"},
        ),
    ):
        assert (
            client.post(path + suffix, json=body, auth=("operator", "password")).status_code == 422
        )
    service.withdrawal.assert_not_called()
    assert (
        client.post(
            path + "review",
            json={"amount": "1000", "destination": "same_account_spot"},
            auth=("operator", "password"),
        ).status_code
        == 200
    )
    service.withdrawal.assert_called_once_with("1000")


@pytest.mark.parametrize("fx", [".99", "1.02"])
def test_historical_usdgo_valuation_profit_is_retained_without_inflating_budget(fx):
    book = Ledger("10000")
    book.cash = money(0)
    book.usdgo = money(10000)
    book.usdgo_cost = money(10000)
    before = deepcopy(book.state())
    book.withdraw_usdgo("withdrawal:test", "1000", ".3", fx)
    assert book.deployable == 10000 - 1000 * money(fx)
    assert book.withdrawn_usdgo_profit == 1000 * money(fx) - 1000
    assert book.summary(".3")["total_profit"] == float(10000 * money(fx) - 10000)
    assert Ledger.restore(book.state()).state() == book.state()
    assert before["initial"] == book.state()["initial"]
