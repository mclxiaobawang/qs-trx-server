"""In-kind reserve conservation, spend isolation, income boundaries and recovery."""

# ruff: noqa: F811 - imported pytest fixture used as test parameters
from decimal import localcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from freqtrade.rpc.api_server import api_trx
from freqtrade.rpc.api_server.app import create_app
from freqtrade.strategy.trx_phase1 import live_contribution, live_reserve
from freqtrade.strategy.trx_phase1.income_allocation import HOUR
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution, digest
from freqtrade.strategy.trx_phase1.live_funding import LiveFunding
from freqtrade.strategy.trx_phase1.live_service import LiveService
from tests.strategy.trx_phase1.test_live_income import batch_credit
from tests.strategy.trx_phase1.test_live_strategy import live  # noqa: F401
from tests.strategy.trx_phase1.test_live_withdrawal import account  # noqa: F401


KEY = "reserve-half-test-01"


def test_reserve_endpoint_requires_authentication_and_fixed_fraction(tmp_path):
    service = SimpleNamespace(reserve_half=Mock(return_value={"ok": True}))
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
    path = "/api/v1/trx/live/reserve/half"
    body = {"account_id": "bound-account", "allocation_id": KEY}
    assert client.post(path, json=body).status_code == 401
    assert (
        client.post(
            path, json={**body, "fraction": "0.9"}, auth=("operator", "password")
        ).status_code
        == 422
    )
    service.reserve_half.assert_not_called()
    assert client.post(path, json=body, auth=("operator", "password")).status_code == 200
    service.reserve_half.assert_called_once_with("bound-account", KEY)


def allocate(account, key=KEY):
    runtime, _, market, _ = account
    market.now += 1000
    return live_reserve.allocate_half(
        runtime.execution, {**runtime.state, "engine": runtime.engine.state()}, key, market.now
    )


def test_service_commits_with_a_reader_and_syncs_existing_engine_without_new_startup(
    account, mocker
):
    runtime, broker, market, root = account
    reader = Mock(return_value=broker)
    svc = LiveService(root, broker_factory=reader, clock=lambda: market.now + 1000)
    svc.root = runtime.execution.root.parents[1]  # The fixture uses LiveExecution's direct root.
    mocker.patch.object(
        svc,
        "load",
        return_value={"account_id": runtime.execution.account_id, "credentials_identity": "bound"},
    )
    mocker.patch.object(svc, "get_runtime", return_value=runtime)
    mocker.patch.object(svc, "credential_identity", return_value="bound")
    runtime.thread = Mock()
    runtime.thread.is_alive.return_value = True
    old, provider = runtime.engine.state(), broker.state()
    receipt = svc.reserve_half(runtime.execution.account_id, KEY)
    assert reader.call_args.args == (False,)
    assert runtime.state["startup_complete"] and runtime.state["desired"] == "running"
    assert runtime.engine.anchor == old["anchor"]
    assert runtime.engine.book.usdgo == money(runtime.execution.status()["book"]["usdgo"])
    assert broker.state() == provider
    saved = runtime.execution.status()
    assert svc.reserve_half(runtime.execution.account_id, KEY) == receipt
    assert runtime.execution.status() == saved
    reader.assert_called_once()


def test_half_is_reclassified_without_any_exchange_write_and_preserves_history(account):
    runtime, broker, _, _ = account
    state = runtime.execution.status()
    before = Ledger.restore(state["book"])
    provider, engine = broker.state(), runtime.engine.state()
    record = allocate(account)
    after = runtime.execution.status()
    book = Ledger.restore(after["book"])
    amount = money(record["quantity"])
    assert amount == (before.usdgo / 2).quantize(money(".00000001"))
    assert before.usdgo - book.usdgo == before.locked["USDGO"] - book.locked["USDGO"] == amount
    assert book.lots == before.lots and book.cash == before.cash
    assert runtime.engine.state() == engine
    assert broker.state() == provider and record["exchange_actions"] == []
    for key in ("created_at", "performance_basis", "anchor", "anchor_version"):
        assert after[key] == state[key]
    assert after["book"]["orders"] == state["book"]["orders"]
    for key, p in after["holdings"].items():
        assert sum(money(p[k]) for k in ("principal", "interest", "unassigned_amount")) == sum(
            money(state["holdings"][key][k]) for k in ("principal", "interest", "unassigned_amount")
        )
    with localcontext() as context:
        context.prec = 50
        fx, price = money(record["mark_usdt"]), money(".3")
        assert abs(
            before.equity(price, fx) / before.fund_units - book.equity(price, fx) / book.fund_units
        ) < money("1e-35")
        assert abs(
            (before.equity(price, fx) - before.initial - before.external_flows)
            - (book.equity(price, fx) - book.initial - book.external_flows)
        ) < money("1e-30")


def test_repeat_and_restart_do_not_halve_the_remaining_funds_again(account):
    runtime, broker, market, root = account
    first = allocate(account)
    saved = runtime.execution.status()
    restored = LiveExecution(root, broker, runtime.execution.account_id)
    assert live_reserve.allocate_half(restored, {}, KEY, market.now + 1000) == first
    assert restored.status() == saved
    with pytest.raises(ExecutionError, match="reserve_already_allocated"):
        live_reserve.allocate_half(restored, {}, "another-request", market.now + 1000)
    assert restored.status() == saved


def test_crash_after_commit_recovers_without_a_second_debit(account, mocker):
    runtime, broker, market, root = account
    save = runtime.execution.store.save

    def crash(*args):
        save(*args)
        raise KeyboardInterrupt

    mocker.patch.object(runtime.execution.store, "save", side_effect=crash)
    with pytest.raises(KeyboardInterrupt):
        allocate(account)
    restored = LiveExecution(root, broker, runtime.execution.account_id)
    saved = restored.status()
    assert (
        live_reserve.allocate_half(restored, {}, KEY, market.now + 1000) == saved["usdgo_reserve"]
    )
    assert restored.status() == saved


def test_reserve_is_not_available_for_redemption_or_capital_budget(account):
    runtime, _, _, _ = account
    record = allocate(account)
    execution = runtime.execution.status()
    funding = LiveFunding(
        execution, runtime.engine.config, runtime.state["markets"], runtime.state["terms"]
    )
    with pytest.raises(ExecutionError, match="redeemable_usdgo_unavailable"):
        funding.redeem_usdgo(funding.book.locked["USDGO"] + 1)
    assert funding.book.deployable == money(record["capital_base_after"])
    assert money(record["capital_base_before"]) - funding.book.deployable == money(
        record["value_usdt"]
    )
    with pytest.raises(ExecutionError, match="no_external_assets_to_contribute"):
        live_contribution.preview(runtime.execution, account[2].now + 1000)


def test_transition_hour_and_following_hour_income_are_owned_once(account):
    runtime, broker, _market, root = account
    _market.now += 2 * HOUR  # Keep the independently pending startup-overlap hour out of this test.
    record = allocate(account)
    boundary = record["allocated_at"] // HOUR * HOUR
    batch_credit(broker, "USDGO", "0.02", boundary + HOUR + 1000)
    runtime.execution.import_interest(boundary + HOUR + 2000)
    rows = runtime.execution.status()["income_accounting"]["allocations"]
    first = next(iter(rows.values()))
    assert money(first.get("reserve_amount", 0)) == 0
    batch_credit(broker, "USDGO", "0.02", boundary + 2 * HOUR + 1000)
    runtime.execution.import_interest(boundary + 2 * HOUR + 2000)
    saved = runtime.execution.status()
    row = max(saved["income_accounting"]["allocations"].values(), key=lambda r: r["timestamp"])
    assert money(row["reserve_amount"]) == money(".01")
    assert money(row["amount"]) + money(row["external_amount"]) == money(".02")
    assert live_reserve.protected_spot(saved) == money(".01")
    assert live_reserve.summary(saved)["interest_receipts"] == 1
    restored = LiveExecution(root, broker, runtime.execution.account_id)
    restored.import_interest(boundary + 2 * HOUR + 3000)
    assert live_reserve.protected_spot(restored.status()) == money(".01")
    assert restored.status()["book"] == saved["book"]
    with pytest.raises(ExecutionError, match="no_external_assets_to_contribute"):
        live_contribution.preview(restored, boundary + 2 * HOUR + 4000)


def test_a_review_cannot_recontribute_protected_earn_even_with_valid_review_hash(account):
    runtime, _, market, _ = account
    record = allocate(account)
    item = record["locations"][0]
    review = {
        "account_id": runtime.execution.account_id,
        "created_at": market.now,
        "expires_at": market.now + 300000,
        "basis": "contribution_market_value",
        "value": "1",
        "assets": [
            {
                "coin": "USDGO",
                "amount": "1",
                "mark": "1.001",
                "value": "1.001",
                "locations": [
                    {
                        "location": "earn",
                        "holding_id": item["holding_id"],
                        "product_id": item["product_id"],
                        "amount": "1",
                    }
                ],
            }
        ],
    }
    review["review_hash"] = digest(review)
    runtime.execution.state["asset_contribution_review"] = review
    runtime.execution.save()
    before = runtime.execution.status()
    with pytest.raises(ExecutionError, match="contribution_assets_changed"):
        live_contribution.apply(runtime.execution, review["review_hash"], market.now + 1000)
    assert runtime.execution.status() == before


@pytest.mark.parametrize(
    "failure", ["intent", "pending", "error", "not_started", "not_running", "no_earn"]
)
def test_busy_or_unverified_state_never_changes_ownership(account, failure):
    runtime, broker, _, _ = account
    if failure == "intent":
        runtime.state["intent"] = {"kind": "trade"}
    elif failure == "pending":
        runtime.engine.pending = object()
    elif failure == "error":
        runtime.state["error"] = "needs_review"
    elif failure == "not_started":
        runtime.state["startup_complete"] = False
    elif failure == "not_running":
        runtime.state["desired"] = "paused"
    else:
        runtime.execution.state["book"]["locked"]["USDGO"] = "0"
        runtime.execution.save()
    before, provider = runtime.execution.status(), broker.state()
    if failure == "pending":
        strategy = {**runtime.state, "engine": {"pending": {"id": "pending"}}}
        with pytest.raises(ExecutionError):
            live_reserve.allocate_half(runtime.execution, strategy, KEY, account[2].now + 1000)
    else:
        with pytest.raises(ExecutionError):
            allocate(account)
    assert runtime.execution.status() == before and broker.state() == provider
