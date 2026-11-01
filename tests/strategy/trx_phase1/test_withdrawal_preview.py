"""Capital outflow economics, Earn precision and an entirely read-only HTTP boundary."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import localcontext
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from freqtrade.strategy.trx_phase1 import live_history_app, startup_basis
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_history import LiveHistory
from freqtrade.strategy.trx_phase1.withdrawal_preview import WithdrawalPreviewError, preview
from tests.strategy.trx_phase1.test_adoption_preview import NOW
from tests.strategy.trx_phase1.test_live_history import save
from tests.strategy.trx_phase1.test_startup_basis import (
    adopted,
    authorize_fixture,
    captured_start,
)


@pytest.fixture
def states(tmp_path):
    service, observed, _ = adopted(tmp_path)
    captured = captured_start(service, observed, NOW + 10000)
    authorize_fixture(service, captured)
    startup_basis.commit(service, captured)
    _, execution, strategy = startup_basis.context(service.root)
    execution["interest_checked_at"] = NOW + 10000
    execution["income_accounting"] = {"status": "active"}
    for rule in execution["products"]:
        rule.update(redemption_minimum="0.1", redemption_decimals=4, redemption_evidence="fixture")
    strategy["engine"]["config"].update(
        buy_quantity_dynamic=False,
        buy_fraction=0.12,
        price_position_enabled=True,
        price_position_buy_multiplier=0.25,
    )
    strategy["price_reference"] = {"value": 0.5}
    for market in strategy["markets"].values():
        market.update(timestamp=NOW + 10000, fee="0")
    return execution, strategy


@pytest.mark.parametrize("fx", ["1.003", "1.001"])
def test_partial_withdrawal_reduces_budget_without_changing_profit_nav_or_trx_cost(states, fx):
    execution, strategy = states
    strategy["markets"]["USDGOUSDT"]["bid"] = fx
    original = deepcopy(states)
    result = preview(execution, strategy, "1000", NOW + 11000)
    assert states == original
    before, after = result["before"], result["after"]
    with localcontext() as context:
        context.prec = 50
        amount = money(1000) * money(fx)
        assert money(before["equity_usdt"]) - money(after["equity_usdt"]) == amount
        assert money(before["net_capital_usdt"]) - money(after["net_capital_usdt"]) == amount
        assert money(before["profit_usdt"]) == money(after["profit_usdt"])
        assert abs(money(before["nav"]) - money(after["nav"])) < money("1e-40")
        assert (
            money(before["buy"]["capital_base_usdt"]) - money(after["buy"]["capital_base_usdt"])
            == amount
        )
        assert money(result["buy_budget_reduction_usdt"]) == amount * money(".12")
        cost = money(execution["book"]["usdgo_cost"]) * 1000 / money(execution["book"]["usdgo"])
        assert money(result["retained_withdrawn_usdgo_valuation_profit_usdt"]) == amount - cost
        old_fx_profit = money(execution["book"]["usdgo"]) * money(fx) - money(
            execution["book"]["usdgo_cost"]
        )
        remaining_fx_profit = money(after["usdgo_quantity"]) * money(fx) - money(
            after["usdgo_cost_usdt"]
        )
        assert abs(old_fx_profit - remaining_fx_profit - (amount - cost)) < money("1e-40")
    assert before["initial_capital_usdt"] == after["initial_capital_usdt"]
    assert before["usdt_cash"] == after["usdt_cash"] == "0"
    assert result["performance_started_at"] == execution["created_at"]
    assert result["preview_only"] and not result["execution_enabled"]


def test_withdrawal_uses_owned_earn_only_and_preserves_minimum_redemption_excess(states):
    execution, strategy = states
    execution["book"]["locked"]["USDGO"] = execution["book"]["usdgo"]
    holding = next(row for row in execution["holdings"].values() if row["coin"] == "USDGO")
    holding["principal"] = execution["book"]["usdgo"]
    holding["unassigned_amount"] = "100000"
    execution["unassigned_offsets"]["USDGO"] = "100000"
    result = preview(execution, strategy, ".01", NOW + 11000)
    assert money(result["sources"]["strategy_earn_to_redeem"]) == money(".1")
    assert money(result["sources"]["strategy_spot_remaining"]) == money(".09")
    assert money(result["withdrawal"]["quantity"]) == money(".01")
    with pytest.raises(WithdrawalPreviewError, match="exceeds_strategy"):
        preview(execution, strategy, "20000", NOW + 11000)


def test_price_position_multiplier_and_available_cash_cap_match_current_budget_convention(states):
    execution, strategy = states
    strategy["price_reference"]["value"] = 0.1
    result = preview(execution, strategy, "1000", NOW + 11000)
    assert money(result["before"]["buy"]["effective_fraction"]) == money(".03")
    strategy["price_reference"]["value"] = 0.5
    result = preview(execution, strategy, "9999", NOW + 11000)
    assert money(result["after"]["buy"]["target_budget_usdt"]) <= money(
        result["after"]["buy"]["available_usdt_estimate"]
    )


def test_recovery_buy_cap_and_conversion_fee_are_included(states):
    execution, strategy = states
    engine = strategy["engine"]
    engine.update(
        recovery_reference=".3", recovery_started_at=0, last_close=NOW, last_price=engine["anchor"]
    )
    engine["config"]["recovery_buy_fraction"] = ".02"
    strategy["markets"]["USDGOUSDT"]["fee"] = ".001"
    result = preview(execution, strategy, "1000", NOW + 11000)
    assert money(result["before"]["buy"]["effective_fraction"]) == money(".02")
    assert money(result["buy_budget_reduction_usdt"]) == money("1000") * money(
        strategy["markets"]["USDGOUSDT"]["bid"]
    ) * money(".02")
    assert money(result["before"]["buy"]["available_usdt_estimate"]) < money(
        execution["book"]["usdgo"]
    ) * money(strategy["markets"]["USDGOUSDT"]["bid"])


@pytest.mark.parametrize("amount", ["0", "-1", "NaN", "Infinity", "0.000000001", "999999999"])
def test_invalid_or_excess_withdrawal_changes_nothing(states, amount):
    before = deepcopy(states)
    with pytest.raises(WithdrawalPreviewError):
        preview(*states, amount, NOW + 11000)
    assert states == before


@pytest.mark.parametrize(
    "problem", ["stale", "operation", "intent", "basis", "income", "precision", "dynamic"]
)
def test_unsafe_snapshot_or_unsupported_sizing_fails_closed(states, problem):
    execution, strategy = states
    if problem == "stale":
        execution["interest_checked_at"] -= 61000
    elif problem == "operation":
        execution["operation"] = {"status": "accepted"}
    elif problem == "intent":
        strategy["intent"] = {"kind": "trade"}
    elif problem == "basis":
        execution["performance_basis"]["started_at"] += 1
    elif problem == "income":
        execution["income_accounting"]["status"] = "holding_evidence_required"
    elif problem == "precision":
        execution["products"][1]["redemption_decimals"] = None
    else:
        strategy["engine"]["config"]["buy_quantity_dynamic"] = True
    before = deepcopy(states)
    with pytest.raises(WithdrawalPreviewError):
        preview(execution, strategy, "1000", NOW + 11000)
    assert states == before


def test_http_preview_requires_authentication_and_cannot_confirm_or_modify_native_files(
    states, tmp_path, monkeypatch
):
    execution, strategy = states
    source = tmp_path / "readonly-source"
    source.mkdir()
    for key, state in (("execution", execution), ("strategy", strategy)):
        save(source / (key + ".sqlite"), key, state)
    archive = LiveHistory(source, tmp_path / "history.sqlite", execution["account_id"])
    monkeypatch.setattr(
        live_history_app, "time", SimpleNamespace(time=lambda: (NOW + 11000) / 1000)
    )
    config = {
        "account_id": execution["account_id"],
        "username": "owner",
        "jwt_secret_key": "test-withdrawal-key-which-is-long-enough",
    }
    headers = {
        "Authorization": "Bearer "
        + jwt.encode(
            {
                "identity": {"u": "owner"},
                "type": "access",
                "exp": datetime.now(UTC) + timedelta(minutes=5),
            },
            config["jwt_secret_key"],
            algorithm="HS256",
        )
    }
    before = {p.name: p.read_bytes() for p in source.glob("*.sqlite")}
    with TestClient(live_history_app.create_app(config, archive)) as client:
        path = "/api/v1/trx/live/withdrawal-preview"
        params = {"account_id": execution["account_id"], "amount": "1000"}
        assert client.get(path, params=params).status_code == 401
        result = client.get(path, params=params, headers=headers)
        assert result.status_code == 200 and result.json()["preview_only"]
        assert (
            client.get(path, params={**params, "account_id": "other"}, headers=headers).status_code
            == 409
        )
        assert (
            client.get(path, params={**params, "amount": "1e3"}, headers=headers).status_code == 422
        )
        assert client.post(path, json=params, headers=headers).status_code == 405
        assert (
            client.post(
                "/api/v1/trx/live/withdrawal/confirm", json=params, headers=headers
            ).status_code
            == 404
        )
    assert {p.name: p.read_bytes() for p in source.glob("*.sqlite")} == before


def test_projection_is_rejected_if_the_execution_book_changes_during_read(states, monkeypatch):
    execution, strategy = states
    updated = deepcopy(execution)
    updated["book"]["external_flows"] = "-1000"
    reads = iter([execution, strategy, updated])
    history = SimpleNamespace(read_source=lambda name: next(reads))
    monkeypatch.setattr(
        live_history_app, "time", SimpleNamespace(time=lambda: (NOW + 11000) / 1000)
    )
    with pytest.raises(HTTPException) as error:
        live_history_app.project_withdrawal(
            history, {"account_id": execution["account_id"]}, execution["account_id"], "1000"
        )
    assert error.value.status_code == 409
    assert error.value.detail == "withdrawal_snapshot_changed"
