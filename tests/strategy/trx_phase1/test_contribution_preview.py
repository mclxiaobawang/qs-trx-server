"""Explicit source selection, capital conservation and a non-confirmable read boundary."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import localcontext

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from freqtrade.strategy.trx_phase1 import live_history_app
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.contribution_preview import (
    DAY,
    HOUR,
    OFFSET,
    ContributionPreviewError,
    fingerprint,
    preview,
)
from freqtrade.strategy.trx_phase1.income_allocation import effective_units
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_history import LiveHistory
from tests.strategy.trx_phase1.test_live_contribution import execution as execution  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_execution import NOW
from tests.strategy.trx_phase1.test_live_history import save


@pytest.fixture
def states(execution):
    now = NOW + 2 * HOUR + 2000
    cfg = ResearchConfig(
        variant="dynamic_period_only",
        buy_fraction=0.12,
        price_position_enabled=True,
        price_position_buy_multiplier=0.25,
    ).model_dump()
    state = execution.status()
    strategy = {
        "account_id": state["account_id"],
        "source_snapshot": cfg,
        "source_sha256": fingerprint(cfg),
        "activated_at": NOW,
        "startup_complete": True,
        "desired": "running",
        "phase": "observing",
        "error": None,
        "intent": None,
        "engine": {
            "config": deepcopy(cfg),
            "anchor": ".3",
            "buy_threshold": ".014",
            "pending": None,
            "last_close": now - 1000,
            "last_price": ".3",
            "recovery_reference": 0,
        },
        "markets": {
            symbol: execution.broker.market(symbol, now) for symbol in ["TRXUSDT", "USDGOUSDT"]
        },
        "price_reference": {"value": ".5", "day": (now + OFFSET) // DAY},
    }
    return state, strategy, now


def project(states, amount="0.5", coin="USDGO", **kwargs):
    e, s, now = states
    return preview(e, s, coin, amount, now, **kwargs)


def test_partial_earn_selection_excludes_all_spot_and_other_coins_without_mutation(states):
    original = deepcopy(states)
    r = project(states)
    assert states == original
    assert r["selection"]["holding_id"] == "USDGO"
    assert money(r["selection"]["amount"]) == money(".5")
    assert money(r["selection"]["external_earn_after"]) == money(".4")
    assert r["excluded"]["all_external_spot"] == states[0]["unassigned_offsets"]
    assert money(r["excluded"]["remaining_external_earn"]["TRX"]["amount"]) == money("4.46553")
    assert r["performance_started_at"] == states[1]["activated_at"]
    assert r["preview_only"] and not r["execution_enabled"]
    assert "review_hash" not in r and "asset_contribution_review" not in states[0]


def test_capital_increases_but_nav_profit_trx_cost_and_old_income_do_not_reset(states):
    r = project(states)
    a, b = r["before"], r["after"]
    with localcontext() as context:
        context.prec = 60
        value = money(r["selection"]["value_usdt"])
        assert money(b["net_capital_usdt"]) - money(a["net_capital_usdt"]) == value
        assert money(b["equity_usdt"]) - money(a["equity_usdt"]) == value
        assert money(b["usdgo_cost_usdt"]) - money(a["usdgo_cost_usdt"]) == value
        assert money(a["profit_usdt"]) == money(b["profit_usdt"])
        assert abs(money(a["nav"]) - money(b["nav"])) < money("1e-40")
        assert money(b["fund_units"]) > money(a["fund_units"])
        assert money(b["buy"]["target_budget_usdt"]) - money(
            a["buy"]["target_budget_usdt"]
        ) == value * money(".12")
    assert a["trx_cost_usdt"] == b["trx_cost_usdt"]
    assert a["initial_capital_usdt"] == b["initial_capital_usdt"]


@pytest.mark.parametrize("amount", ["0", "-1", "1e-8", "NaN", "0.123456789", "1" * 16, 1, True])
def test_invalid_amounts_never_round_or_change_the_selection(states, amount):
    original = deepcopy(states)
    with pytest.raises(ContributionPreviewError, match="invalid_amount"):
        project(states, amount)
    assert states == original


@pytest.mark.parametrize("coin", ["TRX", "USDT", "usdgo"])
def test_coin_filter_does_not_sweep_other_external_assets(states, coin):
    with pytest.raises(ContributionPreviewError, match="selected_coin_unsupported"):
        project(states, coin=coin)


def test_never_uses_external_spot_to_fill_a_shortfall_or_selects_a_reserve(states):
    with pytest.raises(ContributionPreviewError, match="exceeds_selected_external"):
        project(states, "0.90000001")
    states[0]["holdings"]["USDGO"]["reserved_amount"] = "0.5"
    with pytest.raises(ContributionPreviewError, match="exceeds_selected_external"):
        project(states)


def test_holding_selection_is_explicit_when_multiple_sources_exist(states):
    states[0]["holdings"]["second"] = {
        **states[0]["holdings"]["USDGO"],
        "principal": "0",
        "interest": "0",
        "unassigned_amount": "5",
    }
    with pytest.raises(ContributionPreviewError, match="ambiguous"):
        project(states)
    assert project(states, holding_id="USDGO")["selection"]["holding_id"] == "USDGO"
    with pytest.raises(ContributionPreviewError, match="selected_holding"):
        project(states, holding_id="TRX")


@pytest.mark.parametrize(
    "fault,code",
    [
        ("stale", "snapshot_stale"),
        ("pending", "requires_idle"),
        ("operation", "requires_idle"),
        ("income", "income_unresolved"),
        ("source", "binding_required"),
        ("location", "location_mismatch"),
        ("p75", "current_price_reference"),
        ("dynamic", "dynamic_sizing_unsupported"),
    ],
)
def test_stale_or_unverified_state_is_not_a_valid_projection(states, fault, code):
    e, s, now = states
    if fault == "stale":
        e["interest_checked_at"] = now - 60001
    elif fault == "pending":
        s["engine"]["pending"] = {"id": "intent"}
    elif fault == "operation":
        e["operation"] = {"status": "accepted"}
    elif fault == "income":
        e["pending_income"] = {"bill": "unresolved"}
    elif fault == "source":
        s["source_sha256"] = "mismatch"
    elif fault == "location":
        e["book"]["locked"]["USDGO"] = "0"
    elif fault == "p75":
        s["price_reference"]["day"] -= 1
    elif fault == "dynamic":
        s["engine"]["config"]["buy_quantity_dynamic"] = True
    original = deepcopy(states)
    with pytest.raises(ContributionPreviewError, match=code):
        project(states)
    assert states == original


def test_budget_uses_target_price_p75_and_recovery_cap_not_initial_buy_fraction(states):
    states[1]["price_reference"]["value"] = ".1"
    r = project(states)
    assert money(r["after"]["buy"]["effective_fraction"]) == money(".03")
    states[1]["engine"].update(recovery_reference=".29", recovery_started_at=0)
    r = project(states)
    assert money(r["after"]["buy"]["effective_fraction"]) == money(".02")
    assert money(r["after"]["buy"]["target_budget_usdt"]) <= money(
        r["after"]["buy"]["available_usdt_estimate"]
    )


def test_hypothetical_checkpoint_keeps_transition_hour_external_and_next_full_hour_owned(states):
    r = project(states)
    history = deepcopy(states[0]["income_accounting"]["checkpoints"])
    old = history[-1]["positions"]["USDGO"]
    history.append(r["interest"]["hypothetical_checkpoint"])
    full = r["interest"]["first_complete_period_start"]
    transition, _ = effective_units(history, "USDGO", full - HOUR, full)
    new, _ = effective_units(history, "USDGO", full, full + HOUR)
    assert transition == {k: money(old[k]) for k in ["strategy", "external"]}
    assert new["strategy"] == money(old["strategy"]) + money(".5")
    assert new["external"] == money(old["external"]) - money(".5")
    assert r["interest"]["historical_allocations_unchanged"]


def test_authenticated_get_only_route_has_no_financial_writer(states, tmp_path, monkeypatch):
    e, s, now = states
    source = tmp_path / "source"
    source.mkdir()
    save(source / "execution.sqlite", "execution", e)
    save(source / "strategy.sqlite", "strategy", s)
    history = LiveHistory(source, tmp_path / "archive.sqlite", e["account_id"])
    config = {
        "account_id": e["account_id"],
        "username": "owner",
        "jwt_secret_key": "a-long-test-history-secret-for-preview",
    }
    monkeypatch.setattr(live_history_app.time, "time", lambda: now / 1000)
    auth = {
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
    original = {p.name: p.read_bytes() for p in source.glob("*.sqlite")}
    with TestClient(live_history_app.create_app(config, history)) as client:
        path = "/api/v1/trx/live/contribution-preview?coin=USDGO&amount=0.5"
        assert client.get(path).status_code == 401
        response = client.get(path, headers=auth)
        assert response.status_code == 200 and response.json()["execution_enabled"] is False
        assert client.post(path, headers=auth).status_code == 405
        assert client.get(path + "&account_id=other", headers=auth).status_code == 409
        assert client.get(path.replace("USDGO", "TRX"), headers=auth).status_code == 422
        assert client.get(path.replace("0.5", "NaN"), headers=auth).status_code == 422
        assert client.get(path.replace("0.5", "1.0"), headers=auth).status_code == 409
        assert client.get("/contribution-preview").status_code == 200
    assert {p.name: p.read_bytes() for p in source.glob("*.sqlite")} == original


def test_a_live_writer_changing_the_book_invalidates_the_projection(states, monkeypatch):
    e, s, now = states
    changed = deepcopy(e)
    changed["book"]["fund_units"] = "123"
    snapshots = iter([e, s, changed])

    class Reader:
        def read_source(self, key):
            return next(snapshots)

    monkeypatch.setattr(live_history_app.time, "time", lambda: now / 1000)
    with pytest.raises(HTTPException) as failure:
        live_history_app.project_contribution(
            Reader(), {"account_id": e["account_id"]}, e["account_id"], "USDGO", "0.5"
        )
    assert failure.value.status_code == 409
    assert failure.value.detail == "contribution_snapshot_changed"
