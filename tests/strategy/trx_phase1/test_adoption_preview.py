import json
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from freqtrade.rpc.api_server import api_trx
from freqtrade.rpc.api_server.app import create_app
from freqtrade.strategy.trx_phase1.adoption_preview import adoption_preview


NOW = 1800000000000


def source():
    def balance(coin, quantity):
        return {
            "coin": coin,
            "balance": quantity,
            "available": quantity,
            "frozen": "0",
            "locked": "0",
            "restricted_available": "0",
        }

    return {
        "account_type": "classic",
        "credentials_configured": True,
        "read_only": True,
        "stale": False,
        "error": None,
        "checks": {
            name: {"status": "success", "error": None}
            for name in ("balances", "savings", "prices", "orders", "conversion_orders")
        },
        "snapshot": {
            "observed_at": NOW,
            "completed_at": NOW + 1000,
            "account": {"uid_masked": "****1234"},
            "balances": [
                balance("USDT", "700"),
                balance("TRX", "2.12345678"),
                balance("USDGO", "0.20000001"),
            ],
            "savings": {
                "rows": [
                    {
                        "coin": "TRX",
                        "product_id": "p-trx",
                        "order_id": "h-trx",
                        "quantity": "1000.00000001",
                        "status": "in_holding",
                        "total_profit": "100",
                    },
                    {
                        "coin": "USDGO",
                        "product_id": "p-usdgo",
                        "order_id": "h-usdgo",
                        "quantity": "10000.12345678",
                        "status": "in_holding",
                        "total_profit": "200",
                    },
                    {
                        "coin": "USDT",
                        "product_id": "p-usdt",
                        "order_id": "h-usdt",
                        "quantity": "300",
                        "status": "in_holding",
                        "total_profit": "20",
                    },
                ],
                "since": NOW - 90 * 86400000,
                "until": NOW,
                "pages_complete": True,
                "complete_account_history": False,
                "full_account_holdings_verified": True,
                "coverage": {
                    "basis": "account_registration_to_snapshot",
                    "registered_at": NOW - 90 * 86400000,
                    "windows_complete": True,
                    "verified_coins": ["TRX", "USDGO", "USDT"],
                    "verification_method": "all_age_pages_and_stable_earn_overview",
                    "windows": [
                        {
                            "since": NOW - 90 * 86400000,
                            "until": NOW,
                            "pages": 2,
                            "pages_complete": True,
                        }
                    ],
                    "reconciliation": [
                        {
                            "coin": coin,
                            "holdings_quantity": quantity,
                            "overview_before_quantity": quantity,
                            "overview_after_quantity": quantity,
                            "difference_quantity": "0",
                            "matched": True,
                        }
                        for coin, quantity in (
                            ("TRX", "1000.00000001"),
                            ("USDGO", "10000.12345678"),
                        )
                    ],
                },
            },
            "prices": {
                "TRX": {"bid": "0.33333", "ask": "0.33334", "timestamp": NOW - 100},
                "USDGO": {"bid": "1.0009", "ask": "1.0010", "timestamp": NOW - 100},
            },
            "orders": {"rows": [], "limited": False},
            "conversion_orders": {"rows": [], "limited": False},
        },
    }


def preview(account=None, **kwargs):
    return adoption_preview(source() if account is None else account, now_ms=NOW + 2000, **kwargs)


def test_in_kind_quantities_are_not_cash_and_usdt_or_prior_profit_are_not_added():
    account = source()
    original = deepcopy(account)
    result = preview(account)
    assert result["reviewable"] and not result["committed"] and not result["confirmation_available"]
    assert result["startup_actions"] == [] and not result["trading_enabled"]
    manifest = result["manifest"]
    assert manifest["strategy_cash_usdt"] == "0"
    assets = {a["coin"]: a for a in manifest["assets"]}
    assert assets["TRX"]["quantity"] == "1002.12345679"
    assert assets["USDGO"]["quantity"] == "10000.32345679"
    assert len(manifest["locations"]) == 4
    assert all(a["historical_cost_usdt"] is None for a in assets.values())
    assert manifest["excluded_assets"] == [
        {"coin": "USDT", "spot_quantity": "700", "savings_quantity": "300", "quantity": "1000"}
    ]
    expected = Decimal("1002.12345679") * Decimal("0.33333") + Decimal("10000.32345679") * Decimal(
        "1.0009"
    )
    assert Decimal(manifest["initial_equity_usdt"]) == expected
    assert all(v == "0" for v in manifest["hypothetical_opening_profits"].values())
    assert manifest["coverage"]["full_account_holdings_verified"]
    assert not manifest["coverage"]["complete_account_history"]
    assert account == original
    account["snapshot"]["balances"][0]["available"] = "999999"
    account["snapshot"]["balances"][0]["balance"] = "999999"
    account["snapshot"]["savings"]["rows"][0]["total_profit"] = "99999999"
    assert preview(account)["manifest"]["initial_equity_usdt"] == manifest["initial_equity_usdt"]


def test_old_window_only_or_unsubstantiated_boolean_cannot_claim_all_age_completeness():
    account = source()
    account["snapshot"]["savings"].pop("coverage")
    result = preview(account)
    assert not result["reviewable"]
    assert not result["manifest"]["coverage"]["full_account_holdings_verified"]
    assert "full_savings_verification_required" in result["blockers"]
    assert "savings_creation_window_only" in result["warnings"]


@pytest.mark.parametrize("change", ["unmatched_amount", "window_gap", "unfinished_page"])
def test_completeness_requires_actual_contiguous_pages_and_reconciled_quantities(change):
    account = source()
    proof = account["snapshot"]["savings"]["coverage"]
    if change == "unmatched_amount":
        proof["reconciliation"][0]["overview_before_quantity"] = "999999"
    elif change == "window_gap":
        proof["windows"].append({**proof["windows"][0], "until": NOW - 1})
    else:
        proof["windows"][0]["pages_complete"] = False
    result = preview(account)
    assert not result["reviewable"]
    assert not result["manifest"]["coverage"]["full_account_holdings_verified"]


def test_cached_get_is_deterministic_but_quantity_or_valuation_changes_the_manifest():
    original = preview()
    again = adoption_preview(source(), now_ms=NOW + 3000)
    assert again["manifest_id"] == original["manifest_id"]
    account = source()
    account["snapshot"]["savings"]["rows"][0]["quantity"] = "1000.00000002"
    assert preview(account)["manifest_id"] != original["manifest_id"]
    account = source()
    account["snapshot"]["prices"]["USDGO"]["bid"] = "1.0008"
    assert preview(account)["manifest_id"] != original["manifest_id"]


def test_expired_preview_keeps_the_dated_values_but_is_not_reviewable():
    result = adoption_preview(source(), now_ms=NOW + 60001)
    assert result["stale"] and not result["reviewable"]
    assert result["manifest"] is not None and "snapshot_expired" in result["blockers"]


@pytest.mark.parametrize("section", ["balances", "savings", "prices"])
def test_failed_reads_do_not_invent_zero_opening_equity(section):
    account = source()
    account["checks"][section]["status"] = "error"
    result = preview(account)
    assert result["manifest"] is None and not result["reviewable"]
    assert section + "_read_required" in result["blockers"]


@pytest.mark.parametrize(
    "change,code",
    [
        ("duplicate_holding", "duplicate_holding"),
        ("duplicate_asset", "duplicate_asset"),
        ("missing_asset", "spot_balances_incomplete"),
        ("negative", "invalid_snapshot_values"),
        ("nan", "invalid_snapshot_values"),
        ("missing_quote", "invalid_snapshot_values"),
        ("old_quote", "invalid_valuation_quote"),
        ("pagination", "savings_pagination_incomplete"),
    ],
)
def test_unverifiable_amounts_and_coverage_do_not_produce_a_total(change, code):
    account = source()
    snapshot = account["snapshot"]
    if change == "duplicate_holding":
        snapshot["savings"]["rows"].append(deepcopy(snapshot["savings"]["rows"][0]))
    elif change == "duplicate_asset":
        snapshot["balances"].append(deepcopy(snapshot["balances"][1]))
    elif change == "missing_asset":
        snapshot["balances"] = [r for r in snapshot["balances"] if r["coin"] != "TRX"]
    elif change in ("negative", "nan"):
        snapshot["savings"]["rows"][0]["quantity"] = "-1" if change == "negative" else "NaN"
    elif change == "missing_quote":
        del snapshot["prices"]["TRX"]
    elif change == "old_quote":
        snapshot["prices"]["TRX"]["timestamp"] = NOW - 60001
    else:
        snapshot["savings"]["pages_complete"] = False
    result = preview(account)
    assert not result["reviewable"] and result["manifest"] is None
    assert code in result["blockers"]


@pytest.mark.parametrize(
    "change,code",
    [
        ("orders", "open_orders_require_review"),
        ("unverified_orders", "open_orders_unverified"),
        ("locked", "spot_assets_locked"),
        ("restricted", "spot_assets_restricted"),
        ("redeeming", "savings_status_requires_review"),
        ("uta", "classic_account_required"),
    ],
)
def test_visible_values_remain_blocked_when_ownership_or_availability_needs_review(change, code):
    account = source()
    if change == "orders":
        account["snapshot"]["orders"]["rows"] = [{"order_id": "open-order"}]
    elif change == "unverified_orders":
        account["snapshot"]["orders"]["limited"] = True
    elif change == "locked":
        row = account["snapshot"]["balances"][1]
        row["available"], row["locked"] = "1.12345678", "1"
    elif change == "restricted":
        account["snapshot"]["balances"][1]["restricted_available"] = "1"
    elif change == "redeeming":
        account["snapshot"]["savings"]["rows"][0]["status"] = "redeeming"
    else:
        account["account_type"] = "uta"
    result = preview(account)
    assert not result["reviewable"] and code in result["blockers"]
    assert result["manifest"] is not None


def test_no_snapshot_or_existing_strategy_never_becomes_an_acceptance():
    account = source()
    account["snapshot"] = None
    assert preview(account)["manifest"] is None
    result = preview(strategy_initialized=True)
    assert "strategy_already_initialized" in result["blockers"]
    assert not result["reviewable"] and not result["confirmation_available"]


def test_authenticated_get_does_not_connect_or_create_a_strategy_and_has_no_commit_route(
    tmp_path, monkeypatch
):
    from freqtrade.strategy.trx_phase1.account import ReadOnlyBitget
    from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
    from freqtrade.strategy.trx_phase1.live_service import LiveService

    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must not execute, connect or allocate")

    monkeypatch.setattr(ReadOnlyBitget, "request", forbidden)
    monkeypatch.setattr(LiveExecution, "create", forbidden)
    monkeypatch.setattr(LiveService, "prepare", forbidden)
    account = source()
    # A cached snapshot from the past is still a valid GET projection, explicitly stale.
    service = SimpleNamespace(
        config_path=tmp_path / "trx_phase1/bitget-account.local.json",
        status=Mock(return_value=account),
    )
    app = create_app(
        {
            "user_data_dir": str(tmp_path),
            "api_server": {
                "username": "reader",
                "password": "test-password",
                "jwt_secret_key": "x" * 40,
            },
            "trx_phase1": {"existing_live_only": True, "auto_resume_live": False},
        }
    )
    app.dependency_overrides[api_trx.get_account_service] = lambda: service
    client = TestClient(app)
    assert client.get("/api/v1/trx/account/adoption-preview").status_code == 401
    response = client.get("/api/v1/trx/account/adoption-preview", auth=("reader", "test-password"))
    assert response.status_code == 200 and response.json()["preview_only"]
    assert (
        client.post(
            "/api/v1/trx/account/adoption-preview", auth=("reader", "test-password")
        ).status_code
        == 405
    )
    assert not list(tmp_path.rglob("*"))
    assert "test-password" not in json.dumps(response.json())
    service.status.assert_called_once()
