import json
from copy import deepcopy
from unittest.mock import MagicMock

import ccxt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from freqtrade.rpc.api_server import api_trx
from freqtrade.strategy.trx_phase1.account import (
    CONFIG_NAME,
    ENV_KEYS,
    AccountBusy,
    AccountReader,
    AccountReadError,
    AccountService,
    AccountSettings,
    ReadOnlyBitget,
    load_settings,
    normalize_balances,
    normalize_fill,
    normalize_funding_balances,
    normalize_product_terms,
)


def response(data):
    return {"code": "00000", "data": data}


@pytest.fixture(autouse=True)
def no_external_credentials(monkeypatch):
    for name in ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def account(tmp_path):
    path = tmp_path / "trx_phase1" / CONFIG_NAME
    path.parent.mkdir()
    config = {"api_key": "test-key", "secret_key": "test-secret", "passphrase": "test-pass"}
    path.write_text(json.dumps(config))
    path.chmod(0o600)
    client = MagicMock(spec=ReadOnlyBitget)
    client.privateSpotGetV2SpotAccountInfo.return_value = response({"userId": "123456789"})
    client.privateSpotGetV2SpotAccountAssets.return_value = response(
        [
            {
                "coin": "USDT",
                "available": "250.1",
                "frozen": "20",
                "locked": "5",
                "limitAvailable": "2",
            },
            {"coin": "BTC", "available": "99"},
        ]
    )
    client.privateSpotGetV2SpotTradeUnfilledOrders.return_value = response([])
    client.privateSpotGetV2SpotTradeFills.return_value = response([])
    client.privateEarnGetV2EarnSavingsAssets.return_value = response(
        {"resultList": [], "endId": ""}
    )
    client.privateEarnGetV2EarnSavingsProduct.return_value = response([])
    client.privateEarnGetV2EarnSavingsRecords.return_value = response(
        {"resultList": [], "endId": ""}
    )
    client.privateUtaGetV3AccountFundingAssets.return_value = response([])
    client.privateUtaGetV3AccountFundingFinancialRecords.return_value = response(
        {"list": [], "cursor": ""}
    )
    clock = [1800000000.0]
    client.publicSpotGetV2SpotMarketTickers.side_effect = lambda params: response(
        [{"symbol": params["symbol"], "bidPr": "0.31", "askPr": "0.32", "ts": int(clock[0] * 1000)}]
    )
    factory = MagicMock(return_value=client)
    service = AccountService(tmp_path, factory, lambda: clock[0])
    return service, client, path, config, clock, factory


def test_missing_key_never_constructs_client_or_claims_connection(tmp_path):
    factory = MagicMock()
    service = AccountService(tmp_path, factory)
    assert service.status()["status"] == "not_configured"
    assert service.refresh()["status"] == "not_configured"
    factory.assert_not_called()
    assert not (tmp_path / "trx_phase1").exists()


def test_local_status_does_not_call_exchange_and_masks_settings(account):
    service, client, _, _, _, factory = account
    result = service.status()
    assert result["status"] == "ready" and result["credentials_configured"]
    assert result["snapshot"] is None
    factory.assert_not_called()
    assert not client.mock_calls
    settings, _, _ = load_settings(service.config_path)
    assert "test-key" not in repr(settings)
    assert "test-secret" not in settings.model_dump_json()


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "v2/spot/trade/place-order"),
        ("GET", "v2/spot/wallet/withdrawal"),
        ("POST", "v2/earn/savings/subscribe"),
        ("POST", "v2/earn/savings/redeem"),
        ("POST", "v3/account/switch"),
        ("DELETE", "v2/spot/trade/unfilled-orders"),
    ],
)
def test_transport_blocks_financial_writes_before_network(mocker, method, path):
    send = mocker.patch.object(ccxt.bitget, "request")
    with pytest.raises(PermissionError, match="allowlisted account reads"):
        ReadOnlyBitget().request(path, "privateSpot", method)
    send.assert_not_called()


def test_real_ccxt_signing_reads_without_logging_credentials(mocker, caplog):
    client = ReadOnlyBitget(
        {"apiKey": "test-key", "secret": "test-secret", "password": "test-pass", "verbose": True}
    )
    fetch = mocker.patch.object(client, "fetch", return_value=response([]))
    result = client.privateSpotGetV2SpotAccountAssets({"assetType": "all"})
    assert result == response([])
    url, method, headers, _ = fetch.call_args.args[:4]
    assert method == "GET" and url.startswith("https://api.bitget.com/")
    assert headers["ACCESS-KEY"] == "test-key"
    assert headers["ACCESS-PASSPHRASE"] == "test-pass"
    assert headers["ACCESS-SIGN"]
    client.log("test-secret", headers)
    assert "test-pass" not in caplog.text and "test-secret" not in caplog.text


def test_environment_is_complete_and_never_mixed_with_file(account, monkeypatch):
    service, _, _, _, _, _ = account
    monkeypatch.setenv(ENV_KEYS[0], "another-account-key")
    status = service.status()
    assert status["status"] == "invalid_configuration"
    assert status["error"] == "incomplete_credentials"
    for key, value in zip(ENV_KEYS, ("new-key", "new-secret", "new-pass"), strict=True):
        monkeypatch.setenv(key, value)
    settings, source, complete = load_settings(service.config_path)
    assert complete and source == "environment"
    assert settings.secret_key.get_secret_value() == "new-secret"


def test_private_file_permissions_and_invalid_input_are_sanitized(account):
    service, _, path, _, _, factory = account
    path.chmod(0o644)
    assert service.status()["error"] == "private_file_permissions"
    path.chmod(0o600)
    path.write_text('{"secret_key":"secret-in-broken-json",')
    result = service.refresh()
    assert result["error"] == "invalid_configuration"
    assert "secret-in-broken-json" not in json.dumps(result)
    factory.assert_not_called()


def test_refresh_reads_real_components_without_double_counting_or_strategy_import(account):
    service, client, _, _, _, factory = account
    status = service.refresh()
    assert status["status"] == "connected"
    assert status["strategy_allocation"] == "unassigned"
    assert not status["trading_enabled"] and not status["earn_write_enabled"]
    assert status["snapshot"]["account"]["uid_masked"] == "****6789"
    balances = status["snapshot"]["balances"]
    assert len(balances) == 1
    assert balances[0]["balance"] == "275.1"
    assert balances[0]["restricted_available"] == "2"
    assert "return_ratio" not in json.dumps(status)
    assert "test-key" not in json.dumps(status) and "test-secret" not in json.dumps(status)
    assert factory.call_args.args[0]["options"]["defaultType"] == "spot"
    client.privateSpotGetV2SpotAccountAssets.assert_called_once_with({"assetType": "all"})
    client.close.assert_called_once()


def test_balances_failure_is_unknown_not_zero_and_old_sections_are_not_reused(account):
    service, client, _, _, clock, _ = account
    service.refresh()
    clock[0] += 10
    client.privateSpotGetV2SpotAccountAssets.side_effect = ccxt.PermissionDenied("test-secret")
    status = service.refresh()
    assert status["status"] == "partial"
    assert status["snapshot"]["balances"] is None
    assert status["checks"]["balances"] == {"status": "error", "error": "permission_denied"}
    assert "test-secret" not in json.dumps(status)


def test_auth_failure_retains_only_explicitly_stale_previous_snapshot(account):
    service, client, _, _, clock, _ = account
    good = service.refresh()["snapshot"]
    clock[0] += 10
    client.privateSpotGetV2SpotAccountInfo.side_effect = ccxt.AuthenticationError("test-pass")
    status = service.refresh()
    assert status["status"] == "error" and status["stale"]
    assert status["snapshot"] == good
    assert status["error"] == "authentication_failed"
    assert "test-pass" not in json.dumps(status)


def test_uta_mode_rejection_is_not_reported_as_missing_key_permission(account):
    service, client, _, _, _, _ = account
    client.privateSpotGetV2SpotAccountInfo.side_effect = ccxt.PermissionDenied(
        'bitget {"code":"40085","msg":"test-secret"}'
    )
    result = service.refresh()
    assert result["error"] == "uta_classic_api_unsupported"
    assert "test-secret" not in json.dumps(result)


def test_uta_classic_earn_rejection_keeps_supported_reads_visible(account):
    service, client = configure_uta(account)
    for method in (
        client.privateEarnGetV2EarnSavingsAssets,
        client.privateEarnGetV2EarnSavingsProduct,
        client.privateEarnGetV2EarnSavingsRecords,
    ):
        method.side_effect = ccxt.PermissionDenied('bitget {"code":"40085","msg":"hidden"}')
    result = service.refresh()
    assert result["status"] == "partial"
    assert result["checks"]["balances"]["status"] == "success"
    assert result["checks"]["funding_balances"]["status"] == "success"
    for section in ("savings", "savings_records"):
        assert result["checks"][section] == {
            "status": "unsupported",
            "error": "uta_classic_api_unsupported",
        }
        assert result["snapshot"][section] is None
    assert result["checks"]["products"]["status"] == "partial"
    assert all(
        check["error"] == "uta_classic_api_unsupported"
        for check in result["snapshot"]["products"]["coin_checks"].values()
    )


def test_uta_fills_filters_documented_all_spot_response_without_inventing_trx(account):
    service, client = configure_uta(account)
    trx = {
        "category": "SPOT",
        "symbol": "TRXUSDT",
        "execId": "trx-fill",
        "orderId": "o1",
        "side": "buy",
        "execPrice": "0.3",
        "execQty": "100",
        "execValue": "30",
        "createdTime": "1799999999000",
        "feeDetail": [{"feeCoin": "TRX", "fee": "-0.1"}],
    }
    other = {**trx, "symbol": "USDGOUSDT", "execId": "other-fill"}
    client.privateUtaGetV3TradeFills.return_value = response(
        {"list": [other, trx], "cursor": "older"}
    )
    result = service.refresh()
    assert result["checks"]["fills"]["status"] == "success"
    assert [row["trade_id"] for row in result["snapshot"]["fills"]["rows"]] == ["trx-fill"]
    assert result["snapshot"]["fills"]["limited"]
    conversion = result["snapshot"]["conversion_fills"]
    assert [row["trade_id"] for row in conversion["rows"]] == ["other-fill"]
    assert conversion["rows"][0]["base_coin"] == "USDGO" and conversion["limited"]
    client.privateUtaGetV3TradeFills.assert_called_once()
    params = client.privateUtaGetV3TradeFills.call_args.args[0]
    assert "symbol" not in params and params["category"] == "SPOT"
    client.privateUtaGetV3TradeFills.return_value = response(
        {"list": [{**trx, "category": "USDT-FUTURES"}], "cursor": ""}
    )
    result = AccountReader(client, AccountSettings(account_type="uta"), 1800000000000).collect()
    assert result[1]["fills"]["error"] == "unexpected_market"


def test_freshness_cooldown_and_concurrent_refresh(account):
    service, _, _, _, clock, _ = account
    service.refresh()
    with pytest.raises(AccountBusy, match="too_frequent"):
        service.refresh()
    clock[0] += 61
    assert service.status()["status"] == "stale"
    service._refresh_lock.acquire()
    try:
        with pytest.raises(AccountBusy, match="in_progress"):
            service.refresh()
    finally:
        service._refresh_lock.release()
    assert service.refresh()["status"] == "connected"


def test_key_change_or_removal_discards_old_account_data(account):
    service, _, path, config, _, _ = account
    service.refresh()
    config["api_key"] = "changed-key"
    path.write_text(json.dumps(config))
    assert service.status()["snapshot"] is None
    path.unlink()
    status = service.status()
    assert status["status"] == "not_configured" and status["snapshot"] is None


def test_key_change_during_refresh_does_not_publish_previous_account(account):
    service, client, path, config, _, _ = account

    def change_key(*args):
        path.write_text(json.dumps({**config, "api_key": "new-account-key"}))
        service.status()
        return response([])

    client.privateSpotGetV2SpotTradeFills.side_effect = change_key
    status = service.refresh()
    assert status["status"] == "ready" and status["snapshot"] is None


def test_uta_balance_is_separate_and_savings_compatibility_is_probed(account):
    service, client, path, config, _, _ = account
    path.write_text(json.dumps({**config, "account_type": "uta"}))
    client.privateUtaGetV3AccountInfo.return_value = response({"userId": "123456789"})
    client.privateUtaGetV3AccountAssets.return_value = response(
        {
            "assets": [
                {
                    "coin": "USDT",
                    "available": "900",
                    "balance": "1000",
                    "equity": "1100",
                    "locked": "250",
                    "debt": "0",
                }
            ]
        }
    )
    client.privateUtaGetV3TradeUnfilledOrders.return_value = response({"list": [], "cursor": ""})
    client.privateUtaGetV3TradeFills.return_value = response({"list": [], "cursor": ""})
    status = service.refresh()
    row = status["snapshot"]["balances"][0]
    assert row["total"] == "1000" and row["equity"] == "1100"
    assert row["integrity"] == "uta_requires_strategy_allocation"
    assert status["checks"]["savings"]["status"] == "success"
    assert status["earn_access"]["status"] == "read_confirmed"
    assert status["earn_access"]["account_mapping"] == "unverified"
    client.privateEarnGetV2EarnSavingsAssets.assert_called_once()
    client.privateUtaGetV3AccountFundingAssets.assert_called_once()
    client.privateSpotGetV2SpotAccountAssets.assert_not_called()


def test_invalid_numeric_or_duplicate_balances_are_rejected():
    row = {"coin": "TRX", "available": "10", "locked": "0", "frozen": "0", "limitAvailable": "0"}
    with pytest.raises(AccountReadError, match="duplicate_asset"):
        normalize_balances([row, row])
    for value in ("NaN", "Infinity", "-1"):
        with pytest.raises((AccountReadError, ValueError)):
            normalize_balances([{**row, "available": value}])


def fill():
    return {
        "symbol": "TRXUSDT",
        "side": "buy",
        "tradeId": "f1",
        "orderId": "o1",
        "priceAvg": "0.31",
        "size": "100",
        "amount": "31",
        "cTime": "1800000000000",
        "feeDetail": {"feeCoin": "TRX", "totalFee": "-0.1", "totalDeductionFee": ""},
    }


def test_fill_preserves_native_fee_and_market_buy_quantity_units(account):
    service, client, _, _, _, _ = account
    client.privateSpotGetV2SpotTradeFills.return_value = response([fill()])
    client.privateSpotGetV2SpotTradeUnfilledOrders.return_value = response(
        [
            {
                "symbol": "TRXUSDT",
                "side": "buy",
                "orderId": "o1",
                "orderType": "market",
                "status": "partially_filled",
                "priceAvg": "0.31",
                "size": "50",
                "baseVolume": "100",
                "cTime": "1800000000000",
            }
        ]
    )
    result = service.refresh()["snapshot"]
    assert result["orders"]["rows"][0]["quantity_coin"] == "USDT"
    value = result["fills"]["rows"][0]
    assert value["fees"][0] == {"coin": "TRX", "amount": "-0.1", "deduction_amount": "0"}
    assert "rebate" not in json.dumps(value)


def test_uta_fill_field_mapping_and_no_cross_product_import():
    row = {
        "symbol": "TRXUSDT",
        "category": "SPOT",
        "side": "sell",
        "execId": "e1",
        "orderId": "o1",
        "execPrice": "0.33",
        "execQty": "100",
        "execValue": "33",
        "createdTime": "1800000000000",
        "feeDetail": [{"feeCoin": "USDT", "fee": "0.033"}],
    }
    assert normalize_fill(row, True)["turnover"] == "33"
    with pytest.raises(AccountReadError, match="unexpected_market"):
        normalize_fill({**row, "category": "USDT-FUTURES"}, True)


def savings_row(order_id):
    return {
        "productId": "p1",
        "orderId": order_id,
        "productCoin": "TRX",
        "interestCoin": "TRX",
        "periodType": "flexible",
        "holdAmount": "100",
        "totalProfit": "1.1",
        "status": "in_holding",
    }


def test_savings_pagination_keeps_interest_separate_and_marks_creation_coverage(account):
    service, client, _, _, _, _ = account
    first = [savings_row(str(index)) for index in range(100)]
    client.privateEarnGetV2EarnSavingsAssets.side_effect = [
        response({"resultList": first, "endId": "cursor1"}),
        response({"resultList": [savings_row("last")], "endId": "last"}),
        response({"resultList": [], "endId": ""}),
    ]
    result = service.refresh()["snapshot"]["savings"]
    assert len(result["rows"]) == 101
    assert result["rows"][0]["quantity"] == "100"
    assert result["rows"][0]["total_profit"] == "1.1"
    assert result["pages_complete"] and not result["complete_account_history"]
    assert client.privateEarnGetV2EarnSavingsAssets.call_args.kwargs == {}
    assert (
        client.privateEarnGetV2EarnSavingsAssets.call_args_list[1].args[0]["idLessThan"]
        == "cursor1"
    )
    assert client.privateEarnGetV2EarnSavingsAssets.call_args.args[0]["idLessThan"] == "last"


def test_duplicate_pagination_is_an_error_not_partial_asset_total(account):
    service, client, _, _, _, _ = account
    first = [savings_row(str(index)) for index in range(100)]
    client.privateEarnGetV2EarnSavingsAssets.return_value = response(
        {"resultList": first, "endId": "x"}
    )
    result = service.refresh()
    assert result["snapshot"]["savings"] is None
    assert result["checks"]["savings"]["error"] == "duplicate_record"


def test_truncated_fills_cannot_be_presented_as_full_history(account):
    service, client, _, _, _, _ = account
    client.privateSpotGetV2SpotTradeFills.return_value = response(
        [{**fill(), "tradeId": str(index)} for index in range(100)]
    )
    value = service.refresh()["snapshot"]["fills"]
    assert value["limited"] and value["maximum_rows"] == 100


def test_disabled_account_makes_no_requests(account):
    service, _, path, config, _, factory = account
    path.write_text(json.dumps({**config, "enabled": False}))
    assert service.refresh()["status"] == "disabled"
    factory.assert_not_called()


def test_api_uses_same_readonly_service_and_exposes_no_financial_actions(account):
    service = account[0]
    app = FastAPI()
    app.include_router(api_trx.router)
    app.dependency_overrides[api_trx.get_account_service] = lambda: service
    with TestClient(app) as client:
        assert client.get("/trx/account").json()["status"] == "ready"
        data = client.post("/trx/account/refresh").json()
        assert data["read_only"] and not data["trading_enabled"]
        for action in ("start", "buy", "sell", "subscribe", "redeem", "transfer"):
            assert client.post("/trx/account/" + action).status_code == 404


def test_returned_snapshot_cannot_mutate_service_state(account):
    service = account[0]
    original = deepcopy(service.refresh())
    returned = service.status()
    returned["snapshot"]["balances"][0]["available"] = "99999999"
    assert service.status() == original


def test_reader_uses_fixed_spot_and_recent_window(account):
    _, client, _, _, _, _ = account
    reader = AccountReader(client, AccountSettings(), 1800000000000)
    reader.recent_trading(fills=True)
    params = client.privateSpotGetV2SpotTradeFills.call_args.args[0]
    assert params["symbol"] == "TRXUSDT"
    assert int(params["endTime"]) - int(params["startTime"]) == 7 * 86400000


def test_observations_persist_after_restart_and_local_reads_never_add_points(account):
    service, client, path, _, clock, factory = account
    for coin in ("USDGO", "TRX"):
        client.privateSpotGetV2SpotAccountAssets.return_value["data"].append(
            {"coin": coin, "available": "0", "frozen": "0", "locked": "0", "limitAvailable": "0"}
        )
    first = service.refresh()["history"]
    assert first["points"][0]["spot_equity_usdt"] == "275.1"
    assert service.status()["history"] == first
    clock[0] += 6
    restarted = AccountService(path.parent.parent, factory, lambda: clock[0])
    assert restarted.status()["history"]["points"] == []  # identity not yet verified
    reloaded = restarted.refresh()["history"]
    assert reloaded["total_points"] == 2
    assert reloaded["points"][0] == first["points"][0]


def test_failed_read_records_a_gap_and_key_change_hides_previous_history(account):
    service, client, path, config, clock, _ = account
    service.refresh()
    clock[0] += 6
    client.privateSpotGetV2SpotAccountInfo.side_effect = ccxt.NetworkError("private-message")
    failed = service.refresh()
    assert failed["stale"]
    assert failed["history"]["total_points"] == 2
    gap = failed["history"]["points"][-1]
    assert gap["valuation_status"] == "read_failed" and gap["spot_equity_usdt"] is None
    path.write_text(json.dumps({**config, "api_key": "different-account-key"}))
    assert service.status()["history"]["points"] == []
    client.privateSpotGetV2SpotAccountInfo.side_effect = None
    client.privateSpotGetV2SpotAccountInfo.return_value = response({"userId": "987654321"})
    clock[0] += 6
    assert service.refresh()["history"]["total_points"] == 1


def test_history_storage_error_does_not_discard_verified_account_data(account, mocker):
    service = account[0]
    mocker.patch.object(service.history, "append", side_effect=OSError("private-message"))
    result = service.refresh()
    assert result["status"] == "connected" and result["snapshot"]["balances"]
    assert result["history"]["error"] == "history_storage_unavailable"
    assert "private-message" not in json.dumps(result)


@pytest.mark.parametrize("field,value", [("symbol", "BTCUSDT"), ("ts", "1"), ("bidPr", "0")])
def test_valuation_quotes_reject_wrong_market_stale_and_invalid_prices(account, field, value):
    _, client, _, _, clock, _ = account

    def ticker(params):
        row = {
            "symbol": params["symbol"],
            "bidPr": "0.3",
            "askPr": "0.31",
            "ts": int(clock[0] * 1000),
        }
        row[field] = value
        return response([row])

    client.publicSpotGetV2SpotMarketTickers.side_effect = ticker
    data, checks = AccountReader(client, AccountSettings(), int(clock[0] * 1000)).collect()
    assert data["prices"] is None
    assert checks["prices"]["status"] == "error"


def configure_uta(account):
    service, client, path, config, _, _ = account
    path.write_text(json.dumps({**config, "account_type": "uta"}))
    client.privateUtaGetV3AccountInfo.return_value = response({"userId": "123456789"})
    client.privateUtaGetV3AccountAssets.return_value = response({"assets": []})
    client.privateUtaGetV3TradeUnfilledOrders.return_value = response({"list": [], "cursor": ""})
    client.privateUtaGetV3TradeFills.return_value = response({"list": [], "cursor": ""})
    return service, client


def savings_record(order_id="o1", kind="pay_interest", amount="2.5", coin="TRX"):
    return {
        "orderId": order_id,
        "orderType": kind,
        "coinName": coin,
        "settleCoinName": coin,
        "productType": "flexible",
        "amount": amount,
        "ts": "1799999999000",
    }


def product(coin="TRX", product_id="p1"):
    return {
        "coin": coin,
        "productId": product_id,
        "periodType": "flexible",
        "status": "in_progress",
        "apyList": [{"minStepVal": "0", "maxStepVal": "10000000", "currentApy": "10.33"}],
    }


def terms():
    return {
        "singleMinAmount": "10",
        "singleMaxAmount": "2000",
        "remainingAmount": "5000",
        "subscribePrecision": "6",
        "interestTime": "1800003600000",
        "settleTime": "1800086400000",
        "redeemDelay": "",
    }


def test_uta_missing_key_still_never_probes_savings_or_funding(tmp_path):
    path = tmp_path / "trx_phase1" / CONFIG_NAME
    path.parent.mkdir()
    path.write_text(json.dumps({"account_type": "uta"}))
    factory = MagicMock()
    service = AccountService(tmp_path, factory)
    status = service.refresh()
    assert status["earn_access"]["status"] == "pending"
    assert status["snapshot"] is None
    factory.assert_not_called()


def test_uta_savings_mode_failure_is_explicit_and_does_not_hide_funding(account):
    service, client = configure_uta(account)
    client.privateEarnGetV2EarnSavingsAssets.side_effect = ccxt.NotSupported(
        "sensitive-api-message"
    )
    client.privateUtaGetV3AccountFundingAssets.return_value = response(
        [{"coin": "USDGO", "available": "5000", "frozen": "0", "balance": "5000"}]
    )
    result = service.refresh()
    assert result["status"] == "partial"
    assert result["snapshot"]["savings"] is None
    assert result["snapshot"]["funding_balances"][0]["balance"] == "5000"
    assert result["checks"]["savings"]["error"] == "account_api_not_supported"
    assert result["earn_access"]["status"] == "partial"
    assert "sensitive-api-message" not in json.dumps(result)


def test_uta_funding_permission_failure_does_not_invent_zeros(account):
    service, client = configure_uta(account)
    client.privateUtaGetV3AccountFundingAssets.side_effect = ccxt.PermissionDenied(
        "private-message"
    )
    status = service.refresh()
    assert status["snapshot"]["funding_balances"] is None
    assert status["checks"]["funding_balances"]["error"] == "permission_denied"
    assert status["checks"]["savings"]["status"] == "success"


def test_savings_and_funding_records_are_observed_once_without_double_credit(account):
    service, client = configure_uta(account)
    client.privateEarnGetV2EarnSavingsRecords.side_effect = lambda params: response(
        {"resultList": [savings_record()] if params["coin"] == "TRX" else [], "endId": ""}
    )

    def funding(params):
        rows = (
            [
                {
                    "id": "f1",
                    "coin": "TRX",
                    "type": "savings_flexible_profit",
                    "amount": "2.5",
                    "balance": "102.5",
                    "ts": "1799999999000",
                }
            ]
            if (params["type"] == "savings_flexible_profit")
            else []
        )
        return response({"list": rows, "cursor": ""})

    client.privateUtaGetV3AccountFundingFinancialRecords.side_effect = funding
    result = service.refresh()
    savings, cash = result["snapshot"]["savings_records"], result["snapshot"]["funding_records"]
    assert savings["rows"][0]["amount"] == cash["rows"][0]["amount"] == "2.5"
    assert savings["credited_to_strategy"] is False
    assert cash["redemption_arrival_verified"] is False
    assert result["strategy_allocation"] == "unassigned"
    assert "total_interest" not in result["snapshot"] and "equity" not in result["snapshot"]
    assert client.privateUtaGetV3AccountFundingFinancialRecords.call_count == 3


def test_interest_settlement_currency_is_not_assumed_to_match_principal(account):
    _, client, _, _, _, _ = account
    row = {**savings_record(), "settleCoinName": "BGB"}
    client.privateEarnGetV2EarnSavingsRecords.side_effect = lambda params: response(
        {"resultList": [row] if params["coin"] == "TRX" else [], "endId": ""}
    )
    rows = AccountReader(client, AccountSettings(), 1800000000000).savings_records()["rows"]
    assert rows[0]["coin"] == "TRX" and rows[0]["settlement_coin"] == "BGB"


def test_account_history_includes_redemptions_and_interest_when_default_is_subscribe(account):
    _, client, _, _, _, _ = account

    def records(params):
        kind = params.get("orderType", "subscribe")
        row = {**savings_record(kind), "orderType": kind}
        return response({"resultList": [row] if params["coin"] == "TRX" else [], "endId": ""})

    client.privateEarnGetV2EarnSavingsRecords.side_effect = records
    data = AccountReader(client, AccountSettings(), 1800000000000).savings_records()
    assert {r["kind"] for r in data["rows"]} == {"subscribe", "redeem", "pay_interest"}
    assert data["credited_to_strategy"] is False
    assert client.privateEarnGetV2EarnSavingsRecords.call_count == 9


def test_savings_records_follow_short_page_cursor_and_deduplicate_identical_events(account):
    _, client, _, _, _, _ = account
    calls = []

    def records(params):
        calls.append(dict(params))
        if params["coin"] != "TRX":
            return response({"resultList": [], "endId": ""})
        if "idLessThan" not in params:
            return response({"resultList": [savings_record()], "endId": "next"})
        return response({"resultList": [savings_record(), savings_record("o2")], "endId": ""})

    client.privateEarnGetV2EarnSavingsRecords.side_effect = records
    data = AccountReader(client, AccountSettings(), 1800000000000).savings_records()
    assert len(data["rows"]) == 2 and data["pages_complete"]
    assert calls[-1]["idLessThan"] == "next"


def test_conflicting_payout_is_not_silently_deduplicated(account):
    _, client, _, _, _, _ = account
    client.privateEarnGetV2EarnSavingsRecords.side_effect = lambda params: response(
        {
            "resultList": [savings_record(amount="1"), savings_record(amount="2")]
            if (params["coin"] == "TRX")
            else [],
            "endId": "",
        }
    )
    with pytest.raises(AccountReadError, match="conflicting_record"):
        AccountReader(client, AccountSettings(), 1800000000000).savings_records()


def test_funding_cursor_and_signed_redemption_fields(account):
    _, client, _, _, _, _ = account
    row = {
        "id": "f1",
        "coin": "USDT",
        "type": "savings_flexible_subscribe",
        "amount": "-100",
        "balance": "900",
        "ts": "1799999999000",
    }

    def records(params):
        if params["type"] != "savings_flexible_subscribe" or params.get("cursor"):
            return response({"list": [], "cursor": ""})
        return response({"list": [row], "cursor": "next"})

    client.privateUtaGetV3AccountFundingFinancialRecords.side_effect = records
    data = AccountReader(
        client, AccountSettings(account_type="uta"), 1800000000000
    ).funding_records()
    assert data["rows"][0]["amount"] == "-100"
    assert (
        client.privateUtaGetV3AccountFundingFinancialRecords.call_args_list[1].args[0]["cursor"]
        == "next"
    )
    assert data["redemption_arrival_verified"] is False


@pytest.mark.parametrize(
    "change", [{"ts": "1800000000001"}, {"orderType": "deduction"}, {"coinName": "USDGO"}]
)
def test_record_scope_violation_discards_section(account, change):
    _, client, _, _, _, _ = account
    client.privateEarnGetV2EarnSavingsRecords.side_effect = lambda params: response(
        {
            "resultList": [{**savings_record(), **change}] if params["coin"] == "TRX" else [],
            "endId": "",
        }
    )
    with pytest.raises(AccountReadError, match="unexpected_savings_record"):
        AccountReader(client, AccountSettings(), 1800000000000).savings_records()


def test_subscription_terms_are_separate_from_tier_caps_and_eligibility(account):
    service, client = configure_uta(account)
    client.privateEarnGetV2EarnSavingsProduct.side_effect = lambda params: response(
        [product()] if params["coin"] == "TRX" else []
    )
    client.privateEarnGetV2EarnSavingsSubscribeInfo.return_value = response(terms())
    data = service.refresh()["snapshot"]["products"]
    row = data["rows"][0]
    assert row["tiers"][0]["maximum"] == "10000000"
    assert row["terms"]["remaining_amount"] == "5000"
    assert row["terms"]["single_maximum"] == "2000"
    assert row["terms"]["redemption_delay"] is None
    assert row["terms"]["eligibility_verified"] is False and data["quota_verified"] is False
    client.privateEarnGetV2EarnSavingsSubscribeInfo.assert_called_once_with(
        {"productId": "p1", "periodType": "flexible"}
    )


def test_terms_permission_error_preserves_products_and_is_partial(account):
    service, client = configure_uta(account)
    client.privateEarnGetV2EarnSavingsProduct.side_effect = lambda params: response(
        [product()] if params["coin"] == "TRX" else []
    )
    client.privateEarnGetV2EarnSavingsSubscribeInfo.side_effect = ccxt.PermissionDenied("hidden")
    result = service.refresh()
    assert result["status"] == "partial"
    assert result["checks"]["products"]["status"] == "partial"
    assert result["snapshot"]["products"]["rows"][0]["terms"] is None
    assert result["snapshot"]["products"]["rows"][0]["terms_error"] == "permission_denied"
    assert "hidden" not in json.dumps(result)


def test_terms_detail_request_count_is_bounded(account):
    _, client, _, _, _, _ = account
    client.privateEarnGetV2EarnSavingsProduct.side_effect = lambda params: response(
        [product(product_id=str(i)) for i in range(15)] if params["coin"] == "TRX" else []
    )
    client.privateEarnGetV2EarnSavingsSubscribeInfo.return_value = response(terms())
    data = AccountReader(client, AccountSettings(), 1800000000000).products()
    assert data["partial"] and len(data["rows"]) == 15
    assert client.privateEarnGetV2EarnSavingsSubscribeInfo.call_count == 12
    assert data["rows"][-1]["terms_status"] == "not_requested"


def test_funding_balance_components_mismatch_is_reported_not_repaired(account):
    service, client = configure_uta(account)
    client.privateUtaGetV3AccountFundingAssets.return_value = response(
        [{"coin": "USDGO", "available": "100", "frozen": "10", "balance": "112"}]
    )
    data = service.refresh()
    row = data["snapshot"]["funding_balances"][0]
    assert row["component_difference"] == "2" and row["balance"] == "112"
    assert data["checks"]["funding_balances"]["status"] == "partial"
    assert data["status"] == "partial"
    with pytest.raises(AccountReadError, match="duplicate_asset"):
        normalize_funding_balances(
            [{"coin": "TRX", "available": "0", "frozen": "0", "balance": "0"}] * 2
        )


def test_earn_disabled_prevents_all_new_financial_queries(account):
    service, client = configure_uta(account)
    path = account[2]
    path.write_text(json.dumps({**account[3], "account_type": "uta", "include_earn": False}))
    result = service.refresh()
    assert result["earn_access"]["status"] == "disabled"
    assert result["checks"]["funding_balances"]["status"] == "disabled"
    client.privateUtaGetV3AccountFundingAssets.assert_not_called()
    client.privateUtaGetV3AccountFundingFinancialRecords.assert_not_called()
    client.privateEarnGetV2EarnSavingsRecords.assert_not_called()


def test_shared_read_budget_prevents_unbounded_network_calls(account):
    _, client, _, _, _, _ = account
    reader = AccountReader(client, AccountSettings(), 1800000000000)
    reader.deadline = 0
    with pytest.raises(AccountReadError, match="read_budget_exceeded"):
        reader.savings_records()
    client.privateEarnGetV2EarnSavingsRecords.assert_not_called()


def test_repeating_cursor_or_full_page_without_cursor_never_claims_complete(account):
    _, client, _, _, _, _ = account
    client.privateEarnGetV2EarnSavingsRecords.return_value = response(
        {"resultList": [savings_record(coin="USDT")], "endId": "loop"}
    )
    with pytest.raises(AccountReadError, match="pagination_incomplete"):
        AccountReader(client, AccountSettings(), 1800000000000).savings_records()
    client.privateEarnGetV2EarnSavingsRecords.return_value = response(
        {"resultList": [savings_record(str(i), coin="USDT") for i in range(100)], "endId": ""}
    )
    with pytest.raises(AccountReadError, match="pagination_incomplete"):
        AccountReader(client, AccountSettings(), 1800000000000).savings_records()


def test_unfinished_page_limit_discards_all_payout_rows(account):
    _, client, _, _, _, _ = account
    client.privateEarnGetV2EarnSavingsRecords.side_effect = [
        response({"resultList": [savings_record(str(i), coin="USDT")], "endId": str(i)})
        for i in range(20)
    ]
    with pytest.raises(AccountReadError, match="pagination_incomplete"):
        AccountReader(client, AccountSettings(), 1800000000000).savings_records()
    assert client.privateEarnGetV2EarnSavingsRecords.call_count == 20


def test_terms_nonfinite_or_inverted_amounts_are_rejected():
    with pytest.raises(ValueError):
        normalize_product_terms({**terms(), "remainingAmount": "NaN"})
    with pytest.raises(AccountReadError):
        normalize_product_terms({**terms(), "singleMinAmount": "2001"})


def test_new_funding_reads_use_actual_signed_ccxt_requests_only(mocker):
    client = ReadOnlyBitget(
        {"apiKey": "test-key", "secret": "test-secret", "password": "test-pass"}
    )
    fetch = mocker.patch.object(client, "fetch", return_value=response([]))
    client.privateUtaGetV3AccountFundingAssets()
    assert fetch.call_args.args[1] == "GET"
    assert "/api/v3/account/funding-assets" in fetch.call_args.args[0]
    client.privateUtaGetV3AccountFundingFinancialRecords({"type": "savings_flexible_profit"})
    assert fetch.call_args.args[1] == "GET"
    assert "savings_flexible_profit" in fetch.call_args.args[0]
    with pytest.raises(PermissionError):
        client.privateUtaPostV3AccountTransfer({})


def test_both_spot_markets_preserve_native_quantity_and_independent_failures(account):
    service, client, _, _, clock, _ = account
    client.privateSpotGetV2SpotTradeFills.side_effect = lambda params: response(
        [{**fill(), "symbol": params["symbol"], "tradeId": params["symbol"]}]
    )
    result = service.refresh()
    assert result["checks"]["conversion_fills"]["status"] == "success"
    row = result["snapshot"]["conversion_fills"]["rows"][0]
    assert row["symbol"] == "USDGOUSDT" and row["base_coin"] == "USDGO"
    assert row["fees"][0]["coin"] == "TRX"  # Never rewrite a native fee currency.
    assert result["snapshot"]["fills"]["rows"][0]["base_coin"] == "TRX"
    clock[0] += 10

    def query(params):
        if params["symbol"] == "USDGOUSDT":
            raise ccxt.PermissionDenied("private-message")
        return response([fill()])

    client.privateSpotGetV2SpotTradeFills.side_effect = query
    result = service.refresh()
    assert result["checks"]["fills"]["status"] == "success"
    assert result["snapshot"]["conversion_fills"] is None
    assert result["checks"]["conversion_fills"]["error"] == "permission_denied"
    assert "private-message" not in json.dumps(result)


def test_products_include_unheld_coins_and_retain_success_if_another_coin_fails(account):
    service, client, *_ = account

    def query(params):
        assert params["filter"] == "all"
        if params["coin"] == "USDT":
            raise ccxt.PermissionDenied("hidden")
        return response([{**product(product_id=params["coin"]), "coin": params["coin"]}])

    client.privateEarnGetV2EarnSavingsProduct.side_effect = query
    client.privateEarnGetV2EarnSavingsSubscribeInfo.return_value = response(terms())
    data = service.refresh()["snapshot"]["products"]
    assert [row["coin"] for row in data["rows"]] == ["TRX", "USDGO"]
    assert data["partial"] and data["coin_checks"]["USDT"]["error"] == "permission_denied"
    assert all(row["terms_status"] == "success" for row in data["rows"])
    assert "hidden" not in json.dumps(data)


def test_product_detail_business_error_is_explicit_without_hiding_rates(account):
    service, client, *_ = account
    client.privateEarnGetV2EarnSavingsProduct.side_effect = lambda params: response(
        [product()] if params["coin"] == "TRX" else []
    )
    client.privateEarnGetV2EarnSavingsSubscribeInfo.return_value = {
        "code": "59012",
        "msg": "private-message",
        "data": None,
    }
    data = service.refresh()["snapshot"]["products"]
    assert data["rows"][0]["terms_error"] == "product_detail_unavailable"
    assert data["rows"][0]["tiers"] and data["partial"]
    assert "private-message" not in json.dumps(data)
