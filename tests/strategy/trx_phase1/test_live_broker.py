import base64
import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
import requests

from freqtrade.strategy.trx_phase1.account import AccountSettings
from freqtrade.strategy.trx_phase1.live_broker import (
    ClassicBroker,
    ClassicTransport,
    ExecutionError,
    WriteRejected,
    WriteUnknown,
)


@pytest.fixture
def transport(mocker):
    settings = AccountSettings(
        account_type="classic",
        api_key="fixture-key",
        secret_key="fixture-secret",
        passphrase="fixture-passphrase",
    )
    session = mocker.Mock(spec=requests.Session)
    session.request.return_value = SimpleNamespace(
        status_code=200, json=lambda: {"code": "00000", "data": {"orderId": "123"}}
    )
    return ClassicTransport(settings, writes_enabled=True, session=session)


def request_body():
    return {
        "symbol": "USDGOUSDT",
        "side": "buy",
        "orderType": "market",
        "force": "gtc",
        "size": "100",
        "clientOid": "trx-fixture",
        "requestTime": "1792166400000",
        "receiveWindow": "10000",
    }


def test_empty_bill_window_is_local_instead_of_an_invalid_exchange_query(transport):
    broker = ClassicBroker(transport)
    assert broker.bills("TRX", 1792166400000, 1792166400000) == []
    transport.session.request.assert_not_called()


@pytest.mark.parametrize(
    "profit, expected", [("0.01384620", "0.01384620"), (None, None), ("NaN", None)]
)
def test_product_income_observation_does_not_change_owned_holding_or_require_an_extra_read(
    profit, expected, mocker
):
    row = {
        "orderId": "holding",
        "productId": "trx-flex",
        "periodType": "flexible",
        "productCoin": "TRX",
        "holdAmount": "300.89140740",
        "interestCoin": "TRX",
        "totalProfit": profit,
    }
    transport = mocker.Mock()
    transport.call.return_value = {"resultList": [row], "endId": ""}
    broker = ClassicBroker(transport, clock=lambda: 1792247069046)
    holdings = broker.holdings(1792166400000, 1792247069046)
    assert holdings == {
        "holding": {
            "order_id": "holding",
            "product_id": "trx-flex",
            "coin": "TRX",
            "amount": "300.89140740",
        }
    }
    assert broker.savings_observation == {
        "observed_at": 1792247069046,
        "holdings": {
            "holding": {
                "coin": "TRX",
                "product_id": "trx-flex",
                "interest_coin": "TRX",
                "total_profit": expected,
            }
        },
    }
    assert transport.call.call_count == 1
    assert transport.call.call_args.args[0] == "GET"


def test_post_signs_exact_transmitted_body_and_never_retries(transport):
    body = request_body()
    transport.call("POST", "v2/spot/trade/place-order", body)
    call = transport.session.request.call_args
    assert call.args == ("POST", "https://api.bitget.com/api/v2/spot/trade/place-order")
    assert json.loads(call.kwargs["data"]) == body
    stamp = call.kwargs["headers"]["ACCESS-TIMESTAMP"]
    expected = base64.b64encode(
        hmac.new(
            b"fixture-secret",
            (stamp + "POST/api/v2/spot/trade/place-order" + call.kwargs["data"]).encode(),
            hashlib.sha256,
        ).digest()
    ).decode()
    assert call.kwargs["headers"]["ACCESS-SIGN"] == expected
    assert call.kwargs["allow_redirects"] is False
    assert transport.session.request.call_count == 1


@pytest.mark.parametrize(
    "path",
    [
        "v2/spot/wallet/withdrawal",
        "v2/spot/wallet/transfer",
        "v3/trade/place-order",
        "v2/earn/elite/redeem",
    ],
)
def test_non_strategy_write_paths_are_denied_before_transport(transport, path):
    with pytest.raises(ExecutionError, match="transport_scope_denied"):
        transport.call("POST", path, {})
    transport.session.request.assert_not_called()


def test_read_only_execution_client_cannot_write_or_follow_redirect(transport):
    transport.writes_enabled = False
    with pytest.raises(ExecutionError, match="transport_scope_denied"):
        transport.call("POST", "v2/spot/trade/place-order", request_body())
    transport.session.request.assert_not_called()
    transport.session.request.return_value = SimpleNamespace(
        status_code=302, json=lambda: {"code": "00000", "data": {}}
    )
    with pytest.raises(ExecutionError, match="response_unverified"):
        transport.call("GET", "v2/spot/account/info")


@pytest.mark.parametrize(
    "change",
    [{"symbol": "BTCUSDT"}, {"orderType": "limit"}, {"tpslType": "tpsl"}, {"side": "short"}],
)
def test_unsupported_trading_parameters_never_reach_exchange(transport, change):
    with pytest.raises(ExecutionError, match="unsupported_spot_request"):
        transport.call("POST", "v2/spot/trade/place-order", {**request_body(), **change})
    transport.session.request.assert_not_called()


def test_timeouts_and_malformed_acknowledgements_never_expose_secrets_or_retry(transport):
    transport.session.request.side_effect = requests.Timeout("fixture-secret fixture-key")
    with pytest.raises(WriteUnknown) as exc:
        transport.call("POST", "v2/spot/trade/place-order", request_body())
    assert str(exc.value) == "exchange_response_unavailable"
    assert transport.session.request.call_count == 1


@pytest.mark.parametrize(
    ("code", "expected"),
    [("40014", WriteRejected), ("43118", WriteUnknown), ("45001", WriteUnknown)],
)
def test_post_error_classification_is_conservative(transport, code, expected):
    transport.session.request.return_value = SimpleNamespace(
        status_code=400, json=lambda: {"code": code, "msg": "fixture-secret"}
    )
    with pytest.raises(expected) as exc:
        transport.call("POST", "v2/spot/trade/place-order", request_body())
    assert "fixture-secret" not in str(exc.value)
    assert transport.session.request.call_count == 1


def test_uta_credentials_do_not_construct_a_classic_execution_client():
    with pytest.raises(ExecutionError, match="classic_account_required"):
        ClassicTransport(AccountSettings(account_type="uta"))


def test_new_account_can_resolve_an_available_product_without_an_existing_holding(mocker):
    transport = mocker.Mock()
    tiers = [{"minStepVal": "0", "maxStepVal": "10000000", "currentApy": "9.88"}]

    def read(method, path, params):
        assert method == "GET"
        if path.endswith("/product"):
            # The provider's held intersection omits the product before startup.
            return (
                [
                    {
                        "productId": "trx-flex",
                        "coin": "TRX",
                        "periodType": "flexible",
                        "status": "in_progress",
                        "apyList": tiers,
                    }
                ]
                if params["filter"] == "all"
                else []
            )
        return {"singleMinAmount": "1", "subscribePrecision": "8", "redeemDelay": ""}

    transport.call.side_effect = read
    result = ClassicBroker(transport).product("TRX", "trx-flex")
    assert result["tiers"] == tiers
    assert result["singleMinAmount"] == "1"
    assert result["redeemDelay"] == ""  # not a verified redemption minimum or delay


def test_explicit_coin_balance_must_be_observed_and_restricted_amount_is_unspendable(mocker):
    t = mocker.Mock()
    t.call.side_effect = [
        {"userId": "123"},
        [
            {"coin": c, "available": "20", "frozen": "0", "locked": "0", "limitAvailable": "5"}
            for c in ("USDT", "USDGO", "TRX")
        ],
    ]
    broker = ClassicBroker(t)
    assert broker.account_id() == hashlib.sha256(b"123").hexdigest()
    balances = broker.balances()
    assert balances["USDT"]["available"] == "15"
    assert balances["USDT"]["restricted"] == "5"
    t.call.side_effect = None
    t.call.return_value = []
    with pytest.raises(ExecutionError, match="balance_missing"):
        broker.balances()


def test_bulk_balances_keep_explicit_zero_rows_without_coin_filtered_null_restrictions(mocker):
    t = mocker.Mock()
    t.call.return_value = [
        {"coin": c, "available": "0", "frozen": "0", "locked": "0", "limitAvailable": "0"}
        for c in ("USDT", "USDGO", "TRX")
    ]
    balances = ClassicBroker(t).balances()
    assert balances["TRX"] == dict.fromkeys(("available", "frozen", "locked", "restricted"), "0")
    t.call.assert_called_once_with("GET", "v2/spot/account/assets", {"assetType": "all"})
    t.call.return_value.pop()
    with pytest.raises(ExecutionError, match="balance_missing"):
        ClassicBroker(t).balances()


def test_savings_records_preserve_multiple_interest_events_for_one_order(mocker):
    t = mocker.Mock()
    rows = [
        {
            "orderId": "same-order",
            "orderType": "pay_interest",
            "ts": str(1792166400000 + n * 3600000),
            "coinName": "TRX",
            "settleCoinName": "TRX",
            "amount": "1",
        }
        for n in range(2)
    ]
    t.call.side_effect = lambda method, path, params: {
        "resultList": rows if params["orderType"] == "pay_interest" else [],
        "endId": "same-order",
    }
    assert len(ClassicBroker(t).savings_records("TRX", 1792166400000, 1792166400000 + 7200000)) == 2


def test_records_read_all_types_explicitly_when_exchange_defaults_to_subscriptions(mocker):
    t = mocker.Mock()

    def records(method, path, params):
        kind = params.get("orderType", "subscribe")
        return {
            "resultList": [
                {
                    "orderId": "independent-history-id",
                    "orderType": kind,
                    "ts": "1792166400001",
                    "coinName": "USDGO",
                    "settleCoinName": "USDGO",
                    "amount": "0.1",
                }
            ],
            "endId": "",
        }

    t.call.side_effect = records
    broker = ClassicBroker(t)
    rows = broker.savings_records("USDGO", 1792166400000, 1792166401000)
    assert {r["orderType"] for r in rows} == {"subscribe", "redeem", "pay_interest"}
    assert t.call.call_count == 3
    t.call.reset_mock()
    rows = broker.savings_records("USDGO", 1792166400000, 1792166401000, order_type="redeem")
    assert [r["orderType"] for r in rows] == ["redeem"]
    assert t.call.call_count == 1
    with pytest.raises(ExecutionError, match="unsupported_savings_record_type"):
        broker.savings_records("USDGO", 1792166400000, 1792166401000, order_type="unknown")
    assert t.call.call_count == 1


def test_full_page_cursor_that_never_advances_is_not_accepted_as_complete(mocker):
    t = mocker.Mock()
    t.call.return_value = [{"tradeId": str(i)} for i in range(100)]
    with pytest.raises(ExecutionError, match="pagination_incomplete"):
        ClassicBroker(t).pages("v2/spot/trade/fills", {}, cursor_key="tradeId")


def test_fill_fee_sign_and_native_currency_are_preserved(mocker):
    t = mocker.Mock()
    t.call.return_value = [
        {
            "symbol": "TRXUSDT",
            "orderId": "order",
            "tradeId": "trade",
            "side": "buy",
            "size": "10",
            "amount": "3",
            "cTime": "1792166400000",
            "feeDetail": {"feeCoin": "TRX", "totalFee": "-0.01"},
        }
    ]
    fills = ClassicBroker(t).fills("order", "TRXUSDT", 1792166400000, 1792166401000)
    assert fills[0]["fees"] == [{"coin": "TRX", "cost": "0.01"}]
    assert fills[0]["quantity"] == "10"


def test_third_currency_fee_deduction_is_not_silently_ignored(mocker):
    t = mocker.Mock()
    t.call.return_value = [
        {
            "symbol": "TRXUSDT",
            "orderId": "order",
            "tradeId": "trade",
            "side": "buy",
            "size": "10",
            "amount": "3",
            "cTime": "1792166400000",
            "feeDetail": {"feeCoin": "TRX", "totalFee": "-0.01", "totalDeductionFee": "0.005"},
        }
    ]
    with pytest.raises(ExecutionError, match="third_currency_fee_requires_reconciliation"):
        ClassicBroker(t).fills("order", "TRXUSDT", 1792166400000, 1792166401000)


def test_quote_freshness_uses_observation_time_after_slow_prior_reads(mocker):
    t = mocker.Mock()
    t.call.side_effect = [
        [
            {
                "symbol": "USDGOUSDT",
                "status": "online",
                "quantityPrecision": "4",
                "pricePrecision": "4",
                "minTradeUSDT": "1",
            }
        ],
        {"ts": "1792166405000", "bids": [["1.001", "1000"]], "asks": [["1.002", "1000"]]},
        {"takerFeeRate": "0"},
    ]
    broker = ClassicBroker(t, clock=lambda: 1792166405100)
    assert broker.market("USDGOUSDT", 1792166400000)["timestamp"] == 1792166405000


def test_documented_second_precision_fill_times_are_normalized(mocker):
    t = mocker.Mock()
    t.call.return_value = [
        {
            "symbol": "TRXUSDT",
            "orderId": "order",
            "tradeId": "trade",
            "side": "buy",
            "size": "10",
            "amount": "3",
            "cTime": "1792166400",
            "feeDetail": {"feeCoin": "TRX", "totalFee": "0"},
        }
    ]
    assert (
        ClassicBroker(t).fills("order", "TRXUSDT", 1792166400000, 1792166401000)[0]["timestamp"]
        == 1792166400000
    )
