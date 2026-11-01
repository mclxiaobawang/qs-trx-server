from copy import deepcopy
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from freqtrade.strategy.trx_phase1.account import (
    DAY_MS,
    AccountReader,
    AccountReadError,
    AccountSettings,
    ReadOnlyBitget,
    registration_time,
)


NOW = 1800000000000
REGISTERED = NOW - 200 * DAY_MS


def response(data):
    return {"code": "00000", "data": data}


def holding(coin, quantity, order):
    return {
        "productId": "p-" + coin,
        "orderId": order,
        "productCoin": coin,
        "interestCoin": coin,
        "periodType": "flexible",
        "holdAmount": quantity,
        "totalProfit": "9000",
        "status": "in_holding",
    }


def reader_fixture(rows=None):
    rows = rows or [
        holding("TRX", "100.12345678", "trx"),
        holding("USDGO", "1000.00000001", "usdgo"),
        holding("USDT", "2", "usdt"),
    ]
    client = MagicMock(spec=ReadOnlyBitget)
    totals = {
        coin: sum((Decimal(r["holdAmount"]) for r in rows if r["productCoin"] == coin), Decimal(0))
        for coin in ("TRX", "USDGO", "USDT")
    }
    overview = [{"coin": c, "amount": str(q)} for c, q in totals.items()]
    client.privateEarnGetV2EarnAccountAssets.return_value = response(overview)
    # Actual endpoint can repeat the same current orders in every dated window.
    client.privateEarnGetV2EarnSavingsAssets.side_effect = lambda p: response(
        {
            "resultList": [] if p.get("idLessThan") else deepcopy(rows),
            "endId": "" if p.get("idLessThan") else "last",
        }
    )
    return AccountReader(client, AccountSettings(), NOW, account_created_at=REGISTERED), client


def test_repeated_current_holdings_are_counted_once_and_reconciled_without_adding_profit():
    reader, client = reader_fixture()
    data = reader.savings()
    assert data["full_account_holdings_verified"] and not data["partial"]
    assert not data["complete_account_history"]
    assert len(data["rows"]) == 3
    assert data["rows"][2]["quantity"] == "100.12345678"
    proof = data["coverage"]
    assert proof["repeated_window_rows"] == 6
    assert not proof["holding_creation_dates_verified"]
    assert all(
        c["matched"] and Decimal(c["difference_quantity"]) == 0 for c in proof["reconciliation"]
    )
    windows = proof["windows"]
    assert len(windows) == 3 and windows[0]["until"] == NOW and windows[-1]["since"] == REGISTERED
    for window in windows:
        assert window["pages"] == 2 and window["pages_complete"]
        assert 0 <= window["until"] - window["since"] < 90 * DAY_MS
    assert windows[1]["until"] == windows[0]["since"] - 1
    assert windows[2]["until"] == windows[1]["since"] - 1
    assert client.privateEarnGetV2EarnAccountAssets.call_count == 2


def test_distinct_holding_created_more_than_90_days_ago_is_included():
    rows = [
        holding("TRX", "100", "recent"),
        holding("TRX", "10.00000001", "old"),
        holding("USDGO", "1000", "usdgo"),
        holding("USDT", "2", "usdt"),
    ]
    reader, client = reader_fixture(rows)

    def pages(params):
        oldest = int(params["startTime"]) == REGISTERED
        found = [r for r in rows if (r["orderId"] == "old") == oldest]
        # Middle empty window must not stop the scan of older windows.
        if int(params["endTime"]) < NOW and not oldest:
            found = []
        return response({"resultList": found, "endId": ""})

    client.privateEarnGetV2EarnSavingsAssets.side_effect = pages
    result = reader.savings()
    assert result["full_account_holdings_verified"]
    assert {r["order_id"] for r in result["rows"]} == {"recent", "old", "usdgo", "usdt"}
    trx = next(c for c in result["coverage"]["reconciliation"] if c["coin"] == "TRX")
    assert trx["holdings_quantity"] == "110.00000001"
    assert result["coverage"]["windows"][1]["returned_holdings"] == 0


@pytest.mark.parametrize("change", ["missing_old_position", "balance_changed_during_scan"])
def test_incomplete_or_moving_balances_cannot_claim_current_holdings_complete(change):
    reader, client = reader_fixture()
    before = client.privateEarnGetV2EarnAccountAssets.return_value
    after = deepcopy(before)
    after["data"][0]["amount"] = "101.12345678"
    if change == "missing_old_position":
        before = deepcopy(after)
    client.privateEarnGetV2EarnAccountAssets.side_effect = [before, after]
    data = reader.savings()
    assert data["partial"] and not data["full_account_holdings_verified"]
    assert data["coverage"]["verified_coins"] == []
    assert any(not c["matched"] for c in data["coverage"]["reconciliation"])


@pytest.mark.parametrize(
    "change,code",
    [
        ("missing_coin", "savings_overview_incomplete"),
        ("negative", "invalid_exchange_response"),
        ("duplicate_coin", "invalid_exchange_response"),
        ("invalid_number", "finite"),
    ],
)
def test_overview_is_independent_and_missing_or_invalid_amounts_are_not_invented(change, code):
    reader, client = reader_fixture()
    rows = client.privateEarnGetV2EarnAccountAssets.return_value["data"]
    if change == "missing_coin":
        rows.pop()
    elif change == "duplicate_coin":
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]["amount"] = "-1" if change == "negative" else "NaN"
    with pytest.raises((AccountReadError, ValueError), match=code):
        reader.savings()


def test_conflicting_cross_window_order_is_not_silently_deduplicated():
    reader, client = reader_fixture()

    def pages(params):
        quantity = "100" if int(params["endTime"]) == NOW else "101"
        return response({"resultList": [holding("TRX", quantity, "same-order")], "endId": ""})

    client.privateEarnGetV2EarnSavingsAssets.side_effect = pages
    with pytest.raises(AccountReadError, match="savings_holdings_changed"):
        reader.savings()


def test_failed_older_window_or_incomplete_pagination_cannot_produce_a_full_total():
    reader, client = reader_fixture()
    client.privateEarnGetV2EarnSavingsAssets.return_value = response({})

    def pages(params):
        if int(params["endTime"]) < NOW:
            raise AccountReadError("network_error")
        return response({"resultList": [], "endId": ""})

    client.privateEarnGetV2EarnSavingsAssets.side_effect = pages
    with pytest.raises(AccountReadError, match="network_error"):
        reader.savings()
    client.privateEarnGetV2EarnSavingsAssets.side_effect = lambda p: response(
        {"resultList": [holding("TRX", "1", str(i)) for i in range(100)], "endId": ""}
    )
    with pytest.raises(AccountReadError, match="pagination_incomplete"):
        reader.savings()


@pytest.mark.parametrize("value", [None, "", "bad", True, "0", "1800000000001", -1])
def test_registration_lower_bound_requires_a_valid_exchange_timestamp(value):
    assert registration_time(value, NOW) is None
