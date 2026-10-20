from copy import deepcopy
from datetime import datetime

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_ops import activity_summary


def stamp(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


NOW = stamp("2026-10-19T02:00:00+08:00")
DAY = stamp("2026-10-19T00:00:00+08:00")
START = stamp("2026-10-01T00:00:00+08:00")


def test_complete_history_calendar_boundaries_native_income_and_fee_deduplication():
    book = Ledger(10000)
    # More than the view's 20 fills and 30 recent income rows, including an
    # exact midnight fill and one millisecond before the seven-day boundary.
    for i in range(25):
        book.buy(f"buy-{i}", f"order-{i}", START + i, 1, 1, ".001")
    book.buy("midnight", "midnight-order", DAY, 1, 1, ".001")
    book.buy("before-week", "before-week", DAY - 6 * 86400000 - 1, 1, 1)
    book.sell("sold", "sold-order", DAY - 1, 1, 2, [{"lot_id": "order-0", "quantity": 1}], ".002")
    allocations = {}
    for i in range(45):
        key = f"income-allocation:{i}"
        book.credit_interest(key, "TRX", ".001")
        allocations[str(i)] = {
            "bill_id": str(i),
            "coin": "TRX",
            "amount": ".001",
            "timestamp": DAY - 1,
            "recorded_at": NOW,
        }
    book.credit_interest("income-allocation:usdgo", "USDGO", ".2", 1)
    allocations["usdgo"] = {
        "bill_id": "usdgo",
        "coin": "USDGO",
        "amount": ".2",
        "timestamp": DAY,
        "recorded_at": NOW,
    }
    book.credit_interest("income:native", "USDT", ".1")
    fx = {
        "id": "fx-fill",
        "order_id": "fx-order",
        "quantity": "100",
        "quote": "100",
        "timestamp": DAY - 1,
        "fees": [{"coin": "USDGO", "cost": ".1"}],
    }
    book.convert("spot:fx-order:fx-fill", "to_usdgo", 100, "99.9", 1)
    book.fees += money(".1")
    state = {
        "created_at": START,
        "book": book.state(),
        "income_accounting": {"allocations": allocations},
        "events": [
            {"kind": "operation_settled", "fills": [fx]},
            {
                "kind": "credited_interest_imported",
                "receipt_id": "income:native",
                "coin": "USDT",
                "amount": ".1",
                "timestamp": DAY,
            },
            {
                "kind": "unattributed_interest_observed",
                "receipt_id": "pending",
                "coin": "TRX",
                "amount": "999",
                "timestamp": DAY,
            },
            {
                "kind": "external_credit_quarantined",
                "coin": "USDT",
                "amount": "999",
                "timestamp": DAY,
            },
        ],
        "operation": {"fills": {"fx-fill": fx}},
    }
    before = deepcopy(state)
    result = activity_summary(state, NOW)
    assert state == before
    assert all(result["complete"].values())
    today, week, total = (result["periods"][p] for p in ("today", "week", "all"))
    assert today["start"] == DAY and week["start"] == DAY - 6 * 86400000
    assert today["buy_fills"] == 1 and today["sell_fills"] == 0
    assert week["buy_fills"] == 1 and week["sell_fills"] == 1
    assert total["buy_fills"] == 27
    assert money(total["turnover"]) == 29
    assert money(week["realized_profit"]) == book.realized
    assert money(today["realized_profit"]) == 0
    assert money(today["interest"]["TRX"]) == 0  # Late import stays on the original arrival day.
    assert money(week["interest"]["TRX"]) == money(".045")
    assert money(today["interest"]["USDGO"]) == money(".2")
    assert money(today["interest"]["USDT"]) == money(".1")
    assert money(total["conversion_fees"]) == money(".1")  # Repeated evidence isn't credited twice.
    assert money(total["gross_fees"]) == book.fees


def test_missing_evidence_is_unavailable_instead_of_a_partial_total():
    book = Ledger(1000)
    book.credit_interest("income-allocation:missing", "TRX", ".1")
    book.fees = money(".1")
    book.realized = money("2")
    state = {"created_at": DAY, "book": book.state(), "events": []}
    result = activity_summary(state, NOW)
    assert not any(result["complete"].values())
    for row in result["periods"].values():
        assert row["interest"]["TRX"] is None
        assert row["gross_fees"] is None
        assert row["realized_profit"] is None
        assert row["start"] == DAY  # Never imply coverage before this account existed.


def test_realized_profit_includes_verified_rebate_corrections_at_original_trade():
    book = Ledger(100)
    book.buy("buy", "b", START, 10, 1, ".1")
    book.sell("sell", "s", DAY - 1, 5, 2, [{"lot_id": "b", "quantity": 5}], ".1")
    book.queue_fee_rebate("buy", ".05", NOW)
    book.queue_fee_rebate("sell", ".05", NOW)
    book.settle_fee_rebates(NOW)
    result = activity_summary({"created_at": START, "book": book.state(), "events": []}, NOW)
    assert result["complete"]["profit"]
    assert money(result["periods"]["week"]["realized_profit"]) == book.realized
    assert money(result["periods"]["today"]["realized_profit"]) == 0
    assert money(result["periods"]["all"]["gross_fees"]) == money(".2")
