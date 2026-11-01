"""All-age current Savings holdings, reconciled against independent Earn balances.

Date ranges are query coverage, not proof of an order's creation date: Bitget can
return the same current order in several disjoint windows. Such rows are counted
once only. This verifies present holdings, never a complete transaction history.
"""

from decimal import Decimal, localcontext

from freqtrade.strategy.trx_phase1.account import (
    COINS,
    DAY_MS,
    AccountReadError,
    normalize_savings,
)
from freqtrade.strategy.trx_phase1.ledger import money


def overview(reader):
    rows = reader._read(reader.client.privateEarnGetV2EarnAccountAssets)
    if not isinstance(rows, list):
        raise AccountReadError("invalid_exchange_response")
    result = {}
    for row in rows:
        coin, quantity = row["coin"], money(row["amount"])
        if not isinstance(coin, str) or coin in result or quantity < 0:
            raise AccountReadError("invalid_exchange_response")
        result[coin] = quantity
    if any(coin not in result for coin in COINS):
        raise AccountReadError("savings_overview_incomplete")
    return {coin: result[coin] for coin in COINS}


def scan_window(reader, start, end):
    params = {
        "periodType": "flexible",
        "limit": "100",
        "startTime": str(start),
        "endTime": str(end),
    }
    result, seen, pages = [], set(), 0
    for rows in reader._pages(reader.client.privateEarnGetV2EarnSavingsAssets, params):
        pages += 1
        for raw in rows:
            row = normalize_savings(
                raw, include_historical_profit=not getattr(reader, "current_positions_only", False)
            )
            if row is None:
                continue
            if row["order_id"] in seen:
                raise AccountReadError("duplicate_record")
            if money(row["quantity"]) < 0:
                raise AccountReadError("invalid_exchange_response")
            seen.add(row["order_id"])
            result.append(row)
    return result, pages


def scan_all_windows(reader, registered):
    result, windows, repeated = {}, [], 0
    end = reader.now
    # Explicit hard bound; an unexpectedly old registration is not called complete.
    for _ in range(128):
        start = max(registered, end - 90 * DAY_MS + 1)
        rows, pages = scan_window(reader, start, end)
        added = 0
        for row in rows:
            key = row["order_id"]
            if key in result:
                if result[key] != row:
                    raise AccountReadError("savings_holdings_changed")
                repeated += 1
            else:
                result[key] = row
                added += 1
        windows.append(
            {
                "since": start,
                "until": end,
                "pages": pages,
                "returned_holdings": len(rows),
                "new_unique_holdings": added,
                "pages_complete": True,
            }
        )
        if start == registered:
            return list(result.values()), windows, repeated
        end = start - 1
    raise AccountReadError("savings_coverage_incomplete")


def full_savings_holdings(reader, registered):
    if (
        isinstance(registered, bool)
        or not isinstance(registered, int)
        or not 0 < registered <= reader.now
    ):
        raise AccountReadError("savings_registration_required")
    before = overview(reader)
    rows, windows, repeated = scan_all_windows(reader, registered)
    after = overview(reader)
    with localcontext() as context:
        context.prec = 50
        totals = {
            coin: sum((money(r["quantity"]) for r in rows if r["coin"] == coin), Decimal(0))
            for coin in COINS
        }
        checks = [
            {
                "coin": coin,
                "holdings_quantity": str(totals[coin]),
                "overview_before_quantity": str(before[coin]),
                "overview_after_quantity": str(after[coin]),
                "difference_quantity": str(totals[coin] - after[coin]),
                "matched": before[coin] == after[coin] == totals[coin],
            }
            for coin in COINS
        ]
    verified = all(row["matched"] for row in checks)
    return {
        "rows": sorted(rows, key=lambda r: (COINS.index(r["coin"]), r["order_id"])),
        "since": registered,
        "until": reader.now,
        "complete_account_history": False,
        "pages_complete": True,
        "source": "savings_v2",
        "account_location": "unverified",
        "partial": not verified,
        "full_account_holdings_verified": verified,
        "coverage": {
            "basis": "account_registration_to_snapshot",
            "registered_at": registered,
            "windows": windows,
            "windows_complete": True,
            "repeated_window_rows": repeated,
            "holding_creation_dates_verified": False,
            "verification_method": "all_age_pages_and_stable_earn_overview",
            "verified_coins": list(COINS) if verified else [],
            "reconciliation": checks,
        },
    }
