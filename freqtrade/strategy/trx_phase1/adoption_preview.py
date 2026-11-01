"""Pure projection of a cached account snapshot, never an allocation or acceptance.

This module does not construct a broker, execution ledger or strategy. The manifest
identifies the observed quantities and prices for human review; it is not a token
that can initialize or activate an account.
"""

import hashlib
import json
import re
import time
from copy import deepcopy
from decimal import Decimal, localcontext
from itertools import pairwise

from freqtrade.strategy.trx_phase1.ledger import money


COINS = ("TRX", "USDGO")
MAX_AGE_MS = 60000


class PreviewUnavailable(ValueError):
    pass


def amount(value):
    result = money(value)
    if result < 0:
        raise PreviewUnavailable("invalid_snapshot_values")
    return result


def text(value):
    return format(value, "f")


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise PreviewUnavailable("invalid_snapshot_values")
    return value


def spot_positions(snapshot, blockers):
    balances = {}
    for row in snapshot["balances"]:
        coin = row["coin"]
        if coin not in (*COINS, "USDT"):
            continue
        if coin in balances:
            raise PreviewUnavailable("duplicate_asset")
        values = {key: amount(row[key]) for key in ("available", "frozen", "locked")}
        quantity = amount(row["balance"])
        restricted = amount(row["restricted_available"])
        if sum(values.values(), Decimal(0)) != quantity or restricted > values["available"]:
            raise PreviewUnavailable("invalid_snapshot_values")
        balances[coin] = {**values, "quantity": quantity, "restricted": restricted}
        if coin in COINS and any(values[key] for key in ("frozen", "locked")):
            blockers.append("spot_assets_locked")
        if coin in COINS and restricted:
            blockers.append("spot_assets_restricted")
    if any(coin not in balances for coin in COINS):
        raise PreviewUnavailable("spot_balances_incomplete")
    return balances


def savings_positions(snapshot, blockers):
    savings = snapshot["savings"]
    if not savings["pages_complete"]:
        raise PreviewUnavailable("savings_pagination_incomplete")
    rows, seen = [], set()
    for row in savings["rows"]:
        if row["coin"] not in (*COINS, "USDT"):
            continue
        order = identifier(row["order_id"])
        if order in seen:
            raise PreviewUnavailable("duplicate_holding")
        seen.add(order)
        quantity = amount(row["quantity"])
        if quantity == 0:
            continue
        if row["coin"] in COINS and row["status"] != "in_holding":
            blockers.append("savings_status_requires_review")
        rows.append(
            {
                "coin": row["coin"],
                "product_id": identifier(row["product_id"]),
                "order_id": order,
                "quantity": quantity,
            }
        )
    return rows


def full_savings_verified(savings, proof):
    if not (
        savings.get("full_account_holdings_verified") is True
        and proof.get("windows_complete") is True
        and proof.get("basis") == "account_registration_to_snapshot"
        and set(COINS).issubset(proof.get("verified_coins", []))
        and proof.get("verification_method") == "all_age_pages_and_stable_earn_overview"
    ):
        return False
    windows = proof.get("windows", [])
    if not windows or windows[0]["until"] != savings["until"]:
        return False
    if windows[-1]["since"] != savings["since"] or proof["registered_at"] != savings["since"]:
        return False
    if not all(w["pages_complete"] and w["pages"] > 0 for w in windows):
        return False
    if any(a["since"] != b["until"] + 1 for a, b in pairwise(windows)):
        return False
    checks = proof.get("reconciliation", [])
    for coin in COINS:
        rows = [row for row in checks if row["coin"] == coin]
        if len(rows) != 1:
            return False
        row = rows[0]
        quantity = sum(
            (amount(r["quantity"]) for r in savings["rows"] if r["coin"] == coin), Decimal(0)
        )
        if not (
            row["matched"]
            and amount(row["holdings_quantity"])
            == quantity
            == amount(row["overview_before_quantity"])
            == amount(row["overview_after_quantity"])
            and money(row["difference_quantity"]) == 0
        ):
            return False
    return True


def build_manifest(snapshot, blockers):
    balances = spot_positions(snapshot, blockers)
    savings = savings_positions(snapshot, blockers)
    coverage = snapshot["savings"].get("coverage", {})
    verified = full_savings_verified(snapshot["savings"], coverage)
    if not verified:
        blockers.append("full_savings_verification_required")
    assets, locations, expiries = [], [], []
    observed, completed = snapshot["observed_at"], snapshot["completed_at"]
    for coin in COINS:
        quote = snapshot["prices"][coin]
        bid, ask = amount(quote["bid"]), amount(quote["ask"])
        stamp = quote["timestamp"]
        if bid <= 0 or ask < bid or not observed - MAX_AGE_MS <= stamp <= completed + 5000:
            raise PreviewUnavailable("invalid_valuation_quote")
        expiries.append(stamp + MAX_AGE_MS)
        spot = balances[coin]
        held = [row for row in savings if row["coin"] == coin]
        earn_quantity = sum((row["quantity"] for row in held), Decimal(0))
        quantity = spot["quantity"] + earn_quantity
        value = quantity * bid
        assets.append(
            {
                "coin": coin,
                "spot_quantity": text(spot["quantity"]),
                "savings_quantity": text(earn_quantity),
                "quantity": text(quantity),
                "adoption_unit_cost_usdt": text(bid),
                "quote_at": stamp,
                "market_value_usdt": text(value),
                "strategy_cost_usdt": text(value),
                "historical_cost_usdt": None,
            }
        )
        locations.append(
            {
                "coin": coin,
                "location": "spot",
                "quantity": text(spot["quantity"]),
                "available_quantity": text(spot["available"] - spot["restricted"]),
                "frozen_quantity": text(spot["frozen"]),
                "locked_quantity": text(spot["locked"]),
                "product_id": None,
                "order_id": None,
                "market_value_usdt": text(spot["quantity"] * bid),
            }
        )
        for row in sorted(held, key=lambda item: item["order_id"]):
            locations.append(
                {
                    "coin": coin,
                    "location": "flexible_savings",
                    "quantity": text(row["quantity"]),
                    "available_quantity": "0",
                    "frozen_quantity": "0",
                    "locked_quantity": text(row["quantity"]),
                    "product_id": row["product_id"],
                    "order_id": row["order_id"],
                    "market_value_usdt": text(row["quantity"] * bid),
                }
            )
    spot_usdt = balances.get("USDT", {}).get("quantity")
    savings_usdt = sum((row["quantity"] for row in savings if row["coin"] == "USDT"), Decimal(0))
    uid = snapshot["account"]["uid_masked"]
    if not isinstance(uid, str) or not re.fullmatch(r"\*{4}\d{1,4}", uid):
        raise PreviewUnavailable("invalid_snapshot_values")
    manifest = {
        "schema": "trx-opening-adoption-preview-v1",
        "account_type": "classic",
        "uid_masked": uid,
        "snapshot_observed_at": observed,
        "snapshot_completed_at": completed,
        "cost_basis": "adoption_market_value",
        "valuation_basis": "observed_bid_no_assumed_exit_fee",
        "included_coins": list(COINS),
        "assets": assets,
        "locations": locations,
        "excluded_assets": [
            {
                "coin": "USDT",
                "spot_quantity": text(spot_usdt) if spot_usdt is not None else None,
                "savings_quantity": text(savings_usdt),
                "quantity": text(spot_usdt + savings_usdt) if spot_usdt is not None else None,
            }
        ],
        "initial_equity_usdt": text(
            sum((money(a["market_value_usdt"]) for a in assets), Decimal(0))
        ),
        "strategy_cash_usdt": "0",
        "historical_cost_usdt": None,
        "hypothetical_opening_profits": {
            "principal_realized_usdt": "0",
            "principal_unrealized_usdt": "0",
            "earn_net_usdt": "0",
            "usdgo_fx_usdt": "0",
            "total_usdt": "0",
        },
        "coverage": {
            "since": snapshot["savings"]["since"],
            "until": snapshot["savings"]["until"],
            "pages_complete": True,
            "complete_account_history": False,
            "full_account_holdings_verified": verified,
            "verification": deepcopy(coverage) if coverage else None,
        },
    }
    return manifest, min(observed + MAX_AGE_MS, *expiries)


def check_open_orders(snapshot, checks, blockers):
    for name in ("orders", "conversion_orders"):
        value = snapshot.get(name)
        if checks.get(name, {}).get("status") != "success" or not value or value.get("limited"):
            blockers.append("open_orders_unverified")
        elif value["rows"]:
            blockers.append("open_orders_require_review")


def check_account(account, strategy_initialized, blockers):
    if strategy_initialized:
        blockers.append("strategy_already_initialized")
    if account.get("account_type") != "classic":
        blockers.append("classic_account_required")
    if account.get("error") or not account.get("credentials_configured"):
        blockers.append("account_connection_unverified")


def adoption_preview(account, *, strategy_initialized=False, now_ms=None):
    """GET-safe: consume existing normalized observations, with no filesystem writes."""
    now = int(time.time() * 1000) if now_ms is None else now_ms
    report = {
        "preview_only": True,
        "committed": False,
        "confirmation_available": False,
        "read_only": True,
        "trading_enabled": False,
        "earn_write_enabled": False,
        "startup_actions": [],
        "generated_at": now,
        "expires_at": None,
        "reviewable": False,
        "stale": False,
        "manifest_id": None,
        "manifest": None,
        "blockers": [],
        "warnings": [
            "savings_creation_window_only",
            "historical_cost_not_reconstructed",
            "pre_adoption_income_is_opening_capital",
            "cross_period_income_requires_review",
        ],
        "income_policy": {
            "performance_starts_on_formal_adoption": True,
            "past_product_total_profit_is_new_income": False,
            "future_trx_interest_retained_separately": True,
            "usdgo_fx_profit_separate_from_trx_trades": True,
            "earn_net_excludes_usdgo_fx": True,
            "fees_counted_once": True,
            "external_capital_flows_are_profit": False,
        },
    }
    blockers = report["blockers"]
    check_account(account, strategy_initialized, blockers)
    snapshot = account.get("snapshot")
    if not snapshot:
        blockers.append("account_snapshot_required")
        return report
    checks = account.get("checks", {})
    for name in ("balances", "savings", "prices"):
        if checks.get(name, {}).get("status") != "success":
            blockers.append(name + "_read_required")
    if any(name + "_read_required" in blockers for name in ("balances", "savings", "prices")):
        return report
    check_open_orders(snapshot, checks, blockers)
    try:
        observed, completed = snapshot["observed_at"], snapshot["completed_at"]
        if not isinstance(observed, int) or not isinstance(completed, int) or completed < observed:
            raise PreviewUnavailable("invalid_snapshot_values")
        with localcontext() as context:
            context.prec = 50
            manifest, expires = build_manifest(snapshot, blockers)
        if money(manifest["initial_equity_usdt"]) <= 0:
            blockers.append("no_assets_in_scope")
        report.update(manifest=manifest, expires_at=expires)
        report["manifest_id"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        report["stale"] = bool(account.get("stale")) or observed > now + 5000 or now > expires
        if report["stale"]:
            blockers.append("snapshot_expired")
    except PreviewUnavailable as exc:
        blockers.append(str(exc))
    except (KeyError, TypeError, ValueError, ArithmeticError):
        blockers.append("invalid_snapshot_values")
    report["blockers"] = list(dict.fromkeys(blockers))
    if report["manifest"] and report["manifest"]["coverage"]["full_account_holdings_verified"]:
        report["warnings"].remove("savings_creation_window_only")
        report["warnings"].append("holding_creation_dates_unverified")
    report["reviewable"] = not report["blockers"] and report["manifest"] is not None
    return report
