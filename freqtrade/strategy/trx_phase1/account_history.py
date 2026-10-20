"""Persist observed account chart points, separate from strategy ledgers.

Spot balances and the Savings creation-window observation are never added
together. Missing balances/prices remain unknown, including on failed reads.
"""

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from freqtrade.strategy.trx_phase1.ledger import money


COINS = ("USDT", "USDGO", "TRX")
DISPLAY_LIMIT = 10000


def spot_value(quantities, prices, timestamp):
    if any(value is None for value in quantities.values()):
        return None, "missing_balances"
    equity = money(quantities["USDT"])
    for coin in ("USDGO", "TRX"):
        quantity = money(quantities[coin])
        if quantity == 0:
            continue
        quote = (prices or {}).get(coin)
        if quote is None or not -5000 <= timestamp - quote["timestamp"] <= 60000:
            return None, "missing_prices"
        price = money(quote["bid"])
        if price <= 0:
            raise ValueError("Invalid valuation quote")
        equity += quantity * price
    return str(equity), "marked"


def observation(snapshot, account_type, timestamp):
    point = {
        "timestamp": timestamp,
        "completed_at": snapshot["completed_at"] if snapshot else timestamp,
        "spot_equity_usdt": None,
        "spot_quantities": dict.fromkeys(COINS),
        "savings_quantities": dict.fromkeys(COINS),
        "valuation_status": "read_failed" if snapshot is None else "missing_balances",
    }
    if snapshot is None:
        return point
    if account_type != "classic":
        point["valuation_status"] = "unsupported_account"
        return point
    for row in snapshot.get("balances") or []:
        value = money(row["total"])
        if value < 0:
            raise ValueError("Invalid observed balance")
        point["spot_quantities"][row["coin"]] = str(value)
    savings = snapshot.get("savings")
    if savings is not None and savings.get("pages_complete"):
        quantities = dict.fromkeys(COINS, money(0))
        for row in savings["rows"]:
            value = money(row["quantity"])
            if value < 0:
                raise ValueError("Invalid observed Savings balance")
            quantities[row["coin"]] += value
        # holdAmount is already the reported holding. Do not add totalProfit.
        point["savings_quantities"] = {coin: str(value) for coin, value in quantities.items()}
    point["spot_equity_usdt"], point["valuation_status"] = spot_value(
        point["spot_quantities"], snapshot.get("prices"), snapshot["completed_at"]
    )
    return point


class AccountHistory:
    def __init__(self, path):
        self.path = Path(path)

    def append(self, identity, account_type, point):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A private file exists before SQLite opens it; no key or raw UID is stored.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS observations "
                "(account TEXT, mode TEXT, timestamp INTEGER, payload TEXT, "
                "PRIMARY KEY (account, mode, timestamp))"
            )
            db.execute(
                "INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?)",
                (identity, account_type, point["timestamp"], json.dumps(point, allow_nan=False)),
            )

    def read(self, identity, account_type):
        empty = {"points": [], "total_points": 0, "truncated": False, "error": None}
        if identity is None or not self.path.exists():
            return empty
        with (
            closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as db,
            db,
        ):
            count = db.execute(
                "SELECT COUNT(*) FROM observations WHERE account=? AND mode=?",
                (identity, account_type),
            ).fetchone()[0]
            rows = db.execute(
                "SELECT payload FROM observations WHERE account=? AND mode=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (identity, account_type, DISPLAY_LIMIT),
            ).fetchall()
        return {
            "points": [json.loads(row[0]) for row in reversed(rows)],
            "total_points": count,
            "truncated": count > DISPLAY_LIMIT,
            "error": None,
        }
