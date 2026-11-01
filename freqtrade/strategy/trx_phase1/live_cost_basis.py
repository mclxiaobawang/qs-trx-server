"""Display-only TRX cost reconstruction; all financial sources are opened read-only.

Interest is valued at the last completed minute close at its actual receipt time.
This valuation never enters the ledger, profit attribution, or trading decisions.
"""

import bisect
import hashlib
import json
import sqlite3
import threading
import time
from contextlib import closing
from decimal import Decimal
from pathlib import Path


ZERO = Decimal(0)
TOLERANCE = Decimal("0.0000001")
MINUTE = 60000
SAMPLE_BATCH_SIZE = 256


def decimal(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("invalid_cost_source")
    return result


def connection(source):
    db = sqlite3.connect(
        (Path(source) / "strategy.sqlite").resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2
    )
    db.execute("PRAGMA query_only=ON")
    return db


def read_rows(source, sql, parameters=()):
    """Finish each bounded SELECT before releasing any data to Python processing.

    A mode=ro connection still blocks commits on DELETE journals while a SELECT
    cursor remains active. fetchmany on one long-lived cursor does not release it.
    """
    for attempt in range(3):
        try:
            with closing(connection(source)) as db, closing(db.execute(sql, parameters)) as cursor:
                rows = cursor.fetchall()
            return rows
        except sqlite3.OperationalError as exc:
            if attempt == 2 or not any(word in str(exc).lower() for word in ("locked", "busy")):
                raise
            time.sleep(0.05 * (attempt + 1))


def receipt_close(source, stamp):
    rows = read_rows(
        source,
        "SELECT timestamp,close FROM live_candles WHERE timestamp<=? "
        "ORDER BY timestamp DESC LIMIT 1",
        (stamp - MINUTE,),
    )
    return rows[0] if rows else None


def sample_rows(source, begin, end):
    """Keyset pagination over a fixed watermark; no live cursor crosses a yield."""
    after = begin - 1
    while after < end:
        rows = read_rows(
            source,
            "SELECT timestamp,payload FROM live_samples WHERE timestamp>? AND timestamp<=? "
            "ORDER BY timestamp LIMIT ?",
            (after, end, SAMPLE_BATCH_SIZE),
        )
        if not rows:
            return
        after = rows[-1][0]
        yield from rows


class InterestBasis:
    """Weighted average release for actual interest exits; no assumed exits."""

    def __init__(self, source, execution):  # noqa: C901 - conservative evidence reconstruction
        events = set(execution["book"].get("events", []))
        receipts = execution.get("receipts", {})
        credited_rows = {}
        for row in execution.get("income_accounting", {}).get("allocations", {}).values():
            if row.get("coin") != "TRX" or row.get("attribution") != "strategy_allocation":
                continue
            amount = decimal(row["amount"])
            key = "income-allocation:" + str(row["bill_id"])
            if not amount or key not in events or key not in receipts:
                continue
            identity = str(row["bill_id"])
            if identity in credited_rows and credited_rows[identity] != row:
                raise ValueError("conflicting_interest_receipt")
            credited_rows[identity] = row
        self.receipt_count = len(credited_rows)
        self.missing_prices = 0
        changes = []
        for row in credited_rows.values():
            stamp = int(row["timestamp"])
            candle = receipt_close(source, stamp)
            price = None
            if candle and 0 <= stamp - (candle[0] + MINUTE) < MINUTE:
                price = decimal(candle[1]) or None
            if price is None:
                self.missing_prices += 1
            # A delayed allocation only enters historical held quantity when it was booked.
            changes.append(
                (
                    max(stamp, int(row["recorded_at"])),
                    0,
                    str(row["bill_id"]),
                    decimal(row["amount"]),
                    price,
                )
            )
        seen_exits = set()
        for row in execution["book"].get("orders", []):
            if row.get("kind") != "manual_interest_exit":
                continue
            identity = row.get("id", row.get("order_id"))
            if not identity or identity in seen_exits:
                raise ValueError("conflicting_interest_exit")
            seen_exits.add(identity)
            changes.append(
                (int(row["timestamp"]), 1, str(identity), decimal(row["quantity"]), None)
            )
        quantity = basis = unknown = ZERO
        self.times, self.states = [], []
        valid = True
        for stamp, exit_kind, _, amount, price in sorted(changes):
            if exit_kind:
                if amount > quantity + TOLERANCE or not quantity:
                    valid = False
                else:
                    fraction = max(ZERO, (quantity - amount) / quantity)
                    basis *= fraction
                    unknown *= fraction
                quantity = max(ZERO, quantity - amount)
            else:
                quantity += amount
                if price is None:
                    unknown += amount
                else:
                    basis += amount * price
            self.times.append(stamp)
            self.states.append((quantity, basis, unknown, valid))

    def at(self, stamp, principal, interest, cost, price=None):
        index = bisect.bisect_right(self.times, stamp) - 1
        quantity, basis, unknown, valid = (
            self.states[index] if index >= 0 else (ZERO, ZERO, ZERO, True)
        )
        principal, interest, cost = map(decimal, (principal, interest, cost))
        matched = abs(quantity - interest) <= TOLERANCE and valid
        complete = matched and unknown <= TOLERANCE
        total = principal + interest
        average = (cost + basis) / total if complete and total else None
        return {
            "timestamp": stamp,
            "price": float(price) if price is not None and decimal(price) > 0 else None,
            "average_cost": float(average) if average is not None else None,
            "principal_average_cost": float(cost / principal) if principal else None,
            "principal_quantity": str(principal),
            "interest_quantity": str(interest),
            "principal_cost": str(cost),
            "interest_basis": str(basis) if complete else None,
            "total_basis": str(cost + basis) if complete else None,
            "complete": complete,
            "unmatched_interest_quantity": str(abs(quantity - interest)),
            "unpriced_interest_quantity": str(unknown),
        }


def sampled(rows, begin, end, limit):
    """Bounded memory; retain actual first/last and extrema from each time bucket."""
    width = max(1, (end - begin + 1) / max(1, limit // 6))
    bucket, candidates = None, []
    for row in rows:
        group = int((row["timestamp"] - begin) / width)
        if bucket is not None and group != bucket:
            yield from {r["timestamp"]: r for r in candidates}.values()
            candidates = []
        bucket = group
        if not candidates:
            candidates = [row] * 6
        candidates[1] = row
        for field, lo, hi in [("price", 2, 3), ("average_cost", 4, 5)]:
            if row[field] is not None:
                if candidates[lo][field] is None or row[field] < candidates[lo][field]:
                    candidates[lo] = row
                if candidates[hi][field] is None or row[field] > candidates[hi][field]:
                    candidates[hi] = row
        # Missing evidence / observation gaps must survive downsampling.
        if row.get("break_before") or row["average_cost"] is None:
            candidates[0] = row
    yield from {r["timestamp"]: r for r in candidates}.values()


class CostBasisHistory:
    def __init__(self, source, account_id):
        self.source = Path(source)
        self.account_id = account_id
        self.lock = threading.Lock()
        self.cache = None

    def query(self, strategy, execution, range_name="all", limit=1800):
        if any(s.get("account_id") != self.account_id for s in (strategy, execution)):
            raise ValueError("cost_basis_account_mismatch")
        if range_name not in {"summary", "day", "week", "all"} or not 60 <= limit <= 2400:
            raise ValueError("invalid_cost_basis_range")
        book = execution["book"]
        quote = strategy.get("markets", {}).get("TRXUSDT", {})
        identity = hashlib.sha256(
            json.dumps(
                [
                    book,
                    execution.get("income_accounting"),
                    execution.get("receipts"),
                    quote,
                    range_name,
                    limit,
                ],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with self.lock:
            if self.cache and self.cache[0] == identity and time.monotonic() < self.cache[1]:
                return self.cache[2]
            basis = InterestBasis(self.source, execution)
            now = int(time.time() * 1000)
            principal = sum((decimal(lot["quantity"]) for lot in book["lots"]), ZERO)
            cost = sum((decimal(lot["remaining_cost"]) for lot in book["lots"]), ZERO)
            current = basis.at(now, principal, book["interest_trx"], cost, quote.get("bid"))
            current["price_at"] = quote.get("timestamp")
            # Separate indexed endpoint reads avoid a full aggregate scan.
            last = read_rows(
                self.source, "SELECT timestamp FROM live_samples ORDER BY timestamp DESC LIMIT 1"
            )
            end = last[0][0] if last else None
            first = read_rows(
                self.source, "SELECT timestamp FROM live_samples ORDER BY timestamp LIMIT 1"
            )
            start = first[0][0] if first else None
            duration = {"day": 86400000, "week": 7 * 86400000}.get(range_name)
            begin = max(start, end - duration) if start and duration else start
            source_points, incomplete_points = 0, 0

            def points():
                nonlocal source_points, incomplete_points
                previous = None
                for _, payload in sample_rows(self.source, begin, end):
                    sample = json.loads(payload)
                    point = basis.at(
                        sample["timestamp"],
                        sample["principal_trx"],
                        sample["interest_trx"],
                        sample["remaining_cost"],
                        sample.get("price"),
                    )
                    if previous is not None and point["timestamp"] - previous > 3 * MINUTE:
                        point["break_before"] = True
                    previous = point["timestamp"]
                    source_points += 1
                    incomplete_points += not point["complete"]
                    yield point

            curve = (
                []
                if range_name == "summary" or not begin
                else sorted(
                    sampled(points(), begin, end, limit),
                    key=lambda p: p["timestamp"],
                )
            )
            result = {
                "account_id": self.account_id,
                "display_only": True,
                "reconstructed": True,
                "current": current,
                "points": [
                    {
                        k: p[k]
                        for k in (
                            "timestamp",
                            "price",
                            "average_cost",
                            "complete",
                            "break_before",
                        )
                        if k in p
                    }
                    for p in curve
                ],
                "start": begin,
                "end": end,
                "available_start": start,
                "live_started_at": (execution.get("performance_basis") or {}).get("started_at")
                or strategy.get("activated_at")
                or execution["created_at"],
                "source_points": source_points,
                "incomplete_points": incomplete_points,
                "interest_receipts": basis.receipt_count,
                "missing_receipt_prices": basis.missing_prices,
                "basis": "remaining_principal_cost_plus_receipt_close_valued_retained_interest",
                "receipt_price_basis": "last_completed_1m_close_before_receipt_max_age_60s",
                "interest_exit_basis": "weighted_average_release",
                "sampling": "actual_sample_bucket_first_last_price_and_cost_extrema",
            }
            self.cache = (identity, time.monotonic() + 20, result)
            return result
