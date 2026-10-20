"""Approved internal allocation of native account receipts, never an ID mapping.

Only actual, wallet-reconciled BATCH_INTEREST_USER_IN receipts are allocated.
The prior complete hour is an explicit accounting convention. Effective units are
units held throughout that hour; subscriptions enter next hour and redemptions
reduce the completed-hour minimum. Rounding residue stays outside the strategy.
"""

import json
from copy import deepcopy
from decimal import ROUND_DOWN

from .ledger import money


VERSION = "hourly-effective-principal-v1"
HOUR = 3600000
UNIT = money("0.00000001")


def ownership(positions):
    return {
        key: {
            "coin": p["coin"],
            "product_id": p["product_id"],
            "strategy": str(money(p["principal"]) + money(p["interest"])),
            "external": str(money(p["unassigned_amount"])),
        }
        for key, p in positions.items()
    }


def accounting_for(execution, positions, now):  # noqa: C901 - validate one bound policy atomically
    from .live_execution import digest

    previous = execution.state.get("income_accounting")
    path = execution.root.parents[1] / "income-allocation-policy.json"
    if not path.exists():
        if previous:
            return {**deepcopy(previous), "status": "policy_file_missing"}
        return None
    accounting = deepcopy(previous or {"allocations": {}})
    try:
        if path.stat().st_size > 1048576:
            raise ValueError
        policy = json.loads(path.read_text())
        fingerprint = digest(policy)
        if (
            policy["version"] != VERSION
            or policy["account_id"] != execution.account_id
            or not policy["approved_at"]
            or policy["products"] != execution.state["products"]
            or (previous and previous["policy_sha256"] != fingerprint)
        ):
            raise ValueError
        if not previous:
            history = deepcopy(policy["checkpoints"])
            stamps = [p["timestamp"] for p in history]
            if not stamps or stamps != sorted(set(stamps)) or stamps[-1] > now:
                raise ValueError
            for point in history:
                for p in point["positions"].values():
                    if min(money(p["strategy"]), money(p["external"])) < 0:
                        raise ValueError
            # Backfill requires the independently reconstructed final ownership
            # to match the authoritative pre-update execution ledger exactly.
            if history[-1]["positions"] != ownership(execution.state["holdings"]):
                raise ValueError
            if any(money(p["strategy"]) for p in history[0]["positions"].values()):
                raise ValueError
            accounting.update(
                version=VERSION,
                policy_sha256=fingerprint,
                approved_at=policy["approved_at"],
                checkpoints=history,
            )
        history = accounting["checkpoints"]
        current = ownership(positions)
        if history[-1]["positions"] != current:
            operation = execution.state.get("operation") or {}
            stamp = operation.get("created_at", now) if operation.get("kind") == "redeem" else now
            if stamp <= history[-1]["timestamp"]:
                raise ValueError
            history.append({"timestamp": stamp, "positions": current})
        accounting.update(status="active", checked_at=now)
    except (OSError, KeyError, TypeError, ValueError):
        accounting["status"] = "policy_validation_required"
    return accounting


def effective_units(history, coin, start, end):
    before = [p for p in history if p["timestamp"] <= start]
    first = before[-1] if before else history[0]
    points = [first, *(p for p in history if start < p["timestamp"] < end)]
    # In-kind contributions change ownership, not the account's held principal.
    # Under the existing complete-hour convention, transfer ownership becomes
    # effective for the next whole accrual hour. Preserve the external weight in
    # the transition hour instead of losing it to two independent minima.
    transfers, adjusted = {}, []
    for point in points:
        if point["timestamp"] > start:
            for key, amount in point.get("capital_transfers", {}).items():
                transfers[key] = transfers.get(key, money(0)) + money(amount)
        positions = deepcopy(point["positions"])
        for key, amount in transfers.items():
            if key in positions:
                row = positions[key]
                row["strategy"] = str(max(money(0), money(row["strategy"]) - amount))
                row["external"] = str(money(row["external"]) + amount)
        adjusted.append({**point, "positions": positions})
    points = adjusted
    keys = {k for p in points for k, v in p["positions"].items() if v["coin"] == coin}
    totals = {"strategy": money(0), "external": money(0)}
    products = set()
    for key in keys:
        for point in points:
            row = point["positions"].get(key)
            if row:
                products.add(row["product_id"])
        for side in totals:
            totals[side] += min(
                money(point["positions"].get(key, {}).get(side, "0")) for point in points
            )
    return totals, products


def allocate(execution, book, positions, receipts, records, remote, now):  # noqa: C901 - atomic receipt allocation
    from .live_execution import digest

    accounting = accounting_for(execution, positions, now)
    if not accounting or accounting["status"] != "active":
        return book, positions, receipts, records, accounting
    _, pending = execution.income_adjustments(records)
    if not pending:
        return book, positions, receipts, records, accounting
    expected = ownership(positions)
    # Do not adopt unknown manual deposits, other products or changed locations.
    for key, p in expected.items():
        actual = remote.get(key)
        if not actual and money(p["strategy"]) + money(p["external"]) == 0:
            continue
        if (
            not actual
            or actual["coin"] != p["coin"]
            or actual["product_id"] != p["product_id"]
            or money(actual["amount"]) != money(p["strategy"]) + money(p["external"])
        ):
            accounting["status"] = "holding_evidence_required"
            return book, positions, receipts, records, accounting
    history = accounting["checkpoints"]
    buckets = {}
    for credit in [*pending.values(), *accounting["allocations"].values()]:
        bucket = (credit["coin"], credit["timestamp"] // HOUR)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    fx_quote = None
    for pending_key, credit in sorted(pending.items(), key=lambda item: item[1]["timestamp"]):
        if (
            credit.get("type") != "BATCH_INTEREST_USER_IN"
            or credit.get("destination") != "spot"
            or credit.get("group") != "financial"
        ):
            continue
        coin, stamp = credit["coin"], credit["timestamp"]
        end = stamp // HOUR * HOUR
        weights, products = effective_units(history, coin, end - HOUR, end)
        observed_products = {p["product_id"] for p in remote.values() if p["coin"] == coin}
        if (
            len(products) != 1
            or observed_products != products
            or buckets[(coin, stamp // HOUR)] != 1
        ):
            accounting["status"] = "receipt_scope_required"
            continue
        if pending_key in accounting["allocations"]:
            accounting["status"] = "allocation_conflict"
            continue
        total = sum(weights.values(), money(0))
        if total <= 0:
            accounting["status"] = "eligible_principal_required"
            continue
        amount = money(credit["amount"])
        strategy = (amount * weights["strategy"] / total).quantize(UNIT, rounding=ROUND_DOWN)
        key = "income-allocation:" + credit["bill_id"]
        if key in receipts or key in book.events:
            accounting["status"] = "allocation_conflict"
            continue
        if coin == "USDGO" and strategy and fx_quote is None:
            fx_quote = execution.broker.market("USDGOUSDT", now)
        fx = money(fx_quote["bid"]) if coin == "USDGO" and strategy else None
        allocation = {
            "coin": coin,
            "bill_id": credit["bill_id"],
            "timestamp": stamp,
            "recorded_at": now,
            "amount": str(strategy),
            "account_amount": str(amount),
            "external_amount": str(amount - strategy),
            "strategy_units": str(weights["strategy"]),
            "external_units": str(weights["external"]),
            "period_start": end - HOUR,
            "period_end": end,
            "method": VERSION,
            "policy_sha256": accounting["policy_sha256"],
            "destination": "spot",
            "attribution": "strategy_allocation",
            "offset_delta": str(-strategy),
            "resolves_pending": pending_key,
            "usdgo_mark": str(fx) if fx is not None else None,
            "valuation_at": now,
        }
        if strategy:
            book.credit_interest(key, coin, strategy, fx)
        receipts[key] = digest(allocation)
        accounting["allocations"][pending_key] = allocation
        records.append((key, allocation))
    return book, positions, receipts, records, accounting


def summary(accounting):
    if accounting is None:
        return None
    totals = {}
    allocations = accounting.get("allocations", {})
    for row in allocations.values():
        total = totals.setdefault(
            row["coin"], {k: money(0) for k in ("account", "strategy", "external")}
        )
        for name, field in (
            ("account", "account_amount"),
            ("strategy", "amount"),
            ("external", "external_amount"),
        ):
            total[name] += money(row[field])
    return {
        "version": accounting.get("version"),
        "status": accounting["status"],
        "approved_at": accounting.get("approved_at"),
        "allocated_receipts": len(allocations),
        "totals": {
            coin: {k: str(v) for k, v in amounts.items()} for coin, amounts in totals.items()
        },
        "recent_allocations": sorted(allocations.values(), key=lambda r: r["timestamp"])[-30:],
    }
