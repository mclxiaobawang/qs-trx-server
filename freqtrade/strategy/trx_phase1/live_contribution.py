"""Reviewed in-kind capital transfers; never submit an exchange order.

The final commit changes assets available to the live strategy. It is therefore
an explicit operator action, separate from the read-only preview. Previous income
allocations and acquisition fills remain immutable.
"""

from copy import deepcopy

from freqtrade.strategy.trx_phase1.income_allocation import ownership
from freqtrade.strategy.trx_phase1.ledger import Ledger, Lot, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest


COINS = ("USDGO", "TRX")
REVIEW_LIFETIME = 5 * 60000


def candidate(execution, now, *, allow_cross_start_pending=False):
    if execution.broker.account_id() != execution.account_id:
        raise ExecutionError("account_binding_mismatch")
    state = execution.state
    if (
        state["error"]
        or state["operation"]
        or (state["flow"] and state["flow"]["status"] not in {"completed", "expired", "failed"})
    ):
        raise ExecutionError("contribution_requires_idle_execution")
    book, positions, receipts, records, accounting = execution.income_candidate(
        Ledger.restore(state["book"]), state["holdings"], state["receipts"], now
    )
    balances = execution.broker.balances()
    execution.check_wallet(balances, book, records)
    pending = execution.income_adjustments(records)[1]
    crossed = (accounting or {}).get("cross_start_receipts", {})
    excluded = (
        allow_cross_start_pending
        and execution.state.get("performance_basis")
        and all(
            key in crossed
            and crossed[key].get("reason") == "cross_start_accrual_pending"
            and crossed[key]["coin"] == row["coin"]
            and money(crossed[key]["amount"]) == money(row["amount"])
            for key, row in pending.items()
        )
    )
    if (pending and not excluded) or not accounting or accounting["status"] != "active":
        raise ExecutionError("contribution_requires_reconciled_income")
    remote = execution.broker.holdings(state["created_at"] - 1000, now)
    for key, row in positions.items():
        expected = sum(money(row[k]) for k in ("principal", "interest", "unassigned_amount"))
        observed = remote.get(key)
        if (not observed and expected) or (
            observed
            and (
                observed["coin"] != row["coin"]
                or observed["product_id"] != row["product_id"]
                or money(observed["amount"]) != expected
            )
        ):
            raise ExecutionError("contribution_holding_changed")
    execution.commit_candidate(book, positions, receipts, records, now, accounting)
    return book, balances


def quotes(execution, now):
    result = {coin: execution.broker.market(coin + "USDT", now) for coin in COINS}
    observed = max(now, execution.broker.clock()) if hasattr(execution.broker, "clock") else now
    if any(
        not -5000 <= observed - int(q["timestamp"]) <= 60000 or money(q["bid"]) <= 0
        for q in result.values()
    ):
        raise ExecutionError("contribution_quote_stale")
    return result


def preview(execution, now):
    with execution.exclusive():
        execution.reload()
        book, balances = candidate(execution, now)
        markets = quotes(execution, now)
        now = max(now, execution.broker.clock()) if hasattr(execution.broker, "clock") else now
        spot = execution.owned_spot(book)
        assets = []
        for coin in COINS:
            locations = []
            outside = money(execution.state["unassigned_offsets"][coin])
            if coin == "USDGO":
                from .live_reserve import protected_spot

                outside -= protected_spot(execution.state)
            if outside < 0 or money(balances[coin]["available"]) - spot[coin] < outside:
                raise ExecutionError("contribution_external_assets_locked")
            if outside:
                locations.append({"location": "spot", "amount": str(outside)})
            for key, row in execution.state["holdings"].items():
                available = money(row["unassigned_amount"]) - money(row.get("reserved_amount", 0))
                if row["coin"] == coin and available > 0:
                    locations.append(
                        {
                            "location": "earn",
                            "holding_id": key,
                            "product_id": row["product_id"],
                            "amount": str(available),
                        }
                    )
            amount = sum((money(row["amount"]) for row in locations), money(0))
            if amount:
                price = money(markets[coin]["bid"])
                assets.append(
                    {
                        "coin": coin,
                        "amount": str(amount),
                        "mark": str(price),
                        "value": str(amount * price),
                        "locations": locations,
                    }
                )
        if not assets:
            raise ExecutionError("no_external_assets_to_contribute")
        review = {
            "account_id": execution.account_id,
            "created_at": now,
            "expires_at": now + REVIEW_LIFETIME,
            "basis": "contribution_market_value",
            "assets": assets,
            "value": str(sum((money(a["value"]) for a in assets), money(0))),
        }
        review["review_hash"] = digest(review)
        execution.state["asset_contribution_review"] = review
        execution.save()
        return deepcopy(review)


def apply(execution, review_hash, now):  # noqa: C901 - one atomic reviewed contribution
    with execution.exclusive():
        execution.reload()
        completed = execution.state.get("asset_contributions", {})
        if review_hash in completed:
            return deepcopy(completed[review_hash])
        review = execution.state.get("asset_contribution_review")
        if (
            not review
            or review["review_hash"] != review_hash
            or review["account_id"] != execution.account_id
            or digest({k: v for k, v in review.items() if k != "review_hash"}) != review_hash
            or not review["created_at"] <= now < review["expires_at"]
        ):
            raise ExecutionError("contribution_review_expired")
        book, balances = candidate(execution, now)
        markets = quotes(execution, now)
        now = max(now, execution.broker.clock()) if hasattr(execution.broker, "clock") else now
        if now >= review["expires_at"]:
            raise ExecutionError("contribution_review_expired")
        if any(
            abs(money(markets[a["coin"]]["bid"]) / money(a["mark"]) - 1) > money(".01")
            for a in review["assets"]
        ):
            raise ExecutionError("contribution_price_changed")
        positions = execution.state["holdings"]
        offsets = execution.state["unassigned_offsets"]
        spot = execution.owned_spot(book)
        assets = deepcopy(review["assets"])
        for asset in assets:
            coin = asset["coin"]
            price = money(markets[coin]["bid"])
            asset.update(mark=str(price), value=str(money(asset["amount"]) * price))
            for index, row in enumerate(asset["locations"]):
                amount = money(row["amount"])
                holding = (
                    positions.get(row.get("holding_id")) if row["location"] == "earn" else None
                )
                if row["location"] == "spot":
                    from .live_reserve import protected_spot

                    reserve = protected_spot(execution.state) if coin == "USDGO" else money(0)
                    if (
                        amount > money(offsets[coin]) - reserve
                        or amount > money(balances[coin]["available"]) - spot[coin]
                    ):
                        raise ExecutionError("contribution_assets_changed")
                    offsets[coin] = str(money(offsets[coin]) - amount)
                else:
                    if (
                        not holding
                        or holding["coin"] != coin
                        or holding["product_id"] != row["product_id"]
                        or amount
                        > money(holding["unassigned_amount"])
                        - money(holding.get("reserved_amount", 0))
                    ):
                        raise ExecutionError("contribution_assets_changed")
                    holding["unassigned_amount"] = str(money(holding["unassigned_amount"]) - amount)
                    holding["principal"] = str(money(holding["principal"]) + amount)
                if coin == "USDGO":
                    book.usdgo += amount
                    book.usdgo_cost += amount * price
                    if holding:
                        book.locked["USDGO"] += amount
                else:
                    lot_id = "contribution-" + review_hash[:32] + "-" + str(index)
                    book.lots.append(
                        Lot(
                            lot_id,
                            now,
                            amount,
                            price,
                            amount if holding else money(0),
                            amount * price,
                        )
                    )
                    if holding:
                        holding["lots"][lot_id] = str(amount)
        # Issue fund units at the pre-transfer NAV. Added assets are capital, never profit.
        before = Ledger.restore(execution.state["book"])
        before.usdgo_mark = money(markets["USDGO"]["bid"])
        equity = before.equity(markets["TRX"]["bid"])
        value = sum((money(a["value"]) for a in assets), money(0))
        if equity <= 0 or before.fund_units <= 0:
            raise ExecutionError("contribution_requires_positive_nav")
        book.external_flows += value
        book.fund_units += value / (equity / before.fund_units)
        book.usdgo_mark = before.usdgo_mark
        book.events.add("capital-contribution:" + review_hash)
        accounting = execution.state["income_accounting"]
        history = accounting["checkpoints"]
        current = ownership(positions)
        if history[-1]["positions"] != current:
            if now <= history[-1]["timestamp"]:
                raise ExecutionError("contribution_checkpoint_conflict")
            transfers = {
                row["holding_id"]: row["amount"]
                for asset in assets
                for row in asset["locations"]
                if row["location"] == "earn"
            }
            history.append({"timestamp": now, "positions": current, "capital_transfers": transfers})
        record = {
            "review_hash": review_hash,
            "timestamp": now,
            "basis": review["basis"],
            "assets": assets,
            "value": str(value),
            "fund_units_before": str(before.fund_units),
            "fund_units_after": str(book.fund_units),
        }
        execution.state["book"] = book.state()
        execution.state.setdefault("asset_contributions", {})[review_hash] = record
        execution.state["asset_contribution_review"] = None
        execution.event("capital_contribution_recorded", now, contribution=record)
        execution.check_wallet(balances, book)
        execution.save()
        return deepcopy(record)
