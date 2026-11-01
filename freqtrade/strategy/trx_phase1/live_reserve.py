"""One-time in-kind USDGO reserve: no exchange order or savings request."""

from copy import deepcopy
from decimal import ROUND_DOWN, localcontext

from .income_allocation import ownership
from .ledger import Ledger, money


UNIT = money("0.00000001")


def protected_spot(state):
    return sum(
        (
            money(r.get("reserve_amount", 0))
            for r in state.get("income_accounting", {}).get("allocations", {}).values()
        ),
        money(0),
    )


def summary(state):
    record = state.get("usdgo_reserve")
    if not record:
        return None
    rows = list(state.get("income_accounting", {}).get("allocations", {}).values())
    income = [r for r in rows if money(r.get("reserve_amount", 0)) > 0]
    return {
        "allocation_id": record["allocation_id"],
        "allocated_at": record["allocated_at"],
        "allocated_quantity": record["quantity"],
        "allocated_value_usdt": record["value_usdt"],
        "earn_quantity": str(
            sum((money(p.get("reserved_amount", 0)) for p in state["holdings"].values()), money(0))
        ),
        "credited_interest": str(protected_spot(state)),
        "interest_receipts": len(income),
        "latest_interest_at": max((r["timestamp"] for r in income), default=None),
        "strategy_usdgo": state["book"]["usdgo"],
        "recent_income": [
            {
                k: r[k]
                for k in (
                    "bill_id",
                    "timestamp",
                    "account_amount",
                    "reserve_amount",
                    "period_start",
                    "period_end",
                )
            }
            for r in sorted(income, key=lambda r: r["timestamp"])[-20:]
        ],
        "exchange_actions": [],
        "record": deepcopy(record),
    }


def allocate_half(execution, strategy, allocation_id, now):  # noqa: C901 - one atomic capital event
    """Called under the live runtime lock; commit the owned/external split once."""
    from .live_broker import ExecutionError
    from .live_contribution import candidate, quotes

    if not isinstance(allocation_id, str) or not 8 <= len(allocation_id) <= 96:
        raise ExecutionError("reserve_invalid_allocation_id")
    with execution.exclusive():
        execution.reload()
        if execution.state is None:
            raise ExecutionError("existing_live_account_required")
        existing = execution.state.get("usdgo_reserve")
        if existing:
            if existing["allocation_id"] != allocation_id:
                raise ExecutionError("reserve_already_allocated")
            return deepcopy(existing)
        execution.require_trading_authorized()
        if (
            not strategy["startup_complete"]
            or strategy["desired"] != "running"
            or strategy["phase"] != "observing"
            or strategy.get("error")
            or strategy.get("intent")
            or strategy["engine"].get("pending")
            or any(
                r["status"] == "confirmed"
                for r in execution.state.get("asset_withdrawals", {}).values()
            )
        ):
            raise ExecutionError("reserve_requires_idle_strategy")
        old_checkpoints = len(execution.state.get("income_accounting", {}).get("checkpoints", []))
        before, balances = candidate(execution, now, allow_cross_start_pending=True)
        markets = quotes(execution, now)
        now = max(now, execution.broker.clock()) if hasattr(execution.broker, "clock") else now
        quantity = (before.usdgo / 2).quantize(UNIT, rounding=ROUND_DOWN)
        if quantity <= 0 or quantity > before.locked["USDGO"]:
            raise ExecutionError("reserve_owned_earn_insufficient")
        positions = deepcopy(execution.state["holdings"])
        remaining, locations = quantity, []
        for key, position in sorted(positions.items()):
            if position["coin"] != "USDGO" or remaining <= 0:
                continue
            part = min(remaining, money(position["principal"]))
            if part <= 0:
                continue
            account_quantity = sum(
                (money(position[k]) for k in ("principal", "interest", "unassigned_amount")),
                money(0),
            )
            position["principal"] = str(money(position["principal"]) - part)
            position["unassigned_amount"] = str(money(position["unassigned_amount"]) + part)
            position["reserved_amount"] = str(money(position.get("reserved_amount", 0)) + part)
            locations.append(
                {
                    "holding_id": key,
                    "product_id": position["product_id"],
                    "quantity": str(part),
                    "account_holding_quantity": str(account_quantity),
                }
            )
            remaining -= part
        if remaining:
            raise ExecutionError("reserve_owned_earn_insufficient")
        book = Ledger.restore(before.state())
        fx, price = money(markets["USDGO"]["bid"]), money(markets["TRX"]["bid"])
        with localcontext() as context:
            context.prec = 50
            before.usdgo_mark = fx
            nav = before.equity(price) / before.fund_units
            profit = before.equity(price) - before.initial - before.external_flows
            # Reclassify the owned locked amount; its physical location is unchanged.
            book.locked["USDGO"] -= quantity
            book.withdraw_usdgo("capital-reserve:" + allocation_id, quantity, price, fx)
            if (
                abs(book.equity(price) / book.fund_units - nav) > money("1e-35")
                or abs(book.equity(price) - book.initial - book.external_flows - profit)
                > money("1e-30")
                or execution.owned_spot(book) != execution.owned_spot(before)
            ):
                raise ExecutionError("reserve_conservation_failed")
        accounting = deepcopy(execution.state["income_accounting"])
        history = accounting["checkpoints"]
        checkpoint = {
            "timestamp": now,
            "positions": ownership(positions),
            "capital_transfers": {r["holding_id"]: str(-money(r["quantity"])) for r in locations},
            "reserve_transfers": {r["holding_id"]: r["quantity"] for r in locations},
        }
        if history[-1]["timestamp"] < now:
            history.append(checkpoint)
        elif history[-1]["timestamp"] == now and len(history) > old_checkpoints:
            history[-1] = checkpoint  # Only an unsaved checkpoint made by candidate().
        else:
            raise ExecutionError("reserve_checkpoint_conflict")
        record = {
            "schema": "trx-usdgo-earned-reserve-v1",
            "allocation_id": allocation_id,
            "account_id": execution.account_id,
            "allocated_at": now,
            "fraction": "0.5",
            "source_usdgo_quantity": str(before.usdgo),
            "quantity": str(quantity),
            "mark_usdt": str(fx),
            "value_usdt": str(quantity * fx),
            "locations": locations,
            "fund_units_before": str(before.fund_units),
            "fund_units_after": str(book.fund_units),
            "capital_base_before": str(before.deployable),
            "capital_base_after": str(book.deployable),
            "exchange_actions": [],
        }
        execution.state.update(
            book=book.state(),
            holdings=positions,
            income_accounting=accounting,
            usdgo_reserve=record,
        )
        execution.check_wallet(balances, book)
        execution.event("usdgo_reserve_allocated", now, reserve=record)
        execution.save()
        return deepcopy(record)
