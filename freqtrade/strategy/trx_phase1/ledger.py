import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path


def money(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Non-finite money")
    return result


@dataclass
class Lot:
    lot_id: str
    bought_at: int
    quantity: Decimal
    unit_cost: Decimal
    locked: Decimal = Decimal(0)
    remaining_cost: Decimal = Decimal(0)

    def state(self):
        return {
            "lot_id": self.lot_id,
            "bought_at": self.bought_at,
            "quantity": str(self.quantity),
            "unit_cost": str(self.unit_cost),
            "locked": str(self.locked),
            "remaining_cost": str(self.remaining_cost),
        }


class Ledger:
    def __init__(self, initial_cash):
        self.initial = money(initial_cash)
        self.cash = self.initial
        self.usdgo = money(0)
        self.usdgo_cost = money(0)
        self.usdgo_mark = None
        self.interest_trx = money(0)
        self.interest_sale_proceeds = money(0)
        self.locked = {coin: money(0) for coin in ("USDT", "USDGO", "TRX_INTEREST")}
        self.realized = money(0)
        self.stable_interest = money(0)
        self.fx_profit = money(0)
        self.withdrawn_usdgo_profit = money(0)
        self.external_flows = money(0)
        self.fund_units = self.initial
        self.fees = money(0)
        self.fee_rebates = money(0)
        self.pending_rebates = []
        self.rebate_receipts = []
        self.lots: list[Lot] = []
        self.orders: list[dict] = []
        self.events: set[str] = set()

    @property
    def principal(self):
        return sum((lot.quantity for lot in self.lots), money(0))

    @property
    def cost(self):
        return sum((lot.remaining_cost for lot in self.lots), money(0))

    @property
    def deployable(self):
        return (
            self.initial
            + self.external_flows
            + self.realized
            + self.stable_interest
            + self.fx_profit
            + self.interest_sale_proceeds
        )

    @property
    def free_cash(self):
        return self.cash - self.locked["USDT"]

    def equity(self, price, fx=None):
        if self.usdgo and fx is None and self.usdgo_mark is None:
            raise ValueError("USDGO valuation requires an observed exchange price")
        rate = money(fx) if fx is not None else self.usdgo_mark or money(0)
        return self.cash + self.usdgo * rate + (self.principal + self.interest_trx) * money(price)

    def qualifying(self, price, quantity, fee_rate, margin=0):
        price, quantity, fee, margin = map(money, (price, quantity, fee_rate, margin))
        result = []
        for lot in sorted(self.lots, key=lambda item: item.bought_at, reverse=True):
            available = lot.quantity - lot.locked
            if available <= 0 or price * (1 - fee) <= lot.unit_cost * (1 + margin):
                continue
            amount = min(quantity, available)
            if amount > 0:
                result.append({"lot_id": lot.lot_id, "quantity": float(amount)})
                quantity -= amount
            if quantity <= 0:
                break
        return result

    def buy(
        self,
        event_id,
        order_id,
        timestamp,
        quantity,
        price,
        fee_quote=0,
        fee_base=0,
        quote_amount=None,
    ):
        if event_id in self.events:
            return None
        quantity, price, fee_quote, fee_base = map(money, (quantity, price, fee_quote, fee_base))
        turnover = quantity * price if quote_amount is None else money(quote_amount)
        cost = turnover + fee_quote
        net = quantity - fee_base
        if quantity <= 0 or price <= 0 or turnover <= 0 or min(fee_quote, fee_base) < 0 or net <= 0:
            raise ValueError("Invalid buy fill")
        if cost > self.free_cash:
            raise ValueError("Buy exceeds available strategy cash")
        lot = next((item for item in self.lots if item.lot_id == order_id), None)
        if lot:
            cost_total = lot.remaining_cost + cost
            lot.quantity += net
            lot.remaining_cost = cost_total
            lot.unit_cost = cost_total / lot.quantity
        else:
            lot = Lot(order_id, timestamp, net, cost / net)
            lot.remaining_cost = cost
            self.lots.append(lot)
        self.cash -= cost
        self.fees += fee_quote + fee_base * price
        row = {
            "id": event_id,
            "order_id": order_id,
            "timestamp": timestamp,
            "side": "buy",
            "quantity": float(quantity),
            "net_quantity": float(net),
            "price": float(price),
            "fee": float(fee_quote + fee_base * price),
            "cost": float(cost),
            "profit": None,
            "allocations": [],
        }
        self.orders.append(row)
        self.events.add(event_id)
        return row

    def sell(
        self,
        event_id,
        order_id,
        timestamp,
        quantity,
        price,
        allocations,
        fee_quote=0,
        fee_base=0,
        quote_amount=None,
    ):
        """Account for actual fills truthfully, including unexpected negative realized PnL."""
        if event_id in self.events:
            return None
        quantity, price, fee_quote, fee_base = map(money, (quantity, price, fee_quote, fee_base))
        consumed = quantity + fee_base
        turnover = quantity * price if quote_amount is None else money(quote_amount)
        if quantity <= 0 or price <= 0 or turnover <= 0 or min(fee_quote, fee_base) < 0:
            raise ValueError("Invalid sell fill")
        remaining = consumed
        selected = []
        if len({item["lot_id"] for item in allocations}) != len(allocations):
            raise ValueError("Repeated lot allocation")
        for allocation in allocations:
            lot = next(item for item in self.lots if item.lot_id == allocation["lot_id"])
            amount = min(remaining, money(allocation["quantity"]))
            if amount < 0 or amount > lot.quantity - lot.locked:
                raise ValueError("Sell would consume unavailable principal or retained interest")
            if amount:
                selected.append((lot, amount))
                remaining -= amount
            if remaining <= 0:
                break
        if remaining > money("0.000000000001"):
            raise ValueError("Insufficient allocated principal, including base-denominated fee")
        net_proceeds = turnover - fee_quote
        details = []
        total_cost = money(0)
        for lot, amount in selected:
            cost = (
                lot.remaining_cost
                if amount == lot.quantity
                else lot.remaining_cost * amount / lot.quantity
            )
            proceeds = net_proceeds * amount / consumed
            profit = proceeds - cost
            details.append(
                {
                    "lot_id": lot.lot_id,
                    "quantity": float(amount),
                    "cost": float(cost),
                    "net_proceeds": float(proceeds),
                    "profit": float(profit),
                }
            )
            total_cost += cost
            lot.quantity -= amount
            lot.remaining_cost -= cost
        profit = net_proceeds - total_cost
        self.cash += net_proceeds
        self.realized += profit
        self.fees += fee_quote + fee_base * price
        row = {
            "id": event_id,
            "order_id": order_id,
            "timestamp": timestamp,
            "side": "sell",
            "quantity": float(quantity),
            "price": float(price),
            "fee": float(fee_quote + fee_base * price),
            "cost": float(total_cost),
            "profit": float(profit),
            "allocations": details,
        }
        self.orders.append(row)
        self.events.add(event_id)
        return row

    def credit_interest(self, event_id, coin, amount, fx=None):
        if event_id in self.events:
            return
        amount = money(amount)
        if amount < 0:
            raise ValueError("Interest must be a credited nonnegative amount")
        if coin == "TRX":
            self.interest_trx += amount
        elif coin == "USDT":
            self.cash += amount
            self.stable_interest += amount
        elif coin == "USDGO":
            if fx is None or money(fx) <= 0:
                raise ValueError("USDGO interest requires an observed or explicit research value")
            value = amount * money(fx)
            self.usdgo += amount
            self.usdgo_cost += value
            self.usdgo_mark = money(fx)
            self.stable_interest += value
        else:
            raise ValueError("Unsupported interest currency; reconcile its USDT value first")
        self.events.add(event_id)

    def sell_interest(self, event_id, timestamp, quantity, price, fee_quote):
        if event_id in self.events:
            return next(row for row in self.orders if row["id"] == event_id)
        quantity, price, fee_quote = map(money, (quantity, price, fee_quote))
        if (
            quantity <= 0
            or price <= 0
            or fee_quote < 0
            or quantity > self.interest_trx - self.locked["TRX_INTEREST"]
        ):
            raise ValueError("Interest exit exceeds free retained TRX")
        proceeds = quantity * price - fee_quote
        if proceeds < 0:
            raise ValueError("Interest exit fees exceed proceeds")
        self.interest_trx -= quantity
        self.cash += proceeds
        self.interest_sale_proceeds += proceeds
        self.fees += fee_quote
        row = {
            "id": event_id,
            "order_id": event_id,
            "timestamp": timestamp,
            "side": "sell",
            "kind": "manual_interest_exit",
            "quantity": float(quantity),
            "price": float(price),
            "fee": float(fee_quote),
            "cost": 0,
            "profit": None,
            "net_proceeds": float(proceeds),
            "allocations": [],
        }
        self.orders.append(row)
        self.events.add(event_id)
        return row

    def convert(self, event_id, side, paid, received, mark):
        if event_id in self.events:
            return
        paid, received, mark = map(money, (paid, received, mark))
        if min(paid, received, mark) <= 0:
            raise ValueError("Invalid confirmed currency exchange")
        if side == "to_usdgo":
            if paid > self.free_cash:
                raise ValueError("Exchange exceeds available USDT")
            self.cash -= paid
            self.usdgo += received
            self.usdgo_cost += paid
        elif side == "to_usdt":
            if paid > self.usdgo - self.locked["USDGO"]:
                raise ValueError("Exchange exceeds available USDGO")
            cost = self.usdgo_cost if paid == self.usdgo else self.usdgo_cost * paid / self.usdgo
            self.usdgo -= paid
            self.usdgo_cost -= cost
            self.cash += received
            self.fx_profit += received - cost
        else:
            raise ValueError("Unknown conversion direction")
        self.usdgo_mark = mark
        self.events.add(event_id)

    def external_flow(self, event_id, amount, price):
        if event_id in self.events:
            return
        amount = money(amount)
        if amount < 0 and -amount > self.free_cash:
            raise ValueError("Withdrawal exceeds available cash")
        equity = self.equity(price)
        if equity <= 0 or equity + amount <= 0 or self.fund_units <= 0:
            raise ValueError("Capital flows require positive equity before and after the flow")
        unit_nav = equity / self.fund_units
        self.fund_units += amount / unit_nav
        self.cash += amount
        self.external_flows += amount
        self.events.add(event_id)

    def withdraw_usdgo(self, event_id, amount, price, fx):
        """Remove owned spot capital at NAV; retained valuation profit is not new cash."""
        if event_id in self.events:
            return
        with localcontext() as context:
            context.prec = 50
            amount, price, fx = map(money, (amount, price, fx))
            if not 0 < amount <= self.usdgo - self.locked["USDGO"] or min(price, fx) <= 0:
                raise ValueError("Withdrawal exceeds available USDGO")
            equity, value = self.equity(price, fx), amount * fx
            if equity <= value or self.deployable <= value or self.fund_units <= 0:
                raise ValueError("Withdrawal requires positive remaining capital")
            cost = self.usdgo_cost * amount / self.usdgo
            self.fund_units *= (equity - value) / equity
            self.external_flows -= value
            self.usdgo -= amount
            self.usdgo_cost -= cost
            self.withdrawn_usdgo_profit += value - cost
            self.usdgo_mark = fx
            self.events.add(event_id)

    def queue_fee_rebate(self, fill_id, amount, due_at):
        amount = money(amount)
        row = next(fill for fill in self.orders if fill["id"] == fill_id)
        if "rebate_expected" in row:
            return
        if amount < 0 or amount > money(row["fee"]) + money("0.000000000001"):
            raise ValueError("Invalid fee rebate")
        row["rebate_expected"] = float(amount)
        if amount:
            self.pending_rebates.append(
                {"fill_id": fill_id, "amount": str(amount), "due_at": int(due_at)}
            )
            self.pending_rebates.sort(key=lambda entry: entry["due_at"])

    def settle_fee_rebates(self, timestamp):
        credited = money(0)
        while self.pending_rebates and self.pending_rebates[0]["due_at"] <= timestamp:
            pending = self.pending_rebates[0]
            row = next(fill for fill in self.orders if fill["id"] == pending["fill_id"])
            amount = money(pending["amount"])
            receipt = {**pending, "credited_at": pending["due_at"], "cost_adjustments": []}
            if row["side"] == "buy":
                self.allocate_buy_refund(row["order_id"], amount, receipt)
            elif row.get("kind") == "manual_interest_exit":
                self.interest_sale_proceeds += amount
            else:
                self.realized += amount
            self.cash += amount
            self.fee_rebates += amount
            self.rebate_receipts.append(receipt)
            self.pending_rebates.pop(0)
            credited += amount
        return credited

    def allocate_buy_refund(self, lot_id, amount, receipt):
        lot = next(item for item in self.lots if item.lot_id == lot_id)
        sold = [
            (fill, allocation)
            for fill in self.orders
            for allocation in fill["allocations"]
            if allocation["lot_id"] == lot_id
        ]
        total_quantity = lot.quantity + sum(
            (money(allocation["quantity"]) for _, allocation in sold), money(0)
        )
        remaining = amount
        for fill, allocation in sold:
            part = amount * money(allocation["quantity"]) / total_quantity
            receipt["cost_adjustments"].append(
                {"fill_id": fill["id"], "lot_id": lot_id, "amount": str(part)}
            )
            self.realized += part
            remaining -= part
        # The refund reduces held principal cost. Sold cost corrections are separate
        # dated events; the original execution records remain immutable.
        if lot.quantity:
            lot.remaining_cost -= remaining
            lot.unit_cost = lot.remaining_cost / lot.quantity
        else:
            self.realized += remaining
            if receipt["cost_adjustments"]:
                last = receipt["cost_adjustments"][-1]
                last["amount"] = str(money(last["amount"]) + remaining)

    def reported_fills(self):
        rows = deepcopy(self.orders)
        by_id = {row["id"]: row for row in rows}
        for row in rows:
            row["rebate_credited"] = 0.0
        for receipt in self.rebate_receipts:
            row = by_id[receipt["fill_id"]]
            amount = float(receipt["amount"])
            row["rebate_credited"] += amount
            if row["side"] == "buy":
                row["cost"] -= amount
            elif row.get("kind") == "manual_interest_exit":
                row["net_proceeds"] += amount
            else:
                row["profit"] += amount
                consumed = sum(part["quantity"] for part in row["allocations"])
                for part in row["allocations"]:
                    share = amount * part["quantity"] / consumed
                    part["net_proceeds"] += share
                    part["profit"] += share
            for correction in receipt["cost_adjustments"]:
                sale = by_id[correction["fill_id"]]
                adjustment = float(correction["amount"])
                sale["cost"] -= adjustment
                sale["profit"] += adjustment
                part = next(
                    item for item in sale["allocations"] if item["lot_id"] == correction["lot_id"]
                )
                part["cost"] -= adjustment
                part["profit"] += adjustment
        return rows

    def order_summary(self):
        groups = {}
        for fill in self.reported_fills():
            key = fill["order_id"]
            group = groups.setdefault(
                key,
                {
                    "id": key,
                    "order_id": key,
                    "timestamp": fill["timestamp"],
                    "side": fill["side"],
                    "quantity": 0.0,
                    "notional": 0.0,
                    "fee": 0.0,
                    "rebate_expected": 0.0,
                    "rebate_credited": 0.0,
                    "cost": 0.0,
                    "profit": None if fill["profit"] is None else 0.0,
                    "allocations": {},
                    "fill_count": 0,
                },
            )
            for field in (
                "signal_timestamp",
                "signal_anchor",
                "signal_threshold",
                "confirmation_period",
                "equity_after_fill",
                "kind",
            ):
                if field in fill:
                    group[field] = fill[field]
            group["quantity"] += fill["quantity"]
            group["notional"] += fill["quantity"] * fill["price"]
            group["fee"] += fill["fee"]
            group["rebate_expected"] += fill.get("rebate_expected", 0)
            group["rebate_credited"] += fill["rebate_credited"]
            group["cost"] += fill["cost"]
            group["fill_count"] += 1
            if fill["profit"] is not None:
                group["profit"] += fill["profit"]
            for part in fill["allocations"]:
                allocation = group["allocations"].setdefault(
                    part["lot_id"],
                    {
                        "lot_id": part["lot_id"],
                        "quantity": 0.0,
                        "cost": 0.0,
                        "net_proceeds": 0.0,
                        "profit": 0.0,
                    },
                )
                for field in ("quantity", "cost", "net_proceeds", "profit"):
                    allocation[field] += part[field]
        result = []
        for group in groups.values():
            group["price"] = group.pop("notional") / group["quantity"]
            group["net_fee"] = group["fee"] - group["rebate_credited"]
            group["rebate_pending"] = group["rebate_expected"] - group["rebate_credited"]
            group["allocations"] = list(group["allocations"].values())
            result.append(group)
        return result

    def summary(self, price, fx=None):
        equity = self.equity(price, fx)
        return {
            "equity": float(equity),
            "cash": float(self.cash),
            "free_cash": float(self.free_cash),
            "usdgo": float(self.usdgo),
            "usdgo_market_value": float(
                self.usdgo * money(fx if fx is not None else self.usdgo_mark or 0)
            ),
            "usdgo_unrealized_profit": float(
                self.usdgo * money(fx if fx is not None else self.usdgo_mark or 0) - self.usdgo_cost
            ),
            "principal_trx": float(self.principal),
            "interest_trx": float(self.interest_trx),
            "interest_sale_proceeds": float(self.interest_sale_proceeds),
            "remaining_cost": float(self.cost),
            "deployable": float(self.deployable),
            "realized_profit": float(self.realized),
            "unrealized_profit": float(self.principal * money(price) - self.cost),
            "stable_interest": float(self.stable_interest),
            "fx_profit": float(self.fx_profit),
            "withdrawn_usdgo_profit": float(self.withdrawn_usdgo_profit),
            "total_profit": float(equity - self.initial - self.external_flows),
            "return_ratio": float(equity / self.fund_units - 1) if self.fund_units else 0,
            "fees": float(self.fees - self.fee_rebates),
            "gross_fees": float(self.fees),
            "fee_rebates": float(self.fee_rebates),
            "pending_fee_rebates": float(
                sum((money(entry["amount"]) for entry in self.pending_rebates), money(0))
            ),
        }

    def state(self):
        values = {
            name: str(getattr(self, name))
            for name in (
                "initial",
                "usdgo_cost",
                "cash",
                "usdgo",
                "interest_trx",
                "interest_sale_proceeds",
                "realized",
                "stable_interest",
                "fx_profit",
                "withdrawn_usdgo_profit",
                "external_flows",
                "fund_units",
                "fees",
                "fee_rebates",
            )
        }
        return {
            **values,
            "usdgo_mark": str(self.usdgo_mark) if self.usdgo_mark is not None else None,
            "locked": {k: str(v) for k, v in self.locked.items()},
            "lots": [lot.state() for lot in self.lots],
            "orders": self.orders,
            "events": sorted(self.events),
            "pending_rebates": self.pending_rebates,
            "rebate_receipts": self.rebate_receipts,
        }

    @classmethod
    def restore(cls, state):
        book = cls(state["initial"])
        book.withdrawn_usdgo_profit = money(state.get("withdrawn_usdgo_profit", "0"))
        for key in (
            "cash",
            "usdgo_cost",
            "usdgo",
            "interest_trx",
            "realized",
            "stable_interest",
            "fx_profit",
            "external_flows",
            "fund_units",
            "fees",
        ):
            setattr(book, key, money(state[key]))
        book.usdgo_mark = (
            money(state["usdgo_mark"]) if state.get("usdgo_mark") is not None else None
        )
        book.locked = {k: money(v) for k, v in state["locked"].items()}
        book.lots = [
            Lot(
                row["lot_id"],
                row["bought_at"],
                money(row["quantity"]),
                money(row["unit_cost"]),
                money(row["locked"]),
                money(row["remaining_cost"]),
            )
            for row in state["lots"]
        ]
        book.orders = state["orders"]
        book.events = set(state["events"])
        book.fee_rebates = money(state.get("fee_rebates", 0))
        book.pending_rebates = state.get("pending_rebates", [])
        book.rebate_receipts = state.get("rebate_receipts", [])
        book.interest_sale_proceeds = money(state.get("interest_sale_proceeds", 0))
        return book


class StateStore:
    """Atomic SQLite snapshots with optimistic concurrency across worker processes."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS trx_state "
                "(id TEXT PRIMARY KEY, version INTEGER, state TEXT)"
            )

    def load(self, key):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            row = connection.execute(
                "SELECT version, state FROM trx_state WHERE id=?", (key,)
            ).fetchone()
        return (row[0], json.loads(row[1])) if row else (0, None)

    def save(self, key, version, state):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if version == 0:
                connection.execute(
                    "INSERT INTO trx_state VALUES (?, 1, ?)", (key, json.dumps(state))
                )
            else:
                cursor = connection.execute(
                    "UPDATE trx_state SET version=version+1, state=? WHERE id=? AND version=?",
                    (json.dumps(state), key, version),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Concurrent state change; stop and reload before continuing")
        return version + 1
