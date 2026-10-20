"""Read-only attribution of USDGO interest to conversion cohorts.

Eligible interest is allocated proportionally; withdrawals remove waiting cohorts
last-in-first-out, then eligible cohorts proportionally. This accounting convention
is stated explicitly and does not claim an exchange tags income to each conversion.
"""

from collections import deque


class FxAudit:
    def __init__(self):
        self.cohorts, self.cycles = [], []
        self.reinvest = None
        self.removed = deque()

    def subscribe(self, quantity, timestamp):
        allocations = self.reinvest or [(None, quantity)]
        self.reinvest = None
        for cycle, amount in allocations:
            self.cohorts.append(
                {
                    "cycle": cycle,
                    "quantity": amount,
                    "eligible_at": (timestamp // 3600000 + 1) * 3600000,
                }
            )

    def credit(self, quantity, timestamp, mark):
        eligible = [
            row
            for row in self.cohorts
            if row["quantity"] > 1e-12 and row["eligible_at"] <= timestamp - 3600000
        ]
        total = sum(row["quantity"] for row in eligible)
        by_cycle = {}
        for row in eligible:
            value = quantity * row["quantity"] / total if total else 0
            cycle = row["cycle"]
            if cycle is not None:
                by_cycle[cycle] = by_cycle.get(cycle, 0) + value
        self.reinvest = list(by_cycle.items()) or [(None, quantity)]
        for cycle, amount in by_cycle.items():
            self.cycles[cycle]["credited_interest"] += amount
            self.cycles[cycle]["credited_interest_value"] += amount * mark

    def remove(self, quantity, timestamp):
        remaining = quantity
        waiting = [
            row
            for row in reversed(self.cohorts)
            if row["eligible_at"] > (timestamp // 3600000) * 3600000 and row["quantity"] > 0
        ]
        allocations = []
        for row in waiting:
            used = min(remaining, row["quantity"])
            row["quantity"] -= used
            remaining -= used
            if used:
                allocations.append((row["cycle"], used))
        eligible = [row for row in self.cohorts if row["quantity"] > 0 and row not in waiting]
        total = sum(row["quantity"] for row in eligible)
        used = min(total, remaining)
        for row in eligible:
            part = used * row["quantity"] / total if total else 0
            row["quantity"] -= part
            if part:
                allocations.append((row["cycle"], part))
        self.removed.append(allocations)

    def cancel(self):
        if self.removed:
            self.reinvest = self.removed.pop()

    def convert(self, row, mark, forecast=None):
        quantity = row["quantity"]
        if row["side"] == "to_usdgo":
            index = len(self.cycles)
            for cohort in reversed(self.cohorts):
                if cohort["cycle"] is None:
                    cohort["cycle"] = index
                    break
            paid = row.get("paid", quantity)
            self.cycles.append(
                {
                    "id": str(index),
                    "timestamp": row["timestamp"],
                    "quantity": quantity,
                    "paid": paid,
                    "entry_cost": paid - quantity * mark,
                    "exit_cost": 0,
                    "credited_interest": 0,
                    "credited_interest_value": 0,
                    "redeemed_quantity": 0,
                    "last_redemption_at": None,
                    "forecast": forecast,
                }
            )
        elif self.removed:
            removed = self.removed.popleft()
            received = row.get("received", quantity)
            for index, used in removed:
                if index is not None:
                    cycle = self.cycles[index]
                    cycle["redeemed_quantity"] += used
                    cycle["exit_cost"] += (quantity * mark - received) * used / quantity
                    cycle["last_redemption_at"] = row["timestamp"]

    def finish(self, timestamp):
        for index, cycle in enumerate(self.cycles):
            remaining = sum(row["quantity"] for row in self.cohorts if row["cycle"] == index)
            cycle.update(
                remaining_quantity=remaining,
                status="closed" if remaining < 1e-9 else "open",
                held_minutes=(
                    (cycle["last_redemption_at"] if remaining < 1e-9 else timestamp)
                    - cycle["timestamp"]
                )
                / 60000,
                observed_cost=cycle["entry_cost"] + cycle["exit_cost"],
                interest_covers_observed_cost=cycle["credited_interest_value"]
                >= cycle["entry_cost"] + cycle["exit_cost"],
                full_roundtrip_cost_recovered=(
                    cycle["credited_interest_value"] >= cycle["entry_cost"] + cycle["exit_cost"]
                )
                if remaining < 1e-9
                else None,
                attribution=(
                    "eligible pro-rata income; waiting LIFO, eligible pro-rata redemptions; "
                    "open cycles omit future exit cost"
                ),
            )
        return self.cycles
