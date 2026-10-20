"""Explicit historical scenarios; these do not call or authorize an Earn account."""

from collections import deque

from freqtrade.strategy.trx_phase1.cash_policy import CashPolicy
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.rates import annual_interest, effective_apr


HOUR = 3600000
HOURS_PER_YEAR = 365 * 24
ROUNDING_TOLERANCE = money("1e-18")


class HistoricalEarn:
    """TRX subscriptions start next hour and first pay after a complete accrual hour.

    The hourly base includes only principal still held from before that hour's
    beginning. Redeemed principal earns no interest for the incomplete hour.
    Interest is retained as TRX and resubscribed with the same startup delay.
    These timing/redemption conventions are research assumptions, not product facts.
    """

    def __init__(self, book: Ledger, config: ResearchConfig, start_at: int):
        self.book, self.config = book, config
        self.audit = None
        self.start_at = start_at
        self.next_payment = (start_at // HOUR + 1) * HOUR
        self.retained_eligible = money(0)
        self.retained_waiting = deque()
        self.receipts = []
        self.trx_credit_value = money(0)
        self.usdgo_eligible = money(0)
        self.usdgo_waiting = deque()
        self.usdgo_interest_quantity = money(0)
        self.usdgo_credit_value = money(0)
        self.usdt_credit_value = money(0)
        self.usdt_eligible = money(0)
        self.usdt_waiting = deque()
        self.inactive_principal = []
        self.policy = CashPolicy(self) if config.earn_model == "net_benefit_hourly" else None
        self.conversions = []
        if book.usdgo - book.locked["USDGO"] > 0:
            self.subscribe_usdgo(book.usdgo - book.locked["USDGO"], start_at)
        if book.interest_trx:
            self.retained_waiting.append(((start_at // HOUR + 1) * HOUR, book.interest_trx))

    def subscribe_usdgo(self, quantity, timestamp):
        if quantity:
            self.usdgo_waiting.append(((timestamp // HOUR + 1) * HOUR, quantity))
            self.record(
                "subscribe",
                timestamp,
                coin="USDGO",
                quantity=float(quantity),
                starts_at=(timestamp // HOUR + 1) * HOUR,
            )

    def record(self, kind, timestamp, **values):
        if self.audit is not None:
            self.audit.operation(kind, timestamp, **values)

    def observe_conversion(self):
        if self.audit is not None:
            forecast = self.policy.forecasts[-1] if self.policy and self.policy.forecasts else None
            self.audit.fx.convert(self.conversions[-1], float(self.book.usdgo_mark or 1), forecast)
            self.audit.conversion(self.conversions[-1], float(self.book.usdgo_mark or 1), forecast)

    def reconcile_usdgo(self):
        # Decimal additions to the aggregate balance and cohorts can round at
        # different last digits. The actual ledger is authoritative; this only
        # reconciles eligibility and never credits or debits any asset.
        waiting = sum((entry[1] for entry in self.usdgo_waiting), money(0))
        available = self.book.usdgo - self.book.locked["USDGO"]
        difference = available - (self.usdgo_eligible + waiting)
        if abs(difference) > ROUNDING_TOLERANCE:
            raise ValueError("Historical USDGO subscription quantities do not reconcile")
        self.usdgo_eligible = available - waiting
        if self.usdgo_eligible < 0:
            if self.usdgo_eligible < -ROUNDING_TOLERANCE or not self.usdgo_waiting:
                raise ValueError("USDGO waiting subscriptions exceed the ledger balance")
            eligible_at, quantity = self.usdgo_waiting.pop()
            self.usdgo_waiting.append((eligible_at, quantity + self.usdgo_eligible))
            self.usdgo_eligible = money(0)

    def sweep_cash(self, timestamp, reserved=0):
        if self.policy is not None:
            self.policy.sweep(timestamp, reserved)
            self.sync_usdt(timestamp)
            return
        if self.config.earn_model != "simplified_hourly":
            return
        amount = max(money(0), self.book.free_cash - money(reserved))
        if amount:
            self.book.convert(
                f"research-usdgo-subscribe-{timestamp}-{len(self.conversions)}",
                "to_usdgo",
                amount,
                amount,
                1,
            )

            self.subscribe_usdgo(amount, timestamp)
            self.reconcile_usdgo()
            self.conversions.append(
                {
                    "timestamp": timestamp,
                    "side": "to_usdgo",
                    "quantity": float(amount),
                    "rate": 1,
                    "fee": 0,
                    "simulated": True,
                }
            )

            self.observe_conversion()

    def allocate_cash(self, timestamp, plan=None):
        reserved = max(0, plan.budget - plan.filled_cash) if plan and plan.side == "buy" else 0
        self.sweep_cash(timestamp, reserved)

    def liquid_cash(self):
        available = self.book.free_cash
        if self.config.earn_model in ("simplified_hourly", "net_benefit_hourly"):
            rate = self.book.usdgo_mark or money(1)
            available += (self.book.usdgo - self.book.locked["USDGO"]) * rate
        return float(available)

    def fundable_cash(self, timestamp):
        if self.policy is None:
            return self.liquid_cash()
        if not self.policy.quote_usable(timestamp):
            return float(self.book.free_cash)
        _, bid, _ = self.policy.quote(timestamp)
        return float(
            self.book.free_cash
            + (self.book.usdgo - self.book.locked["USDGO"])
            * money(bid)
            * (1 - money(self.config.fx_fee))
        )

    def remove_usdgo_eligibility(self, amount):
        if self.audit is not None:
            self.audit.fx.remove(float(amount), self.next_payment - 2 * HOUR)
        remaining = amount
        while self.usdgo_waiting and remaining:
            eligible_at, quantity = self.usdgo_waiting.pop()
            used = min(quantity, remaining)
            remaining -= used
            if quantity > used:
                self.usdgo_waiting.append((eligible_at, quantity - used))
        if remaining > self.usdgo_eligible + ROUNDING_TOLERANCE:
            raise ValueError("Historical USDGO subscription quantities do not reconcile")
        self.usdgo_eligible -= remaining

    def prepare_cash(self, plan_id, timestamp, required, expires=None):
        if self.policy is not None:
            available = self.policy.prepare_cash(plan_id, timestamp, required, expires)
            self.sync_usdt(timestamp)
            return available
        if self.config.earn_model != "simplified_hourly":
            return
        amount = min(
            max(money(0), money(required) - self.book.free_cash),
            self.book.usdgo - self.book.locked["USDGO"],
        )
        if not amount:
            return
        self.reconcile_usdgo()
        self.remove_usdgo_eligibility(amount)
        self.record(
            "redemption_requested",
            timestamp,
            coin="USDGO",
            quantity=float(amount),
            plan_id=plan_id,
            due_at=timestamp,
        )
        self.book.convert(f"research-usdgo-fund-{plan_id}", "to_usdt", amount, amount, 1)
        self.record(
            "redemption_received",
            timestamp,
            coin="USDGO",
            quantity=float(amount),
            plan_id=plan_id,
            received=float(amount),
            due_at=timestamp,
        )
        self.reconcile_usdgo()
        self.conversions.append(
            {
                "timestamp": timestamp,
                "side": "to_usdt",
                "quantity": float(amount),
                "rate": 1,
                "fee": 0,
                "simulated": True,
            }
        )

        self.observe_conversion()

    def advance(self, timestamp: int, price: float):
        if self.policy is not None:
            self.policy.advance(timestamp)
            self.sync_usdt(timestamp)
        if self.config.earn_model == "none":
            return
        while self.next_payment <= timestamp:
            paid_at = self.next_payment
            beginning = paid_at - HOUR
            self.prune_completed_redemptions(beginning)
            while self.retained_waiting and self.retained_waiting[0][0] <= beginning:
                _, quantity = self.retained_waiting.popleft()
                self.retained_eligible += quantity
            principal = sum(
                (
                    self.eligible_principal(lot, beginning)
                    for lot in self.book.lots
                    if lot.bought_at < beginning
                ),
                money(0),
            )
            base = principal + self.retained_eligible
            if self.config.trx_earn_cap:
                base = min(base, money(self.config.trx_earn_cap))
            if not self.config.trx_product_available:
                base = money(0)
            interest = annual_interest(self.config, "TRX", base) / HOURS_PER_YEAR
            if interest:
                value = interest * money(price)
                self.book.credit_interest(f"research-trx-interest-{paid_at}", "TRX", interest)
                self.trx_credit_value += value
                self.receipts.append(
                    {
                        "timestamp": paid_at,
                        "coin": "TRX",
                        "eligible_quantity": float(base),
                        "apr": effective_apr(self.config, "TRX", base),
                        "quantity": float(interest),
                        "reference_price": price,
                        "credited_value_usdt": float(value),
                        "simulated": True,
                    }
                )
                # Reinvestment is a new subscription, not immediate automatic compounding.
                self.retained_waiting.append((paid_at + HOUR, interest))
            self.pay_usdgo(paid_at, beginning)
            self.pay_usdt(paid_at)
            self.next_payment += HOUR

    def prune_completed_redemptions(self, beginning):
        # A cohort eligible by this hour's beginning cannot constrain any future
        # payment. Keep in-flight withdrawals and incomplete resubscription hours.
        # The immutable operation log and actual lot/cost ledger remain intact.
        self.inactive_principal = [
            row
            for row in self.inactive_principal
            if row["eligible_at"] is None or row["eligible_at"] > beginning
        ]

    def pay_usdgo(self, paid_at, beginning):
        if self.config.earn_model not in ("simplified_hourly", "net_benefit_hourly"):
            return
        while self.usdgo_waiting and self.usdgo_waiting[0][0] <= beginning:
            _, quantity = self.usdgo_waiting.popleft()
            self.usdgo_eligible += quantity
        self.reconcile_usdgo()
        stable_base = self.usdgo_eligible
        if self.config.usdgo_earn_cap:
            stable_base = min(stable_base, money(self.config.usdgo_earn_cap))
        if not self.config.usdgo_product_available or not self.config.fx_quote_available:
            stable_base = money(0)
        stable_interest = annual_interest(self.config, "USDGO", stable_base) / HOURS_PER_YEAR
        if stable_interest:
            mark = float(self.book.usdgo_mark or 1)
            if self.audit is not None:
                self.audit.fx.credit(float(stable_interest), paid_at, mark)
            self.book.credit_interest(
                f"research-usdgo-interest-{paid_at}", "USDGO", stable_interest, fx=mark
            )
            self.usdgo_interest_quantity += stable_interest
            self.usdgo_credit_value += stable_interest * money(mark)
            self.subscribe_usdgo(stable_interest, paid_at)
            self.reconcile_usdgo()
            self.receipts.append(
                {
                    "timestamp": paid_at,
                    "coin": "USDGO",
                    "eligible_quantity": float(stable_base),
                    "apr": effective_apr(self.config, "USDGO", stable_base),
                    "quantity": float(stable_interest),
                    "reference_price": mark,
                    "credited_value_usdt": float(stable_interest * money(mark)),
                    "simulated": True,
                }
            )

    def eligible_principal(self, lot, beginning):
        inactive = sum(
            (
                money(row["quantity"])
                for row in self.inactive_principal
                if row["lot_id"] == lot.lot_id
                and row.get("since_at", 0) < beginning + HOUR
                and (row["eligible_at"] is None or row["eligible_at"] > beginning)
            ),
            money(0),
        )
        return max(money(0), lot.quantity - inactive)

    def prepare_plan(self, plan):
        ready = plan.created_at
        if plan.side == "buy" and self.policy is not None:
            request = self.policy.requests.get(plan.plan_id)
            if request and request["status"] == "awaiting_arrival":
                ready = request["due_at"]
        if plan.side == "sell" and self.config.earn_model != "none":
            ready += self.config.trx_redemption_delay * 60000
            self.refresh_plan(plan)
            self.record(
                "redemption_requested",
                plan.created_at,
                coin="TRX",
                quantity=plan.quantity,
                plan_id=plan.plan_id,
                due_at=ready,
            )
        return ready

    def refresh_plan(self, plan):
        self.inactive_principal = [
            row for row in self.inactive_principal if row.get("plan_id") != plan.plan_id
        ]
        if plan.side == "sell":
            self.inactive_principal.extend(
                {
                    "plan_id": plan.plan_id,
                    "lot_id": row["lot_id"],
                    "quantity": str(row["quantity"]),
                    "eligible_at": None,
                    "since_at": plan.created_at,
                }
                for row in plan.allocations
                if row["quantity"] > 0
            )

    def finish_plan(self, plan, timestamp):
        if plan.side == "sell" and self.config.earn_model != "none":
            self.record(
                "redemption_consumed" if plan.filled_quantity else "redemption_cancelled",
                timestamp,
                coin="TRX",
                quantity=plan.filled_quantity,
                plan_id=plan.plan_id,
                due_at=plan.ready_at,
            )
        self.cancel_funding(plan.plan_id, timestamp)
        self.inactive_principal = [
            row for row in self.inactive_principal if row.get("plan_id") != plan.plan_id
        ]
        if plan.side == "sell":
            self.inactive_principal.extend(
                {
                    "lot_id": row["lot_id"],
                    "quantity": str(row["quantity"]),
                    "eligible_at": (timestamp // HOUR + 1) * HOUR,
                    "since_at": timestamp,
                }
                for row in plan.allocations
                if row["quantity"] > 0
            )

    def cancel_funding(self, plan_id, timestamp):
        if self.policy is not None:
            self.policy.cancel(plan_id, timestamp)

    def remove_retained_interest(self, amount):
        remaining = amount
        while self.retained_waiting and remaining:
            when, quantity = self.retained_waiting.pop()
            used = min(quantity, remaining)
            remaining -= used
            if quantity > used:
                self.retained_waiting.append((when, quantity - used))
        self.retained_eligible = max(money(0), self.retained_eligible - remaining)

    def sync_usdt(self, timestamp):
        if self.policy is None:
            return
        tracked = self.usdt_eligible + sum((row[1] for row in self.usdt_waiting), money(0))
        difference = self.book.free_cash - tracked
        if difference > ROUNDING_TOLERANCE:
            self.usdt_waiting.append(((timestamp // HOUR + 1) * HOUR, difference))
            self.record(
                "subscribe",
                timestamp,
                coin="USDT",
                quantity=float(difference),
                starts_at=(timestamp // HOUR + 1) * HOUR,
            )
        elif difference < -ROUNDING_TOLERANCE:
            remaining = -difference
            while self.usdt_waiting and remaining:
                when, quantity = self.usdt_waiting.pop()
                used = min(remaining, quantity)
                remaining -= used
                if quantity > used:
                    self.usdt_waiting.append((when, quantity - used))
            self.usdt_eligible = max(money(0), self.usdt_eligible - remaining)

    def pay_usdt(self, paid_at):
        if self.policy is None:
            return
        while self.usdt_waiting and self.usdt_waiting[0][0] <= paid_at - HOUR:
            _, quantity = self.usdt_waiting.popleft()
            self.usdt_eligible += quantity
        interest = annual_interest(self.config, "USDT", self.usdt_eligible) / HOURS_PER_YEAR
        if interest:
            self.book.credit_interest(f"research-usdt-interest-{paid_at}", "USDT", interest)
            self.usdt_credit_value += interest
            self.usdt_waiting.append((paid_at + HOUR, interest))
            self.receipts.append(
                {
                    "timestamp": paid_at,
                    "coin": "USDT",
                    "eligible_quantity": float(self.usdt_eligible),
                    "apr": effective_apr(self.config, "USDT", self.usdt_eligible),
                    "quantity": float(interest),
                    "reference_price": 1,
                    "credited_value_usdt": float(interest),
                    "simulated": True,
                }
            )

    def summary(self, price: float):
        market_value = self.book.interest_trx * money(price)
        return {
            "trx_interest_credited_value": float(self.trx_credit_value),
            "trx_interest_market_value": float(market_value),
            "trx_interest_price_pnl": float(
                market_value + self.book.interest_sale_proceeds - self.trx_credit_value
            ),
            "usdgo_interest_quantity": float(self.usdgo_interest_quantity),
            "usdgo_interest_credited_value": float(self.usdgo_credit_value),
            "usdt_interest_credited_value": float(self.usdt_credit_value),
            "earn_income_credited_value": float(
                self.trx_credit_value + self.usdgo_credit_value + self.usdt_credit_value
            ),
            "earn_income_market_value": float(
                market_value
                + self.usdgo_credit_value
                + self.usdt_credit_value
                + self.book.interest_sale_proceeds
            ),
            "interest_receipt_count": len(self.receipts),
            "cash_policy_reason": self.policy.reason
            if self.policy is not None
            else "simplified_cash_policy",
            "cash_forecast_count": len(self.policy.forecasts) if self.policy is not None else 0,
            "fx_valuation_known": self.config.fx_quote_available,
        }

    def state(self):
        return {
            "start_at": self.start_at,
            "next_payment": self.next_payment,
            "retained_eligible": str(self.retained_eligible),
            "retained_waiting": [[t, str(q)] for t, q in self.retained_waiting],
            "usdgo_eligible": str(self.usdgo_eligible),
            "usdgo_waiting": [[t, str(q)] for t, q in self.usdgo_waiting],
            "usdt_eligible": str(self.usdt_eligible),
            "usdt_waiting": [[t, str(q)] for t, q in self.usdt_waiting],
            "receipts": self.receipts,
            "trx_credit_value": str(self.trx_credit_value),
            "usdgo_interest_quantity": str(self.usdgo_interest_quantity),
            "usdgo_credit_value": str(self.usdgo_credit_value),
            "usdt_credit_value": str(self.usdt_credit_value),
            "inactive_principal": self.inactive_principal,
            "conversions": self.conversions,
            "policy": self.policy.state() if self.policy is not None else None,
        }

    @classmethod
    def restore(cls, book, config, state):
        earn = cls(book, config, state["start_at"])
        earn.next_payment = state["next_payment"]
        for name in (
            "retained_eligible",
            "usdgo_eligible",
            "usdt_eligible",
            "trx_credit_value",
            "usdgo_interest_quantity",
            "usdgo_credit_value",
            "usdt_credit_value",
        ):
            setattr(earn, name, money(state[name]))
        for name in ("retained_waiting", "usdgo_waiting", "usdt_waiting"):
            setattr(earn, name, deque((t, money(q)) for t, q in state[name]))
        earn.receipts = state["receipts"]
        earn.inactive_principal = state["inactive_principal"]
        earn.conversions = state["conversions"]
        if earn.policy is not None and state["policy"] is not None:
            earn.policy.restore(state["policy"])
        return earn
