"""Bounded, causal cash-allocation decisions for an explicitly simulated product."""

import math
from decimal import ROUND_DOWN

from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.rates import annual_interest


class CashPolicy:
    def __init__(self, earn):
        self.earn = earn
        self.requests = {}
        self.pending_requests = set()
        self.forecasts = []
        self.last_redemption = None
        self.last_record = None
        self.reason = "cash_waiting"

    @property
    def config(self):
        return self.earn.config

    @property
    def book(self):
        return self.earn.book

    def quote(self, timestamp):
        cfg = self.config
        mark = cfg.fx_rate
        if (
            cfg.fx_shock_rate
            and timestamp >= self.earn.start_at + cfg.fx_shock_after_minutes * 60000
        ):
            mark = cfg.fx_shock_rate
        bid = mark * (1 - cfg.fx_spread / 2) * (1 - cfg.fx_slippage)
        ask = mark * (1 + cfg.fx_spread / 2) * (1 + cfg.fx_slippage)
        return mark, bid, ask

    def quote_usable(self, timestamp):
        mark, _, _ = self.quote(timestamp)
        return self.config.fx_quote_available and abs(mark - 1) <= self.config.fx_deviation_limit

    def forecast(self, timestamp, amount):
        cfg = self.config
        mark, bid, ask = self.quote(timestamp)
        result = {
            "timestamp": timestamp,
            "amount": float(amount),
            "forecast_hours": cfg.cash_forecast_hours,
            "mark": mark,
            "eligible": False,
            "simulated": True,
        }
        if not cfg.fx_quote_available:
            return {**result, "reason": "usdgo_quote_missing"}
        if abs(mark - 1) > cfg.fx_deviation_limit:
            return {**result, "reason": "usdgo_price_deviation"}
        if not cfg.usdgo_product_available:
            return {**result, "reason": "usdgo_product_unavailable"}
        if amount < cfg.cash_min_switch_value:
            return {**result, "reason": "minimum_switch_value"}
        if (
            self.last_redemption is not None
            and timestamp - self.last_redemption < cfg.cash_min_hold_minutes * 60000
        ):
            return {**result, "reason": "cash_hold_not_elapsed"}
        # No future trade timestamps or future shock quotes enter the forecast.
        quantity = float(amount) / (ask * (1 + cfg.fx_fee))
        if cfg.usdgo_earn_cap:
            quantity = min(quantity, max(0, cfg.usdgo_earn_cap - float(self.book.usdgo)))
        if quantity <= 0:
            return {**result, "reason": "usdgo_quota_exhausted"}
        spend = quantity * ask * (1 + cfg.fx_fee)
        if spend < cfg.cash_min_switch_value:
            return {**result, "reason": "minimum_switch_value"}
        idle_hours = max(
            0, cfg.cash_forecast_hours - 1 - math.ceil(cfg.redemption_delay_minutes / 60)
        )
        interest = quantity * cfg.usdgo_apr * idle_hours / (365 * 24)
        alternative = spend * cfg.usdt_apr * max(0, cfg.cash_forecast_hours - 1) / (365 * 24)
        if cfg.usdgo_apr_tiers or cfg.usdt_apr_tiers or cfg.apr_scenario_scale != 1:
            usdgo = self.book.usdgo - self.book.locked["USDGO"]
            interest = (
                float(
                    annual_interest(cfg, "USDGO", usdgo + money(quantity))
                    - annual_interest(cfg, "USDGO", usdgo)
                )
                * idle_hours
                / (365 * 24)
            )
            cash = self.book.free_cash
            alternative = (
                float(
                    annual_interest(cfg, "USDT", cash)
                    - annual_interest(cfg, "USDT", max(money(0), cash - money(spend)))
                )
                * max(0, cfg.cash_forecast_hours - 1)
                / (365 * 24)
            )
        proceeds = (quantity + interest) * bid * (1 - cfg.fx_fee)
        benefit = proceeds - spend - alternative
        return {
            **result,
            "quantity": quantity,
            "spend": spend,
            "expected_usdgo_interest": interest,
            "usdt_alternative_interest": alternative,
            "roundtrip_cost": spend - quantity * bid * (1 - cfg.fx_fee),
            "expected_net_benefit": benefit,
            "eligible": benefit > cfg.cash_switch_margin,
            "reason": "positive_cash_net_benefit"
            if benefit > cfg.cash_switch_margin
            else "cash_cost_exceeds_benefit",
        }

    def record(self, decision):
        self.reason = decision["reason"]
        signature = (
            decision["reason"],
            round(decision["amount"], 2),
            decision["timestamp"] // 3600000,
        )
        if signature != self.last_record:
            self.forecasts.append(decision)
            self.last_record = signature

    def sweep(self, timestamp, reserved):
        amount = max(money(0), self.book.free_cash - money(reserved))
        if not amount:
            return
        decision = self.forecast(timestamp, amount)
        self.record(decision)
        if not decision["eligible"]:
            return
        _, _, ask = self.quote(timestamp)
        quantity = min(
            money(decision["quantity"]), amount / (money(ask) * (1 + money(self.config.fx_fee)))
        )
        quantity = quantity.quantize(
            money(10) ** -self.config.fx_quantity_decimals, rounding=ROUND_DOWN
        )
        if quantity <= 0:
            return
        spend = quantity * money(ask) * (1 + money(self.config.fx_fee))
        self.book.convert(
            f"research-cash-subscribe-{timestamp}-{len(self.earn.conversions)}",
            "to_usdgo",
            spend,
            quantity,
            decision["mark"],
        )
        self.earn.subscribe_usdgo(quantity, timestamp)
        self.earn.reconcile_usdgo()
        self.earn.conversions.append(
            {
                "timestamp": timestamp,
                "side": "to_usdgo",
                "quantity": float(quantity),
                "paid": float(spend),
                "received": float(quantity),
                "rate": ask,
                "fee": float(spend - quantity * money(ask)),
                "simulated": True,
            }
        )

        self.earn.observe_conversion()

    def prepare_cash(self, plan_id, timestamp, required, expires=None):
        if self.config.usdt_redemption_delay_minutes:
            return self.prepare_waiting_cash(plan_id, timestamp, required, expires)
        available = float(self.book.free_cash)
        shortage = max(money(0), money(required) - self.book.free_cash)
        if not shortage or not self.quote_usable(timestamp):
            return available
        _, bid, _ = self.quote(timestamp)
        net_price = money(bid) * (1 - money(self.config.fx_fee))
        quantity = min(shortage / net_price, self.book.usdgo - self.book.locked["USDGO"])
        quantity = quantity.quantize(
            money(10) ** -self.config.fx_quantity_decimals, rounding=ROUND_DOWN
        )
        if quantity <= 0:
            return available
        self.book.locked["USDGO"] += quantity
        self.earn.remove_usdgo_eligibility(quantity)
        self.earn.reconcile_usdgo()
        self.requests[plan_id] = {
            "quantity": str(quantity),
            "due_at": timestamp + self.config.redemption_delay_minutes * 60000,
            "expires_at": expires,
            "status": "awaiting_arrival",
        }
        self.pending_requests.add(plan_id)
        self.earn.record(
            "redemption_requested",
            timestamp,
            coin="USDGO",
            plan_id=plan_id,
            quantity=float(quantity),
            due_at=self.requests[plan_id]["due_at"],
            expires_at=expires,
        )
        self.advance(timestamp)
        return available + float(quantity * net_price)

    def prepare_waiting_cash(self, plan_id, timestamp, required, expires):
        available = self.book.free_cash
        invested = self.earn.usdt_eligible + sum((q for _, q in self.earn.usdt_waiting), money(0))
        liquid = max(money(0), available - invested)
        usdt = min(max(money(0), money(required) - liquid), available, invested)
        shortage = max(money(0), money(required) - available)
        net_price = money(self.quote(timestamp)[1]) * (1 - money(self.config.fx_fee))
        quantity = money(0)
        if shortage and self.quote_usable(timestamp):
            quantity = min(shortage / net_price, self.book.usdgo - self.book.locked["USDGO"])
            quantity = quantity.quantize(
                money(10) ** -self.config.fx_quantity_decimals, rounding=ROUND_DOWN
            )
        if not usdt and not quantity:
            return float(available)
        delay = max(
            self.config.usdt_redemption_delay_minutes if usdt else 0,
            self.config.redemption_delay_minutes if quantity else 0,
        )
        due = timestamp + delay * 60000
        self.book.locked["USDT"] += usdt
        self.earn.sync_usdt(timestamp)
        self.book.locked["USDGO"] += quantity
        if quantity:
            self.earn.remove_usdgo_eligibility(quantity)
            self.earn.reconcile_usdgo()
        self.requests[plan_id] = {
            "quantity": str(quantity),
            "usdt_quantity": str(usdt),
            "due_at": due,
            "expires_at": expires,
            "status": "awaiting_arrival",
        }
        self.pending_requests.add(plan_id)
        for coin, amount in (("USDT", usdt), ("USDGO", quantity)):
            if amount:
                self.earn.record(
                    "redemption_requested",
                    timestamp,
                    coin=coin,
                    plan_id=plan_id,
                    quantity=float(amount),
                    due_at=due,
                    expires_at=expires,
                )
        self.advance(timestamp)
        return float(available + quantity * net_price)

    def advance(self, timestamp):
        if self.config.fx_quote_available:
            self.book.usdgo_mark = money(self.quote(timestamp)[0])
        for plan_id in tuple(self.pending_requests):
            request = self.requests[plan_id]
            if request["expires_at"] is not None and timestamp >= request["expires_at"]:
                self.cancel(plan_id, timestamp)
                continue
            if timestamp < request["due_at"] or (
                money(request["quantity"]) and not self.quote_usable(timestamp)
            ):
                continue
            quantity = money(request["quantity"])
            usdt = money(request.get("usdt_quantity", 0))
            if usdt:
                self.book.locked["USDT"] -= usdt
                self.earn.record(
                    "redemption_received",
                    timestamp,
                    coin="USDT",
                    plan_id=plan_id,
                    quantity=float(usdt),
                    received=float(usdt),
                    due_at=request["due_at"],
                )
            if not quantity:
                request.update(status="arrived", received=str(usdt))
                self.pending_requests.discard(plan_id)
                self.last_redemption = timestamp
                continue
            mark, bid, _ = self.quote(timestamp)
            received = quantity * money(bid) * (1 - money(self.config.fx_fee))
            self.book.locked["USDGO"] -= quantity
            self.book.convert(f"research-cash-fund-{plan_id}", "to_usdt", quantity, received, mark)
            self.earn.reconcile_usdgo()
            request.update(status="arrived", received=str(received))
            self.pending_requests.discard(plan_id)
            self.last_redemption = timestamp
            self.earn.record(
                "redemption_received",
                timestamp,
                coin="USDGO",
                plan_id=plan_id,
                quantity=float(quantity),
                received=float(received),
                due_at=request["due_at"],
            )
            self.earn.conversions.append(
                {
                    "timestamp": timestamp,
                    "side": "to_usdt",
                    "quantity": float(quantity),
                    "paid": float(quantity),
                    "received": float(received),
                    "rate": bid,
                    "fee": float(quantity * money(bid) - received),
                    "simulated": True,
                }
            )

            self.earn.observe_conversion()

    def cancel(self, plan_id, timestamp):
        request = self.requests.get(plan_id)
        if request and request["status"] == "awaiting_arrival":
            quantity = money(request["quantity"])
            usdt = money(request.get("usdt_quantity", 0))
            if usdt:
                self.book.locked["USDT"] -= usdt
                self.earn.sync_usdt(timestamp)
                self.earn.record(
                    "redemption_cancelled",
                    timestamp,
                    coin="USDT",
                    plan_id=plan_id,
                    quantity=float(usdt),
                    due_at=request["due_at"],
                )
            self.book.locked["USDGO"] -= quantity
            if quantity and self.earn.audit is not None:
                self.earn.audit.fx.cancel()
            self.earn.subscribe_usdgo(quantity, timestamp)
            self.earn.reconcile_usdgo()
            request["status"] = "canceled_before_arrival"
            if quantity:
                self.earn.record(
                    "redemption_cancelled",
                    timestamp,
                    coin="USDGO",
                    plan_id=plan_id,
                    quantity=float(quantity),
                    due_at=request["due_at"],
                )
            self.pending_requests.discard(plan_id)

    def state(self):
        return {
            "requests": self.requests,
            "forecasts": self.forecasts,
            "last_redemption": self.last_redemption,
            "last_record": list(self.last_record) if self.last_record else None,
            "reason": self.reason,
        }

    def restore(self, state):
        self.requests = state["requests"]
        self.pending_requests = {
            plan_id
            for plan_id, request in self.requests.items()
            if request["status"] == "awaiting_arrival"
        }
        self.forecasts = state["forecasts"]
        self.last_redemption = state["last_redemption"]
        self.last_record = tuple(state["last_record"]) if state["last_record"] else None
        self.reason = state["reason"]
