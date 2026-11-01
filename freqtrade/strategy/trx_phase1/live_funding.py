"""Translate a frozen C1 decision into owned, receipt-checked funding steps."""

from decimal import ROUND_UP

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError, rounded
from freqtrade.strategy.trx_phase1.live_execution import ProductRule


def upward(value, decimals):
    return money(value).quantize(money(10) ** -decimals, rounding=ROUND_UP)


class LiveFunding:
    def __init__(self, execution, config, markets, terms):
        self.execution, self.config = execution, config
        self.markets, self.terms = markets, terms
        self.book = Ledger.restore(execution["book"])
        self.rules = {p["coin"]: ProductRule.model_validate(p) for p in execution["products"]}

    def planning_book(self):
        book = Ledger.restore(self.book.state())
        # Unlock only an independently owned and mapped Savings location for
        # planning. The execution ledger remains locked until arrival is proved.
        for lot in book.lots:
            mapped = sum(
                (
                    money(p.get("lots", {}).get(lot.lot_id, "0"))
                    for p in self.execution["holdings"].values()
                ),
                money(0),
            )
            if mapped != lot.locked:
                raise ExecutionError("principal_location_mismatch")
            lot.locked = money(0)
        book.usdgo_mark = money(self.markets["USDGOUSDT"]["bid"])
        return book

    def available_cash(self):
        market = self.markets["USDGOUSDT"]
        fee = money(market["fee"])
        return self.book.free_cash + self.book.usdgo * money(market["bid"]) * (1 - fee) / (1 + fee)

    def trade(self, symbol, side, amount, plan=None):
        market = self.markets[symbol]
        price = money(market["ask" if side == "buy" else "bid"])
        fee, deviation = money(market["fee"]), money(self.config.maximum_execution_deviation)
        minimum = (
            (
                money(amount) / (price * (1 + deviation))
                if side == "buy"
                else money(amount) * price * (1 - deviation)
            )
            * (1 - fee)
            / (1 + fee)
        )
        result = {
            "kind": "trade",
            "symbol": symbol,
            "side": side,
            "amount": str(amount),
            "reference_price": str(price),
            "maximum_deviation": str(deviation),
            "minimum_receive": str(minimum),
            "profit_margin": str(self.config.execution_margin),
        }
        if plan is not None:
            result.update(signal_anchor=str(plan.anchor), signal_threshold=str(plan.threshold))
        return result

    def subscribe(self, coin, amount, source="principal"):
        return {
            "kind": "subscribe",
            "coin": coin,
            "product_id": self.rules[coin].product_id,
            "amount": str(amount),
            "source": source,
        }

    def redemption(self, coin, amount, holding_id, lots=None):
        return {
            "kind": "redeem",
            "coin": coin,
            "product_id": self.rules[coin].product_id,
            "holding_id": holding_id,
            "amount": str(amount),
            "redeem_lots": {k: str(v) for k, v in (lots or {}).items()},
        }

    def redeem_usdgo(self, required):
        rule = self.rules["USDGO"]
        if rule.redemption_minimum is None:
            raise ExecutionError("redemption_terms_unverified")
        steps = []
        for key, position in self.execution["holdings"].items():
            if required <= 0:
                break
            if position["coin"] != "USDGO" or position["product_id"] != rule.product_id:
                continue
            owned = rounded(position["principal"], rule.redemption_decimals)
            amount = max(rule.redemption_minimum, upward(required, rule.redemption_decimals))
            amount = min(amount, owned)
            if amount >= rule.redemption_minimum:
                steps.append(self.redemption("USDGO", amount, key))
                required -= amount
        if required > 0:
            raise ExecutionError("redeemable_usdgo_unavailable")
        return steps

    def redeem_trx(self, allocations):
        rule = self.rules["TRX"]
        if rule.redemption_minimum is None:
            raise ExecutionError("redemption_terms_unverified")
        free = {lot.lot_id: lot.quantity - lot.locked for lot in self.book.lots}
        required = {
            a["lot_id"]: max(money(0), money(a["quantity"]) - free.get(a["lot_id"], money(0)))
            for a in allocations
        }
        steps = []
        for key, position in self.execution["holdings"].items():
            if position["coin"] != "TRX" or position["product_id"] != rule.product_id:
                continue
            lots = {k: money(v) for k, v in position.get("lots", {}).items()}
            selected = {k: min(q, required.get(k, money(0))) for k, q in lots.items()}
            target = sum(selected.values(), money(0))
            if target == 0:
                continue
            target = max(rule.redemption_minimum, upward(target, rule.redemption_decimals))
            extra = target - sum(selected.values(), money(0))
            for lot_id, amount in lots.items():
                part = min(extra, amount - selected[lot_id])
                selected[lot_id] += part
                extra -= part
            if extra > 0:
                raise ExecutionError("redemption_terms_not_satisfied")
            steps.append(
                self.redemption("TRX", target, key, {k: v for k, v in selected.items() if v})
            )
            for lot_id, q in selected.items():
                required[lot_id] = max(money(0), required.get(lot_id, money(0)) - q)
        if any(required.values()):
            raise ExecutionError("redeemable_profitable_principal_unavailable")
        return steps

    def for_plan(self, plan):
        if plan.side == "sell":
            return [
                *self.redeem_trx(plan.allocations),
                self.trade("TRXUSDT", "sell", money(plan.quantity), plan),
            ]
        budget = money(plan.budget)
        shortage = max(money(0), budget - self.book.free_cash)
        steps = []
        if shortage:
            market = self.markets["USDGOUSDT"]
            fee = money(market["fee"])
            rate = money(market["bid"]) * (1 - fee) / (1 + fee)
            quantity = upward(shortage / rate, market["quantity_decimals"])
            free = self.book.usdgo - self.book.locked["USDGO"]
            steps.extend(self.redeem_usdgo(max(money(0), quantity - free)))
            steps.append(self.trade("USDGOUSDT", "sell", quantity))
        steps.append(self.trade("TRXUSDT", "buy", budget, plan))
        return steps

    def startup(self):
        capital = money(self.book.initial)
        usdgo = capital * (1 - money(self.config.initial_buy_fraction))
        trx = capital - usdgo
        return [
            self.trade("USDGOUSDT", "buy", usdgo),
            self.subscribe("USDGO", "previous"),
            self.trade("TRXUSDT", "buy", trx),
            self.subscribe("TRX", "previous"),
        ]

    def eligible_subscription(self, coin, amount):
        terms = self.terms[coin]
        value = rounded(
            min(money(amount), money(terms["singleMaxAmount"]), money(terms["remainingAmount"])),
            int(terms["subscribePrecision"]),
        )
        return value if value >= money(terms["singleMinAmount"]) else money(0)

    def apr_income(self, coin, amount):
        tiers = self.terms[coin].get("tiers", [])
        if not tiers:
            raise ExecutionError("current_apr_tiers_unverified")
        held = sum(
            (
                money(p["principal"]) + money(p["interest"]) + money(p.get("unassigned_amount", 0))
                for p in self.execution["holdings"].values()
                if p["coin"] == coin
            ),
            money(0),
        )
        income = money(0)
        previous = money(0)
        for tier in tiers:
            low, high, rate = (money(tier[k]) for k in ("minStepVal", "maxStepVal", "currentApy"))
            if low != previous or high <= low or not 0 <= rate <= 100:
                raise ExecutionError("current_apr_tiers_unverified")
            previous = high
            income += max(money(0), min(held + amount, high) - max(held, low)) * rate / 100
        return income * money(self.config.cash_forecast_hours) / (365 * 24)

    def idle(self, now, last_cash_switch):
        free_principal = self.book.principal - sum((lot.locked for lot in self.book.lots), money(0))
        for source, amount in (
            ("principal", free_principal),
            ("interest", self.book.interest_trx - self.book.locked["TRX_INTEREST"]),
        ):
            amount = self.eligible_subscription("TRX", amount)
            if amount and self.apr_income("TRX", amount) > 0:
                return [self.subscribe("TRX", amount, source)]
        usdgo = self.eligible_subscription("USDGO", self.book.usdgo - self.book.locked["USDGO"])
        if usdgo and self.apr_income("USDGO", usdgo) > 0:
            return [self.subscribe("USDGO", usdgo)]
        if now - last_cash_switch < self.config.cash_min_hold_minutes * 60000:
            return []
        cash, market = self.book.free_cash, self.markets["USDGOUSDT"]
        fee, bid, ask = (money(market[k]) for k in ("fee", "bid", "ask"))
        amount = rounded(
            cash / (ask * (1 + fee)) * (1 - fee),
            int(self.terms["USDGO"]["subscribePrecision"]),
        )
        eligible = self.eligible_subscription("USDGO", amount)
        if cash < money(self.config.cash_min_switch_value) or eligible < amount:
            return []
        benefit = (
            self.apr_income("USDGO", eligible) * bid + eligible * bid * (1 - fee) / (1 + fee) - cash
        )
        if benefit <= money(self.config.cash_switch_margin):
            return []
        return [self.trade("USDGOUSDT", "buy", cash), self.subscribe("USDGO", "previous")]
