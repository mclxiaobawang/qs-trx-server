"""Translate a frozen C1 decision into owned, receipt-checked funding steps."""

from decimal import ROUND_UP

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError, rounded
from freqtrade.strategy.trx_phase1.live_execution import ProductRule
from freqtrade.strategy.trx_phase1.stablecoin_policy import (
    REDEMPTION_SPACING_MS,
    policy_for,
)


def upward(value, decimals):
    return money(value).quantize(money(10) ** -decimals, rounding=ROUND_UP)


class LiveFunding:
    def __init__(self, execution, config, markets, terms):
        self.execution, self.config = execution, config
        self.markets, self.terms = markets, terms
        self.book = Ledger.restore(execution["book"])
        self.policy = policy_for(execution)
        products = execution["products"] + (self.policy or {}).get("products", [])
        self.rules = {p["coin"]: ProductRule.model_validate(p) for p in products}

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
        if self.policy:
            book.usdc_mark = money(self.markets["USDCUSDT"]["bid"])
        return book

    def available_cash(self):
        if self.policy:
            return self.owned_buying_power()
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

    def stable_trade(self, coin, side, amount, quote="USDT"):
        step = self.trade(coin + quote, side, amount)
        market = self.markets[step["symbol"]]
        price, fee = money(step["reference_price"]), money(market["fee"])
        # Limit stablecoin price drift independently of the original TRX protection.
        deviation = min(money("0.0002"), money(step["maximum_deviation"]))
        step["maximum_deviation"] = str(deviation)
        precision = market["price_decimals"] if side == "buy" else market["quantity_decimals"]
        tradable = rounded(amount / (1 + fee), precision)
        step["minimum_receive"] = str(
            tradable * (1 - fee) / (price * (1 + deviation))
            if side == "buy"
            else tradable * (1 - fee) * price * (1 - deviation)
        )
        return step

    def stable_spot(self, coin):
        if coin == "USDT":
            return self.book.free_cash
        if coin == "USDC":
            return self.book.usdc - self.book.usdc_locked
        return self.book.usdgo - self.book.locked["USDGO"]

    def stable_holding_amount(self, position, outside):
        if outside:
            from freqtrade.strategy.trx_phase1.external_stablecoin import holding_amount

            return holding_amount(position)
        return money(position["principal"])

    def redeemable_stable(self, coin, *, outside=False):
        rule = self.rules[coin]
        if rule.redemption_minimum is None:
            return money(0)
        amounts = (
            rounded(self.stable_holding_amount(p, outside), rule.redemption_decimals)
            for p in self.execution["holdings"].values()
            if p["coin"] == coin and p["product_id"] == rule.product_id
        )
        return sum((q for q in amounts if q >= rule.redemption_minimum), money(0))

    def owned_buying_power(self):
        total = self.book.free_cash
        for coin in ("USDT", "USDC", "USDGO"):
            amount = money(0) if coin == "USDT" else self.stable_spot(coin)
            amount += self.redeemable_stable(coin)
            if coin == "USDT":
                total += amount
            elif amount:
                total += money(self.stable_trade(coin, "sell", amount)["minimum_receive"])
        return total

    def redeem_stable(self, coin, required, *, outside=False):
        rule = self.rules[coin]
        if rule.redemption_minimum is None:
            raise ExecutionError("redemption_terms_unverified")
        steps = []
        for key, position in sorted(self.execution["holdings"].items()):
            if required <= 0:
                break
            if position["coin"] != coin or position["product_id"] != rule.product_id:
                continue
            owned = rounded(self.stable_holding_amount(position, outside), rule.redemption_decimals)
            amount = min(
                owned, upward(max(required, rule.redemption_minimum), rule.redemption_decimals)
            )
            if amount >= rule.redemption_minimum:
                steps.append(self.redemption(coin, amount, key))
                required -= amount
        if required > 0:
            raise ExecutionError("stablecoin_owned_redemption_unavailable")
        return steps

    def buy_funding(self, plan, now):
        budget = money(plan.budget)
        gap = max(money(0), budget - self.book.free_cash)
        steps, exchanges = [], []
        redeemable_usdt = self.redeemable_stable("USDT")
        if gap and redeemable_usdt:
            amount = min(gap, redeemable_usdt)
            try:
                steps = self.redeem_stable("USDT", amount)
            except ExecutionError:
                pass
            else:
                gap = max(money(0), gap - sum((money(s["amount"]) for s in steps), money(0)))
        for coin in ("USDC", "USDGO"):
            if not gap:
                break
            market = self.markets[coin + "USDT"]
            rate = money(self.stable_trade(coin, "sell", money(1))["minimum_receive"])
            spot = self.stable_spot(coin)
            available = spot + self.redeemable_stable(coin)
            decimals = market["quantity_decimals"]
            minimum = money(market["minimum_notional"]) / money(market["bid"])
            amount = min(upward(max(gap / rate, minimum), decimals), rounded(available, decimals))
            if amount < minimum:
                continue
            if amount > spot:
                try:
                    steps.extend(self.redeem_stable(coin, amount - spot))
                except ExecutionError:
                    amount = rounded(spot, decimals)
            if amount >= minimum:
                exchange = self.stable_trade(coin, "sell", amount)
                exchanges.append(exchange)
                gap = max(money(0), gap - money(exchange["minimum_receive"]))
        if gap:
            raise ExecutionError("stablecoin_buy_funding_unavailable")
        # Earliest possible admission, not an assumed venue arrival guarantee.
        if steps and now + (len(steps) - 1) * REDEMPTION_SPACING_MS >= plan.expires_at:
            raise ExecutionError("stablecoin_buy_funding_window_too_short")
        return [*steps, *exchanges, self.trade("TRXUSDT", "buy", budget, plan)]

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

    def for_plan(self, plan, now=None):
        if plan.side == "sell":
            return [
                *self.redeem_trx(plan.allocations),
                self.trade("TRXUSDT", "sell", money(plan.quantity), plan),
            ]
        if self.policy:
            return self.buy_funding(plan, plan.created_at if now is None else now)
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

    def subscription(self, coin, amount, observed):
        terms = observed[coin]
        if amount <= 0 or self.rules[coin].redemption_minimum is None:
            return money(0)
        value = rounded(
            min(amount, money(terms["singleMaxAmount"]), money(terms["remainingAmount"])),
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
        if self.policy:
            return []
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
