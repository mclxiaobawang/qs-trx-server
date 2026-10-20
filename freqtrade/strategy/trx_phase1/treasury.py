from dataclasses import dataclass

from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import money


@dataclass(frozen=True)
class ProductTerms:
    product_id: str
    coin: str
    verified: bool
    minimum: float
    maximum: float
    remaining_quota: float
    interest_start_ms: int
    redemption_delay_ms: int
    # Verified annual rates are decimal fractions; tiers are cumulative quantity caps.
    tiers: tuple[tuple[float, float], ...]

    def expected_interest(self, quantity, hours):
        if not self.verified or hours < 0 or hours > 24 * 30:
            raise ValueError("Verified terms and a bounded forecast window are required")
        remaining, previous, interest = quantity, 0.0, 0.0
        for cap, rate in self.tiers:
            if cap <= previous or rate < 0:
                raise ValueError("Invalid verified rate tiers")
            amount = min(remaining, cap - previous)
            interest += amount * rate * hours / (365 * 24)
            remaining -= amount
            previous = cap
            if remaining <= 0:
                return interest
        if remaining > 1e-12:
            raise ValueError("Quantity exceeds verified rate coverage")
        return interest


class BitgetSavings:
    """Classic-account API adapter. Writes require a separately activated account."""

    def __init__(self, ccxt_client, activated=False):
        self.client, self.activated = ccxt_client, activated

    @staticmethod
    def data(response):
        if response.get("code") != "00000":
            raise ValueError(f"Bitget savings error: {response.get('code')}")
        return response["data"]

    def products(self, coin):
        return self.data(self.client.privateEarnGetV2EarnSavingsProduct({"coin": coin}))

    def terms(self, product_id):
        return self.data(
            self.client.privateEarnGetV2EarnSavingsSubscribeInfo(
                {"productId": product_id, "periodType": "flexible"}
            )
        )

    def interest_records(self, coin, start_ms, end_ms):
        if end_ms - start_ms > 90 * 86400000:
            raise ValueError("Savings record query must be split into 90-day windows")
        params = {
            "coin": coin,
            "periodType": "flexible",
            "orderType": "pay_interest",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 100,
        }
        records, cursors = [], set()
        while True:
            page = self.data(self.client.privateEarnGetV2EarnSavingsRecords(params))
            records.extend(page["resultList"])
            cursor = page.get("endId")
            if len(page["resultList"]) < 100 or not cursor:
                return records
            if cursor in cursors:
                raise ValueError("Savings pagination did not advance")
            cursors.add(cursor)
            params["idLessThan"] = cursor

    def subscribe(self, product_id, quantity):
        if not self.activated:
            raise PermissionError("Real-account Earn operations are not activated")
        return self.data(
            self.client.privateEarnPostV2EarnSavingsSubscribe(
                {"productId": product_id, "periodType": "flexible", "amount": str(quantity)}
            )
        )

    def redeem(self, product_id, asset_order_id, quantity):
        if not self.activated:
            raise PermissionError("Real-account Earn operations are not activated")
        return self.data(
            self.client.privateEarnPostV2EarnSavingsRedeem(
                {
                    "productId": product_id,
                    "orderId": asset_order_id,
                    "periodType": "flexible",
                    "amount": str(quantity),
                }
            )
        )


class Treasury:
    """Location accounting without double-counting NAV; no invented rates or arrivals."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.pending = {}

    def lock_subscription(self, event_id, terms: ProductTerms, quantity, source="principal"):
        book = self.engine.book
        if event_id in book.events:
            return
        if self.engine.pending:
            raise ValueError("A principal decision has priority over treasury allocation")
        if not terms.verified or not terms.minimum <= quantity <= min(
            terms.maximum, terms.remaining_quota
        ):
            raise ValueError("Subscription eligibility, limits or quota are not satisfied")
        amount = money(quantity)
        if terms.coin == "TRX":
            self.lock_trx(amount, source)
        elif terms.coin in ("USDT", "USDGO"):
            available = (
                book.free_cash if terms.coin == "USDT" else book.usdgo - book.locked["USDGO"]
            )
            if amount > available:
                raise ValueError("Insufficient free stablecoin balance")
            book.locked[terms.coin] += amount
        else:
            raise ValueError("Unsupported treasury source")
        book.events.add(event_id)

    def lock_trx(self, amount, source):
        book = self.engine.book
        if source == "principal":
            free = sum((lot.quantity - lot.locked for lot in book.lots), money(0))
            if amount > free:
                raise ValueError("Insufficient unreserved principal")
            remaining = amount
            for lot in book.lots:
                part = min(remaining, lot.quantity - lot.locked)
                lot.locked += part
                remaining -= part
                if remaining <= 0:
                    break
        elif source == "interest":
            if amount > book.interest_trx - book.locked["TRX_INTEREST"]:
                raise ValueError("Insufficient retained interest")
            book.locked["TRX_INTEREST"] += amount
        else:
            raise ValueError("Unsupported treasury source")

    def request_redemption(self, request_id, coin, quantity, source="principal"):
        if request_id in self.pending:
            return
        if quantity <= 0:
            raise ValueError("Invalid redemption amount")
        self.pending[request_id] = {
            "coin": coin,
            "quantity": quantity,
            "source": source,
            "status": "awaiting_actual_arrival",
        }

    def confirm_arrival(self, request_id, event_id, credited_quantity, wallet_verified=False):
        book = self.engine.book
        if event_id in book.events:
            return
        if not wallet_verified:
            raise ValueError("Success response alone does not prove spendable wallet arrival")
        request = self.pending[request_id]
        amount = money(credited_quantity)
        if amount <= 0 or amount > money(request["quantity"]):
            raise ValueError("Arrival exceeds the pending redemption")
        if request["coin"] == "TRX" and request["source"] == "principal":
            remaining = amount
            if remaining > sum((lot.locked for lot in book.lots), money(0)):
                raise ValueError("Arrival exceeds locked principal")
            for lot in book.lots:
                part = min(remaining, lot.locked)
                lot.locked -= part
                remaining -= part
        else:
            key = "TRX_INTEREST" if request["coin"] == "TRX" else request["coin"]
            if amount > book.locked[key]:
                raise ValueError("Arrival exceeds locked assets")
            book.locked[key] -= amount
        request["quantity"] -= float(amount)
        request["status"] = "arrived" if request["quantity"] <= 1e-12 else "partial_arrival"
        book.events.add(event_id)

    def import_credited_interest(self, records):
        for record in records:
            if record.get("orderType") != "pay_interest":
                continue
            coin = record.get("settleCoinName")
            if not coin or not record.get("orderId") or not record.get("ts"):
                raise ValueError("Interest record lacks settlement currency or unique identifier")
            self.engine.book.credit_interest(
                f"bitget-interest-{record['orderId']}-{record['ts']}-{coin}", coin, record["amount"]
            )

    def state(self):
        return {"pending": self.pending}

    def restore(self, state):
        self.pending = state["pending"]


def cash_switch_quote(
    amount,
    hours,
    bid,
    ask,
    fee_rate,
    usdt_terms: ProductTerms,
    usdgo_terms: ProductTerms,
    redemption_wait_hours=0,
    opportunity_cost=0,
    snapshot_penalty=0,
    principal_pending=False,
    price_deviation_limit=0.01,
):
    if principal_pending:
        return {"eligible": False, "reason": "principal_opportunity_has_priority"}
    if amount <= 0 or not 0 <= hours <= 24 * 30 or not 0 < bid <= ask:
        raise ValueError("Invalid cash/quote/forecast inputs")
    if (
        not 0 <= fee_rate < 0.1
        or min(redemption_wait_hours, opportunity_cost, snapshot_penalty) < 0
    ):
        raise ValueError("Invalid execution friction")
    if abs((bid + ask) / 2 - 1) > price_deviation_limit:
        return {"eligible": False, "reason": "stablecoin_deviation"}
    if not usdt_terms.verified or not usdgo_terms.verified:
        return {"eligible": False, "reason": "unverified_product_terms"}
    quantity = amount / (ask * (1 + fee_rate))
    if quantity > usdgo_terms.remaining_quota or quantity < usdgo_terms.minimum:
        return {"eligible": False, "reason": "quota_or_minimum"}
    earned = usdgo_terms.expected_interest(quantity, max(0, hours - redemption_wait_hours))
    alternative = usdt_terms.expected_interest(amount, hours)
    expected_proceeds = (quantity + earned) * bid * (1 - fee_rate)
    benefit = expected_proceeds - amount - alternative - opportunity_cost - snapshot_penalty
    return {
        "eligible": benefit > 0,
        "reason": "positive_expected_net_benefit"
        if benefit > 0
        else "cost_exceeds_expected_benefit",
        "expected_net_benefit": benefit,
        "expected_usdgo_interest": earned,
        "usdt_alternative_interest": alternative,
        "usdgo_quantity": quantity,
        "forecast_only": True,
    }
