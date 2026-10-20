"""Bounded Classic REST transport. Importing this module never connects or submits.

The account-page read-only client remains separate. A caller must explicitly enable
writes on this client; no credentials, signed headers or exchange error bodies are
included in exceptions. POST requests are never automatically retried.
"""

import base64
import hashlib
import hmac
import json
import re
import time
from decimal import ROUND_DOWN, InvalidOperation
from urllib.parse import urlencode

import requests

from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.request_limits import (
    RECONCILIATION,
    RateLimitWait,
    RequestLimiter,
    coalesced,
    credential_id,
)


COINS = ("USDT", "USDGO", "TRX")
SYMBOLS = {"TRXUSDT": "TRX", "USDGOUSDT": "USDGO"}
GET_PATHS = {
    "v2/public/time",
    "v2/spot/account/info",
    "v2/spot/account/assets",
    "v2/spot/account/bills",
    "v2/spot/public/symbols",
    "v2/spot/market/orderbook",
    "v2/common/trade-rate",
    "v2/spot/trade/orderInfo",
    "v2/spot/trade/fills",
    "v2/earn/savings/product",
    "v2/earn/savings/subscribe-info",
    "v2/earn/savings/assets",
    "v2/earn/savings/records",
    "v2/earn/savings/subscribe-result",
    "v2/earn/savings/redeem-result",
}
POST_PATHS = {
    "v2/spot/trade/place-order",
    "v2/spot/trade/cancel-order",
    "v2/earn/savings/subscribe",
    "v2/earn/savings/redeem",
}
PUBLIC_PATHS = {"v2/public/time", "v2/spot/public/symbols", "v2/spot/market/orderbook"}
DAY = 86400000


class ExecutionError(Exception):
    """Only fixed codes, never raw private responses, may escape the adapter."""


class WriteUnknown(ExecutionError):
    pass


class WriteRejected(ExecutionError):
    pass


class WriteNotSent(ExecutionError):
    """Proven local deferral before the HTTP request, safe to prepare again."""


def identifier(value):
    if not isinstance(value, (str, int)) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(value)):
        raise ExecutionError("invalid_identifier")
    return str(value)


def positive(value):
    result = money(value)
    if result <= 0:
        raise ExecutionError("nonpositive_quantity")
    return result


class ClassicTransport:
    origin = "https://api.bitget.com/api/"

    def __init__(self, settings, *, writes_enabled=False, session=None, limiter=None):
        if settings.account_type != "classic":
            raise ExecutionError("classic_account_required")
        self.settings = settings
        self.writes_enabled = writes_enabled
        self.session = session or requests.Session()
        # In particular, do not install an HTTP adapter that retries POST requests.
        self.offset = 0
        self.limiter = limiter or RequestLimiter()
        self.credential = credential_id(
            *(
                v.get_secret_value()
                for v in (settings.api_key, settings.secret_key, settings.passphrase)
            )
        )
        self._write_permit = None

    def reserve_write(self, path, lifetime):
        self._write_permit = None
        if lifetime <= 0:
            raise WriteNotSent("request_rate_wait")
        started = self.limiter.clock()
        try:
            self.limiter.acquire(path, self.credential, priority=1, write=True)
        except RateLimitWait:
            raise WriteNotSent("request_rate_wait") from None
        self._write_permit = (path, min(started + lifetime, self.limiter.clock() + 0.05))

    def sync_clock(self):
        server = int(self.call("GET", "v2/public/time")["serverTime"])
        local = int(time.time() * 1000)
        if abs(server - local) > 30000:
            raise ExecutionError("exchange_clock_mismatch")
        self.offset = server - local

    def call(self, method, path, params=None):
        if (
            (method == "GET" and path not in GET_PATHS)
            or (method == "POST" and (path not in POST_PATHS or not self.writes_enabled))
            or method not in {"GET", "POST"}
        ):
            raise ExecutionError("transport_scope_denied")
        params = dict(params or {})
        if method == "POST":
            self.validate_write(path, params)
        if method == "GET" and path == "v2/spot/public/symbols":
            return coalesced(
                ("classic", str(self.limiter.path), None, path, json.dumps(params, sort_keys=True)),
                lambda: self._admitted_call(method, path, params),
                self.limiter.clock,
            )
        return self._admitted_call(method, path, params)

    def _admitted_call(self, method, path, params):
        for attempt in range(4 if method == "GET" else 1):
            if method == "POST":
                if self._write_permit is None:
                    self.reserve_write(path, 10)
                permit, self._write_permit = self._write_permit, None
                if permit[0] != path or self.limiter.clock() >= permit[1]:
                    raise WriteNotSent("request_rate_wait")
            else:
                try:
                    self.limiter.acquire(
                        path,
                        None if path in PUBLIC_PATHS else self.credential,
                        priority=0 if path in RECONCILIATION else 1,
                    )
                except RateLimitWait:
                    raise ExecutionError("request_rate_wait") from None
            try:
                return self._call_once(method, path, params, attempt)
            except RateLimitWait:
                if method == "POST":
                    raise WriteUnknown("exchange_write_429") from None
                if attempt == 3:
                    raise ExecutionError("request_rate_wait") from None

    def _call_once(self, method, path, params, attempt):
        query = urlencode(params) if method == "GET" else ""
        body = json.dumps(params, separators=(",", ":")) if method == "POST" else ""
        route = "/api/" + path + ("?" + query if query else "")
        headers = {"Content-Type": "application/json", "locale": "en-US"}
        if path not in PUBLIC_PATHS:
            stamp = str(int(time.time() * 1000) + self.offset)
            signature = base64.b64encode(
                hmac.new(
                    self.settings.secret_key.get_secret_value().encode(),
                    (stamp + method + route + body).encode(),
                    hashlib.sha256,
                ).digest()
            ).decode()
            headers.update(
                {
                    "ACCESS-KEY": self.settings.api_key.get_secret_value(),
                    "ACCESS-SIGN": signature,
                    "ACCESS-TIMESTAMP": stamp,
                    "ACCESS-PASSPHRASE": self.settings.passphrase.get_secret_value(),
                }
            )
        try:
            response = self.session.request(
                method,
                self.origin + path + ("?" + query if query else ""),
                data=body or None,
                headers=headers,
                timeout=(5, 15),
                allow_redirects=False,
            )
            if response.status_code == 429:
                self.limiter.limited(getattr(response, "headers", {}), attempt)
                raise RateLimitWait()
            payload = response.json()
        except RateLimitWait:
            raise
        except Exception:
            error = WriteUnknown if method == "POST" else ExecutionError
            raise error("exchange_response_unavailable") from None
        if isinstance(payload, dict) and str(payload.get("code")) == "429":
            self.limiter.limited(getattr(response, "headers", {}), attempt)
            raise RateLimitWait()
        if not isinstance(payload, dict) or payload.get("code") != "00000":
            code = str(payload.get("code", "")) if isinstance(payload, dict) else ""
            code = code if re.fullmatch(r"\d{1,10}", code) else "invalid"
            if method == "GET":
                raise ExecutionError("exchange_read_" + code)
            # All other server errors, including duplicate-client-ID errors, may
            # follow acceptance. Resolve by query, never repeat the write.
            error = WriteRejected if code in {"40014", "40085", "40006", "40009"} else WriteUnknown
            raise error("exchange_write_" + code)
        if not 200 <= response.status_code < 300 or "data" not in payload:
            error = WriteUnknown if method == "POST" else ExecutionError
            raise error("exchange_response_unverified")
        return payload["data"]

    @staticmethod
    def validate_write(path, params):
        if path.endswith("place-order"):
            allowed = {
                "symbol",
                "side",
                "orderType",
                "force",
                "size",
                "clientOid",
                "requestTime",
                "receiveWindow",
            }
            if (
                set(params) - allowed
                or params.get("symbol") not in SYMBOLS
                or params.get("side") not in {"buy", "sell"}
                or params.get("orderType") != "market"
            ):
                raise ExecutionError("unsupported_spot_request")
            identifier(params["clientOid"])
            positive(params["size"])
        elif path.endswith("cancel-order"):
            if set(params) != {"clientOid", "symbol"} or params["symbol"] not in SYMBOLS:
                raise ExecutionError("unsupported_cancel_request")
            identifier(params["clientOid"])
        else:
            allowed = {"productId", "periodType", "amount"}
            if path.endswith("redeem"):
                allowed.add("orderId")
                identifier(params["orderId"])
            if set(params) != allowed or params.get("periodType") != "flexible":
                raise ExecutionError("unsupported_savings_request")
            identifier(params["productId"])
            positive(params["amount"])


class ClassicBroker:
    def __init__(self, transport, clock=None):
        self.transport = transport
        self.clock = clock or (lambda: int(time.time() * 1000) + self.transport.offset)
        self.savings_observation = None

    @property
    def writes_enabled(self):
        return self.transport.writes_enabled

    def get(self, path, params=None):
        return self.transport.call("GET", path, params)

    def account_id(self):
        if isinstance(self.transport, ClassicTransport):
            cached = self.transport.limiter.identity(self.transport.credential)
            if cached:
                return cached
        row = self.get("v2/spot/account/info")
        account = hashlib.sha256(identifier(row["userId"]).encode()).hexdigest()
        if isinstance(self.transport, ClassicTransport):
            self.transport.limiter.bind(self.transport.credential, account)
        return account

    def balances(self):
        result = {}
        # Coin-filtered zero balances can carry limitAvailable=null. The full
        # assetType=all response explicitly reports every requested balance and
        # its restriction, without guessing that an unknown restriction is zero.
        rows = self.get("v2/spot/account/assets", {"assetType": "all"})
        for coin in COINS:
            matches = [r for r in rows if r.get("coin") == coin]
            if len(matches) != 1:
                raise ExecutionError("balance_missing_or_duplicated")
            row = matches[0]
            values = {k: money(row[k]) for k in ("available", "frozen", "locked")}
            if min(values.values()) < 0:
                raise ExecutionError("negative_balance")
            restricted = money(row["limitAvailable"])
            if not 0 <= restricted <= values["available"]:
                raise ExecutionError("restricted_balance_unverified")
            values["available"] -= restricted
            values["restricted"] = restricted
            result[coin] = {k: str(v) for k, v in values.items()}
        return result

    def pages(self, path, params, *, list_key=None, cursor_key, limit=100, unique_keys=None):
        params = {**params, "limit": str(limit)}
        result, seen, cursors = [], {}, set()
        for _ in range(30):
            payload = self.get(path, params)
            rows = payload[list_key] if list_key else payload
            if not isinstance(rows, list):
                raise ExecutionError("invalid_page")
            for row in rows:
                key = tuple(str(row[k]) for k in (unique_keys or (cursor_key,)))
                digest = json.dumps(row, sort_keys=True)
                if key in seen and seen[key] != digest:
                    raise ExecutionError("conflicting_page_record")
                if key not in seen:
                    seen[key] = digest
                    result.append(row)
            if len(rows) < limit:
                return result
            cursor = payload.get("endId") if list_key else rows[-1][cursor_key]
            if not cursor or str(cursor) in cursors:
                raise ExecutionError("pagination_incomplete")
            cursors.add(str(cursor))
            params["idLessThan"] = str(cursor)
        raise ExecutionError("pagination_read_limit")

    def market(self, symbol, now):
        if symbol not in SYMBOLS:
            raise ExecutionError("unsupported_market")
        rows = self.get("v2/spot/public/symbols", {"symbol": symbol})
        matches = [r for r in rows if r.get("symbol") == symbol and r.get("status") == "online"]
        if len(matches) != 1:
            raise ExecutionError("market_unavailable")
        row = matches[0]
        book = self.get(
            "v2/spot/market/orderbook", {"symbol": symbol, "type": "step0", "limit": "20"}
        )
        observed_at = self.clock()
        if not -2000 <= observed_at - int(book["ts"]) <= 10000:
            raise ExecutionError("quote_stale")
        bid, ask = positive(book["bids"][0][0]), positive(book["asks"][0][0])
        if bid > ask:
            raise ExecutionError("crossed_orderbook")
        rate = self.get("v2/common/trade-rate", {"symbol": symbol, "businessType": "spot"})
        fee = money(rate["takerFeeRate"])
        if not 0 <= fee < money("0.1"):
            raise ExecutionError("fee_rate_unverified")
        quantity_precision, price_precision = (
            int(row["quantityPrecision"]),
            int(row["pricePrecision"]),
        )
        if (
            not 0
            <= min(quantity_precision, price_precision)
            <= max(quantity_precision, price_precision)
            <= 18
        ):
            raise ExecutionError("invalid_market_precision")
        return {
            "bid": str(bid),
            "ask": str(ask),
            "fee": str(fee),
            "quantity_decimals": quantity_precision,
            "price_decimals": price_precision,
            "minimum_notional": str(positive(row["minTradeUSDT"])),
            "bids": book["bids"],
            "asks": book["asks"],
            "timestamp": int(book["ts"]),
        }

    def product(self, coin, product_id):
        # A new strategy may hold neither product yet. Include selected available
        # and held/retired products, then verify exact identity and status below.
        rows = self.get("v2/earn/savings/product", {"coin": coin, "filter": "all"})
        matches = [
            r
            for r in rows
            if str(r.get("productId")) == product_id
            and r.get("coin") == coin
            and r.get("periodType") == "flexible"
        ]
        if len(matches) != 1:
            raise ExecutionError("product_unverified")
        terms = self.get(
            "v2/earn/savings/subscribe-info", {"productId": product_id, "periodType": "flexible"}
        )
        return {"status": matches[0]["status"], "tiers": matches[0].get("apyList", []), **terms}

    def holdings(self, start, end):
        result = {}
        observations = {}
        # Assets are filtered by creation time, not last update time. Query each
        # window since this execution account was allocated, including old lots.
        for _ in range(200):
            stop = min(end, start + 89 * DAY)
            rows = self.pages(
                "v2/earn/savings/assets",
                {"periodType": "flexible", "startTime": str(start), "endTime": str(stop)},
                list_key="resultList",
                cursor_key="orderId",
            )
            for row in rows:
                if row.get("periodType") == "flexible" and row.get("productCoin") in COINS:
                    key = identifier(row["orderId"])
                    normalized = {
                        "order_id": key,
                        "product_id": identifier(row["productId"]),
                        "coin": row["productCoin"],
                        "amount": str(money(row["holdAmount"])),
                    }
                    if key in result and result[key] != normalized:
                        raise ExecutionError("conflicting_holding")
                    result[key] = normalized
                    observations[key] = {
                        "coin": normalized["coin"],
                        "product_id": normalized["product_id"],
                        "interest_coin": row.get("interestCoin"),
                        "total_profit": self.optional_observed_amount(row.get("totalProfit")),
                    }
            if stop == end:
                # Account-level product earnings are display evidence only. They
                # may include manual assets and never establish strategy income.
                self.savings_observation = {"observed_at": self.clock(), "holdings": observations}
                return result
            start = stop
        raise ExecutionError("holding_history_read_limit")

    @staticmethod
    def optional_observed_amount(value):
        if value is None:
            return None
        try:
            return str(money(value))
        except (InvalidOperation, TypeError, ValueError):
            return None

    def savings_records(self, coin, start, end, *, order_type=None):
        self.history_window(start, end)
        kinds = ("subscribe", "redeem", "pay_interest")
        if order_type is not None and order_type not in kinds:
            raise ExecutionError("unsupported_savings_record_type")
        # Omitting orderType returned subscriptions only on the real account.
        # Read each required stream explicitly, with an independent cursor.
        result = {}
        keys = ("orderId", "orderType", "ts", "coinName", "settleCoinName")
        for kind in (order_type,) if order_type else kinds:
            rows = self.pages(
                "v2/earn/savings/records",
                {
                    "coin": coin,
                    "periodType": "flexible",
                    "orderType": kind,
                    "startTime": str(start),
                    "endTime": str(end),
                },
                list_key="resultList",
                cursor_key="orderId",
                unique_keys=keys,
            )
            for row in rows:
                key = tuple(str(row[k]) for k in keys)
                if key in result and result[key] != row:
                    raise ExecutionError("conflicting_page_record")
                result[key] = row
        return list(result.values())

    def bills(self, coin, start, end):
        self.history_window(start, end)
        if start == end:
            return []  # The exchange rejects a zero-duration history query.
        return self.pages(
            "v2/spot/account/bills",
            {"coin": coin, "startTime": str(start), "endTime": str(end)},
            cursor_key="billId",
            limit=500,
        )

    @staticmethod
    def history_window(start, end):
        if not 0 <= end - start < 90 * DAY:
            raise ExecutionError("history_window_exceeded")

    def order(self, client_id):
        try:
            rows = self.get("v2/spot/trade/orderInfo", {"clientOid": client_id})
        except ExecutionError as exc:
            # A successful empty query and a documented not-found code are both
            # absence of evidence, never proof a timed-out POST was not accepted.
            if str(exc) == "exchange_read_43025":
                return None
            raise
        if not rows:
            return None
        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("clientOid") != client_id:
            raise ExecutionError("unexpected_order")
        return rows[0]

    def fills(self, order_id, symbol, start, end):
        self.history_window(start, end)
        rows = self.pages(
            "v2/spot/trade/fills",
            {"orderId": order_id, "symbol": symbol, "startTime": str(start), "endTime": str(end)},
            cursor_key="tradeId",
        )
        result = []
        for row in rows:
            if row.get("symbol") != symbol or str(row.get("orderId")) != order_id:
                raise ExecutionError("unexpected_fill")
            fees = row["feeDetail"]
            fees = [fees] if isinstance(fees, dict) else fees
            if not isinstance(fees, list) or not fees:
                raise ExecutionError("fill_fees_unverified")
            if any(money(fee.get("totalDeductionFee") or "0") != 0 for fee in fees):
                raise ExecutionError("third_currency_fee_requires_reconciliation")
            stamp = int(row["cTime"])
            if 0 < stamp < 100000000000:
                stamp *= 1000
            result.append(
                {
                    "id": identifier(row["tradeId"]),
                    "order_id": order_id,
                    "side": row["side"],
                    "quantity": str(positive(row["size"])),
                    "quote": str(positive(row["amount"])),
                    "timestamp": stamp,
                    "fees": [
                        {"coin": fee["feeCoin"], "cost": str(-money(fee["totalFee"]))}
                        for fee in fees
                    ],
                }
            )
        return sorted(result, key=lambda r: (r["timestamp"], r["id"]))

    def submit(self, operation):
        kind, request = operation["kind"], operation["request"]
        path = "v2/spot/trade/place-order" if kind == "trade" else "v2/earn/savings/" + kind
        return self.transport.call("POST", path, request)

    def reserve_submission(self, operation, expires_at):
        kind = operation["kind"]
        path = "v2/spot/trade/place-order" if kind == "trade" else "v2/earn/savings/" + kind
        remaining = expires_at - self.clock()
        if kind == "trade":
            remaining = min(remaining, operation["market"]["timestamp"] + 10000 - self.clock())
        self.transport.reserve_write(path, remaining / 1000)

    def cancel(self, client_id, symbol):
        return self.transport.call(
            "POST", "v2/spot/trade/cancel-order", {"clientOid": client_id, "symbol": symbol}
        )

    def reserve_cancellation(self):
        self.transport.reserve_write("v2/spot/trade/cancel-order", 10)

    def result(self, kind, order_id):
        return self.get(
            "v2/earn/savings/" + kind + "-result", {"orderId": order_id, "periodType": "flexible"}
        )


def rounded(value, decimals):
    return money(value).quantize(money(10) ** -decimals, rounding=ROUND_DOWN)
