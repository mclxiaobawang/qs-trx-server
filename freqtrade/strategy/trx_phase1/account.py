"""Read-only Bitget account inspection, isolated from research and execution wallets.

No account assets are imported into Engine/Ledger. Credentials and raw exchange
responses never leave this module. A transport allowlist also blocks writes if a
caller accidentally invokes a CCXT trading, transfer or Earn method.
"""

import hashlib
import json
import os
import re
import stat
import threading
import time
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Literal

import ccxt
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from freqtrade.strategy.trx_phase1.account_history import AccountHistory, observation
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.request_limits import LimitedReadMixin, RequestLimiter


CONFIG_NAME = "bitget-account.local.json"
ENV_KEYS = ("TRX_BITGET_API_KEY", "TRX_BITGET_SECRET_KEY", "TRX_BITGET_PASSPHRASE")
COINS = ("USDT", "USDGO", "TRX")
MARKETS = {"TRXUSDT": "TRX", "USDGOUSDT": "USDGO"}
DAY_MS = 86400000
FUNDING_SAVINGS_TYPES = (
    "savings_flexible_subscribe",
    "savings_flexible_redeem",
    "savings_flexible_profit",
)
READ_PATHS = frozenset(
    {
        "v2/public/time",
        "v2/spot/market/tickers",
        "v2/spot/account/info",
        "v2/spot/account/assets",
        "v2/spot/trade/unfilled-orders",
        "v2/spot/trade/fills",
        "v3/account/info",
        "v3/account/settings",
        "v3/account/assets",
        "v3/account/funding-assets",
        "v3/account/funding-financial-records",
        "v3/trade/unfilled-orders",
        "v3/trade/fills",
        "v2/earn/savings/assets",
        "v2/earn/savings/product",
        "v2/earn/savings/subscribe-info",
        "v2/earn/savings/records",
    }
)


class AccountSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    account_type: Literal["classic", "uta"] = "classic"
    include_earn: bool = True
    history_days: int = Field(default=7, ge=1, le=30)
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)
    secret_key: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)
    passphrase: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)


class AccountReadError(Exception):
    """Only a fixed, non-sensitive code is safe to expose to the API."""


class AccountBusy(Exception):
    pass


class ReadOnlyBitget(LimitedReadMixin, ccxt.bitget):
    def request(
        self, path, api="public", method="GET", params=None, headers=None, body=None, config=None
    ):
        if method.upper() != "GET" or path not in READ_PATHS:
            raise PermissionError("This connection permits allowlisted account reads only")
        return super().request(path, api, method, params or {}, headers, body, config or {})

    def log(self, *args):
        # CCXT debug/verbose logging can include signed headers and raw responses.
        pass


def load_settings(path: Path):
    """Use one complete credential source, never combine parts of different keys."""
    try:
        raw = {}
        if path.exists():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
                raise AccountReadError("invalid_configuration")
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict):
                raise AccountReadError("invalid_configuration")
            if any(raw.get(key) for key in ("api_key", "secret_key", "passphrase")) and (
                info.st_mode & 0o077
            ):
                raise AccountReadError("private_file_permissions")
        values = [os.environ.get(key, "").strip() for key in ENV_KEYS]
        source = "environment" if any(values) else "local_file" if path.exists() else "none"
        if any(values):
            raw.update(zip(("api_key", "secret_key", "passphrase"), values, strict=True))
        settings = AccountSettings.model_validate(raw)
        credentials = [
            settings.api_key.get_secret_value(),
            settings.secret_key.get_secret_value(),
            settings.passphrase.get_secret_value(),
        ]
        if any(credentials) and not all(credentials):
            raise AccountReadError("incomplete_credentials")
        return settings, source, all(credentials)
    except AccountReadError:
        raise
    except Exception:
        # Validation errors include their input; JSON errors may include file text.
        raise AccountReadError("invalid_configuration") from None


def failure_code(exc):
    if isinstance(exc, AccountReadError):
        return str(exc)
    if re.search(r'"code"\s*:\s*"?59012(?:"|[,}])', str(exc)):
        return "product_detail_unavailable"
    if isinstance(exc, ccxt.PermissionDenied):
        # CCXT also classifies an account-mode mismatch as PermissionDenied.
        # Extract only the numeric code; never return the exception or response.
        if re.search(r'"code"\s*:\s*"?40085(?:"|[,}])', str(exc)):
            return "uta_classic_api_unsupported"
        return "permission_denied"
    if isinstance(exc, ccxt.AuthenticationError):
        return "authentication_failed"
    if isinstance(exc, ccxt.NotSupported):
        return "account_api_not_supported"
    if isinstance(exc, ccxt.RateLimitExceeded):
        return "rate_limited"
    if isinstance(exc, ccxt.NetworkError):
        return "network_error"
    return "invalid_exchange_response"


def response_data(response):
    if isinstance(response, dict) and response.get("code") == "59012":
        raise AccountReadError("product_detail_unavailable")
    if not isinstance(response, dict) or response.get("code") != "00000":
        raise AccountReadError("invalid_exchange_response")
    return response["data"]


def decimal_text(value):
    return str(money(value))


def text_id(value):
    if not isinstance(value, (str, int)) or not str(value) or len(str(value)) > 128:
        raise AccountReadError("invalid_exchange_response")
    return str(value)


def mask_uid(value):
    value = text_id(value)
    if not value.isdigit():
        raise AccountReadError("invalid_exchange_response")
    return "****" + value[-4:]


def normalize_balances(data, uta=False):
    rows = data["assets"] if uta else data
    if not isinstance(rows, list):
        raise AccountReadError("invalid_exchange_response")
    result, seen = [], set()
    for row in rows:
        coin = row["coin"]
        if coin not in COINS:
            continue
        if coin in seen:
            raise AccountReadError("duplicate_asset")
        seen.add(coin)
        if uta:
            # UTA available/equity can include margin and derivative effects.
            # They are reported separately, never added into a fictitious spot total.
            values = {
                key: decimal_text(row[key])
                for key in ("available", "locked", "balance", "equity", "debt")
            }
            values.update(
                frozen=None,
                restricted_available=None,
                total=values["balance"],
                integrity="uta_requires_strategy_allocation",
            )
        else:
            values = {
                key: decimal_text(row[key])
                for key in ("available", "frozen", "locked", "limitAvailable")
            }
            if any(money(value) < 0 for value in values.values()):
                raise AccountReadError("invalid_exchange_response")
            # limitAvailable is a restriction within available, not another asset.
            values["restricted_available"] = values.pop("limitAvailable")
            if money(values["restricted_available"]) > money(values["available"]):
                raise AccountReadError("invalid_exchange_response")
            total = sum((money(values[key]) for key in ("available", "frozen", "locked")), money(0))
            values.update(
                total=str(total),
                balance=str(total),
                equity=None,
                debt="0",
                integrity="components_observed",
            )
        result.append({"coin": coin, **values})
    # Absence is not invented as a zero: callers can distinguish omitted currencies.
    return sorted(result, key=lambda row: COINS.index(row["coin"]))


def normalize_order(row, uta=False, symbol="TRXUSDT"):
    if (
        symbol not in MARKETS
        or row.get("symbol") != symbol
        or (uta and row.get("category") != "SPOT")
    ):
        raise AccountReadError("unexpected_market")
    if row.get("side") not in ("buy", "sell") or row.get("orderType") not in ("limit", "market"):
        raise AccountReadError("invalid_exchange_response")
    quantity = decimal_text(row["qty" if uta else "size"])
    return {
        "symbol": symbol,
        "base_coin": MARKETS[symbol],
        "order_id": text_id(row["orderId"]),
        "side": row["side"],
        "order_type": row["orderType"],
        "status": str(row["orderStatus" if uta else "status"])[:40],
        "price": decimal_text(row.get("price") or row.get("priceAvg") or "0"),
        "quantity": quantity,
        "quantity_coin": "USDT"
        if row["side"] == "buy" and row["orderType"] == "market"
        else MARKETS[symbol],
        "filled_quantity": decimal_text(row["cumExecQty" if uta else "baseVolume"]),
        # Retain the old TRX field for existing API consumers, never label USDGO as TRX.
        "filled_trx": decimal_text(row["cumExecQty" if uta else "baseVolume"])
        if symbol == "TRXUSDT"
        else None,
        "timestamp": int(row["createdTime" if uta else "cTime"]),
    }


def normalize_fill(row, uta=False, symbol="TRXUSDT"):
    if (
        symbol not in MARKETS
        or row.get("symbol") != symbol
        or (uta and row.get("category") != "SPOT")
    ):
        raise AccountReadError("unexpected_market")
    if row.get("side") not in ("buy", "sell"):
        raise AccountReadError("invalid_exchange_response")
    fees = row.get("feeDetail")
    if isinstance(fees, dict):
        fees = [fees]
    if not isinstance(fees, list):
        raise AccountReadError("invalid_exchange_response")
    normalized = []
    for fee in fees:
        coin = fee["feeCoin"]
        if not isinstance(coin, str) or not coin.isalnum() or len(coin) > 20:
            raise AccountReadError("invalid_exchange_response")
        # Preserve the native API sign and denomination. Never assume USDT fees
        # or invent the 50% research rebate for a real account.
        normalized.append(
            {
                "coin": coin,
                "amount": decimal_text(fee["fee" if uta else "totalFee"]),
                "deduction_amount": decimal_text(fee.get("totalDeductionFee") or "0"),
            }
        )
    return {
        "symbol": symbol,
        "base_coin": MARKETS[symbol],
        "trade_id": text_id(row["execId" if uta else "tradeId"]),
        "order_id": text_id(row["orderId"]),
        "side": row["side"],
        "price": decimal_text(row["execPrice" if uta else "priceAvg"]),
        "quantity": decimal_text(row["execQty" if uta else "size"]),
        "turnover": decimal_text(row["execValue" if uta else "amount"]),
        "timestamp": int(row["createdTime" if uta else "cTime"]),
        "fees": normalized,
    }


def normalize_savings(row):
    coin = row["productCoin"]
    if coin not in COINS or row["periodType"] != "flexible":
        return None
    return {
        "product_id": text_id(row["productId"]),
        "order_id": text_id(row["orderId"]),
        "coin": coin,
        "interest_coin": str(row["interestCoin"])[:20],
        "quantity": decimal_text(row["holdAmount"]),
        "total_profit": decimal_text(row["totalProfit"]),
        "status": str(row["status"])[:40],
    }


def normalize_funding_balances(data):
    if not isinstance(data, list):
        raise AccountReadError("invalid_exchange_response")
    result, seen = [], set()
    for row in data:
        coin = row["coin"]
        if coin not in COINS:
            continue
        if coin in seen:
            raise AccountReadError("duplicate_asset")
        seen.add(coin)
        values = {key: money(row[key]) for key in ("available", "frozen", "balance")}
        if min(values.values()) < 0:
            raise AccountReadError("invalid_exchange_response")
        delta = values["balance"] - values["available"] - values["frozen"]
        result.append(
            {
                "coin": coin,
                **{key: str(value) for key, value in values.items()},
                "component_difference": str(delta),
                "location": "funding",
                "integrity": "components_match" if delta == 0 else "components_mismatch",
            }
        )
    return sorted(result, key=lambda row: COINS.index(row["coin"]))


def normalize_savings_record(row):
    if row["coinName"] not in COINS or row["productType"] != "flexible":
        return None
    if row["orderType"] not in ("subscribe", "redeem", "pay_interest"):
        raise AccountReadError("unexpected_savings_record")
    coin = row["settleCoinName"]
    if not isinstance(coin, str) or not coin.isalnum() or len(coin) > 20:
        raise AccountReadError("invalid_exchange_response")
    amount, timestamp = money(row["amount"]), int(row["ts"])
    if amount < 0 or timestamp <= 0:
        raise AccountReadError("invalid_exchange_response")
    order_id = text_id(row["orderId"])
    event_id = f"{order_id}:{timestamp}:{row['orderType']}:{row['coinName']}:{coin}"
    return {
        "event_id": event_id,
        "order_id": order_id,
        "coin": row["coinName"],
        "settlement_coin": coin,
        "kind": row["orderType"],
        "amount": str(amount),
        "timestamp": timestamp,
        "source": "savings_v2",
        "account_location": "unverified",
    }


def normalize_funding_record(row, expected_kind):
    if row["type"] != expected_kind:
        raise AccountReadError("unexpected_savings_record")
    if row["coin"] not in COINS:
        return None
    timestamp = int(row["ts"])
    if timestamp <= 0:
        raise AccountReadError("invalid_exchange_response")
    return {
        "record_id": text_id(row["id"]),
        "coin": row["coin"],
        "kind": row["type"],
        "amount": decimal_text(row["amount"]),
        "balance_after": decimal_text(row["balance"]),
        "timestamp": timestamp,
        "source": "funding_v3",
        "location": "funding",
    }


def normalize_product_terms(data):
    values = {
        key: money(data[key]) for key in ("singleMinAmount", "singleMaxAmount", "remainingAmount")
    }
    if min(values.values()) < 0 or values["singleMinAmount"] > values["singleMaxAmount"]:
        raise AccountReadError("invalid_exchange_response")
    precision = int(data["subscribePrecision"])
    if not 0 <= precision <= 30:
        raise AccountReadError("invalid_exchange_response")
    # Preserve the exchange's delay description. An empty value is unknown,
    # never the backtest's assumed one-minute arrival.
    delay = data.get("redeemDelay") or None
    if delay is not None and (not isinstance(delay, str) or len(delay) > 40):
        raise AccountReadError("invalid_exchange_response")
    return {
        "single_minimum": str(values["singleMinAmount"]),
        "single_maximum": str(values["singleMaxAmount"]),
        "remaining_amount": str(values["remainingAmount"]),
        "subscription_precision": precision,
        "redemption_delay": delay,
        "interest_start_at": int(data["interestTime"]) if data.get("interestTime") else None,
        "settlement_at": int(data["settleTime"]) if data.get("settleTime") else None,
        "eligibility_verified": False,
    }


class AccountReader:
    def __init__(self, client, settings: AccountSettings, now_ms, clock_ms=None):
        self.client, self.settings, self.now = client, settings, now_ms
        self.clock_ms = clock_ms or (lambda: now_ms)
        self.deadline = time.monotonic() + 75
        self._uta_fills = None

    def _read(self, method, params=None):
        if time.monotonic() >= self.deadline:
            raise AccountReadError("read_budget_exceeded")
        return response_data(method(dict(params or {})))

    def collect(self):
        uta = self.settings.account_type == "uta"
        sections = {
            "balances": lambda: normalize_balances(
                self._read(
                    self.client.privateUtaGetV3AccountAssets
                    if uta
                    else self.client.privateSpotGetV2SpotAccountAssets,
                    {} if uta else {"assetType": "all"},
                ),
                uta,
            ),
            "orders": self.recent_trading,
            "fills": lambda: self.recent_trading(fills=True),
            "conversion_orders": lambda: self.recent_trading(symbol="USDGOUSDT"),
            "conversion_fills": lambda: self.recent_trading(fills=True, symbol="USDGOUSDT"),
            "prices": self.prices,
        }
        if self.settings.include_earn:
            # Ordinary Savings remains the documented v2 product family. Try
            # these read-only endpoints for UTA instead of claiming that the
            # unrelated v3 On-chain Elite API represents TRX/USDGO Savings.
            sections.update(
                savings=self.savings, products=self.products, savings_records=self.savings_records
            )
            if uta:
                sections.update(
                    funding_balances=lambda: normalize_funding_balances(
                        self._read(self.client.privateUtaGetV3AccountFundingAssets)
                    ),
                    funding_records=self.funding_records,
                )
        data, checks = {}, {}
        for section, read in sections.items():
            try:
                data[section] = read()
                partial = isinstance(data[section], dict) and data[section].get("partial", False)
                if section == "funding_balances":
                    partial = any(
                        row["integrity"] == "components_mismatch" for row in data[section]
                    )
                checks[section] = {"status": "partial" if partial else "success", "error": None}
            except Exception as exc:
                data[section] = None
                error = failure_code(exc)
                checks[section] = {
                    "status": "unsupported"
                    if error in ("uta_classic_api_unsupported", "account_api_not_supported")
                    else "error",
                    "error": error,
                }
        for section in (
            "savings",
            "products",
            "savings_records",
            "funding_balances",
            "funding_records",
        ):
            if section not in checks:
                data[section] = None
                checks[section] = {
                    "status": "not_applicable" if self.settings.include_earn else "disabled",
                    "error": None,
                }
        return data, checks

    def prices(self):
        result = {}
        for coin in ("USDGO", "TRX"):
            symbol = coin + "USDT"
            rows = self._read(self.client.publicSpotGetV2SpotMarketTickers, {"symbol": symbol})
            if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("symbol") != symbol:
                raise AccountReadError("invalid_exchange_response")
            row = rows[0]
            bid, ask, timestamp = money(row["bidPr"]), money(row["askPr"]), int(row["ts"])
            if bid <= 0 or ask < bid or not -5000 <= self.clock_ms() - timestamp <= 60000:
                raise AccountReadError("invalid_exchange_response")
            result[coin] = {"bid": str(bid), "ask": str(ask), "timestamp": timestamp}
        return result

    def _pages(
        self,
        method,
        params,
        list_key="resultList",
        cursor_key="endId",
        cursor_parameter="idLessThan",
    ):
        """Do not infer that a short page with a cursor is the last page."""
        params, cursors = dict(params), set()
        for _ in range(20):
            data = self._read(method, params)
            rows = data[list_key]
            if not isinstance(rows, list):
                raise AccountReadError("invalid_exchange_response")
            yield rows
            if not rows:
                return
            cursor = data.get(cursor_key)
            if not cursor:
                if len(rows) >= 100:
                    raise AccountReadError("pagination_incomplete")
                return
            cursor = text_id(cursor)
            if cursor in cursors:
                raise AccountReadError("pagination_incomplete")
            cursors.add(cursor)
            params[cursor_parameter] = cursor
        raise AccountReadError("pagination_incomplete")

    def recent_trading(self, fills=False, symbol="TRXUSDT"):
        if symbol not in MARKETS:
            raise AccountReadError("unexpected_market")
        uta = self.settings.account_type == "uta"
        params = {"symbol": symbol, "limit": "100"}
        if fills:
            params.update(
                startTime=str(self.now - self.settings.history_days * DAY_MS), endTime=str(self.now)
            )
        if uta:
            params["category"] = "SPOT"
            if fills:
                # v3 fill history has no symbol parameter: it returns all SPOT
                # markets. Filter locally rather than treating other pairs as
                # corruption or incorrectly importing them into TRX history.
                params.pop("symbol")
            method = (
                self.client.privateUtaGetV3TradeFills
                if fills
                else self.client.privateUtaGetV3TradeUnfilledOrders
            )
        else:
            method = (
                self.client.privateSpotGetV2SpotTradeFills
                if fills
                else self.client.privateSpotGetV2SpotTradeUnfilledOrders
            )
        if uta and fills and self._uta_fills is not None:
            data = self._uta_fills
        else:
            data = self._read(method, params)
            if uta and fills:
                self._uta_fills = data
        rows = data["list"] if uta else data
        if not isinstance(rows, list):
            raise AccountReadError("invalid_exchange_response")
        normalize = normalize_fill if fills else normalize_order
        selected = rows
        if uta and fills:
            if any(
                row.get("category") != "SPOT"
                or not isinstance(row.get("symbol"), str)
                or not row["symbol"]
                for row in rows
            ):
                raise AccountReadError("unexpected_market")
            selected = [row for row in rows if row["symbol"] == symbol]
        normalized = [normalize(row, uta, symbol) for row in selected]
        key = "trade_id" if fills else "order_id"
        if len({row[key] for row in normalized}) != len(normalized):
            raise AccountReadError("duplicate_record")
        return {
            "symbol": symbol,
            "base_coin": MARKETS[symbol],
            "rows": normalized,
            "limited": len(rows) >= 100 or bool(data.get("cursor")) if uta else len(rows) >= 100,
            "since": self.now - self.settings.history_days * DAY_MS if fills else None,
            "until": self.now,
            "maximum_rows": 100,
        }

    def savings(self):
        # Bitget limits assets queries by order creation time to three months.
        # Report that coverage explicitly; old positions must not silently become 0.
        params = {
            "periodType": "flexible",
            "limit": "100",
            "startTime": str(self.now - 90 * DAY_MS),
            "endTime": str(self.now),
        }
        result, seen = [], set()
        for rows in self._pages(self.client.privateEarnGetV2EarnSavingsAssets, params):
            for row in rows:
                value = normalize_savings(row)
                if value is None:
                    continue
                key = value["order_id"]
                if key in seen:
                    raise AccountReadError("duplicate_record")
                seen.add(key)
                result.append(value)
        return {
            "rows": result,
            "since": self.now - 90 * DAY_MS,
            "until": self.now,
            "complete_account_history": False,
            "pages_complete": True,
            "source": "savings_v2",
            "account_location": "unverified",
        }

    def savings_records(self):
        result = {}
        since = self.now - self.settings.history_days * DAY_MS
        for coin in COINS:
            # The exchange can default to subscriptions when orderType is absent.
            for kind in ("subscribe", "redeem", "pay_interest"):
                params = {
                    "coin": coin,
                    "periodType": "flexible",
                    "orderType": kind,
                    "limit": "100",
                    "startTime": str(since),
                    "endTime": str(self.now),
                }
                for rows in self._pages(self.client.privateEarnGetV2EarnSavingsRecords, params):
                    for row in rows:
                        value = normalize_savings_record(row)
                        if value is None:
                            continue
                        if value["coin"] != coin or not since <= value["timestamp"] <= self.now:
                            raise AccountReadError("unexpected_savings_record")
                        key = value["event_id"]
                        if key in result and result[key] != value:
                            raise AccountReadError("conflicting_record")
                        result[key] = value
        return {
            "rows": sorted(result.values(), key=lambda row: (row["timestamp"], row["event_id"])),
            "since": since,
            "until": self.now,
            "pages_complete": True,
            "source": "savings_v2",
            "credited_to_strategy": False,
        }

    def funding_records(self):
        result = {}
        since = self.now - self.settings.history_days * DAY_MS
        for kind in FUNDING_SAVINGS_TYPES:
            params = {
                "type": kind,
                "limit": "100",
                "startTime": str(since),
                "endTime": str(self.now),
            }
            for rows in self._pages(
                self.client.privateUtaGetV3AccountFundingFinancialRecords,
                params,
                "list",
                "cursor",
                "cursor",
            ):
                for row in rows:
                    value = normalize_funding_record(row, kind)
                    if value is None:
                        continue
                    if not since <= value["timestamp"] <= self.now:
                        raise AccountReadError("unexpected_savings_record")
                    key = value["record_id"]
                    if key in result and result[key] != value:
                        raise AccountReadError("conflicting_record")
                    result[key] = value
        return {
            "rows": sorted(result.values(), key=lambda row: (row["timestamp"], row["record_id"])),
            "since": since,
            "until": self.now,
            "pages_complete": True,
            "redemption_arrival_verified": False,
        }

    def products(self):
        result = []
        coin_checks = {}
        for coin in ("TRX", "USDGO", "USDT"):
            try:
                rows = self._read(
                    self.client.privateEarnGetV2EarnSavingsProduct,
                    # Include unheld products as well as held/delisted products. A list
                    # entry is information, not proof of subscription eligibility.
                    {"coin": coin, "filter": "all"},
                )
                if not isinstance(rows, list):
                    raise AccountReadError("invalid_exchange_response")
            except Exception as exc:
                coin_checks[coin] = {"status": "error", "error": failure_code(exc)}
                continue
            coin_checks[coin] = {"status": "success", "error": None}
            for row in rows:
                if row["coin"] != coin or row["periodType"] != "flexible":
                    continue
                result.append(
                    {
                        "coin": coin,
                        "product_id": text_id(row["productId"]),
                        "status": str(row["status"])[:40],
                        "level": str(row.get("productLevel", "unknown"))[:40],
                        "tiers": [
                            {
                                "minimum": decimal_text(tier["minStepVal"]),
                                "maximum": decimal_text(tier["maxStepVal"]),
                                "apr_percent": decimal_text(tier["currentApy"]),
                            }
                            for tier in row["apyList"]
                        ],
                    }
                )
        if len({row["product_id"] for row in result}) != len(result):
            raise AccountReadError("duplicate_record")
        for index, row in enumerate(result):
            row.update(terms=None, terms_error=None, terms_status="not_requested")
            if index >= 12:
                continue
            try:
                data = self._read(
                    self.client.privateEarnGetV2EarnSavingsSubscribeInfo,
                    {"productId": row["product_id"], "periodType": "flexible"},
                )
                row.update(terms=normalize_product_terms(data), terms_status="success")
            except Exception as exc:
                row.update(terms_error=failure_code(exc), terms_status="error")
        return {
            "rows": result,
            "quota_verified": False,
            "partial": any(row["terms_status"] != "success" for row in result)
            or any(check["status"] != "success" for check in coin_checks.values()),
            "coin_checks": coin_checks,
            "maximum_term_reads": 12,
        }


class AccountService:
    """Explicit refresh only; no startup polling, orders or background activation."""

    def __init__(self, user_dir: Path, client_factory=ReadOnlyBitget, clock=time.time):
        self.config_path = user_dir / "trx_phase1" / CONFIG_NAME
        self.client_factory, self.clock = client_factory, clock
        self._refresh_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._identity = None
        self._snapshot = None
        self._checks = {}
        self._attempt = None
        self._error = None
        self._account_id = None
        self._history_error = None
        self.history = AccountHistory(user_dir / "trx_phase1" / "account-observations.sqlite")

    def _configuration(self):
        settings, source, complete = load_settings(self.config_path)
        identity = hashlib.sha256(settings.model_dump_json().encode()).hexdigest()
        # SecretStr intentionally masks serialization, so include secrets only in
        # this private cache identifier. It is never returned or persisted.
        identity += hashlib.sha256(
            "\0".join(
                value.get_secret_value()
                for value in (settings.api_key, settings.secret_key, settings.passphrase)
            ).encode()
        ).hexdigest()
        with self._state_lock:
            if identity != self._identity:
                self._identity = identity
                self._snapshot, self._checks = None, {}
                self._attempt, self._error = None, None
                self._account_id, self._history_error = None, None
        return settings, source, complete

    def status(self):
        with self._state_lock:
            base = {
                "read_only": True,
                "trading_enabled": False,
                "earn_write_enabled": False,
                "config_file": "user_data/trx_phase1/" + CONFIG_NAME,
                "credential_source": "none",
                "credentials_configured": False,
                "account_type": None,
                "include_earn": False,
                "history_days": None,
                "snapshot": None,
                "checks": {},
                "last_attempt_at": None,
                "status": "not_configured",
                "error": None,
                "stale": False,
                "strategy_allocation": "unassigned",
                "history": {"points": [], "total_points": 0, "truncated": False, "error": None},
                "earn_access": {
                    "route": "savings_v2_compatibility",
                    "status": "pending",
                    "account_mapping": "unverified",
                    "redemption_arrival": "unverified",
                },
            }
            try:
                settings, source, complete = self._configuration()
            except AccountReadError as exc:
                self._snapshot, self._checks = None, {}
                self._identity = None
                return {**base, "status": "invalid_configuration", "error": str(exc)}
            base.update(
                account_type=settings.account_type,
                include_earn=settings.include_earn,
                history_days=settings.history_days,
                credential_source=source,
                credentials_configured=complete,
            )
            if not settings.enabled:
                return {**base, "status": "disabled"}
            if not settings.include_earn:
                base["earn_access"]["status"] = "disabled"
            if not complete:
                return base
            base.update(
                status="ready",
                snapshot=deepcopy(self._snapshot),
                checks=deepcopy(self._checks),
                last_attempt_at=self._attempt,
                error=self._error,
            )
            try:
                base["history"] = self.history.read(self._account_id, settings.account_type)
                base["history"]["error"] = self._history_error
            except Exception:
                base["history"]["error"] = "history_storage_unavailable"
            if self._snapshot:
                stale = self.clock() * 1000 - self._snapshot["observed_at"] > 60000
                base.update(
                    stale=stale or bool(self._error),
                    status="stale"
                    if stale
                    else "partial"
                    if any(
                        row["status"] in ("error", "partial", "unsupported")
                        for row in self._checks.values()
                    )
                    else "connected",
                )
                if settings.include_earn:
                    states = [
                        self._checks.get(name, {}).get("status")
                        for name in ("savings", "products", "savings_records")
                    ]
                    base["earn_access"]["status"] = (
                        "read_confirmed"
                        if all(state == "success" for state in states)
                        else "partial"
                        if any(state in ("success", "partial") for state in states)
                        else "failed"
                    )
            if self._error:
                base["status"] = "error"
            return base

    def allocation_context(self):
        """Return the same verified snapshot and UID binding for server-side preflight."""
        with self._state_lock:
            return self.status(), self._account_id

    def refresh(self):
        if not self._refresh_lock.acquire(blocking=False):
            raise AccountBusy("refresh_in_progress")
        client = None
        identity = None
        try:
            settings, _, complete = self._configuration()
            identity = self._identity
            if not complete or not settings.enabled:
                return self.status()
            now = int(self.clock() * 1000)
            with self._state_lock:
                if self._attempt and now - self._attempt < 5000:
                    raise AccountBusy("refresh_too_frequent")
                self._attempt, self._error, self._checks = now, None, {}
                # No stale section is ever mixed into the new snapshot.
            client = self.client_factory(
                {
                    "apiKey": settings.api_key.get_secret_value(),
                    "secret": settings.secret_key.get_secret_value(),
                    "password": settings.passphrase.get_secret_value(),
                    "enableRateLimit": True,
                    "timeout": 8000,
                    "verbose": False,
                    "options": {"defaultType": "spot", "adjustForTimeDifference": True},
                }
            )
            client._shared_limiter = RequestLimiter(
                self.config_path.parent / "request-limits.sqlite"
            )
            client.load_time_difference()
            uta = settings.account_type == "uta"
            info = response_data(
                client.privateUtaGetV3AccountInfo()
                if uta
                else client.privateSpotGetV2SpotAccountInfo()
            )
            snapshot = {
                "observed_at": now,
                "completed_at": None,
                "account": {
                    "uid_masked": mask_uid(info["userId"]),
                    "permission_type": info.get("permType")
                    if info.get("permType") in ("read-only", "read-and-write")
                    else "unknown",
                },
            }
            reader = AccountReader(client, settings, now, lambda: int(self.clock() * 1000))
            data, checks = reader.collect()
            snapshot.update(data)
            snapshot["completed_at"] = int(self.clock() * 1000)
            self._configuration()
            with self._state_lock:
                if self._identity == identity:
                    self._snapshot, self._checks = snapshot, checks
                    self._account_id = hashlib.sha256(text_id(info["userId"]).encode()).hexdigest()
                    self._record_observation(settings.account_type, snapshot, now)
        except AccountBusy:
            raise
        except Exception as exc:
            with self._state_lock:
                self._error = failure_code(exc)
                if self._account_id and self._identity == identity:
                    self._record_observation(settings.account_type, None, now)
        finally:
            with suppress(Exception):
                if client is not None:
                    client.close()
            self._refresh_lock.release()
        return self.status()

    def _record_observation(self, account_type, snapshot, timestamp):
        try:
            self.history.append(
                self._account_id, account_type, observation(snapshot, account_type, timestamp)
            )
            self._history_error = None
        except Exception:
            self._history_error = "history_storage_unavailable"
