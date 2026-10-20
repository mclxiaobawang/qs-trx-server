"""Public-only Bitget market access for prospective virtual observations."""

import time

import ccxt
import numpy as np
import pandas as pd

from freqtrade.strategy.trx_phase1.candles import append_closed_minutes, validate_minutes
from freqtrade.strategy.trx_phase1.history_coverage import frame_from_rows
from freqtrade.strategy.trx_phase1.request_limits import LimitedReadMixin


PUBLIC_PATHS = {
    "v2/public/time",
    "v2/spot/market/history-candles",
    "v2/spot/market/candles",
    "v2/spot/market/orderbook",
    "v2/spot/public/symbols",
}


class PublicBitget(LimitedReadMixin, ccxt.bitget):
    def request(
        self, path, api="public", method="GET", params=None, headers=None, body=None, config=None
    ):
        public = api == "public" or (isinstance(api, list) and api and api[0] == "public")
        if method != "GET" or path not in PUBLIC_PATHS or not public:
            raise PermissionError("Shadow market transport is public GET only")
        return super().request(path, api, method, params or {}, headers, body, config or {})

    def log(self, *args):
        pass


class PublicShadowMarket:
    def __init__(self, *, limiter=None):
        self.client = PublicBitget({"enableRateLimit": True, "timeout": 8000})
        if limiter is not None:
            self.client._shared_limiter = limiter
        self.client.request_priority = 2
        self.provisional = {}

    def close(self):
        self.client.close()

    @staticmethod
    def data(response):
        if response.get("code") != "00000":
            raise ValueError("public_market_response_failed")
        return response["data"]

    def snapshot(self):
        server = int(self.client.fetch_time())
        local = int(time.time() * 1000)
        if abs(local - server) > 30000:
            raise ValueError("market_clock_mismatch")
        raw = self.data(
            self.client.publicSpotGetV2SpotMarketOrderbook(
                {"symbol": "TRXUSDT", "type": "step0", "limit": "20"}
            )
        )
        quote = {
            "timestamp": int(raw["ts"]),
            "asks": [[float(p), float(q)] for p, q in raw["asks"]],
            "bids": [[float(p), float(q)] for p, q in raw["bids"]],
        }
        now = int(time.time() * 1000)
        validate_quote(quote, now)
        return now, quote

    def rules(self):
        rows = self.data(self.client.publicSpotGetV2SpotPublicSymbols({"symbol": "TRXUSDT"}))
        matches = [row for row in rows if row["symbol"] == "TRXUSDT"]
        if len(matches) != 1 or matches[0]["status"] != "online":
            raise ValueError("market_unavailable")
        row = matches[0]
        result = {
            "quantity_decimals": int(row["quantityPrecision"]),
            # Bitget documents minTradeAmount as deprecated; the notional minimum
            # and quantity precision are the current public constraints.
            "min_quantity": 0.0,
            "min_notional": float(row["minTradeUSDT"]),
            "observed_at": int(time.time() * 1000),
        }
        if not 0 <= result["quantity_decimals"] <= 18 or not all(
            np.isfinite(result[key]) and result[key] >= 0
            for key in ("min_quantity", "min_notional")
        ):
            raise ValueError("market_rules_unverified")
        return result

    def update(self, existing, now):
        last = int(pd.Timestamp(existing.date.iloc[-1]).timestamp() * 1000)
        if now - last > 30 * 86400000:
            raise ValueError("shadow_history_too_old")
        if last + 60000 >= (now - 15000) // 60000 * 60000:
            return existing
        rows, end = [], now
        for page_number in range(250):
            fetch = (
                self.client.publicSpotGetV2SpotMarketCandles
                if page_number == 0
                else self.client.publicSpotGetV2SpotMarketHistoryCandles
            )
            page = self.data(
                fetch(
                    {
                        "symbol": "TRXUSDT",
                        "granularity": "1min",
                        "endTime": str(end),
                        "limit": "200",
                    }
                )
            )
            if not page:
                raise ValueError("public_history_incomplete")
            rows.extend(page)
            earliest = min(int(row[0]) for row in page)
            if earliest <= last:
                incoming = frame_from_rows(rows)
                duplicates = incoming[incoming.duplicated("date", keep=False)]
                if len(duplicates) and (duplicates.groupby("date").nunique() > 1).any().any():
                    raise ValueError("conflicting_history_pages")
                incoming = incoming.drop_duplicates("date").reset_index(drop=True)
                incoming = self.confirm_recent(incoming, last, now)
                return append_closed_minutes(
                    existing, incoming, pd.Timestamp(now, unit="ms", tz="UTC")
                )
            if earliest >= end:
                raise ValueError("public_history_cursor_stalled")
            end = earliest
        raise ValueError("public_history_read_limit")

    def confirm_recent(self, incoming, last, now):
        """The latest closed bars remain provisional until two spaced reads agree.

        This operational confirmation is independent of the frozen strategy's
        60/240/1440m signal periods. Older data still undergo strict overlap checks.
        """
        values = incoming[["open", "high", "low", "close", "volume"]].to_numpy()
        stamps = pd.DatetimeIndex(incoming.date).as_unit("ms").asi8
        limit = (now - 15000) // 60000 * 60000 - 60000
        for index, timestamp in enumerate(stamps):
            if timestamp <= last or timestamp + 60000 <= now - 120000:
                continue
            previous = self.provisional.get(int(timestamp))
            same = previous is not None and np.array_equal(previous[1], values[index])
            if not same:
                self.provisional[int(timestamp)] = (now, values[index].copy())
            if not same or now - previous[0] < 10000:
                limit = min(limit, int(timestamp) - 60000)
        self.provisional = {
            stamp: value
            for stamp, value in self.provisional.items()
            if stamp > last and stamp + 60000 > now - 180000
        }
        return incoming.loc[stamps <= limit].reset_index(drop=True)


def validate_quote(quote, now):
    if now - quote["timestamp"] > 10000 or quote["timestamp"] > now + 2000:
        raise ValueError("stale_execution_quote")
    for side in ("asks", "bids"):
        if not quote[side] or not all(
            np.isfinite(p) and np.isfinite(q) and p > 0 and q > 0 for p, q in quote[side]
        ):
            raise ValueError("invalid_orderbook")
    if quote["bids"][0][0] > quote["asks"][0][0]:
        raise ValueError("crossed_orderbook")
    if any(a[0] > b[0] for a, b in zip(quote["asks"], quote["asks"][1:], strict=False)) or any(
        a[0] < b[0] for a, b in zip(quote["bids"], quote["bids"][1:], strict=False)
    ):
        raise ValueError("unsorted_orderbook")


def seed_history(path, now, lookback_days):
    data = pd.read_feather(path)
    cutoff = pd.Timestamp(now, unit="ms", tz="UTC")
    data = data.loc[
        (data.date >= cutoff.floor("D") - pd.Timedelta(days=lookback_days + 3))
        & (data.date + pd.Timedelta("1min") <= cutoff)
    ].reset_index(drop=True)
    return validate_minutes(data)
