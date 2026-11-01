"""Read-only indicator explanations/history. Never imports or constructs a trading worker."""

import math
import sqlite3
import threading
import time
from array import array
from contextlib import closing
from pathlib import Path


if __package__:
    from .ledger import Ledger, money
else:
    from ledger import Ledger, money

MINUTE = 60000
DAY = 86400000
OFFSET = 480 * MINUTE


def indicator_snapshot(state, engine, book):
    """Explain a saved observation; counterfactual targets are not past schedule evidence."""
    cfg = engine["config"]
    observed = engine.get("last_close") or None
    value = engine.get("last_sigma")
    value = float(value) if observed and value is not None and math.isfinite(value) else None
    threshold = cfg.get("volatility_high")
    threshold = float(threshold) if threshold is not None else None
    high = value >= threshold if value is not None and threshold is not None else None
    price = engine.get("last_price")
    occupation = None
    if price and book.deployable > 0 and book.equity(price) > 0:
        occupation = min(
            1.0,
            max(
                float((book.principal + book.interest_trx) * money(price) / book.equity(price)),
                float(book.cost / book.deployable),
            ),
        )
    periods = cfg.get("periods", [])
    enabled = cfg.get("variant") in {
        "dynamic_period_only",
        "dynamic_threshold_period",
        "dynamic_quantity_period",
        "dynamic_period",
    }
    base = target = None
    if occupation is not None and periods and high is not None:
        base = min(len(periods) - 1, sum(occupation >= b for b in (0.25, 0.55, 0.8)))
        target = min(len(periods) - 1, base + int(high)) if enabled else 0
    return {
        "observed_at": observed,
        "window_minutes": cfg.get("volatility_window"),
        "hourly_volatility": value,
        "high_threshold": threshold,
        "high_volatility": high,
        "dynamic_periods": enabled,
        "occupation": occupation,
        "base_tier": base,
        "target_tier": target,
        "volatility_target_increment": target - base if enabled and target is not None else 0,
        "at_tier_cap": bool(enabled and high and base == len(periods) - 1),
        "scheduled_tier": engine.get("tier"),
        "buy_period": engine.get("buy_period"),
        "sell_period": engine.get("sell_period"),
        "relax_count": engine.get("relax_count"),
        "minimum_period_hold": cfg.get("minimum_period_hold"),
        "assessment_basis": "current_snapshot_target_not_historical_schedule_attribution",
    }


class IndicatorHistory:
    """Bounded reads of existing finalized candles; only in-memory caches are written."""

    def __init__(self, source, account_id):
        self.source = Path(source)
        self.account_id = account_id
        self.lock = threading.RLock()
        self.references = {}
        self.responses = {}

    def connect(self):
        db = sqlite3.connect(
            (self.source / "strategy.sqlite").resolve().as_uri() + "?mode=ro",
            uri=True,
            timeout=1,
        )
        db.execute("PRAGMA query_only=ON")
        return db

    def rows(self, begin, end):
        """Release each short SQLite read before processing, including on DELETE journals."""
        cursor = begin
        while cursor < end:
            for attempt in range(3):
                try:
                    with closing(self.connect()) as db:
                        batch = db.execute(
                            "SELECT timestamp,close FROM live_candles "
                            "WHERE timestamp>=? AND timestamp<? ORDER BY timestamp LIMIT 2048",
                            (cursor, end),
                        ).fetchall()
                    break
                except sqlite3.OperationalError as exc:
                    if attempt == 2 or not any(s in str(exc).lower() for s in ("locked", "busy")):
                        raise
                    time.sleep(0.05 * (attempt + 1))
            if not batch:
                return
            cursor = batch[-1][0] + MINUTE
            yield from batch  # No live database connection or read transaction is held here.

    def reference_values(self, requested_days, cfg):
        import numpy as np

        days, quantile = cfg["price_position_lookback_days"], cfg["price_position_quantile"]
        result = {day: self.references.get((day, days, quantile)) for day in requested_days}
        missing = [day for day in requested_days if result[day] is None]
        if not missing:
            return result
        # Read the union once, rather than rereading a full year for every displayed day.
        stamps, values = array("q"), array("d")
        for stamp, value in self.rows(
            (min(missing) - days) * DAY - OFFSET, max(missing) * DAY - OFFSET
        ):
            stamps.append(stamp)
            values.append(value)
        times = np.frombuffer(stamps, dtype="int64")
        prices = np.frombuffer(values, dtype="float64")
        for day in missing:
            end = day * DAY - OFFSET
            begin = end - days * DAY
            left, right = np.searchsorted(times, [begin, end])
            complete = right - left == days * 1440 and left < len(times)
            if complete:
                complete = (
                    times[left] == begin
                    and times[right - 1] == end - MINUTE
                    and np.all(np.diff(times[left:right]) == MINUTE)
                    and np.all(np.isfinite(prices[left:right]) & (prices[left:right] > 0))
                )
            if complete:
                value = float(np.quantile(prices[left:right], quantile))
                if len(self.references) >= 400:
                    self.references.pop(next(iter(self.references)))
                self.references[(day, days, quantile)] = result[day] = value
        # Missing windows are not cached: later synchronization may supply them.
        return result

    def query(self, state, execution, range_name="all", limit=1800, *, historical_volatility=False):
        if (
            state.get("account_id") != self.account_id
            or execution.get("account_id") != self.account_id
        ):
            raise ValueError("indicator_account_mismatch")
        engine = state["engine"]
        book = Ledger.restore(execution["book"])
        fx = state.get("markets", {}).get("USDGOUSDT", {}).get("bid")
        if fx:
            book.usdgo_mark = money(fx)
        current = indicator_snapshot(state, engine, book)
        if range_name == "summary":
            return {"account_id": self.account_id, "current": current}
        cfg = engine["config"]
        end = engine.get("last_close") or 0
        begin = (execution.get("performance_basis") or {}).get("started_at") or execution[
            "created_at"
        ]
        metadata = {}
        if historical_volatility:
            ranges = {
                "day": 1,
                "week": 7,
                "month": 30,
                "quarter": 90,
                "half_year": 180,
                "year": 365,
            }
            if range_name not in {*ranges, "all"}:
                raise ValueError("volatility_range_invalid")
            live_started = (
                (execution.get("performance_basis") or {}).get("started_at")
                or state.get("activated_at")
                or execution["created_at"]
            )
            with closing(self.connect()) as db:
                first = db.execute(
                    "SELECT MIN(timestamp) FROM live_candles WHERE timestamp<?", (end,)
                ).fetchone()[0]
            available_start = first + MINUTE if first is not None else None
            requested_start = (
                end - ranges[range_name] * DAY if range_name != "all" else available_start
            )
            begin = max(available_start, requested_start) if available_start is not None else end
            metadata = {
                "history_scope": "saved_market_candles",
                "requested_range": range_name,
                "requested_start": requested_start,
                "available_start": available_start,
                "live_started_at": live_started,
                "includes_pre_live_history": begin < live_started,
                "history_truncated": available_start is not None
                and requested_start < available_start,
                "reconstructed": True,
            }
        elif range_name != "all":
            if range_name not in {"day", "week"}:
                raise ValueError("indicator_range_requires_volatility_view")
            begin = max(begin, end - (DAY if range_name == "day" else 7 * DAY))
        key = (historical_volatility, range_name, begin, end, limit, state.get("source_sha256"))
        with self.lock:
            if key not in self.responses:
                result = self.build(
                    begin,
                    end,
                    cfg,
                    limit,
                    include_p75=not historical_volatility,
                    retain_price_extrema=historical_volatility,
                    live_started_at=metadata.get("live_started_at"),
                )
                result.update(
                    account_id=self.account_id, source_sha256=state.get("source_sha256"), **metadata
                )
                if len(self.responses) >= 6:
                    self.responses.pop(next(iter(self.responses)))
                self.responses[key] = result
            return {**self.responses[key], "current": current}

    def build(  # noqa: C901 - paired extrema and gap sampling share one bounded pass
        self,
        begin,
        end,
        cfg,
        limit,
        *,
        include_p75=True,
        retain_price_extrema=False,
        live_started_at=None,
    ):
        import numpy as np

        window = int(cfg["volatility_window"])
        stamps, closes = array("q"), array("d")
        for stamp, close in self.rows(begin - (window + 1) * MINUTE, end):
            stamps.append(stamp + MINUTE)
            closes.append(close)
        times = np.frombuffer(stamps, dtype="int64")
        prices = np.frombuffer(closes, dtype="float64")
        volatility = np.full(len(times), np.nan)
        if len(times) > window:
            valid = np.isfinite(prices) & (prices > 0)
            returns = np.diff(np.log(np.where(valid, prices, 1)))
            good = valid[1:] & valid[:-1] & (np.diff(times) == MINUTE)
            returns = np.where(good, returns, 0)
            sums = np.r_[0, np.cumsum(returns)]
            squares = np.r_[0, np.cumsum(returns * returns)]
            counts = np.r_[0, np.cumsum(good)]
            mean = (sums[window:] - sums[:-window]) / window
            variance = (squares[window:] - squares[:-window]) / window - mean * mean
            complete = counts[window:] - counts[:-window] == window
            volatility[window:] = np.where(
                complete, np.sqrt(np.maximum(0, variance)) * np.sqrt(60), np.nan
            )
        indices = np.flatnonzero((times >= begin) & (times <= end))
        daily = []
        if include_p75 and end >= begin:
            days = list(range((begin + OFFSET) // DAY, (end + OFFSET) // DAY + 1))
            references = self.reference_values(days, cfg)
            for day in days:
                effective = int(day * DAY - OFFSET)
                daily.append(
                    {
                        "timestamp": max(begin, effective),
                        "effective_at": effective,
                        "value": references[day],
                    }
                )
        selected = set()
        # Retain extrema and gaps in each bucket, so high-volatility spikes are not averaged away.
        bucket_count = max(1, (limit - 2) // 8) if retain_price_extrema else max(1, limit // 6)
        for bucket in np.array_split(indices, bucket_count):
            if not len(bucket):
                continue
            selected.update((int(bucket[0]), int(bucket[-1])))
            finite = bucket[np.isfinite(volatility[bucket])]
            if len(finite):
                selected.add(int(finite[np.argmin(volatility[finite])]))
                selected.add(int(finite[np.argmax(volatility[finite])]))
            missing = bucket[~np.isfinite(volatility[bucket])]
            if len(missing):
                selected.add(int(missing[0]))
                if retain_price_extrema:
                    selected.add(int(missing[-1]))
            if retain_price_extrema:
                valid_prices = bucket[np.isfinite(prices[bucket]) & (prices[bucket] > 0)]
                if len(valid_prices):
                    selected.add(int(valid_prices[np.argmin(prices[valid_prices])]))
                    selected.add(int(valid_prices[np.argmax(prices[valid_prices])]))
            else:
                gaps = bucket[np.r_[False, np.diff(times[bucket]) != MINUTE]]
                if len(gaps):
                    selected.add(int(gaps[0]))
        if retain_price_extrema and live_started_at is not None and len(indices):
            edge = int(np.searchsorted(times, live_started_at))
            selected.update(i for i in [edge - 1, edge] if indices[0] <= i <= indices[-1])
        gaps = (
            np.r_[
                0,
                np.cumsum(
                    (np.diff(times) != MINUTE)
                    | ~np.isfinite(prices[1:])
                    | ~np.isfinite(prices[:-1])
                    | (prices[1:] <= 0)
                    | (prices[:-1] <= 0)
                ),
            ]
            if len(times)
            else np.array([], dtype="int64")
        )
        points = []
        previous = None
        for i in sorted(selected):
            points.append(
                {
                    "timestamp": int(times[i]),
                    "price": float(prices[i])
                    if math.isfinite(prices[i]) and prices[i] > 0
                    else None,
                    "volatility": float(volatility[i]) if math.isfinite(volatility[i]) else None,
                    **(
                        {"break_before": bool(previous is not None and gaps[i] > gaps[previous])}
                        if retain_price_extrema
                        else {}
                    ),
                }
            )
            previous = i
        return {
            "start": begin,
            "end": end,
            "observed_at": end or None,
            "window_minutes": window,
            "high_threshold": cfg.get("volatility_high"),
            "lookback_days": cfg["price_position_lookback_days"],
            "quantile": cfg["price_position_quantile"],
            "timezone": "Asia/Shanghai",
            "points": points,
            "p75": daily,
            "source_points": len(indices),
            "missing_p75_days": sum(p["value"] is None for p in daily),
            "missing_volatility_points": int(np.count_nonzero(~np.isfinite(volatility[indices]))),
            "basis": "existing_local_finalized_candles_reconstructed_not_trade_decision_history",
            "sampling": (
                "bucket_first_last_price_and_volatility_min_max_missing_and_live_boundary"
                if retain_price_extrema
                else "bucket_first_last_volatility_min_max_and_missing; p75_daily_unsampled"
            ),
        }
