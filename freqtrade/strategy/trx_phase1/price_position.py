"""Causal daily price references, separate from trading signals and account state."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from freqtrade.strategy.trx_phase1.candles import data_fingerprint, validate_minutes


DAY = 86_400_000
OFFSET = 480 * 60_000


@dataclass(frozen=True)
class PricePositionReference:
    lookback_days: int
    quantile: float
    source_start: str
    source_end: str
    source_fingerprint: str
    values: dict[int, float]

    def value_at(self, timestamp):
        return self.values.get((int(timestamp) + OFFSET) // DAY)

    def describe(self, start, end):
        rows = [
            {
                "effective_at": day * DAY - OFFSET,
                "history_start": (day - self.lookback_days) * DAY - OFFSET,
                "history_end_exclusive": day * DAY - OFFSET,
                "reference_price": value,
            }
            for day, value in sorted(self.values.items())
            if (int(start) + OFFSET) // DAY <= day <= (int(end) + OFFSET) // DAY
        ]
        return {
            "lookback_days": self.lookback_days,
            "quantile": self.quantile,
            "timezone": "Asia/Shanghai",
            "update": "Daily at 00:00 UTC+8 using only the previous complete calendar days.",
            "source_start_utc": self.source_start,
            "source_end_utc": self.source_end,
            "source_fingerprint": self.source_fingerprint,
            "daily_references": rows,
        }


def build_price_reference(frame, config):
    data = validate_minutes(frame)
    stamps = pd.DatetimeIndex(data.date).as_unit("ms").asi8
    closes = data.close.to_numpy()
    values = {}
    first_day = (int(stamps[0]) + OFFSET) // DAY
    # Include the final close: at midnight the preceding 23:59 candle is available.
    final_day = (int(stamps[-1]) + 60_000 + OFFSET) // DAY
    required = config.price_position_lookback_days * 1440
    for day in range(first_day, final_day + 1):
        end = day * DAY - OFFSET
        begin = end - config.price_position_lookback_days * DAY
        left, right = np.searchsorted(stamps, [begin, end])
        if right - left != required or left == len(stamps) or int(stamps[left]) != begin:
            continue
        if int(stamps[right - 1]) != end - 60_000:
            continue
        values[day] = float(np.quantile(closes[left:right], config.price_position_quantile))
    return PricePositionReference(
        config.price_position_lookback_days,
        config.price_position_quantile,
        str(data.date.iloc[0]),
        str(data.date.iloc[-1]),
        data_fingerprint(data),
        values,
    )
