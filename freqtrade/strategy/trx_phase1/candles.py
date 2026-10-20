import hashlib

import numpy as np
import pandas as pd
from pandas import DataFrame


def data_fingerprint(frame: DataFrame) -> str:
    canonical = frame[["open", "high", "low", "close", "volume"]].astype("float64").copy()
    canonical.insert(0, "date", pd.DatetimeIndex(frame["date"]).as_unit("ms").asi8)
    rows = pd.util.hash_pandas_object(canonical, index=False).to_numpy().astype("<u8")
    return hashlib.sha256(rows.tobytes()).hexdigest()


def validate_minutes(frame: DataFrame) -> DataFrame:
    data = frame[["date", "open", "high", "low", "close", "volume"]].copy()
    data["date"] = pd.to_datetime(data["date"], utc=True)
    dates = pd.DatetimeIndex(data["date"]).as_unit("ms").asi8
    if len(data) == 0 or len(set(dates)) != len(dates) or (dates % 60000).any():
        raise ValueError("Minute data must be nonempty, unique and aligned to UTC minutes")
    if len(dates) > 1 and not (np.diff(dates) == 60000).all():
        raise ValueError("Minute data contains gaps or is not sorted; trading must pause")
    numeric = data[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Missing or non-finite OHLCV")
    if (
        (data["high"] < data[["open", "low", "close"]].max(axis=1)).any()
        or (data["low"] > data[["open", "high", "close"]].min(axis=1)).any()
        or (data[["open", "high", "low", "close"]] <= 0).any().any()
        or (data["volume"] < 0).any()
    ):
        raise ValueError("Invalid OHLCV")
    return data


def aggregate_minutes(frame: DataFrame, minutes: int, asof=None) -> DataFrame:
    """Label by close time, retain only complete groups known at asof."""
    if minutes < 1 or 1440 % minutes:
        raise ValueError("Period must divide a UTC day")
    data = validate_minutes(frame)
    end = pd.Timestamp(asof) if asof is not None else data["date"].iloc[-1] + pd.Timedelta("1min")
    if end.tzinfo is None:
        raise ValueError("asof must be timezone aware")
    data = data.loc[data["date"] + pd.Timedelta("1min") <= end]
    groups = data.set_index("date").resample(
        f"{minutes}min", origin="epoch", closed="left", label="right"
    )
    result = groups.agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    result = result.loc[(groups.size() == minutes) & (result.index <= end)]
    return result.reset_index().rename(columns={"date": "close_time"})


def normalized_volatility(frame: DataFrame, window: int) -> pd.Series:
    """Hourly-equivalent realized volatility, always estimated on the 1m clock."""
    returns = np.log(frame["close"]).diff()
    return returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(60)


def append_closed_minutes(existing: DataFrame, incoming: DataFrame, now) -> DataFrame:
    """Merge exchange records without synthetic filling; reject contradictory finalized bars."""
    cutoff = pd.Timestamp(now)
    if cutoff.tzinfo is None:
        raise ValueError("now must be timezone aware")
    incoming = incoming.loc[incoming["date"] + pd.Timedelta("1min") <= cutoff]
    overlap = existing.merge(incoming, on="date", suffixes=("_old", "_new"))
    for column in ["open", "high", "low", "close", "volume"]:
        if not np.allclose(overlap[column + "_old"], overlap[column + "_new"], rtol=1e-9):
            raise ValueError("Exchange revised a finalized candle; reconcile before trading")
    merged = pd.concat([existing, incoming]).drop_duplicates("date").sort_values("date")
    return validate_minutes(merged.reset_index(drop=True))
