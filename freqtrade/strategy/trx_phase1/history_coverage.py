"""Public Bitget spot history with raw provenance and explicit gap inventory."""

import gzip
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from freqtrade.strategy.trx_phase1.candles import data_fingerprint, validate_minutes


MINUTE = 60000
ENDPOINT = "https://api.bitget.com/api/v2/spot/market/history-candles"
SOURCE = {"exchange": "bitget", "market": "spot", "pair": "TRX/USDT", "timeframe": "1m"}


def frame_from_rows(rows):
    frame = pd.DataFrame(
        [[int(row[0]), *map(float, row[1:6])] for row in rows],
        columns=["date", "open", "high", "low", "close", "volume"],
    )
    frame["date"] = pd.to_datetime(frame["date"], unit="ms", utc=True)
    return frame.sort_values("date").reset_index(drop=True)


def coverage(frame):
    """Inventory discontinuities; never synthesize a missing candle."""
    if frame.empty:
        return {"rows": 0, "intervals": [], "gaps": []}
    frame = frame.sort_values("date").reset_index(drop=True)
    stamps = pd.DatetimeIndex(frame.date).as_unit("ms").asi8
    if frame.date.duplicated().any() or (stamps % MINUTE).any():
        raise ValueError("History contains duplicate or unaligned minutes")
    numeric = frame[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric.to_numpy()).all() or (
        (frame.high < frame[["open", "low", "close"]].max(axis=1)).any()
        or (frame.low > frame[["open", "high", "close"]].min(axis=1)).any()
        or (frame[["open", "high", "low", "close"]] <= 0).any().any()
        or (frame.volume < 0).any()
    ):
        raise ValueError("History contains invalid actual OHLCV")
    breaks = np.flatnonzero(np.diff(stamps) != MINUTE) + 1
    bounds = np.r_[0, breaks, len(frame)]
    intervals = [
        {
            "start_utc": str(frame.date.iloc[a]),
            "end_utc": str(frame.date.iloc[b - 1]),
            "start_ms": int(stamps[a]),
            "end_ms": int(stamps[b - 1]),
            "rows": int(b - a),
        }
        for a, b in pairwise(bounds)
    ]
    gaps = [
        {
            "start_utc": str(pd.Timestamp(int(stamps[index - 1] + MINUTE), unit="ms", tz="UTC")),
            "end_utc": str(pd.Timestamp(int(stamps[index] - MINUTE), unit="ms", tz="UTC")),
            "missing_minutes": int((stamps[index] - stamps[index - 1]) // MINUTE - 1),
        }
        for index in breaks
    ]
    return {
        "rows": len(frame),
        "earliest_utc": str(frame.date.iloc[0]),
        "latest_utc": str(frame.date.iloc[-1]),
        "duplicates": 0,
        "invalid_ohlcv": 0,
        "synthetic_minutes": 0,
        "intervals": intervals,
        "gaps": gaps,
        "missing_minutes": sum(row["missing_minutes"] for row in gaps),
    }


def select_continuous_interval(report, original_start, original_end):
    begin = int(pd.Timestamp(original_start).timestamp() * 1000)
    end = int(pd.Timestamp(original_end).timestamp() * 1000)
    choices = [
        row for row in report["intervals"] if row["start_ms"] <= begin <= end <= row["end_ms"]
    ]
    if len(choices) != 1 or choices[0]["start_ms"] >= begin:
        raise ValueError("No earlier continuous same-market component contains the original year")
    return {
        **choices[0],
        "end_ms": end,
        "end_utc": str(pd.Timestamp(end, unit="ms", tz="UTC")),
        "rows": (end - choices[0]["start_ms"]) // MINUTE + 1,
    }


def download_history(directory, original_file, launch_at, proxy=None, workers=12):  # noqa: C901
    """Collect all available minutes from official launch, keeping the original year intact."""
    directory = Path(directory)
    cache = directory / "raw_history"
    cache.mkdir(parents=True, exist_ok=True)
    original = validate_minutes(pd.read_feather(original_file)).reset_index(drop=True)
    end = int(original.date.iloc[0].timestamp() * 1000)
    begin = int(launch_at)
    if begin % MINUTE or begin >= end:
        raise ValueError("Invalid official launch boundary")
    total = math.ceil((end - begin) / (200 * MINUTE))
    plan = {
        **SOURCE,
        "endpoint": ENDPOINT,
        "start_ms": begin,
        "end_exclusive_ms": end,
        "blocks": total,
        "workers": workers,
        "max_requests_per_second": 1 / 0.06,
        "original_file_sha256": hashlib.sha256(Path(original_file).read_bytes()).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
        "selection": (
            "Earliest contiguous component containing the entire saved original year; "
            "coverage only, before strategy results."
        ),
    }
    plan_file = directory / "acquisition-plan.json"
    if plan_file.exists():
        saved = json.loads(plan_file.read_text())
        for key in (*SOURCE, "start_ms", "end_exclusive_ms", "original_file_sha256"):
            if plan[key] != saved[key]:
                raise ValueError("Acquisition resume differs from its original declaration")
    else:
        plan_file.write_text(json.dumps(plan, indent=2))
    gate, local = threading.Lock(), threading.local()
    next_request = 0.0

    def request_slot():
        nonlocal next_request
        with gate:
            delay = next_request - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_request = time.monotonic() + 0.06

    def fetch(index):
        left, right = begin + index * 200 * MINUTE, min(end, begin + (index + 1) * 200 * MINUTE)
        path = cache / f"{left}-{right}.json.gz"
        if path.exists():
            return path, True
        if not hasattr(local, "session"):
            local.session = requests.Session()
            if proxy:
                local.session.proxies.update(https=proxy)
        last = None
        for attempt in range(6):
            try:
                request_slot()
                response = local.session.get(
                    ENDPOINT,
                    params={
                        "symbol": "TRXUSDT",
                        "granularity": "1min",
                        "endTime": right,
                        "limit": (right - left) // MINUTE,
                    },
                    timeout=25,
                )
                response.raise_for_status()
                raw = response.json()
                if raw.get("code") != "00000":
                    raise ValueError(f"History response code {raw.get('code')}")
                rows = [row for row in raw.get("data", []) if left <= int(row[0]) < right]
                coverage(frame_from_rows(rows))
                payload = {
                    **SOURCE,
                    "url": response.url,
                    "fetched_at_utc": datetime.now(UTC).isoformat(),
                    "requestTime": raw.get("requestTime"),
                    "requested_start_ms": left,
                    "requested_end_ms": right,
                    "data": rows,
                }
                temporary = path.with_suffix(".tmp")
                with gzip.open(temporary, "wt") as stream:
                    json.dump(payload, stream)
                temporary.replace(path)
                return path, False
            except (requests.RequestException, ValueError, KeyError) as exc:
                last = type(exc).__name__
                if attempt < 5:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Public history block {left}-{right} failed: {last}")

    started, completed, reused = time.monotonic(), 0, 0
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch, index): index for index in range(total)}
        for future in as_completed(pending):
            try:
                _, existing = future.result()
                reused += existing
            except Exception as exc:
                errors.append({"block_index": pending[future], "error": str(exc)})
            completed += 1
            if completed % 500 == 0 or completed == total:
                progress = {
                    "completed": completed,
                    "total": total,
                    "cached": reused,
                    "failed": len(errors),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
                (directory / "acquisition-progress.json").write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
    if errors:
        (directory / "acquisition-errors.json").write_text(json.dumps(errors, indent=2))
        raise ValueError("History acquisition incomplete; preserve raw cache and resume")
    chunks = []
    for index in range(total):
        left, right = begin + index * 200 * MINUTE, min(end, begin + (index + 1) * 200 * MINUTE)
        with gzip.open(cache / f"{left}-{right}.json.gz", "rt") as stream:
            saved = json.load(stream)
        if any(saved[key] != value for key, value in SOURCE.items()):
            raise ValueError("Historical source changed exchange, market, pair or timeframe")
        chunks.append(frame_from_rows(saved["data"]))
    full = (
        pd.concat([*chunks, original], ignore_index=True).sort_values("date").reset_index(drop=True)
    )
    report = coverage(full)
    selected = select_continuous_interval(report, original.date.iloc[0], original.date.iloc[-1])
    actual = full.loc[
        (full.date >= pd.Timestamp(selected["start_utc"]))
        & (full.date <= pd.Timestamp(selected["end_utc"]))
    ].reset_index(drop=True)
    validate_minutes(actual)
    overlap = actual.loc[actual.date >= original.date.iloc[0]].reset_index(drop=True)
    if data_fingerprint(overlap) != data_fingerprint(original):
        raise ValueError("Acquisition altered the pinned original control year")
    path = directory / "inputs/continuous-1m.feather"
    actual.to_feather(path)
    report.update(
        source=SOURCE,
        endpoint=ENDPOINT,
        raw_blocks=total,
        selected=selected,
        selected_data_fingerprint=data_fingerprint(actual),
        selected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_control_preserved=True,
        completed_at=datetime.now(UTC).isoformat(),
    )
    (directory / "data-coverage.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({"complete": True, "selected": selected, "gaps": len(report["gaps"])}),
        flush=True,
    )
    return report
