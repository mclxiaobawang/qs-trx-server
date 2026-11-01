"""Causal, daily maximum-mass price bands; no account access or state writes."""

import hashlib

import numpy as np
import pandas as pd

from .live_broker import ExecutionError


DAY = 86400000
OFFSET = 480 * 60000
SCALE = 1000000


class BandHistory:
    def __init__(self, first_open_at, ticks):
        self.first = int(first_open_at)
        self.ticks = np.asarray(ticks)
        if (
            self.first % 60000
            or self.ticks.ndim != 1
            or not len(self.ticks)
            or self.ticks.dtype != np.dtype("int64")
            or (self.ticks <= 0).any()
            or self.ticks.max() > np.iinfo(np.int64).max // 104000
        ):
            raise ExecutionError("range_history_invalid")
        self.cached = None

    @classmethod
    def load(cls, path, expected_sha256, first, rows):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ExecutionError("range_history_fingerprint_mismatch")
        with np.load(path, allow_pickle=False) as data:
            history = cls(int(data["first_open_at"].item()), data["close_ticks"])
        if history.first != first or len(history.ticks) != rows:
            raise ExecutionError("range_history_scope_mismatch")
        return history

    def at(self, timestamp, recent):
        cutoff = ((int(timestamp) + OFFSET) // DAY) * DAY - OFFSET
        if self.cached is not None and self.cached["effective_at"] == cutoff:
            return dict(self.cached)
        stamps = pd.DatetimeIndex(recent.date).as_unit("ms").asi8
        known = stamps + 60000 <= cutoff
        stamps = stamps[known]
        prices = recent.close.to_numpy(dtype=float)[known]
        rounded = np.rint(prices * SCALE)
        if (
            not len(stamps)
            or (stamps % 60000).any()
            or (np.diff(stamps) != 60000).any()
            or not np.isfinite(prices).all()
            or (prices <= 0).any()
            or (np.abs(rounded / SCALE - prices) > 1e-12).any()
        ):
            raise ExecutionError("range_history_incomplete")
        ticks = rounded.astype(np.int64)
        seed_end = self.first + (len(self.ticks) - 1) * 60000
        overlap = (stamps >= self.first) & (stamps <= seed_end)
        if not np.array_equal(ticks[overlap], self.ticks[(stamps[overlap] - self.first) // 60000]):
            raise ExecutionError("range_history_conflict")
        count = min(len(self.ticks), (cutoff - self.first) // 60000)
        if count <= 0:
            raise ExecutionError("range_history_incomplete")
        combined = self.ticks[:count]
        if seed_end + 60000 < cutoff:
            added = stamps > seed_end
            if (
                not added.any()
                or stamps[added][0] != seed_end + 60000
                or stamps[added][-1] + 60000 != cutoff
            ):
                raise ExecutionError("range_history_incomplete")
            combined = np.r_[combined, ticks[added]]
        if self.first + len(combined) * 60000 != cutoff or len(combined) < 365 * 1440:
            raise ExecutionError("range_history_incomplete")
        close_times = self.first + (np.arange(len(combined)) + 1) * 60000
        weights = np.exp2((close_times - cutoff) / DAY / 90)
        values, inverse = np.unique(combined, return_inverse=True)
        masses = np.bincount(inverse, weights=weights, minlength=len(values))
        right = np.searchsorted(values * 100000, values * 104000, side="right") - 1
        cumulative = np.r_[0.0, np.cumsum(masses)]
        scores = cumulative[right + 1] - cumulative[:-1]
        left = int(np.argmax(scores))  # Sorted prices: ties choose the lowest lower bound.
        upper = int(right[left])
        self.cached = {
            "effective_at": cutoff,
            "half_life_days": 90,
            "width_cap_pct": 4,
            "lower": float(values[left] / SCALE),
            "upper": float(values[upper] / SCALE),
            "weighted_coverage": float(scores[left] / cumulative[-1]),
            "training_first_open_at": self.first,
            "latest_training_close_at": cutoff,
            "training_minutes": len(combined),
            "training_sha256": hashlib.sha256(combined.tobytes()).hexdigest(),
        }
        return dict(self.cached)
