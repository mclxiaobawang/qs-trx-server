"""Archive persisted daily price bands for display; financial sources remain read-only."""

import hashlib
import json
import math
import sqlite3
import threading
import time
from contextlib import closing
from itertools import pairwise
from pathlib import Path


if __package__:
    from .live_indicators import DAY, MINUTE, OFFSET, IndicatorHistory
else:
    from live_indicators import DAY, MINUTE, OFFSET, IndicatorHistory


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class RangeBandHistory:
    def __init__(self, source, archive, account_id, clock=None):
        self.source, self.archive = Path(source), Path(archive)
        self.account_id = account_id
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.reader = IndicatorHistory(source, account_id)
        self.lock = threading.RLock()
        self.last_seen = None
        self.archive.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.archive)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS range_bands ("
                "scope TEXT NOT NULL,effective_at INTEGER NOT NULL,payload TEXT NOT NULL,"
                "recorded_at INTEGER NOT NULL,PRIMARY KEY(scope,effective_at))"
            )
        with closing(sqlite3.connect(self.archive)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reconstructed_range_bands ("
                "scope TEXT NOT NULL,effective_at INTEGER NOT NULL,payload TEXT NOT NULL,"
                "computed_at INTEGER NOT NULL,evidence_sha256 TEXT NOT NULL,"
                "PRIMARY KEY(scope,effective_at))"
            )
        self.archive.chmod(0o600)

    def binding(self, state, execution):
        if (
            state.get("account_id") != self.account_id
            or execution.get("account_id") != self.account_id
        ):
            raise ValueError("range_history_account_mismatch")
        binding = state.get("range_policy")
        if not binding:
            return None
        policy = binding["policy"]
        if (
            binding["account_id"] != self.account_id
            or binding["source_sha256"] != state["source_sha256"]
            or digest(state["source_snapshot"]) != state["source_sha256"]
            or binding["profit_started_at"] != state["activated_at"]
            or binding["execution_created_at"] != execution["created_at"]
            or binding["initial_capital"] != execution["book"]["initial"]
            or not state.get("range_policy_activated_at")
            or policy["version"] != "trx-weighted-range-v1"
            or policy["half_life_days"] != 90
            or policy["width_cap_pct"] != 4
        ):
            raise ValueError("range_history_binding_mismatch")
        return digest(binding)

    def sync(self, state, execution):
        """Copy an already persisted reference into this reader's own display archive."""
        scope = self.binding(state, execution)
        band = state.get("active_price_band")
        if not scope or not band:
            return False
        at = band["effective_at"]
        lower, upper = band["lower"], band["upper"]
        if (
            not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (at, lower, upper))
            or (at + OFFSET) % DAY
            or not 0 < lower <= upper <= lower * 1.04 + 1e-12
            or band["latest_training_close_at"] > at
            or band["half_life_days"] != 90
            or band["width_cap_pct"] != 4
        ):
            raise ValueError("range_history_invalid_band")
        last = state["engine"].get("last_close", 0)
        if not at <= last < at + DAY:
            return False  # A daily reference waiting for confirmation is not plotted as current.
        payload = json.dumps(band, sort_keys=True, allow_nan=False)
        key = (scope, at, payload)
        with self.lock:
            if key == self.last_seen:
                return False
            with closing(sqlite3.connect(self.archive, timeout=2)) as db, db:
                row = db.execute(
                    "SELECT payload FROM range_bands WHERE scope=? AND effective_at=?", (scope, at)
                ).fetchone()
                if row and row[0] != payload:
                    raise ValueError("range_history_daily_reference_changed")
                db.execute(
                    "INSERT OR IGNORE INTO range_bands VALUES(?,?,?,?)",
                    (scope, at, payload, self.clock()),
                )
            self.last_seen = key
        return True

    def import_reconstruction(self, state, execution, evidence):
        """Operator-only display import. Never changes saved live references or account state."""
        scope = self.binding(state, execution)
        binding = state["range_policy"]
        if (
            evidence.get("version") != "range-band-reconstruction-v1"
            or evidence.get("account_id") != self.account_id
            or evidence.get("policy_sha256") != scope
            or evidence.get("history_sha256") != binding["history_sha256"]
            or evidence.get("source_sha256") != state["source_sha256"]
            or evidence.get("half_life_days") != 90
            or evidence.get("width_cap_pct") != 4
        ):
            raise ValueError("range_reconstruction_binding_mismatch")
        bands = evidence.get("bands", [])
        if not bands or len(bands) > 32:
            raise ValueError("range_reconstruction_scope_invalid")
        rows = []
        previous = None
        for band in bands:
            at, lower, upper = band["effective_at"], band["lower"], band["upper"]
            if (
                not all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in (at, lower, upper)
                )
                or (at + OFFSET) % DAY
                or not 0 < lower <= upper <= lower * 1.04 + 1e-12
                or band["latest_training_close_at"] != at
                or band["half_life_days"] != 90
                or band["width_cap_pct"] != 4
                or at > state["engine"]["last_close"]
                or (previous is not None and at != previous + DAY)
            ):
                raise ValueError("range_reconstruction_invalid_band")
            previous = at
            rows.append(
                (
                    scope,
                    at,
                    json.dumps(band, sort_keys=True, allow_nan=False),
                    evidence["generated_at"],
                    digest(evidence),
                )
            )
        with self.lock, closing(sqlite3.connect(self.archive, timeout=2)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            for row in rows:
                live = db.execute(
                    "SELECT payload FROM range_bands WHERE scope=? AND effective_at=?", row[:2]
                ).fetchone()
                if live and json.loads(live[0]) != json.loads(row[2]):
                    raise ValueError("range_reconstruction_live_reference_mismatch")
                old = db.execute(
                    "SELECT payload FROM reconstructed_range_bands "
                    "WHERE scope=? AND effective_at=?",
                    row[:2],
                ).fetchone()
                if old and old[0] != row[2]:
                    raise ValueError("range_reconstruction_already_imported_different_value")
                db.execute("INSERT OR IGNORE INTO reconstructed_range_bands VALUES(?,?,?,?,?)", row)
        return {"references": len(rows), "evidence_sha256": digest(evidence)}

    def query(self, state, execution, range_name="all", limit=1800, *, include_reconstructed=True):
        scope = self.binding(state, execution)
        if not scope:
            return {"account_id": self.account_id, "enabled": False, "points": [], "bands": []}
        self.sync(state, execution)
        end = int(state["engine"].get("last_close") or 0)
        activated = int(state["range_policy_activated_at"])
        with self.lock, closing(sqlite3.connect(self.archive, timeout=2)) as db:
            first = (
                db.execute(
                    "SELECT MIN(effective_at) FROM reconstructed_range_bands WHERE scope=?",
                    (scope,),
                ).fetchone()[0]
                if include_reconstructed
                else None
            )
            earliest = min(activated, first) if first is not None else activated
            begin = (
                max(earliest, end - (DAY if range_name == "day" else 7 * DAY))
                if range_name != "all"
                else earliest
            )
            rows = db.execute(
                "SELECT effective_at,payload FROM range_bands WHERE scope=? "
                "AND effective_at<=? AND effective_at+?>=? ORDER BY effective_at",
                (scope, end, DAY, begin),
            ).fetchall()
            reconstructed = (
                db.execute(
                    "SELECT effective_at,payload,computed_at FROM reconstructed_range_bands "
                    "WHERE scope=? "
                    "AND effective_at<=? AND effective_at+?>=? ORDER BY effective_at",
                    (scope, end, DAY, begin),
                ).fetchall()
                if include_reconstructed
                else []
            )
        known = {at: json.loads(payload) for at, payload in rows}
        past = {at: (json.loads(payload), computed) for at, payload, computed in reconstructed}
        bands = []
        if end >= begin:
            for day in range((begin + OFFSET) // DAY, (end + OFFSET) // DAY + 1):
                at = day * DAY - OFFSET
                left, right = max(begin, at), min(end, at + DAY)
                live_start = max(activated, at)
                cuts = [left] + ([live_start] if left < live_start < right else []) + [right]
                for start, stop in pairwise(cuts):
                    live = known.get(at) if start >= live_start else None
                    backfill, computed = past.get(at, (None, None))
                    band = live or backfill
                    bands.append(
                        {
                            "timestamp": start,
                            "end_at": stop,
                            "effective_at": at,
                            "lower": band["lower"] if band else None,
                            "upper": band["upper"] if band else None,
                            "training_end": band["latest_training_close_at"] if band else None,
                            "provenance": "live_record"
                            if live
                            else "historical_reconstruction"
                            if backfill
                            else "missing",
                            "computed_at": computed if backfill and not live else None,
                        }
                    )
        points = self.points(begin, end, limit)
        return {
            "account_id": self.account_id,
            "enabled": True,
            "policy_sha256": scope,
            "activated_at": activated,
            "start": begin,
            "end": end,
            "observed_at": end or None,
            "half_life_days": 90,
            "width_cap_pct": 4,
            "points": points["points"],
            "bands": bands,
            "source_points": points["source_points"],
            "missing_band_days": len({b["effective_at"] for b in bands if b["lower"] is None}),
            "reconstructed_band_days": len(
                {b["effective_at"] for b in bands if b["provenance"] == "historical_reconstruction"}
            ),
            "recorded_band_days": len(
                {b["effective_at"] for b in bands if b["provenance"] == "live_record"}
            ),
            "timezone": "Asia/Shanghai",
            "basis": "causal_historical_reconstruction_and_persisted_live_daily_bands",
            "sampling": "bucket_first_last_price_min_max_and_gaps; daily_bands_unsampled",
        }

    def points(self, begin, end, limit):
        if begin > end:
            return {"points": [], "source_points": 0}
        rows = list(self.reader.rows(max(0, begin - MINUTE), end))
        points = []
        count = 0
        previous = None
        for at, price in rows:
            stamp = at + MINUTE
            if stamp < begin:
                continue
            if previous is not None and stamp > previous + MINUTE:
                points.append({"timestamp": previous + MINUTE, "price": None})
            points.append({"timestamp": stamp, "price": price})
            previous = stamp
            count += 1
        if len(points) <= limit:
            return {"points": points, "source_points": count}
        selected = set()
        buckets = max(1, limit // 5)
        for i in range(buckets):
            start, stop = len(points) * i // buckets, len(points) * (i + 1) // buckets
            if start == stop:
                continue
            selected.update((start, stop - 1))
            good = [j for j in range(start, stop) if points[j]["price"] is not None]
            missing = [j for j in range(start, stop) if points[j]["price"] is None]
            if good:
                selected.add(min(good, key=lambda j: points[j]["price"]))
                selected.add(max(good, key=lambda j: points[j]["price"]))
            if missing:
                selected.add(missing[0])
        return {"points": [points[i] for i in sorted(selected)], "source_points": count}
