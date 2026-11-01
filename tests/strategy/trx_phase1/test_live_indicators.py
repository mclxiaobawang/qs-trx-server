import hashlib
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import jwt
import numpy as np
import pytest
from fastapi.testclient import TestClient

from freqtrade.strategy.trx_phase1.ledger import Ledger
from freqtrade.strategy.trx_phase1.live_history_app import create_app
from freqtrade.strategy.trx_phase1.live_indicators import (
    DAY,
    MINUTE,
    IndicatorHistory,
    indicator_snapshot,
)


@pytest.fixture
def evidence(tmp_path):
    start = int(datetime(2026, 10, 20, tzinfo=UTC).timestamp() * 1000) - 480 * MINUTE
    closes = 0.3 * np.exp(np.sin(np.arange(1840) / 4) * 0.002 + np.arange(1840) * 0.000001)
    cfg = {
        "periods": [60, 240, 1440],
        "variant": "dynamic_period_only",
        "volatility_window": 120,
        "volatility_high": 0.012,
        "minimum_period_hold": 3,
        "price_position_lookback_days": 1,
        "price_position_quantile": 0.75,
    }
    book = Ledger(1000)
    book.buy("fill", "lot", start, 1000, 0.3)
    engine = {
        "config": cfg,
        "last_close": start + len(closes) * MINUTE,
        "last_price": float(closes[-1]),
        "last_sigma": 0.012,
        "buy_period": 1440,
        "sell_period": 240,
        "tier": 2,
        "relax_count": 0,
    }
    state = {"account_id": "account", "engine": engine, "source_sha256": "frozen", "events": []}
    execution = {
        "account_id": "account",
        "created_at": start + DAY + 20 * MINUTE,
        "book": book.state(),
        "events": [],
    }
    for name, value in (("strategy", state), ("execution", execution)):
        with closing(sqlite3.connect(tmp_path / f"{name}.sqlite")) as db, db:
            db.execute("CREATE TABLE trx_state(id TEXT PRIMARY KEY,version INTEGER,state TEXT)")
            db.execute("INSERT INTO trx_state VALUES(?,1,?)", (name, json.dumps(value)))
    with closing(sqlite3.connect(tmp_path / "strategy.sqlite")) as db, db:
        db.execute("CREATE TABLE live_candles(timestamp INTEGER PRIMARY KEY,close REAL)")
        db.executemany(
            "INSERT INTO live_candles VALUES(?,?)",
            ((start + i * MINUTE, float(p)) for i, p in enumerate(closes)),
        )
    return tmp_path, state, execution, book, start, closes


@pytest.mark.parametrize(
    "sigma,high,increment", [(0.01199, False, 0), (0.012, True, 1), (0.02, True, 1)]
)
def test_snapshot_threshold_is_inclusive_and_does_not_change_saved_schedule(
    evidence, sigma, high, increment
):
    _, state, _, book, _, _ = evidence
    state["engine"]["last_sigma"] = sigma
    original = deepcopy((state, book.state()))
    result = indicator_snapshot(state, state["engine"], book)
    assert (state, book.state()) == original
    assert result["high_volatility"] is high
    assert result["volatility_target_increment"] == increment
    assert result["buy_period"] == 1440  # Explain current targets, never replace actual scheduling.
    assert result["assessment_basis"].startswith("current_snapshot_target")
    book.buy("more", "more", 1, 1000, 0.3)
    result = indicator_snapshot(state, state["engine"], book)
    if high:
        assert result["at_tier_cap"] and result["volatility_target_increment"] == 0


def test_history_matches_exact_causal_quantile_and_120_return_formula_without_writes(evidence):
    path, state, execution, _, start, closes = evidence
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.sqlite")}
    original = deepcopy((state, execution))
    reader = IndicatorHistory(path, "account")
    result = reader.query(state, execution, "all", 1800)
    assert result["p75"][0]["effective_at"] == start + DAY
    assert result["p75"][0]["value"] == pytest.approx(float(np.quantile(closes[:1440], 0.75)))
    assert result["p75"][0]["timestamp"] == execution["created_at"]
    for point in result["points"]:
        i = (point["timestamp"] - start) // MINUTE - 1
        expected = np.std(np.diff(np.log(closes[i - 120 : i + 1])), ddof=0) * np.sqrt(60)
        assert point["volatility"] == pytest.approx(float(expected), abs=1e-12)
    assert len(result["points"]) <= 1800
    assert result["points"][-1]["timestamp"] == state["engine"]["last_close"]
    assert (state, execution) == original
    assert files == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.sqlite")
    }
    assert reader.query(state, execution, "day")["points"]
    assert reader.query(state, execution, "summary")["current"]["hourly_volatility"] == 0.012


def test_gaps_and_missing_year_are_not_filled_or_mistaken_for_zero_volatility(evidence):
    path, state, execution, _, start, _ = evidence
    with closing(sqlite3.connect(path / "strategy.sqlite")) as db, db:
        db.execute(
            "DELETE FROM live_candles WHERE timestamp IN (?,?)", (start, start + 1465 * MINUTE)
        )
    result = IndicatorHistory(path, "account").query(state, execution)
    assert result["p75"][0]["value"] is None
    assert result["missing_p75_days"] == 1
    assert result["missing_volatility_points"] > 0
    assert any(p["volatility"] is None for p in result["points"])


def test_processing_a_candle_batch_holds_no_live_database_read_lock(evidence):
    path, _, _, _, start, _ = evidence
    reader = IndicatorHistory(path, "account")
    rows = reader.rows(start, start + DAY)
    assert next(rows)[0] == start
    # A paused Python consumer must not block an independent writer, even in DELETE mode.
    with closing(sqlite3.connect(path / "strategy.sqlite", timeout=0)) as db, db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        db.execute("UPDATE trx_state SET version=version+1 WHERE id='strategy'")
    assert len(list(rows)) == 1439


def test_indicator_route_is_authenticated_read_only_and_account_bound(evidence):
    path, _, _, _, _, _ = evidence
    config = {
        "source": str(path),
        "archive": str(path / "history.sqlite"),
        "account_id": "account",
        "username": "owner",
        "jwt_secret_key": "test-indicator-key-at-least-32-characters",
    }
    token = jwt.encode(
        {
            "type": "access",
            "identity": {"u": "owner"},
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        config["jwt_secret_key"],
        algorithm="HS256",
    )
    headers = {"Authorization": "Bearer " + token}
    with TestClient(create_app(config)) as client:
        route = "/api/v1/trx/live/indicators?account_id=account"
        assert client.get(route).status_code == 401
        assert client.get(route, headers=headers).status_code == 200
        assert (
            client.get(
                route.replace("account_id=account", "account_id=other"), headers=headers
            ).status_code
            == 409
        )
        assert client.get(route + "&limit=9999", headers=headers).status_code == 422
        assert client.post(route, headers=headers).status_code == 405


def test_market_volatility_history_includes_pre_live_prices_and_skips_p75_work(
    evidence, monkeypatch
):
    path, state, execution, _, start, closes = evidence
    state["activated_at"] = start + DAY + 30 * MINUTE
    reader = IndicatorHistory(path, "account")
    monkeypatch.setattr(
        reader,
        "reference_values",
        lambda *a: pytest.fail("Volatility history must not calculate daily P75"),
    )
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.sqlite")}
    result = reader.query(state, execution, "all", 60, historical_volatility=True)
    assert result["start"] == start + MINUTE < execution["created_at"]
    assert result["live_started_at"] == state["activated_at"]
    assert result["includes_pre_live_history"] and result["reconstructed"]
    assert result["history_scope"] == "saved_market_candles" and result["p75"] == []
    assert len(result["points"]) <= 60
    assert result["window_minutes"] == 120 and result["high_threshold"] == 0.012
    assert result["missing_volatility_points"] == 120
    assert min(closes) in [p["price"] for p in result["points"]]
    assert max(closes) in [p["price"] for p in result["points"]]
    for point in result["points"]:
        i = (point["timestamp"] - start) // MINUTE - 1
        if i >= 120:
            expected = np.std(np.diff(np.log(closes[i - 120 : i + 1])), ddof=0) * np.sqrt(60)
            assert point["volatility"] == pytest.approx(float(expected), abs=1e-12)
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.sqlite")
    }


@pytest.mark.parametrize(
    "range_name,days", [("month", 30), ("quarter", 90), ("half_year", 180), ("year", 365)]
)
def test_extended_volatility_ranges_keep_actual_coverage_explicit(evidence, range_name, days):
    path, state, execution, _, start, _ = evidence
    result = IndicatorHistory(path, "account").query(
        state, execution, range_name, 60, historical_volatility=True
    )
    assert result["requested_start"] == state["engine"]["last_close"] - days * DAY
    assert result["available_start"] == start + MINUTE
    assert result["start"] == result["available_start"] and result["history_truncated"]
    with pytest.raises(ValueError, match="requires_volatility"):
        IndicatorHistory(path, "account").query(state, execution, range_name)


def test_sparse_candle_gap_breaks_price_and_volatility_linked_series(evidence):
    path, state, execution, _, start, _ = evidence
    with closing(sqlite3.connect(path / "strategy.sqlite")) as db, db:
        db.execute("DELETE FROM live_candles WHERE timestamp=?", (start + 800 * MINUTE,))
    r = IndicatorHistory(path, "account").query(
        state, execution, "all", 60, historical_volatility=True
    )
    assert any(p["break_before"] for p in r["points"])
    assert r["missing_volatility_points"] > 120
    assert any(p["volatility"] is None for p in r["points"])
    assert len(r["points"]) <= 60


def test_extended_history_route_is_only_available_to_volatility_and_stays_get_only(evidence):
    path, _, _, _, _, _ = evidence
    config = {
        "source": str(path),
        "archive": str(path / "history.sqlite"),
        "account_id": "account",
        "username": "owner",
        "jwt_secret_key": "test-indicator-key-at-least-32-characters",
    }
    token = jwt.encode(
        {
            "type": "access",
            "identity": {"u": "owner"},
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        config["jwt_secret_key"],
        algorithm="HS256",
    )
    headers = {"Authorization": "Bearer " + token}
    with TestClient(create_app(config)) as client:
        route = "/api/v1/trx/live/indicators?account_id=account&range=quarter"
        assert client.get(route, headers=headers).status_code == 422
        assert client.get(route + "&view=range_band", headers=headers).status_code == 422
        response = client.get(route + "&view=volatility", headers=headers)
        assert response.status_code == 200 and response.json()["requested_range"] == "quarter"
        assert response.json()["reconstructed"] and response.json()["p75"] == []
        assert client.post(route + "&view=volatility", headers=headers).status_code == 405
        assert client.get(route + "&view=volatility").status_code == 401
        assert (
            client.get(
                route.replace("account_id=account", "account_id=other") + "&view=volatility",
                headers=headers,
            ).status_code
            == 409
        )
