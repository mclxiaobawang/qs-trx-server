import hashlib
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from freqtrade.strategy.trx_phase1.live_history_app import create_app
from freqtrade.strategy.trx_phase1.live_range_history import DAY, MINUTE, RangeBandHistory, digest
from tests.strategy.trx_phase1.test_live_indicators import evidence as original_evidence


@pytest.fixture
def evidence(tmp_path):
    path, state, execution, _book, start, _closes = original_evidence.__wrapped__(tmp_path)
    state.update(source_snapshot=state["engine"]["config"], activated_at=execution["created_at"])
    state["source_sha256"] = digest(state["source_snapshot"])
    state["range_policy_activated_at"] = execution["created_at"] + 10 * MINUTE
    state["range_policy"] = {
        "account_id": "account",
        "source_sha256": state["source_sha256"],
        "profit_started_at": state["activated_at"],
        "execution_created_at": execution["created_at"],
        "initial_capital": execution["book"]["initial"],
        "history_sha256": "test-seed-fingerprint",
        "policy": {"version": "trx-weighted-range-v1", "half_life_days": 90, "width_cap_pct": 4},
    }
    state["active_price_band"] = {
        "effective_at": start + DAY,
        "lower": 0.3,
        "upper": 0.312,
        "latest_training_close_at": start + DAY,
        "half_life_days": 90,
        "width_cap_pct": 4,
    }
    with closing(sqlite3.connect(path / "strategy.sqlite")) as db, db:
        db.execute("UPDATE trx_state SET state=?", (json.dumps(state),))
    return path, state, execution, start


def test_prices_and_persisted_bands_start_at_activation_and_never_write_financial_sources(evidence):
    path, state, execution, start = evidence
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in path.glob("*.sqlite")}
    original = deepcopy((state, execution))
    reader = RangeBandHistory(path, path / "display" / "bands.sqlite", "account")
    result = reader.query(state, execution)
    assert result["start"] == state["range_policy_activated_at"]
    assert min(p["timestamp"] for p in result["points"]) >= result["start"]
    assert result["bands"] == [
        {
            "timestamp": result["start"],
            "end_at": result["end"],
            "effective_at": start + DAY,
            "lower": 0.3,
            "upper": 0.312,
            "training_end": start + DAY,
            "provenance": "live_record",
            "computed_at": None,
        }
    ]
    assert result["missing_band_days"] == 0
    assert len(result["points"]) <= 1800
    assert (state, execution) == original
    assert files == {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in path.glob("*.sqlite")
    }
    assert reader.sync(state, execution) is False
    restarted = RangeBandHistory(path, path / "display" / "bands.sqlite", "account")
    assert restarted.query(state, execution)["bands"] == result["bands"]


def test_missing_days_and_candles_stay_empty_and_daily_boundaries_never_rewrite(evidence):
    path, state, execution, start = evidence
    reader = RangeBandHistory(path, path / "display" / "bands.sqlite", "account")
    reader.sync(state, execution)
    state["active_price_band"]["upper"] = 0.311
    with pytest.raises(ValueError, match="daily_reference_changed"):
        reader.sync(state, execution)
    state["active_price_band"] = {
        **state["active_price_band"],
        "effective_at": start + 3 * DAY,
        "latest_training_close_at": start + 3 * DAY,
    }
    state["engine"]["last_close"] = start + 3 * DAY + 60 * MINUTE
    result = reader.query(state, execution)
    assert result["missing_band_days"] == 1
    assert result["bands"][1]["lower"] is None
    assert result["bands"][2]["lower"] == 0.3
    with closing(sqlite3.connect(path / "strategy.sqlite")) as db, db:
        db.execute("DELETE FROM live_candles WHERE timestamp=?", (start + DAY + 40 * MINUTE,))
    assert any(p["price"] is None for p in reader.query(state, execution)["points"])


@pytest.mark.parametrize(
    "field", ["account_id", "profit_started_at", "source_sha256", "initial_capital"]
)
def test_another_account_or_financial_origin_cannot_supply_band_history(evidence, field):
    path, state, execution, _ = evidence
    state["range_policy"][field] = "wrong"
    with pytest.raises(ValueError, match="binding_mismatch"):
        RangeBandHistory(path, path / "display.sqlite", "account").query(state, execution)


def test_future_training_data_is_rejected_and_oversized_price_history_retains_extrema(evidence):
    path, state, execution, _ = evidence
    reader = RangeBandHistory(path, path / "display.sqlite", "account")
    state["active_price_band"]["latest_training_close_at"] += MINUTE
    with pytest.raises(ValueError, match="invalid_band"):
        reader.query(state, execution)
    state["active_price_band"]["latest_training_close_at"] -= MINUTE
    result = reader.query(state, execution, limit=60)
    full = reader.query(state, execution, limit=1800)
    assert len(result["points"]) <= 60
    assert max(p["price"] for p in full["points"]) == max(p["price"] for p in result["points"])
    assert min(p["price"] for p in full["points"]) == min(p["price"] for p in result["points"])
    state.pop("range_policy")
    assert reader.query(state, execution)["enabled"] is False


def test_range_history_route_requires_authentication_is_get_only_and_keeps_standard_indicators(
    evidence,
):
    path, _, _, _ = evidence
    config = {
        "source": str(path),
        "archive": str(path / "display" / "history.sqlite"),
        "account_id": "account",
        "username": "owner",
        "jwt_secret_key": "test-band-secret-at-least-32-characters",
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
    route = "/api/v1/trx/live/indicators?account_id=account&view=range_band"
    with TestClient(create_app(config)) as client:
        assert client.get(route).status_code == 401
        assert client.get(route, headers=headers).json()["bands"][0]["lower"] == 0.3
        assert (
            client.get(
                route.replace("account_id=account", "account_id=other"), headers=headers
            ).status_code
            == 409
        )
        assert client.post(route, headers=headers).status_code == 405
        assert client.get(
            route.replace("view=range_band", "view=standard"), headers=headers
        ).json()["p75"]
        assert (
            client.get(route.replace("view=range_band", "view=bad"), headers=headers).status_code
            == 422
        )


def reconstruction(state, start):
    return {
        "version": "range-band-reconstruction-v1",
        "account_id": state["account_id"],
        "policy_sha256": digest(state["range_policy"]),
        "source_sha256": state["source_sha256"],
        "history_sha256": state["range_policy"]["history_sha256"],
        "half_life_days": 90,
        "width_cap_pct": 4,
        "generated_at": state["engine"]["last_close"],
        "bands": [
            {**state["active_price_band"], "effective_at": at, "latest_training_close_at": at}
            for at in range(start - 6 * DAY, start + 2 * DAY, DAY)
        ],
    }


def test_full_seven_day_backfill_is_reference_only_and_never_rewrites_live_records(evidence):
    path, state, execution, start = evidence
    with closing(sqlite3.connect(path / "strategy.sqlite")) as db, db:
        db.executemany(
            "INSERT INTO live_candles VALUES(?,?)",
            [(at, 0.3) for at in range(start - 6 * DAY, start, MINUTE)],
        )
    source_hashes = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in path.glob("*.sqlite")
    }
    reader = RangeBandHistory(path, path / "display" / "range-bands.sqlite", "account")
    reader.sync(state, execution)
    with closing(sqlite3.connect(reader.archive)) as db:
        live_before = db.execute("SELECT * FROM range_bands").fetchall()
    history = reconstruction(state, start)
    original = deepcopy((state, execution))
    reader.import_reconstruction(state, execution, history)
    result = reader.query(state, execution, "week")
    assert result["end"] - result["start"] == 7 * DAY
    assert result["source_points"] == 7 * 1440 + 1
    assert result["points"][0]["timestamp"] == result["start"]
    assert result["points"][-1]["timestamp"] == result["end"]
    assert len(result["points"]) <= 1800 and result["missing_band_days"] == 0
    assert result["bands"][0]["provenance"] == "historical_reconstruction"
    assert result["bands"][-1]["provenance"] == "live_record"
    assert result["bands"][-1]["timestamp"] == state["range_policy_activated_at"]
    assert result["bands"][-2]["end_at"] == state["range_policy_activated_at"]
    assert result["reconstructed_band_days"] == 8 and result["recorded_band_days"] == 1
    legacy = reader.query(state, execution, "week", include_reconstructed=False)
    assert legacy["start"] == state["range_policy_activated_at"]
    assert all(b["provenance"] == "live_record" for b in legacy["bands"])
    reader.import_reconstruction(state, execution, history)  # Idempotent display import.
    with closing(sqlite3.connect(reader.archive)) as db:
        assert db.execute("SELECT * FROM range_bands").fetchall() == live_before
        assert db.execute("SELECT COUNT(*) FROM reconstructed_range_bands").fetchone()[0] == 8
    assert (state, execution) == original
    assert source_hashes == {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in path.glob("*.sqlite")
    }
    assert (
        reader.query(state, execution, "day")["end"]
        - reader.query(state, execution, "day")["start"]
        == DAY
    )
    restarted = RangeBandHistory(path, reader.archive, "account")
    assert restarted.query(state, execution, "week")["bands"] == result["bands"]

    config = {
        "source": str(path),
        "archive": str(path / "display" / "history.sqlite"),
        "account_id": "account",
        "username": "owner",
        "jwt_secret_key": "test-seven-day-secret-at-least-32-characters",
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
    route = "/api/v1/trx/live/indicators?account_id=account&view=range_band&range=week"
    with TestClient(create_app(config)) as client:
        old_client = client.get(route, headers=headers).json()
        new_client = client.get(route + "&include_history=true", headers=headers).json()
        assert old_client["start"] == state["range_policy_activated_at"]
        assert new_client["end"] - new_client["start"] == 7 * DAY
        assert new_client["reconstructed_band_days"] == 8


@pytest.mark.parametrize("problem", ["scope", "future", "live_mismatch"])
def test_bad_backfills_are_rejected_atomically_without_reclassifying_live_data(evidence, problem):
    path, state, execution, start = evidence
    reader = RangeBandHistory(path, path / "display" / "bands.sqlite", "account")
    reader.sync(state, execution)
    data = reconstruction(state, start)
    if problem == "scope":
        data["policy_sha256"] = "wrong"
    elif problem == "future":
        data["bands"][0]["latest_training_close_at"] += MINUTE
    else:
        data["bands"][-1]["upper"] = 0.311
    with pytest.raises(ValueError):
        reader.import_reconstruction(state, execution, data)
    with closing(sqlite3.connect(reader.archive)) as db:
        assert db.execute("SELECT COUNT(*) FROM reconstructed_range_bands").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM range_bands").fetchone()[0] == 1
