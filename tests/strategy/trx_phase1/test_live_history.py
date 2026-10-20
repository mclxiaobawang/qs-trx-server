import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from freqtrade.strategy.trx_phase1.live_history import HistoryUnavailable, LiveHistory
from freqtrade.strategy.trx_phase1.live_history_app import create_app


def save(path, key, state):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS trx_state(id TEXT PRIMARY KEY,version INTEGER,state TEXT)"
        )
        db.execute(
            "INSERT INTO trx_state VALUES(?,1,?) ON CONFLICT(id) "
            "DO UPDATE SET state=excluded.state,version=version+1",
            (key, json.dumps(state)),
        )


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "account"
    source.mkdir()
    execution = {
        "account_id": "test-account",
        "created_at": 100,
        "events": [],
        "book": {"orders": []},
    }
    strategy = {"account_id": "test-account", "events": []}
    save(source / "execution.sqlite", "execution", execution)
    save(source / "strategy.sqlite", "strategy", strategy)
    archive = LiveHistory(
        source, tmp_path / "archive/history.sqlite", "test-account", clock=lambda: 99999
    )
    return source, archive, execution, strategy


def test_complete_source_and_backups_are_deduplicated_without_changing_ledger(setup):
    source, archive, execution, strategy = setup
    execution["events"] = [
        {"kind": "operation_settled", "timestamp": 1000 + n, "operation_id": str(n)}
        for n in range(85)
    ]
    execution["book"]["orders"] = [{"timestamp": 500, "order_id": "first"}]
    strategy["events"] = [{"kind": "strategy_poll_blocked", "timestamp": 200}]
    save(source / "execution.sqlite", "execution", execution)
    save(source / "strategy.sqlite", "strategy", strategy)
    before = {p.name: p.read_bytes() for p in source.glob("*.sqlite")}
    archive.seed(
        {
            "account_id": "test-account",
            "snapshots": [
                {
                    "execution": execution["events"][:20]
                    + [{"kind": "write_attempt", "timestamp": 150}]
                }
            ],
        }
    )
    archive.sync()
    archive.sync()
    assert archive.query()["total"] == 88
    assert archive.query(start=1000, end=1001)["rows"][0]["data"]["operation_id"] == "0"
    assert archive.query(kind="exception")["total"] == 1
    assert archive.query(kind="fill")["rows"][0]["data"]["order_id"] == "first"
    assert {p.name: p.read_bytes() for p in source.glob("*.sqlite")} == before


def test_pagination_has_no_missing_ties_or_live_insert_duplicates(setup):
    _, archive, _, _ = setup
    events = [
        {"kind": "operation_settled", "timestamp": 1000, "operation_id": str(n)} for n in range(75)
    ]
    archive.ingest({"execution": events})
    page = archive.query(limit=30)
    found = page["rows"][:]
    archive.ingest({"execution": events + [{"kind": "write_attempt", "timestamp": 999}]})
    while page["next_cursor"]:
        page = archive.query(limit=30, cursor=page["next_cursor"])
        assert page["total"] == 75
        found.extend(page["rows"])
    assert len(found) == len({r["id"] for r in found}) == 75
    assert {r["data"]["operation_id"] for r in found} == {str(n) for n in range(75)}
    assert archive.query()["total"] == 76


def test_archive_survives_source_rollover_and_restart_and_detects_gaps(setup):
    source, archive, _, strategy = setup
    strategy["events"] = [{"kind": "strategy_poll_blocked", "timestamp": n} for n in range(10)]
    save(source / "strategy.sqlite", "strategy", strategy)
    archive.sync()
    strategy["events"] = [{"kind": "strategy_poll_blocked", "timestamp": n} for n in range(5, 505)]
    save(source / "strategy.sqlite", "strategy", strategy)
    archive = LiveHistory(source, archive.archive, "test-account")
    archive.sync()
    assert archive.query()["total"] == 505
    assert not archive.query()["coverage"]["possible_strategy_gap"]
    strategy["events"] = [
        {"kind": "strategy_poll_blocked", "timestamp": n} for n in range(600, 1100)
    ]
    save(source / "strategy.sqlite", "strategy", strategy)
    archive.sync()
    assert archive.query()["coverage"]["possible_strategy_gap"]


def test_late_allocation_uses_recording_time_and_preserves_identical_events(setup):
    _, archive, _, _ = setup
    repeated = {"kind": "write_attempt", "timestamp": 300}
    archive.ingest(
        {
            "execution": [
                repeated,
                repeated,
                {"kind": "interest_allocation_recorded", "timestamp": 100, "recorded_at": 500},
            ]
        }
    )
    archive.ingest({"execution": [repeated, repeated]})
    assert archive.query()["total"] == 3
    assert archive.query(start=500, end=501)["total"] == 1
    assert archive.query(start=100, end=101)["total"] == 0


def test_invalid_range_cursor_account_and_missing_source_fail_closed(setup):
    source, archive, _, _ = setup
    for arguments in ({"start": 5, "end": 4}, {"limit": 101}, {"cursor": "broken"}):
        with pytest.raises(ValueError):
            archive.query(**arguments)
    archive.ingest({"execution": [{"kind": "write_attempt", "timestamp": n} for n in range(3)]})
    cursor = archive.query(limit=1)["next_cursor"]
    with pytest.raises(ValueError, match="cursor"):
        archive.query(kind="exception", cursor=cursor)
    with pytest.raises(HistoryUnavailable):
        archive.seed({"account_id": "other", "snapshots": []})
    with pytest.raises(HistoryUnavailable, match="ledger"):
        archive.seed({"account_id": "test-account", "created_at": 99, "snapshots": []})
    (source / "execution.sqlite").unlink()
    with pytest.raises(HistoryUnavailable):
        archive.sync()
    assert not (source / "execution.sqlite").exists()


def test_reusing_exchange_identity_for_another_ledger_does_not_mix_history(setup):
    source, archive, execution, _ = setup
    archive.sync()
    execution["created_at"] += 1
    save(source / "execution.sqlite", "execution", execution)
    with pytest.raises(HistoryUnavailable, match="ledger"):
        archive.sync()


def test_authenticated_api_rejects_wrong_expired_refresh_tokens_and_mutations(setup):
    _, archive, _, _ = setup
    config = {
        "account_id": "test-account",
        "jwt_secret_key": "test-history-key-which-is-long-enough",
        "username": "owner",
    }
    app = create_app(config, archive)

    def token(**changes):
        payload = {
            "identity": {"u": "owner"},
            "type": "access",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        }
        payload.update(changes)
        return {
            "Authorization": "Bearer "
            + jwt.encode(payload, config["jwt_secret_key"], algorithm="HS256")
        }

    with TestClient(app) as client:
        path = "/api/v1/trx/live/history?account_id=test-account"
        assert client.get(path).status_code == 401
        for headers in (
            token(type="refresh"),
            token(exp=datetime.now(UTC) - timedelta(seconds=10)),
            token(identity={"u": "other"}),
        ):
            assert client.get(path, headers=headers).status_code == 401
        assert client.get(path, headers=token()).status_code == 200
        assert client.get(path.replace("test-account", "other"), headers=token()).status_code == 409
        assert client.get(path + "&start=2&end=1", headers=token()).status_code == 422
        assert client.get(path + "&limit=1000", headers=token()).status_code == 422
        assert client.post(path, headers=token()).status_code == 405


def test_read_archive_does_not_prevent_an_independent_account_writer(setup):
    source, archive, execution, _ = setup

    def writer():
        for n in range(30):
            execution["events"].append({"kind": "operation_settled", "timestamp": n})
            save(source / "execution.sqlite", "execution", execution)

    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(writer)
        for _ in range(20):
            archive.sync()
            archive.query(limit=2)
        task.result(timeout=5)
    archive.sync()
    assert archive.query()["total"] == 30
    assert archive.read_source("execution")["events"] == execution["events"]
