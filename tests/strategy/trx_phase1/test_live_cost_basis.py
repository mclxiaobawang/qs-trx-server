import hashlib
import json
import sqlite3
import threading
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from freqtrade.strategy.trx_phase1.ledger import StateStore
from freqtrade.strategy.trx_phase1.live_cost_basis import (
    CostBasisHistory,
    InterestBasis,
    connection,
)
from freqtrade.strategy.trx_phase1.live_history import LiveHistory
from freqtrade.strategy.trx_phase1.live_history_app import create_app


@pytest.fixture
def source(tmp_path):
    allocation = {
        "coin": "TRX",
        "attribution": "strategy_allocation",
        "bill_id": "receipt",
        "timestamp": 125000,
        "recorded_at": 130000,
        "amount": "2",
    }
    execution = {
        "account_id": "account",
        "created_at": 100000,
        "events": [],
        "receipts": {"income-allocation:receipt": "verified-fingerprint"},
        "income_accounting": {"allocations": {"receipt": allocation}},
        "book": {
            "lots": [{"quantity": "100", "remaining_cost": "30"}],
            "interest_trx": "2",
            "orders": [],
            "events": ["income-allocation:receipt"],
        },
    }
    strategy = {
        "account_id": "account",
        "events": [],
        "markets": {"TRXUSDT": {"bid": "0.5", "timestamp": 200000}},
    }
    with closing(sqlite3.connect(tmp_path / "strategy.sqlite")) as db, db:
        db.executescript(
            "CREATE TABLE live_candles(timestamp INTEGER PRIMARY KEY,close REAL);"
            "CREATE TABLE live_samples(timestamp INTEGER PRIMARY KEY,payload TEXT);"
        )
        db.executemany("INSERT INTO live_candles VALUES(?,?)", [(60000, 0.4), (120000, 0.9)])
        for stamp, interest in [(120000, 0), (200000, 2)]:
            sample = {
                "timestamp": stamp,
                "principal_trx": 100,
                "interest_trx": interest,
                "remaining_cost": 30,
                "price": 0.5,
            }
            db.execute("INSERT INTO live_samples VALUES(?,?)", (stamp, json.dumps(sample)))
    return tmp_path, execution, strategy


def query(source):
    path, execution, strategy = source
    return CostBasisHistory(path, "account").query(strategy, execution)


def test_actual_receipt_price_no_future_candle_and_no_ledger_writes(source):
    path, execution, _ = source
    before = hashlib.sha256((path / "strategy.sqlite").read_bytes()).hexdigest()
    original = json.dumps(execution, sort_keys=True)
    result = query(source)
    assert result["current"]["average_cost"] == pytest.approx(30.8 / 102)
    assert result["current"]["interest_basis"] == "0.8"
    assert result["points"][0]["average_cost"] == pytest.approx(0.3)
    assert result["points"][1]["average_cost"] == pytest.approx(30.8 / 102)
    assert result["display_only"] and result["reconstructed"]
    assert original == json.dumps(execution, sort_keys=True)
    assert before == hashlib.sha256((path / "strategy.sqlite").read_bytes()).hexdigest()
    with closing(connection(path)) as db, pytest.raises(sqlite3.OperationalError):
        db.execute("DELETE FROM live_samples")


def test_only_verified_strategy_share_counted_and_receipts_deduplicated(source):
    _, execution, _ = source
    allocations = execution["income_accounting"]["allocations"]
    allocations["duplicate"] = dict(allocations["receipt"])
    for key, attribution in [("external", "external_account_credit"), ("unassigned", "unassigned")]:
        allocations[key] = {
            **allocations["receipt"],
            "bill_id": key,
            "amount": "1000",
            "attribution": attribution,
        }
    allocations["not-booked"] = {
        **allocations["receipt"],
        "bill_id": "not-booked",
        "amount": "1000",
    }
    result = query(source)
    assert result["interest_receipts"] == 1
    assert result["current"]["average_cost"] == pytest.approx(30.8 / 102)


def test_delayed_receipt_uses_receipt_quote_but_enters_holdings_at_recorded_time(source):
    path, execution, _ = source
    execution["income_accounting"]["allocations"]["receipt"]["recorded_at"] = 250000
    basis = InterestBasis(path, execution)
    assert basis.at(200000, 100, 0, 30)["average_cost"] == pytest.approx(0.3)
    assert basis.at(300000, 100, 2, 30)["average_cost"] == pytest.approx(30.8 / 102)


def test_actual_partial_interest_exit_releases_weighted_average_basis(source):
    _, execution, _ = source
    execution["book"]["interest_trx"] = "1"
    execution["book"]["orders"] = [
        {"id": "exit", "timestamp": 240000, "kind": "manual_interest_exit", "quantity": "1"}
    ]
    result = query(source)
    assert result["current"]["average_cost"] == pytest.approx(30.4 / 101)
    assert result["current"]["interest_basis"] == "0.40"
    assert result["points"][-1]["average_cost"] == pytest.approx(30.8 / 102)


@pytest.mark.parametrize("reason", ["missing_price", "unmatched_quantity"])
def test_incomplete_evidence_is_not_reported_as_zero_cost(source, reason):
    path, execution, _ = source
    if reason == "missing_price":
        with closing(sqlite3.connect(path / "strategy.sqlite")) as db, db:
            db.execute("DELETE FROM live_candles WHERE timestamp=60000")
    else:
        execution["book"]["interest_trx"] = "3"
    result = query(source)
    assert result["current"]["average_cost"] is None
    assert result["current"]["interest_basis"] is None
    assert not result["current"]["complete"]
    assert result["current"]["principal_average_cost"] == pytest.approx(0.3)


def test_zero_holdings_and_account_guard(source):
    _, execution, strategy = source
    execution["book"]["lots"] = []
    execution["book"]["interest_trx"] = "0"
    execution["income_accounting"]["allocations"] = {}
    assert query(source)["current"]["average_cost"] is None
    strategy["account_id"] = "other"
    with pytest.raises(ValueError, match="account_mismatch"):
        query(source)


@pytest.mark.parametrize("performance_start,expected", [(None, 110000), (115000, 115000)])
def test_actual_activation_or_performance_start_precedes_execution_container_creation(
    source, performance_start, expected
):
    _, execution, strategy = source
    strategy["activated_at"] = 110000
    execution["performance_basis"] = {"started_at": performance_start}
    assert query(source)["live_started_at"] == expected


def test_cost_route_is_authenticated_account_bound_and_get_only(source, tmp_path):
    path, execution, strategy = source
    for name, state in [("execution", execution), ("strategy", strategy)]:
        with closing(sqlite3.connect(path / f"{name}.sqlite")) as db, db:
            db.execute("CREATE TABLE trx_state(id TEXT PRIMARY KEY,state TEXT)")
            db.execute("INSERT INTO trx_state VALUES(?,?)", (name, json.dumps(state)))
    history = LiveHistory(path, tmp_path / "archive/history.sqlite", "account")
    secret = "cost-test-secret-which-is-longer-than-thirty-two-bytes"
    client = TestClient(
        create_app(
            {"account_id": "account", "username": "reader", "jwt_secret_key": secret}, history
        )
    )
    token = jwt.encode(
        {
            "identity": {"u": "reader"},
            "type": "access",
            "exp": datetime.now(UTC) + timedelta(minutes=1),
        },
        secret,
        algorithm="HS256",
    )
    url = "/api/v1/trx/live/indicators?view=cost_basis&account_id=account"
    assert client.get(url).status_code == 401
    headers = {"Authorization": "Bearer " + token}
    assert client.get(url, headers=headers).json()["current"]["average_cost"] == pytest.approx(
        30.8 / 102
    )
    assert (
        client.get(
            url.replace("account_id=account", "account_id=other"), headers=headers
        ).status_code
        == 409
    )
    assert client.post(url, headers=headers).status_code == 405


def populate_samples(path, count):
    with closing(sqlite3.connect(path / "strategy.sqlite")) as db, db:
        db.execute("DELETE FROM live_samples")
        db.executemany(
            "INSERT INTO live_samples VALUES(?,?)",
            (
                (
                    stamp,
                    json.dumps(
                        {
                            "timestamp": stamp,
                            "price": 0.5,
                            "principal_trx": 100,
                            "interest_trx": 0,
                            "remaining_cost": 30,
                        }
                    ),
                )
                for stamp in range(1, count + 1)
            ),
        )


def no_interest(execution):
    execution["income_accounting"]["allocations"] = {}
    execution["book"]["interest_trx"] = "0"


def test_all_sample_processing_happens_after_connections_and_cursors_are_closed(
    source, monkeypatch
):
    from freqtrade.strategy.trx_phase1 import live_cost_basis as module

    path, _, _ = source
    populate_samples(path, 1200)
    original_connect, original_at = module.connection, module.InterestBasis.at
    active = 0
    statements = []

    class Tracked:
        def __init__(self):
            nonlocal active
            self.db = original_connect(path)
            active += 1

        def execute(self, sql, parameters):
            statements.append(sql)
            return self.db.execute(sql, parameters)

        def close(self):
            nonlocal active
            self.db.close()
            active -= 1

    def guarded_at(self, *args, **kwargs):
        assert active == 0, "source read connection survived into calculation"
        return original_at(self, *args, **kwargs)

    monkeypatch.setattr(module, "connection", lambda _: Tracked())
    monkeypatch.setattr(module.InterestBasis, "at", guarded_at)
    result = query(source)
    assert active == 0 and result["source_points"] == 1200
    assert all("LIMIT" in sql for sql in statements if "live_samples" in sql)
    assert sum("SELECT timestamp,payload" in sql for sql in statements) >= 5


def test_slow_cost_scan_does_not_block_native_state_commits(source, monkeypatch):
    from freqtrade.strategy.trx_phase1 import live_cost_basis as module

    path, execution, strategy = source
    no_interest(execution)
    populate_samples(path, 4096)
    store = StateStore(path / "strategy.sqlite")
    version = store.save("concurrency-probe", 0, {"heartbeat": 0})
    started = threading.Event()
    original = module.InterestBasis.at
    results, failures, commit_times = [], [], []

    def slow(self, stamp, *args, **kwargs):
        if stamp <= 4096:
            started.set()
            time.sleep(0.0015)  # Reader remains busy longer than SQLite's 5-second write timeout.
        return original(self, stamp, *args, **kwargs)

    monkeypatch.setattr(module.InterestBasis, "at", slow)

    def read():
        try:
            results.append(CostBasisHistory(path, "account").query(strategy, execution))
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=read)
    thread.start()
    assert started.wait(3)
    for heartbeat in range(40):
        at = time.monotonic()
        version = store.save("concurrency-probe", version, {"heartbeat": heartbeat + 1})
        commit_times.append(time.monotonic() - at)
        time.sleep(0.003)
    # Writer must finish while the deliberately slow reader is still computing.
    assert thread.is_alive()
    thread.join(15)
    assert not thread.is_alive() and not failures
    assert len(results) == 1 and results[0]["source_points"] == 4096
    assert max(commit_times) < 1, commit_times
    assert store.load("concurrency-probe") == (41, {"heartbeat": 40})


def test_scan_pins_end_watermark_while_new_observations_arrive(source, monkeypatch):
    from freqtrade.strategy.trx_phase1 import live_cost_basis as module

    path, execution, strategy = source
    no_interest(execution)
    populate_samples(path, 600)
    original = module.InterestBasis.at
    appended = False

    def append_after_scan_started(self, stamp, *args, **kwargs):
        nonlocal appended
        if stamp == 1 and not appended:
            with closing(sqlite3.connect(path / "strategy.sqlite", timeout=0.2)) as db, db:
                sample = {
                    "timestamp": 601,
                    "price": 0.6,
                    "principal_trx": 100,
                    "interest_trx": 0,
                    "remaining_cost": 30,
                }
                db.execute("INSERT INTO live_samples VALUES(?,?)", (601, json.dumps(sample)))
            appended = True
        return original(self, stamp, *args, **kwargs)

    monkeypatch.setattr(module.InterestBasis, "at", append_after_scan_started)
    result = CostBasisHistory(path, "account").query(strategy, execution)
    assert appended and result["end"] == 600 and result["source_points"] == 600
    assert result["points"][-1]["timestamp"] == 600
