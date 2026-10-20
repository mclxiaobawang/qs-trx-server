"""Connections must close even when garbage collection cannot reclaim them."""

import sqlite3

import pytest

from freqtrade.strategy.trx_phase1.ledger import StateStore
from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategyStore
from freqtrade.strategy.trx_phase1.request_limits import RequestLimiter


def test_repeated_page_and_worker_reads_do_not_retain_database_descriptors(tmp_path, monkeypatch):
    connect = sqlite3.connect
    opened = []

    def tracked(*args, **kwargs):
        db = connect(*args, **kwargs)
        opened.append(db)  # Deliberately retain references: GC cannot hide a leak.
        return db

    monkeypatch.setattr(sqlite3, "connect", tracked)
    store = LiveStrategyStore(tmp_path / "strategy.sqlite")
    limiter = RequestLimiter(tmp_path / "limits.sqlite")
    store.save("strategy", 0, {"account": "test"})
    for _ in range(100):
        assert store.load("strategy")[0] == 1
        assert store.observations()["equity_curve"] == []
        assert limiter.status(limiter.path)["queued"] == 0
    with pytest.raises(RuntimeError, match="Concurrent state change"):
        store.save("strategy", 9, {"account": "must-not-commit"})
    assert StateStore(store.path).load("strategy") == (1, {"account": "test"})
    for db in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            db.execute("SELECT 1")
