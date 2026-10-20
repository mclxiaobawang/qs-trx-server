import json
import sqlite3
from copy import deepcopy

import pytest

from freqtrade.rpc.api_server import api_trx
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_restore import ExistingLiveRestore, existing_state
from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategy
from tests.strategy.trx_phase1.test_live_service import service as base_service
from tests.strategy.trx_phase1.test_live_service import setup


@pytest.fixture
def service(tmp_path):
    return base_service.__wrapped__(tmp_path)


@pytest.fixture
def completed(service):
    svc, broker, clock, reads, tiers = service
    svc.prepare(setup())
    runtime = svc.runtime
    broker.writes_enabled = True
    runtime.state["desired"] = "running"
    for _ in range(90):
        clock[0] += 15000
        runtime.market.now = clock[0]
        runtime.tick(clock[0])
        if runtime.state["startup_complete"] and not runtime.state["intent"]:
            break
    assert runtime.state["startup_complete"]
    assert runtime.execution.state["book"]["orders"]
    svc.existing_only = True
    svc.runtime = None
    reads.clear()
    return svc, broker, clock, reads, tiers


def change_state(svc, name, mutate):
    control = svc.load()
    path = svc.root / "live_execution" / control["account_id"] / f"{name}.sqlite"
    with sqlite3.connect(path) as db:
        state = json.loads(
            db.execute("SELECT state FROM trx_state WHERE id=?", (name,)).fetchone()[0]
        )
        mutate(state)
        db.execute("UPDATE trx_state SET state=? WHERE id=?", (json.dumps(state), name))
    return path


def test_missing_account_cannot_be_created_by_cloud_prepare_or_restore(service):
    svc, broker, _, reads, _ = service
    svc.existing_only = True
    with pytest.raises(ExecutionError, match="existing_live_account_required"):
        svc.prepare(setup())
    with pytest.raises(ExecutionError, match="existing_live_account_required"):
        ExistingLiveRestore(svc).once()
    assert reads == [] and broker.state()["writes"] == []
    assert not (svc.root / "live_execution").exists()


@pytest.mark.parametrize("name", ["execution", "strategy"])
def test_missing_migrated_database_is_not_reinitialized(completed, name):
    svc, broker, _, reads, _ = completed
    before = broker.state()["writes"]
    path = svc.root / "live_execution" / svc.load()["account_id"] / f"{name}.sqlite"
    path.unlink()
    with pytest.raises(ExecutionError, match="existing_live_account_required"):
        svc.prepare(setup())
    assert not path.exists()
    assert reads == [] and broker.state()["writes"] == before


@pytest.mark.parametrize("field,value", [("startup_complete", False), ("activated_at", None)])
def test_partial_startup_cannot_be_automatically_started(completed, field, value):
    svc, _, _, reads, _ = completed
    change_state(svc, "strategy", lambda state: state.update({field: value}))
    with pytest.raises(ExecutionError, match="completed_live_startup_required"):
        ExistingLiveRestore(svc).once()
    assert reads == []


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("strategy", "account_id", "f" * 64),
        ("execution", "account_id", "f" * 64),
        ("strategy", "source_sha256", "f" * 64),
        ("strategy", "source_snapshot", {}),
        ("execution", "products", []),
    ],
)
def test_mismatched_migration_fails_before_any_exchange_access(completed, name, field, value):
    svc, _, _, reads, _ = completed
    change_state(svc, name, lambda state: state.update({field: value}))
    with pytest.raises(ExecutionError, match="frozen_live_setup_mismatch"):
        ExistingLiveRestore(svc).once()
    assert reads == []


@pytest.mark.parametrize("desired,phase", [("paused", "paused"), ("running", "blocked")])
def test_manual_pause_and_review_blocks_survive_restart(completed, desired, phase):
    svc, _, _, reads, _ = completed
    change_state(svc, "strategy", lambda state: state.update(desired=desired, phase=phase))
    assert not ExistingLiveRestore(svc).once()
    assert reads == []


def test_completed_startup_book_anchor_and_orders_survive_restoration(completed, mocker):
    svc, broker, clock, reads, _ = completed
    control, _ = existing_state(svc)
    runtime = svc.get_runtime(control)
    before = deepcopy(runtime.execution.state)
    writes = broker.state()["writes"]
    start = mocker.patch.object(LiveStrategy, "start")
    assert ExistingLiveRestore(svc).once()
    start.assert_called_once()
    assert reads == [False, True]
    assert runtime.state["startup_complete"]
    runtime.tick(clock[0])
    after = runtime.execution.state
    assert after["book"]["orders"] == before["book"]["orders"]
    assert after["book"]["lots"] == before["book"]["lots"]
    assert after["anchor"] == before["anchor"]
    assert after["anchor_version"] == before["anchor_version"]
    assert broker.state()["writes"] == writes
    assert runtime.state["indicator_initialization"]["signals_after"] >= clock[0]


def test_terms_change_requires_manual_review_and_cannot_self_approve_on_retry(completed, mocker):
    svc, _, _, reads, tiers = completed
    original = svc.load()["review_hash"]
    tiers[0]["currentApy"] = "3"
    start = mocker.patch.object(LiveStrategy, "start")
    restore = ExistingLiveRestore(svc)
    assert not restore.once()
    assert svc.runtime.state["error"] == "live_terms_changed_review_required"
    assert svc.runtime.state["desired"] == "paused"
    assert not restore.once()
    assert reads == [False]
    assert svc.load()["review_hash"] == original
    start.assert_not_called()


def test_transient_read_failure_keeps_restart_intent_and_can_retry(completed, mocker):
    svc, broker, _, reads, _ = completed
    account = broker.account_id
    broker.account_id = mocker.Mock(side_effect=ExecutionError("exchange_response_unavailable"))
    restore = ExistingLiveRestore(svc)
    with pytest.raises(ExecutionError, match="exchange_response_unavailable"):
        restore.once()
    assert existing_state(svc)[1]["desired"] == "running"
    broker.account_id = account
    start = mocker.patch.object(LiveStrategy, "start")
    assert restore.once()
    start.assert_called_once()
    assert reads == [False, False, True]


def test_changed_credentials_cannot_automatically_enable_writes(completed):
    svc, _, _, reads, _ = completed
    control = svc.load()
    control["credentials_identity"] = "f" * 64
    svc.save(control)
    with pytest.raises(ExecutionError, match="credentials_changed_requires_preflight"):
        ExistingLiveRestore(svc).once()
    assert reads == []


def test_stopping_restorer_cannot_start_worker(completed, mocker):
    svc, _, _, reads, _ = completed
    restore = ExistingLiveRestore(svc)
    restore.stop()
    start = mocker.patch.object(LiveStrategy, "start")
    assert not restore.once()
    start.assert_not_called()
    assert reads == []


def test_server_recovery_is_explicit_opt_in_and_requires_existing_only_mode(mocker):
    factory = mocker.patch("freqtrade.strategy.trx_phase1.live_restore.ExistingLiveRestore")
    api_trx.start_live_restore({})
    factory.assert_not_called()
    with pytest.raises(RuntimeError, match="existing-live-only"):
        api_trx.start_live_restore({"trx_phase1": {"auto_resume_live": True}})
    factory.assert_not_called()
