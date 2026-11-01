import fcntl
import json
import os
import sqlite3
import time
from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.auto_rebate import (
    AdmissionDeferred,
    batch_policy,
    eligible,
    fingerprint,
    read_status,
    supervised_commit,
)
from freqtrade.strategy.trx_phase1.credit_contribution import candidate
from tests.strategy.trx_phase1.test_credit_contribution import NOW, sample


def standing(state):
    return {
        "version": "auto-rebate-capital-v1",
        "enabled": True,
        "account_id": state["account_id"],
        "ledger_created_at": state["created_at"],
        "approved_at": NOW - 1000,
        "starts_after_recorded_at": NOW - 1000,
        "scope": "verified_native_usdt_rebates",
        "classification": "new_capital",
        "source_sha256": "frozen",
        "worker_image_id": "sha256:worker",
    }


def test_native_credit_adopts_once_even_with_changed_batch_time_and_restart():
    state, _, markets, wallet = sample()
    policy = standing(state)
    original = deepcopy(state)
    batch = batch_policy(state, policy, NOW)
    work, record = candidate(state, batch, markets, wallet, NOW)
    assert state == original
    assert record["classification"] == "automatic_capital_unlinked_rebate"
    assert record["authorization_policy_sha256"] == fingerprint(policy)
    assert record["authorization_approved_at"] == policy["approved_at"]
    assert work["receipts"] == state["receipts"]
    assert work["quarantined_account_credits"] == state["quarantined_account_credits"]
    # A response lost after COMMIT and process restart still scans the journal.
    restarted = json.loads(json.dumps(work))
    assert eligible(restarted, policy, NOW + 5000) == []
    assert batch_policy(restarted, policy, NOW + 5000) is None
    assert len(restarted["asset_contributions"]) == 1
    assert restarted["book"]["fee_rebates"] == state["book"]["fee_rebates"]


@pytest.mark.parametrize(
    "problem", ["disabled", "account", "ledger", "scope", "fingerprint", "negative", "timestamp"]
)
def test_invalid_authorization_or_evidence_never_selects_credit(problem):
    state, _, _, _ = sample()
    policy = standing(state)
    if problem == "disabled":
        policy["enabled"] = False
    elif problem == "account":
        policy["account_id"] = "other"
    elif problem == "ledger":
        policy["ledger_created_at"] += 1
    elif problem == "scope":
        policy["scope"] = "all_credits"
    elif problem == "fingerprint":
        state["receipts"]["account-credit:bill"] = "wrong"
    elif problem == "negative":
        state["quarantined_account_credits"]["account-credit:bill"]["offset_delta"] = "-1"
    else:
        state["quarantined_account_credits"]["account-credit:bill"]["recorded_at"] = NOW + 1
    original = deepcopy(state)
    with pytest.raises(AdmissionDeferred):
        eligible(state, policy, NOW)
    assert state == original


def test_old_other_currency_and_manually_adopted_receipts_are_excluded():
    state, manual, markets, wallet = sample()
    policy = standing(state)
    state["quarantined_account_credits"]["account-credit:bill"]["recorded_at"] = NOW - 2000
    assert eligible(state, policy, NOW) == []
    state["quarantined_account_credits"]["account-credit:bill"]["recorded_at"] = NOW
    work, _ = candidate(state, manual, markets, wallet, NOW)
    assert eligible(work, policy, NOW) == []
    state["quarantined_account_credits"]["account-credit:bill"]["coin"] = "TRX"
    assert eligible(state, policy, NOW) == []
    state["quarantined_account_credits"]["account-credit:bill"].update(
        coin="USDT", type="BATCH_INTEREST_USER_IN"
    )
    assert eligible(state, policy, NOW) == []


def test_observer_requires_matching_fresh_status_and_counts_durable_records(tmp_path):
    state, _, markets, wallet = sample()
    policy = standing(state)
    assert read_status(tmp_path, state, NOW)["phase"] == "unavailable"
    (tmp_path / "auto-rebate-policy.json").write_text(json.dumps(policy))
    status = {
        "account_id": state["account_id"],
        "policy_sha256": fingerprint(policy),
        "checked_at": NOW,
        "phase": "idle",
    }
    path = tmp_path / "auto-rebate-status.json"
    path.write_text(json.dumps(status))
    view = read_status(tmp_path, state, NOW)
    assert view["fresh"] and view["enabled"] and view["pending_receipts"] == 1
    assert view["adopted_receipts"] == 0
    assert read_status(tmp_path, state, NOW + 90001)["phase"] == "stale"
    work, _ = candidate(state, batch_policy(state, policy, NOW), markets, wallet, NOW)
    view = read_status(tmp_path, work, NOW)
    assert view["adopted_receipts"] == 1 and view["adopted_amount"] == "0.046358358466"
    status["policy_sha256"] = "wrong"
    path.write_text(json.dumps(status))
    assert not read_status(tmp_path, work, NOW)["fresh"]


def test_supervisor_timeout_releases_lock_and_rolls_back_uncommitted_transaction(tmp_path):
    path, lock_path, started = (
        tmp_path / "ledger.sqlite",
        tmp_path / "execution.lock",
        tmp_path / "started",
    )
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE capital (amount INTEGER)")

    def delayed_commit():
        with lock_path.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            with sqlite3.connect(path) as db:
                db.execute("INSERT INTO capital VALUES (1)")
                started.touch()
                time.sleep(5)

    begin = time.monotonic()
    with pytest.raises(AdmissionDeferred, match="commit_deadline"):
        supervised_commit(delayed_commit, timeout=0.2)
    assert time.monotonic() - begin < 1
    assert started.exists()
    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM capital").fetchone()[0] == 0


def test_commit_then_lost_ack_still_leaves_one_durable_admission(tmp_path):
    state, _, markets, wallet = sample()
    policy = standing(state)
    path = tmp_path / "ledger.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE state (payload TEXT)")

    def lost_response():
        work, _ = candidate(state, batch_policy(state, policy, NOW), markets, wallet, NOW)
        with sqlite3.connect(path) as db:
            db.execute("INSERT INTO state VALUES (?)", (json.dumps(work),))
        os._exit(0)

    with pytest.raises(AdmissionDeferred, match="commit_response_lost"):
        supervised_commit(lost_response)
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT payload FROM state").fetchall()
    assert len(rows) == 1
    assert eligible(json.loads(rows[0][0]), policy, NOW + 1) == []
