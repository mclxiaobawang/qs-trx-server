"""Confirmed replay preserves capital, rejects stale evidence and never resumes."""

import json
import sqlite3
from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.account_movement_receipts import (
    external_usdt_income_allowed,
    known_movement_bill,
)
from freqtrade.strategy.trx_phase1.account_movement_reconcile import VERSION_ID, candidate, persist
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution, digest
from tests.strategy.trx_phase1.test_account_movement_preview import (
    snapshot as snapshot,  # noqa: PLC0414
)


def prepared(snapshot):
    state, evidence, _ = snapshot
    state.update(
        events=[],
        receipts={},
        pending_income={},
        error=None,
        flow=None,
        flows=[],
        anchor=None,
        anchor_version=0,
    )
    state["income_accounting"].update(status="active", policy_sha256="policy", allocations={})
    for position in state["holdings"].values():
        position.update(subscription_ids=[], lots={})
    evidence["execution_state"] = deepcopy(state)
    evidence["markets"] = {"USDGOUSDT": {"bid": "1.0011"}}
    policy = {
        "version": VERSION_ID,
        "account_id": state["account_id"],
        "reconciliation_id": "approved-1",
        "baseline_sha256": digest(state),
        "approved_at": evidence["observed_at_ms"],
        "external_first": True,
        "usdt_outside_strategy": True,
        "manual_bill_ids": ["deposit", "sub-usdt", "sub-usdgo", "fx-quote", "fx-base"],
    }
    return state, evidence, policy


def test_confirmed_replay_is_idempotent_and_preserves_existing_capital(snapshot):
    state, evidence, policy = prepared(snapshot)
    before = deepcopy(state)
    result, record = candidate(state, evidence, policy)
    assert state == before
    for field in ["initial", "lots", "orders", "fees", "realized", "external_flows", "fund_units"]:
        assert result["book"][field] == state["book"][field]
    assert result["created_at"] == state["created_at"]
    assert record["resume_authorized"] is False and record["exchange_writes"] == 0
    assert candidate(result, evidence, policy) == (result, record)
    assert external_usdt_income_allowed(result)
    work = object.__new__(LiveExecution)
    work.state = result
    assert known_movement_bill(work, evidence["bills"]["USDT"][1])
    changed = {**evidence["bills"]["USDT"][1], "size": "101"}
    with pytest.raises(ExecutionError, match="conflicting_confirmed_account_movement"):
        known_movement_bill(work, changed)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("stale", "movement_baseline_changed"),
        ("scope", "movement_manual_scope_changed"),
        ("outside", "movement_ownership_policy_required"),
        ("snapshot", "movement_snapshot_changed"),
    ],
)
def test_unapproved_or_stale_replay_never_changes_input(snapshot, fault, code):
    state, evidence, policy = prepared(snapshot)
    if fault == "stale":
        policy["baseline_sha256"] = "stale"
    elif fault == "scope":
        policy["manual_bill_ids"] = []
    elif fault == "outside":
        policy["usdt_outside_strategy"] = False
    elif fault == "snapshot":
        evidence["execution_state"]["events"].append({"unexpected": True})
    before = deepcopy(state)
    with pytest.raises(ExecutionError, match=code):
        candidate(state, evidence, policy)
    assert state == before


def test_existing_reconciliation_cannot_be_reused_with_changed_policy(snapshot):
    state, evidence, policy = prepared(snapshot)
    result, _ = candidate(state, evidence, policy)
    with pytest.raises(ExecutionError, match="movement_policy_changed"):
        candidate(result, evidence, {**policy, "approved_at": policy["approved_at"] + 1})


@pytest.mark.parametrize("fault", ["hash", "account", "principal", "amount", "product"])
def test_outside_usdt_interest_is_bound_to_verified_holding_and_account(snapshot, fault):
    state, evidence, policy = prepared(snapshot)
    result, _ = candidate(state, evidence, policy)
    assert external_usdt_income_allowed(result)
    if fault == "hash":
        result["outside_usdt_income_policy"]["sha256"] = "bad"
    elif fault == "account":
        result["account_id"] = "other"
    elif fault == "principal":
        result["holdings"]["usdt-hold"]["principal"] = "1"
    elif fault == "amount":
        result["holdings"]["usdt-hold"]["unassigned_amount"] = "51"
    else:
        result["holdings"]["usdt-hold"]["product_id"] = "other"
    assert not external_usdt_income_allowed(result)


def mixed_usdt_subscription(snapshot):
    state, evidence, policy = prepared(snapshot)
    # .10 external + .15 strategy principal move together, without changing ownership.
    row = evidence["bills"]["USDT"][2]
    row.update(size="-.25", balance=".05")
    evidence["bills"] = {"USDT": [row], "USDGO": [], "TRX": []}
    evidence["holdings"]["usdt-hold"]["amount"] = ".25"
    evidence["holdings"]["usdgo-hold"]["amount"] = "500"
    evidence["earn_overview"] = [
        {"coin": "USDT", "amount": ".25"},
        {"coin": "USDGO", "amount": "500"},
        {"coin": "TRX", "amount": "100"},
    ]
    evidence["balances"]["USDT"]["available"] = ".05"
    policy["manual_bill_ids"] = ["sub-usdt"]
    policy["preserve_owned_usdt_location"] = True
    return state, evidence, policy


def test_mixed_manual_usdt_subscription_preserves_principal_and_moves_only_location(snapshot):
    from freqtrade.strategy.trx_phase1.ledger import money

    state, evidence, policy = mixed_usdt_subscription(snapshot)
    old = deepcopy(state)
    new, record = candidate(state, evidence, policy)
    assert state == old
    assert new["book"]["cash"] == old["book"]["cash"]
    assert money(new["book"]["locked"]["USDT"]) == money(".15")
    assert money(new["holdings"]["usdt-hold"]["principal"]) == money(".15")
    assert money(new["holdings"]["usdt-hold"]["unassigned_amount"]) == money(".10")
    for key in ["initial", "external_flows", "fund_units", "lots", "orders", "realized"]:
        assert new["book"][key] == old["book"][key]
    assert external_usdt_income_allowed(new)
    assert candidate(new, evidence, policy) == (new, record)


@pytest.mark.parametrize("fault", ["locked", "principal", "interest", "cash", "yield_policy"])
def test_mixed_usdt_policy_cannot_hide_unverified_strategy_amounts(snapshot, fault):
    from freqtrade.strategy.trx_phase1.ledger import money

    state, evidence, policy = mixed_usdt_subscription(snapshot)
    new, _ = candidate(state, evidence, policy)
    if fault == "locked":
        new["book"]["locked"]["USDT"] = "0"
    elif fault == "principal":
        new["holdings"]["usdt-hold"]["principal"] = ".16"
        new["book"]["locked"]["USDT"] = ".16"
    elif fault == "interest":
        new["holdings"]["usdt-hold"]["interest"] = ".01"
    elif fault == "cash":
        new["book"]["cash"] = str(money(".14"))
    else:
        p = new["outside_usdt_income_policy"]
        p["interest_is_outside_strategy"] = False
        p["sha256"] = digest({k: v for k, v in p.items() if k != "sha256"})
    assert not external_usdt_income_allowed(new)


def test_later_usdt_interest_remains_entirely_external_even_with_owned_principal(snapshot):
    from freqtrade.strategy.trx_phase1.ledger import money
    from tests.strategy.trx_phase1.test_account_movement_preview import HOUR, START

    state, evidence, policy = mixed_usdt_subscription(snapshot)
    new, _ = candidate(state, evidence, policy)
    next_evidence = deepcopy(evidence)
    row = {
        **evidence["bills"]["USDT"][0],
        "billId": "later-usdt-interest",
        "bizOrderId": "later-usdt-payout",
        "businessType": "BATCH_INTEREST_USER_IN",
        "size": ".01",
        "balance": ".06",
        "cTime": str(START + 2 * HOUR + 60000),
    }
    next_evidence.update(
        execution_state=deepcopy(new),
        observed_at_ms=START + 3 * HOUR,
        bills={"USDT": [row], "USDGO": [], "TRX": []},
    )
    next_evidence["balances"]["USDT"]["available"] = ".06"
    next_policy = {
        **policy,
        "reconciliation_id": "later-yield",
        "baseline_sha256": digest(new),
        "manual_bill_ids": [],
    }
    final, _ = candidate(new, next_evidence, next_policy)
    assert final["book"] == new["book"]
    assert money(final["unassigned_offsets"]["USDT"]) == money(".01")
    receipt = final["quarantined_account_credits"]["account-credit:later-usdt-interest"]
    assert receipt["reason"] == "outside_strategy_usdt_interest"
    assert external_usdt_income_allowed(final)


def databases(tmp_path):
    states = {
        "execution": {"book": {"initial": "1000"}},
        "strategy": {"desired": "paused", "phase": "blocked"},
    }
    paths, expected = {}, {}
    for key, state in states.items():
        paths[key] = tmp_path / (key + ".sqlite")
        with sqlite3.connect(paths[key]) as db:
            db.execute("CREATE TABLE trx_state(id TEXT PRIMARY KEY,version INTEGER,state TEXT)")
            db.execute("INSERT INTO trx_state VALUES(?,?,?)", (key, 3, json.dumps(state)))
            db.execute("CREATE TABLE protected_history(value TEXT)")
            db.execute("INSERT INTO protected_history VALUES('preserve')")
        expected[key] = {"version": 3, "sha256": digest(state)}
    return states, paths, expected


def test_atomic_commit_backs_up_both_databases_and_preserves_other_tables(tmp_path):
    old, paths, expected = databases(tmp_path)
    new_strategy = {"desired": "paused", "phase": "paused", "error": None}
    versions = persist(
        paths["execution"],
        paths["strategy"],
        expected,
        {"book": {"initial": "1000"}, "record": 1},
        new_strategy,
        tmp_path / "backup",
    )
    assert versions == {"execution": 4, "strategy": 4}
    for key, path in paths.items():
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT version FROM trx_state").fetchone()[0] == 4
            assert db.execute("SELECT value FROM protected_history").fetchone()[0] == "preserve"
        with sqlite3.connect(tmp_path / "backup" / (key + ".sqlite")) as db:
            assert json.loads(db.execute("SELECT state FROM trx_state").fetchone()[0]) == old[key]


def test_failed_second_state_encoding_rolls_back_first_state_update(tmp_path):
    old, paths, expected = databases(tmp_path)
    with pytest.raises(TypeError):
        persist(
            paths["execution"],
            paths["strategy"],
            expected,
            {"record": 1},
            {"desired": "paused", "phase": "paused", "error": None, "not_json": {1}},
            tmp_path / "backup",
        )
    for key, path in paths.items():
        with sqlite3.connect(path) as db:
            version, raw = db.execute("SELECT version,state FROM trx_state").fetchone()
        assert version == 3 and json.loads(raw) == old[key]


def test_stale_version_never_writes_and_running_strategy_is_rejected(tmp_path):
    old, paths, expected = databases(tmp_path)
    expected["execution"]["version"] = 2
    with pytest.raises(ExecutionError, match="movement_backup_baseline_changed"):
        persist(
            paths["execution"],
            paths["strategy"],
            expected,
            {"record": 1},
            {"desired": "paused", "phase": "paused", "error": None},
            tmp_path / "backup",
        )
    with pytest.raises(ExecutionError, match="movement_commit_must_remain_paused"):
        persist(
            paths["execution"],
            paths["strategy"],
            expected,
            {"record": 1},
            {"desired": "running", "phase": "paused", "error": None},
            tmp_path / "another-backup",
        )
    for key, path in paths.items():
        with sqlite3.connect(path) as db:
            version, raw = db.execute("SELECT version,state FROM trx_state").fetchone()
        assert version == 3 and json.loads(raw) == old[key]
