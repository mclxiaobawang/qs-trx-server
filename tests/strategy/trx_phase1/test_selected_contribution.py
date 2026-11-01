"""Fixed source, native commit, preserved accounting and uncertain-result recovery."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.selected_contribution import (
    CoreRejected,
    CoreUnavailable,
    SelectedContribution,
    SelectedContributionError,
    fixed_policy,
)
from tests.strategy.trx_phase1.test_contribution_preview import states as states  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_contribution import execution as execution  # noqa: PLC0414
from tests.strategy.trx_phase1.test_live_history import save


@pytest.fixture
def setup(execution, states, tmp_path):
    e, s, now = states
    save(execution.root / "strategy.sqlite", "strategy", s)
    config = {
        "source": str(execution.root),
        "state": str(tmp_path / "private-reviews"),
        "account_id": e["account_id"],
        "ledger_created_at": e["created_at"],
        "performance_started_at": s["activated_at"],
        "source_sha256": s["source_sha256"],
        "initial_capital": e["book"]["initial"],
        "selection": {
            "batch_id": "one-fixed-batch",
            "coin": "USDGO",
            "amount": "0.5",
            "holding_id": "USDGO",
            "product_id": "USDGO-flex",
        },
        "core_origin": "http://trx-phase1-live:8080",
        "public_origin": "https://trx.octo-agent.com",
        "username": "test-user",
        "jwt_secret_key": "test-key-" * 8,
    }
    # Exercise the legacy core used by the unchanged old trading container.
    path = Path("output/trx-cloud-deployment-2026-10-18/contribution-release/live_contribution.py")
    if path.exists():
        spec = importlib.util.spec_from_file_location("legacy_contribution", path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        native_apply = legacy.apply
    else:
        from freqtrade.strategy.trx_phase1.live_contribution import apply

        native_apply = apply

    class Core:
        mode = "success"
        calls = 0

        def confirm(self, review_hash, authorization):
            assert authorization == "Bearer test"
            self.calls += 1
            if self.mode == "reject":
                raise CoreRejected("contribution_requires_idle_strategy")
            if self.mode == "timeout":
                raise CoreUnavailable()
            try:
                record = native_apply(execution, review_hash, clock[0])
            except ExecutionError as exc:
                raise CoreRejected(str(exc)) from None
            if self.mode == "committed_timeout":
                raise CoreUnavailable()
            if self.mode == "committed_error":
                raise CoreRejected("runtime_sync_failed")
            return record

    clock = [now]
    core = Core()
    service = SelectedContribution(config, core=core, clock=lambda: clock[0])
    return service, core, clock, execution, config


def test_prepare_is_private_and_never_writes_native_journals(setup):
    service, core, _, execution, _ = setup
    before = {p: p.read_bytes() for p in execution.root.glob("*.sqlite")}
    provider = execution.broker.state()
    review = service.prepare()
    assert {p: p.read_bytes() for p in before} == before
    assert execution.broker.state() == provider and core.calls == 0
    assert review["fixed_selection"] and review["requires_user_confirmation"]
    assert len(review["assets"]) == 1 and review["assets"][0]["amount"] == "0.5"
    assert review["assets"][0]["locations"] == [
        {"location": "earn", "holding_id": "USDGO", "product_id": "USDGO-flex", "amount": "0.5"}
    ]
    assert service.status()["status"] == "not_confirmed"


@pytest.mark.parametrize(
    "selection",
    [
        {"coin": "TRX"},
        {"amount": "0.6"},
        {"holding_id": "TRX"},
        {"product_id": "wrong"},
        {"account_id": "other"},
        {"amount": None},
    ],
)
def test_selection_cannot_be_broadened(setup, selection):
    service, core, _, _, _ = setup
    with pytest.raises(SelectedContributionError, match="fixed_selection_mismatch"):
        service.prepare(selection)
    assert core.calls == 0


def test_private_policy_preserves_approved_batch_fingerprint(setup):
    service, core, clock, execution, config = setup
    policy = {
        key: deepcopy(config[key])
        for key in (
            "account_id",
            "ledger_created_at",
            "performance_started_at",
            "source_sha256",
            "initial_capital",
            "selection",
        )
    }
    (execution.root / "selected-contribution-policy.local.json").write_text(json.dumps(policy))
    restored = SelectedContribution(fixed_policy(execution.root), core=core, clock=lambda: clock[0])
    assert restored.policy_hash == service.policy_hash
    assert restored.scope == service.scope
    assert restored.source == execution.root
    assert restored.folder == execution.root / "selected-contribution"
    assert core.calls == 0


@pytest.mark.parametrize("invalid", ["missing", "broken", "source", "coin", "amount", "capital"])
def test_private_policy_cannot_default_or_broaden_a_financial_batch(setup, invalid):
    _, core, _, execution, config = setup
    policy = {
        key: deepcopy(config[key])
        for key in (
            "account_id",
            "ledger_created_at",
            "performance_started_at",
            "source_sha256",
            "initial_capital",
            "selection",
        )
    }
    path = execution.root / "selected-contribution-policy.local.json"
    before = execution.status()
    if invalid == "source":
        policy["source"] = "/another/account"
    elif invalid == "coin":
        policy["selection"]["coin"] = "TRX"
    elif invalid == "amount":
        policy["selection"]["amount"] = "-0.5"
    elif invalid == "capital":
        policy["initial_capital"] = "NaN"
    if invalid != "missing":
        path.write_text("{" if invalid == "broken" else json.dumps(policy))
    with pytest.raises(SelectedContributionError, match="private_policy_required"):
        fixed_policy(execution.root)
    assert execution.status() == before and core.calls == 0


@pytest.mark.parametrize("mode", ["success", "committed_timeout", "committed_error"])
def test_exact_batch_commits_once_through_native_writer_and_keeps_basis(setup, mode):
    service, core, clock, execution, _ = setup
    review = service.prepare()
    before = deepcopy(execution.status())
    provider = execution.broker.state()
    core.mode = mode
    clock[0] += 1000
    record = service.confirm(review["review_hash"], "Bearer test")
    after = execution.status()
    old, new = Ledger.restore(before["book"]), Ledger.restore(after["book"])
    value = money(record["value"])
    assert new.usdgo - old.usdgo == money(".5")
    assert new.usdgo_cost - old.usdgo_cost == value
    assert new.locked["USDGO"] - old.locked["USDGO"] == money(".5")
    assert new.external_flows - old.external_flows == value
    assert new.fund_units > old.fund_units
    for key in ["lots", "initial", "orders"]:
        assert before["book"][key] == after["book"][key]
    assert before["holdings"]["TRX"] == after["holdings"]["TRX"]
    assert before["unassigned_offsets"] == after["unassigned_offsets"]
    assert money(after["holdings"]["USDGO"]["unassigned_amount"]) == money(".4")
    assert before["income_accounting"]["allocations"] == after["income_accounting"]["allocations"]
    assert service.read("strategy")[1]["activated_at"] == service.scope["performance_started_at"]
    before_nav = money(review["projection"]["before"]["nav"])
    new.usdgo_mark = money(record["assets"][0]["mark"])
    assert abs(new.equity(".3") / new.fund_units - before_nav) < money("1e-24")
    assert service.confirm(review["review_hash"], "Bearer test") == record
    assert execution.status() == after and core.calls == 1
    assert execution.broker.state() == provider
    assert service.status()["status"] == "confirmed"
    with pytest.raises(SelectedContributionError, match="already_confirmed"):
        service.prepare()


def test_multiple_drafts_and_restart_cannot_contribute_same_batch_twice(setup):
    service, core, clock, execution, config = setup
    first = service.prepare()
    clock[0] += 1
    second = service.prepare()
    record = service.confirm(first["review_hash"], "Bearer test")
    restarted = SelectedContribution(config, core=core, clock=lambda: clock[0])
    assert restarted.confirm(second["review_hash"], "Bearer test") == record
    assert core.calls == 1 and len(execution.status()["asset_contributions"]) == 1


@pytest.mark.parametrize("resume_expired", [False, True])
def test_unknown_result_keeps_same_review_and_reconciles_without_duplicate(setup, resume_expired):
    service, core, clock, execution, config = setup
    first = service.prepare()
    clock[0] += 1
    second = service.prepare()
    before = execution.status()
    core.mode = "timeout"
    with pytest.raises(SelectedContributionError, match="pending_check_status"):
        service.confirm(first["review_hash"], "Bearer test")
    assert execution.status()["book"] == before["book"]
    restarted = SelectedContribution(config, core=core, clock=lambda: clock[0])
    assert restarted.prepare()["review_hash"] == first["review_hash"]
    with pytest.raises(SelectedContributionError, match="previous_confirmation_pending"):
        restarted.confirm(second["review_hash"], "Bearer test")
    core.mode = "success"
    if resume_expired:
        clock[0] = first["expires_at"] + 1
        with pytest.raises(SelectedContributionError, match="review_expired"):
            restarted.confirm(first["review_hash"], "Bearer test")
        assert restarted.status()["status"] == "not_confirmed"
        assert execution.status()["book"] == before["book"]
    else:
        restarted.confirm(first["review_hash"], "Bearer test")
        assert restarted.status()["status"] == "confirmed"
        assert len(execution.status()["asset_contributions"]) == 1


def test_definitive_rejection_removes_only_own_metadata(setup):
    service, core, _, execution, _ = setup
    review = service.prepare()
    before = execution.status()
    core.mode = "reject"
    with pytest.raises(SelectedContributionError, match="requires_idle_strategy"):
        service.confirm(review["review_hash"], "Bearer test")
    after = execution.status()
    assert service.status()["status"] == "not_confirmed"
    assert {k: v for k, v in before.items() if k not in service.METADATA} == {
        k: v for k, v in after.items() if k not in service.METADATA
    }


@pytest.mark.parametrize(
    "fault,code",
    [
        ("expiry", "expired"),
        ("price", "price_changed"),
        ("basis", "basis_changed"),
        ("busy", "requires_idle"),
        ("hash", "not_found"),
    ],
)
def test_changed_or_unknown_review_never_calls_core(setup, fault, code):
    service, core, clock, execution, _ = setup
    review = service.prepare()
    if fault == "expiry":
        clock[0] = review["expires_at"]
    elif fault in {"price", "busy"}:
        s = service.read("strategy")[1]
        if fault == "price":
            s["markets"]["USDGOUSDT"]["bid"] = "1.02"
        else:
            s["engine"]["pending"] = {"id": "new-order"}
        save(execution.root / "strategy.sqlite", "strategy", s)
    elif fault == "basis":
        e = execution.status()
        e["book"]["initial"] = "9"
        save(execution.root / "execution.sqlite", "execution", e)
    else:
        review["review_hash"] = "0" * 64
    before = execution.status()
    with pytest.raises(SelectedContributionError, match=code):
        service.confirm(review["review_hash"], "Bearer test")
    assert core.calls == 0 and execution.status() == before


def test_native_service_serializes_confirmation_with_worker_and_syncs_book(setup, monkeypatch):
    import threading
    from types import SimpleNamespace

    from freqtrade.strategy.trx_phase1 import live_contribution, selected_contribution
    from freqtrade.strategy.trx_phase1.config import ResearchConfig
    from freqtrade.strategy.trx_phase1.live_service import LiveService

    helper, _, clock, execution, config = setup
    state = helper.read("strategy")[1]
    runtime = SimpleNamespace(
        lock=threading.RLock(),
        state=state,
        execution=execution,
        engine=SimpleNamespace(
            pending=None, config=ResearchConfig.from_saved(state["engine"]["config"])
        ),
    )
    runtime.state["terms"] = {}
    synced = []
    runtime.sync_book = lambda funding: synced.append(funding.book.state())
    runtime.persist = lambda: None
    service = LiveService.__new__(LiveService)
    service.lock = threading.RLock()
    service.root = execution.root.parents[1]
    service.clock = lambda: clock[0]
    service.load = lambda: {"account_id": config["account_id"], "credentials_identity": "same"}
    service.credential_identity = lambda: "same"
    service.get_runtime = lambda control: runtime
    reads = []
    service.broker_factory = lambda writes: reads.append(writes) or execution.broker
    service.close_broker = lambda broker: None
    monkeypatch.setattr(selected_contribution, "fixed_policy", lambda source: config)
    apply = live_contribution.apply

    def locked_apply(*args):
        assert runtime.lock._is_owned() and service.lock._is_owned()
        return apply(*args)

    monkeypatch.setattr(live_contribution, "apply", locked_apply)
    review = service.selected_contribution()
    before = execution.status()
    assert not synced and reads == [False]
    assert execution.status() == before
    result = service.selected_contribution(
        review["review_hash"], {"coin": "USDGO", "amount": "0.5"}
    )
    assert synced[-1] == execution.status()["book"]
    assert service.selected_contribution(status_only=True)["record"] == result
    assert reads == [False, False, False]
    assert before["book"]["lots"] == execution.status()["book"]["lots"]


def test_selected_api_rejects_changed_amount_and_unknown_review(setup):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from freqtrade.rpc.api_server import api_trx
    from freqtrade.strategy.trx_phase1.live_broker import ExecutionError

    helper, _, _, _, _ = setup

    class Service:
        def selected_contribution(self, review_hash=None, selection=None, status_only=False):
            try:
                helper.selection(selection)
                return (
                    helper.status()
                    if status_only
                    else helper.confirm(review_hash, "Bearer test")
                    if review_hash
                    else helper.prepare(selection)
                )
            except SelectedContributionError as exc:
                raise ExecutionError(str(exc)) from None

    app = FastAPI()
    app.include_router(api_trx.router, prefix="/api/v1")
    app.dependency_overrides[api_trx.get_live_service] = lambda: Service()
    client = TestClient(app)
    prefix = "/api/v1/trx/live/contribution/selected/"
    assert client.post(prefix + "preview", json={"coin": "TRX"}).status_code == 409
    assert client.post(prefix + "preview", json={"amount": 1}).status_code == 422
    assert client.post(prefix + "preview", json={"all": True}).status_code == 422
    assert client.post(prefix + "confirm", json={}).status_code == 422
    assert client.post(prefix + "confirm", json={"review_hash": "0" * 64}).status_code == 409
    assert (
        client.post(prefix + "confirm", json={"review_hash": "0" * 64, "amount": ".9"}).status_code
        == 409
    )
    assert client.post(prefix + "preview").json()["assets"][0]["amount"] == "0.5"
    assert client.get(prefix + "status").json()["status"] == "not_confirmed"
