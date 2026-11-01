"""Actual-start scope, atomic revaluation and post-start-only native income."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from freqtrade.rpc.api_server import api_trx
from freqtrade.rpc.api_server.app import create_app
from freqtrade.strategy.trx_phase1 import opening_adoption, startup_basis
from freqtrade.strategy.trx_phase1.account import AccountReader, AccountSettings
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategy
from tests.strategy.trx_phase1.test_adoption_preview import NOW
from tests.strategy.trx_phase1.test_live_execution import ExchangeFixture
from tests.strategy.trx_phase1.test_live_income import batch_credit
from tests.strategy.trx_phase1.test_live_strategy import Market
from tests.strategy.trx_phase1.test_opening_adoption import ACCOUNT, capture_context, context


def adopted(tmp_path):
    svc, observed, captured = context(tmp_path)
    receipt = opening_adoption.adopt(None, svc, capture=captured)
    return svc, observed, receipt


def fresh(observed, now, price="0.4"):
    observed = deepcopy(observed)
    snap = observed["snapshot"]
    snap.update(observed_at=now - 1000, completed_at=now, history_scope="current_positions_only")
    snap["savings"]["until"] = now - 1000
    snap["savings"]["coverage"]["windows"][0]["until"] = now - 1000
    for coin, quote in snap["prices"].items():
        quote.update(
            timestamp=now,
            bid=price if coin == "TRX" else "1.002",
            ask="0.4001" if coin == "TRX" else "1.0021",
        )
    for row in snap["balances"]:
        if row["coin"] == "TRX":
            row.update(balance="3", available="3")
    return observed


def captured_start(svc, observed, now):
    current = fresh(observed, now)
    value = startup_basis.project(current, *startup_basis.context(svc.root), now)
    value.update(
        captured_at=now,
        opening_quotes=deepcopy(current["snapshot"]["prices"]),
        credentials_identity=svc.credential_identity(),
    )
    assert value["reviewable"]
    return value


def authorize_fixture(svc, captured):
    """Test-only authorization; no real credentials or broker are used."""
    svc.clock = lambda: captured["captured_at"]
    svc.save(
        {
            **svc.load(),
            "trading_authorized": True,
            "review_hash": "a" * 64,
            "review_expires_at": svc.clock() + 300000,
            "startup_scope_hash": captured["scope_hash"],
        }
    )


def test_preview_is_fresh_provisional_capital_and_never_changes_the_adoption(tmp_path):
    svc, observed, original = adopted(tmp_path)
    before = {
        p.name: p.read_bytes() for p in (svc.root / "live_execution" / ACCOUNT).glob("*.sqlite")
    }
    first = captured_start(svc, observed, NOW + 10000)
    assert first["started_at"] is None and not first["performance_started"]
    assert first["startup_actions"] == [] and not first["confirmation_available"]
    assert first["manifest"]["strategy_cash_usdt"] == "0"
    assert first["manifest"]["excluded_assets"][0]["quantity"] == "1000"
    trx = first["manifest"]["assets"][0]
    assert money(trx["quantity"]) == money("1003.00000001")
    assert trx["startup_unit_cost_usdt"] == "0.4"
    assert first["manifest"]["hypothetical_opening_profits"]["earn_net_usdt"] == "0"
    second = startup_basis.project(
        fresh(observed, NOW + 20000, "0.39"), *startup_basis.context(svc.root), NOW + 20000
    )
    assert first["manifest"]["initial_equity_usdt"] != second["manifest"]["initial_equity_usdt"]
    assert first["scope_hash"] == second["scope_hash"]  # Repricing does not freeze an old cost.
    assert before == {
        p.name: p.read_bytes() for p in (svc.root / "live_execution" / ACCOUNT).glob("*.sqlite")
    }
    assert opening_adoption.status(svc.root)["receipt"] == original


def test_authorized_but_unstarted_account_cannot_import_pre_start_income(tmp_path):
    svc, _, _ = adopted(tmp_path)
    svc.save({**svc.load(), "trading_authorized": True})
    before = {
        p.name: p.read_bytes() for p in (svc.root / "live_execution" / ACCOUNT).glob("*.sqlite")
    }
    broker_factory = MagicMock()
    svc.broker_factory = broker_factory
    with pytest.raises(ExecutionError, match="actual_start_required_before_reconciliation"):
        svc.reconcile()
    broker_factory.assert_not_called()
    assert before == {
        p.name: p.read_bytes() for p in (svc.root / "live_execution" / ACCOUNT).glob("*.sqlite")
    }


def test_actual_start_rebuilds_exact_costs_and_anchor_without_a_buy(tmp_path):
    svc, observed, original = adopted(tmp_path)
    value = captured_start(svc, observed, NOW + 10000)
    authorize_fixture(svc, value)
    basis = startup_basis.commit(svc, value)
    control, execution, strategy = startup_basis.context(svc.root)
    book = Ledger.restore(execution["book"])
    assert basis["started_at"] == strategy["activated_at"] == execution["created_at"] == NOW + 10000
    assert execution["opening_adoption"] == original
    assert book.principal == money("1003.00000001") and book.usdgo == money("10000.32345679")
    assert book.initial == book.principal * money(".4") + book.usdgo * money("1.002")
    assert book.equity(".4") == book.initial == book.fund_units
    assert book.cash == book.fees == book.realized == book.stable_interest == book.interest_trx == 0
    assert book.orders == [] and execution["flows"] == []
    assert execution["flow"] is execution["operation"] is strategy["intent"] is None
    assert execution["anchor"] == "0.4" and strategy["engine"]["anchor"] == 0.4
    assert strategy["startup_complete"] and strategy["engine"]["pending"] is None
    assert all(lot.remaining_cost == lot.quantity * money(".4") for lot in book.lots)
    assert book.summary(".39")["unrealized_profit"] < 0
    assert execution["unassigned_offsets"]["USDT"] == "700"
    assert control["setup"]["capital"] == str(book.initial)
    assert opening_adoption.status(svc.root)["receipt"] == original
    assert startup_basis.commit(svc, None) == basis  # Recovery cannot reprice the position.


def test_no_authorization_or_stale_quote_can_commit_a_start(tmp_path):
    svc, observed, _ = adopted(tmp_path)
    value = captured_start(svc, observed, NOW + 10000)
    before = svc.control.read_bytes()
    with pytest.raises(ExecutionError, match="not_authorized"):
        startup_basis.commit(svc, value)
    assert svc.control.read_bytes() == before
    authorize_fixture(svc, value)
    svc.clock = lambda: value["captured_at"] + 20000
    with pytest.raises(ExecutionError, match="snapshot_expired"):
        startup_basis.commit(svc, value)
    assert startup_basis.context(svc.root)[1].get("performance_basis") is None


def test_mid_transaction_failure_rolls_back_both_journals(tmp_path, monkeypatch):
    svc, observed, _ = adopted(tmp_path)
    value = captured_start(svc, observed, NOW + 10000)
    authorize_fixture(svc, value)
    before = startup_basis.context(svc.root)
    dumps = json.dumps

    def crash(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("schema") == "trx-c1-live-strategy-v1":
            raise KeyboardInterrupt("Simulated failure after execution UPDATE")
        return dumps(value, *args, **kwargs)

    monkeypatch.setattr(startup_basis.json, "dumps", crash)
    with pytest.raises(KeyboardInterrupt):
        startup_basis.commit(svc, value)
    assert startup_basis.context(svc.root) == before


def test_control_marker_failure_recovers_the_same_committed_basis(tmp_path, monkeypatch):
    svc, observed, _ = adopted(tmp_path)
    value = captured_start(svc, observed, NOW + 10000)
    authorize_fixture(svc, value)
    write = startup_basis.atomic_json
    monkeypatch.setattr(
        startup_basis, "atomic_json", MagicMock(side_effect=OSError("marker failed"))
    )
    with pytest.raises(OSError):
        startup_basis.commit(svc, value)
    stored = startup_basis.context(svc.root)[1]["performance_basis"]
    monkeypatch.setattr(startup_basis, "atomic_json", write)
    svc.recover_startup_control(svc.load())
    assert svc.load()["performance_basis_sha256"] == stored["sha256"]
    assert startup_basis.context(svc.root)[1]["performance_basis"] == stored


def test_current_positions_reader_never_queries_old_income_or_fills():
    client = MagicMock()
    reader = AccountReader(
        client, AccountSettings(include_earn=True), NOW, current_positions_only=True
    )
    reader.savings = MagicMock(return_value={"rows": []})
    reader.products = MagicMock(return_value={"rows": []})
    reader.prices = MagicMock(return_value={})
    reader.recent_trading = MagicMock(return_value={"rows": []})
    reader.savings_records = MagicMock(side_effect=AssertionError("Historical income is forbidden"))
    reader.funding_records = MagicMock(side_effect=AssertionError("Historical income is forbidden"))
    data, checks = reader.collect()
    assert data["savings_records"] is data["fills"] is data["conversion_fills"] is None
    assert checks["savings_records"]["status"] == "excluded_before_start"
    reader.savings_records.assert_not_called()
    assert all(not call.kwargs.get("fills") for call in reader.recent_trading.call_args_list)


def test_final_capture_uses_current_positions_only_and_latest_quotes(tmp_path):
    svc, account, client = capture_context(tmp_path)
    raw = account.refresh.return_value
    initial = startup_basis.adoption_preview(raw, now_ms=svc.clock())
    initial["opening_quotes"] = raw["snapshot"]["prices"]
    opening_adoption.adopt(
        None, svc, capture=lambda *_: (initial, ACCOUNT, svc.credential_identity(), svc.clock())
    )
    account.status = MagicMock(return_value=raw)
    value = startup_basis.capture(account, svc)
    account.refresh.assert_called_once_with(current_positions_only=True)
    assert value["manifest"]["assets"][0]["startup_unit_cost_usdt"] == "0.4"
    client.privateEarnGetV2EarnSavingsRecords.assert_not_called()


@pytest.mark.parametrize("bgb_on_at_start", [False, True])
def test_future_explicit_start_reprices_after_review_without_first_purchase(
    tmp_path, monkeypatch, bgb_on_at_start
):
    svc, observed, _ = adopted(tmp_path)
    clock = [NOW + 10000]
    svc.clock = lambda: clock[0]
    products = [
        {
            **r,
            "redemption_minimum": "0.1",
            "redemption_decimals": 8,
            "redemption_evidence": "fixture-only",
        }
        for r in svc.load()["setup"]["products"]
    ]
    svc.save(
        {
            **svc.load(),
            "trading_authorized": True,
            "setup": {**svc.load()["setup"], "products": products},
        }
    )
    broker = ExchangeFixture(tmp_path / "provider.sqlite")
    broker.account_id = lambda: ACCOUNT
    terms = broker.product
    broker.product = lambda coin, _: {**terms(coin, coin + "-flex"), "tiers": []}
    reads = []

    def factory(writes):
        reads.append(writes)
        broker.writes_enabled = writes
        broker.bgb_deduction = "on" if writes and bgb_on_at_start else "off"
        return broker

    svc.broker_factory = factory
    svc.market_factory = lambda: Market(broker, clock[0])
    svc.seed = svc.root / "fixture-history.feather"
    svc.market_factory().data.to_feather(svc.seed)
    svc.redemption_verified = lambda *_: clock[0]
    price = ["0.4"]

    def current():
        captured = captured_start(svc, observed, clock[0])
        if price[0] != "0.4":
            raw = fresh(observed, clock[0], price[0])
            for row in raw["snapshot"]["balances"]:
                if row["coin"] == "TRX":
                    row.update(balance="4", available="4")
                if row["coin"] == "USDT":
                    row.update(balance="1111", available="1111")
            captured = startup_basis.project(raw, *startup_basis.context(svc.root), clock[0])
            captured.update(
                captured_at=clock[0],
                opening_quotes=raw["snapshot"]["prices"],
                credentials_identity=svc.credential_identity(),
            )
        return captured

    svc.capture_startup = current
    before = startup_basis.context(svc.root)[1]
    checked = svc.prepare(svc.load()["setup"])
    assert startup_basis.context(svc.root)[1] == before
    assert reads == [False] and broker.state()["writes"] == []
    # Quotes move after review. The actual start must use .39, not .4 or the adoption price.
    price[0] = ".39"
    clock[0] += 1000
    started = []
    monkeypatch.setattr(
        LiveStrategy, "start", lambda runtime: started.append(runtime.state["performance_basis"])
    )
    if bgb_on_at_start:
        with pytest.raises(ExecutionError, match="disable_bgb_deduction_before_start"):
            svc.start(checked["prepared"]["review_hash"])
        assert started == [] and startup_basis.context(svc.root)[1] == before
        assert broker.state()["writes"] == []
        return
    svc.start(checked["prepared"]["review_hash"])
    assert len(started) == 1
    assert started[0]["quotes"]["TRX"]["bid"] == ".39"
    _, execution, strategy = startup_basis.context(svc.root)
    assert execution["book"]["orders"] == [] and execution["flows"] == []
    assert strategy["startup_complete"] and strategy["intent"] is None
    assert strategy["engine"]["anchor"] == 0.39
    assert strategy["engine"]["accept_new_buys"]
    assert money(execution["book"]["lots"][0]["quantity"]) == 4
    assert execution["unassigned_offsets"]["USDT"] == "1111"
    assert reads == [False, False, True] and broker.state()["writes"] == []


def test_authenticated_cached_preview_cannot_start_or_write(tmp_path, monkeypatch):
    svc, observed, _ = adopted(tmp_path)
    current = fresh(observed, NOW + 10000)
    service = SimpleNamespace(
        config_path=svc.root / "bitget-account.local.json",
        allocation_context=lambda: (current, ACCOUNT),
        clock=lambda: (NOW + 10000) / 1000,
    )
    monkeypatch.setattr(
        startup_basis, "commit", MagicMock(side_effect=AssertionError("GET must not commit"))
    )
    app = create_app(
        {
            "user_data_dir": str(tmp_path),
            "api_server": {
                "username": "reader",
                "password": "password",
                "jwt_secret_key": "x" * 40,
            },
        }
    )
    app.dependency_overrides[api_trx.get_account_service] = lambda: service
    client = TestClient(app)
    url = "/api/v1/trx/account/startup-preview"
    assert client.get(url).status_code == 401
    result = client.get(url, auth=("reader", "password"))
    assert result.status_code == 200 and result.json()["preview_only"]
    assert not any(
        r.path.endswith("/account/startup-preview") and "POST" in r.methods
        for r in api_trx.router.routes
    )
    assert startup_basis.context(svc.root)[2]["activated_at"] is None


@pytest.mark.parametrize("coin", ["TRX", "USDGO"])
def test_pre_start_and_cross_start_receipts_excluded_and_later_interest_counted_once(
    tmp_path, coin
):
    svc, observed, _ = adopted(tmp_path)
    hour = startup_basis.HOUR
    start = NOW // hour * hour + hour + 15 * 60000
    value = captured_start(svc, observed, start)
    authorize_fixture(svc, value)
    startup_basis.commit(svc, value)
    broker = ExchangeFixture(tmp_path / "provider.sqlite")
    broker.account_id = lambda: ACCOUNT
    runtime = LiveExecution(svc.root, broker, ACCOUNT)
    remote = broker.state()
    remote["wallet"] = {"TRX": "3", "USDGO": ".20000001", "USDT": "700"}
    remote["holdings"] = {
        key: {
            "order_id": key,
            "coin": p["coin"],
            "product_id": p["product_id"],
            "amount": str(money(p["principal"]) + money(p["interest"])),
        }
        for key, p in runtime.state["holdings"].items()
    }
    broker.put(remote)
    broker.savings_records = MagicMock(
        side_effect=AssertionError("Daily history must not be imported")
    )
    # The bill arrives after start but pays the prior, entirely pre-start hour.
    batch_credit(broker, coin, ".01", start + 1000)
    runtime.import_interest(start + 2000)
    assert money(runtime.state["book"]["interest_trx"]) == 0
    assert money(runtime.state["book"]["stable_interest"]) == 0
    assert not runtime.state["pending_income"]
    # The next bill overlaps the start boundary: preserve the asset outside PnL.
    overlap = start // hour * hour + hour + 1000
    batch_credit(broker, coin, ".02", overlap)
    runtime.import_interest(overlap + 1000)
    assert len(runtime.state["pending_income"]) == 1
    assert len(runtime.state["income_accounting"]["cross_start_receipts"]) == 1
    full = overlap + hour
    batch_credit(broker, coin, ".03", full)
    runtime.import_interest(full + 1000)
    book = Ledger.restore(runtime.state["book"])
    assert book.interest_trx == (money(".03") if coin == "TRX" else 0)
    assert book.stable_interest == (money(".03") * money("1.001") if coin == "USDGO" else 0)
    before = deepcopy(runtime.state["book"])
    runtime.import_interest(full + 2000)
    assert runtime.state["book"] == before and broker.state()["writes"] == []
    assert runtime.state["unassigned_offsets"][coin] == ".03" or money(
        runtime.state["unassigned_offsets"][coin]
    ) == money(".03")


@pytest.mark.parametrize(
    "problem", [None, "wrong_group", "fee", "missing_wallet", "deposit", "changed_scope"]
)
def test_external_usdt_interest_reconciles_without_changing_start_basis(tmp_path, problem):
    svc, observed, _ = adopted(tmp_path)
    start = NOW + 10000
    value = captured_start(svc, observed, start)
    authorize_fixture(svc, value)
    basis = startup_basis.commit(svc, value)
    broker = ExchangeFixture(tmp_path / "provider.sqlite")
    broker.account_id = lambda: ACCOUNT
    runtime = LiveExecution(svc.root, broker, ACCOUNT)
    remote = broker.state()
    remote["wallet"] = {"TRX": "3", "USDGO": ".20000001", "USDT": "700"}
    remote["holdings"] = {
        key: {
            "order_id": key,
            "coin": p["coin"],
            "product_id": p["product_id"],
            "amount": p["principal"],
        }
        for key, p in runtime.state["holdings"].items()
    }
    broker.put(remote)
    batch_credit(broker, "USDT", ".00026979", start + 1000)
    remote = broker.state()
    if problem == "wrong_group":
        remote["bills"][-1]["groupType"] = "other"
    elif problem == "fee":
        remote["bills"][-1]["fees"] = ".00000001"
    elif problem == "missing_wallet":
        remote["wallet"]["USDT"] = "700"
    elif problem == "deposit":
        remote["bills"][-1]["businessType"] = "DEPOSIT"
    elif problem == "changed_scope":
        runtime.state["performance_basis"]["income_policy"]["usdt_is_outside_strategy"] = False
        runtime.save()
    broker.put(remote)
    before = runtime.status()
    if problem:
        with pytest.raises(ExecutionError):
            runtime.import_interest(start + 2000)
        assert runtime.status() == before
        return
    assert runtime.import_interest(start + 2000) == 1
    once = runtime.status()
    assert once["book"] == before["book"] and once["performance_basis"] == basis
    assert once["holdings"] == before["holdings"] and once["pending_income"] == {}
    assert money(once["unassigned_offsets"]["USDT"]) == money("700.00026979")
    credit = next(iter(once["quarantined_account_credits"].values()))
    assert credit["type"] == "BATCH_INTEREST_USER_IN"
    assert credit["reason"] == "outside_strategy_usdt_interest"
    assert money(credit["offset_delta"]) == money(".00026979")
    recovered = LiveExecution(svc.root, broker, ACCOUNT)
    assert recovered.import_interest(start + 3000) == 0
    assert recovered.status()["quarantined_account_credits"] == once["quarantined_account_credits"]
    assert recovered.status()["book"] == before["book"]
    assert broker.state()["writes"] == []
    remote["bills"][-1]["size"] = ".00026980"
    broker.put(remote)
    with pytest.raises(ExecutionError, match="conflicting_account_credit"):
        recovered.import_interest(start + 4000)
    assert recovered.status()["book"] == before["book"]
