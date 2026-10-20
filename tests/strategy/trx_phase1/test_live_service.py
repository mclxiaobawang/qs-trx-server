import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from freqtrade.rpc.api_server import api_trx
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_service import LiveService, LiveSetup
from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategy
from tests.strategy.trx_phase1.test_live_execution import NOW, RULES, ExchangeFixture
from tests.strategy.trx_phase1.test_live_strategy import Market, candidate


@pytest.fixture
def service(tmp_path):
    broker = ExchangeFixture(tmp_path / "provider.sqlite")
    original = broker.product
    tiers = [{"minStepVal": "0", "maxStepVal": "10000000", "currentApy": "10"}]
    broker.product = lambda coin, pid: {**original(coin, pid), "tiers": tiers}
    clock = [NOW]
    reads = []

    def factory(writes):
        reads.append(writes)
        broker.writes_enabled = writes
        return broker

    market = Market(broker, NOW)
    result = LiveService(
        tmp_path, broker_factory=factory, market_factory=lambda: market, clock=lambda: clock[0]
    )
    result.root.mkdir()
    result.source_path = result.root / "source.json"
    result.source_path.write_text(json.dumps({"config": candidate().model_dump()}))
    result.seed = result.root / "seed.feather"
    market.data.to_feather(result.seed)
    private = result.root / "bitget-account.local.json"
    private.write_text(
        json.dumps(
            {
                "account_type": "classic",
                "api_key": "test-key",
                "secret_key": "test-secret",
                "passphrase": "test-pass",
            }
        )
    )
    private.chmod(0o600)
    return result, broker, clock, reads, tiers


def setup():
    return {"capital": "1000", "products": RULES}


def test_page_status_does_not_construct_broker_allocate_or_start(service):
    svc, broker, _, reads, _ = service
    result = svc.status()
    assert result["prepared"] is None and result["runtime"] is None
    assert not result["real_trading_running"] and reads == []
    assert not (svc.root / "live_execution").exists()
    assert broker.state()["writes"] == []


def test_contribution_preview_and_confirm_use_reader_and_preserve_strategy(service):
    from freqtrade.strategy.trx_phase1.ledger import Ledger, money
    from tests.strategy.trx_phase1.test_income_allocation import approve

    svc, broker, clock, reads, _ = service
    state = broker.state()
    state["wallet"].update(TRX="4.46933112", USDGO="1.000153")
    broker.put(state)
    svc.prepare(setup())
    runtime = svc.runtime
    runtime.state.update(startup_complete=True, phase="prepared")
    runtime.persist()
    approve(runtime.execution)
    clock[0] += 2 * 3600000
    before = runtime.execution.status()
    engine = runtime.engine.state()
    review = svc.contribution()
    assert runtime.execution.status()["book"] == before["book"]
    clock[0] += 1000
    record = svc.contribution(review["review_hash"])
    assert money(record["value"]) > 2
    assert reads == [False, False, False] and broker.state()["writes"] == []
    assert runtime.state["startup_complete"]
    assert runtime.state["phase"] == "prepared" and runtime.state["desired"] == "paused"
    assert runtime.engine.anchor == engine["anchor"]
    assert runtime.engine.buy_at == engine["buy_at"]
    assert runtime.engine.sell_at == engine["sell_at"]
    assert runtime.execution.status()["book"]["orders"] == before["book"]["orders"]
    assert runtime.engine.book.external_flows == money(record["value"])
    assert Ledger.restore(runtime.execution.status()["book"]).initial == money("1000")
    status = svc.status()
    assert len(status["runtime"]["execution"]["asset_contributions"]) == 1
    assert status["runtime"]["summary"]["added_capital"] == float(record["value"])


def test_contribution_never_prepares_or_restarts_incomplete_startup(service):
    svc, broker, _, _, _ = service
    svc.prepare(setup())
    with pytest.raises(ExecutionError, match="contribution_requires_idle_strategy"):
        svc.contribution()
    assert not svc.runtime.state["startup_complete"]
    assert broker.state()["writes"] == []


def test_paused_reconciliation_updates_credits_observations_without_starting(service):
    from tests.strategy.trx_phase1.test_account_repair import rebate

    svc, broker, clock, reads, _ = service
    svc.prepare(setup())
    runtime = svc.runtime
    runtime.state.update(startup_complete=True, phase="prepared")
    runtime.persist()
    before = runtime.execution.status()["book"]
    clock[0] += 1000
    broker.savings_observation = {"observed_at": clock[0], "holdings": {}}
    rebate(broker, clock[0])
    result = svc.reconcile()
    assert result["runtime"]["startup_complete"]
    assert result["runtime"]["desired"] == "paused"
    assert not result["runtime"]["worker_running"]
    assert result["runtime"]["savings_observation"]["observed_at"] == clock[0]
    assert len(result["runtime"]["execution"]["quarantined_account_credits"]) == 1
    assert runtime.execution.status()["book"] == before
    assert result["runtime"]["valuation"]["observed_at"] == clock[0]
    assert svc.reconcile()["runtime"]["summary"] == result["runtime"]["summary"]
    assert reads == [False, False, False] and not broker.state()["writes"]


def test_prepare_is_read_only_and_exposes_actual_config_without_credentials(service):
    svc, broker, _, reads, _ = service
    result = svc.prepare(setup())
    assert reads == [False]
    assert not result["runtime"]["worker_running"] and not result["real_trading_running"]
    assert result["runtime"]["effective_config"]["initial_cash"] == 1000
    assert result["prepared"]["setup"]["capital"] == "1000"
    assert result["runtime"]["effective_config"]["fee_rebate_ratio"] == 0
    assert (
        result["runtime"]["indicator_initialization"]["rows_used"]
        == candidate().volatility_window + 1
    )
    assert result["runtime"]["warmup_until"] is None
    assert result["runtime"]["equity_curve"] == []
    assert result["runtime"]["summary"]["max_drawdown"] is None
    assert result["runtime"]["summary"]["trading_turnover"] == 0
    assert "credentials_identity" not in json.dumps(result)
    assert "test-key" not in json.dumps(result) and "test-secret" not in json.dumps(result)
    assert svc.control.stat().st_mode & 0o777 == 0o600
    assert broker.state()["writes"] == []
    restarted = LiveService(svc.root.parent, broker_factory=MagicMock())
    restarted.source_path = svc.source_path
    assert restarted.status()["runtime"]["phase"] == "prepared"
    restarted.broker_factory.assert_not_called()


def test_status_reads_saved_live_curve_without_sampling_or_exchange_requests(service):
    svc, broker, _, reads, _ = service
    svc.prepare(setup())
    runtime = svc.runtime
    runtime.store.observe({"timestamp": NOW, "equity": 1000, "price": 0.3}, 1)
    runtime.store.observe({"timestamp": NOW + 60000, "equity": 950, "price": 0.29}, 0.95)
    calls = reads[:]
    before = runtime.execution.status()
    for _ in range(3):
        result = svc.status()["runtime"]
        assert result["history"]["samples"] == 2
        assert result["summary"]["max_drawdown"] == pytest.approx(0.05)
        assert [p["equity"] for p in result["equity_curve"]] == [1000, 950]
    assert reads == calls
    assert runtime.execution.status() == before
    assert broker.state()["writes"] == []


def test_missing_redemption_evidence_prevents_allocation(service):
    svc, broker, _, reads, _ = service
    bad = setup()
    bad["products"] = [{**p, "redemption_minimum": None} for p in RULES]
    with pytest.raises(ValidationError):
        svc.prepare(bad)
    assert reads == [] and broker.state()["writes"] == []
    assert not (svc.root / "live_execution").exists()


def test_expired_or_wrong_review_cannot_enable_transport(service):
    svc, broker, clock, reads, _ = service
    result = svc.prepare(setup())
    with pytest.raises(ExecutionError, match="fresh_preflight_required"):
        svc.start("f" * 64)
    clock[0] = result["prepared"]["review_expires_at"]
    with pytest.raises(ExecutionError, match="fresh_preflight_required"):
        svc.start(result["prepared"]["review_hash"])
    assert all(writes is False for writes in reads)
    assert broker.state()["writes"] == []


def test_only_explicit_start_enables_writes_and_rechecks_terms(service, mocker):
    svc, broker, _, reads, _ = service
    result = svc.prepare(setup())
    start = mocker.patch.object(LiveStrategy, "start")
    svc.start(result["prepared"]["review_hash"])
    assert reads == [False, False, True]
    start.assert_called_once()
    assert svc.load()["review_expires_at"] == 0
    assert broker.state()["writes"] == []  # mocked worker; no financial activity


def test_changed_fee_or_apr_requires_showing_the_updated_review(service):
    svc, broker, _, reads, tiers = service
    result = svc.prepare(setup())
    tiers[0]["currentApy"] = "5"
    with pytest.raises(ExecutionError, match="live_terms_changed_review_required"):
        svc.start(result["prepared"]["review_hash"])
    assert all(writes is False for writes in reads)
    assert broker.state()["writes"] == []
    assert svc.status()["prepared"]["review_hash"] != result["prepared"]["review_hash"]


def test_budget_and_source_are_frozen_after_allocation(service):
    svc, broker, _, _, _ = service
    svc.prepare(setup())
    with pytest.raises(ExecutionError, match="frozen_live_setup_mismatch"):
        svc.prepare({**setup(), "capital": "900"})
    assert broker.state()["writes"] == []


def test_pause_and_existing_order_query_never_enable_new_writes(service):
    svc, broker, _, reads, _ = service
    svc.prepare(setup())
    svc.pause()
    svc.reconcile()
    assert all(w is False for w in reads)
    assert broker.state()["writes"] == []


def test_invalid_or_insufficient_product_stops_before_local_allocation(service):
    svc, broker, _, _, _ = service
    original = broker.product
    broker.product = lambda coin, pid: {**original(coin, pid), "remainingAmount": "0"}
    with pytest.raises(ExecutionError, match="startup_product_quota_or_minimum_unsatisfied"):
        svc.prepare(setup())
    assert not (svc.root / "live_execution").exists()
    assert broker.state()["writes"] == []


def test_setup_validation_accepts_only_both_products():
    with pytest.raises(ValidationError):
        LiveSetup.model_validate({"capital": "-1", "products": RULES})
    with pytest.raises(ValidationError):
        LiveSetup.model_validate({"capital": "1000", "products": [RULES[0], RULES[0]]})


def test_management_api_status_prepare_start_pause_and_reconcile_are_separate(service, mocker):
    svc, broker, _, reads, _ = service
    app = FastAPI()
    app.include_router(api_trx.router)
    app.dependency_overrides[api_trx.get_live_service] = lambda: svc
    reader = MagicMock()
    reader.allocation_context.return_value = ({}, None)
    app.dependency_overrides[api_trx.get_account_service] = lambda: reader
    with TestClient(app) as client:
        assert client.get("/trx/live").status_code == 200
        assert reads == []
        assert client.post("/trx/live/start", json={}).status_code == 422
        assert client.post("/trx/live/start", json={"review_hash": "f" * 64}).status_code == 409
        response = client.post("/trx/live/prepare", json=setup())
        assert response.status_code == 200
        assert broker.state()["writes"] == []
        start = mocker.patch.object(LiveStrategy, "start")
        assert (
            client.post(
                "/trx/live/start", json={"review_hash": response.json()["prepared"]["review_hash"]}
            ).status_code
            == 200
        )
        start.assert_called_once()
        assert client.post("/trx/live/pause").status_code == 200
        assert client.post("/trx/live/reconcile", json={}).status_code == 200
        assert broker.state()["writes"] == []


def test_api_never_exposes_a_raw_provider_error(service, mocker):
    svc = service[0]
    mocker.patch.object(svc, "products", side_effect=RuntimeError("private-api-secret"))
    app = FastAPI()
    app.include_router(api_trx.router)
    app.dependency_overrides[api_trx.get_live_service] = lambda: svc
    app.dependency_overrides[api_trx.get_account_service] = lambda: MagicMock(status=dict)
    with TestClient(app) as client:
        response = client.post("/trx/live/products")
        assert response.status_code == 409
        assert response.json()["detail"] == "live_operation_failed"
        assert "private-api-secret" not in response.text


def test_preflight_reviews_only_unfinished_startup_steps_after_pause(service):
    svc, broker, clock, _, _ = service
    svc.prepare(setup())
    runtime = svc.runtime
    broker.writes_enabled = True
    runtime.state["desired"] = "running"
    runtime.tick(clock[0])
    for _ in range(2):
        clock[0] += 15000
        runtime.market.now = clock[0]
        runtime.tick(clock[0])
    assert len(broker.state()["writes"]) == 1
    runtime.pause()
    for _ in range(4):
        clock[0] += 15000
        runtime.market.now = clock[0]
        runtime.tick(clock[0])
    assert runtime.state["error"] == "startup_incomplete_requires_review"
    result = svc.prepare(setup())
    assert len(result["runtime"]["startup_resume"]) == 3
    assert result["runtime"]["phase"] == "prepared"
    broker.writes_enabled = True
    runtime.state["desired"] = "running"
    for _ in range(40):
        clock[0] += 15000
        runtime.market.now = clock[0]
        runtime.tick(clock[0])
        if runtime.state["startup_complete"]:
            break
    assert runtime.state["startup_complete"]
    assert len(broker.state()["writes"]) == 4


def test_source_snapshot_is_not_mutable_through_status(service):
    svc = service[0]
    result = svc.status()
    result["candidate"]["snapshot"]["periods"].append(1)
    assert svc.status()["candidate"]["snapshot"]["periods"] == [60, 240, 1440]


def test_live_page_preserves_quote_age_cost_lots_and_total_receipt_count(service):
    svc, broker, clock, reads, _ = service
    result = svc.prepare(setup())
    runtime = svc.runtime
    with runtime.execution.exclusive():
        runtime.execution.reload()
        from freqtrade.strategy.trx_phase1.ledger import Ledger

        book = Ledger.restore(runtime.execution.state["book"])
        for index in range(21):
            book.buy(f"fill-{index}", f"order-{index}", clock[0], "1", ".3", ".001")
        runtime.execution.state["book"] = book.state()
        runtime.execution.save()
    result = svc.status()["runtime"]
    assert result["execution"]["order_count"] == 21
    assert len(result["execution"]["orders"]) == 20
    assert len(result["execution"]["lots"]) == 21
    assert result["summary"]["principal_trx"] == 21
    assert result["summary"]["remaining_cost"] == pytest.approx(6.321)
    assert result["valuation"]["stale"] is False
    clock[0] += 61000
    assert svc.status()["runtime"]["valuation"]["stale"] is True
    assert reads == [False] and broker.state()["writes"] == []


def account_configuration(service, cash="1000.00000000", restricted="0"):
    svc, _, clock, *_ = service
    rules = [dict(p) for p in RULES]
    for rule in rules:
        rule["redemption_decimals"] = 2 if rule["coin"] == "USDGO" else 8
    (svc.root / "live-product-rules.local.json").write_text(
        json.dumps({"schema": 1, "products": rules, "notes": {"USDGO": "operating precision only"}})
    )
    return {
        "status": "partial",
        "account_type": "classic",
        "stale": False,
        "error": None,
        "checks": {"balances": {"status": "success"}, "savings": {"status": "success"}},
        "snapshot": {
            "observed_at": clock[0],
            "balances": [
                {
                    "coin": "USDT",
                    "available": cash,
                    "restricted_available": restricted,
                    "frozen": "30",
                    "locked": "10",
                }
            ],
            "savings": {
                "rows": [
                    {"coin": "TRX", "product_id": "TRX-flex", "quantity": "4.46553"},
                    {"coin": "USDGO", "product_id": "USDGO-flex", "quantity": "1"},
                ]
            },
            "products": {
                "rows": [
                    {
                        "coin": coin,
                        "product_id": coin + "-flex",
                        "status": "in_progress",
                        "terms_status": "success",
                        "terms": {"single_minimum": "1" if coin == "TRX" else ".1"},
                    }
                    for coin in ("TRX", "USDGO")
                ]
            },
        },
    }


def test_automatic_configuration_preserves_exact_spendable_cash_and_manual_holdings(service):
    svc, broker, _, reads, _ = service
    account = account_configuration(service, "997.8987614200000000", "0.0000000100000000")
    result = svc.status(account)
    draft = result["configuration"]
    assert draft["capital"] == "997.8987614100000000"
    assert draft["ready"] and draft["capital_source"] == "available_usdt_less_restricted"
    products = {p["coin"]: p for p in draft["products"]}
    assert products["TRX"]["account_savings_quantity"] == "4.46553"
    assert products["USDGO"]["redemption_decimals"] == 2
    assert not products["USDGO"]["redemption_api_verified"]
    assert result["prepared"] is None and reads == [] and broker.state()["writes"] == []
    assert not (svc.root / "live_execution").exists()


@pytest.mark.parametrize(
    "change,expected",
    [
        ("stale", "account_refresh_required"),
        ("missing_usdt", "account_balance_unavailable"),
        ("failed_balance", "account_balance_unavailable"),
        ("no_cash", "insufficient_available_usdt"),
        ("missing_product", "selected_product_terms_unavailable"),
        ("uta", "classic_account_required"),
    ],
)
def test_automatic_configuration_blocks_incomplete_evidence_before_allocation(
    service, change, expected
):
    svc, broker, clock, reads, _ = service
    account = account_configuration(service)
    if change == "stale":
        clock[0] += 61000
    if change == "missing_usdt":
        account["snapshot"]["balances"] = []
    if change == "failed_balance":
        account["checks"]["balances"]["status"] = "error"
    if change == "no_cash":
        account["snapshot"]["balances"][0]["available"] = "0"
    if change == "missing_product":
        account["snapshot"]["products"]["rows"] = []
    if change == "uta":
        account["account_type"] = "uta"
    with pytest.raises(ExecutionError, match=expected):
        svc.prepare_account(account, broker.account_id())
    assert reads == [] and broker.state()["writes"] == []
    assert not (svc.root / "live_execution").exists()


def test_automatic_preflight_freezes_budget_and_never_adopts_manual_savings(service):
    svc, broker, _, reads, _ = service
    account = account_configuration(service, restricted="2.5")
    result = svc.prepare_account(account, broker.account_id())
    assert result["prepared"]["setup"]["capital"] == "997.50000000"
    assert result["runtime"]["summary"]["principal_trx"] == 0
    assert result["runtime"]["execution"]["holdings"] == {}
    account["snapshot"]["balances"][0]["available"] = "10"
    draft = svc.status(account)["configuration"]
    assert draft["capital"] == "997.50000000" and draft["capital_source"] == "frozen_strategy"
    assert reads == [False] and broker.state()["writes"] == []


def test_automatic_preflight_binds_the_account_snapshot_to_the_live_broker(service):
    svc, broker, *_ = service
    account = account_configuration(service)
    with pytest.raises(ExecutionError, match="account_binding_mismatch"):
        svc.prepare_account(account, "different-account")
    assert not (svc.root / "live_execution").exists() and broker.state()["writes"] == []


def test_missing_or_invalid_saved_rules_remain_unknown_not_subscription_defaults(service):
    svc, broker, *_ = service
    account = account_configuration(service)
    path = svc.root / "live-product-rules.local.json"
    path.unlink()
    assert svc.configuration(account)["status"] == "missing_product_rules"
    path.write_text('{"schema": 1, "products": [{"redemption_decimals": "private-raw-error"}]}')
    result = svc.configuration(account)
    assert result["status"] == "invalid_product_rules" and result["products"] == []
    assert "private-raw-error" not in json.dumps(result) and broker.state()["writes"] == []


def test_api_automatic_preflight_uses_server_snapshot_and_only_read_permissions(service):
    svc, broker, *_ = service
    account = account_configuration(service, restricted="2.5")
    reader = MagicMock()
    reader.status.return_value = account
    reader.allocation_context.return_value = (account, broker.account_id())
    app = FastAPI()
    app.include_router(api_trx.router)
    app.dependency_overrides[api_trx.get_live_service] = lambda: svc
    app.dependency_overrides[api_trx.get_account_service] = lambda: reader
    with TestClient(app) as client:
        draft = client.get("/trx/live").json()["configuration"]
        assert draft["capital"] == "997.50000000"
        result = client.post("/trx/live/prepare-account")
        assert result.status_code == 200
        assert result.json()["prepared"]["setup"]["capital"] == "997.50000000"
        assert not result.json()["real_trading_running"]
    reader.refresh.assert_not_called()  # The verified snapshot was less than five seconds old.
    assert broker.state()["writes"] == []


def test_cold_preflight_confirms_recent_bars_on_same_client_before_allocation(service, mocker):
    from freqtrade.strategy.trx_phase1.candles import append_closed_minutes
    from freqtrade.strategy.trx_phase1.shadow_market import PublicShadowMarket

    svc, broker, clock, reads, _ = service
    market = svc.market_factory()
    original = market.data.copy()
    market.data.iloc[:-5].to_feather(svc.seed)
    clock[0] += 30000
    market.now = clock[0]
    confirmer = PublicShadowMarket.__new__(PublicShadowMarket)
    confirmer.provisional = {}
    updates = []

    def update(existing, now):
        updates.append(id(confirmer))
        incoming = confirmer.confirm_recent(
            original, int(existing.date.iloc[-1].timestamp() * 1000), now
        )
        return append_closed_minutes(existing, incoming, pd.Timestamp(now, unit="ms", tz="UTC"))

    def wait(seconds):
        assert not (svc.root / "live_execution").exists()
        clock[0] += int(seconds * 1000)
        market.now = clock[0]

    mocker.patch.object(market, "update", side_effect=update)
    sleep = mocker.patch("freqtrade.strategy.trx_phase1.live_service.time.sleep", side_effect=wait)
    result = svc.prepare(setup())
    assert len(updates) == 2 and len(set(updates)) == 1
    sleep.assert_called_once_with(11)
    assert result["runtime"]["phase"] == "prepared"
    assert reads == [False] and broker.state()["writes"] == []


def test_preflight_can_retry_after_an_empty_journal_was_created(service):
    from freqtrade.strategy.trx_phase1.live_execution import LiveExecution

    svc, broker, _, _, _ = service
    empty = LiveExecution(svc.root, broker, broker.account_id())
    assert empty.state is None
    assert svc.prepare(setup())["runtime"]["phase"] == "prepared"
    assert broker.state()["writes"] == []


def test_minimum_redemption_display_is_bound_to_account_product_and_rules(service):
    from freqtrade.strategy.trx_phase1.live_execution import digest

    svc, broker, _, _, _ = service
    account = account_configuration(service)
    rule = json.loads((svc.root / "live-product-rules.local.json").read_text())["products"][0]
    path = svc.root / "execution_checks/minimum-redemption-2026-10-17" / (rule["coin"] + ".json")
    path.parent.mkdir(parents=True)
    evidence = {
        "account_id": broker.account_id(),
        "rules_sha256": digest(rule),
        "verification_version": 2,
        "status": "verified",
        "exchange_id": "confirmed-order",
        "request": {"productId": rule["product_id"], "amount": rule["redemption_minimum"]},
        "checked_at": NOW,
        "checks": {
            k: True
            for k in ("exchange_result", "arrival_bill", "holding_reduction", "spendable_arrival")
        },
    }
    path.write_text(json.dumps(evidence))
    products = svc.status(account, broker.account_id())["configuration"]["products"]
    assert next(p for p in products if p["coin"] == rule["coin"])["redemption_api_verified"]
    assert not next(p for p in products if p["coin"] != rule["coin"])["redemption_api_verified"]
    assert not any(
        p["redemption_api_verified"]
        for p in svc.status(account, "other-account")["configuration"]["products"]
    )
    evidence["checks"]["spendable_arrival"] = False
    path.write_text(json.dumps(evidence))
    assert not any(
        p["redemption_api_verified"]
        for p in svc.status(account, broker.account_id())["configuration"]["products"]
    )


def test_strategy_and_acceptance_cannot_allocate_the_same_account_concurrently(service):
    from freqtrade.strategy.trx_phase1.acceptance_check import MinimumChainCheck

    svc, broker, clock, _, _ = service
    account_configuration(service)
    broker.writes_enabled = False
    check = MinimumChainCheck(svc.root, broker, clock=lambda: clock[0])
    check.prepare()
    with pytest.raises(ExecutionError, match="finish_acceptance_before_strategy_allocation"):
        svc.prepare(setup())
    assert not (svc.root / "live_execution").exists()
    check.cancel()
    assert svc.prepare(setup())["runtime"]["phase"] == "prepared"
    with pytest.raises(ExecutionError, match="standalone_test_requires_unallocated_account"):
        check.prepare()
    assert broker.state()["writes"] == []
