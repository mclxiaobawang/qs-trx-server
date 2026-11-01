import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from freqtrade.rpc.api_server.app import create_app
from freqtrade.strategy.trx_phase1 import opening_adoption
from freqtrade.strategy.trx_phase1.account import ENV_KEYS, AccountReadError, ReadOnlyBitget
from freqtrade.strategy.trx_phase1.adoption_preview import adoption_preview
from freqtrade.strategy.trx_phase1.ledger import Ledger
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from freqtrade.strategy.trx_phase1.live_restore import ExistingLiveRestore
from freqtrade.strategy.trx_phase1.live_service import LiveService
from freqtrade.strategy.trx_phase1.opening_adoption_capture import capture
from tests.strategy.trx_phase1.test_adoption_preview import NOW, source
from tests.strategy.trx_phase1.test_live_strategy import candidate


ACCOUNT = hashlib.sha256(b"123456789").hexdigest()


@pytest.fixture(autouse=True)
def no_secrets_or_execution(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Opening adoption must not construct a broker, buy or start")

    monkeypatch.setattr(LiveExecution, "create", forbidden)
    monkeypatch.setattr(LiveService, "broker", forbidden)


def context(tmp_path):
    svc = LiveService(tmp_path, clock=lambda: NOW + 2000, existing_only=True)
    svc.root.mkdir()
    svc._source = candidate().model_dump()
    private = svc.root / "bitget-account.local.json"
    private.write_text(
        json.dumps({"api_key": "test-key", "secret_key": "test-secret", "passphrase": "test-pass"})
    )
    private.chmod(0o600)
    raw = source()
    preview = adoption_preview(raw, now_ms=NOW + 2000)
    preview["opening_quotes"] = deepcopy(raw["snapshot"]["prices"])
    current = MagicMock(return_value=(preview, ACCOUNT, svc.credential_identity(), NOW + 2000))
    return svc, raw, current


def test_adopts_native_journals_at_market_cost_with_no_cash_or_fabricated_fills(tmp_path):
    svc, account, captured = context(tmp_path)
    original = (svc.root / "bitget-account.local.json").read_bytes()
    receipt = opening_adoption.adopt(None, svc, capture=captured)
    state = LiveExecution(svc.root, None, ACCOUNT).status()
    book = Ledger.restore(state["book"])
    m = receipt["manifest"]
    assert receipt["committed"] and not receipt["preview_only"]
    assert not receipt["trading_authorized"] and receipt["startup_actions"] == []
    assert book.initial == Decimal(m["initial_equity_usdt"]) and book.cash == 0
    assert book.principal == Decimal("1002.12345679")
    assert book.usdgo == Decimal("10000.32345679")
    assert book.equity("0.33333") == book.initial == book.fund_units
    assert book.realized == book.stable_interest == book.interest_trx == book.fx_profit == 0
    assert book.fees == 0 and not book.orders
    assert len(book.lots) == 2 and sum(lot.locked for lot in book.lots) == Decimal("1000.00000001")
    assert book.locked["USDGO"] == Decimal("10000.12345678")
    assert state["unassigned_offsets"] == {"TRX": "0", "USDGO": "0", "USDT": "700"}
    assert set(state["holdings"]) == {"h-trx", "h-usdgo"}
    assert all(p["interest"] == p["unassigned_amount"] == "0" for p in state["holdings"].values())
    assert state["flow"] is state["operation"] is None
    assert state["flows"] == []
    assert (svc.root / "bitget-account.local.json").read_bytes() == original
    view = svc.status(account, ACCOUNT)
    assert view["runtime"]["summary"]["total_profit"] == 0
    assert view["runtime"]["summary"]["unrealized_profit"] == 0
    assert view["runtime"]["summary"]["usdgo_unrealized_profit"] == 0
    assert view["runtime"]["startup_complete"]
    assert view["runtime"]["phase"] == view["runtime"]["desired"] == "paused"
    assert not view["runtime"]["worker_running"] and not view["real_trading_running"]
    assert view["configuration"]["status"] == "adopted_trading_not_authorized"
    assert svc.runtime.state["activated_at"] is None and svc.runtime.engine.pending is None
    assert not svc.runtime.engine.accept_new_buys
    assert svc.runtime.state["markets"]["TRXUSDT"]["ask"] == "0.33334"


def test_repeated_call_never_reads_again_or_revalues_or_adds_second_lots(tmp_path):
    svc, _, captured = context(tmp_path)
    first = opening_adoption.adopt(None, svc, capture=captured)
    second = opening_adoption.adopt(None, svc, capture=captured)
    assert first == second
    captured.assert_called_once()
    assert len(LiveExecution(svc.root, None, ACCOUNT).status()["book"]["lots"]) == 2


def test_crash_after_journal_publish_recovers_same_cost_without_new_exchange_reads(
    tmp_path, monkeypatch
):
    svc, _, captured = context(tmp_path)
    real = opening_adoption.atomic_json

    def crash(path, value):
        if path == svc.control:
            raise OSError("simulated power loss before control marker")
        return real(path, value)

    monkeypatch.setattr(opening_adoption, "atomic_json", crash)
    with pytest.raises(OSError):
        opening_adoption.adopt(None, svc, capture=captured)
    assert not svc.control.exists()
    persisted = LiveExecution(svc.root, None, ACCOUNT).status()["opening_adoption"]
    monkeypatch.setattr(opening_adoption, "atomic_json", real)
    result = opening_adoption.adopt(None, svc, capture=captured)
    assert result == persisted and opening_adoption.status(svc.root)["committed"]
    captured.assert_called_once()


def test_concurrent_operators_publish_only_one_account_and_one_cost_basis(tmp_path):
    svc, _, captured = context(tmp_path)
    entered, proceed = threading.Event(), threading.Event()
    value = captured.return_value

    def waiting(*args):
        entered.set()
        assert proceed.wait(5)
        return value

    captured.side_effect = waiting
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(opening_adoption.adopt, None, svc, capture=captured)
        assert entered.wait(5)
        second = pool.submit(opening_adoption.adopt, None, svc, capture=captured)
        proceed.set()
        assert first.result(5) == second.result(5)
    captured.assert_called_once()


def test_unreviewable_or_failed_capture_leaves_no_account_allocation(tmp_path):
    svc, _, captured = context(tmp_path)
    captured.return_value[0]["reviewable"] = False
    with pytest.raises(ExecutionError, match="fresh_complete"):
        opening_adoption.adopt(None, svc, capture=captured)
    assert not svc.control.exists() and not (svc.root / "live_execution").exists()


def test_another_existing_allocation_is_never_replaced(tmp_path):
    svc, _, captured = context(tmp_path)
    svc.save({"account_id": "some-existing-account"})
    original = svc.control.read_bytes()
    with pytest.raises(ExecutionError, match="already_allocated"):
        opening_adoption.adopt(None, svc, capture=captured)
    assert svc.control.read_bytes() == original
    captured.assert_not_called()


def test_no_worker_prepare_queue_tick_or_income_import_can_bypass_paused_adoption(tmp_path):
    svc, _, captured = context(tmp_path)
    opening_adoption.adopt(None, svc, capture=captured)
    runtime = svc.get_runtime(svc.load())
    execution = runtime.execution
    for fn in (
        lambda: svc.start("x" * 64),
        lambda: svc.prepare({}),
        runtime.start,
        lambda: execution.queue([], now=NOW, expires_at=NOW + 1000, label="forbidden"),
        lambda: execution.tick(NOW),
    ):
        with pytest.raises(ExecutionError, match="adopted_trading_not_authorized"):
            fn()
    with pytest.raises(ExecutionError, match="adoption_income_attribution_pending"):
        execution.import_interest(NOW)
    with pytest.raises(ExecutionError, match="completed_live_startup_required"):
        ExistingLiveRestore(svc).once()
    assert runtime.thread is None and execution.status()["book"]["orders"] == []


def capture_context(tmp_path):
    svc, account, _ = context(tmp_path)
    client = MagicMock(spec=ReadOnlyBitget)

    def response(data):
        return {"code": "00000", "data": data}

    client.privateSpotGetV2SpotAccountInfo.return_value = response({"userId": "123456789"})
    client.privateSpotGetV2SpotAccountAssets.return_value = response(
        [
            {
                "coin": r["coin"],
                "available": r["available"],
                "frozen": r["frozen"],
                "locked": r["locked"],
                "limitAvailable": r["restricted_available"],
            }
            for r in account["snapshot"]["balances"]
        ]
    )
    client.privateSpotGetV2SpotTradeUnfilledOrders.return_value = response([])
    client.privateEarnGetV2EarnAccountAssets.return_value = response(
        [
            {"coin": "TRX", "amount": "1000.00000001"},
            {"coin": "USDGO", "amount": "10000.12345678"},
            {"coin": "USDT", "amount": "300"},
        ]
    )
    client.publicSpotGetV2SpotMarketTickers.side_effect = lambda p: response(
        [
            {
                "symbol": p["symbol"],
                "bidPr": "0.4" if p["symbol"] == "TRXUSDT" else "1.001",
                "askPr": "0.4001" if p["symbol"] == "TRXUSDT" else "1.0011",
                "ts": NOW + 1999,
            }
        ]
    )
    service = SimpleNamespace(
        refresh=MagicMock(return_value=deepcopy(account)),
        allocation_context=lambda: (account, ACCOUNT),
        config_path=svc.root / "bitget-account.local.json",
        client_factory=MagicMock(return_value=client),
    )
    return svc, service, client


def test_capture_uses_latest_quotes_and_checks_binding_balances_and_pending_orders(tmp_path):
    svc, account, client = capture_context(tmp_path)
    result, account_id, _, now = capture(account, svc)
    assert account_id == ACCOUNT and now == NOW + 2000
    trx = next(r for r in result["manifest"]["assets"] if r["coin"] == "TRX")
    assert trx["adoption_unit_cost_usdt"] == "0.4"
    assert Decimal(trx["strategy_cost_usdt"]) == Decimal("1002.12345679") * Decimal("0.4")
    calls = [c[0] for c in client.mock_calls]
    assert calls.index("privateSpotGetV2SpotAccountAssets") < calls.index(
        "publicSpotGetV2SpotMarketTickers"
    )
    assert calls[-1] == "close"


@pytest.mark.parametrize(
    "change", ["wrong_account", "changed_spot", "changed_earn", "pending_order", "stale_quote"]
)
def test_capture_refuses_changed_or_unverified_assets_without_creating_a_ledger(tmp_path, change):
    svc, account, client = capture_context(tmp_path)
    if change == "wrong_account":
        client.privateSpotGetV2SpotAccountInfo.return_value["data"]["userId"] = "98765"
    elif change == "changed_spot":
        client.privateSpotGetV2SpotAccountAssets.return_value["data"][1]["available"] = "99"
    elif change == "changed_earn":
        client.privateEarnGetV2EarnAccountAssets.return_value["data"][0]["amount"] = "9999"
    elif change == "pending_order":
        client.privateSpotGetV2SpotTradeUnfilledOrders.return_value["data"] = [
            {
                "symbol": "TRXUSDT",
                "orderId": "open-order",
                "side": "buy",
                "orderType": "limit",
                "status": "live",
                "size": "1",
                "baseVolume": "0",
                "priceAvg": "0.4",
                "cTime": str(NOW),
            }
        ]
    else:
        client.publicSpotGetV2SpotMarketTickers.side_effect = lambda p: {
            "code": "00000",
            "data": [{"symbol": p["symbol"], "bidPr": "0.4", "askPr": "0.41", "ts": NOW - 600000}],
        }
    with pytest.raises((ExecutionError, AccountReadError)):
        capture(account, svc)
    assert not svc.control.exists() and not (svc.root / "live_execution").exists()


def test_receipt_get_is_authenticated_and_has_no_commit_post_route(tmp_path):
    svc, _, captured = context(tmp_path)
    config = {
        "user_data_dir": str(tmp_path),
        "api_server": {"username": "reader", "password": "password", "jwt_secret_key": "x" * 40},
    }
    app = create_app(config)
    client = TestClient(app)
    url = "/api/v1/trx/account/adoption"
    assert client.get(url).status_code == 401
    assert client.get(url, auth=("reader", "password")).json()["committed"] is False
    opening_adoption.adopt(None, svc, capture=captured)
    result = client.get(url, auth=("reader", "password"))
    assert result.status_code == 200 and result.json()["committed"]
    assert "test-secret" not in result.text and "credentials_identity" not in result.text
    assert client.post(url, auth=("reader", "password")).status_code == 405
