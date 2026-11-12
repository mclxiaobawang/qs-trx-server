# ruff: noqa: RUF001 - notification assertions use user-facing Chinese punctuation

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient

from freqtrade.strategy.trx_phase1 import live_backup, live_ops
from freqtrade.strategy.trx_phase1.ledger import Ledger
from freqtrade.strategy.trx_phase1.live_ops import LiveView, Operations, ViewUnavailable
from freqtrade.strategy.trx_phase1.live_ops_app import create_app
from tests.strategy.trx_phase1.test_live_strategy import live, startup  # noqa: F401


@pytest.fixture
def observed(live, tmp_path):  # noqa: F811
    runtime, _, broker, _ = live
    startup(live)
    runtime.capture_observation(broker.now)
    runtime.persist()
    return (
        runtime,
        broker,
        LiveView(runtime.execution.root, runtime.execution.account_id, clock=lambda: broker.now),
    )


def test_view_is_read_only_and_has_no_worker_or_curve_and_preserves_accounting(observed):
    runtime, _, view = observed
    original = {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")}
    first = view.view()
    assert "worker_running" not in first["runtime"]
    assert "equity_curve" not in first["runtime"]
    assert first["account_id"] == runtime.execution.account_id
    assert first["runtime"]["summary"]["principal_trx"] == float(runtime.engine.book.principal)
    assert first["runtime"]["dashboard"]["attribution"]["reconciled"]
    assert first["runtime"]["order_preview"]["fresh"]
    assert {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")} == original
    assert len(json.dumps(first)) < 50000
    assert view.history(curve=True)["equity_curve"]
    with pytest.raises(ViewUnavailable, match="account"):
        LiveView(view.source, "wrong-account").view()
    missing = view.source / "absent.sqlite"
    with pytest.raises(ViewUnavailable):
        live_ops.read_state(missing, "execution")
    assert not missing.exists()


def test_view_includes_only_strategy_usdc_and_reconciles_its_mark_to_market(observed, monkeypatch):
    _, broker, view = observed
    ev, sv, execution, state = view.snapshot()
    book = Ledger.restore(execution["book"])
    book.credit_interest("income-allocation:usdc", "USDC", "25", "1")
    execution["book"] = book.state()
    execution.setdefault("unassigned_offsets", {})["USDC"] = "500"
    state["markets"]["USDCUSDT"] = {"bid": "1.01", "timestamp": broker.now}
    monkeypatch.setattr(view, "snapshot", lambda: (ev, sv, execution, state))

    result = view.view()["runtime"]
    assert result["summary"]["usdc_market_value"] == 25.25
    dashboard = result["dashboard"]
    location = next(row for row in dashboard["locations"] if row["coin"] == "USDC")
    assert location["total"] == "25"
    assert location["outside_strategy"] == "500"
    assert dashboard["attribution"]["usdc_unrealized"] == "0.25"
    assert dashboard["attribution"]["reconciled"]


def test_activity_reports_booked_usdc_and_batched_usdt_interest():
    now = int(datetime(2026, 11, 4, 10, 16, tzinfo=UTC).timestamp() * 1000)
    book = Ledger(1000)
    book.credit_interest("income-allocation:usdc", "USDC", "0.33763838", "1")
    book.credit_interest("income-product-batch:usdt", "USDT", "0.01369862")
    execution = {
        "book": book.state(),
        "created_at": now - 3600000,
        "income_accounting": {
            "allocations": {
                "income-bill:usdc": {
                    "bill_id": "usdc",
                    "coin": "USDC",
                    "amount": "0.33763838",
                    "timestamp": now,
                },
                "income-product-batch:usdt": {
                    "bill_id": "usdt",
                    "coin": "USDT",
                    "amount": "0.01369862",
                    "timestamp": now,
                },
                "income-product-batch:outside": {
                    "bill_id": "outside",
                    "coin": "USDT",
                    "amount": "0",
                    "external_amount": "0.00031476",
                    "timestamp": now,
                },
            },
        },
    }
    activity = live_ops.activity_summary(execution, now)
    assert activity["complete"]["interest"]
    assert activity["periods"]["today"]["interest"] == {
        "TRX": "0",
        "USDGO": "0",
        "USDT": "0.01369862",
        "USDC": "0.33763838",
    }


def test_unused_adopted_account_can_poll_without_market_fees_or_starting_trading(tmp_path):
    from tests.strategy.trx_phase1.test_startup_basis import adopted

    service, _, _ = adopted(tmp_path)
    service.get_runtime(service.load())
    source = service.root / "live_execution" / service.load()["account_id"]
    original = {p.name: p.read_bytes() for p in source.glob("*.sqlite")}
    view = LiveView(source, service.load()["account_id"], clock=service.clock)
    first = view.view()
    second = view.view()
    assert (
        first["runtime"]["performance"]
        == second["runtime"]["performance"]
        == {
            "status": "not_started",
            "started_at": None,
            "basis": "actual_strategy_start_market_value",
            "previous_adoption_is_performance_basis": False,
        }
    )
    assert first["runtime"]["order_preview"]["buy"] is None
    assert first["runtime"]["order_preview"]["sell"] is None
    assert first["runtime"]["desired"] == "paused"
    assert first["runtime"]["execution"]["orders"] == []
    assert {p.name: p.read_bytes() for p in source.glob("*.sqlite")} == original


def test_standalone_observer_has_no_trading_imports_and_matches_saved_account(observed):
    _, broker, view = observed
    before = {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")}
    code = """
import importlib.abc, json, sys
class ReadOnlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ('freqtrade', 'pandas', 'numpy', 'ccxt', 'sqlalchemy',
                     'trx_phase1.live_execution', 'trx_phase1.live_strategy')
        if any(fullname == p or fullname.startswith(p + '.') for p in forbidden):
            raise ImportError('Observer imported trading dependency: ' + fullname)
sys.meta_path.insert(0, ReadOnlyImports())
from trx_phase1.live_ops_app import create_app
from trx_phase1.live_ops import LiveView
view = LiveView(sys.argv[1], sys.argv[2], clock=lambda: int(sys.argv[3]))
print(json.dumps(view.view()))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(view.source), view.account_id, str(broker.now)],
        env={**os.environ, "PYTHONPATH": str(Path(live_ops.__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert json.loads(result.stdout) == view.view()
    assert {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")} == before


def test_cgroup_memory_budget_is_finite_and_missing_metrics_are_safe(tmp_path):
    assert live_ops.memory_usage(tmp_path) is None
    (tmp_path / "memory.current").write_text("850")
    (tmp_path / "memory.max").write_text("1000")
    assert live_ops.memory_usage(tmp_path) == {"current": 850, "limit": 1000, "percent": 85}
    (tmp_path / "memory.max").write_text("max")
    assert live_ops.memory_usage(tmp_path) is None


def test_host_metrics_bind_to_configured_container_and_still_reject_stale_or_wrong_data(tmp_path):
    path = tmp_path / "trading-memory.json"
    value = {
        "container_name": "trx-new-api",
        "checked_at": 100000,
        "running": True,
        "current": 100,
        "limit": 1000,
        "error": None,
    }
    path.write_text(json.dumps(value))
    assert live_ops.trading_memory(path, 100000, container_name="trx-new-api")["percent"] == 10
    assert live_ops.trading_memory(path, 100000) == {"available": False}
    assert live_ops.trading_memory(path, 200001, container_name="trx-new-api") == {
        "available": False
    }


def test_preview_accounts_for_fee_reserve_and_never_sells_retained_interest(observed):
    runtime, broker, view = observed
    data = view.view()["runtime"]
    preview = data["order_preview"]
    from decimal import Decimal as D

    buy = preview["buy"]
    assert D(buy["order_quote"]) * (1 + D(buy["fee_rate"])) <= D(buy["budget"])
    assert D(buy["gross_trx"]) > D(buy["net_trx"]) > 0
    # Raise only the view's target price to make every principal lot eligible.
    _, _, execution, state = view.snapshot()
    engine, book = state["engine"], runtime.engine.book
    panel = deepcopy(data["dashboard"])
    panel["signal"]["sides"]["sell"].update(
        target="10", target_fraction=".21", target_eligible_trx=str(book.principal)
    )
    first = live_ops.preview(state, engine, execution, book, panel, broker.now)["sell"]
    book.credit_interest("extra-interest", "TRX", 100)
    second = live_ops.preview(state, engine, execution, book, panel, broker.now)["sell"]
    assert first == second
    assert D(first["order_trx"]) * (1 + D(first["fee_rate"])) <= D(first["target_trx"])
    panel["signal"]["principal_location_verified"] = False
    assert live_ops.preview(state, engine, execution, book, panel, broker.now)["sell"] is None
    assert not live_ops.preview(
        state, engine, execution, book, data["dashboard"], broker.now + 61000
    )["fresh"]


def monitor(observed, tmp_path, monkeypatch):
    runtime, broker, view = observed
    monkeypatch.setattr(
        live_ops.shutil, "disk_usage", lambda _: SimpleNamespace(total=100, used=80, free=20)
    )
    backup = tmp_path / "backup.json"
    backup.write_text(json.dumps({"last_success_at": broker.now, "last_error": None}))
    ops = Operations(view, tmp_path / "ops.sqlite", backup, tmp_path, clock=lambda: broker.now)
    return runtime, broker, view, ops


def test_independent_alarms_debounce_deduplicate_survive_restart_and_recover(
    observed, tmp_path, monkeypatch
):
    runtime, broker, view, ops = monitor(observed, tmp_path, monkeypatch)
    broker.now += 240000
    assert not any(a["active"] for a in ops.check()["alerts"])
    assert any(a["code"] == "worker_stale" and a["active"] for a in ops.check()["alerts"])
    pending = ops.status()["notification"]["pending"]
    restarted = Operations(
        view, ops.database, ops.backup_status, tmp_path, clock=lambda: broker.now
    )
    restarted.check()
    assert restarted.status()["notification"]["pending"] == pending
    runtime.state.update(desired="paused", phase="paused", error=None)
    runtime.persist()
    assert not restarted.check()["alerts"]
    assert restarted.status()["notification"]["pending"] == 1
    with closing(sqlite3.connect(ops.database)) as db:
        assert db.execute(
            "SELECT transition FROM notifications WHERE delivery_state='pending'"
        ).fetchall() == [("paused",)]
        assert (
            db.execute(
                "SELECT count(*) FROM notifications WHERE transition='recovered'"
            ).fetchone()[0]
            > 0
        )
    assert runtime.state["desired"] == "paused"  # Monitor never starts a strategy.


def test_backup_failure_and_disk_alerts_do_not_mutate_source(observed, tmp_path, monkeypatch):
    _, _, view, ops = monitor(observed, tmp_path, monkeypatch)
    ops.backup_status.write_text(json.dumps({"last_error": "TimeoutError"}))
    monkeypatch.setattr(
        live_ops.shutil, "disk_usage", lambda _: SimpleNamespace(total=100, used=95, free=5)
    )
    before = {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")}
    ops.check()
    alerts = ops.check()["alerts"]
    assert {a["code"] for a in alerts} >= {"backup_failed", "backup_stale", "disk_usage"}
    assert next(a for a in alerts if a["code"] == "disk_usage")["severity"] == "critical"
    assert {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")} == before


@pytest.mark.parametrize("code", ["disk_usage", "strategy_error", "minute_stale"])
def test_recovery_delivers_verified_status_and_event_time_after_restart(
    code, observed, tmp_path, monkeypatch
):
    _, broker, view, ops = monitor(observed, tmp_path, monkeypatch)
    snapshot = list(view.snapshot())
    execution, state = snapshot[2:]
    before = {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")}
    monkeypatch.setattr(view, "snapshot", lambda: tuple(snapshot))
    original = {
        "disk_usage": "宿主机磁盘使用率达到85%",
        "strategy_error": "策略报告运行异常",
        "minute_stale": "确认分钟行情超过3分钟未更新",
    }[code]
    if code == "disk_usage":
        monkeypatch.setattr(
            live_ops.shutil, "disk_usage", lambda _: SimpleNamespace(total=100, used=90, free=10)
        )
    elif code == "strategy_error":
        state["error"] = "market_network_unavailable"
    else:
        state["engine"]["last_close"] = broker.now - 240000
    config = tmp_path / "telegram.json"
    config.write_text(json.dumps({"enabled": True, "bot_token": "123:fake", "chat_id": "test"}))
    ops.notification_file = config
    sent = []
    monkeypatch.setattr(
        live_ops, "send_notification", lambda cfg, title, body: sent.append((title, body))
    )
    ops.check()
    ops.check()
    ops.deliver()
    assert len(sent) == 1 and original in sent[0][1]
    broker.now += 60000
    recovered_at = broker.now
    state.update(error=None, phase="observing", last_poll_at=broker.now)
    state["engine"]["last_close"] = broker.now
    execution["interest_checked_at"] = broker.now
    monkeypatch.setattr(
        live_ops.shutil, "disk_usage", lambda _: SimpleNamespace(total=100, used=20, free=80)
    )
    assert not ops.check()["alerts"]
    with sqlite3.connect(ops.database) as db:
        payload = json.loads(
            db.execute(
                "SELECT payload FROM notifications WHERE code=? AND transition='recovered'",
                (code,),
            ).fetchone()[0]
        )
    assert payload["active"] is False and payload["recovered_at"] == recovered_at
    assert original not in payload["message"]
    broker.now += 3600000  # Sending later must not change the recorded recovery time.
    restarted = Operations(
        view,
        ops.database,
        ops.backup_status,
        tmp_path,
        notification_file=config,
        clock=lambda: broker.now,
    )
    restarted.deliver()
    restarted.deliver()
    assert len(sent) == 2
    title, body = sent[1]
    expected_title = {
        "disk_usage": "磁盘空间告警已解除",
        "strategy_error": "策略错误已解除",
        "minute_stale": "行情更新告警已解除",
    }[code]
    assert title == "[TRX策略恢复] " + expected_title
    assert code not in title
    assert "恢复时间：" + live_ops.beijing_time(recovered_at) in body
    assert live_ops.beijing_time(broker.now) not in body
    assert "当前状态：" in body and original not in body
    assert {"disk_usage": "20.00%", "strategy_error": "等待新信号", "minute_stale": "已恢复更新"}[
        code
    ] in body
    assert {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")} == before


@pytest.mark.parametrize("code", ["worker_stale", "minute_stale", "reconciliation_stale"])
def test_recovery_does_not_claim_paused_strategy_resumed(code):
    message = live_ops.recovery_status(code, {"desired": "paused"}, {}, {})
    assert "已暂停" in message and "时效监测不适用" in message
    assert "已恢复更新" not in message


def test_legacy_queued_recovery_hides_old_alarm_and_uses_original_event_time(
    observed, tmp_path, monkeypatch
):
    _, _, _, ops = monitor(observed, tmp_path, monkeypatch)
    ops.check()
    config = tmp_path / "telegram.json"
    config.write_text(json.dumps({"enabled": True, "bot_token": "123:fake", "chat_id": "test"}))
    ops.notification_file = config
    recovered_at = int(datetime(2026, 10, 18, 16, 1, 2, tzinfo=UTC).timestamp() * 1000)
    with sqlite3.connect(ops.database) as db:
        db.execute(
            "INSERT INTO notifications(code,transition,created_at,payload) VALUES(?,?,?,?)",
            (
                "strategy_error",
                "recovered",
                recovered_at,
                json.dumps({"message": "策略报告运行异常，请查看真实账户页面。", "active": True}),
            ),
        )
    sent = []
    monkeypatch.setattr(live_ops, "send_notification", lambda cfg, title, body: sent.append(body))
    ops.deliver()
    assert len(sent) == 1
    assert "2026-10-19 00:01:02 北京时间（UTC+8）" in sent[0]
    assert "当前状态：该项告警已解除。" in sent[0]
    assert "策略报告运行异常" not in sent[0]


@pytest.mark.parametrize(
    "code,transition,test,expected",
    [
        ("strategy_error", "opened", False, "[TRX策略告警] 策略运行异常"),
        ("strategy_error", "recovered", False, "[TRX策略恢复] 策略错误已解除"),
        ("minute_stale", "opened", False, "[TRX策略告警] 分钟行情更新超时"),
        ("minute_stale", "recovered", False, "[TRX策略恢复] 行情更新告警已解除"),
        ("signal_overdue_buy", "opened", False, "[TRX策略告警] 买入信号判断延迟"),
        ("signal_overdue_sell", "recovered", False, "[TRX策略恢复] 卖出信号延迟告警已解除"),
        ("new-unmapped-type", "opened", False, "[TRX策略告警] 其他运行异常"),
        ("new-unmapped-type", "recovered", False, "[TRX策略恢复] 其他运行告警已解除"),
        ("minute_stale", "recovered", True, "【测试】[TRX策略恢复] 行情更新告警已解除"),
    ],
)
def test_chinese_titles_preserve_internal_codes_delivery_and_test_marker(
    code, transition, test, expected, observed, tmp_path, monkeypatch
):
    _, broker, view, ops = monitor(observed, tmp_path, monkeypatch)
    ops.check()
    before = {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")}
    config = tmp_path / "telegram.json"
    config.write_text(json.dumps({"enabled": True, "bot_token": "123:fake", "chat_id": "test"}))
    ops.notification_file = config
    with sqlite3.connect(ops.database) as db:
        db.execute(
            "INSERT INTO notifications(code,transition,created_at,payload) VALUES(?,?,?,?)",
            (
                code,
                transition,
                broker.now,
                json.dumps(
                    {"message": "已核验的通知正文", "test": test, "recovered_at": broker.now}
                ),
            ),
        )
    sent = []
    monkeypatch.setattr(
        live_ops, "send_notification", lambda cfg, title, body: sent.append((title, body))
    )
    ops.deliver()
    ops.deliver()
    assert len(sent) == 1 and sent[0][0] == expected
    assert code not in sent[0][0]
    assert "已核验的通知正文" in sent[0][1]
    with sqlite3.connect(ops.database) as db:
        assert db.execute(
            "SELECT code,transition,delivery_state,attempts FROM notifications"
        ).fetchone() == (code, transition, "sent", 1)
    assert {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")} == before


def test_memory_pressure_is_debounced_recovers_and_never_changes_trading_state(
    observed, tmp_path, monkeypatch
):
    _, _, view, ops = monitor(observed, tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")}
    monkeypatch.setattr(
        live_ops, "memory_usage", lambda: {"current": 850, "limit": 1000, "percent": 85}
    )
    assert not any(a["active"] for a in ops.check()["alerts"])
    alert = next(a for a in ops.check()["alerts"] if a["code"] == "observer_memory")
    assert alert["active"] and alert["severity"] == "warning"
    monkeypatch.setattr(
        live_ops, "memory_usage", lambda: {"current": 950, "limit": 1000, "percent": 95}
    )
    alert = next(a for a in ops.check()["alerts"] if a["code"] == "observer_memory")
    assert alert["severity"] == "critical"
    monkeypatch.setattr(
        live_ops, "memory_usage", lambda: {"current": 200, "limit": 1000, "percent": 20}
    )
    assert not ops.check()["alerts"]
    assert ops.status()["memory"]["percent"] == 20
    assert {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")} == before


def test_trading_memory_warning_critical_recovery_and_stale_sampler(
    observed, tmp_path, monkeypatch
):
    _, broker, view, ops = monitor(observed, tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")}
    ops.resource_file = tmp_path / "resources.json"

    def sample(current, stamp=None):
        ops.resource_file.write_text(
            json.dumps(
                {
                    "container_name": "trx-phase1-live",
                    "checked_at": broker.now if stamp is None else stamp,
                    "running": True,
                    "current": current,
                    "limit": 1000,
                    "error": None,
                }
            )
        )

    sample(850)
    assert not any(a["active"] for a in ops.check()["alerts"])
    alarm = next(a for a in ops.check()["alerts"] if a["code"] == "trading_memory")
    assert alarm["active"] and alarm["severity"] == "warning"
    sample(950)
    alarm = next(a for a in ops.check()["alerts"] if a["code"] == "trading_memory")
    assert alarm["severity"] == "critical"
    sample(200)
    assert not ops.check()["alerts"]
    sample(200, broker.now - 90001)
    ops.check()
    assert any(
        a["active"] and a["code"] == "trading_memory_unavailable" for a in ops.check()["alerts"]
    )
    assert {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")} == before


def test_trade_notifications_baseline_partial_fills_restart_and_delivery(
    observed, tmp_path, monkeypatch
):
    _, broker, view, ops = monitor(observed, tmp_path, monkeypatch)
    ops.check()
    assert ops.status()["trade_notifications"]["baseline_at"] == broker.now
    assert ops.status()["trade_notifications"]["queued_fills"] == 0
    before = {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")}
    _, _, execution, _ = view.snapshot()
    for index, side in enumerate(("buy", "buy", "sell")):
        execution["book"]["orders"].append(
            {
                "id": f"spot:new-order:{index}",
                "order_id": "new-order",
                "timestamp": broker.now,
                "side": side,
                "quantity": 10,
                "net_quantity": 9.99 if side == "buy" else None,
                "price": 0.3,
                "fee": 0.003,
                "profit": 0.5 if side == "sell" else None,
            }
        )
    ops.observe_trades(execution, broker.now)
    restarted = Operations(
        view, ops.database, ops.backup_status, tmp_path, clock=lambda: broker.now
    )
    restarted.observe_trades(execution, broker.now)
    assert restarted.status()["trade_notifications"]["queued_fills"] == 3
    config = tmp_path / "telegram.json"
    config.write_text(
        json.dumps(
            {
                "enabled": True,
                "channel": "telegram",
                "bot_token": "123:fake-token",
                "chat_id": "test-chat",
            }
        )
    )
    restarted.notification_file = config
    sent = []
    monkeypatch.setattr(
        live_ops, "send_notification", lambda cfg, title, body: sent.append((title, body))
    )
    for _ in range(4):
        restarted.deliver()
    assert len(sent) == 3
    assert all("TRX策略成交" in title for title, _ in sent)
    assert "净到账数量" in sent[0][1] and "9.99000000 TRX" in sent[0][1]
    assert "该笔本金已实现利润" in sent[-1][1] and "0.50000000 USDT" in sent[-1][1]
    assert {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")} == before


def test_trade_notification_lost_ack_is_not_replayed(observed, tmp_path, monkeypatch):
    _, broker, view, ops = monitor(observed, tmp_path, monkeypatch)
    ops.check()
    _, _, execution, _ = view.snapshot()
    execution["book"]["orders"].append(
        {
            "id": "spot:later:fill",
            "order_id": "later",
            "timestamp": broker.now,
            "side": "buy",
            "quantity": 1,
            "net_quantity": 0.999,
            "price": 0.3,
            "fee": 0.0003,
            "profit": None,
        }
    )
    ops.observe_trades(execution, broker.now)
    config = tmp_path / "telegram.json"
    config.write_text(
        json.dumps(
            {
                "enabled": True,
                "channel": "telegram",
                "bot_token": "123:fake-token",
                "chat_id": "test-chat",
            }
        )
    )
    ops.notification_file = config
    attempted = []

    def lose_ack(*args):
        attempted.append(True)
        raise TimeoutError()

    monkeypatch.setattr(live_ops, "send_notification", lose_ack)
    ops.deliver()
    restarted = Operations(
        view,
        ops.database,
        ops.backup_status,
        tmp_path,
        notification_file=config,
        clock=lambda: broker.now,
    )
    restarted.observe_trades(execution, broker.now)
    restarted.deliver()
    assert len(attempted) == 1
    assert restarted.status()["notification"]["last_error"] == "telegram_delivery_unconfirmed"


def test_operational_alerts_are_delivered_before_trade_notices(observed, tmp_path, monkeypatch):
    _, _, _, ops = monitor(observed, tmp_path, monkeypatch)
    ops.check()
    config = tmp_path / "telegram.json"
    config.write_text(
        json.dumps(
            {"enabled": True, "channel": "telegram", "bot_token": "123:fake", "chat_id": "test"}
        )
    )
    ops.notification_file = config
    with sqlite3.connect(ops.database) as db:
        db.execute(
            "INSERT INTO notifications(code,transition,payload,delivery_state) VALUES(?,?,?,?)",
            ("trade-first", "trade", json.dumps({"side": "buy", "message": "fill"}), "pending"),
        )
        db.execute(
            "INSERT INTO notifications(code,transition,payload,delivery_state) VALUES(?,?,?,?)",
            ("memory", "opened", json.dumps({"message": "pressure"}), "pending"),
        )
    sent = []
    monkeypatch.setattr(live_ops, "send_notification", lambda cfg, title, body: sent.append(title))
    ops.deliver()
    ops.deliver()
    assert "告警" in sent[0] and "成交" in sent[1]


def test_telegram_delivery_uses_configured_recipient_and_ambiguous_send_is_not_retried(
    observed, tmp_path, monkeypatch
):
    _, broker, _, ops = monitor(observed, tmp_path, monkeypatch)
    cfg = tmp_path / "telegram.json"
    cfg.write_text(
        json.dumps(
            {
                "enabled": True,
                "channel": "telegram",
                "bot_token": "123:fake-token",
                "chat_id": "test-chat",
            }
        )
    )
    ops.notification_file = cfg
    broker.now += 300000
    ops.check()
    ops.check()
    calls = []

    def lost_result(config, subject, body):
        calls.append((subject, body))
        raise TimeoutError()

    monkeypatch.setattr(live_ops, "send_notification", lost_result)
    ops.deliver()
    assert ops.status()["notification"]["last_error"] == "telegram_delivery_unconfirmed"
    with closing(sqlite3.connect(ops.database)) as db, db:
        # Isolate this one message: recovery/other alarms have their own identities.
        db.execute("DELETE FROM notifications WHERE attempted_at IS NULL")
    ops.deliver()
    assert len(calls) == 1
    assert "fake-token" not in json.dumps(ops.status())
    assert "test-chat" not in json.dumps(ops.status())


def test_read_api_requires_valid_access_token_and_rejects_all_mutations(
    observed, tmp_path, monkeypatch
):
    _, _, view, ops = monitor(observed, tmp_path, monkeypatch)
    config = {
        "source": str(view.source),
        "account_id": view.account_id,
        "username": "operator",
        "jwt_secret_key": "s" * 32,
    }
    token = jwt.encode(
        {
            "identity": {"u": "operator"},
            "type": "access",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        config["jwt_secret_key"],
    )
    app = create_app(config, view, ops)
    with TestClient(app) as client:
        assert client.get("/api/v1/trx/live/view").status_code == 401
        auth = {"Authorization": "Bearer " + token}
        assert client.get("/api/v1/trx/live/view", headers=auth).status_code == 200
        assert client.post("/api/v1/trx/live/view", headers=auth).status_code == 405
        assert client.get("/api/v1/trx/live/ops", headers=auth).status_code == 200
        assert client.get("/api/v1/trx/live/start", headers=auth).status_code == 404


def backup_config(observed, tmp_path):
    runtime, _, view = observed
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({"account_id": view.account_id, "source_sha256": runtime.state["source_sha256"]})
    )
    return {
        "source": str(view.source),
        "control": str(control),
        "account_id": view.account_id,
        "destination": str(tmp_path / "backups"),
        "status": str(tmp_path / "backup-status.json"),
    }


def test_online_backup_is_verified_in_isolation_and_preserves_source(observed, tmp_path):
    config = backup_config(observed, tmp_path)
    source = Path(config["source"])
    before = {p.name: p.read_bytes() for p in source.glob("*.sqlite")}
    result = live_backup.run_backup(config)
    archive = Path(config["destination"]) / result["last_archive"]
    restore = tmp_path / "isolated-copy"
    restore.mkdir()
    live_backup.unpack_archive(archive, restore)
    restored = live_backup.verify(restore, config["account_id"])
    assert restored["startup_complete"]
    assert restored["order_count"] == result["verification"]["order_count"]
    assert {p.name: p.read_bytes() for p in source.glob("*.sqlite")} == before
    assert archive.stat().st_mode & 0o077 == 0
    assert not result["credential_files_included"]
    with pytest.raises(ValueError, match="account"):
        live_backup.verify(restore, "wrong-account")


def test_backup_timeout_retains_last_success_without_touching_source(
    observed, tmp_path, monkeypatch
):
    config = backup_config(observed, tmp_path)
    Path(config["status"]).write_text(json.dumps({"last_success_at": 1234}))

    def timeout(*args):
        raise TimeoutError()

    monkeypatch.setattr(live_backup, "copy_database", timeout)
    with pytest.raises(TimeoutError):
        live_backup.run_backup(config)
    status = json.loads(Path(config["status"]).read_text())
    assert status["last_success_at"] == 1234
    assert status["last_error"] == "TimeoutError"
    assert not list(Path(config["destination"]).glob("*.tar.gz"))


@pytest.fixture
def backup_for_timeout_checks(observed, tmp_path):
    config = backup_config(observed, tmp_path)
    result = live_backup.run_backup(config)
    directory = tmp_path / "isolated-timeout-check"
    directory.mkdir()
    live_backup.unpack_archive(Path(config["destination"]) / result["last_archive"], directory)
    with closing(sqlite3.connect(directory / "execution.sqlite")) as db, db:
        # Ensure quick_check runs enough VM instructions to invoke its real
        # SQLite progress handler, without changing the source databases.
        db.execute("CREATE TABLE timeout_payload (payload TEXT)")
        db.executemany("INSERT INTO timeout_payload VALUES (?)", [("data",)] * 5000)
    return directory, config["account_id"]


def test_backup_verification_can_finish_after_old_twenty_second_limit(
    backup_for_timeout_checks, monkeypatch
):
    directory, account = backup_for_timeout_checks
    ticks = iter([100.0, 121.0])
    monkeypatch.setattr(
        live_backup,
        "time",
        SimpleNamespace(
            monotonic=lambda: next(ticks, 121.0), process_time=live_backup.time.process_time
        ),
    )
    checks = []
    assert live_backup.verify(directory, account, checks=checks)["startup_complete"]
    assert checks[0]["database"] == "execution"
    assert checks[0]["wall_seconds"] == 21


@pytest.mark.parametrize("deadline,elapsed", [(None, 61.0), (110.0, 21.0)])
def test_backup_verification_keeps_per_database_and_total_deadlines(
    backup_for_timeout_checks, monkeypatch, deadline, elapsed
):
    directory, account = backup_for_timeout_checks
    ticks = iter([100.0, 100.0 + elapsed])
    monkeypatch.setattr(
        live_backup,
        "time",
        SimpleNamespace(
            monotonic=lambda: next(ticks, 100.0 + elapsed),
            process_time=live_backup.time.process_time,
        ),
    )
    checks = []
    with pytest.raises(TimeoutError, match="backup_database_check_timeout:execution"):
        live_backup.verify(directory, account, deadline=deadline, checks=checks)
    assert checks[0]["database"] == "execution"
    assert checks[0]["wall_seconds"] == elapsed


def test_archived_backup_timeout_records_stage_and_retains_last_success(
    observed, tmp_path, monkeypatch
):
    config = backup_config(observed, tmp_path)
    Path(config["status"]).write_text(json.dumps({"last_success_at": 1234}))
    source = Path(config["source"])
    before = {p.name: p.read_bytes() for p in source.glob("*.sqlite")}
    verify = live_backup.verify

    def timeout_on_readback(directory, account, **kwargs):
        if Path(directory).name.startswith(".trx-check-"):
            raise TimeoutError("backup_database_check_timeout:strategy")
        return verify(directory, account, **kwargs)

    monkeypatch.setattr(live_backup, "verify", timeout_on_readback)
    with pytest.raises(TimeoutError, match="backup_database_check_timeout:strategy"):
        live_backup.run_backup(config)
    status = json.loads(Path(config["status"]).read_text())
    assert status["last_success_at"] == 1234
    assert status["last_error_stage"] == "archived_database_verification"
    assert status["last_error_message"] == "backup_database_check_timeout:strategy"
    assert len(status["last_attempt_database_checks"]["copied"]) == 2
    assert not list(Path(config["destination"]).glob("*.tar.gz"))
    assert {p.name: p.read_bytes() for p in source.glob("*.sqlite")} == before


def test_backup_verification_still_rejects_corrupt_database(backup_for_timeout_checks):
    directory, account = backup_for_timeout_checks
    (directory / "execution.sqlite").write_bytes(b"invalid SQLite database")
    with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
        live_backup.verify(directory, account)


def test_backup_accepts_updates_completed_during_large_strategy_copy(
    observed, tmp_path, monkeypatch
):
    config = backup_config(observed, tmp_path)
    source = Path(config["source"])
    copy_database = live_backup.copy_database
    copies = []

    def copy_with_worker_update(path, destination, deadline):
        if Path(path).name == "strategy.sqlite":
            # Simulate a normal worker completing its writes before the large
            # database's SQLite backup snapshot is established.
            for name in ("execution", "strategy"):
                with closing(sqlite3.connect(source / (name + ".sqlite"))) as db, db:
                    db.execute("UPDATE trx_state SET version=version+1 WHERE id=?", (name,))
            copies.append(True)
        copy_database(path, destination, deadline)

    monkeypatch.setattr(live_backup, "copy_database", copy_with_worker_update)
    result = live_backup.run_backup(config)
    restore = tmp_path / "isolated-copy"
    restore.mkdir()
    live_backup.unpack_archive(Path(config["destination"]) / result["last_archive"], restore)
    for name in ("execution", "strategy"):
        assert live_backup.state(restore / (name + ".sqlite"), name) == live_backup.state(
            source / (name + ".sqlite"), name
        )
    assert len(copies) == 1
    assert result["last_error"] is None


@pytest.mark.parametrize("changed", ["execution", "strategy", "control"])
def test_backup_rejects_changes_across_execution_copy(observed, tmp_path, monkeypatch, changed):
    config = backup_config(observed, tmp_path)
    Path(config["status"]).write_text(json.dumps({"last_success_at": 1234}))
    copy_database = live_backup.copy_database
    changes = []

    def copy_with_concurrent_change(path, destination, deadline):
        copy_database(path, destination, deadline)
        if Path(path).name != "execution.sqlite":
            return
        changes.append(True)
        if changed == "control":
            control = Path(config["control"])
            value = json.loads(control.read_text())
            value["concurrent_change"] = len(changes)
            control.write_text(json.dumps(value))
        else:
            database = Path(config["source"]) / (changed + ".sqlite")
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("UPDATE trx_state SET version=version+1 WHERE id=?", (changed,))

    monkeypatch.setattr(live_backup, "copy_database", copy_with_concurrent_change)
    with pytest.raises(ValueError, match="backup_source_changed_retry_later"):
        live_backup.run_backup(config)
    assert len(changes) == 3
    status = json.loads(Path(config["status"]).read_text())
    assert status["last_success_at"] == 1234
    assert status["last_error"] == "ValueError"
    assert not list(Path(config["destination"]).glob("*.tar.gz"))


def test_retention_preserves_two_verified_backups_and_unrelated_files(tmp_path):
    unrelated = tmp_path / "deployment-important.tar.gz"
    unrelated.write_bytes(b"do not prune")
    for n in range(5):
        (tmp_path / f"trx-auto-{n}.tar.gz").write_bytes(b"archive" * 100)
        (tmp_path / f"trx-auto-{n}.json").write_text(
            json.dumps({"format": "trx-online-backup-v1", "created_at": n})
        )
    result = live_backup.prune(tmp_path, 1, keep_hourly=1, keep_daily=0)
    assert result["archives"] == 2
    assert result["over_limit"]
    assert unrelated.read_bytes() == b"do not prune"
