"""Observer-only lifecycle notices must reflect progress and survive lost acknowledgments."""
# ruff: noqa: RUF001 - notification assertions use user-facing Chinese punctuation

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1 import live_ops
from freqtrade.strategy.trx_phase1.live_ops import Operations
from tests.strategy.trx_phase1.test_live_ops import monitor
from tests.strategy.trx_phase1.test_live_ops import observed as source_observed  # noqa: F401
from tests.strategy.trx_phase1.test_live_strategy import live  # noqa: F401


@pytest.fixture
def observed(source_observed):  # noqa: F811 - imported pytest fixture
    return source_observed


def fixture(observed, tmp_path, monkeypatch):
    _, broker, view, ops = monitor(observed, tmp_path, monkeypatch)
    snapshot = deepcopy(view.snapshot())
    execution, state = snapshot[2:]
    monkeypatch.setattr(view, "snapshot", lambda: snapshot)
    state.update(desired="running", phase="observing", error=None, startup_complete=True)
    state["engine"].update(last_close=broker.now, buy_at=broker.now + 3600000)
    state["last_poll_at"] = execution["interest_checked_at"] = broker.now
    state["indicator_initialization"] = {"initialized_at": broker.now}
    config = tmp_path / "telegram.json"
    config.write_text(json.dumps({"enabled": True, "bot_token": "fixture", "chat_id": "fixture"}))
    ops.notification_file = config
    sent = []
    monkeypatch.setattr(
        live_ops, "send_notification", lambda cfg, title, body: sent.append((title, body))
    )
    return broker, view, ops, execution, state, sent


def notices(ops):
    with sqlite3.connect(ops.database) as db:
        return db.execute(
            "SELECT transition,delivery_state,attempts FROM notifications "
            "WHERE transition IN ('started','paused','resumed') ORDER BY id"
        ).fetchall()


def restarted(ops, view, broker, tmp_path):
    return Operations(
        view,
        ops.database,
        ops.backup_status,
        tmp_path,
        ops.notification_file,
        clock=lambda: broker.now,
    )


def test_rollout_baseline_and_observer_restart_never_announce_a_fictitious_start(
    observed, tmp_path, monkeypatch
):
    broker, view, ops, _, _, sent = fixture(observed, tmp_path, monkeypatch)
    original = {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")}
    ops.check()
    restored = restarted(ops, view, broker, tmp_path)
    restored.check()
    restored.deliver()
    assert notices(ops) == [] and sent == []
    assert {p.name: p.read_bytes() for p in view.source.glob("*.sqlite")} == original


def test_first_start_waits_for_real_progress_and_delivers_once_across_restart(
    observed, tmp_path, monkeypatch
):
    broker, view, ops, execution, state, sent = fixture(observed, tmp_path, monkeypatch)
    state.update(desired="paused", phase="prepared", activated_at=None, startup_complete=False)
    ops.check()
    state.update(desired="running", phase="connecting")
    ops.check()
    state.update(phase="executing", startup_complete=True, activated_at=broker.now)
    execution["interest_checked_at"] = broker.now - 180001
    ops.check()
    assert notices(ops) == []
    execution["interest_checked_at"] = broker.now
    ops.check()
    confirmed_at = broker.now
    ops.check()
    restored = restarted(ops, view, broker, tmp_path)
    restored.check()
    restored.deliver()
    restored.deliver()
    assert notices(ops) == [("started", "sent", 1)]
    assert len(sent) == 1 and sent[0][0] == "[TRX策略状态] 策略启动成功"
    assert "状态变更确认时间：" + live_ops.beijing_time(confirmed_at) in sent[0][1]
    assert "当前状态：运行中，正在执行既有流程" in sent[0][1]


def test_pause_resume_history_keeps_event_time_and_current_status_distinct(
    observed, tmp_path, monkeypatch
):
    broker, view, ops, _, state, sent = fixture(observed, tmp_path, monkeypatch)
    ops.check()
    state.update(desired="paused", phase="pausing")
    broker.now += 1000
    ops.check()
    paused_at = broker.now
    state["phase"] = "paused"
    ops.check()
    state.update(desired="running", phase="connecting")
    ops.check()
    assert notices(ops) == [("paused", "pending", 0)]
    state["phase"] = "observing"
    broker.now += 1000
    ops.check()
    restored = restarted(ops, view, broker, tmp_path)
    restored.check()
    restored.deliver()
    restored.deliver()
    restored.deliver()
    assert notices(ops) == [("paused", "sent", 1), ("resumed", "sent", 1)]
    assert [title for title, _ in sent] == [
        "[TRX策略状态] 已暂停新增委托",
        "[TRX策略状态] 策略已恢复运行",
    ]
    assert "变更确认时状态：已暂停新增委托" in sent[0][1]
    assert "当前状态：运行中，等待新信号" in sent[0][1]
    assert live_ops.beijing_time(paused_at) in sent[0][1]


def test_short_pause_resume_is_recognized_from_the_persisted_initialization_marker(
    observed, tmp_path, monkeypatch
):
    broker, view, ops, _, state, sent = fixture(observed, tmp_path, monkeypatch)
    ops.check()
    # The worker was resumed between monitor samples; do not invent a pause time.
    state["indicator_initialization"]["initialized_at"] += 1
    ops.check()
    restarted(ops, view, broker, tmp_path).check()
    ops.deliver()
    assert notices(ops) == [("resumed", "sent", 1)]
    assert "已恢复运行" in sent[0][0]


def test_temporary_error_without_new_run_marker_does_not_duplicate_alarm_recovery(
    observed, tmp_path, monkeypatch
):
    _, _, ops, _, state, _ = fixture(observed, tmp_path, monkeypatch)
    state.update(error="market_network_unavailable", phase="recovering")
    ops.check()
    state.update(error=None, phase="observing")
    ops.check()
    assert notices(ops) == []


def test_ambiguous_delivery_and_repeated_checks_never_resend_lifecycle_notice(
    observed, tmp_path, monkeypatch
):
    broker, view, ops, _, state, _ = fixture(observed, tmp_path, monkeypatch)
    ops.check()
    state.update(desired="paused", phase="paused")
    ops.check()
    attempted = []

    def lost_ack(*args):
        attempted.append(True)
        raise TimeoutError()

    monkeypatch.setattr(live_ops, "send_notification", lost_ack)
    ops.deliver()
    restored = restarted(ops, view, broker, tmp_path)
    restored.check()
    restored.deliver()
    assert attempted == [True]
    assert notices(ops) == [("paused", "unknown", 1)]
    assert restored.status()["notification"]["last_error"] == "telegram_delivery_unconfirmed"


def test_concurrent_checks_and_deliveries_claim_one_event_once(observed, tmp_path, monkeypatch):
    _, view, ops, execution, state, sent = fixture(observed, tmp_path, monkeypatch)
    ops.check()
    state.update(desired="paused", phase="paused")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: ops.observe_lifecycle(execution, state, ops.clock()), range(2)))
        list(pool.map(lambda _: ops.deliver(), range(2)))
    assert notices(ops) == [("paused", "sent", 1)] and len(sent) == 1
    assert view.account_id == execution["account_id"]


@pytest.mark.parametrize("field", ["last_poll_at", "startup_complete"])
def test_no_success_notice_with_stale_heartbeat_or_incomplete_startup(
    observed, tmp_path, monkeypatch, field
):
    _, _, ops, _, state, _ = fixture(observed, tmp_path, monkeypatch)
    state.update(desired="paused", phase="prepared", activated_at=None, startup_complete=False)
    ops.check()
    state.update(desired="running", phase="observing", startup_complete=True)
    state[field] = 0 if field == "last_poll_at" else False
    ops.check()
    assert notices(ops) == []


def test_paused_rollout_then_actual_resume_is_not_announced_as_first_start(
    observed, tmp_path, monkeypatch
):
    _, _, ops, _, state, sent = fixture(observed, tmp_path, monkeypatch)
    state.update(desired="paused", phase="paused")
    ops.check()
    assert notices(ops) == []
    state.update(desired="running", phase="observing")
    ops.check()
    ops.deliver()
    assert notices(ops) == [("resumed", "sent", 1)]
    assert sent[0][0] == "[TRX策略状态] 策略已恢复运行"


def test_unreadable_current_snapshot_does_not_misrepresent_the_queued_event(
    observed, tmp_path, monkeypatch
):
    _, view, ops, _, state, sent = fixture(observed, tmp_path, monkeypatch)
    ops.check()
    state.update(desired="paused", phase="paused")
    ops.check()

    def unavailable():
        raise live_ops.ViewUnavailable()

    monkeypatch.setattr(view, "snapshot", unavailable)
    ops.deliver()
    assert notices(ops) == [("paused", "sent", 1)]
    assert "当前状态：暂时无法读取" in sent[0][1]
