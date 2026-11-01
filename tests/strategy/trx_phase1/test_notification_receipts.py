"""Receipts never retry uncertain deliveries, mutate trading state, or accept foreign clicks."""

import io
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient

from freqtrade.strategy.trx_phase1 import live_ops
from freqtrade.strategy.trx_phase1 import notification_receipts as receipts
from freqtrade.strategy.trx_phase1.live_ops_app import create_app


@pytest.fixture
def monitor(tmp_path):
    path = tmp_path / "telegram.json"
    config = {
        "enabled": True,
        "channel": "telegram",
        "bot_token": "123:synthetic",
        "chat_id": "456",
        "source_label": "Max",
        "source_domain": "trx.octo-agent.com",
        "acknowledgement_enabled": True,
        "callback_secret": "s" * 32,
    }
    path.write_text(json.dumps(config))
    view = SimpleNamespace(account_id="old-account")
    ops = live_ops.Operations(
        view,
        tmp_path / "ops.sqlite",
        tmp_path / "backup",
        tmp_path,
        notification_file=path,
        clock=lambda: 1792930000000,
    )
    with closing(sqlite3.connect(ops.database)) as db, db:
        db.execute(
            "INSERT INTO ops_state VALUES(1,?)",
            (
                json.dumps(
                    {
                        "alerts": [{"code": "worker_stale", "active": True}],
                        "notification_error": "telegram_delivery_unconfirmed",
                    }
                ),
            ),
        )
        db.execute(
            "INSERT INTO notifications(code,transition,created_at,payload,attempted_at,"
            "attempts,delivery_state) VALUES('worker_stale','opened',1000,?,1001,1,'unknown')",
            (json.dumps({"message": "old uncertain notification"}),),
        )
    return ops


def records(ops):
    with closing(sqlite3.connect(ops.database)) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM notifications ORDER BY id")]


def add_pending(ops):
    with closing(sqlite3.connect(ops.database)) as db, db:
        db.execute(
            "INSERT INTO notifications(code,transition,created_at,payload) "
            "VALUES('worker_stale','opened',2000,?)",
            (json.dumps({"message": "new"}),),
        )


def test_legacy_migration_and_manual_receipt_preserve_attempts_and_active_fault(monitor):
    before = records(monitor)
    with closing(sqlite3.connect(monitor.database)) as db, db:
        receipts.migrate(db)
    assert records(monitor) == before
    listing = receipts.public_rows(monitor, unresolved=True)
    item = listing["items"][0]
    assert item["reference"] and not item["has_telegram_button"]
    assert not item["exact_sent_text"]
    assert "callback_token" not in json.dumps(listing)
    assert monitor.status()["notification"]["last_error"] == "telegram_delivery_unconfirmed"
    with pytest.raises(ValueError):
        receipts.acknowledge_web(monitor, 1, "foreign-account-reference")
    ack = receipts.acknowledge_web(monitor, 1, item["reference"])
    assert receipts.acknowledge_web(monitor, 1, item["reference"]) == ack
    after = records(monitor)[0]
    assert after["attempts"] == 1 and after["delivery_state"] == "unknown"
    assert after["sent_at"] is None and after["acknowledgment_source"] == "web"
    status = monitor.status()
    assert status["notification"]["pending"] == 0
    assert status["notification"]["last_error"] is None
    assert status["alerts"] == [{"code": "worker_stale", "active": True}]
    assert not receipts.public_rows(monitor, unresolved=True)["items"]


def deliver_with_lost_reply(ops, monkeypatch):
    add_pending(ops)
    requests = []

    def lose_response(request, timeout):
        requests.append(json.loads(request.data))
        raise TimeoutError()

    monkeypatch.setattr(live_ops, "urlopen", lose_response)
    ops.deliver()
    ops.deliver()
    assert len(requests) == 1  # no historical resend or ambiguous-result retry
    payload = requests[0]
    assert payload["chat_id"] == "456"
    callback = payload["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    assert len(callback.encode()) <= 64
    assert receipts.reference(ops.view.account_id, 2) in payload["text"]
    assert payload["reply_markup"]["inline_keyboard"][1][0]["url"].startswith(
        "https://trx.octo-agent.com/notifications#TRX-"
    )
    return {
        "update_id": 1,
        "callback_query": {
            "id": "query-one",
            "data": callback,
            "from": {"id": 456},
            "message": {
                "message_id": 999,
                "chat": {"id": 456, "type": "private"},
                "from": {"is_bot": True},
                "text": payload["text"],
            },
        },
    }


def test_lost_send_reply_can_be_confirmed_in_telegram_without_resend(monitor, monkeypatch):
    update = deliver_with_lost_reply(monitor, monkeypatch)
    calls = []
    monkeypatch.setattr(receipts, "telegram_call", lambda c, m, p: calls.append((m, p)))
    result = receipts.receive_callback(monitor, update)
    row = records(monitor)[1]
    assert result["acknowledged"] and row["telegram_message_id"] == 999
    assert row["delivery_state"] == "unknown" and row["acknowledgment_source"] == "telegram"
    assert row["attempts"] == 1
    assert [m for m, _ in calls] == ["answerCallbackQuery", "editMessageReplyMarkup"]
    assert "已确认收到" in calls[1][1]["reply_markup"]["inline_keyboard"][0][0]["text"]
    receipts.receive_callback(monitor, update)
    assert records(monitor)[1] == row  # replay does not change receipt timestamp or outbox
    assert monitor.status()["notification"]["pending"] == 1  # legacy still unacknowledged


@pytest.mark.parametrize("fault", ["user", "chat", "group", "message", "text", "token"])
def test_foreign_or_modified_telegram_click_is_rejected(monitor, monkeypatch, fault):
    update = deliver_with_lost_reply(monitor, monkeypatch)
    q = update["callback_query"]
    if fault == "user":
        q["from"]["id"] = 777
    elif fault == "chat":
        q["message"]["chat"]["id"] = 777
    elif fault == "group":
        q["message"]["chat"]["type"] = "group"
    elif fault == "message":
        with closing(sqlite3.connect(monitor.database)) as db, db:
            db.execute("UPDATE notifications SET telegram_message_id=123 WHERE id=2")
    elif fault == "text":
        q["message"]["text"] = "another notice"
    else:
        q["data"] = q["data"][:-3] + "abc"
    before = records(monitor)
    monkeypatch.setattr(
        receipts, "telegram_call", lambda *a: pytest.fail("must reject before network")
    )
    with pytest.raises(ValueError):
        receipts.receive_callback(monitor, update)
    assert records(monitor) == before


def test_feedback_failure_keeps_receipt_and_successful_send_retains_message_id(
    monitor, monkeypatch
):
    update = deliver_with_lost_reply(monitor, monkeypatch)
    monkeypatch.setattr(receipts, "telegram_call", lambda *a: (_ for _ in ()).throw(TimeoutError()))
    assert receipts.receive_callback(monitor, update)["feedback_confirmed"] is False
    assert records(monitor)[1]["acknowledged_at"]
    add_pending(monitor)
    monkeypatch.setattr(
        live_ops,
        "urlopen",
        lambda *a, **k: io.BytesIO(b'{"ok":true,"result":{"message_id":1000,"chat":{"id":456}}}'),
    )
    monitor.deliver()
    row = records(monitor)[2]
    assert row["delivery_state"] == "sent" and row["telegram_message_id"] == 1000
    assert row["rendered_subject"] and row["rendered_body"]


def test_api_auth_secret_scope_and_explicit_confirmation(monitor, monkeypatch):
    app = create_app(
        {"username": "operator", "jwt_secret_key": "s" * 32}, view=monitor.view, operations=monitor
    )
    token = jwt.encode(
        {
            "identity": {"u": "operator"},
            "type": "access",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        "s" * 32,
    )
    auth = {"Authorization": "Bearer " + token}
    with TestClient(app) as client:
        assert client.get("/notifications").status_code == 200
        assert client.get("/api/v1/trx/live/notifications").status_code == 401
        assert client.post("/api/v1/trx/live/notifications/1/ack", json={}).status_code == 401
        assert client.post("/api/v1/trx/live/telegram-callback", json={}).status_code == 403
        secret = {"X-Telegram-Bot-Api-Secret-Token": "s" * 32}
        assert client.post("/api/v1/trx/live/telegram-callback", headers=secret, json={}).json()[
            "ignored"
        ]
        assert (
            client.post("/api/v1/trx/live/telegram-callback", headers=secret, json=[]).status_code
            == 400
        )
        data = client.get("/api/v1/trx/live/notifications", headers=auth).json()
        assert (
            client.get("/api/v1/trx/live/notifications?limit=51", headers=auth).status_code == 400
        )
        assert "synthetic" not in json.dumps(data) and "chat_id" not in json.dumps(data)
        payload = {"reference": data["items"][0]["reference"], "received": True}
        assert (
            client.post("/api/v1/trx/live/notifications/1/ack", headers=auth, json={}).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v1/trx/live/notifications/1/ack",
                headers={**auth, "Origin": "https://evil.example"},
                json=payload,
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v1/trx/live/notifications/1/ack", headers=auth, json=payload
            ).status_code
            == 200
        )
        assert client.post("/api/v1/trx/live/start", headers=auth).status_code == 404
