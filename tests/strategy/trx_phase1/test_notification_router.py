"""Shared callbacks retain the original account receiver and fail closed on outages."""

import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from freqtrade.strategy.trx_phase1 import notification_router as router
from freqtrade.strategy.trx_phase1.live_ops_app import create_app
from tests.strategy.trx_phase1.test_notification_receipts import monitor  # noqa: F401


@pytest.fixture
def routed(monitor):  # noqa: F811
    config = monitor.notification()
    config.update(
        source_label="齿轮",
        source_domain="trxqs.octo-agent.com",
        callback_peers=[
            {
                "scope": "63c08d12",
                "secret": "s" * 40,
                "url": "https://trx.octo-agent.com/api/v1/trx/live/telegram-callback",
            }
        ],
    )
    monitor.notification_file.write_text(json.dumps(config))
    return monitor


def update(data):
    return {"update_id": 999, "callback_query": {"id": "test", "data": data}}


def test_local_and_legacy_scopes_route_to_their_own_receiver(routed, monkeypatch):
    local, peer = [], []
    monkeypatch.setattr(
        router.receipts,
        "receive_callback",
        lambda ops, value: local.append(value) or {"ok": True, "local": True},
    )
    monkeypatch.setattr(
        router,
        "forward_callback",
        lambda cfg, value: peer.append((cfg, value)) or {"ok": True, "legacy": True},
    )
    first = update(f"n1:{router.receipts.scope(routed.view.account_id)}:local")
    second = update("n1:63c08d12:legacy")
    assert router.dispatch_callback(routed, first)["local"]
    assert router.dispatch_callback(routed, second)["legacy"]
    assert local == [first] and peer[0][1] == second
    assert peer[0][0] == routed.notification()["callback_peers"][0]


@pytest.mark.parametrize(
    "value",
    [{}, {"callback_query": []}, update(None), update("n1:00000000:unknown"), update("other")],
)
def test_unknown_callbacks_cannot_touch_any_account(routed, monkeypatch, value):
    def unexpected(*args):
        pytest.fail("unrecognized callback must not leave this receiver")

    monkeypatch.setattr(router.receipts, "receive_callback", unexpected)
    monkeypatch.setattr(router, "forward_callback", unexpected)
    assert router.dispatch_callback(routed, value) == {"ok": True, "ignored": True}


def test_forward_preserves_update_and_peer_secret_without_redirects(routed, monkeypatch):
    peer = routed.notification()["callback_peers"][0]
    value = update("n1:63c08d12:legacy")
    sent = []

    def open_request(request, timeout):
        sent.append((request, timeout))
        return io.BytesIO(b'{"ok":true,"acknowledged":true}')

    def opener(*handlers):
        assert any(isinstance(h, router.NoRedirect) for h in handlers)
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(router, "build_opener", opener)
    assert router.forward_callback(peer, value)["acknowledged"]
    request, timeout = sent[0]
    assert request.full_url == peer["url"] and timeout == 12
    assert request.get_method() == "POST" and json.loads(request.data) == value
    assert request.get_header("X-telegram-bot-api-secret-token") == peer["secret"]
    assert (
        router.NoRedirect().redirect_request(None, None, 302, None, {}, "https://evil.example")
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        {"url": "http://127.0.0.1/"},
        {"url": "https://evil.example/"},
        {"scope": "bad-scope"},
        {"secret": "short"},
    ],
)
def test_invalid_peer_cannot_contact_network(routed, monkeypatch, change):
    monkeypatch.setattr(router, "build_opener", lambda *a: pytest.fail("invalid peer"))
    peer = {**routed.notification()["callback_peers"][0], **change}
    with pytest.raises(ValueError, match="notification_peer_invalid"):
        router.forward_callback(peer, update("n1:63c08d12:legacy"))


@pytest.mark.parametrize(
    "fault",
    [
        TimeoutError(),
        HTTPError("redacted", 403, "rejected", {}, None),
        HTTPError("redacted", 302, "redirect", {}, None),
    ],
)
def test_peer_outage_does_not_acknowledge_locally(routed, monkeypatch, fault):
    def fail(*args, **kwargs):
        raise fault

    monkeypatch.setattr(router, "build_opener", lambda *a: SimpleNamespace(open=fail))
    peer = routed.notification()["callback_peers"][0]
    with pytest.raises(router.CallbackForwardUnavailable):
        router.forward_callback(peer, update("n1:63c08d12:legacy"))


def test_http_receiver_requests_retry_for_outage_but_keeps_old_faults(routed, monkeypatch):
    def fail(*args):
        raise router.CallbackForwardUnavailable()

    monkeypatch.setattr(router, "forward_callback", fail)
    app = create_app(
        {"username": "operator", "jwt_secret_key": "k" * 32}, view=routed.view, operations=routed
    )
    with TestClient(app) as client:
        path = "/api/v1/trx/live/telegram-callback"
        payload = update("n1:63c08d12:legacy")
        assert client.post(path, json=payload).status_code == 403
        response = client.post(
            path, json=payload, headers={"X-Telegram-Bot-Api-Secret-Token": "s" * 32}
        )
        assert (
            response.status_code == 503
            and response.json()["detail"] == "notification_peer_unavailable"
        )
        assert routed.status()["alerts"] == [{"code": "worker_stale", "active": True}]


def test_receiver_can_be_ready_before_outgoing_buttons_are_enabled(routed):
    config = routed.notification()
    config.update(acknowledgement_enabled=False, callback_receiver_enabled=True)
    routed.notification_file.write_text(json.dumps(config))
    app = create_app(
        {"username": "operator", "jwt_secret_key": "k" * 32}, view=routed.view, operations=routed
    )
    with TestClient(app) as client:
        path = "/api/v1/trx/live/telegram-callback"
        assert client.post(path, json={}).status_code == 403
        response = client.post(path, json={}, headers={"X-Telegram-Bot-Api-Secret-Token": "s" * 32})
        assert response.status_code == 200 and response.json()["ignored"]
