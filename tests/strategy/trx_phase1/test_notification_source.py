"""Notification source labels apply to every event without changing routing."""

import io
import json

import pytest

from freqtrade.strategy.trx_phase1 import live_ops


@pytest.mark.parametrize(
    "label,domain",
    [("Max", "trx.octo-agent.com"), ("齿轮", "trxqs.octo-agent.com"), ("XBW", "xbw-apr.shop")],
)
@pytest.mark.parametrize(
    "transition,item",
    [
        ("opened", {"message": "余额核对异常"}),
        ("recovered", {"message": "已恢复", "recovered_at": 1792923000000}),
        ("trade", {"message": "已核对买入成交", "side": "buy"}),
        ("trade", {"message": "已核对卖出成交", "side": "sell"}),
        ("started", {"observed_at": 1792923000000, "message": "已启动"}),
        ("paused", {"observed_at": 1792923000000, "message": "已暂停"}),
        ("resumed", {"observed_at": 1792923000000, "message": "已恢复运行"}),
    ],
)
def test_source_label_preserves_each_event_and_recipient(
    monkeypatch, label, domain, transition, item
):
    sent = []

    def accept(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr(live_ops, "urlopen", accept)
    config = {
        "enabled": True,
        "channel": "telegram",
        "bot_token": "123:fake",
        "chat_id": "same-chat",
        "source_label": label,
        "source_domain": domain,
    }
    title, body = live_ops.notification_content(
        "strategy_error", transition, item, 1792923000000, "运行中"
    )
    assert live_ops.send_notification(config, title, body)
    assert sent == [
        {
            "chat_id": "same-chat",
            "text": f"【{label}｜{domain}】{title}\n来源：https://{domain}/trx\n{body}",  # noqa: RUF001
            "disable_web_page_preview": True,
        }
    ]


def test_legacy_notification_and_disabled_configuration(monkeypatch):
    sent = []

    def accept(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr(live_ops, "urlopen", accept)
    config = {
        "enabled": True,
        "channel": "telegram",
        "bot_token": "123:fake",
        "chat_id": "same-chat",
    }
    assert live_ops.send_notification(config, "标题", "正文")
    assert sent[0]["text"] == "标题\n正文"
    assert live_ops.send_notification({**config, "enabled": False}, "标题", "正文") is False
    assert len(sent) == 1


@pytest.mark.parametrize(
    "fields",
    [
        {"source_label": "Max"},
        {"source_domain": "trx.octo-agent.com"},
        {"source_label": "Max\n伪造标题", "source_domain": "trx.octo-agent.com"},
        {"source_label": "齿轮", "source_domain": "wrong.example.com"},
    ],
)
def test_invalid_source_cannot_send(monkeypatch, fields):
    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid source must not contact Telegram")

    monkeypatch.setattr(live_ops, "urlopen", unexpected)
    config = {
        "enabled": True,
        "channel": "telegram",
        "bot_token": "123:fake",
        "chat_id": "same-chat",
        **fields,
    }
    with pytest.raises(ValueError, match="notification_source_invalid"):
        live_ops.send_notification(config, "标题", "正文")
