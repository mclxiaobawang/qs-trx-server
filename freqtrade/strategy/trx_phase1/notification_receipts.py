"""Receipt acknowledgments in the independent monitor DB; no trading dependencies."""

# ruff: noqa: RUF001 - Chinese notification copy
import hashlib
import json
import secrets
import sqlite3
from contextlib import closing
from urllib.request import Request, urlopen


COLUMNS = {
    "callback_token": "TEXT",
    "telegram_message_id": "INTEGER",
    "rendered_subject": "TEXT",
    "rendered_body": "TEXT",
    "acknowledged_at": "INTEGER",
    "acknowledgment_source": "TEXT",
}


def migrate(db):
    columns = {row[1] for row in db.execute("PRAGMA table_info(notifications)")}
    for name, kind in COLUMNS.items():
        if name not in columns:
            db.execute(f"ALTER TABLE notifications ADD COLUMN {name} {kind}")
    db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS notification_callback_token "
        "ON notifications(callback_token) WHERE callback_token IS NOT NULL"
    )


def scope(account):
    return hashlib.sha256(account.encode()).hexdigest()[:8]


def reference(account, notification_id):
    return f"TRX-{scope(account)}-{notification_id:06d}"


def format_notification(config, subject, body):
    label, domain = config.get("source_label"), config.get("source_domain")
    if label or domain:
        if (
            not isinstance(label, str)
            or not label.strip()
            or len(label) > 32
            or any(c in label for c in "\r\n【】")
            or domain not in {"trx.octo-agent.com", "trxqs.octo-agent.com"}
        ):
            raise ValueError("notification_source_invalid")
        subject = f"【{label}｜{domain}】" + subject
        body = f"来源：https://{domain}/trx\n" + body
    if config.get("_notification_reference"):
        body += "\n通知编号：" + config["_notification_reference"]
        body += "\n下方确认仅表示已收到通知，不会解除故障或执行交易。"
    return subject, body


def send_payload(config, subject, body):
    subject, body = format_notification(config, subject, body)
    payload = {
        "chat_id": config["chat_id"],
        "text": subject + "\n" + body,
        "disable_web_page_preview": True,
    }
    callback = config.get("_notification_callback")
    if callback:
        payload["reply_markup"] = markup(config, callback)
    return payload


def markup(config, callback, acknowledged=False):
    ref = config["_notification_reference"]
    return {
        "inline_keyboard": [
            [
                {
                    "text": "✅ 已确认收到" if acknowledged else "确认收到",
                    "callback_data": callback,
                }
            ],
            [
                {
                    "text": "查看对应通知",
                    "url": f"https://{config['source_domain']}/notifications#{ref}",
                }
            ],
        ]
    }


def public_rows(operations, *, before=None, limit=30, unresolved=False):
    from .live_ops import notification_content

    query, args = "SELECT * FROM notifications WHERE 1=1", []
    if before is not None:
        query += " AND id < ?"
        args.append(before)
    if unresolved:
        query += " AND delivery_state IN ('unknown','sending') AND acknowledged_at IS NULL"
    query += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    config = operations.notification()
    with closing(sqlite3.connect(operations.database)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(query, args).fetchall()
    result = []
    for row in rows:
        exact = row["rendered_subject"] is not None
        if exact:
            title, body = row["rendered_subject"], row["rendered_body"]
        else:
            title, _ = notification_content(
                row["code"], row["transition"], json.loads(row["payload"]), row["created_at"]
            )
            # Legacy rows did not retain the final rendered text or current status.
            body = json.loads(row["payload"]).get("message", "")
        result.append(
            {
                "id": row["id"],
                "reference": reference(operations.view.account_id, row["id"]),
                "created_at": row["created_at"],
                "attempted_at": row["attempted_at"],
                "sent_at": row["sent_at"],
                "delivery_state": row["delivery_state"],
                "acknowledged_at": row["acknowledged_at"],
                "acknowledgment_source": row["acknowledgment_source"],
                "has_telegram_button": bool(row["callback_token"]),
                "title": title,
                "body": body,
                "exact_sent_text": exact,
            }
        )
    return {
        "items": result,
        "next_before": rows[-1]["id"] if len(rows) == limit else None,
        "source_label": config.get("source_label"),
        "source_domain": config.get("source_domain"),
    }


def acknowledge_web(operations, notification_id, expected_reference):
    if expected_reference != reference(operations.view.account_id, notification_id):
        raise ValueError("notification_reference_mismatch")
    with closing(sqlite3.connect(operations.database)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT acknowledged_at,attempted_at FROM notifications WHERE id=?", (notification_id,)
        ).fetchone()
        if not row or row[1] is None:
            raise ValueError("notification_not_attempted")
        stamp = row[0] or operations.clock()
        if row[0] is None:
            db.execute(
                "UPDATE notifications SET acknowledged_at=?,acknowledgment_source='web' WHERE id=?",
                (stamp, notification_id),
            )
    return {"reference": expected_reference, "acknowledged_at": stamp}


def telegram_call(config, method, payload):
    # This receiver has no sendMessage/trading capability. Never log error URLs.
    if method not in {"answerCallbackQuery", "editMessageReplyMarkup"}:
        raise ValueError("notification_method_forbidden")
    token = config["bot_token"]
    if any(
        c not in "0123456789:abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-" for c in token
    ):
        raise ValueError("telegram_token_invalid")
    req = Request(
        "https://api.telegram.org/bot" + token + "/" + method,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=5) as response:  # noqa: S310 - fixed Telegram HTTPS host
        result = json.loads(response.read(65536))
    if result.get("ok") is not True:
        raise ValueError("telegram_receipt_response_unconfirmed")


def receive_callback(operations, update):
    config = operations.notification()
    query = update.get("callback_query")
    if not isinstance(query, dict):
        return {"ok": True, "ignored": True}
    callback = query.get("data", "")
    prefix = f"n1:{scope(operations.view.account_id)}:"
    if not isinstance(callback, str) or not callback.startswith(prefix):
        return {"ok": True, "ignored": True}
    message = query.get("message") or {}
    recipient = str(config.get("chat_id", ""))
    message_id = message.get("message_id")
    if (
        not recipient.isdigit()
        or int(recipient) <= 0
        or str((query.get("from") or {}).get("id")) != recipient
        or str((message.get("chat") or {}).get("id")) != recipient
        or (message.get("chat") or {}).get("type") != "private"
        or (message.get("from") or {}).get("is_bot") is not True
        or type(message_id) is not int
        or message_id <= 0
        or not isinstance(query.get("id"), str)
    ):
        raise ValueError("notification_recipient_mismatch")
    with closing(sqlite3.connect(operations.database)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT id,telegram_message_id,acknowledged_at,attempted_at FROM notifications "
            "WHERE callback_token=?",
            (callback,),
        ).fetchone()
        if not row or row[3] is None or (row[1] is not None and row[1] != message_id):
            raise ValueError("notification_callback_mismatch")
        ref = reference(operations.view.account_id, row[0])
        # A delivered message with a lost sendMessage response can still be acknowledged.
        # Match its persisted reference and full text before binding the message id.
        rendered = db.execute(
            "SELECT rendered_subject,rendered_body FROM notifications WHERE id=?", (row[0],)
        ).fetchone()
        if message.get("text") != rendered[0] + "\n" + rendered[1]:
            raise ValueError("notification_message_mismatch")
        if row[2] is None:
            db.execute(
                "UPDATE notifications SET acknowledged_at=?,acknowledgment_source='telegram',"
                "telegram_message_id=COALESCE(telegram_message_id,?) WHERE id=?",
                (operations.clock(), message_id, row[0]),
            )
    # Persist first. A Telegram response timeout never reverses acknowledgment or resends a notice.
    feedback = True
    for method, payload in [
        (
            "answerCallbackQuery",
            {"callback_query_id": query["id"], "text": "已记录收到此通知；策略状态不变。"},
        ),
        (
            "editMessageReplyMarkup",
            {
                "chat_id": config["chat_id"],
                "message_id": message_id,
                "reply_markup": markup({**config, "_notification_reference": ref}, callback, True),
            },
        ),
    ]:
        try:
            telegram_call(config, method, payload)
        except Exception:
            feedback = False
    return {"ok": True, "acknowledged": True, "feedback_confirmed": feedback}


def prepare(operations, notification_id, config, title, body):
    if not config.get("acknowledgement_enabled"):
        return config
    if not config.get("callback_secret") or not str(config.get("chat_id", "")).isdigit():
        raise ValueError("notification_ack_configuration_incomplete")
    callback = f"n1:{scope(operations.view.account_id)}:{secrets.token_urlsafe(24)}"
    config = {
        **config,
        "_notification_callback": callback,
        "_notification_reference": reference(operations.view.account_id, notification_id),
    }
    subject, body = format_notification(config, title, body)
    if len(subject + "\n" + body) > 4096:
        raise ValueError("notification_too_long")
    with closing(sqlite3.connect(operations.database)) as db, db:
        db.execute(
            "UPDATE notifications SET callback_token=?,rendered_subject=?,rendered_body=? "
            "WHERE id=?",
            (callback, subject, body, notification_id),
        )
    return config
