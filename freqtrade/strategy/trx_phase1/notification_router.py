"""Route a shared bot's callbacks without changing another account's receiver."""

import json
import re
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from . import notification_receipts as receipts


class CallbackForwardUnavailable(Exception):
    """Ask Telegram to retry when the original receiver cannot be reached."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def forward_callback(peer, update):
    # Fixed peer allowlist prevents SSRF and sending its secret to another host.
    if (
        peer.get("url") != "https://trx.octo-agent.com/api/v1/trx/live/telegram-callback"
        or not re.fullmatch(r"[a-f0-9]{8}", peer.get("scope", ""))
        or not isinstance(peer.get("secret"), str)
        or not 32 <= len(peer["secret"]) <= 256
        or not re.fullmatch(r"[A-Za-z0-9_-]+", peer["secret"])
    ):
        raise ValueError("notification_peer_invalid")
    request = Request(  # noqa: S310 - exact HTTPS peer allowlist above
        peer["url"],
        data=json.dumps(update).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Telegram-Bot-Api-Secret-Token": peer["secret"],
        },
        method="POST",
    )
    try:
        opener = build_opener(ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=12) as response:
            result = json.loads(response.read(65536))
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise CallbackForwardUnavailable()
        return result
    except HTTPError as exc:
        if exc.code == 400:
            raise ValueError("notification_callback_mismatch") from None
        raise CallbackForwardUnavailable() from None
    except (OSError, ValueError):
        raise CallbackForwardUnavailable() from None


def dispatch_callback(operations, update):
    query = update.get("callback_query")
    data = query.get("data") if isinstance(query, dict) else None
    if not isinstance(data, str):
        return {"ok": True, "ignored": True}
    local_scope = receipts.scope(operations.view.account_id)
    if data.startswith(f"n1:{local_scope}:"):
        return receipts.receive_callback(operations, update)
    peers = operations.notification().get("callback_peers", [])
    for peer in peers:
        if isinstance(peer, dict) and data.startswith(f"n1:{peer.get('scope')}:"):
            return forward_callback(peer, update)
    return {"ok": True, "ignored": True}
