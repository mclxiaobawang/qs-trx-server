"""Authenticated independent read API and non-trading monitoring loop."""

import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import notification_receipts as receipts
from . import notification_router as router
from .live_ops import LiveView, Operations, ViewUnavailable


def authorize(credentials, config):
    try:
        if credentials is None:
            raise ValueError()
        payload = jwt.decode(
            credentials.credentials,
            config["jwt_secret_key"],
            algorithms=["HS256"],
            options={"require": ["exp", "type", "identity"]},
        )
        username = payload.get("identity", {}).get("u")
        if (
            payload["type"] != "access"
            or not isinstance(username, str)
            or not secrets.compare_digest(username, config["username"])
        ):
            raise ValueError()
    except (jwt.PyJWTError, ValueError, TypeError, AttributeError):
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"}) from None


def create_app(config, view=None, operations=None):  # noqa: C901 - independent monitor routes
    view = view or LiveView(config["source"], config["account_id"])
    operations = operations or Operations(
        view,
        config["state"],
        config["backup_status"],
        config["disk_path"],
        config.get("notification_file"),
        resource_file=config.get("resource_file"),
        trading_container_name=config.get("trading_container_name", "trx-phase1-live"),
    )
    health = {"checked_at": None}

    async def watch():
        while True:
            try:
                await asyncio.to_thread(operations.check)
                health["checked_at"] = int(time.time() * 1000)
                await asyncio.to_thread(operations.deliver)
            except Exception:
                logging.getLogger(__name__).warning("Independent monitor check failed")
            await asyncio.sleep(30)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(watch())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    security = HTTPBearer(auto_error=False)

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
        authorize(credentials, config)

    @app.get("/health")
    def healthcheck():
        if not health["checked_at"] or time.time() * 1000 - health["checked_at"] > 90000:
            raise HTTPException(503, "monitor_stale")
        return {"ok": True}

    @app.get("/api/v1/trx/live/view", dependencies=[Depends(authenticate)])
    def lightweight():
        try:
            return {**view.view(), "operations": operations.status()}
        except (ViewUnavailable, KeyError, TypeError, ValueError):
            raise HTTPException(503, "live_view_unavailable") from None

    @app.get("/api/v1/trx/live/ops", dependencies=[Depends(authenticate)])
    def ops():
        return operations.status()

    @app.get("/notifications")
    def notification_page():
        return FileResponse(
            Path(__file__).with_name("notification_center.html"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/trx/live/notifications", dependencies=[Depends(authenticate)])
    def notifications(before: int | None = None, limit: int = 30, unresolved: bool = False):
        if not 1 <= limit <= 50 or (before is not None and before < 1):
            raise HTTPException(400, "invalid_notification_page")
        return receipts.public_rows(operations, before=before, limit=limit, unresolved=unresolved)

    @app.post(
        "/api/v1/trx/live/notifications/{notification_id}/ack", dependencies=[Depends(authenticate)]
    )
    async def acknowledge(notification_id: int, request: Request):
        settings = operations.notification()
        origin = request.headers.get("origin")
        if origin and origin != "https://" + str(settings.get("source_domain")):
            raise HTTPException(403, "notification_origin_mismatch")
        raw = await request.body()
        if len(raw) > 1024:
            raise HTTPException(413, "notification_request_too_large")
        try:
            body = json.loads(raw)
            if not isinstance(body, dict) or body.get("received") is not True:
                raise ValueError()
            return receipts.acknowledge_web(operations, notification_id, body.get("reference"))
        except (ValueError, TypeError):
            raise HTTPException(409, "notification_confirmation_mismatch") from None

    @app.post("/api/v1/trx/live/telegram-callback")
    async def telegram_callback(request: Request):
        settings = operations.notification()
        expected = settings.get("callback_secret")
        actual = request.headers.get("x-telegram-bot-api-secret-token", "")
        if (
            not settings.get("callback_receiver_enabled", settings.get("acknowledgement_enabled"))
            or not expected
            or not secrets.compare_digest(actual, expected)
        ):
            raise HTTPException(403, "notification_callback_unauthorized")
        raw = await request.body()
        if len(raw) > 65536:
            raise HTTPException(413, "notification_request_too_large")
        try:
            update = json.loads(raw)
            if not isinstance(update, dict):
                raise ValueError()
            return await asyncio.to_thread(router.dispatch_callback, operations, update)
        except router.CallbackForwardUnavailable:
            raise HTTPException(503, "notification_peer_unavailable") from None
        except (ValueError, TypeError, KeyError, AttributeError):
            raise HTTPException(400, "notification_callback_mismatch") from None

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_app(json.loads(Path("/config/observer.json").read_text())),
        host="0.0.0.0",  # noqa: S104 - internal Docker network, no published port
        port=8091,
        access_log=False,
        limit_concurrency=8,
        backlog=32,
    )
