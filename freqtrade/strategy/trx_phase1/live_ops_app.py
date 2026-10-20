"""Authenticated independent read API and non-trading monitoring loop."""

import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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


def create_app(config, view=None, operations=None):
    view = view or LiveView(config["source"], config["account_id"])
    operations = operations or Operations(
        view,
        config["state"],
        config["backup_status"],
        config["disk_path"],
        config.get("notification_file"),
        resource_file=config.get("resource_file"),
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
