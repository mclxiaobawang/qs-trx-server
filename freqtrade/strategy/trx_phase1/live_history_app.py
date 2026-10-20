"""Read-only account-history HTTP service, independently deployable from trading."""

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import jwt
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


if __package__:
    from freqtrade.strategy.trx_phase1.live_history import HistoryUnavailable, LiveHistory
else:
    # Direct execution keeps the reader independent of strategy package imports.
    from live_history import HistoryUnavailable, LiveHistory


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


async def collect_history(history, health):
    while True:
        try:
            await asyncio.to_thread(history.sync)
            health["ok"] = True
        except Exception:
            health["ok"] = False
            logging.getLogger(__name__).warning("History source temporarily unavailable")
        await asyncio.sleep(1)


def create_app(config, history=None):
    history = history or LiveHistory(config["source"], config["archive"], config["account_id"])
    security = HTTPBearer(auto_error=False)
    health = {"ok": False}

    def initialize():
        if config.get("seed"):
            history.seed(json.loads(Path(config["seed"]).read_text()))
        history.sync()
        health["ok"] = True

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(initialize)
        task = asyncio.create_task(collect_history(history, health))
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
        authorize(credentials, config)

    @app.get("/health")
    def healthcheck():
        if not health["ok"]:
            raise HTTPException(503, "history_source_unavailable")
        return {"ok": True}

    @app.get("/api/v1/trx/live/history", dependencies=[Depends(authenticate)])
    def query(
        account_id: str,
        start: int | None = Query(default=None, ge=0, le=253402300799999),
        end: int | None = Query(default=None, ge=0, le=253402300799999),
        kind: Literal["all", "fill", "exception"] = "all",
        limit: int = Query(default=30, ge=1, le=100),
        cursor: str | None = Query(default=None, max_length=1024),
    ):
        if account_id != config["account_id"]:
            raise HTTPException(409, "history_account_mismatch")
        try:
            history.sync()
            return history.query(start=start, end=end, kind=kind, limit=limit, cursor=cursor)
        except HistoryUnavailable:
            raise HTTPException(503, "history_source_unavailable") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    return app


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    uvicorn.run(
        create_app(json.loads(Path(args.config).read_text())),
        host="0.0.0.0",  # noqa: S104 - container network only; no published host port.
        port=8090,
        access_log=False,
    )
