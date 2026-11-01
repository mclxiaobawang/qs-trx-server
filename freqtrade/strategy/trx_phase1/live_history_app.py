"""Read-only account-history HTTP service, independently deployable from trading."""

import asyncio
import json
import logging
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import jwt
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


if __package__:
    from freqtrade.strategy.trx_phase1.contribution_preview import (
        ContributionPreviewError,
    )
    from freqtrade.strategy.trx_phase1.contribution_preview import (
        preview as contribution_preview,
    )
    from freqtrade.strategy.trx_phase1.live_cost_basis import CostBasisHistory
    from freqtrade.strategy.trx_phase1.live_history import HistoryUnavailable, LiveHistory
    from freqtrade.strategy.trx_phase1.live_indicators import IndicatorHistory
    from freqtrade.strategy.trx_phase1.live_range_history import RangeBandHistory
    from freqtrade.strategy.trx_phase1.withdrawal_preview import WithdrawalPreviewError, preview
else:
    # Direct execution keeps the reader independent of strategy package imports.
    from contribution_preview import ContributionPreviewError
    from contribution_preview import preview as contribution_preview
    from live_cost_basis import CostBasisHistory
    from live_history import HistoryUnavailable, LiveHistory
    from live_indicators import IndicatorHistory
    from live_range_history import RangeBandHistory
    from withdrawal_preview import WithdrawalPreviewError, preview


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


def sync_band_snapshot(bands, snapshot):
    if bands is not None and snapshot:
        try:
            bands.sync(*snapshot)
        except (ValueError, KeyError, TypeError):
            logging.getLogger(__name__).warning("Price band evidence temporarily unavailable")


async def collect_history(history, health, bands=None):
    while True:
        try:
            snapshot = await asyncio.to_thread(history.sync)
            await asyncio.to_thread(sync_band_snapshot, bands, snapshot)
            health["ok"] = True
        except Exception:
            health["ok"] = False
            logging.getLogger(__name__).warning("History source temporarily unavailable")
        await asyncio.sleep(1)


def project_withdrawal(history, config, account_id, amount):
    if account_id != config["account_id"]:
        raise HTTPException(409, "history_account_mismatch")
    try:
        execution = history.read_source("execution")
        strategy = history.read_source("strategy")
        result = preview(execution, strategy, amount, int(time.time() * 1000))
        current = history.read_source("execution")
        for key in ("book", "holdings", "unassigned_offsets", "operation", "performance_basis"):
            if current.get(key) != execution.get(key):
                raise WithdrawalPreviewError("withdrawal_snapshot_changed")
        return result
    except WithdrawalPreviewError as exc:
        raise HTTPException(409, str(exc)) from None
    except HistoryUnavailable:
        raise HTTPException(503, "history_source_unavailable") from None


def project_contribution(history, config, account_id, coin, amount, holding_id=None):
    if account_id is not None and account_id != config["account_id"]:
        raise HTTPException(409, "history_account_mismatch")
    try:
        execution = history.read_source("execution")
        strategy = history.read_source("strategy")
        result = contribution_preview(
            execution, strategy, coin, amount, int(time.time() * 1000), holding_id=holding_id
        )
        current = history.read_source("execution")
        for key in ("book", "holdings", "unassigned_offsets", "operation", "pending_income"):
            if current.get(key) != execution.get(key):
                raise ContributionPreviewError("contribution_snapshot_changed")
        current_strategy = history.read_source("strategy")
        for key in ("engine", "markets", "activated_at", "intent", "source_sha256"):
            if current_strategy.get(key) != strategy.get(key):
                raise ContributionPreviewError("contribution_snapshot_changed")
        return result
    except ContributionPreviewError as exc:
        raise HTTPException(409, str(exc)) from None
    except HistoryUnavailable:
        raise HTTPException(503, "history_source_unavailable") from None
    except (KeyError, TypeError, ArithmeticError, ValueError):
        raise HTTPException(503, "contribution_source_unavailable") from None


def create_app(config, history=None):  # noqa: C901 - shared authentication for isolated read routes
    history = history or LiveHistory(config["source"], config["archive"], config["account_id"])
    security = HTTPBearer(auto_error=False)
    health = {"ok": False}
    indicators = IndicatorHistory(history.source, config["account_id"])
    costs = CostBasisHistory(history.source, config["account_id"])
    bands = RangeBandHistory(
        history.source,
        history.archive.with_name("range-bands.sqlite"),
        config["account_id"],
    )

    def initialize():
        if config.get("seed"):
            history.seed(json.loads(Path(config["seed"]).read_text()))
        snapshot = history.sync()
        sync_band_snapshot(bands, snapshot)
        health["ok"] = True

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(initialize)
        task = asyncio.create_task(collect_history(history, health, bands))
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

    @app.get("/contribution-preview", response_class=HTMLResponse)
    def contribution_preview_page():
        return HTMLResponse(
            Path(__file__).with_name("contribution_preview.html").read_text(),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/trx/live/contribution-preview", dependencies=[Depends(authenticate)])
    def selected_contribution_preview(
        coin: Literal["USDGO"],
        amount: str = Query(min_length=1, max_length=24, pattern=r"^\d{1,15}(\.\d{1,8})?$"),
        holding_id: str | None = Query(default=None, min_length=1, max_length=128),
        account_id: str | None = None,
    ):
        return project_contribution(history, config, account_id, coin, amount, holding_id)

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

    @app.get("/api/v1/trx/live/withdrawal-preview", dependencies=[Depends(authenticate)])
    def withdrawal_preview(
        account_id: str,
        amount: str = Query(min_length=1, max_length=32, pattern=r"^\d{1,15}(\.\d{1,8})?$"),
    ):
        return project_withdrawal(history, config, account_id, amount)

    @app.get("/api/v1/trx/live/indicators", dependencies=[Depends(authenticate)])
    def indicator_history(
        account_id: str,
        view: Literal["standard", "range_band", "volatility", "cost_basis"] = "standard",
        include_history: bool = False,
        range_name: Literal[
            "summary", "day", "week", "month", "quarter", "half_year", "year", "all"
        ] = Query(default="all", alias="range"),
        limit: int = Query(default=1800, ge=60, le=2400),
    ):
        if account_id != config["account_id"]:
            raise HTTPException(409, "history_account_mismatch")
        try:
            if view == "cost_basis":
                return costs.query(
                    history.read_source("strategy"),
                    history.read_source("execution"),
                    range_name,
                    limit,
                )
            if view != "volatility" and range_name not in {"summary", "day", "week", "all"}:
                raise HTTPException(422, "extended_range_requires_volatility_view")
            if view == "range_band":
                return bands.query(
                    history.read_source("strategy"),
                    history.read_source("execution"),
                    range_name,
                    limit,
                    include_reconstructed=include_history,
                )
            return indicators.query(
                history.read_source("strategy"),
                history.read_source("execution"),
                range_name,
                limit,
                historical_volatility=view == "volatility",
            )
        except (HistoryUnavailable, OSError, ValueError, KeyError, sqlite3.Error):
            raise HTTPException(503, "indicator_history_unavailable") from None

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
