"""Standalone authenticated backend for the TRX strategy."""

import argparse
import json
import os
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from freqtrade import __version__
from freqtrade.rpc.api_server import api_trx
from freqtrade.rpc.api_server.api_auth import http_basic_or_jwt_token, router_login
from freqtrade.rpc.api_server.trx_diagnostic_reader import diagnostic_reader


def load_config(path):
    config = json.loads(Path(path).read_text())
    api = config.setdefault("api_server", {})
    for key in ("username", "password", "jwt_secret_key"):
        value = os.environ.get("TRX_API_" + key.upper())
        if value is not None:
            api[key] = value
    return config


def create_app(config):
    config = {**config, "api_server": dict(config.get("api_server", {}))}
    api = config["api_server"]
    if not all(isinstance(api.get(key), str) and api[key] for key in ("username", "password")):
        raise ValueError("Configure the TRX API username and password")
    if not isinstance(api.get("jwt_secret_key"), str) or len(api["jwt_secret_key"]) < 32:
        raise ValueError("Configure a JWT secret with at least 32 characters")
    if config.get("exchange", {"name": "bitget"}).get("name") != "bitget":
        raise ValueError("TRX backend requires Bitget spot")
    if config.get("trading_mode", "spot") != "spot":
        raise ValueError("TRX backend requires Bitget spot")
    config["exchange"] = {"name": "bitget"}
    config["trading_mode"] = "spot"
    config["user_data_dir"] = Path(config.get("user_data_dir") or "user_data").resolve()
    config["datadir"] = Path(
        config.get("datadir") or config["user_data_dir"] / "data/bitget"
    ).resolve()

    @asynccontextmanager
    async def lifespan(app):
        try:
            api_trx.start_live_restore(config)
            yield
        finally:
            api_trx.shutdown_live_services()
            diagnostic_reader.close()

    app = FastAPI(
        title="TRX Strategy API",
        version=__version__,
        docs_url="/docs" if api.get("enable_openapi", False) else None,
        redoc_url=None,
        openapi_url="/openapi.json" if api.get("enable_openapi", False) else None,
        lifespan=lifespan,
    )
    app.state.config = config
    origins = api.get("CORS_origins", [])
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    app.include_router(router_login, prefix="/api/v1", tags=["Auth"])
    app.include_router(
        api_trx.router,
        prefix="/api/v1",
        dependencies=[Depends(http_basic_or_jwt_token)],
    )

    @app.get("/health")
    @app.get("/api/v1/ping")
    def health():
        return {"status": "pong"}

    @app.get("/api/v1/version", dependencies=[Depends(http_basic_or_jwt_token)])
    def version():
        return {"version": __version__}

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the TRX strategy backend")
    # Existing production containers invoke this exact webserver command.
    parser.add_argument("command", nargs="?", choices=["webserver"])
    parser.add_argument("--config", type=Path, default=Path("user_data/config.trx.json"))
    parser.add_argument("--userdir", type=Path)
    parser.add_argument("--logfile", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.userdir is not None:
        config["user_data_dir"] = args.userdir
    app = create_app(config)
    api = config["api_server"]
    log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    if args.logfile is not None:
        args.logfile.parent.mkdir(parents=True, exist_ok=True)
        log_config["handlers"]["trx_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(args.logfile),
            "formatter": "default",
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 10,
        }
        for name in ("uvicorn", "uvicorn.access"):
            log_config["loggers"][name]["handlers"].append("trx_file")
        log_config["root"] = {"handlers": ["default", "trx_file"], "level": "INFO"}
    uvicorn.run(
        app,
        host=api.get("listen_ip_address", "127.0.0.1"),
        port=api.get("listen_port", 8088),
        workers=1,
        limit_concurrency=api.get("limit_concurrency", 8),
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
